//! One shared engine per folder and options (2026-11-27, MCP stability fix). Two coding-agent
//! sessions open on one project each start their own `serve`; each used to open its own engine:
//! its own read of the material, its own caches and its own four MicroModels on the same graphics
//! card, which a 4 GB card cannot hold twice and which, with caches sized to the free memory,
//! took a 32 GB machine past its memory (both processes ended in an allocation failure within the
//! same second). Now the first `serve` for a (home, folder, options) holds the engine and shares
//! it over loopback (`127.0.0.1`, a port the system picks, written to
//! `<home>/scratch/engines/<key>.json` with the holder's process id); a later `serve` with the
//! same key attaches to it and forwards its client's lines both ways, so every client is served
//! by the one engine (one set of MicroModels on the card, one call at a time through
//! `mcp::CALL_GATE`). The holder keeps serving while any client is attached after its own client
//! leaves. A registration whose engine does not answer is stale and is taken over. A `serve` with
//! other options (`--whole`, `--memory-dir`, ...) is another engine with its own key.
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// How this `serve` takes part: it holds the engine (with its listener and registration), or it
/// attaches to the engine another process holds.
#[derive(Debug)]
pub enum Part {
    Own(Holder),
    Attach { stream: TcpStream, port: u16, pid: u32 },
}

/// The engine this process holds: its listener and the registration file it wrote.
#[derive(Debug)]
pub struct Holder {
    pub listener: TcpListener,
    pub port: u16,
    pub file: PathBuf,
}
impl Drop for Holder {
    fn drop(&mut self) {
        // Removed only while it still names this process (a later holder may have taken over).
        if read(&self.file).is_some_and(|(_, pid)| pid == std::process::id()) {
            let _ = std::fs::remove_file(&self.file);
        }
    }
}

/// The registration key of a (home, folder, options): the hash of the canonical folder and the
/// serve arguments other than `--root <folder>`, in order.
#[must_use]
pub fn key(root: &Path, args: &[String]) -> String {
    let mut text = format!("{}\n", root.display().to_string().to_lowercase());
    let mut skip = false;
    for a in args {
        if skip {
            skip = false;
            continue;
        }
        if a == "--root" {
            skip = true;
            continue;
        }
        text.push_str(a);
        text.push('\n');
    }
    nmhash::sha256_bytes(text.as_bytes()).chars().take(16).collect()
}

/// How long a registration may go unanswered before it is taken as stale.
const STALE: Duration = Duration::from_secs(4);

fn registry(home: &Path, key: &str) -> PathBuf {
    home.join("scratch").join("engines").join(format!("{key}.json"))
}

/// (port, pid) of a registration file.
fn read(file: &Path) -> Option<(u16, u32)> {
    let v = nmjson::parse(&std::fs::read(file).ok()?).ok()?;
    let port = u16::try_from(v.get("port").and_then(nmjson::Value::integer).ok()?).ok()?;
    let pid = u32::try_from(v.get("pid").and_then(nmjson::Value::integer).ok()?).ok()?;
    Some((port, pid))
}

/// Whether the engine on `port` answers: a connection and a `ping` reply within `wait`.
fn answers(port: u16, wait: Duration) -> Option<TcpStream> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let stream = TcpStream::connect_timeout(&addr, wait).ok()?;
    stream.set_read_timeout(Some(wait)).ok()?;
    let mut w = stream.try_clone().ok()?;
    writeln!(w, r#"{{"jsonrpc":"2.0","id":"nano-engine-probe","method":"ping"}}"#).ok()?;
    w.flush().ok()?;
    let mut line = String::new();
    BufReader::new(stream.try_clone().ok()?).read_line(&mut line).ok()?;
    stream.set_read_timeout(None).ok()?;
    line.contains("nano-engine-probe").then_some(stream)
}

/// Decides this `serve`'s part for `key` under `home`: attach to a live registered engine, else
/// hold the engine (bind a loopback listener and register it). A stale registration (its engine
/// does not answer within a few seconds) is replaced.
/// # Errors
/// The listener could not be bound or the registration could not be written.
pub fn claim(home: &Path, key: &str) -> std::io::Result<Part> {
    let file = registry(home, key);
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // A holder answers a ping as soon as it has registered (the handshake needs no engine), so a
    // registration that does not answer for `STALE` is left from a process that ended.
    let deadline = Instant::now() + STALE;
    loop {
        if let Some((port, pid)) = read(&file) {
            if let Some(stream) = answers(port, Duration::from_secs(3)) {
                return Ok(Part::Attach { stream, port, pid });
            }
            if Instant::now() < deadline {
                std::thread::park_timeout(Duration::from_millis(250));
                continue;
            }
            let _ = std::fs::remove_file(&file);
        }
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let body = format!("{{\"pid\":{},\"port\":{port},\"started_ms\":{}}}\n", std::process::id(), nmtype::now_ms());
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&file) {
            Ok(mut f) => {
                f.write_all(body.as_bytes())?;
                return Ok(Part::Own(Holder { listener, port, file }));
            }
            // Another `serve` registered first: attach to it (next turn of the loop).
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && Instant::now() < deadline => {}
            Err(e) => return Err(e),
        }
    }
}

/// Serves the clients that attach to this engine until `stop` is set and none is attached, each
/// connection on its own thread through `mcp::run_as` (attached). `api` is the served contract.
pub fn listen(api: &crate::Api, holder: &Holder, stop: &AtomicBool) {
    if holder.listener.set_nonblocking(true).is_err() {
        eprintln!("engine: the listener could not be set to poll; clients cannot attach");
        return;
    }
    std::thread::scope(|scope| {
        loop {
            match holder.listener.accept() {
                Ok((stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    scope.spawn(move || {
                        let Ok(writer) = stream.try_clone() else { return };
                        eprintln!("engine: a client attached ({} now)", crate::mcp::attached() + 1);
                        if let Err(e) = crate::mcp::run_as(api, BufReader::new(stream), &writer, true) {
                            eprintln!("engine: an attached client's connection ended with an error: {e}");
                        }
                        let _ = writer.shutdown(std::net::Shutdown::Write);
                        eprintln!("engine: a client left ({} attached)", crate::mcp::attached());
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if stop.load(Ordering::SeqCst) && crate::mcp::attached() == 0 {
                        break;
                    }
                    std::thread::park_timeout(Duration::from_millis(50));
                }
                Err(e) => {
                    eprintln!("engine: accept failed: {e}");
                    std::thread::park_timeout(Duration::from_millis(200));
                }
            }
        }
    });
}

/// Forwards this process's client to the engine on `stream`: stdin lines to the engine, the
/// engine's lines to stdout, until the client's input ends (then the engine's last replies are
/// read out) or the engine ends.
/// # Errors
/// The engine's connection could not be cloned, or ended while the client was still connected.
pub fn attach(stream: TcpStream, input: impl Read, mut output: impl Write + Send) -> std::io::Result<()> {
    let from_engine = stream.try_clone()?;
    let mut to_engine = stream;
    let engine_ended = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let back = scope.spawn(|| -> std::io::Result<()> {
            let mut r = BufReader::new(from_engine);
            let mut line = Vec::new();
            loop {
                line.clear();
                if r.read_until(b'\n', &mut line)? == 0 {
                    engine_ended.store(true, Ordering::SeqCst);
                    return Ok(());
                }
                output.write_all(&line)?;
                output.flush()?;
            }
        });
        let mut r = BufReader::new(input);
        let mut line = Vec::new();
        let forward = loop {
            line.clear();
            match r.read_until(b'\n', &mut line) {
                Ok(0) => break Ok(()),
                Ok(_) => {
                    if let Err(e) = to_engine.write_all(&line).and_then(|()| to_engine.flush()) {
                        break Err(e);
                    }
                }
                Err(e) => break Err(e),
            }
        };
        let _ = to_engine.shutdown(std::net::Shutdown::Write);
        let back = back.join().unwrap_or_else(|_| Err(std::io::Error::other("the engine reader stopped")));
        forward?;
        back
    })?;
    if engine_ended.load(Ordering::SeqCst) {
        Ok(())
    } else {
        Err(std::io::Error::other("the shared engine's connection ended"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_key_ignores_root_flag_but_not_options() {
        let root = Path::new("D:/x/repo");
        let a = key(root, &["--root".into(), "D:/x/repo".into(), "--flow".into()]);
        let b = key(root, &["--flow".into(), "--root".into(), "elsewhere".into()]);
        let c = key(root, &["--root".into(), "D:/x/repo".into(), "--whole".into()]);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, key(Path::new("D:/x/other"), &["--flow".into()]));
    }

    #[test]
    fn engine_second_serve_attaches_to_the_first() {
        let home = nmtype::check_dir("nmapi-engine-claim").unwrap();
        let k = "test-claim";
        let first = claim(&home, k).unwrap();
        let Part::Own(holder) = first else { panic!("the first serve must hold the engine") };
        let stop = AtomicBool::new(false);
        let api = crate::tools::api();
        std::thread::scope(|scope| {
            scope.spawn(|| listen(&api, &holder, &stop));
            let second = claim(&home, k).unwrap();
            let Part::Attach { stream, port, pid } = second else { panic!("the second serve must attach") };
            assert_eq!(port, holder.port);
            assert_eq!(pid, std::process::id());
            let input = b"{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n".to_vec();
            let mut out = Vec::new();
            attach(stream, &input[..], &mut out).unwrap();
            assert!(String::from_utf8_lossy(&out).contains("\"id\":7"));
            stop.store(true, Ordering::SeqCst);
        });
        drop(holder);
        assert!(read(&registry(&home, k)).is_none(), "the holder's registration is removed when it ends");
    }
}
