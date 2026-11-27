//! Live launcher: `neuromorph-nano launch <command> [args]` copies the newest release build to a
//! fresh file in `scratch/live/` and runs that copy with the same arguments, so a serving session
//! never holds `target/release` open and a rebuild is never blocked. Standard input, output and
//! error pass straight through; the launcher itself writes only to standard error, so standard
//! output carries nothing but the served program's MCP messages.
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

/// The launcher copy `setup` writes and `.mcp.json` names.
#[must_use]
pub fn launcher_path(home: &Path) -> PathBuf {
    home.join("bin").join(format!("neuromorph-nano{}", std::env::consts::EXE_SUFFIX))
}

/// The newest release build.
#[must_use]
pub fn build_path(home: &Path) -> PathBuf {
    home.join("target").join("release").join(format!("neuromorph-nano{}", std::env::consts::EXE_SUFFIX))
}

fn live_dir(home: &Path) -> PathBuf {
    home.join("scratch").join("live")
}

/// A live copy's file name: `nm-` with the time and process id, so two sessions never share one.
fn copy_name() -> String {
    format!("nm-{}-{}{}", nmtype::now_ms(), std::process::id(), std::env::consts::EXE_SUFFIX)
}

fn is_copy(name: &str) -> bool {
    name.starts_with("nm-") && name.ends_with(std::env::consts::EXE_SUFFIX) && name.len() > 3 + std::env::consts::EXE_SUFFIX.len()
}

/// Removes live copies no longer running; a copy still in use cannot be removed and is skipped.
fn clear_old(dir: &Path) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if is_copy(&e.file_name().to_string_lossy()) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
}

/// Runs the launcher: copy, then run the copy with `args` and return its exit code.
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("usage: neuromorph-nano launch <command> [args], for example launch serve --root <repo>");
        return ExitCode::FAILURE;
    }
    let src = build_path(home);
    if !src.is_file() {
        eprintln!("neuromorph-nano launch: no build at {}; run cargo build --release in {}", src.display(), home.display());
        return ExitCode::FAILURE;
    }
    let dir = live_dir(home);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("neuromorph-nano launch: {}: {e}", dir.display());
        return ExitCode::FAILURE;
    }
    clear_old(&dir);
    let exe = dir.join(copy_name());
    if let Err(e) = std::fs::copy(&src, &exe) {
        eprintln!("neuromorph-nano launch: copy to {}: {e}", exe.display());
        return ExitCode::FAILURE;
    }
    let mut cmd = Command::new(&exe);
    cmd.args(args).stdin(Stdio::inherit()).stdout(Stdio::inherit()).stderr(Stdio::inherit());
    // Started by an app with no console (the Claude Code desktop app runs MCP servers so), the copy
    // would open a console window of its own; it runs without one when its input is not a terminal.
    #[cfg(windows)]
    if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let status = cmd.status();
    match status {
        Ok(s) => match s.code() {
            Some(c) => u8::try_from(c).map_or_else(
                |_| {
                    eprintln!("neuromorph-nano launch: child exited abnormally with status 0x{c:08x}");
                    ExitCode::FAILURE
                },
                ExitCode::from,
            ),
            None => ExitCode::FAILURE,
        },
        Err(e) => {
            eprintln!("neuromorph-nano launch: run {}: {e}", exe.display());
            ExitCode::FAILURE
        }
    }
}

/// Writes the launcher copy from the newest build (or from this program when there is none).
/// A launcher still in use by a running session cannot be replaced; the one in place is kept.
/// Returns what happened, for `setup` to print.
/// # Errors
/// No launcher could be written and none is in place.
pub fn write_launcher(home: &Path) -> Result<String, String> {
    let to = launcher_path(home);
    let from = Some(build_path(home)).filter(|p| p.is_file()).or_else(|| std::env::current_exe().ok()).ok_or("no build to copy")?;
    if let Some(d) = to.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    if nmsubs::repo::canon(&from) == nmsubs::repo::canon(&to) {
        return Ok(format!("launcher {} is this program", to.display()));
    }
    match std::fs::copy(&from, &to) {
        Ok(_) => Ok(format!("launcher written: {}", to.display())),
        Err(e) if to.is_file() => Ok(format!("launcher kept: {} (in use, not replaced: {e})", to.display())),
        Err(e) => Err(format!("launcher {}: {e}", to.display())),
    }
}

/// Section check (processor only): copy names are recognised and unique per process and time,
/// other files are never taken for copies, and the paths sit in the Nano's own folder.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let name = copy_name();
    if !is_copy(&name) || is_copy("nm-") || is_copy("neuromorph-nano.exe") || is_copy("notes.txt") {
        return Err(format!("launch: copy names ({name})"));
    }
    let home = Path::new("/x/nano");
    if !launcher_path(home).starts_with(home.join("bin"))
        || !build_path(home).starts_with(home.join("target").join("release"))
        || !live_dir(home).starts_with(home.join("scratch"))
    {
        return Err("launch: paths must stay in the Nano's folder".into());
    }
    Ok(())
}
