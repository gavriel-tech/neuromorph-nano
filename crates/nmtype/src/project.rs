//! Memory per studied project (Neuromorph Nano, 2026-11-15, build plan E1). The Nano serves any
//! repository; what it keeps about a repository that is not the Nano itself goes to
//! `<nano>/memory/projects/<folder name>-<first 8 hex of the SHA-256 of the absolute path>/`, so
//! two repositories never share a memory and the same repository always finds its own again.
//! The Nano studying its own folder keeps `<nano>/memory/`.
use std::path::{Path, PathBuf};

/// The path as one comparable text: no `\\?\` prefix, forward slashes, no trailing slash, and
/// lower case on Windows (its file names ignore case, so one folder never gets two memories).
#[must_use]
pub fn normal(p: &Path) -> String {
    let t = p.to_string_lossy();
    let t = t.trim_start_matches(r"\\?\").replace('\\', "/");
    let t = t.trim_end_matches('/');
    if cfg!(windows) { t.to_lowercase() } else { t.to_owned() }
}

/// The project key, `<folder name>-<8 hex>`: the folder name with anything but letters, digits,
/// `-`, `_` and `.` replaced by `_` (empty becomes `root`), then the first 8 hex digits of the
/// SHA-256 of [`normal`] of the path. Pass an absolute, canonical path.
#[must_use]
pub fn key(root: &Path) -> String {
    let name: String = root
        .file_name()
        .map(|n| {
            n.to_string_lossy().chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' }).collect()
        })
        .unwrap_or_default();
    let name = if name.is_empty() { "root".to_owned() } else { name };
    let hash = nmhash::sha256_bytes(normal(root).as_bytes());
    format!("{name}-{}", hash.get(..8).unwrap_or(&hash))
}

/// The folder holding every project memory, `<home>/memory/projects`.
#[must_use]
pub fn projects_dir(home: &Path) -> PathBuf {
    home.join("memory").join("projects")
}

/// True when the studied root is the Nano's own folder.
#[must_use]
pub fn is_home(home: &Path, root: &Path) -> bool {
    normal(home) == normal(root)
}

/// The default memory folder for a studied root: `<home>/memory` for the Nano itself, else
/// `<home>/memory/projects/<key>`. Pass absolute, canonical paths.
#[must_use]
pub fn memory_dir(home: &Path, root: &Path) -> PathBuf {
    if is_home(home, root) { home.join("memory") } else { projects_dir(home).join(key(root)) }
}

/// The project name shown in the audit log: `nano` for the Nano itself, else the key.
#[must_use]
pub fn label(home: &Path, root: &Path) -> String {
    if is_home(home, root) { "nano".to_owned() } else { key(root) }
}

/// Section check: the Nano's own folder keeps `memory/`; another root gets a folder under
/// `memory/projects/` named by its folder and 8 hex digits; the key ignores slashes, a trailing
/// slash, the `\\?\` prefix and (on Windows) the case of the path; two roots with the same folder name get different keys.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let home = Path::new("/x/neuromorph-nano");
    if memory_dir(home, Path::new("/x/neuromorph-nano/")) != home.join("memory") {
        return Err("nmtype.project: the Nano's own folder must keep memory/".into());
    }
    let a = key(Path::new("/w/one/shop"));
    let b = key(Path::new("/w/two/shop"));
    if a == b || !a.starts_with("shop-") || a.len() != "shop-".len() + 8 || !a.bytes().skip(5).all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("nmtype.project: keys {a} and {b} are not distinct <name>-<8 hex>"));
    }
    if key(Path::new("/w/one/shop/")) != a {
        return Err("nmtype.project: a trailing slash changed the key".into());
    }
    // The folder name keeps its own case (canonical paths give the real one); the hash does not.
    if cfg!(windows) && key(Path::new(r"\\?\C:\W\Shop")) != key(Path::new("c:/w/Shop")) {
        return Err("nmtype.project: prefix, slashes or case changed the key on Windows".into());
    }
    let d = memory_dir(home, Path::new("/w/one/shop"));
    if d != home.join("memory").join("projects").join(&a) {
        return Err(format!("nmtype.project: memory folder {} is not under memory/projects", d.display()));
    }
    if key(Path::new("/w/my repo!")).split('-').next() != Some("my_repo_") || label(home, home) != "nano" {
        return Err("nmtype.project: folder name not cleaned or label wrong".into());
    }
    Ok(())
}
