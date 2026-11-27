//! Detectors over stores, weights, numbers and paths: C-CLASS (class tags on every store),
//! C-MUSCLE (muscle memory placed apart and refusing sensory items), C-WEIGHTS (released
//! weights unchanged, cheap form), C-NUMBERS (fixed operating numbers of plan section 6) and
//! C-DOWNLOAD (nothing points outside the mini folder).
use super::Finding;
use super::text::{crate_sources, is_comment, list, read, rel, s, so, strs, walk};
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// C-CLASS.
pub(super) fn class(root: &Path, p: &Value) -> Result<Finding, String> {
    let mut f = Finding::default();
    if !read(&root.join(s(p, "tag_file")?)).contains(s(p, "tag_mark")?.as_str()) {
        f.not_yet = Some(format!("no class tag type yet; will look for {}", so(p, "will_look_for")));
        return Ok(f);
    }
    let marks = strs(p, "store_marks");
    for st in list(p, "stores") {
        let (name, dir) = (s(st, "name")?, s(st, "dir")?);
        let mut files = Vec::new();
        walk(&root.join(&dir), &["rs".to_owned()], &mut files);
        if files.is_empty() {
            f.notes.push(format!("{name}: not present ({dir})"));
            continue;
        }
        let tagged = files.iter().any(|x| {
            let t = read(x);
            marks.iter().any(|m| t.contains(m.as_str()))
        });
        f.notes.push(format!("{name}: {}", if tagged { "references the class tag" } else { "no class check" }));
        if !tagged {
            f.add(&format!("store:{name}"), format!("the {name} ({dir}) references no class tag, so it accepts both classes"));
        }
    }
    Ok(f)
}

/// Entries under `dir` (recursive) whose name contains `mark`, case-insensitive.
fn named(root: &Path, dir: &Path, mark: &str, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if e.file_name().to_string_lossy().to_lowercase().contains(mark) {
            out.push(rel(root, &p));
        }
        if p.is_dir() {
            named(root, &p, mark, out);
        }
    }
}

/// C-MUSCLE.
pub(super) fn muscle(root: &Path, p: &Value) -> Result<Finding, String> {
    let mark = s(p, "name_mark")?.to_lowercase();
    let mut f = Finding::default();
    for dir in strs(p, "forbidden") {
        let mut found = Vec::new();
        named(root, &root.join(&dir), &mark, &mut found);
        for x in found {
            f.add(&format!("placed:{x}"), format!("{x}: muscle memory under {dir}/"));
        }
    }
    let store_marks = strs(p, "store_marks");
    let mut stores: BTreeMap<String, String> = BTreeMap::new();
    for src in crate_sources(root) {
        let r = rel(root, &src);
        if store_marks.iter().any(|m| r.to_lowercase().contains(m.as_str())) {
            let krate = r.split('/').nth(1).unwrap_or_default().to_owned();
            stores.entry(krate).or_default().push_str(&read(&src).to_lowercase());
        }
    }
    if stores.is_empty() && f.violations.is_empty() {
        f.not_yet = Some("no muscle-memory store in the repository yet; will look for its placement outside memory/, weights/ and units/ and a refusal of sensory items in its source".into());
        return Ok(f);
    }
    for (krate, text) in &stores {
        let refuses = text.contains("sensory") && text.contains("refus");
        f.notes.push(format!("store {krate}: {}", if refuses { "refuses sensory items" } else { "no sensory refusal found" }));
        if !refuses {
            f.add(&format!("no-sensory-refusal:{krate}"), format!("the muscle-memory store {krate} has no refusal of sensory items"));
        }
    }
    Ok(f)
}

fn mtime(m: &std::fs::Metadata) -> u64 {
    m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs())
}

/// (size, modification time, hash) of a stamp or cache entry.
fn stat_of(v: &Value) -> Option<(u64, u64, String)> {
    Some((v.get("bytes").ok()?.integer().ok()?, v.get("mtime").ok()?.integer().ok()?, v.get("sha256").ok()?.string().ok()?.to_owned()))
}

/// C-WEIGHTS, cheap form: size and modification time against a cached record, never a hash.
pub(super) fn weights(root: &Path, p: &Value) -> Result<Finding, String> {
    let rows = read(&root.join(s(p, "integrity")?));
    let prefix = format!("| `{}", s(p, "prefix")?);
    let stamp_dir = root.join(s(p, "stamp_dir")?);
    let cache_path = root.join(s(p, "cache")?);
    let cached = std::fs::read(&cache_path).ok().and_then(|b| nmjson::parse(&b).ok());
    let mut cache: BTreeMap<String, (u64, u64, String)> = cached
        .as_ref()
        .and_then(|c| c.get("files").ok())
        .and_then(|x| x.object().ok())
        .map(|o| o.iter().filter_map(|(k, v)| stat_of(v).map(|s| (k.clone(), s))).collect())
        .unwrap_or_default();
    let mut f = Finding::default();
    let (mut stamped, mut by_cache, mut seeded) = (0, 0, 0);
    for line in rows.lines().filter(|l| l.starts_with(prefix.as_str())) {
        let cells: Vec<&str> = line.split('`').collect();
        let (Some(path), Some(sha)) = (cells.get(1), cells.get(3)) else { continue };
        // A `models/` row is a path in the models folder, wherever the models are kept.
        let file = path.strip_prefix("models/").map_or_else(|| root.join(path), |rest| nmtype::models_dir(root).join(rest));
        let Ok(meta) = std::fs::metadata(&file) else {
            f.add(&format!("missing:{path}"), format!("{path} is listed in INTEGRITY.md and missing"));
            continue;
        };
        let now = (meta.len(), mtime(&meta), (*sha).to_owned());
        let mut parts = path.split('/');
        let (id, name) = (parts.nth(1).unwrap_or_default(), path.rsplit('/').next().unwrap_or_default());
        let stamp = [format!("{id}--{name}.json"), format!("{id}.json")]
            .into_iter()
            .filter_map(|n| std::fs::read(stamp_dir.join(n)).ok())
            .filter_map(|b| nmjson::parse(&b).ok())
            .filter_map(|v| stat_of(&v))
            .find(|x| x.2 == *sha);
        let known = stamp.clone().or_else(|| cache.get(*path).filter(|c| c.2 == *sha).cloned());
        match known {
            Some(k) if k == now => {
                if stamp.is_some() { stamped += 1 } else { by_cache += 1 }
            },
            Some(k) => f.add(&format!("changed:{path}"), format!("{path}: size or modification time differs from the cached record ({} bytes, mtime {} now {} bytes, mtime {}); run check integrity", k.0, k.1, now.0, now.1)),
            None => {
                cache.insert((*path).to_owned(), now);
                seeded += 1;
            },
        }
    }
    if seeded > 0 {
        let body: Vec<String> =
            cache.iter().map(|(k, (b, m, h))| format!("    \"{k}\": {{\"bytes\": {b}, \"mtime\": {m}, \"sha256\": \"{h}\"}}")).collect();
        if let Some(d) = cache_path.parent() {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        std::fs::write(&cache_path, format!("{{\n  \"version\": 1,\n  \"note\": \"C-WEIGHTS cheap form: size and modification time of each weights row of INTEGRITY.md, keyed with its recorded hash; disposable, reseeded when missing.\",\n  \"files\": {{\n{}\n  }}\n}}\n", body.join(",\n"))).map_err(|e| format!("{}: {e}", cache_path.display()))?;
    }
    f.notes.push(format!(
        "cheap form (size and modification time, no hash): {stamped} by NPCS stamp, {by_cache} by cache, {seeded} newly seeded"
    ));
    Ok(f)
}

/// C-NUMBERS.
pub(super) fn numbers(root: &Path, p: &Value) -> Result<Finding, String> {
    let mut f = Finding::default();
    let (mut fixed, mut derived, mut gone) = (0, 0, 0);
    for c in list(p, "constants") {
        let (id, what, file) = (s(c, "id")?, s(c, "what")?, s(c, "file")?);
        let text: String = read(&root.join(&file)).lines().filter(|l| !is_comment(l)).collect::<Vec<_>>().join("\n");
        let (fx, dv) = (so(c, "fixed"), so(c, "derived"));
        if !dv.is_empty() && text.contains(dv.as_str()) {
            derived += 1;
            f.notes.push(format!("{id}: derived ({what}; {dv} in {file})"));
        } else if !fx.is_empty() && text.contains(fx.as_str()) {
            fixed += 1;
            f.add(&id, format!("{what} is a fixed number in {file}"));
        } else {
            gone += 1;
            f.notes.push(format!("{id}: not found in {file} (renamed, removed or converted: look again by hand)"));
        }
    }
    for x in list(p, "exempt") {
        f.notes.push(format!("exempt: {} ({})", so(x, "what"), so(x, "why")));
    }
    f.notes.insert(0, format!("{fixed} fixed, {derived} derived, {gone} not found"));
    Ok(f)
}

/// Does `line` hold an absolute path (drive letter plus slash, UNC, or a home folder)?
fn absolute(line: &str) -> bool {
    let b = line.as_bytes();
    let drive = (1..b.len().saturating_sub(1)).any(|i| {
        b[i] == b':'
            && b[i - 1].is_ascii_alphabetic()
            && (b[i + 1] == b'/' || b[i + 1] == b'\\')
            && (i < 2 || !(b[i - 2].is_ascii_alphanumeric() || b[i - 2] == b'_' || b[i - 2] == b'\\'))
    });
    // Built at run time so this file does not match itself. Backslash escapes and the
    // verbatim prefix that path code trims are not paths, so only drive letters and home
    // folders count.
    let homes = ["Users", "home"].map(|h| format!("/{h}/"));
    drive || homes.iter().any(|h| line.contains(h.as_str()))
}

/// C-DOWNLOAD.
pub(super) fn download(root: &Path, p: &Value) -> Result<Finding, String> {
    let mut f = Finding::default();
    let mut files = Vec::new();
    for sc in list(p, "scan") {
        let path = root.join(s(sc, "path")?);
        if path.is_file() {
            files.push(path);
        } else {
            walk(&path, &strs(sc, "ext"), &mut files);
        }
    }
    let canon = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    for file in &files {
        let r = rel(root, file);
        let text = read(file);
        // A section check's fixtures (from its `fn check(` line on) use synthetic paths.
        let end = text.lines().position(|l| l.contains("fn check(")).unwrap_or(usize::MAX);
        for (i, line) in text.lines().enumerate() {
            if i < end && !is_comment(line) && absolute(line) {
                f.add(&format!("abs:{r}:{}", i + 1), format!("{r}:{} holds an absolute path", i + 1));
            }
            if r.ends_with("Cargo.toml")
                && let Some(dep) = line.split("path = \"").nth(1).and_then(|x| x.split('"').next())
                && !file.parent().unwrap_or(root).join(dep).canonicalize().is_ok_and(|c| c.starts_with(&canon))
            {
                f.add(&format!("cargo-path:{r}:{dep}"), format!("{r}: path dependency {dep} resolves outside the Nano folder"));
            }
        }
    }
    let mut links = 0;
    let mut level: Vec<std::path::PathBuf> = vec![root.to_path_buf()];
    for _ in 0..2 {
        let mut next = Vec::new();
        for d in &level {
            for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
                let path = e.path();
                if std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
                    links += 1;
                    if !path.canonicalize().is_ok_and(|c| c.starts_with(&canon)) {
                        f.add(&format!("link:{}", rel(root, &path)), format!("{} links outside the Nano folder", rel(root, &path)));
                    }
                } else if path.is_dir() {
                    next.push(path);
                }
            }
        }
        level = next;
    }
    f.notes.push(format!("{} files scanned, {links} links seen", files.len()));
    Ok(f)
}
