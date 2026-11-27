//! Incremental synchronization (guide 6.1, 17.3): re-read the root, compare content hashes,
//! bump the one version counter only when something changed, and emit one `invalidation`
//! packet per changed, added or removed file so the Core Brain can drop stale packets.
use crate::Substrate;
use crate::repo::{self, FileInfo};
use nmtype::{Packet, Provenance, Status};
use std::collections::BTreeMap;

/// What a sync changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncReport {
    pub version: u64,
    pub changed: Vec<String>,
    pub added: Vec<String>,
    pub removed: Vec<String>,
}

impl Substrate {
    /// Apply a new file list: unchanged hashes keep the version; any change bumps it.
    pub fn apply(&mut self, next: Vec<FileInfo>) -> (SyncReport, Vec<Packet>) {
        let old: BTreeMap<&str, &str> = self.files.iter().map(|f| (f.path.as_str(), f.hash.as_str())).collect();
        let new: BTreeMap<&str, &str> = next.iter().map(|f| (f.path.as_str(), f.hash.as_str())).collect();
        let changed: Vec<String> = new.iter().filter(|(p, h)| old.get(*p).is_some_and(|o| o != *h)).map(|(p, _)| (*p).to_owned()).collect();
        let added: Vec<String> = new.keys().filter(|p| !old.contains_key(*p)).map(|p| (*p).to_owned()).collect();
        let removed: Vec<String> = old.keys().filter(|p| !new.contains_key(*p)).map(|p| (*p).to_owned()).collect();
        if changed.is_empty() && added.is_empty() && removed.is_empty() {
            return (SyncReport { version: self.version, changed, added, removed }, Vec::new());
        }
        self.index(next);
        self.version += 1;
        // 2026-11-10 (refinement step 1): the reference graph follows the new version when it is
        // in use; nothing happens for a substrate that never built one.
        self.refresh_refs();
        let packets = changed
            .iter()
            .chain(&added)
            .chain(&removed)
            .map(|p| {
                Packet::new(
                    "invalidation",
                    &format!("file:{p}"),
                    &format!("{p} changed at substrate version {}", self.version),
                    Provenance::new(&format!("{p}:1"), "nmsubs.sync#g0", self.version, nmtype::now_ms()),
                )
                .status(Status::Observed)
                .evidence(vec![p.clone()])
            })
            .collect();
        (SyncReport { version: self.version, changed, added, removed }, packets)
    }
    /// Re-read the root folder and apply it. Roadmap Phase 1 item 2 (2026-11-24): a fingerprint of
    /// the tree's metadata (every file's path, size and modification time, outside `.git`,
    /// `target` and `node_modules`) is taken first; when it equals the last sync's, nothing is read
    /// or hashed and the version stays (a superset of the files read, so a real change always
    /// fails the fast path). Every `FULL_EVERY`th sync reads in full regardless.
    pub fn sync(&mut self) -> (SyncReport, Vec<Packet>) {
        let fp = metadata_fingerprint(&self.root);
        let fast = {
            let mut m = FINGERPRINTS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let e = m.entry(self.root.clone()).or_insert((0, 0));
            let same = fp != 0 && e.0 == fp && !e.1.is_multiple_of(FULL_EVERY);
            e.0 = fp;
            e.1 = e.1.wrapping_add(1);
            same
        };
        if fast {
            return (SyncReport { version: self.version, changed: Vec::new(), added: Vec::new(), removed: Vec::new() }, Vec::new());
        }
        let mut files = Vec::new();
        let mut skipped = Vec::new();
        let root = self.root.clone();
        let skip = self.skip.clone();
        repo::walk_counted(&root, &root, &skip, &mut files, &mut skipped);
        self.skipped = skipped;
        self.apply(files)
    }
}

/// Every this many syncs of a root read in full even when the metadata fingerprint is unchanged
/// (an edit that keeps a file's size and modification time).
const FULL_EVERY: u32 = 20;

/// Per root: the last metadata fingerprint and the syncs counted.
static FINGERPRINTS: std::sync::Mutex<BTreeMap<std::path::PathBuf, (u64, u32)>> = std::sync::Mutex::new(BTreeMap::new());

/// A fingerprint of every file's path, size and modification time under `root` (outside the
/// folders the walk never reads), in a stable order; 0 when the root cannot be read.
fn metadata_fingerprint(root: &std::path::Path) -> u64 {
    fn walk(dir: &std::path::Path, out: &mut Vec<(String, u64, u128)>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            // The entry's own metadata: a link is not followed (no loop through a folder link).
            let Ok(md) = std::fs::symlink_metadata(&p) else { continue };
            if md.is_dir() {
                // Roadmap Phase 4.6 (2026-11-25): the folders the walk never reads (build output,
                // installed packages, caches, hidden folders unless `NANO_HIDDEN=on`) are not
                // walked here either, so a large `.venv` or `dist` costs nothing per sync.
                let hidden = name.starts_with('.') && (name == ".git" || !crate::repo::hidden_on());
                if !hidden && !crate::repo::SKIP.contains(&name.as_str()) {
                    walk(&p, out);
                }
            } else {
                let t = md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
                out.push((p.to_string_lossy().into_owned(), md.len(), t));
            }
        }
    }
    if !root.is_dir() {
        return 0;
    }
    let mut all = Vec::new();
    walk(root, &mut all);
    all.sort();
    // FNV-1a over the sorted entries.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (p, n, t) in &all {
        for b in p.bytes().chain(n.to_le_bytes()).chain(t.to_le_bytes()) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h.max(1)
}

/// Self-check of the `sync` sub-component: no change keeps the version; one changed file
/// bumps it once and yields one invalidation.
pub(crate) fn check() -> Result<(), String> {
    let embed: crate::Embed = std::sync::Arc::new(|_: &str| vec![1.0]);
    let mut s = crate::from_files(&[("a.rs", "fn a() {}"), ("b.rs", "fn b() {}")], embed);
    let same = s.files.clone();
    let (r, p) = s.apply(same);
    if r.version != 1 || !p.is_empty() {
        return Err("nmsubs.sync: unchanged files bumped the version".into());
    }
    let mut next = s.files.clone();
    if let Some(f) = next.get_mut(1) {
        f.text = "fn b2() {}".into();
        f.hash = nmhash::sha256_bytes(f.text.as_bytes());
    }
    let (r, p) = s.apply(next);
    if r.version != 2 || r.changed != ["b.rs"] || p.len() != 1 || !s.symbols.iter().any(|x| x.name == "b2") {
        return Err(format!("nmsubs.sync: one change gave {r:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Roadmap Phase 4.6 (2026-11-25): the per-sync fingerprint walks the awkward fixture (a
    /// folder link back to the root, a long path, large files) and returns, the same twice.
    #[test]
    fn sync_fingerprint_resilience() {
        let (root, _) = crate::repo::resilience_fixture("nmsubs-sync-resilience").unwrap();
        let a = super::metadata_fingerprint(&root);
        assert!(a != 0 && a == super::metadata_fingerprint(&root));
        let _ = std::fs::remove_dir_all(&root);
    }
}
