//! Territories (guide 4.1, 17.1; PLAN 4d item 8): non-overlapping regions for one task,
//! coarse first. A territory never changes the anatomy; it is a plan for one task.
use nmtype::Region;
use std::collections::BTreeMap;

/// Folder prefix of a path at `depth` components (`a/b/c.rs`, 2 gives `a/b/`).
fn prefix(path: &str, depth: usize) -> String {
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() <= depth {
        return path.to_owned();
    }
    let mut p = parts.iter().take(depth).copied().collect::<Vec<_>>().join("/");
    p.push('/');
    p
}

/// Cut a sorted file list into `n` territories of folder prefixes: groups at depth 2 are
/// dealt largest first to the territory with the fewest files, so territories never overlap
/// and together cover every file.
#[must_use]
pub fn partition(files: &[String], n: usize) -> Vec<Vec<Region>> {
    let n = n.max(1);
    let mut groups: BTreeMap<String, usize> = BTreeMap::new();
    for f in files {
        *groups.entry(prefix(f, 2)).or_default() += 1;
    }
    let mut order: Vec<(String, usize)> = groups.into_iter().collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut out: Vec<Vec<Region>> = vec![Vec::new(); n];
    let mut load = vec![0_usize; n];
    for (path, count) in order {
        let i = (0..n).min_by_key(|i| (load[*i], *i)).unwrap_or(0);
        load[i] += count;
        let territory = u8::try_from(i).unwrap_or(u8::MAX);
        #[expect(clippy::cast_precision_loss, reason = "file counts are small")]
        let priority = count as f32;
        out[i].push(Region { territory, path, start: 0, end: count, priority });
    }
    out
}

/// Self-check of the `territory` sub-component: territories cover every file exactly once.
pub(crate) fn check() -> Result<(), String> {
    let files: Vec<String> =
        ["a/b/1.rs", "a/b/2.rs", "a/c/3.rs", "d/4.rs", "e/f/g/5.rs", "top.md"].iter().map(|s| (*s).to_owned()).collect();
    let parts = partition(&files, 3);
    for f in &files {
        let owners = parts.iter().filter(|t| t.iter().any(|r| f.starts_with(r.path.as_str()))).count();
        if owners != 1 {
            return Err(format!("nmpod.territory: {f} owned by {owners} territories"));
        }
    }
    Ok(())
}
