//! One writer per memory file. The journal is the system of record: its views (the tier folders
//! `1-core/`, `2-active/<topic>/`, `3-archive/<topic>/`, `index.json`, `temp/captured.json`,
//! `undo.json`, `snapshot.json`) are written by `book::Book::write_views` alone. The hierarchy of
//! use counts keeps `hierarchy.json` and the temporary notes keep `temp/notes.json`; neither
//! writes nor reads the journal's views as its own.
//!
//! An older layout let the hierarchy and the book write the same tier files, so one item could be
//! listed in two tiers and each side could read the other's records. `migrate` brings such a
//! folder to the new layout at open, before anything writes: it keeps a copy of every tier file,
//! the index and the notes file as they were (`migrated/one-writer-<ms>.json`, written once, never
//! read by the memory), takes the hierarchy's own items into `hierarchy.json`, and writes the notes
//! file back in its own form. The book's next `write_views` then rewrites the tier folders from
//! the journal and empties any tier file that holds no memory.
use crate::hierarchy::{Hierarchy, HierarchyPolicy, is_own_item, tier_files};
use crate::store::{self, obj, s, u};
use crate::temp::{Temp, TempPolicy};
use nmjson::Value;
use nmtype::Fault;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// What one migration did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// The copy of the mixed files, when the folder was in the older layout.
    pub kept: Option<PathBuf>,
    /// Hierarchy items taken from the tier files into `hierarchy.json`.
    pub adopted: usize,
    /// The notes file held the book's form and was written back in its own.
    pub notes_rewritten: bool,
}

/// True when the folder is in the older mixed layout: a tier file holds a hierarchy item, or the
/// notes file holds another writer's form.
#[must_use]
pub fn is_mixed(dir: &Path) -> bool {
    let own_in_tiers = tier_files(dir)
        .iter()
        .any(|(_, f)| store::get(f).is_some_and(|v| v.get("items").and_then(Value::array).unwrap_or_default().iter().any(is_own_item)));
    let notes = store::get(&dir.join("temp").join("notes.json")).is_some_and(|v| v.get("notes").is_err() || v.get("items").is_ok());
    own_in_tiers || notes
}

/// Keep a copy of every tier file, `index.json` and `temp/notes.json` of a mixed folder in
/// `migrated/one-writer-<ms>.json` (each file's JSON under its path relative to `dir`). Returns
/// the copy's path; nothing when the folder is not mixed.
/// # Errors
/// The copy cannot be written.
pub fn keep_mixed(dir: &Path) -> Result<Option<PathBuf>, Fault> {
    if !is_mixed(dir) {
        return Ok(None);
    }
    let mut files: BTreeMap<String, Value> = BTreeMap::new();
    let others = [dir.join("index.json"), dir.join("temp").join("notes.json")];
    for f in tier_files(dir).into_iter().map(|(_, f)| f).chain(others.into_iter().filter(|f| f.is_file())) {
        let rel = f.strip_prefix(dir).unwrap_or(&f).to_string_lossy().replace('\\', "/");
        let v = store::get(&f).unwrap_or_else(|| s(&String::from_utf8_lossy(&std::fs::read(&f).unwrap_or_default())));
        files.insert(rel, v);
    }
    let at = nmtype::now_ms();
    let folder = dir.join("migrated");
    let mut path = folder.join(format!("one-writer-{at}.json"));
    let mut n = 1;
    while path.exists() {
        path = folder.join(format!("one-writer-{at}-{n}.json"));
        n += 1;
    }
    store::put_safe(
        &path,
        &obj(vec![
            ("record", s("memory-migration")),
            ("to", s("one writer per file: tier views from the journal, hierarchy.json, temp/notes.json")),
            ("at", u(at)),
            ("files", Value::Object(files)),
        ]),
    )?;
    Ok(Some(path))
}

/// Bring a memory folder to the one-writer layout and open the hierarchy and the temporary notes
/// on it (see the module text). A folder already in the layout is only opened.
/// # Errors
/// The copy, `hierarchy.json` or the notes file cannot be written.
pub fn migrate(dir: &Path, hp: HierarchyPolicy, tp: TempPolicy) -> Result<(Hierarchy, Temp, Report), Fault> {
    let kept = keep_mixed(dir)?;
    let mut h = Hierarchy::open(dir, hp);
    let adopted = h.adopted();
    h.settle()?;
    let mut t = Temp::open(dir, tp);
    let notes_rewritten = t.settle()?;
    Ok((h, t, Report { kept, adopted, notes_rewritten }))
}

/// Item ids listed in more than one tier (core, active, archive) of a memory folder's tier files.
#[must_use]
pub fn tiers_overlap(dir: &Path) -> usize {
    let mut per: [BTreeSet<String>; 3] = Default::default();
    for (level, f) in tier_files(dir) {
        if let (Some(set), Some(v)) = (per.get_mut(level), store::get(&f)) {
            set.extend(
                v.get("items").and_then(Value::array).unwrap_or_default().iter().map(|x| store::text(x, "id")).filter(|x| !x.is_empty()),
            );
        }
    }
    let mut seen: BTreeMap<&String, usize> = BTreeMap::new();
    for id in per.iter().flatten() {
        *seen.entry(id).or_default() += 1;
    }
    seen.values().filter(|c| **c > 1).count()
}

/// Every id listed in a memory folder's tier files and notes file.
#[must_use]
pub fn listed_ids(dir: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for f in tier_files(dir).into_iter().map(|(_, f)| f).chain([dir.join("temp").join("notes.json"), crate::book::captured_file(dir)]) {
        if let Some(v) = store::get(&f) {
            for k in ["items", "notes"] {
                out.extend(
                    v.get(k).and_then(Value::array).unwrap_or_default().iter().map(|x| store::text(x, "id")).filter(|x| !x.is_empty()),
                );
            }
        }
    }
    out
}

/// Section check: a folder in the older mixed layout (the book's view in `2-active/context`, the
/// hierarchy's file in `3-archive/context` holding its own item and a copy of a journal record,
/// the book's temporary tier in `temp/notes.json`) lists an item in two tiers; after `migrate` and
/// the journal's open the overlap is 0, every id is kept (journal, hierarchy or notes), the
/// hierarchy item keeps its counts, the copy holds every old file, and a second open changes
/// nothing.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    let e = |x: Fault| x.to_string();
    let root = nmtype::check_dir("nmmem-onewriter-root").map_err(|x| x.to_string())?;
    std::fs::create_dir_all(root.join("src")).map_err(|x| x.to_string())?;
    std::fs::write(root.join("src/x.rs"), "pub fn decide_switch(x: u8) -> bool {\n    x > 3\n}\n").map_err(|x| x.to_string())?;
    let dir = nmtype::check_dir("nmmem-onewriter").map_err(|x| x.to_string())?;
    let ep = crate::episodic::Episode::new(
        "neuromorph",
        "b0",
        "where is the switch decided",
        "src/x.rs:1 decide_switch",
        vec!["src/x.rs:1 lines 1..3".into()],
        1,
    );
    crate::episodic::Episodic::open(&dir).write(ep.clone()).map_err(e)?;
    let rid = crate::mem2::episode_record_id(&ep.id);
    {
        let mut m = crate::mem2::Memory2::open(&dir, &root).map_err(e)?;
        m.capture("the switch reads x above 3", "tagged").map_err(e)?;
    }
    // The older layout: the hierarchy's file in the archive (its own item with counts, plus the
    // copy of the journal record it once read as its own), the temporary tier's view in notes.json.
    let rev1 = format!(
        r#"{{"record":"memory-items","level":"3-archive","topic":"context","items":[
        {{"id":"{}","topic":"context","query":"where is the switch decided","text":"src/x.rs:1","source":"b0","uses":5,"useful":2,"failures":1,"first":10,"last":20}},
        {{"id":"{rid}","topic":"context","query":"","text":"copy","source":"","uses":0,"useful":0,"failures":0,"first":0,"last":0}}]}}"#,
        ep.id
    );
    std::fs::create_dir_all(dir.join("3-archive/context")).map_err(|x| x.to_string())?;
    std::fs::write(dir.join("3-archive/context/items.json"), &rev1).map_err(|x| x.to_string())?;
    let captured = std::fs::read(crate::book::captured_file(&dir)).map_err(|x| x.to_string())?;
    std::fs::write(dir.join("temp/notes.json"), &captured).map_err(|x| x.to_string())?;
    let before = listed_ids(&dir);
    if tiers_overlap(&dir) != 1 || !is_mixed(&dir) {
        return Err(format!("nmmem.onewriter: the mixed folder was not mixed (overlap {})", tiers_overlap(&dir)));
    }
    let (h, t, rep) = migrate(&dir, HierarchyPolicy::default(), TempPolicy::default()).map_err(e)?;
    let m = crate::mem2::Memory2::open(&dir, &root).map_err(e)?;
    if tiers_overlap(&dir) != 0 || is_mixed(&dir) {
        return Err(format!("nmmem.onewriter: overlap {} after migration", tiers_overlap(&dir)));
    }
    let lost: Vec<&String> = before
        .iter()
        .filter(|id| !m.book.records.contains_key(*id) && h.get(id).is_none() && !t.list().iter().any(|n| &n.id == *id))
        .collect();
    if !lost.is_empty() {
        return Err(format!("nmmem.onewriter: ids lost by the migration: {lost:?}"));
    }
    let item = h.get(&ep.id).map(|i| (i.level, i.uses, i.useful, i.failures, i.last));
    if item != Some((2, 5, 2, 1, 20)) || h.get(&rid).is_some() || rep.adopted != 1 || !rep.notes_rewritten {
        return Err(format!("nmmem.onewriter: hierarchy item {item:?}, report {rep:?}"));
    }
    let kept = rep.kept.as_ref().and_then(|p| store::get(p)).ok_or("no copy of the mixed files")?;
    let files = kept.get("files").map_err(|x| x.to_string())?;
    if files.get("3-archive/context/items.json").is_err()
        || files.get("temp/notes.json").is_err()
        || files.get("2-active/context/items.json").is_err()
    {
        return Err("nmmem.onewriter: the copy misses an old file".into());
    }
    drop(m);
    let snapshot = |d: &Path| -> Vec<(String, Vec<u8>)> {
        let mut v: Vec<(String, Vec<u8>)> = tier_files(d)
            .into_iter()
            .map(|(_, f)| f)
            .chain([d.join("index.json"), d.join("hierarchy.json"), d.join("temp/notes.json")])
            .map(|f| (f.to_string_lossy().into_owned(), std::fs::read(&f).unwrap_or_default()))
            .collect();
        v.sort();
        v
    };
    let first = snapshot(&dir);
    let (_, _, again) = migrate(&dir, HierarchyPolicy::default(), TempPolicy::default()).map_err(e)?;
    drop(crate::mem2::Memory2::open(&dir, &root).map_err(e)?);
    if again != Report::default() || snapshot(&dir) != first {
        return Err(format!("nmmem.onewriter: a second open changed the folder ({again:?})"));
    }
    Ok(())
}
