//! Moving revision 1 into revision 2 (2026-11-12, memory revision 2, section 15). When the journal
//! is empty, every revision 1 episode becomes a universal record (with its verdict as a verdict
//! event), its tier taken from the revision 1 hierarchy, and every temporary note becomes a temp
//! memory. Nothing of revision 1 is changed or removed; its files stay in place until Luis Jake
//! Gabriel III agrees to retire them.
//!
//! An episode's cited lines (`path:line lines a..b`, or `path:line text`) become file anchors whose
//! part hash is taken from the file as it is now (the code decides; a part already gone gets an
//! empty hash and fails its first re-read). The stable name is the first defined name in the part
//! (after `fn`, `struct`, `enum`, `trait`, `impl`, `mod`, `const`, `static`, `type`, `def`, `class`
//! or a Markdown heading).
use crate::episodic::{Episode, Episodic};
use crate::record::{Anchor, Class, Kind, Part, Record, Retention, Source, State, Tier, Validity, lines_hash};
use crate::retention::{S0_ACTIVE, S0_CORE, S0_TEMP, d0};
use crate::store::{obj, s};
use nmjson::Value;
use std::path::Path;

/// One cited line as an anchor, read against the source root `root`.
#[must_use]
pub fn anchor_of(cite: &str, root: &Path) -> Option<Anchor> {
    let (loc, rest) = cite.split_once(' ').unwrap_or((cite, ""));
    let (path, line) = loc.rsplit_once(':')?;
    let line: usize = line.parse().ok()?;
    let (start, end) = rest
        .strip_prefix("lines ")
        .and_then(|r| r.split_once(".."))
        .and_then(|(a, b)| Some((a.trim().parse().ok()?, b.split_whitespace().next()?.parse().ok()?)))
        .unwrap_or((line, line));
    let text = std::fs::read_to_string(root.join(path)).ok();
    let lines: Vec<&str> = text.as_deref().map(|t| t.lines().collect()).unwrap_or_default();
    let part = lines.get(start.saturating_sub(1)..end.min(lines.len())).unwrap_or(&[]);
    let name = part.iter().find_map(|l| defined_name(l)).unwrap_or_default();
    Some(Anchor {
        source: Source::File,
        location: path.to_owned(),
        part: Part::Lines { start, end },
        part_hash: if part.is_empty() { String::new() } else { lines_hash(part) },
        name,
        excerpt: None,
    })
}

/// The name a line defines, if any.
#[must_use]
pub fn defined_name(line: &str) -> Option<String> {
    let t = line.trim_start();
    // A Markdown heading is hashes then a space (`#[derive]` and `#!` are Rust attributes).
    if let Some(h) = t.strip_prefix('#') {
        let h = h.trim_start_matches('#');
        return h.starts_with(' ').then(|| h.trim().to_owned()).filter(|h| !h.is_empty());
    }
    let mut words = t.split(|c: char| !(c.is_alphanumeric() || c == '_')).filter(|w| !w.is_empty());
    while let Some(w) = words.next() {
        if matches!(w, "fn" | "struct" | "enum" | "trait" | "impl" | "mod" | "const" | "static" | "type" | "def" | "class") {
            return words.next().map(str::to_owned);
        }
    }
    None
}

/// The record of a revision 1 episode on run-day `day`, in tier `tier`.
#[must_use]
pub fn record_of(ep: &Episode, tier: Tier, day: u64, root: &Path) -> Record {
    let anchors: Vec<Anchor> = ep.evidence.iter().filter(|c| !c.starts_with("note:")).filter_map(|c| anchor_of(c, root)).collect();
    let head = ep.answer.lines().next().unwrap_or("");
    let text = format!("{} | {head}", ep.query);
    let mut keys: Vec<String> = anchors.iter().flat_map(|a| [a.location.clone(), a.name.clone()]).filter(|k| !k.is_empty()).collect();
    keys.sort();
    keys.dedup();
    let s0 = match tier {
        Tier::Core => S0_CORE,
        Tier::Temp => S0_TEMP,
        _ => S0_ACTIVE,
    };
    Record {
        id: format!("ep:{}", ep.id.get(..16).unwrap_or(&ep.id)),
        content: Record::content_hash(Kind::Fact, &text, &anchors),
        kind: Kind::Fact,
        class: if ep.correct == Some(true) { Class::Protected } else { Class::Normal },
        tier,
        state: if ep.correct == Some(false) { State::Invalid } else { State::Active },
        topic: if ep.level == "temp" { "temp".into() } else { "context".into() },
        text,
        anchors,
        keys,
        links: Vec::new(),
        validity: Validity { from_day: day, from_version: ep.version, recorded_day: day, ..Validity::default() },
        retention: Retention { stability: s0, difficulty: d0(), last_use_day: day, ..Retention::default() },
        verdicts: Vec::new(),
    }
}

/// The import events of a revision 1 memory folder `dir` (episodes with their tiers and verdicts,
/// and temporary notes), read against the source root `root`, on run-day `day`.
#[must_use]
pub fn import_events(dir: &Path, root: &Path, day: u64) -> Vec<(String, Value)> {
    let episodic = Episodic::open(dir);
    // Revision 1's level of each item (its own file, or its own items in an older layout's tier files).
    let levels = crate::hierarchy::Hierarchy::open(dir, crate::hierarchy::HierarchyPolicy::default());
    let tier_of = |id: &str| -> Tier {
        match levels.get(id).map(|i| i.level) {
            Some(0) => Tier::Core,
            Some(2) => Tier::Archive,
            _ => Tier::Active,
        }
    };
    let mut out = Vec::new();
    for ep in episodic.latest(usize::MAX) {
        let r = record_of(ep, tier_of(&ep.id), day, root);
        let id = r.id.clone();
        out.push(("record".to_owned(), r.to_value()));
        if let Some(c) = ep.correct {
            out.push(("verdict".to_owned(), obj(vec![("id", s(&id)), ("right", Value::Bool(c)), ("by", s("revision 1"))])));
        }
    }
    if let Some(v) = crate::store::get(&dir.join("temp").join("notes.json"))
        && let Ok(notes) = v.get("notes").and_then(Value::array)
    {
        for n in notes {
            let Ok(text) = n.get("text").and_then(Value::string) else { continue };
            let ep = Episode::new("temp", "temp", text, text, Vec::new(), 0);
            out.push(("record".to_owned(), record_of(&ep, Tier::Temp, day, root).to_value()));
        }
    }
    out
}

/// Section check: a cited line becomes an anchor hashed against the file with its defined name;
/// an episode becomes a record (verified right: protected; wrong: invalid); a folder's import gives
/// one record per episode plus its verdict.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    let root = nmtype::check_dir("nmmem-migrate-root").map_err(|e| e.to_string())?;
    std::fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    std::fs::write(root.join("src/x.rs"), "use a;\npub fn decide_switch(x: u8) -> bool {\n    x > 3\n}\n").map_err(|e| e.to_string())?;
    let a = anchor_of("src/x.rs:2 lines 2..4", &root).ok_or("no anchor")?;
    if a.name != "decide_switch" || a.part_hash.is_empty() || a.part != (Part::Lines { start: 2, end: 4 }) {
        return Err(format!("nmmem.migrate: anchor {a:?}"));
    }
    if defined_name("#[derive(Clone)]").is_some() || defined_name("## Gate rules").as_deref() != Some("Gate rules") {
        return Err("nmmem.migrate: attribute read as a heading, or a heading missed".into());
    }
    let mut ep = Episode::new(
        "neuromorph",
        "b0",
        "where is the switch decided",
        "src/x.rs:2 decide_switch",
        vec!["src/x.rs:2 lines 2..4".into()],
        3,
    );
    ep.correct = Some(true);
    let r = record_of(&ep, Tier::Active, 1, &root);
    if r.class != Class::Protected || r.anchors.len() != 1 || !r.keys.contains(&"decide_switch".to_owned()) {
        return Err(format!("nmmem.migrate: record {r:?}"));
    }
    let mem = nmtype::check_dir("nmmem-migrate-mem").map_err(|e| e.to_string())?;
    let mut epi = Episodic::open(&mem);
    epi.write(ep).map_err(|e| e.to_string())?;
    let ev = import_events(&mem, &root, 1);
    if ev.iter().filter(|(k, _)| k == "record").count() != 1 {
        return Err(format!("nmmem.migrate: import gave {} events", ev.len()));
    }
    Ok(())
}
