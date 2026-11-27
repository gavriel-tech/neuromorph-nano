//! The nightly pass (2026-11-12, memory revision 2, sections 7, 9 and 11): once per run-day, from
//! the book's state, the events that keep the memory in shape. Pure: it reads the book and returns
//! journal events; the caller appends them and the book replays them, so every step is logged and
//! the state changes ones (tier, state) can be undone.
//!
//! Steps, in order:
//! 1. Retention by tier (run-days since last use against each tier's threshold): a temporary note
//!    used on 3 run-days or more is saved into 2-active; an unused one past its 7 run-days is
//!    archived (out of the views); 1-core past its threshold drops to 2-active; 2-active past its
//!    threshold drops to 3-archive. Pinned memories never move.
//! 2. Rising: an active memory used on 3 run-days or more whose stability has reached core's
//!    starting stability rises into 1-core.
//! 3. Duplicates (same content hash): the later ones are compacted into the earliest, with a link.
//! 4. General notes: when 3 or more live memories rest on the same named part of the same source,
//!    one general memory is written that links to them as its evidence (built from their anchors,
//!    no model writes it).
//! 5. Salience: pending salience is folded into stability and cleared.
//! 6. Homeostasis: 1-core holds at most `CORE_CAP`; the weakest above it go down to 2-active.
use crate::book::Book;
use crate::classes::{may_drop, may_merge};
use crate::record::{Class, Kind, Record, State, Tier};
use crate::retention::{Level, S0_ACTIVE, S0_CORE, leaves};
use crate::store::{obj, s};
use nmjson::Value;
use std::collections::BTreeMap;

/// Most memories 1-core holds.
pub const CORE_CAP: usize = 32;
/// Run-days of use a temporary note needs to be saved, and an active memory to rise.
pub const SAVE_DAYS: u64 = 3;
/// Live memories sharing one named source part before a general note is written.
pub const GENERAL_MIN: usize = 3;

fn change(kind: &str, r: &Record, to: &str, from: &str, why: &str) -> (String, Value) {
    (kind.to_owned(), obj(vec![("id", s(&r.id)), ("to", s(to)), ("from", s(from)), ("why", s(why))]))
}

/// The events of one nightly pass on run-day `day`.
#[must_use]
pub fn nightly(book: &Book, day: u64) -> Vec<(String, Value)> {
    let mut out: Vec<(String, Value)> = Vec::new();
    let live = |r: &&Record| matches!(r.state, State::Active);
    let unused = |r: &Record| day.saturating_sub(r.retention.last_use_day.max(r.validity.recorded_day));
    let mut tier_now: BTreeMap<String, Tier> = book.records.values().map(|r| (r.id.clone(), r.tier)).collect();
    // 1 and 2: retention and rising.
    for r in book.records.values().filter(live).filter(|r| may_drop(r.class)) {
        let t = unused(r);
        let st = r.retention.stability;
        match r.tier {
            Tier::Temp if r.retention.days_used >= SAVE_DAYS => {
                out.push(change("tier", r, Tier::Active.label(), Tier::Temp.label(), "temporary note used on 3 run-days: saved"));
                tier_now.insert(r.id.clone(), Tier::Active);
            }
            Tier::Temp if leaves(Level::Temp, t, st) => {
                out.push(change("state", r, State::Archived.label(), State::Active.label(), "temporary note unused 7 run-days: cleared"))
            }
            Tier::Core if leaves(Level::Core, t, st) => {
                out.push(change("tier", r, Tier::Active.label(), Tier::Core.label(), "1-core unused past its 28 run-days"));
                tier_now.insert(r.id.clone(), Tier::Active);
            }
            Tier::Active if leaves(Level::Active, t, st) => {
                out.push(change("tier", r, Tier::Archive.label(), Tier::Active.label(), "2-active unused past its 14 run-days"));
                out.push(change("state", r, State::Archived.label(), State::Active.label(), "moved to the archive"));
                tier_now.insert(r.id.clone(), Tier::Archive);
            }
            Tier::Active if r.retention.days_used >= SAVE_DAYS && st >= S0_CORE => {
                out.push(change("tier", r, Tier::Core.label(), Tier::Active.label(), "used on 3 run-days and as stable as core"));
                tier_now.insert(r.id.clone(), Tier::Core);
            }
            _ => {}
        }
    }
    // 3: duplicates by content hash, kept in the earliest.
    let mut by_content: BTreeMap<&str, Vec<&Record>> = BTreeMap::new();
    for r in book.records.values().filter(live) {
        by_content.entry(r.content.as_str()).or_default().push(r);
    }
    for list in by_content.values_mut().filter(|l| l.len() > 1) {
        list.sort_by(|a, b| a.validity.recorded_day.cmp(&b.validity.recorded_day).then(a.id.cmp(&b.id)));
        let Some((keep, rest)) = list.split_first() else { continue };
        for r in rest.iter().filter(|r| may_merge(r.class)) {
            out.push(change("state", r, State::Compacted.label(), State::Active.label(), &format!("duplicate of {}", keep.id)));
            out.push(("link".into(), obj(vec![("id", s(&keep.id)), ("rel", s("merged-from")), ("to", s(&r.id))])));
        }
    }
    // 4: general notes over a shared named part.
    let mut by_part: BTreeMap<(String, String), Vec<&Record>> = BTreeMap::new();
    for r in book.records.values().filter(live).filter(|r| r.kind != Kind::Fact || !r.links.iter().any(|(rel, _)| rel == "derived-from")) {
        for a in r.anchors.iter().filter(|a| !a.name.is_empty()) {
            by_part.entry((a.location.clone(), a.name.clone())).or_default().push(r);
        }
    }
    for ((loc, name), list) in by_part.into_iter().filter(|(_, l)| l.len() >= GENERAL_MIN) {
        let gid = format!("general:{}", nmhash::sha256_bytes(format!("{loc}#{name}").as_bytes()).get(..16).unwrap_or(""));
        if book.records.contains_key(&gid) {
            continue;
        }
        let Some(first) = list.first() else { continue };
        let anchor = first.anchors.iter().find(|a| a.location == loc && a.name == name).cloned();
        let text = format!("{} memories rest on {name} in {loc}", list.len());
        let anchors: Vec<_> = anchor.into_iter().collect();
        let rec = Record {
            id: gid.clone(),
            content: Record::content_hash(Kind::Fact, &text, &anchors),
            kind: Kind::Fact,
            class: Class::Normal,
            tier: Tier::Active,
            state: State::Active,
            topic: first.topic.clone(),
            text,
            anchors,
            keys: vec![name.clone(), loc.clone()],
            links: list.iter().map(|r| ("derived-from".to_owned(), r.id.clone())).collect(),
            validity: crate::record::Validity {
                from_day: day,
                from_version: first.validity.from_version,
                recorded_day: day,
                ..Default::default()
            },
            retention: crate::record::Retention {
                stability: S0_ACTIVE,
                difficulty: crate::retention::d0(),
                last_use_day: day,
                ..Default::default()
            },
            verdicts: Vec::new(),
        };
        out.push(("record".into(), rec.to_value()));
    }
    // 5: salience folded into stability.
    for r in book.records.values().filter(live).filter(|r| r.retention.salience_pending > 0.0) {
        let st = r.retention.stability * (1.0 + crate::retention::SALIENCE_GAIN * r.retention.salience_pending.clamp(0.0, 1.0));
        out.push((
            "retention".into(),
            obj(vec![("id", s(&r.id)), ("stability", Value::Number(format!("{st}"))), ("salience_pending", Value::Number("0".into()))]),
        ));
    }
    // 6: homeostasis: core at most CORE_CAP, the weakest go down.
    let mut core: Vec<&Record> = book.records.values().filter(live).filter(|r| tier_now.get(&r.id) == Some(&Tier::Core)).collect();
    if core.len() > CORE_CAP {
        core.sort_by(|a, b| b.retention.stability.total_cmp(&a.retention.stability).then(a.id.cmp(&b.id)));
        for r in core.iter().skip(CORE_CAP).filter(|r| r.class != Class::Pinned) {
            out.push(change("tier", r, Tier::Active.label(), Tier::Core.label(), "1-core over its 32: the weakest go down"));
        }
    }
    out
}

/// Whether a memory shows in the views: live ones, and archived, superseded or invalid ones until
/// 28 run-days after they were retired (archive cleanup; the journal keeps everything).
#[must_use]
pub fn visible(r: &Record, day: u64) -> bool {
    match r.state {
        State::Active => true,
        State::Compacted => false,
        State::Archived => r.tier != Tier::Temp,
        State::Superseded | State::Invalid => {
            r.class != Class::Normal || day.saturating_sub(r.validity.retired_day.unwrap_or(day)) < crate::retention::CORE_DAYS as u64
        }
    }
}

/// Section check: on a synthetic book after 30 run-days, each rule fires once and nothing else
/// moves; a pinned memory never moves; replaying the pass's events gives the expected tiers.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    use crate::record::{Anchor, Part, Retention, Source, Validity};
    let mk = |id: &str, tier: Tier, class: Class, days_used: u64, last: u64, st: f64, content: &str, name: &str| Record {
        id: id.into(),
        content: content.into(),
        kind: Kind::Fact,
        class,
        tier,
        state: State::Active,
        topic: "context".into(),
        text: id.into(),
        anchors: vec![Anchor {
            source: Source::File,
            location: "a.rs".into(),
            part: Part::Lines { start: 1, end: 3 },
            part_hash: "h".into(),
            name: name.into(),
            excerpt: None,
        }],
        keys: Vec::new(),
        links: Vec::new(),
        validity: Validity { recorded_day: 1, ..Validity::default() },
        retention: Retention { stability: st, difficulty: 5.0, last_use_day: last, days_used, ..Retention::default() },
        verdicts: Vec::new(),
    };
    let mut b = Book::default();
    for r in [
        mk("temp-old", Tier::Temp, Class::Normal, 0, 1, crate::retention::S0_TEMP, "c1", ""),
        mk("temp-used", Tier::Temp, Class::Normal, 3, 29, 5.0, "c2", ""),
        mk("active-old", Tier::Active, Class::Normal, 1, 1, S0_ACTIVE, "c3", ""),
        mk("core-old", Tier::Core, Class::Normal, 1, 1, S0_CORE, "c4", ""),
        mk("pinned", Tier::Core, Class::Pinned, 0, 1, S0_CORE, "c5", ""),
        mk("dup-a", Tier::Active, Class::Normal, 1, 29, 30.0, "same", ""),
        mk("dup-b", Tier::Active, Class::Normal, 1, 29, 30.0, "same", ""),
        mk("g1", Tier::Active, Class::Normal, 1, 29, 30.0, "g1", "decide"),
        mk("g2", Tier::Active, Class::Normal, 1, 29, 30.0, "g2", "decide"),
        mk("g3", Tier::Active, Class::Normal, 1, 29, 30.0, "g3", "decide"),
    ] {
        b.records.insert(r.id.clone(), r);
    }
    // Promotion by use: used on 3 run-days and as stable as core. The archive is the floor: a long
    // unused archived memory gets no event and stays shown in the archive view.
    for r in [
        mk("active-used", Tier::Active, Class::Normal, 3, 29, S0_CORE + 1.0, "c6", ""),
        Record { state: State::Archived, ..mk("archive-old", Tier::Archive, Class::Normal, 0, 1, S0_ACTIVE, "c7", "") },
    ] {
        b.records.insert(r.id.clone(), r);
    }
    let ev = nightly(&b, 30);
    let untouched = |id: &str| !ev.iter().any(|(_, v)| v.get("id").and_then(Value::string).ok() == Some(id));
    let shown = b.tier(Tier::Archive, 30).values().flatten().any(|r| r.id == "archive-old");
    if !untouched("archive-old") || !shown {
        return Err("nmmem.pass: an archived memory was moved or hidden".into());
    }
    let has = |kind: &str, id: &str, to: &str| {
        ev.iter().any(|(k, v)| {
            k == kind && v.get("id").and_then(Value::string).ok() == Some(id) && v.get("to").and_then(Value::string).ok() == Some(to)
        })
    };
    let ok = has("state", "temp-old", "archived")
        && has("tier", "temp-used", "2-active")
        && has("tier", "active-old", "3-archive")
        && has("tier", "core-old", "2-active")
        && has("tier", "active-used", "1-core")
        && has("state", "dup-b", "compacted")
        && !ev.iter().any(|(_, v)| v.get("id").and_then(Value::string).ok() == Some("pinned"));
    let general = ev.iter().filter(|(k, _)| k == "record").count();
    if !ok || general != 1 {
        return Err(format!(
            "nmmem.pass: events {:?}",
            ev.iter().map(|(k, v)| format!("{k}:{}", v.get("id").and_then(Value::string).unwrap_or(""))).collect::<Vec<_>>()
        ));
    }
    let mut over = Book::default();
    for i in 0..(CORE_CAP + 3) {
        #[expect(clippy::cast_precision_loss, reason = "test stabilities")]
        let r = mk(&format!("c{i:02}"), Tier::Core, Class::Normal, 3, 30, S0_CORE + i as f64, &format!("k{i}"), "");
        over.records.insert(r.id.clone(), r);
    }
    let down =
        nightly(&over, 30).iter().filter(|(k, v)| k == "tier" && v.get("to").and_then(Value::string).ok() == Some("2-active")).count();
    if down != 3 {
        return Err(format!("nmmem.pass: homeostasis moved {down}, want 3"));
    }
    Ok(())
}
