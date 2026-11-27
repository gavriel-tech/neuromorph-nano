//! Unit cuts part test UC1 (2026-11-10; Gavriel Technologies, architecture by Luis Jake Gabriel
//! III). One variable: where the files of the saved tree are cut into units. Processor only: the
//! release tokenizer and token table for the drones' intake, no model run. The drones' BM25F
//! order (`nmpod::intake`, unchanged) ranks the units of each cut for the 30 development
//! questions. Hit rule, fixed before the run: a unit is right when it lies in a file the
//! reviewers named for the question and its lines overlap the reviewers' line range (the
//! question's `file` and `lines`, or any entry of `also`). Cuts compared: (a) 30-line windows
//! (today); (b) the existing structural `segments`; (c) `segments_budgeted` (cAST split-then-merge,
//! opening notes and imports as their own unit) at the budget derived from the mean size of
//! today's units; (d) the same at cAST's published 2,000 non-whitespace characters. Questions
//! are named by index and id; no question text is written. Nothing is written to disk but the
//! report.
//!
//! Run: `cargo test --release -p nmapi --test unit_cuts -- --ignored --nocapture`
//! (`NM_UC_N=2` for the smoke run). Writes `runs/2026-11-10-UC1-unit-cuts-<n>q-<ms>.txt`.
use nmapi::scan_dump;
use nmpod::intake::{Bf16Table, Drone, Kind, Rarity, Record, Source};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

type R<T> = Result<T, String>;
/// A reviewers' answer range: file path, first line, last line.
type Key = (String, usize, usize);

/// The development half by index: (id, question text, answer ranges), as the locked harness
/// orders it.
fn dev_half(home: &Path) -> R<Vec<(String, String, Vec<Key>)>> {
    const TYPES: [&str; 5] = ["behaviour", "name", "value", "explain", "relation"];
    let mut out = Vec::new();
    for area in ["context", "engine", "tools"] {
        let p = home.join(format!("tests/capability-gates/cx1-context/questions-{area}.json"));
        let v = nmjson::parse(&std::fs::read(&p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let mut qs = Vec::new();
        for x in v.get("questions").and_then(|y| y.array()).map_err(|e| e.to_string())? {
            let s = |k: &str| x.get(k).and_then(|y| y.string()).map(str::to_owned).map_err(|e| e.to_string());
            let range = |y: &nmjson::Value| -> R<Key> {
                let f = y.get("file").and_then(|z| z.string()).map_err(|e| e.to_string())?.to_owned();
                let l: Vec<usize> = y
                    .get("lines")
                    .and_then(|z| z.array())
                    .map_err(|e| e.to_string())?
                    .iter()
                    .filter_map(|z| z.integer().ok().and_then(|n| usize::try_from(n).ok()))
                    .collect();
                Ok((f, l.first().copied().unwrap_or(1), l.last().copied().unwrap_or(1)))
            };
            let mut keys = vec![range(x)?];
            if let Ok(also) = x.get("also").and_then(|y| y.array()) {
                for a in also {
                    keys.push(range(a)?);
                }
            }
            let t = s("type")?;
            qs.push((TYPES.iter().position(|y| *y == t).unwrap_or(usize::MAX), s("id")?, s("q")?, keys));
        }
        qs.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        out.extend(qs.into_iter().enumerate().filter(|(pos, _)| pos % 2 == 0).map(|(_, (_, id, q, k))| (id, q, k)));
    }
    Ok(out)
}

/// One cut of the tree: per unit (file index, first line, last line).
struct Cut {
    name: String,
    units: Vec<(usize, usize, usize)>,
}

#[test]
#[ignore = "part test UC1: needs the saved scans and the release tokenizer and table; run by hand"]
#[expect(clippy::too_many_lines, reason = "one study with its report")]
fn unit_cuts_uc1() -> R<()> {
    let home = nmtype::nano_root();
    let dir = home.join("scratch/tc1-dump/1794288995065");
    let n_want: usize = std::env::var("NM_UC_N").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let started = Instant::now();
    let items = dev_half(&home)?;
    let sub = scan_dump::load_substrate(&dir).map_err(|e| e.to_string())?;
    // Each file's lines, rebuilt from the saved 30-line units in order.
    let mut by_file: BTreeMap<usize, Vec<(usize, &str)>> = BTreeMap::new();
    for c in &sub.chunks {
        by_file.entry(c.file).or_default().push((c.start, c.text.as_str()));
    }
    let mut texts: Vec<Vec<&str>> = vec![Vec::new(); sub.files.len()];
    for (f, mut parts) in by_file {
        parts.sort_unstable_by_key(|p| p.0);
        if let Some(slot) = texts.get_mut(f) {
            for (_, t) in parts {
                slot.extend(t.lines());
            }
        }
    }
    let nws = |f: usize, s: usize, e: usize| -> usize {
        texts.get(f).map_or(0, |l| l.iter().skip(s - 1).take(e + 1 - s).map(|x| x.chars().filter(|c| !c.is_whitespace()).count()).sum())
    };
    let today: Vec<(usize, usize, usize)> = sub.chunks.iter().map(|c| (c.file, c.start, c.end)).collect();
    let mean = today.iter().map(|u| nws(u.0, u.1, u.2)).sum::<usize>() / today.len().max(1);
    let item_lines = |path: &str| -> Vec<usize> { sub.symbols.iter().filter(|s| s.file == path).map(|s| s.line).collect() };
    type Cutter<'a> = dyn Fn(&str, &[&str], &[usize]) -> Vec<(usize, usize)> + 'a;
    let cut_with = |f: &Cutter<'_>| -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for (i, file) in sub.files.iter().enumerate() {
            let lines = texts.get(i).cloned().unwrap_or_default();
            if lines.is_empty() {
                continue;
            }
            let its = if file.path.ends_with(".rs") { item_lines(&file.path) } else { Vec::new() };
            out.extend(f(&file.path, &lines, &its).into_iter().map(|(s, e)| (i, s, e)));
        }
        out
    };
    let cuts = vec![
        Cut { name: "(a) 30-line windows (today)".into(), units: today.clone() },
        Cut { name: "(b) existing structural segments".into(), units: cut_with(&|p, l, i| nmsubs::segments(p, l, i)) },
        Cut {
            name: format!("(c) budgeted structural units, budget {mean} (mean of today's units)"),
            units: cut_with(&|p, l, i| nmsubs::segments_budgeted(p, l, i, mean)),
        },
        Cut {
            name: "(d) budgeted structural units, budget 2000 (cAST)".into(),
            units: cut_with(&|p, l, i| nmsubs::segments_budgeted(p, l, i, 2000)),
        },
    ];
    let wdir = nmtype::models_dir(&home).join("qwen3-0.6b/files");
    let tok = nmtokz::Tokenizer::open(&wdir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let drone = Drone::new(Arc::new(tok), Bf16Table::open(&wdir).map_err(|e| e.to_string())?);
    let qidx: Vec<usize> = scan_dump::indices(&dir).into_iter().filter(|i| items.get(*i).is_some()).take(n_want).collect();

    let mut out = String::new();
    let total_lines: usize = texts.iter().map(Vec::len).sum();
    let _ = writeln!(
        out,
        "UC1 unit cuts, processor only; dump 1794288995065; {} files, {total_lines} lines; order = drones' BM25F (boost 5), unchanged; hit rule: a unit in a file the reviewers named whose lines overlap their range",
        sub.files.len()
    );
    let _ = writeln!(
        out,
        "right-file misses of the three-judge form (question numbers 6, 8, 11, 13, 18, 24) are listed per cut as place of the first right unit"
    );
    for cut in &cuts {
        let t = Instant::now();
        let covered: usize = cut.units.iter().map(|u| u.2 + 1 - u.1).sum();
        let mut sizes: Vec<usize> = cut.units.iter().map(|u| nws(u.0, u.1, u.2)).collect();
        sizes.sort_unstable();
        let mut pieces: Vec<Record> = Vec::with_capacity(cut.units.len());
        for (i, (f, s, e)) in cut.units.iter().enumerate() {
            let path = sub.files.get(*f).map(|x| x.path.clone()).unwrap_or_default();
            let text =
                texts.get(*f).map(|l| l.iter().skip(s - 1).take(e + 1 - s).copied().collect::<Vec<_>>().join("\n")).unwrap_or_default();
            pieces.push(
                drone
                    .take_in(Kind::Code, Source::Lines { path, start: *s, end: *e }, i + 1, &text)
                    .map_err(|e| e.to_string())?
                    .into_inner(),
            );
        }
        let rarity = Rarity::from_records(pieces.iter());
        let (mut places, mut whole, mut keys_n) = (Vec::new(), 0_usize, 0_usize);
        let mut focus = Vec::new();
        for &qi in &qidx {
            let Some((id, text, keys)) = items.get(qi) else { continue };
            let req = drone
                .take_in(Kind::Request, Source::Message { label: format!("q{qi:02}") }, 0, text)
                .map_err(|e| e.to_string())?
                .into_inner();
            let mut probe = drone.probe(&req, &rarity);
            let score: Vec<f32> = pieces.iter().map(|p| probe.sense(p).bm25f).collect();
            let right = |u: &(usize, usize, usize)| {
                keys.iter().any(|(f, a, b)| sub.files.get(u.0).is_some_and(|x| &x.path == f) && u.1 <= *b && u.2 >= *a)
            };
            let mut idx: Vec<usize> = (0..score.len()).collect();
            idx.sort_by(|a, b| score.get(*b).copied().unwrap_or(0.0).total_cmp(&score.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
            let place = idx.iter().position(|i| cut.units.get(*i).is_some_and(right)).map_or(usize::MAX, |p| p + 1);
            places.push(place);
            for (f, a, b) in keys {
                keys_n += 1;
                whole += usize::from(cut.units.iter().any(|u| sub.files.get(u.0).is_some_and(|x| &x.path == f) && u.1 <= *a && u.2 >= *b));
            }
            if [5, 7, 10, 12, 17, 23].contains(&qi) {
                focus.push(format!("q{} {id}: {place}", qi + 1));
            }
        }
        let within = |m: usize| places.iter().filter(|r| **r <= m).count();
        let _ = writeln!(
            out,
            "{}: {} units, lines covered {covered} of {total_lines}; non-whitespace size median {} largest {}; answer ranges inside one unit {whole} of {keys_n}",
            cut.name,
            cut.units.len(),
            sizes.get(sizes.len() / 2).copied().unwrap_or(0),
            sizes.last().copied().unwrap_or(0)
        );
        let _ = writeln!(
            out,
            "  right unit first {} top 6 {} top 12 {} top 50 {} of {}; right-file misses: {}; {:.1} s",
            within(1),
            within(6),
            within(12),
            within(50),
            places.len(),
            focus.join(", "),
            t.elapsed().as_secs_f64()
        );
    }
    let _ = writeln!(out, "whole test {:.1} s", started.elapsed().as_secs_f64());
    print!("{out}");
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    std::fs::write(home.join(format!("runs/2026-11-10-UC1-unit-cuts-{}q-{ms}.txt", qidx.len())), &out).map_err(|e| e.to_string())?;
    Ok(())
}
