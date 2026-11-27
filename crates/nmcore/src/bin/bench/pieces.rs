//! `bench --mode pieces`: the piece model and its reference graph on one tree, measured. Prints
//! the time to index the tree (the substrate alone, the reference graph, the drones' intake), the
//! piece count by kind with a size histogram, the graph's typed edges, the graph's pass rule over a
//! question set (every answer range of a question with more than one range joined by an edge to
//! another answer range of the same question), and the request drone's kind and aspects for each
//! question.
use super::{arg, load};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// `bench --mode pieces ...`.
#[expect(clippy::too_many_lines, reason = "one report, written top to bottom")]
pub fn run(home: &Path, own: &[String]) -> Result<String, String> {
    let root = match arg(own, "--root") {
        Some(r) if Path::new(r).is_dir() => nmsubs::repo::canon(Path::new(r)),
        Some(r) => return Err(format!("--root {r} is not a folder")),
        None => home.to_path_buf(),
    };
    let qfile = arg(own, "--questions").map_or_else(|| home.join("tests/dev/questions.json"), PathBuf::from);
    let questions = load(&root, &qfile)?;
    let mut out = String::new();
    let _ = writeln!(out, "pieces: {} (NANO_PIECES {}), root {}", if nmsubs::piece::on() { "on" } else { "off" }, std::env::var("NANO_PIECES").unwrap_or_default(), root.display());

    // Index times: the substrate alone (best of three), the reference graph, the drones' intake.
    let mut scan_s = f64::MAX;
    let mut sub = None;
    for _ in 0..3 {
        let t = Instant::now();
        let s = nmsubs::Substrate::scan(&root, &["neuromorph-nano".to_owned()], std::sync::Arc::new(|_: &str| Vec::new()));
        scan_s = scan_s.min(t.elapsed().as_secs_f64());
        sub = Some(s);
    }
    let Some(sub) = sub else { return Err("no substrate".into()) };
    let t = Instant::now();
    let g = sub.refs();
    let graph_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut intake = nmapi::drone_windows::Keep::open(&nmtype::models_dir(home).join("qwen3-0.6b/files")).and_then(|mut k| k.sync(&sub).map(|_| k));
    let intake_s = t.elapsed().as_secs_f64();
    let _ = writeln!(
        out,
        "index: substrate {scan_s:.3} s (best of 3), reference graph {graph_s:.3} s, drones' intake {intake_s:.3} s{}; total {:.3} s",
        if intake.is_err() { " (intake failed)" } else { "" },
        scan_s + graph_s + intake_s
    );
    // The items (belief per item): every line of a file in exactly one item, their links to the
    // pieces, and the items' own intake through the drones.
    let t = Instant::now();
    let items = nmsubs::items::build(&sub);
    let items_s = t.elapsed().as_secs_f64();
    let (code_c, other_c, kinds) = nmsubs::items::partition(&sub, &items);
    let t = Instant::now();
    let item_intake = intake.as_mut().map_err(|e| e.to_string()).and_then(|k| k.sync_items(&sub).map_err(|e| e.to_string()));
    let item_intake_s = t.elapsed().as_secs_f64();
    let multi_piece = items.pieces_of.iter().filter(|v| v.len() > 1).count();
    let multi_item = items.of_piece.iter().filter(|v| v.len() > 1).count();
    let _ = writeln!(
        out,
        "items: {} (definitions {}, heads {}, sections {}, pieces {}), built {items_s:.3} s, intake {item_intake_s:.3} s{}; code files {}: lines in exactly one item {} of {}; other files {}: {} of {}; items meeting more than one piece {multi_piece} of {}; pieces meeting more than one item {multi_item} of {}",
        items.items.len(),
        kinds[0],
        kinds[1],
        kinds[2],
        kinds[3],
        if item_intake.is_err() { " (failed)" } else { "" },
        code_c[0],
        code_c[2],
        code_c[1],
        other_c[0],
        other_c[2],
        other_c[1],
        items.items.len(),
        sub.chunks.len()
    );

    // Pieces by kind, with sizes (tokens, characters / 4).
    let code: Vec<&nmsubs::Chunk> = sub.chunks.iter().filter(|c| sub.files.get(c.file).is_some_and(nmsubs::repo::FileInfo::is_code)).collect();
    let mut by_kind: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for c in &code {
        by_kind.entry(c.kind).or_default().push(c.tokens);
    }
    let docs = sub.chunks.len() - code.len();
    let all: Vec<usize> = code.iter().map(|c| c.tokens).collect();
    let mean = |v: &[usize]| if v.is_empty() { 0.0 } else { v.iter().sum::<usize>() as f64 / v.len() as f64 };
    let median = |v: &[usize]| {
        let mut w = v.to_vec();
        w.sort_unstable();
        w.get(w.len() / 2).copied().unwrap_or(0)
    };
    let _ = writeln!(out, "code pieces {} (documents {docs}), tokens mean {:.0} median {} largest {}", code.len(), mean(&all), median(&all), all.iter().max().copied().unwrap_or(0));
    let h = nmsubs::piece::histogram(&all);
    let _ = writeln!(
        out,
        "sizes: under {} {}, under 96 {}, under 192 {}, under 288 {}, up to {} {}, over {}",
        nmsubs::piece::SMALL,
        h[0],
        h[1],
        h[2],
        h[3],
        nmsubs::piece::MAX,
        h[4],
        h[5]
    );
    for (k, v) in &by_kind {
        let _ = writeln!(out, "  {k:<10} {:>5}  mean {:>4.0}  median {:>4}", v.len(), mean(v), median(v));
    }

    // The graph: typed edges, and its pass rule on the questions.
    let s = &g.specific;
    let _ = writeln!(out, "{}", s.describe());
    let covers = |c: usize, a: &super::Place| {
        sub.chunks.get(c).is_some_and(|ch| sub.files.get(ch.file).is_some_and(|f| f.path.replace('\\', "/") == a.file) && (a.whole_file || (ch.start <= a.last && ch.end >= a.first)))
    };
    let mut multi = 0;
    let mut joined_q = 0;
    let mut lines = Vec::new();
    for q in questions.iter().filter(|q| q.answers.len() > 1) {
        multi += 1;
        let holders: Vec<Vec<usize>> = q.answers.iter().map(|a| (0..sub.chunks.len()).filter(|c| covers(*c, a)).collect()).collect();
        let mut joined = Vec::new();
        for (r, hs) in holders.iter().enumerate() {
            let others: Vec<usize> = holders.iter().enumerate().filter(|(o, _)| *o != r).flat_map(|(_, v)| v.iter().copied()).collect();
            let ok = hs.iter().any(|h| others.contains(h) || s.edges_of(*h).iter().any(|e| others.contains(&(e.to as usize))));
            joined.push(ok);
        }
        let all_joined = joined.iter().all(|x| *x);
        if all_joined {
            joined_q += 1;
        }
        lines.push(format!(
            "  {} {} ranges, joined {}",
            q.id,
            q.answers.len(),
            joined.iter().map(|x| if *x { '1' } else { '0' }).collect::<String>()
        ));
    }
    let _ = writeln!(out, "graph rule: every answer range joined to another of its question on {joined_q} of {multi} questions with more than one range");
    for l in lines {
        let _ = writeln!(out, "{l}");
    }

    // The request drone on each question.
    let _ = writeln!(out, "request drone:");
    for q in &questions {
        let r = nmpod::intake::Request::read(&q.text, &[]);
        let _ = writeln!(out, "  {} [{}] {}  |  {}", q.id, r.kind, r.aspects.iter().map(|a| format!("({})", a.text)).collect::<Vec<_>>().join(" "), q.text);
    }
    Ok(out)
}
