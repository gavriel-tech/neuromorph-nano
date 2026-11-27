//! `bench --mode evidence`: the drones' calibrated evidence over one tree, measured on its own
//! (processor, no models). Per question: the time of the raw sensing and of the calibrated
//! evidence, the place of the best answer piece in the tree's order by the BM25F flag, by the
//! fused shares the stack reads today and by the calibrated evidence, how many of the first 12 are
//! file notes or heads, and for a question with no answer which aspects no piece is accepted for.
use super::{Question, arg, load};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// `bench --mode evidence ...`.
#[expect(clippy::too_many_lines, reason = "one report, written top to bottom")]
pub fn run(home: &Path, own: &[String]) -> Result<String, String> {
    let root = match arg(own, "--root") {
        Some(r) if Path::new(r).is_dir() => nmsubs::repo::canon(Path::new(r)),
        Some(r) => return Err(format!("--root {r} is not a folder")),
        None => home.to_path_buf(),
    };
    let qfile = arg(own, "--questions").map_or_else(|| home.join("tests/dev/questions.json"), PathBuf::from);
    let questions: Vec<Question> = load(&root, &qfile)?;
    let memory = home.join("scratch").join("bench-memory").join(nmtype::project::label(home, &root));
    let org = nmapi::forage::Organism::open_with(home, &root, &["neuromorph-nano".to_owned()], false, &memory).map_err(|e| format!("organism: {e}"))?;
    let sub = org.sub();
    let mut keep = nmapi::drone_windows::Keep::open(&nmtype::models_dir(home).join("qwen3-0.6b/files")).map_err(|e| format!("intake: {e}"))?;
    let t = Instant::now();
    keep.sync(&sub).map_err(|e| format!("intake: {e}"))?;
    let mut out = String::new();
    let _ = writeln!(out, "evidence: {} pieces (NANO_PIECES {}), intake {:.2} s", sub.chunks.len(), std::env::var("NANO_PIECES").unwrap_or_default(), t.elapsed().as_secs_f64());
    let covers = |c: usize, q: &Question| {
        sub.chunks.get(c).is_some_and(|ch| {
            sub.files.get(ch.file).is_some_and(|f| {
                let p = f.path.replace('\\', "/");
                q.answers.iter().any(|a| p == a.file && (a.whole_file || (ch.start <= a.last && ch.end >= a.first)))
            })
        })
    };
    let head = |c: usize| sub.chunks.get(c).is_some_and(|ch| ch.start <= 1 || matches!(ch.kind, "notes" | "imports"));
    let order = |v: &[f32]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|a, b| v.get(*b).copied().unwrap_or(0.0).total_cmp(&v.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
        idx
    };
    let mut tally = [[0_usize; 3]; 6];
    let mut heads = [0_usize; 6];
    let (mut raw_s, mut ev_s) = (Vec::new(), Vec::new());
    for q in &questions {
        let t = Instant::now();
        let signals = keep.signals(&q.text).map_err(|e| format!("{}: {e}", q.id))?;
        raw_s.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let ev = keep.evidence(&q.text).map_err(|e| format!("{}: {e}", q.id))?;
        ev_s.push(t.elapsed().as_secs_f64());
        let flags: Vec<f32> = signals.iter().map(|s| s.bm25f).collect();
        let fused = nmapi::drone_path::fused_shares(&signals);
        let e = ev.pieces();
        let fam = |f: nmpod::evidence::Family| -> Vec<f32> {
            (0..ev.len())
                .map(|i| {
                    (0..ev.aspects.len())
                        .map(|a| {
                            nmpod::evidence::FAMILY.iter().enumerate().filter(|(_, g)| **g == f).map(|(s, _)| ev.surprise(a, s, i)).fold(0.0_f32, f32::max)
                        })
                        .sum()
                })
                .collect()
        };
        let (wo, na, me) = (fam(nmpod::evidence::Family::Wording), fam(nmpod::evidence::Family::Naming), fam(nmpod::evidence::Family::Meaning));
        let mut line = format!("{:<5}", q.id);
        for (k, v) in [&flags, &fused, &e, &wo, &na, &me].into_iter().enumerate() {
            let o = order(v);
            let place = o.iter().position(|c| covers(*c, q));
            if let Some(p) = place {
                for (j, cut) in [1_usize, 12, 50].iter().enumerate() {
                    if p < *cut
                        && let Some(x) = tally.get_mut(k).and_then(|t| t.get_mut(j))
                    {
                        *x += 1;
                    }
                }
            }
            let h = o.iter().take(12).filter(|c| head(**c)).count();
            if let Some(x) = heads.get_mut(k) {
                *x += h;
            }
            let _ = write!(line, "  {}{}", place.map_or_else(|| "-".to_owned(), |p| (p + 1).to_string()), if h > 0 { format!("/h{h}") } else { String::new() });
        }
        if arg(own, "--why").is_some_and(|w| *w == q.id) {
            let o = order(&e);
            let show = |c: usize| -> String {
                let ch = sub.chunks.get(c);
                let f = ch.and_then(|x| sub.files.get(x.file)).map_or("", |f| f.path.as_str());
                let cells: Vec<String> = (0..ev.aspects.len())
                    .map(|a| {
                        let row: Vec<String> = (0..nmpod::evidence::SIGNALS.len())
                            .map(|s| format!("{:.1}/{:.1}", ev.value.get(a).and_then(|x| x.get(s)).and_then(|x| x.get(c)).copied().unwrap_or(0.0), ev.band.get(a).and_then(|x| x.get(s)).copied().unwrap_or(0.0)))
                            .collect();
                        format!("[{}]", row.join(" "))
                    })
                    .collect();
                format!(
                    "    {f}:{}-{} {} E {:.2} raw bm25f {:.2} {}",
                    ch.map_or(0, |x| x.start),
                    ch.map_or(0, |x| x.end),
                    ch.map_or("", |x| x.kind),
                    e.get(c).copied().unwrap_or(0.0),
                    signals.get(c).map_or(0.0, |s| s.bm25f),
                    cells.join(" ")
                )
            };
            let _ = writeln!(out, "  why {}: nulls {:?}", q.id, ev.nulls);
            for &c in o.iter().take(8) {
                let _ = writeln!(out, "{}", show(c));
            }
            for c in (0..sub.chunks.len()).filter(|c| covers(*c, q)) {
                let _ = writeln!(out, "  answer at {}:{}", o.iter().position(|x| *x == c).map_or(0, |p| p + 1), show(c));
            }
        }
        let accepted: Vec<String> = (0..ev.aspects.len())
            .map(|a| {
                let n = (0..ev.len()).filter(|i| ev.accepted(a, *i)).count();
                format!("{}={n}", ev.aspects.get(a).map_or("", String::as_str))
            })
            .collect();
        let _ = write!(line, "  aspects {}", accepted.join("; "));
        if q.answers.is_empty() {
            let _ = write!(line, "  (decoy)");
        }
        let _ = writeln!(out, "{line}");
    }
    let n = questions.iter().filter(|q| !q.answers.is_empty()).count();
    let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    let warm = |v: &[f64]| mean(v.get(1..).unwrap_or_default());
    let _ = writeln!(out, "answer piece first / in 12 / in 50 (of {n}); file heads in the first 12 (all questions):");
    for (k, name) in ["bm25f flag", "fused shares", "calibrated", "wording", "naming", "meaning"].iter().enumerate() {
        let t = tally.get(k).copied().unwrap_or_default();
        let _ = writeln!(out, "  {name:<13} {} / {} / {}   heads {}", t[0], t[1], t[2], heads.get(k).copied().unwrap_or(0));
    }
    let _ = writeln!(out, "seconds per question, warm: raw signals {:.3}, calibrated evidence {:.3}", warm(&raw_s), warm(&ev_s));
    Ok(out)
}
