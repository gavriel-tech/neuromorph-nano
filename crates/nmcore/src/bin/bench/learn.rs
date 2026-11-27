//! `bench --mode muscle`: G3 and muscle memory (`serve --muscle on`) judged on held-out folds. No
//! continuous learning layer runs here (they are separate, 2026-11-18). One organism serves the
//! question set as `serve` would (the options after `--`):
//! 1. G3 off: every question once; the placements hash is the board's hash for that path.
//! 2. Per fold: G3 on with an empty store; the other folds' questions are served, each judged by the
//!    organism's own measure (`--verdicts own`, the default: each call its own reference, a replay
//!    judged by the served places it keeps and its time) or, with `--verdicts key`, by the answer
//!    key standing for Claude Code's verdicts (the mean reciprocal place of the answer ranges); one
//!    idle cycle replays what G3 judges, mines, selects and promotes; then the fold's own questions
//!    are served with what was learned.
//!
//! The report pools the held-out calls, G3 on against off on the same questions: right first, top 3,
//! in the pack, every answer range, reply tokens, what each fold promoted, and for the calls G3
//! decided, their time with G3's choice against the rule's (each run again fresh). Folds:
//! `tests/dev/folds.json` for the questions it lists, else one of five by the SHA-256 of the id.
use super::{Question, arg, cited, hits, load};
use nmjson::Value;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// What one reply measured.
#[derive(Clone, Default)]
struct Mark {
    first: bool,
    top3: bool,
    pack: bool,
    every: bool,
    rr: f64,
    tokens: usize,
    place: String,
}

fn mark(q: &Question, text: &str) -> Mark {
    let c = cited(text);
    let covers = |a: &super::Place, x: &(String, usize, usize)| a.file == x.0 && (a.whole_file || (x.1 <= a.last && x.2 >= a.first));
    #[expect(clippy::cast_precision_loss, reason = "places and counts")]
    let rr = if q.answers.is_empty() {
        0.0
    } else {
        q.answers.iter().map(|a| c.iter().position(|x| covers(a, x)).map_or(0.0, |p| 1.0 / (p + 1) as f64)).sum::<f64>() / q.answers.len() as f64
    };
    Mark {
        first: c.first().is_some_and(|x| hits(q, x)),
        top3: c.iter().take(3).any(|x| hits(q, x)),
        pack: c.iter().any(|x| hits(q, x)),
        every: !q.answers.is_empty() && q.answers.iter().all(|a| c.iter().any(|x| covers(a, x))),
        rr,
        tokens: text.chars().count() / 4,
        place: c.iter().map(|x| format!("{}:{}-{}", x.0, x.1, x.2)).collect::<Vec<_>>().join(","),
    }
}

/// The fold of each question: `folds.json` when it lists the id, else the SHA-256 of the id.
fn folds(home: &Path, questions: &[Question]) -> Vec<usize> {
    let listed: BTreeMap<String, usize> = match std::fs::read(home.join("tests/dev/folds.json")).ok().and_then(|b| nmjson::parse(&b).ok()) {
        Some(v) => match v.get("folds") {
            Ok(Value::Object(m)) => m.iter().filter_map(|(id, f)| Some((id.clone(), usize::try_from(f.integer().ok()?).ok()?))).collect(),
            _ => BTreeMap::new(),
        },
        None => BTreeMap::new(),
    };
    questions
        .iter()
        .map(|q| listed.get(&q.id).copied().unwrap_or_else(|| nmhash::sha256_bytes(q.id.as_bytes()).bytes().map(usize::from).sum::<usize>() % 5))
        .collect()
}

fn reply(api: &nmapi::Api, input: &Value) -> String {
    match api.dispatch("nm_context_pack", input) {
        Ok(Value::String(s)) => s,
        Ok(v) => v.encode().unwrap_or_default(),
        Err(e) => format!("error {}: {}", e.code, e.message),
    }
}

/// Wait until the meaning lane, when bound, holds a vector for every piece (as the board does).
fn lane_full(org: &nmapi::forage::Organism) {
    let sub = org.sub();
    let Some(lane) = sub.dense.read().ok().and_then(|d| d.clone()) else { return };
    let (mut last, mut since) = (0_usize, Instant::now());
    loop {
        let (have, total) = lane.coverage(&sub);
        if have >= total || since.elapsed().as_secs() >= 600 {
            eprintln!("meaning lane: {have} of {total} pieces with a vector");
            return;
        }
        if have > last {
            (last, since) = (have, Instant::now());
        }
        std::thread::park_timeout(std::time::Duration::from_millis(500));
    }
}

/// `bench --mode muscle [--root <repo>] [--questions <file>] [--store <dir>] [--form <form>] [--verdicts own|key] [-- <serve options>]`.
#[expect(clippy::too_many_lines, reason = "one run, written top to bottom")]
pub fn run(home: &Path, own: &[String], extra: &[String]) -> Result<String, String> {
    let root = match arg(own, "--root") {
        Some(r) if Path::new(r).is_dir() => nmsubs::repo::canon(Path::new(r)),
        Some(r) => return Err(format!("--root {r} is not a folder")),
        None => home.to_path_buf(),
    };
    let qfile = arg(own, "--questions").map_or_else(|| home.join("tests/dev/questions.json"), PathBuf::from);
    let mut questions = load(&root, &qfile)?;
    if let Some(limit) = arg(own, "--n").and_then(|v| v.parse::<usize>().ok()) {
        questions.truncate(limit);
    }
    let label = nmtype::project::label(home, &root);
    let store_root = arg(own, "--store").map_or_else(|| home.join("scratch").join("lane-b2").join("g3-bench").join(&label), PathBuf::from);
    let form = arg(own, "--form").filter(|f| *f != "none").cloned();
    let key = match arg(own, "--verdicts").map(String::as_str) {
        None | Some("own") => false,
        Some("key") => true,
        Some(o) => return Err(format!("--verdicts takes own or key, got {o}")),
    };
    let input = |q: &Question| {
        let mut m = BTreeMap::from([("question".to_owned(), Value::String(q.text.clone()))]);
        if let Some(f) = &form {
            m.insert("form".to_owned(), Value::String(f.clone()));
        }
        Value::Object(m)
    };
    let memory = home.join("scratch").join("bench-memory").join(&label);
    let org = nmapi::forage::Organism::open_with(home, &root, &["neuromorph-nano".to_owned()], false, &memory).map_err(|e| format!("organism: {e}"))?;
    super::super::configure(&org, extra)?;
    lane_full(&org);
    nmapi::tools::install(org).map_err(|e| format!("{}: {}", e.code, e.message))?;
    let api = nmapi::tools::api();
    let t0 = Instant::now();

    // 1. Learning off.
    let _ = nmapi::g3::disable();
    let off: Vec<Mark> = questions.iter().map(|q| mark(q, &reply(&api, &input(q)))).collect();
    let places: Vec<String> = questions.iter().zip(&off).map(|(q, m)| format!("plain {} {}", q.id, m.place)).collect();
    let placements = nmhash::sha256_bytes(places.join("\n").as_bytes());
    eprintln!("muscle: off arm {:.1} s", t0.elapsed().as_secs_f64());

    // 2. Folds.
    let fold = folds(home, &questions);
    let k = fold.iter().copied().max().map_or(0, |m| m + 1);
    let mut on: Vec<Option<(Mark, String)>> = vec![None; questions.len()];
    let mut learned = String::new();
    // Calls G3 decided: (ms with G3's choice, ms with the rule's choice right after).
    let mut timed: Vec<(f64, f64)> = Vec::new();
    for f in 0..k {
        let dir = store_root.join(format!("fold{f}"));
        let _ = std::fs::remove_dir_all(&dir);
        nmapi::g3::enable(home, Some(&dir)).map_err(|e| e.to_string())?;
        let mut by_call: BTreeMap<String, usize> = BTreeMap::new();
        let mut served: BTreeMap<String, nmapi::g3::Served> = BTreeMap::new();
        for (i, q) in questions.iter().enumerate() {
            if fold.get(i) == Some(&f) || q.answers.is_empty() {
                continue;
            }
            let text = reply(&api, &input(q));
            let m = mark(q, &text);
            if let Some(id) = nmapi::g3::last_call() {
                if key {
                    nmapi::g3::verdict(&id, m.rr).map_err(|e| e.to_string())?;
                }
                served.insert(id.clone(), nmapi::g3::Served { cited: nmapi::g3::cited(&text), ms: 0 });
                by_call.insert(id, i);
            }
        }
        // Replays are scored by the organism's own measure, or by the answer key (standing for the
        // ledger's verified places), with time as a cost below one place (`nmapi::g3::outcome`).
        let mut score = |id: &str, _: &[(nmmusc::priors::Site, String)], text: &str| {
            if key {
                let q = questions.get(*by_call.get(id)?)?;
                Some((mark(q, text).rr, q.answers.len()))
            } else {
                nmapi::g3::own_score(served.get(id)?, None, text)
            }
        };
        let cy = nmapi::g3::cycle(&mut score).map_err(|e| e.to_string())?;
        let _ = writeln!(
            learned,
            "fold {f}: trained on {} calls; replays {}; promoted {:?}; provisional {:?}; refused {} by gain, {} by negative selection",
            by_call.len(),
            cy.replays,
            cy.promoted.iter().map(|p| p.1.clone()).collect::<Vec<_>>(),
            cy.provisional,
            cy.refused_gain.len(),
            cy.refused_self.len()
        );
        for (i, q) in questions.iter().enumerate() {
            if fold.get(i) != Some(&f) {
                continue;
            }
            let m = mark(q, &reply(&api, &input(q)));
            let cfg = nmapi::tools::take_diagnostics().and_then(|d| d.get("config").ok().cloned());
            let field = |k: &str| cfg.as_ref().and_then(|c| c.get(k).ok()).and_then(|v| v.string().ok()).unwrap_or_default().to_owned();
            let by = format!("{} by {}", field("allocation"), field("by"));
            // A call G3 decided: G3's choice and the rule's, each run again the same way (fresh,
            // no answer cache), in alternating order, for the time the decision bought.
            if field("allocation") != field("rule") && !field("rule").is_empty() {
                let run = |m: &str| {
                    let t = Instant::now();
                    let _ = nmapi::g3::forced(m, || reply(&api, &input(q)));
                    t.elapsed().as_secs_f64() * 1000.0
                };
                let (g, r) = if timed.len().is_multiple_of(2) {
                    let g = run(&field("allocation"));
                    (g, run(&field("rule")))
                } else {
                    let r = run(&field("rule"));
                    (run(&field("allocation")), r)
                };
                timed.push((g, r));
            }
            if let Some(slot) = on.get_mut(i) {
                *slot = Some((m, by));
            }
        }
        let _ = nmapi::g3::disable();
        eprintln!("muscle: fold {f} done {:.1} s", t0.elapsed().as_secs_f64());
    }

    // Report: held-out calls with answers, on against off.
    let mut out = String::new();
    let _ = writeln!(out, "muscle: {} questions, {k} folds, verdicts {}, off placements {placements}", questions.len(), if key { "key" } else { "own" });
    out.push_str(&learned);
    let (mut a, mut b) = ([0_usize; 4], [0_usize; 4]);
    let (mut rr_a, mut rr_b, mut tok_a, mut tok_b, mut n, mut moved) = (0.0, 0.0, 0_usize, 0_usize, 0_usize, Vec::new());
    for ((q, m0), m1) in questions.iter().zip(&off).zip(&on) {
        let Some((m1, by)) = m1 else { continue };
        if q.answers.is_empty() {
            continue;
        }
        n += 1;
        for (i, (x, y)) in [(m0.first, m1.first), (m0.top3, m1.top3), (m0.pack, m1.pack), (m0.every, m1.every)].into_iter().enumerate() {
            if let (Some(p), Some(r)) = (a.get_mut(i), b.get_mut(i)) {
                *p += usize::from(x);
                *r += usize::from(y);
            }
        }
        (rr_a, rr_b, tok_a, tok_b) = (rr_a + m0.rr, rr_b + m1.rr, tok_a + m0.tokens, tok_b + m1.tokens);
        if m0.place != m1.place {
            moved.push(format!("{} ({by}): rr {:.3} to {:.3}", q.id, m0.rr, m1.rr));
        }
    }
    let _ = writeln!(
        out,
        "held-out calls with answers {n}: off first {} top3 {} pack {} every {} of {n}, rr {rr_a:.3}, tokens {tok_a}\n                        on  first {} top3 {} pack {} every {} of {n}, rr {rr_b:.3}, tokens {tok_b}",
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]
    );
    let _ = writeln!(out, "moved {} of {n}: {}", moved.len(), moved.join("; "));
    if !timed.is_empty() {
        let med = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(0.0)
        };
        let (g, r): (Vec<f64>, Vec<f64>) = timed.iter().copied().unzip();
        let faster = timed.iter().filter(|(g, r)| g < r).count();
        let _ = writeln!(
            out,
            "decided by G3 {} calls, each run again fresh with both choices: median {:.0} ms with G3's choice against {:.0} ms with the rule's; faster on {faster} of {}; total {:.1} against {:.1} s",
            timed.len(),
            med(g.clone()),
            med(r.clone()),
            timed.len(),
            g.iter().sum::<f64>() / 1000.0,
            r.iter().sum::<f64>() / 1000.0
        );
    }
    #[expect(clippy::cast_precision_loss, reason = "token counts")]
    let tokens_ok = (tok_b as f64) <= 1.1 * tok_a as f64;
    let holds = (0..4).all(|i| b.get(i) >= a.get(i)) && tokens_ok;
    let _ = writeln!(out, "pass (no count lower, tokens not up more than 10 percent): {holds}; {:.1} s", t0.elapsed().as_secs_f64());
    Ok(out)
}
