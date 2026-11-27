//! `bench --mode board`: the scoreboard, one table for the paths asked over one question set.
//! Processor paths (`plain`, `stack`) run in this process first; card paths (`fast`, `full`) run
//! only when asked, in a child process of this program, since one process serves one
//! configuration. One header line names the program hash, the serve options and a hash of every
//! placement (no times in it), so two runs of the same build on the same tree give the same hash.
use super::{Question, arg, cited, hits, load};
use nmjson::Value;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

const PATHS: [&str; 5] = ["plain", "flow", "stack", "fast", "full"];
const CARD: [&str; 2] = ["fast", "full"];

/// What one served reply measured.
struct Mark {
    id: String,
    first: bool,
    top3: bool,
    pack: bool,
    every: bool,
    cited: bool,
    tokens: usize,
    secs: f64,
    withheld: bool,
    low: bool,
    /// Every cited place in reply order (the placement).
    place: String,
    /// Mean over the answer ranges of the reciprocal place of the first cited span holding it.
    rr: f64,
}

/// Where the answer sits in the Globalized Context Stack for one question.
struct StackMark {
    id: String,
    /// Place in what G2 sends up (0-based).
    up: Option<usize>,
    every_up: bool,
    /// The tentacle holding it best, its place in that `g0_tx` order, and whether it was reported.
    window: Option<(usize, usize, bool)>,
    g1: Option<usize>,
    secs: f64,
    /// The request's aspects the up set covers, and the request's aspects (its drone).
    aspects: (usize, usize),
    /// File notes and heads (a file's first piece, or notes and imports) in the up set, for a
    /// question whose answers are all in code.
    heads: Option<usize>,
    /// Tentacles whose report names an aspect with no information, and tentacles reporting.
    absent: (usize, usize),
    /// What the team took in: an answer range in some window; every answer range in some window;
    /// answer ranges in the windows and in all.
    in_win: bool,
    every_win: bool,
    ranges: (usize, usize),
    /// Distinct pieces and tokens over every round, and the rounds of windows.
    pieces: usize,
    tokens: usize,
    window_rounds: usize,
    /// Specific relations between two pieces in the windows: joining two different windows, and all.
    cross: (usize, usize),
    /// The request's aspects some piece in the windows holds, and all of them.
    aspects_win: (usize, usize),
    /// Place of the answer in G1's hand-over.
    hand: Option<usize>,
    /// Invariants: pieces in two windows, held pieces not forwarded or queued once, tentacles
    /// whose report changed with the others absent, refusals at the isolation gate.
    invariants: [usize; 4],
    /// Per tentacle, answer ranges only its windows hold (what the team would miss without it).
    only: Vec<usize>,
    /// Labels the tentacles had, one per window.
    labels: Vec<String>,
    /// Every answer range in G1's hand-over, and the hand-over's length.
    every_hand: bool,
    hand_len: usize,
    /// G1's race at the end: committed, ties, aspects uncovered (0 without the race).
    race: (usize, usize, usize),
    /// With `--why`: each answer range missing from the up set, where it was and how it relates
    /// to the up set.
    lost: Vec<String>,
    /// G2: its gate (empty without its comprehension), the covering set's size, every answer
    /// range in the covering set, orders sent, pieces its orders had read, orders refused for a
    /// missing copy to G1, edges refused by class, and its milliseconds.
    g2gate: String,
    g2set: usize,
    every_set: bool,
    g2orders: usize,
    g2read: usize,
    motor_refused: usize,
    class_refused: usize,
    g2ms: f64,
    /// The judge at the inclusion edge: pieces judged, milliseconds, pieces moved, tokens read.
    judge: (usize, f64, usize, usize),
    up_len: usize,
    /// Tokens sent up; G2's F over its aspects and relations, and why its selection stopped.
    up_tokens: usize,
    g2value: (f32, usize),
    g2stop: String,
    /// Time terms in seconds: the drones (the request's drone and the drones' evidence), forming
    /// the windows, G1 with the windows' reads.
    times: [f64; 3],
    /// The answer-first pack written from what went up, marked as a served reply is.
    pack: Mark,
    pack_s: f64,
    /// With `--why-r`: the selection's diagnosis, one line each (G2's chance per part for the
    /// answer pieces and for the candidates that beat them).
    diag: Vec<String>,
    /// With `--why-r`: every (candidate, part) pair with a chance, as (the drones' part chance,
    /// the settle chance, the candidate is an answer piece); per part, the place of the first
    /// answer piece among the candidates by each chance.
    pairs: Vec<(f32, f32, bool)>,
    ranks: Vec<(Option<usize>, Option<usize>)>,
}

impl StackMark {
    fn place(&self) -> String {
        let p = |x: Option<usize>| x.map_or_else(|| "-".to_owned(), |v| (v + 1).to_string());
        let w = self.window.map_or_else(|| "-".to_owned(), |(t, r, rep)| format!("t{}#{}{}", t + 1, r + 1, if rep { "*" } else { "" }));
        format!("up {} every {} window {w} g1 {}", p(self.up), u8::from(self.every_up), p(self.g1))
    }
}

/// `path:first-last` (or `path:line`) at the start of a line: an outline line of a pack.
fn outline(reply: &str) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    for l in reply.lines() {
        if l.starts_with("@@ ") || l.starts_with("~~ ") || l.starts_with(' ') {
            continue;
        }
        let Some((path, at)) = l.split_whitespace().next().and_then(|h| h.rsplit_once(':')) else { continue };
        if path.is_empty() || !(path.contains('.') || path.contains('/')) {
            continue;
        }
        let (a, b) = at.split_once('-').unwrap_or((at, at));
        if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
            out.push((path.replace('\\', "/"), a, b));
        }
    }
    out
}

fn mark(q: &Question, text: &str, secs: f64) -> Mark {
    let c = cited(text);
    let covers = |a: &super::Place, x: &(String, usize, usize)| a.file == x.0 && (a.whole_file || (x.1 <= a.last && x.2 >= a.first));
    let lines = outline(text);
    Mark {
        id: q.id.clone(),
        first: c.first().is_some_and(|x| hits(q, x)),
        top3: c.iter().take(3).any(|x| hits(q, x)),
        pack: c.iter().any(|x| hits(q, x)),
        every: !q.answers.is_empty() && q.answers.iter().all(|a| c.iter().any(|x| covers(a, x))),
        // A place cited: an answer range present whole in a span, or named by an outline line.
        cited: q.answers.iter().any(|a| {
            c.iter().any(|x| a.file == x.0 && (a.whole_file || (x.1 <= a.first && x.2 >= a.last))) || lines.iter().any(|x| covers(a, x))
        }),
        tokens: text.chars().count() / 4,
        secs,
        withheld: text.lines().next().is_some_and(|l| l.contains(" withheld")),
        low: text.lines().any(|l| l.starts_with("confidence: low")),
        place: c.iter().map(|x| format!("{}:{}-{}", x.0, x.1, x.2)).collect::<Vec<_>>().join(","),
        rr: range_rr(q, &c),
    }
}

/// Mean over the answer ranges of 1 / place of the first cited span holding a line of it (0 when
/// none does).
fn range_rr(q: &Question, c: &[(String, usize, usize)]) -> f64 {
    if q.answers.is_empty() {
        return 0.0;
    }
    let covers = |a: &super::Place, x: &(String, usize, usize)| a.file == x.0 && (a.whole_file || (x.1 <= a.last && x.2 >= a.first));
    #[expect(clippy::cast_precision_loss, reason = "places and counts")]
    let s: f64 = q.answers.iter().map(|a| c.iter().position(|x| covers(a, x)).map_or(0.0, |p| 1.0 / (p + 1) as f64)).sum::<f64>() / q.answers.len() as f64;
    s
}

/// Serve every question on each of `modes` in this process: one organism, configured with
/// `serve_args`, models warm before the first question when the whole path is on.
/// The reply form the served paths are asked for (`--form`, default full; `adaptive` is what a
/// caller naming no form gets).
static FORM: std::sync::RwLock<String> = std::sync::RwLock::new(String::new());

fn serve_marks(home: &Path, root: &Path, questions: &[Question], modes: &[&str], serve_args: &[String]) -> Result<Vec<(String, Vec<Mark>)>, String> {
    let memory = home.join("scratch").join("bench-memory").join(nmtype::project::label(home, root));
    let org = nmapi::forage::Organism::open_with(home, root, &["neuromorph-nano".to_owned()], false, &memory).map_err(|e| format!("organism: {e}"))?;
    super::super::configure(&org, serve_args)?;
    lane_full(&org);
    // Stage 5 WP-C4: a configured note store's notes join the meaning lane before the first question.
    let seeded = nmapi::writer_idle::seed_aux_now(&org.sub());
    if seeded > 0 {
        eprintln!("writer: {seeded} held notes and likely names added to the meaning lane before the questions");
    }
    nmapi::tools::install(org).map_err(|e| format!("{}: {}", e.code, e.message))?;
    if let Some(w) = nmapi::tools::installed().and_then(|o| o.whole.read().ok().and_then(|w| w.clone())) {
        w.warm().map_err(|e| format!("warm: {e}"))?;
    }
    // The bench measures the warm served state: it waits (at most 120 s) until the background
    // loads of the close-call judge and the tentacles' reader have settled (on the card, or no
    // card), so the first questions are served as the rest are. Served calls never wait.
    let t = std::time::Instant::now();
    while t.elapsed().as_secs() < 120
        && (nmapi::drone_path::warm_close_judge() == nmapi::drone_path::JudgeState::Loading
            || nmapi::drone_path::warm_reader() == nmapi::drone_path::JudgeState::Loading)
    {
        std::thread::park_timeout(std::time::Duration::from_millis(200));
    }
    eprintln!(
        "bench: warm state after {:.1} s: judge {:?}, reader {:?}",
        t.elapsed().as_secs_f64(),
        nmapi::drone_path::warm_close_judge(),
        nmapi::drone_path::warm_reader()
    );
    let api = nmapi::tools::api();
    let mut out = Vec::new();
    for mode in modes {
        let mut marks = Vec::new();
        for q in questions {
            let form = FORM.read().map_or_else(|_| "full".to_owned(), |f| f.clone());
            let mut input = vec![("question".to_owned(), Value::String(q.text.clone()))];
            if !form.is_empty() {
                input.push(("form".to_owned(), Value::String(form)));
            }
            if *mode != "plain" {
                input.push(("mode".to_owned(), Value::String((*mode).to_owned())));
            }
            let input = Value::Object(input.into_iter().collect());
            let t = Instant::now();
            let text = match api.dispatch("nm_context_pack", &input) {
                Ok(Value::String(s)) => s,
                Ok(v) => v.encode().unwrap_or_default(),
                Err(e) => format!("error {}: {}", e.code, e.message),
            };
            let secs = t.elapsed().as_secs_f64();
            // A measurement aid (`NANO_BENCH_TEXT=<dir>`): each reply as served, one file a question.
            if let Some(dir) = std::env::var_os("NANO_BENCH_TEXT") {
                let _ = std::fs::create_dir_all(&dir);
                let _ = std::fs::write(Path::new(&dir).join(format!("{mode}-{}.txt", q.id)), &text);
            }
            let _ = nmapi::tools::take_diagnostics();

            let m = mark(q, &text, secs);
            eprintln!("{mode} {} {:.2} s first {} every {}", q.id, secs, u8::from(m.first), u8::from(m.every));
            marks.push(m);
        }
        out.push(((*mode).to_owned(), marks));
    }
    // Shutdown: background card passes stop before the process ends (the intake of relevance keys
    // and drone readings runs on its own thread; left mid-job it hung the bench at exit).
    let keys = nmapi::tools::installed().and_then(|o| o.whole.read().ok().and_then(|w| w.clone())).and_then(|w| w.relevance_keys());
    nmapi::key_scan::stop_passes(keys.as_deref(), std::time::Duration::from_secs(30));
    Ok(out)
}


/// With the meaning lane bound, wait until every piece has a vector before the first question,
/// so no question is measured on a lane still being built (stored vectors load at once; a
/// material with none is embedded in the background). Progress on standard error each minute; the
/// wait ends early when no vector is added for ten minutes (a stated watchdog for a stalled fill,
/// reported, not a quota on the material).
fn lane_full(org: &nmapi::forage::Organism) {
    let sub = org.sub();
    let Some(lane) = sub.dense.read().ok().and_then(|d| d.clone()) else { return };
    // A lane held at its loaded coverage (`NANO_DENSE_FILL=off`) is measured as it is.
    if !nmsubs::dense::FILL.load(std::sync::atomic::Ordering::Relaxed) {
        let (have, total) = lane.coverage(&sub);
        eprintln!("meaning lane: {have} of {total} pieces with a vector, held there (NANO_DENSE_FILL=off)");
        return;
    }
    // `NANO_BENCH_WAIT=off`: the questions start at once while the lane builds, as served calls on
    // a first start do (the check that no call waits for the build).
    if std::env::var("NANO_BENCH_WAIT").as_deref() == Ok("off") {
        let (have, total) = lane.coverage(&sub);
        eprintln!("meaning lane: {have} of {total} pieces with a vector, not waited for (NANO_BENCH_WAIT=off)");
        return;
    }
    let t = Instant::now();
    let (mut last, mut since, mut told) = (0_usize, Instant::now(), 0_u64);
    loop {
        let (have, total) = lane.coverage(&sub);
        if have >= total {
            eprintln!("meaning lane: {have} of {total} pieces with a vector, {:.1} s waited", t.elapsed().as_secs_f64());
            return;
        }
        if have > last {
            (last, since) = (have, Instant::now());
        }
        if since.elapsed().as_secs() >= 600 {
            eprintln!("meaning lane: {have} of {total} pieces with a vector, no vector added for 600 s; measured as it is");
            return;
        }
        if t.elapsed().as_secs() / 60 > told {
            told = t.elapsed().as_secs() / 60;
            eprintln!("meaning lane: {have} of {total} pieces with a vector after {told} min");
        }
        std::thread::park_timeout(std::time::Duration::from_millis(500));
    }
}

/// The Globalized Context Stack alone over every question (processor, no models).
#[expect(clippy::too_many_lines, reason = "one mark per question, every measure of the stack")]
#[expect(clippy::too_many_arguments, reason = "the material, the questions, the arm and the diagnosis switches")]
fn stack_marks(
    home: &Path,
    root: &Path,
    questions: &[Question],
    arm: nmapi::drone_path::Windows,
    matched: bool,
    why: bool,
    why_r: bool,
    dump: Option<&Path>,
    force: Option<nmapi::drone_path::ForceAt>,
    extra: &[String],
) -> Result<(Vec<StackMark>, Vec<usize>), String> {
    let memory = home.join("scratch").join("bench-memory").join(nmtype::project::label(home, root));
    let org = nmapi::forage::Organism::open_with(home, root, &["neuromorph-nano".to_owned()], false, &memory).map_err(|e| format!("organism: {e}"))?;
    // `-- --dense`: the meaning lane bound as on the served path (its vectors from
    // `NANO_DENSE_DIR`), so the stack runs beside it; nothing else of the serve options applies.
    if extra.iter().any(|a| a == "--dense") {
        let dense: Vec<String> = extra.iter().skip_while(|a| *a != "--dense").take(2).filter(|a| *a == "--dense" || *a == "on" || *a == "off").cloned().collect();
        super::super::configure(&org, &dense)?;
        lane_full(&org);
    // Stage 5 WP-C4: a configured note store's notes join the meaning lane before the first question.
    let seeded = nmapi::writer_idle::seed_aux_now(&org.sub());
    if seeded > 0 {
        eprintln!("writer: {seeded} held notes and likely names added to the meaning lane before the questions");
    }
    }
    let sub = org.sub();
    let mut keep = nmapi::drone_windows::Keep::open(&nmtype::models_dir(home).join("qwen3-0.6b/files")).map_err(|e| format!("intake: {e}"))?;
    let t_in = std::time::Instant::now();
    let taken = keep.sync(&sub).map_err(|e| format!("intake: {e}"))?;
    eprintln!("intake: {taken} pieces taken in by the drones in {:.2} s", t_in.elapsed().as_secs_f64());
    let graph = sub.refs();
    let mut per_t = vec![0_usize; nmsubs::lens::NAMES.len()];
    let mut out = Vec::new();
    let mut dump_file = match dump {
        Some(dir) => {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            Some(std::fs::File::create(dir.join("candidates.jsonl")).map_err(|e| format!("{}: {e}", dir.display()))?)
        }
        None => None,
    };
    for q in questions {
        use nmapi::drone_path::Windows;
        // A comparison arm matched to the team takes in the tokens the team took on this question.
        let total = if matched && matches!(arm, Windows::FirstFit | Windows::Single) {
            let t = nmapi::drone_path::stack_run_arm(&sub, &keep, &q.text, None, Windows::Team, None).map_err(|e| format!("{}: {e}", q.id))?;
            Some(t.windows.iter().flatten().map(|c| keep.tokens(*c).max(1)).sum::<usize>())
        } else {
            None
        };
        let covers = |c: usize, a: &super::Place| {
            sub.chunks.get(c).is_some_and(|ch| {
                sub.files.get(ch.file).is_some_and(|f| f.path.replace('\\', "/") == a.file) && (a.whole_file || (ch.start <= a.last && ch.end >= a.first))
            })
        };
        let ans: Vec<usize> = (0..sub.chunks.len()).filter(|c| q.answers.iter().any(|a| covers(*c, a))).collect();
        nmapi::drone_path::set_force(force.map(|f| (f, ans.clone())));
        let run = nmapi::drone_path::stack_run_arm(&sub, &keep, &q.text, None, arm, total).map_err(|e| format!("{}: {e}", q.id));
        nmapi::drone_path::set_force(None);
        let run = run?;
        let best = |list: &[usize]| list.iter().position(|c| ans.contains(c));
        let window = run
            .orders
            .iter()
            .enumerate()
            .filter_map(|(t, o)| best(o).map(|r| (t, r)))
            .min_by_key(|(t, r)| (*r, *t))
            .map(|(t, r)| (t, r, r < run.depths.get(t).copied().unwrap_or(0)));
        if let Some((t, _, true)) = window {
            let tentacle = run.owners.get(t).copied().unwrap_or(t);
            if per_t.len() <= tentacle {
                per_t.resize(tentacle + 1, 0);
            }
            if let Some(x) = per_t.get_mut(tentacle) {
                *x += 1;
            }
        }
        // What the team took in.
        let mut inw: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for (w, win) in run.windows.iter().enumerate() {
            for c in win {
                inw.entry(*c).or_insert(w);
            }
        }
        let range_in = |a: &super::Place| inw.keys().any(|c| covers(*c, a));
        let ranges = (q.answers.iter().filter(|a| range_in(a)).count(), q.answers.len());
        let tentacles = run.owners.iter().copied().max().map_or(0, |m| m + 1);
        let only: Vec<usize> = (0..tentacles)
            .map(|t| {
                q.answers
                    .iter()
                    .filter(|a| {
                        let holders: std::collections::BTreeSet<usize> =
                            inw.iter().filter(|(c, _)| covers(**c, a)).filter_map(|(_, w)| run.owners.get(*w).copied()).collect();
                        holders.len() == 1 && holders.contains(&t)
                    })
                    .count()
            })
            .collect();
        let (mut cross, mut inner) = (0_usize, 0_usize);
        for (c, w) in &inw {
            for e in graph.specific.edges_of(*c) {
                let j = e.to as usize;
                if j <= *c {
                    continue;
                }
                if let Some(wj) = inw.get(&j) {
                    if wj == w {
                        inner += 1;
                    } else {
                        cross += 1;
                    }
                }
            }
        }
        let mut g1_order = run.handover.clone();
        g1_order.extend(run.up.iter().chain(&run.queue).filter(|c| !run.handover.contains(c)));
        let lost: Vec<String> = if why {
            q.answers
                .iter()
                .filter(|a| !run.up.iter().any(|c| covers(*c, a)))
                .map(|a| {
                    let held: Vec<usize> = (0..sub.chunks.len()).filter(|c| covers(*c, a)).collect();
                    let at = |list: &[usize]| held.iter().filter_map(|c| list.iter().position(|x| x == c)).min().map_or("-".to_owned(), |p| (p + 1).to_string());
                    let win = held.iter().any(|c| inw.contains_key(c));
                    let rel: Vec<String> = held
                        .iter()
                        .flat_map(|c| graph.neighbours(*c).into_iter().filter_map(|(j, r, _)| run.up.iter().position(|u| *u == j).map(|p| format!("{r:?} u{}", p + 1))))
                        .collect();
                    format!(
                        "{}:{}-{} window {} hand-over {} queue {} rel [{}]",
                        a.file,
                        a.first,
                        a.last,
                        u8::from(win),
                        at(&run.handover),
                        at(&run.queue),
                        rel.join(", ")
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        let diag = if why_r { diagnose(&sub, &keep, q, &run, &ans) } else { Vec::new() };
        if let Some(f) = dump_file.as_mut() {
            use std::io::Write as _;
            writeln!(f, "{}", dump_line(&sub, &keep, &graph, q, &run, &ans)).map_err(|e| e.to_string())?;
        }
        let (pairs, ranks) = if why_r { reliability(&run, &ans) } else { (Vec::new(), Vec::new()) };
        let m = StackMark {
            diag,
            pairs,
            ranks,
            every_hand: !q.answers.is_empty() && q.answers.iter().all(|a| run.handover.iter().any(|c| covers(*c, a))),
            hand_len: run.handover.len(),
            g2gate: run.g2run.gate.clone(),
            g2set: run.g2run.set.len(),
            every_set: !q.answers.is_empty() && q.answers.iter().all(|a| run.g2run.set.iter().any(|c| covers(*c, a))),
            g2orders: run.g2run.motor.sent.len(),
            g2read: run.g2run.read,
            motor_refused: run.g2run.motor.refused,
            class_refused: run.g2run.class_refusals,
            g2ms: run.g2run.ms,
            judge: (run.g2run.judged, run.g2run.judge_ms, run.g2run.judge_moved, run.g2run.judge_tokens),
            up_len: run.up.len(),
            up_tokens: run.up.iter().map(|c| keep.tokens(*c).max(1)).sum(),
            g2value: run.g2run.value,
            g2stop: run.g2run.stop.clone(),
            times: [run.sense_s, run.compose_s, run.g1_s],
            pack: mark(q, &run.pack, run.total_s),
            pack_s: run.pack_s,
            race: run.verdict.as_ref().map_or((0, 0, 0), |v| (v.committed.len(), v.ties.len(), v.uncovered.len())),
            lost,
            id: q.id.clone(),
            up: best(&run.up),
            every_up: !q.answers.is_empty() && q.answers.iter().all(|a| run.up.iter().any(|c| covers(*c, a))),
            window,
            g1: best(&g1_order),
            secs: run.total_s,
            heads: q.answers.iter().all(|a| !nmtype::is_document(&a.file)).then(|| {
                run.up.iter().filter(|c| sub.chunks.get(**c).is_some_and(|ch| ch.start <= 1 || matches!(ch.kind, "notes" | "imports"))).count()
            }),
            absent: (run.absent.iter().filter(|a| !a.is_empty()).count(), run.absent.len()),
            aspects: {
                let req = nmpod::intake::Request::read(&q.text, &[]);
                let up_text: Vec<String> = run
                    .up
                    .iter()
                    .filter_map(|c| sub.chunks.get(*c).map(|ch| format!("{} {} {}", sub.files.get(ch.file).map_or("", |f| f.path.as_str()), ch.heading, ch.text)))
                    .collect();
                (req.covered_by(&up_text.join(" ")).iter().filter(|x| **x).count(), req.aspects.len())
            },
            in_win: ranges.0 > 0,
            every_win: ranges.1 > 0 && ranges.0 == ranges.1,
            ranges,
            pieces: inw.len(),
            tokens: inw.keys().map(|c| keep.tokens(*c).max(1)).sum(),
            window_rounds: run.taken.len(),
            cross: (cross, cross + inner),
            aspects_win: run.aspects_held,
            hand: best(&run.handover),
            invariants: [run.overlap, run.unconserved, run.solo, run.gate],
            only,
            labels: run.labels.clone(),
        };
        eprintln!("stack {} {} judged {} in {:.2} s", q.id, m.place(), m.judge.0, m.judge.1 / 1000.0);
        // What the judge decided, on standard error only: each judged piece (place, the shift it
        // needed to cross G2's edge, the judge's calibrated evidence) and what moved.
        if !run.g2run.judge_pairs.is_empty() {
            let at = |c: usize| {
                let mark = if ans.contains(&c) { "*" } else { "" };
                sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file).map(|f| format!("{mark}{}:{}-{}", f.path.replace('\\', "/"), ch.start, ch.end))).unwrap_or_default()
            };
            let pairs: Vec<String> = run.g2run.judge_pairs.iter().map(|(c, d, y)| format!("{} need {d:.2} judge {y:+.2}", at(*c))).collect();
            let moved = |v: &[usize]| v.iter().map(|c| at(*c)).collect::<Vec<_>>().join(" ");
            eprintln!("judge {} pairs [{}] in [{}] out [{}]", q.id, pairs.join("; "), moved(&run.g2run.judge_in), moved(&run.g2run.judge_out));
            // The duels at the edge: entrant against holder, the shift the entrant needed, the
            // judge's difference (entrant less holder; above 0 the entrant took the place).
            if !run.g2run.judge_duels.is_empty() {
                let duels: Vec<String> = run.g2run.judge_duels.iter().map(|(x, y, c, d)| format!("{} vs {} need {c:.2} diff {d:+.2}", at(*x), at(*y))).collect();
                eprintln!("duel {} [{}]", q.id, duels.join("; "));
            }
        }
        out.push(m);
    }
    Ok((out, per_t))
}

/// The selection's diagnosis for one question: the request's parts and relations; G2's chosen
/// set; each answer piece with its place among G2's candidates and, per part, its chance, its
/// epistemic tag and the share of the part's stems it holds; per part, the candidates with the
/// largest chance (an answer piece marked `*`).
fn diagnose(sub: &nmsubs::Substrate, keep: &nmapi::drone_windows::Keep, q: &Question, run: &nmapi::drone_path::StackRun, ans: &[usize]) -> Vec<String> {
    let Some(ev) = run.evidence.as_ref() else { return Vec::new() };
    let req = nmpod::intake::Request::read(&q.text, &[]);
    let parts: Vec<&nmpod::intake::Part> = req.aspects.iter().flat_map(|a| a.parts.iter()).collect();
    let chance = nmapi::drone_path::Chance::new(ev);
    let place = |c: usize| {
        sub.chunks.get(c).map_or_else(String::new, |ch| format!("{}:{}-{}", sub.files.get(ch.file).map_or("", |f| f.path.as_str()).replace('\\', "/"), ch.start, ch.end))
    };
    let cell = |k: usize, c: usize| -> String {
        let r = chance.first_form(ev, k, c);
        let s = ev.settles(k, c);
        let stems: Vec<String> = parts.get(k).map(|p| p.stems.clone()).unwrap_or_default();
        let doc = nmtype::is_document(&place(c));
        let tag = keep.record(c).map_or("-", |rec| nmpod::evidence::tag(rec, &stems, doc).name());
        let share = ev.parts.get(k).and_then(|x| x.2.get(c)).copied().unwrap_or(0.0);
        let acc = ev.parts.get(k).is_some_and(|x| ev.accepted(x.0, c));
        let why = ev.settle_why.get(k).and_then(|v| v.get(c)).map_or_else(String::new, |(h, p)| format!("/{:.2},{:.2},{:.2}/p{p:.1}", h[0], h[1], h[2]));
        format!("{r:.2}/{s:.3}/{tag}/{share:.2}{why}{}", if acc { "" } else { "/na" })
    };
    let mut out = Vec::new();
    out.push(format!(
        "parts: {}; relations: {}",
        parts
            .iter()
            .enumerate()
            .map(|(k, p)| {
                let (pm, nm) = ev.settle_mass.get(k).copied().unwrap_or_default();
                format!("k{k} {} ({:?}, mass {pm:.1} against {nm:.1} nats)", p.text, p.role)
            })
            .collect::<Vec<_>>()
            .join(", "),
        req.relations.iter().map(|r| r.name()).collect::<Vec<_>>().join(", ")
    ));
    let cands = &run.cands;
    let at = |c: usize| cands.iter().position(|x| *x == c).map_or_else(|| "-".to_owned(), |p| (p + 1).to_string());
    let row = |c: usize| (0..parts.len()).map(|k| cell(k, c)).collect::<Vec<_>>().join(" ");
    let mark = |c: usize| if ans.contains(&c) { "*" } else { "" };
    out.push(format!(
        "set ({}): {}",
        run.g2run.stop,
        run.g2run.set.iter().map(|c| format!("{}{} t{} g{} [{}]", mark(*c), place(*c), keep.tokens(*c), at(*c), row(*c))).collect::<Vec<_>>().join(" | ")
    ));
    for c in ans {
        out.push(format!("ans {} t{} g{} set {} [{}]", place(*c), keep.tokens(*c), at(*c), u8::from(run.g2run.set.contains(c)), row(*c)));
    }
    for k in 0..parts.len() {
        let mut v: Vec<(f32, usize)> = cands.iter().map(|c| (ev.settles(k, *c), *c)).filter(|x| x.0 > 0.0).collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let best_ans = v.iter().position(|(_, c)| ans.contains(c)).map_or_else(|| "-".to_owned(), |p| (p + 1).to_string());
        out.push(format!(
            "k{k} {}: positive {}, first answer at {best_ans}: {}",
            parts.get(k).map_or("", |p| p.text.as_str()),
            v.len(),
            v.iter().take(4).map(|(_, c)| format!("{}{} t{} [{}]", mark(*c), place(*c), keep.tokens(*c), cell(k, *c))).collect::<Vec<_>>().join(" | ")
        ));
    }
    out
}

/// One question's whole selection input for the replay bench (`tests/replay/`).
#[expect(clippy::too_many_lines, reason = "one JSON object, every field the replay reads")]
fn dump_line(sub: &nmsubs::Substrate, keep: &nmapi::drone_windows::Keep, graph: &nmsubs::refgraph::RefGraph, q: &Question, run: &nmapi::drone_path::StackRun, ans: &[usize]) -> String {
    use std::collections::BTreeMap;
    let num = |x: f64| if x.is_finite() { Value::Number(format!("{x:.6}")) } else { Value::Null };
    let n = |x: usize| Value::Number(x.to_string());
    let s = |x: &str| Value::String(x.to_owned());
    let obj = |v: Vec<(&str, Value)>| Value::Object(v.into_iter().map(|(k, v)| (k.to_owned(), v)).collect::<BTreeMap<_, _>>());
    let key = |c: usize| {
        sub.chunks.get(c).map_or_else(String::new, |ch| format!("{}:{}-{}", sub.files.get(ch.file).map_or("", |f| f.path.as_str()).replace('\\', "/"), ch.start, ch.end))
    };
    let req = nmpod::intake::Request::read(&q.text, &[]);
    let parts: Vec<&nmpod::intake::Part> = req.aspects.iter().flat_map(|a| a.parts.iter()).collect();
    let mut out = vec![("id", s(&q.id)), ("kind", s(&q.kind)), ("question", s(&q.text))];
    out.push((
        "answers",
        Value::Array(
            q.answers
                .iter()
                .map(|a| obj(vec![("file", s(&a.file)), ("first", n(a.first)), ("last", n(a.last)), ("whole_file", Value::Bool(a.whole_file)), ("resolved", Value::Bool(true))]))
                .collect(),
        ),
    ));
    let Some(ev) = run.evidence.as_ref() else { return obj(out).encode().unwrap_or_default() };
    let chance = nmapi::drone_path::Chance::new(ev);
    out.push((
        "request",
        obj(vec![
            ("kind", s(req.kind)),
            (
                "parts",
                Value::Array(
                    parts
                        .iter()
                        .enumerate()
                        .map(|(k, p)| {
                            let (pm, nm) = ev.settle_mass.get(k).copied().unwrap_or_default();
                            obj(vec![
                                ("text", s(&p.text)),
                                ("role", s(&format!("{:?}", p.role))),
                                ("stems", Value::Array(p.stems.iter().map(|x| s(x)).collect())),
                                ("mass", num(f64::from(pm))),
                                ("against", num(f64::from(nm))),
                                ("base", num(f64::from(ev.settle_base.get(k).copied().unwrap_or(0.0)))),
                                ("conc", ev.concentration.get(k).copied().flatten().map_or(Value::Null, |c| num(f64::from(c)))),
                                (
                                    "conc_null",
                                    Value::Array(ev.concentration_null.get(k).map_or_else(Vec::new, |v| v.iter().map(|c| c.map_or(Value::Null, |c| num(f64::from(c)))).collect())),
                                ),
                                ("assoc", ev.association.get(k).copied().flatten().map_or(Value::Null, |c| num(f64::from(c)))),
                                (
                                    "assoc_null",
                                    Value::Array(ev.association_null.get(k).map_or_else(Vec::new, |v| v.iter().map(|c| c.map_or(Value::Null, |c| num(f64::from(c)))).collect())),
                                ),
                                ("holders", n(ev.holders.get(k).map_or(0, |h| h.0))),
                                ("files", n(ev.holders.get(k).map_or(0, |h| h.1))),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("relations", Value::Array(req.relations.iter().map(|r| s(r.name())).collect())),
        ]),
    ));
    out.push(("budget", n(run.g2run.room)));
    out.push(("band", num(f64::from(run.g2run.band))));
    // G2's own pool: its candidates at its last selection (G1's final hand-over, then its queue,
    // and pieces forced into the set), each piece once. `run.queue` is the gate's queue after G2,
    // so chained with `run.handover` it repeats hand-over pieces and drops the queue pieces G2 sent
    // up. Without G2 (`cands` empty): G1's hand-over and queue.
    let mut g1_order: Vec<usize> = Vec::new();
    let pool: Vec<usize> = if run.cands.is_empty() { run.handover.iter().chain(&run.queue).copied().collect() } else { run.cands.clone() };
    for c in pool {
        if !g1_order.contains(&c) {
            g1_order.push(c);
        }
    }
    let mut win: BTreeMap<usize, usize> = BTreeMap::new();
    for (w, cs) in run.windows.iter().enumerate() {
        for c in cs {
            win.entry(*c).or_insert(run.owners.get(w).copied().unwrap_or(w));
        }
    }
    let mut all: Vec<usize> = g1_order.clone();
    for c in ans {
        if !all.contains(c) {
            all.push(*c);
        }
    }
    let cell = |k: usize, c: usize| {
        let doc = nmtype::is_document(&key(c));
        let stems: Vec<String> = parts.get(k).map(|p| p.stems.clone()).unwrap_or_default();
        let tag = keep.record(c).map_or("-", |rec| nmpod::evidence::tag(rec, &stems, doc).name());
        let (h, p) = ev.settle_why.get(k).and_then(|v| v.get(c)).copied().unwrap_or(([0.0; 3], 0.0));
        // Per family (wording, naming, meaning), the best signal's surprise for the part's aspect.
        let a = ev.parts.get(k).map_or(0, |x| x.0);
        let fam = [nmpod::evidence::Family::Wording, nmpod::evidence::Family::Naming, nmpod::evidence::Family::Meaning]
            .iter()
            .map(|f| num(f64::from(nmpod::evidence::FAMILY.iter().enumerate().filter(|(_, g)| *g == f).map(|(s, _)| ev.surprise(a, s, c)).fold(0.0_f32, f32::max))))
            .collect();
        obj(vec![
            ("f", Value::Array(fam)),
            ("r", num(f64::from(chance.first_form(ev, k, c)))),
            ("s", num(f64::from(ev.settles(k, c)))),
            ("tag", s(tag)),
            ("share", num(f64::from(ev.parts.get(k).and_then(|x| x.2.get(c)).copied().unwrap_or(0.0)))),
            ("h", Value::Array(h.iter().map(|x| num(f64::from(*x))).collect())),
            ("p", num(f64::from(p))),
            ("accepted", Value::Bool(ev.parts.get(k).is_some_and(|x| ev.accepted(x.0, c)))),
            ("holds", Value::Bool(ev.part_holds(k, c))),
        ])
    };
    // Edges among the pool only: an appended answer piece is not reached through a pool piece.
    let index: BTreeMap<usize, usize> = g1_order.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    out.push((
        "candidates",
        Value::Array(
            all.iter()
                .map(|&c| {
                    let ch = sub.chunks.get(c);
                    let edges: Vec<Value> = graph
                        .specific
                        .edges_of(c)
                        .iter()
                        .filter(|e| index.contains_key(&(e.to as usize)))
                        .map(|e| obj(vec![("to", s(&key(e.to as usize))), ("kind", s(&format!("{:?}", e.kind))), ("dir", s(&format!("{:?}", e.dir)))]))
                        .collect();
                    obj(vec![
                        ("key", s(&key(c))),
                        ("path", s(&key(c).rsplit_once(':').map_or(String::new(), |x| x.0.to_owned()))),
                        ("first", n(ch.map_or(0, |x| x.start))),
                        ("last", n(ch.map_or(0, |x| x.end))),
                        ("tokens", n(keep.tokens(c).max(1))),
                        ("g1", g1_order.iter().position(|x| *x == c).map_or(Value::Null, |p| n(p + 1))),
                        ("handover", Value::Bool(run.handover.contains(&c))),
                        ("added", Value::Bool(!g1_order.contains(&c))),
                        ("window", win.get(&c).map_or(Value::Null, |t| n(t + 1))),
                        ("piece_kind", s(ch.map_or("", |x| x.kind))),
                        ("cells", Value::Array((0..parts.len()).map(|k| cell(k, c)).collect())),
                        ("edges", Value::Array(edges)),
                    ])
                })
                .collect(),
        ),
    ));
    out.push((
        "facts",
        Value::Array(run.g2run.facts.iter().map(|(k, v)| obj(vec![("relation", n(*k)), ("pieces", Value::Array(v.iter().map(|c| s(&key(*c))).collect()))])).collect()),
    ));
    out.push(("rel_pairs", Value::Array(run.g2run.rel_pairs.iter().map(|(p, qq, k)| Value::Array(vec![s(&key(*p)), s(&key(*qq)), n(*k)])).collect())));
    out.push((
        "served",
        obj(vec![
            ("set", Value::Array(run.g2run.set.iter().map(|c| s(&key(*c))).collect())),
            ("up", Value::Array(run.up.iter().map(|c| s(&key(*c))).collect())),
            ("rivals", Value::Array(run.g2run.rivals.iter().map(|c| s(&key(*c))).collect())),
            ("stop", s(&run.g2run.stop)),
            ("gate", s(&run.g2run.gate)),
            ("pack_cited", Value::Array(cited(&run.pack).iter().map(|x| s(&format!("{}:{}-{}", x.0, x.1, x.2))).collect())),
        ]),
    ));
    obj(out).encode().unwrap_or_default()
}

/// The pairs and places of `StackMark::pairs` and `ranks` for one question.
#[expect(clippy::type_complexity, reason = "two plain tables")]
fn reliability(run: &nmapi::drone_path::StackRun, ans: &[usize]) -> (Vec<(f32, f32, bool)>, Vec<(Option<usize>, Option<usize>)>) {
    let Some(ev) = run.evidence.as_ref() else { return (Vec::new(), Vec::new()) };
    let chance = nmapi::drone_path::Chance::new(ev);
    let mut pairs = Vec::new();
    let mut ranks = Vec::new();
    for k in 0..ev.parts.len() {
        for c in &run.cands {
            let (a, b) = (chance.first_form(ev, k, *c), ev.settles(k, *c));
            if a > 0.0 || b > 0.0 {
                pairs.push((a, b, ans.contains(c)));
            }
        }
        let place = |f: &dyn Fn(usize) -> f32| {
            let mut v: Vec<(f32, usize)> = run.cands.iter().map(|c| (f(*c), *c)).filter(|x| x.0 > 0.0).collect();
            v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            v.iter().position(|(_, c)| ans.contains(c))
        };
        ranks.push((place(&|c| chance.first_form(ev, k, c)), place(&|c| ev.settles(k, c))));
    }
    (pairs, ranks)
}

/// The reliability table of `--why-r`: the pairs binned by each chance; per bin, the pairs, the
/// mean chance and the share that are answer pieces; then the places of the first answer piece.
fn reliability_lines(sm: &[StackMark]) -> Vec<String> {
    let pairs: Vec<(f32, f32, bool)> = sm.iter().flat_map(|m| m.pairs.iter().copied()).collect();
    let edges = [0.0_f32, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.000_001];
    let mut out = Vec::new();
    for (name, f) in [("part chance (served r)", 0_usize), ("settle chance (new r)", 1)] {
        let mut row = Vec::new();
        for w in edges.windows(2) {
            let (lo, hi) = (w.first().copied().unwrap_or(0.0), w.get(1).copied().unwrap_or(1.0));
            let b: Vec<&(f32, f32, bool)> = pairs
                .iter()
                .filter(|p| {
                    let x = if f == 0 { p.0 } else { p.1 };
                    x > 0.0 && x >= lo && x < hi
                })
                .collect();
            if b.is_empty() {
                continue;
            }
            #[expect(clippy::cast_precision_loss, reason = "counts")]
            let mean = b.iter().map(|p| if f == 0 { p.0 } else { p.1 }).sum::<f32>() / b.len() as f32;
            let pos = b.iter().filter(|p| p.2).count();
            #[expect(clippy::cast_precision_loss, reason = "counts")]
            row.push(format!("[{lo:.2},{hi:.2}) n {} mean {mean:.3} answer {pos} ({:.3})", b.len(), pos as f32 / b.len() as f32));
        }
        out.push(format!("reliability, {name}: {}", row.join("; ")));
    }
    type RankPair = (Option<usize>, Option<usize>);
    let ranks: Vec<RankPair> = sm.iter().flat_map(|m| m.ranks.iter().copied()).collect();
    let summ = |g: &dyn Fn(&RankPair) -> Option<usize>| {
        let first = ranks.iter().filter(|r| g(r) == Some(0)).count();
        let top3 = ranks.iter().filter(|r| g(r).is_some_and(|x| x < 3)).count();
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let mrr = ranks.iter().map(|r| g(r).map_or(0.0, |x| 1.0 / (x + 1) as f32)).sum::<f32>() / ranks.len().max(1) as f32;
        format!("first {first}, top 3 {top3}, mean reciprocal place {mrr:.3}")
    };
    out.push(format!("first answer piece per part among G2's candidates ({} parts): served r {}; settle chance {}", ranks.len(), summ(&|r| r.0), summ(&|r| r.1)));
    out
}

/// Card paths in a child process of this program; its rows come back on standard output.
fn card_marks(root: &Path, qfile: &Path, own: &[String], extra: &[String], modes: &[&str]) -> Result<Vec<(String, Vec<Mark>)>, String> {
    let exe = std::env::current_exe().map_err(|e| format!("program path: {e}"))?;
    let mut args: Vec<String> =
        vec!["bench".into(), "--mode".into(), "board".into(), "--rows".into(), "--paths".into(), modes.join(","), "--root".into(), root.display().to_string()];
    args.extend(["--questions".into(), qfile.display().to_string()]);
    if let Some(n) = arg(own, "--n") {
        args.extend(["--n".into(), n.clone()]);
    }
    args.push("--".into());
    args.extend(extra.iter().cloned());
    let o = std::process::Command::new(exe)
        .args(&args)
        .stderr(std::process::Stdio::inherit())
        .output()
        .map_err(|e| format!("card paths: {e}"))?;
    if !o.status.success() {
        return Err(format!("card paths: child process failed ({})", o.status));
    }
    let mut out: Vec<(String, Vec<Mark>)> = modes.iter().map(|m| ((*m).to_owned(), Vec::new())).collect();
    for l in String::from_utf8_lossy(&o.stdout).lines() {
        let f: Vec<&str> = l.split('\t').collect();
        let [_, mode, id, first, top3, pack, every, cited, tokens, secs, withheld, low, place] = f.as_slice() else { continue };
        if !l.starts_with("row\t") {
            continue;
        }
        let b = |x: &str| x == "1";
        if let Some((_, v)) = out.iter_mut().find(|(m, _)| m == mode) {
            v.push(Mark {
                id: (*id).to_owned(),
                first: b(first),
                top3: b(top3),
                pack: b(pack),
                every: b(every),
                cited: b(cited),
                tokens: tokens.parse().unwrap_or(0),
                secs: secs.parse().unwrap_or(0.0),
                withheld: b(withheld),
                low: b(low),
                place: (*place).to_owned(),
                rr: 0.0,
            });
        }
    }
    Ok(out)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    match v.len() {
        0 => 0.0,
        n if n % 2 == 1 => v.get(n / 2).copied().unwrap_or(0.0),
        n => (v.get(n / 2 - 1).copied().unwrap_or(0.0) + v.get(n / 2).copied().unwrap_or(0.0)) / 2.0,
    }
}

/// Seconds after the first call (the first pays for anything left cold), or all with one call.
fn warm(secs: &[f64]) -> (f64, f64) {
    let mut v: Vec<f64> = if secs.len() > 1 { secs.get(1..).unwrap_or_default().to_vec() } else { secs.to_vec() };
    let worst = v.iter().copied().fold(0.0_f64, f64::max);
    (median(&mut v), worst)
}

/// `bench --mode board ...`.
#[expect(clippy::too_many_lines, reason = "one report, written top to bottom")]
#[expect(clippy::cast_precision_loss, reason = "token counts for a report")]
pub fn run(home: &Path, own: &[String], extra: &[String]) -> Result<String, String> {
    let root = match arg(own, "--root") {
        Some(r) if Path::new(r).is_dir() => nmsubs::repo::canon(Path::new(r)),
        Some(r) => return Err(format!("--root {r} is not a folder")),
        None => home.to_path_buf(),
    };
    let qfile = arg(own, "--questions").map_or_else(|| home.join("tests/dev/questions.json"), PathBuf::from);
    let mut questions = load(&root, &qfile)?;
    if let Some(n) = arg(own, "--n") {
        questions.truncate(n.parse::<usize>().map_err(|_| format!("--n takes a count, got {n}"))?);
    }
    let asked: Vec<&str> = arg(own, "--paths").map_or("plain,stack", String::as_str).split(',').map(str::trim).collect();
    if let Some(p) = asked.iter().find(|p| !PATHS.contains(p)) {
        return Err(format!("--paths takes plain, flow, stack, fast and full, got {p}"));
    }
    // Processor paths first, card paths after, each once, in a fixed order.
    let paths: Vec<&str> = PATHS.iter().copied().filter(|p| asked.contains(p)).collect();
    let card: Vec<&str> = paths.iter().copied().filter(|p| CARD.contains(p)).collect();
    let whole: Vec<String> = ["--whole", "--neural"].iter().map(|s| (*s).to_owned()).chain(extra.iter().cloned()).collect();

    // `--rows`: the child process of a board run, serving the card paths and printing raw rows.
    if own.iter().any(|a| a == "--rows") {
        let marks = serve_marks(home, &root, &questions, &card, &whole)?;
        let mut out = String::new();
        for (mode, ms) in &marks {
            for m in ms {
                let b = |x: bool| u8::from(x);
                let _ = writeln!(
                    out,
                    "row\t{mode}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{}\t{}",
                    m.id,
                    b(m.first),
                    b(m.top3),
                    b(m.pack),
                    b(m.every),
                    b(m.cited),
                    m.tokens,
                    m.secs,
                    b(m.withheld),
                    b(m.low),
                    m.place
                );
            }
        }
        return Ok(out);
    }

    let lenses = arg(own, "--lenses").is_some_and(|v| v == "on");
    // `--windows flags|lenses|team|firstfit|single` forms the stack's windows that way (default:
    // lenses with `--lenses on`, else flags); `--match team` gives a comparison arm the tokens
    // the team took on each question.
    let arm = match arg(own, "--windows") {
        Some(w) => nmapi::drone_path::Windows::parse(w).ok_or_else(|| format!("--windows takes flags, lenses, team, firstfit or single, got {w}"))?,
        None if lenses => nmapi::drone_path::Windows::Lenses,
        None => nmapi::drone_path::Windows::Flags,
    };
    // TEAM-2: the comparison arms also form the served flow's windows.
    if matches!(arm, nmapi::drone_path::Windows::FirstFit | nmapi::drone_path::Windows::Single) {
        nmapi::drone_path::set_windows(Some(arm));
    }
    let matched = arg(own, "--match").is_some_and(|v| v == "team");
    // `--g1 pooled|set|product`: how G1 ranks (default: pooled for the team, product otherwise);
    // `pooled` values every piece on the scale shared by all tentacles, `set` pools over the
    // merged set of the round's reports (the comparison form).
    match arg(own, "--g1").map(String::as_str) {
        Some("pooled") => nmapi::drone_path::set_pooled(Some(true)),
        Some("set") => {
            nmapi::drone_path::set_pooled(Some(true));
            nmapi::drone_path::set_common(false);
        }
        Some("product") => nmapi::drone_path::set_pooled(Some(false)),
        Some(g) => return Err(format!("--g1 takes pooled, set or product, got {g}")),
        None => {}
    }
    // `--race off`: G1 forwards by the round's capacity in place of its race (comparison form).
    if arg(own, "--race").is_some_and(|v| v == "off") {
        nmapi::drone_path::set_race(false);
    }
    // `--g2 cover`: G2 comprehends by coverage and sends orders down the motor nervous system;
    // without it G2 passes G1's order up to the capacity.
    if arg(own, "--g2").is_some_and(|v| v == "cover") {
        nmapi::drone_path::set_g2(true);
    }
    // `--r settle`: G2's selection reads the settle chance in place of the first form.
    if arg(own, "--r").is_some_and(|v| v == "settle") {
        nmapi::drone_path::set_settle(true);
    }
    // `--facts off`: G2 reads no relation as a fact of the code's structure (for comparison).
    if arg(own, "--facts").is_some_and(|v| v == "off") {
        nmapi::drone_path::set_facts(false);
    }
    // `--dump-candidates <dir>`: every question's whole selection input for the replay bench, one
    // JSON line per question in `<dir>/candidates.jsonl`; `--force-answer windows|handover|set`:
    // the known answer pieces placed by hand at that stage, the later stages run as normal (guide
    // section 13). Neither changes anything when not given.
    let dump = arg(own, "--dump-candidates").map(PathBuf::from);
    let force = match arg(own, "--force-answer").map(String::as_str) {
        None => None,
        Some("windows") => Some(nmapi::drone_path::ForceAt::Windows),
        Some("handover") => Some(nmapi::drone_path::ForceAt::Handover),
        Some("set") => Some(nmapi::drone_path::ForceAt::Set),
        Some(s) => return Err(format!("--force-answer takes windows, handover or set, got {s}")),
    };
    // `--judge on`: the Core Brain's judge reads the pieces at the inclusion edge of G2's selection.
    if arg(own, "--judge").is_some_and(|v| v == "on") {
        nmapi::drone_path::set_judge(true);
    }
    // `--call-s <seconds>`: the caller's time budget per call (the judge's round fills what is left).
    if let Some(v) = arg(own, "--call-s") {
        nmapi::drone_path::set_call_budget(Some(v.parse::<f64>().map_err(|_| format!("--call-s takes seconds, got {v}"))?));
    }
    // `--form adaptive|cited|answer|full|none`: the form the served paths are asked for.
    if let Ok(mut f) = FORM.write() {
        *f = match arg(own, "--form").map(String::as_str) {
            None => "full".to_owned(),
            Some("none") => String::new(),
            Some(x) => x.to_owned(),
        };
    }
    let mut served: Vec<(String, Vec<Mark>)> = Vec::new();
    let mut stack: Option<(Vec<StackMark>, Vec<usize>)> = None;
    if paths.contains(&"plain") {
        served.extend(serve_marks(home, &root, &questions, &["plain"], extra)?);
    }
    // `flow`: the served one flow (`serve --flow`), end to end through the tools as `plain` is.
    if paths.contains(&"flow") {
        let flow: Vec<String> = std::iter::once("--flow".to_owned()).chain(extra.iter().cloned()).collect();
        served.extend(serve_marks(home, &root, &questions, &["flow"], &flow)?);
    }
    if paths.contains(&"stack") {
        stack = Some(stack_marks(home, &root, &questions, arm, matched, own.iter().any(|a| a == "--why"), own.iter().any(|a| a == "--why-r"), dump.as_deref(), force, extra)?);
    }
    if !card.is_empty() {
        served.extend(card_marks(&root, &qfile, own, extra, &card)?);
    }

    // The placements, in path order, with no times: the run's hash.
    let mut places: Vec<String> = Vec::new();
    for p in &paths {
        if *p == "stack" {
            if let Some((sm, _)) = &stack {
                places.extend(sm.iter().map(|m| format!("stack {} {}", m.id, m.place())));
            }
        } else if let Some((_, ms)) = served.iter().find(|(m, _)| m == p) {
            places.extend(ms.iter().map(|m| format!("{p} {} {}", m.id, m.place)));
        }
    }
    let placements = nmhash::sha256_bytes(places.join("\n").as_bytes());
    let program = std::env::current_exe()
        .ok()
        .and_then(|e| std::fs::File::open(e).ok())
        .and_then(|f| nmhash::sha256(f).ok())
        .unwrap_or_else(|| "unknown".into());
    let opts = |p: &str| if CARD.contains(&p) { whole.join(" ") } else if extra.is_empty() { "(none)".to_owned() } else { extra.join(" ") };
    let serve: Vec<String> = paths.iter().filter(|p| **p != "stack").map(|p| format!("{p}: {}", opts(p))).collect();
    let serve = if serve.is_empty() { "(stack only)".to_owned() } else { serve.join("; ") };

    let answered: Vec<&Question> = questions.iter().filter(|q| !q.answers.is_empty()).collect();
    let n = answered.len();
    let is_answered = |id: &str| answered.iter().any(|q| q.id == id);
    let decoys = questions.len() - n;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "board: program {program}  serve {serve}  stack --windows {}{}{}  placements {placements}",
        arm.name(),
        arg(own, "--g1").map_or(String::new(), |g| format!(" --g1 {g}")),
        if matched { " --match team" } else { "" }
    );
    let _ = writeln!(
        out,
        "questions: {} from {} ({n} with answers, {decoys} decoys), root {}",
        questions.len(),
        qfile.strip_prefix(home).unwrap_or(&qfile).display(),
        root.display()
    );
    let _ = writeln!(out, "counts over the {n} questions with answers; tokens over all; seconds after the first call");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "path    first  top3  pack every cited   tok med tok total   s med s worst  decoys |    up    f4 every-up aspects"
    );
    for p in &paths {
        if *p == "stack" {
            let Some((sm, _)) = &stack else { continue };
            let a: Vec<&StackMark> = sm.iter().filter(|m| is_answered(&m.id)).collect();
            let c = |f: &dyn Fn(&StackMark) -> bool| a.iter().filter(|m| f(m)).count();
            let (med, worst) = warm(&sm.iter().map(|m| m.secs).collect::<Vec<_>>());
            let mut tok: Vec<f64> = sm.iter().map(|m| m.pack.tokens as f64).collect();
            let held = sm.iter().filter(|m| !is_answered(&m.id) && (m.pack.withheld || m.pack.low)).count();
            let decoys = sm.len() - a.len();
            let _ = writeln!(
                out,
                "{:<6} {:>6} {:>5} {:>5} {:>5} {:>5} {:>9.0} {:>9} {:>7.3} {:>7.3} {:>7} | {:>5} {:>5} {:>8} {:>7}",
                p,
                c(&|m| m.up == Some(0)),
                c(&|m| m.up.is_some_and(|x| x < 3)),
                c(&|m| m.pack.pack),
                c(&|m| m.pack.every),
                c(&|m| m.pack.cited),
                median(&mut tok),
                sm.iter().map(|m| m.pack.tokens).sum::<usize>(),
                med,
                worst,
                format!("{held}/{decoys}"),
                c(&|m| m.up.is_some()),
                c(&|m| m.up.is_some_and(|x| x < nmapi::whole::FAST_K)),
                c(&|m| m.every_up),
                format!("{}/{}", a.iter().map(|m| m.aspects.0).sum::<usize>(), a.iter().map(|m| m.aspects.1).sum::<usize>())
            );
            continue;
        }
        let Some((_, ms)) = served.iter().find(|(m, _)| m == p) else { continue };
        let a: Vec<&Mark> = ms.iter().filter(|m| is_answered(&m.id)).collect();
        let c = |f: &dyn Fn(&Mark) -> bool| a.iter().filter(|m| f(m)).count();
        let mut tok: Vec<f64> = ms.iter().map(|m| m.tokens as f64).collect();
        let total: usize = ms.iter().map(|m| m.tokens).sum();
        let (med, worst) = warm(&ms.iter().map(|m| m.secs).collect::<Vec<_>>());
        let held = ms.iter().filter(|m| !is_answered(&m.id) && (m.withheld || m.low)).count();
        let _ = writeln!(
            out,
            "{:<6} {:>6} {:>5} {:>5} {:>5} {:>5} {:>9.0} {:>9} {:>7.3} {:>7.3} {:>7} | {:>5} {:>5} {:>8} {:>7}",
            p,
            c(&|m| m.first),
            c(&|m| m.top3),
            c(&|m| m.pack),
            c(&|m| m.every),
            c(&|m| m.cited),
            median(&mut tok),
            total,
            med,
            worst,
            format!("{held}/{decoys}"),
            "-",
            "-",
            "-",
            "-"
        );
    }
    if let Some((sm, per_t)) = &stack {
        let _ = writeln!(
            out,
            "stack: answer reported by tentacle {}",
            per_t.iter().enumerate().map(|(t, c)| format!("t{} {c}", t + 1)).collect::<Vec<_>>().join(", ")
        );
        let heads: usize = sm.iter().filter(|m| is_answered(&m.id)).filter_map(|m| m.heads).sum();
        let code = sm.iter().filter(|m| is_answered(&m.id) && m.heads.is_some()).count();
        let decoys: Vec<String> =
            sm.iter().filter(|m| !is_answered(&m.id)).map(|m| format!("{} {}/{}", m.id, m.absent.0, m.absent.1)).collect();
        let _ = writeln!(
            out,
            "stack: file heads in the up set {heads} over {code} code questions; decoys, tentacles naming an aspect with no information: {}",
            if decoys.is_empty() { "none".to_owned() } else { decoys.join(", ") }
        );
        // The team: what its windows took in, and the design's invariants.
        let a: Vec<&StackMark> = sm.iter().filter(|m| is_answered(&m.id)).collect();
        let c = |f: &dyn Fn(&StackMark) -> bool| a.iter().filter(|m| f(m)).count();
        let sum2 = |f: &dyn Fn(&StackMark) -> (usize, usize)| a.iter().map(|m| f(m)).fold((0, 0), |s, x| (s.0 + x.0, s.1 + x.1));
        let med = |f: &dyn Fn(&StackMark) -> usize| {
            let mut v: Vec<f64> = sm.iter().map(|m| f(m) as f64).collect();
            median(&mut v)
        };
        let (rh, rt) = sum2(&|m| m.ranges);
        let (ah, at) = sum2(&|m| m.aspects_win);
        let (xc, xe) = sm.iter().map(|m| m.cross).fold((0, 0), |s, x| (s.0 + x.0, s.1 + x.1));
        let _ = writeln!(
            out,
            "team: answer in windows {}, every answer range in windows {}, ranges {rh}/{rt}, aspects held {ah}/{at}, in G1 hand-over {} (first {}, top 3 {}); pieces med {:.0}, tokens med {:.0}, rounds of windows med {:.0}; relations across windows {xc} of {xe} ({:.1}%)",
            c(&|m| m.in_win),
            c(&|m| m.every_win),
            c(&|m| m.hand.is_some()),
            c(&|m| m.hand == Some(0)),
            c(&|m| m.hand.is_some_and(|x| x < 3)),
            med(&|m| m.pieces),
            med(&|m| m.tokens),
            med(&|m| m.window_rounds),
            if xe > 0 { 100.0 * xc as f64 / xe as f64 } else { 0.0 }
        );
        let _ = writeln!(
            out,
            "G1: every answer range in the hand-over {}; hand-over med {:.0}; race: committed med {:.0}, ties {}, aspects uncovered {}",
            c(&|m| m.every_hand),
            med(&|m| m.hand_len),
            med(&|m| m.race.0),
            sm.iter().map(|m| m.race.1).sum::<usize>(),
            sm.iter().map(|m| m.race.2).sum::<usize>()
        );
        // The accuracy equation's factors (every answer range at each stage) and the time terms.
        let plain_every = served.iter().find(|(m, _)| m == "plain").map_or_else(|| "-".to_owned(), |(_, ms)| ms.iter().filter(|m| is_answered(&m.id) && m.every).count().to_string());
        let medf = |f: &dyn Fn(&StackMark) -> f64| {
            let mut v: Vec<f64> = sm.iter().map(f).collect();
            median(&mut v)
        };
        let worstf = |f: &dyn Fn(&StackMark) -> f64| sm.iter().map(f).fold(0.0_f64, f64::max);
        let _ = writeln!(
            out,
            "factors (every answer range): in windows {}, in G1's hand-over {}, sent up {}, in the stack's pack {}, in the plain pack {plain_every}; up tokens med {:.0}",
            c(&|m| m.every_win),
            c(&|m| m.every_hand),
            c(&|m| m.every_up),
            c(&|m| m.pack.every),
            med(&|m| m.up_tokens)
        );
        let _ = writeln!(
            out,
            "time terms, s med / worst: drones (request and evidence) {:.3} / {:.3}, windows formed {:.3} / {:.3}, G1 with the window reads {:.3} / {:.3}, G2 {:.4} / {:.4}, pack {:.4} / {:.4}; judge, recall: not on this path",
            medf(&|m| m.times[0]),
            worstf(&|m| m.times[0]),
            medf(&|m| m.times[1]),
            worstf(&|m| m.times[1]),
            medf(&|m| m.times[2]),
            worstf(&|m| m.times[2]),
            medf(&|m| m.g2ms / 1000.0),
            worstf(&|m| m.g2ms / 1000.0),
            medf(&|m| m.pack_s),
            worstf(&|m| m.pack_s)
        );
        if sm.iter().any(|m| !m.g2gate.is_empty()) {
            let mut stops: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
            for m in &sm[..] {
                *stops.entry(m.g2stop.as_str()).or_default() += 1;
            }
            let _ = writeln!(
                out,
                "G2 selection: F med {:.2} of aspects and relations med {:.0}; stopped by {}",
                medf(&|m| f64::from(m.g2value.0)),
                med(&|m| m.g2value.1),
                stops.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", ")
            );
            let gate = |g: &str| sm.iter().filter(|m| m.g2gate == g).count();
            let settled: Vec<&&StackMark> = a.iter().filter(|m| m.g2gate == "settled").collect();
            let mut ms: Vec<f64> = sm.iter().map(|m| m.g2ms).collect();
            let _ = writeln!(
                out,
                "G2: gate settled {} (every answer range up on {} of them), tie {}, absent {}; every answer range in the chosen set {}; chosen set med {:.0}, up med {:.0}; orders {}, pieces read on orders {}; refused for a missing copy to G1 {}, edges refused by class {}; G2 ms med {:.2}, worst {:.2}",
                gate("settled"),
                settled.iter().filter(|m| m.every_up).count(),
                gate("tie"),
                gate("absent"),
                c(&|m| m.every_set),
                med(&|m| m.g2set),
                med(&|m| m.up_len),
                sm.iter().map(|m| m.g2orders).sum::<usize>(),
                sm.iter().map(|m| m.g2read).sum::<usize>(),
                sm.iter().map(|m| m.motor_refused).sum::<usize>(),
                sm.iter().map(|m| m.class_refused).sum::<usize>(),
                median(&mut ms),
                ms.iter().copied().fold(0.0_f64, f64::max)
            );
        }
        if sm.iter().any(|m| m.judge.0 > 0) {
            let mut js: Vec<f64> = sm.iter().map(|m| m.judge.1 / 1000.0).collect();
            let _ = writeln!(
                out,
                "judge at the inclusion edge: pieces judged {} (med {:.0} per question), moved into or out of the set {}, s med {:.3}, worst {:.3}; tokens read {}, s per piece judged {:.3}, ms per token read {:.3}",
                sm.iter().map(|m| m.judge.0).sum::<usize>(),
                med(&|m| m.judge.0),
                sm.iter().map(|m| m.judge.2).sum::<usize>(),
                median(&mut js),
                js.iter().copied().fold(0.0_f64, f64::max),
                sm.iter().map(|m| m.judge.3).sum::<usize>(),
                js.iter().sum::<f64>() / sm.iter().map(|m| m.judge.0).sum::<usize>().max(1) as f64,
                js.iter().sum::<f64>() * 1000.0 / sm.iter().map(|m| m.judge.3).sum::<usize>().max(1) as f64
            );
        }
        let inv = sm.iter().fold([0_usize; 4], |mut s, m| {
            for (x, y) in s.iter_mut().zip(m.invariants) {
                *x += y;
            }
            s
        });
        let mut only = vec![0_usize; per_t.len()];
        for m in &a {
            for (t, x) in m.only.iter().enumerate() {
                if only.len() <= t {
                    only.resize(t + 1, 0);
                }
                if let Some(o) = only.get_mut(t) {
                    *o += x;
                }
            }
        }
        let mut seen: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
        for m in sm {
            for l in &m.labels {
                *seen.entry(l.clone()).or_default() += 1;
            }
        }
        let _ = writeln!(
            out,
            "team: invariants (each must be 0): pieces in two windows {}, held pieces not forwarded or queued once {}, reports changed with the others absent {}, isolation gate {}",
            inv[0], inv[1], inv[2], inv[3]
        );
        for l in reliability_lines(sm) {
            if sm.iter().any(|m| !m.pairs.is_empty()) {
                let _ = writeln!(out, "{l}");
            }
        }
        let _ = writeln!(
            out,
            "team: answer ranges only one tentacle held (diagnosis): {}; functions taken: {}",
            only.iter().enumerate().map(|(t, x)| format!("t{} {x}", t + 1)).collect::<Vec<_>>().join(", "),
            seen.iter().map(|(l, n)| format!("{l} {n}")).collect::<Vec<_>>().join(", ")
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "per question: served paths first top3 pack every cited (w withheld, l low); stack: place in the up set, every, tentacle#place (* reported)");
    for q in &questions {
        let mut line = format!("{:<5} {:<8}", q.id, q.kind);
        for p in &paths {
            if *p == "stack" {
                if let Some(m) = stack.as_ref().and_then(|(sm, _)| sm.iter().find(|m| m.id == q.id)) {
                    let w = m.window.map_or_else(|| "-".to_owned(), |(t, r, rep)| format!("t{}#{}{}", t + 1, r + 1, if rep { "*" } else { "" }));
                    let _ = write!(
                        line,
                        "  stack u{} e{} {w} w{}/{} h{}",
                        m.up.map_or_else(|| "-".to_owned(), |x| (x + 1).to_string()),
                        u8::from(m.every_up),
                        m.ranges.0,
                        m.ranges.1,
                        m.hand.map_or_else(|| "-".to_owned(), |x| (x + 1).to_string())
                    );
                    if !m.g2gate.is_empty() {
                        let _ = write!(line, " g2 {} set {} up {}", m.g2gate, m.g2set, m.up_len);
                    }
                    let b = |x: bool| if x { '1' } else { '0' };
                    let _ = write!(line, " pack {}{}{}{}{}", b(m.pack.first), b(m.pack.top3), b(m.pack.pack), b(m.pack.every), b(m.pack.cited));
                    let _ = write!(line, " rr {:.4} tok {} s {:.3}", m.pack.rr, m.pack.tokens, m.pack.secs);
                    if m.judge.0 > 0 {
                        let _ = write!(line, " judged {} in {:.2} s moved {}", m.judge.0, m.judge.1 / 1000.0, m.judge.2);
                    }
                    for l in &m.lost {
                        let _ = write!(line, "\n        lost {l}");
                    }
                    for l in &m.diag {
                        let _ = write!(line, "\n        {l}");
                    }
                }
            } else if let Some(m) = served.iter().find(|(x, _)| x == p).and_then(|(_, ms)| ms.iter().find(|m| m.id == q.id)) {
                let b = |x: bool| if x { '1' } else { '0' };
                let _ = write!(
                    line,
                    "  {p} {}{}{}{}{}{}{}",
                    b(m.first),
                    b(m.top3),
                    b(m.pack),
                    b(m.every),
                    b(m.cited),
                    if m.withheld { "w" } else { "" },
                    if m.low { "l" } else { "" }
                );
                let _ = write!(line, " rr {:.4} tok {} s {:.3}", m.rr, m.tokens, m.secs);
            }
        }
        let _ = writeln!(out, "{line}");
    }
    let runs = home.join("runs");
    let _ = std::fs::create_dir_all(&runs);
    let file = runs.join(format!("bench-board-{}.txt", nmtype::now_ms()));
    if std::fs::write(&file, &out).is_ok() {
        let _ = writeln!(out, "written: {}", file.strip_prefix(home).unwrap_or(&file).display());
    }
    Ok(out)
}
