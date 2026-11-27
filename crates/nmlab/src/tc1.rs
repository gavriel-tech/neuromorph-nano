//! `nano-lab run tc1 [n]` (2026-11-10): the first measurement of the tentacle core, first
//! form (design `docs/2026-11-10-neumor-design-tentacle-core-v1.md`, architecture by Luis Jake
//! Gabriel III). Short and isolated: the first `n` questions of the CX1 development half (default
//! 10, by index; the file `run g1` reads), one scan per question with raw scores recorded and the
//! windows kept, then two arms on that one scan:
//! - control: today's served pass (score `mass`, shortlist 6, the reranker judging inside each
//!   tentacle, two chunks forwarded per tentacle);
//! - core: one tentacle core per window reporting through its `g0_tx` to the G1 loop, G1's
//!   hand-over forwarded (`nmapi::tentacle_core::judge`).
//!
//! Per question: right chunk first inside its window, first across all windows, forwarded (and
//! its place), seconds, G1 rounds and the isolation gate, memory. The run stops before a question
//! that would take it past the test limit (`LIMIT_S`) at the mean time per question so far, and
//! says how many questions it read. Records hold question ids and indices, never question text.
//! Writes a new `runs/TC1-dev-<ms>.txt`.
//!
//! Scan dump and replay (2026-11-10, phase B: decision rules improved by replay, plan section 11):
//! - `run tc1 dump [n]`: the same two arms over the first `n` development questions (default all
//!   30), no test limit (an end-of-day run), and per question everything the cores and G1 decide
//!   from, saved by `nmapi::scan_dump` under `scratch/tc1-dump/<ms>/` as each question finishes
//!   (the live core loop's device reads on a tape). Record `runs/TC1-dump-<ms>.txt`, one progress
//!   line per question.
//! - `run tc1 replay [folder]`: the cores and the G1 loop over a dump on the processor only (no
//!   model loaded, no device), the same summary table as `run tc1`, and per question whether the
//!   replay equals the live core arm of the dump run (hand-over, rankings, rounds, gate, reads).
//!   Default folder: the newest under `scratch/tc1-dump/`. Record `runs/TC1-replay-<ms>.txt`.
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use nmapi::forage::Organism;
use nmapi::scan_dump::{self, ControlArm, CoreArm, QuestionDump, Tape};
use nmapi::tentacle_core::CoreRun;
use nmapi::tentacle_pass::{Pass, Scan, Score, Settings, TentaclePass};
use nmtype::Fault;

use crate::e2::Item;

/// Questions read by default (the brief of this measurement: the first 10 development questions).
const QUESTIONS: usize = 10;
/// The test limit for one graphics-card job (about 12 minutes), a fixed test gate.
const LIMIT_S: f64 = 690.0;
/// Seconds the run waits for the dense lane to index every chunk before the first question (so
/// the windows are laid out as served; the P1 tests' rule).
const DENSE_WAIT_S: u64 = 240;
/// Where dumps go (disposable work).
const DUMP_DIR: &str = "scratch/tc1-dump";

/// RAM and device memory now.
fn memory_line() -> String {
    let ram = nmffi::dynlib::process_memory().map_or_else(|e| e, |(peak, now)| format!("RAM peak {} MB, now {} MB", peak >> 20, now >> 20));
    let dev = nmapi::neural::model(&nmtype::nano_root()).ok().and_then(|m| m.pop.gpu.clone()).map_or_else(
        || "no device".to_owned(),
        |o| {
            let s = o.gpu.pool_stats();
            let free = o.gpu.memory().map_or_else(|e| e, |(f, t)| format!("device free {} MB of {} MB", f >> 20, t >> 20));
            format!(
                "device held by the process {} MB (kept {} MB, working peak {} MB), {free}",
                (s.live_kept + s.live_work + s.idle_bytes) >> 20,
                s.live_kept >> 20,
                s.peak_work >> 20
            )
        },
    );
    format!("{ram}; {dev}")
}

/// RAM only (the replay opens no device and loads no model).
fn ram_line() -> String {
    nmffi::dynlib::process_memory().map_or_else(|e| e, |(peak, now)| format!("RAM peak {} MB, now {} MB", peak >> 20, now >> 20))
}

fn show(k: Option<usize>) -> String {
    k.map_or_else(|| "-".to_owned(), |x| x.to_string())
}

/// One question's row.
#[derive(Default)]
struct Row {
    id: String,
    held: bool,
    /// (control, core): place inside its window, across all windows, in the forwarded list.
    in_window: (Option<usize>, Option<usize>),
    across: (Option<usize>, Option<usize>),
    /// Across windows by the calibrated score without the anchor subtracted (information).
    across_raw: Option<usize>,
    forwarded: (Option<usize>, Option<usize>),
    /// Seconds: scan (shared), control judge, core loop.
    scan_s: f64,
    control_s: f64,
    core_s: f64,
    rounds: u32,
    gate: usize,
    stop: String,
    no_info: usize,
    forwarded_n: (usize, usize),
    loop_reads: usize,
    enc_tokens: usize,
    memory: String,
}

fn place(v: &mut [(f32, bool)]) -> Option<usize> {
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    v.iter().position(|y| y.1)
}

/// The control arm's fields of a row: the scan's in-window and across-window places under the
/// served score (mass) and the raw calibrated score, and the control's forwarded list.
fn control_part(row: &mut Row, scan: &Scan, right: &dyn Fn(usize) -> bool, order: &[usize]) {
    row.held = scan.windows.iter().any(|w| w.place.iter().any(|c| right(*c)));
    row.enc_tokens = scan.enc_tokens;
    row.forwarded.0 = order.iter().position(|c| right(*c));
    row.forwarded_n.0 = order.len();
    let mut inside: Option<usize> = None;
    let mut all: Vec<(f32, bool)> = Vec::new();
    let mut raw: Vec<(f32, bool)> = Vec::new();
    for w in &scan.windows {
        let ranked = w.ranked(Score::Mass);
        if let Some(r) = ranked.iter().position(|(k, _)| w.place.get(*k).is_some_and(|c| right(*c))) {
            inside = Some(inside.map_or(r, |b| b.min(r)));
        }
        all.extend(ranked.iter().map(|(k, s)| (*s, w.place.get(*k).is_some_and(|c| right(*c)))));
        raw.extend(w.scores(Score::Calibrated).iter().enumerate().map(|(k, s)| (*s, w.place.get(k).is_some_and(|c| right(*c)))));
    }
    row.in_window.0 = inside;
    row.across.0 = place(&mut all);
    row.across_raw = place(&mut raw);
}

/// The core arm's fields of a row.
fn core_part(row: &mut Row, cp: &Pass, cr: &CoreRun, right: &dyn Fn(usize) -> bool) {
    row.forwarded.1 = cp.order.iter().position(|c| right(*c));
    row.forwarded_n.1 = cp.order.len();
    let mut inside: Option<usize> = None;
    let mut all: Vec<(f32, bool)> = Vec::new();
    for (_, list) in &cr.ranked {
        if let Some(r) = list.iter().position(|(c, _)| right(*c)) {
            inside = Some(inside.map_or(r, |b| b.min(r)));
        }
        all.extend(list.iter().map(|(c, s)| (*s, right(*c))));
    }
    row.in_window.1 = inside;
    row.across.1 = place(&mut all);
    row.rounds = cr.outcome.handover.rounds;
    row.gate = cr.outcome.handover.gate_direct;
    row.stop.clone_from(&cr.outcome.handover.stop);
    row.no_info = cr.no_info;
    row.loop_reads = cr.reads;
}

fn row_line(i: usize, row: &Row) -> String {
    format!(
        "q{} ({}): held {}; in window control {} core {}; across control {} core {} (raw {}); forwarded control {}/{} core {}/{}; seconds scan {:.1} control {:.1} core {:.1}; rounds {} stop {}; gate {}; no information {}; loop reads {}; encoded {} tokens; {}",
        i + 1,
        row.id,
        row.held,
        show(row.in_window.0),
        show(row.in_window.1),
        show(row.across.0),
        show(row.across.1),
        show(row.across_raw),
        show(row.forwarded.0),
        row.forwarded_n.0,
        show(row.forwarded.1),
        row.forwarded_n.1,
        row.scan_s,
        row.control_s,
        row.core_s,
        row.rounds,
        row.stop,
        row.gate,
        row.no_info,
        row.loop_reads,
        row.enc_tokens,
        row.memory
    )
}

/// The summary table over the rows (the same in every mode).
fn summary(out: &mut String, rows: &[Row]) {
    let k = rows.len();
    let first = |f: &dyn Fn(&Row) -> Option<usize>| rows.iter().filter(|r| f(r) == Some(0)).count();
    let any = |f: &dyn Fn(&Row) -> Option<usize>| rows.iter().filter(|r| f(r).is_some()).count();
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let mean = |f: &dyn Fn(&Row) -> f64| rows.iter().map(f).sum::<f64>() / k.max(1) as f64;
    let _ = writeln!(out, "\nsummary over {k} questions (right chunk in some window: {})", rows.iter().filter(|r| r.held).count());
    let _ = writeln!(out, "| measure | control (today's pass, mass) | tentacle core |");
    let _ = writeln!(out, "|---|---|---|");
    let _ = writeln!(out, "| right chunk first inside its window | {} | {} |", first(&|r| r.in_window.0), first(&|r| r.in_window.1));
    let _ = writeln!(
        out,
        "| right chunk first across all windows | {} | {} (raw calibrated, no anchor: {}) |",
        first(&|r| r.across.0),
        first(&|r| r.across.1),
        first(&|r| r.across_raw)
    );
    let _ = writeln!(out, "| right chunk forwarded (any place) | {} | {} |", any(&|r| r.forwarded.0), any(&|r| r.forwarded.1));
    let _ = writeln!(out, "| right chunk forwarded first | {} | {} |", first(&|r| r.forwarded.0), first(&|r| r.forwarded.1));
    let _ = writeln!(
        out,
        "| spans forwarded per question (mean) | {:.1} | {:.1} |",
        mean(&|r| r.forwarded_n.0 as f64),
        mean(&|r| r.forwarded_n.1 as f64)
    );
    let _ = writeln!(
        out,
        "| seconds per question: scan (shared) plus arm | {:.1} + {:.1} | {:.1} + {:.1} |",
        mean(&|r| r.scan_s),
        mean(&|r| r.control_s),
        mean(&|r| r.scan_s),
        mean(&|r| r.core_s)
    );
    let _ = writeln!(
        out,
        "| G1 rounds used (mean, max) | - | {:.2}, {} |",
        mean(&|r| f64::from(r.rounds)),
        rows.iter().map(|r| r.rounds).max().unwrap_or(0)
    );
    let _ = writeln!(out, "| isolation gate (sum, must be 0) | - | {} |", rows.iter().map(|r| r.gate).sum::<usize>());
    let _ = writeln!(out, "| first reports with no information (mean of 8) | - | {:.2} |", mean(&|r| r.no_info as f64));
}

/// The organism with the dense lane indexed (served layout), and its opening lines.
fn open(home: &Path, out: &mut String) -> Result<Organism, Fault> {
    let org = Organism::open_with(home, home, &["neuromorph-nano".to_owned()], false, &home.join("memory"))?;
    let _ = writeln!(out, "neural: {}", nmapi::neural::bind(&org)?);
    let _ = writeln!(out, "dense: {}", nmapi::neural::start_dense(&org)?);
    let sub = org.sub();
    let lane = sub.dense.read().ok().and_then(|g| g.clone());
    let missing = || lane.as_ref().map_or(0, |l| sub.chunks.iter().filter(|c| l.vector(&c.key).is_none()).count());
    let (first, t) = (missing(), Instant::now());
    while missing() > 0 && t.elapsed().as_secs() < DENSE_WAIT_S {
        #[expect(
            clippy::disallowed_methods,
            reason = "synchronous measurement polling a background indexing thread; no async runtime here"
        )]
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    let _ = writeln!(
        out,
        "dense lane: {first} chunks lacked a vector at start; waited {:.1} s; still missing {}",
        t.elapsed().as_secs_f64(),
        missing()
    );
    Ok(org)
}

/// The gold rule of the run: a chunk is right when its file is a key file of the question and
/// its lines overlap the key lines.
pub(crate) fn gold(sub: &nmsubs::Substrate, x: &Item) -> Vec<usize> {
    let path = |i: usize| sub.chunks.get(i).and_then(|c| sub.files.get(c.file)).map_or_else(String::new, |f| f.path.replace('\\', "/"));
    (0..sub.chunks.len())
        .filter(|&c| sub.chunks.get(c).is_some_and(|ch| x.keys.iter().any(|(f, a, b)| *f == path(c) && ch.start <= *b && ch.end >= *a)))
        .collect()
}

fn settings_line(control: &Settings) -> String {
    format!(
        "control: score {}, judge {}, shortlist {}, {} per tentacle; core: score calibrated above the anchor, shortlist derived per tentacle (gate 1), G1 loop, budget derived (forward = pack room of today's cut / mean chunk tokens; no time limit, the loop's ceiling bounds it)",
        control.score.name(),
        control.judge.name(),
        control.shortlist,
        control.per_tentacle
    )
}

/// The live run, and with `dump` the scan dump of every question.
fn test(home: &Path, n: usize, rec: &Path, dump: Option<&Path>, span: usize) -> Result<String, Fault> {
    let mut items: Vec<Item> = crate::e2::suite(home)?;
    items.truncate(n);
    let mut out = String::new();
    let started = Instant::now();
    match dump {
        None => {
            let _ = writeln!(
                out,
                "TC1 first measurement of the tentacle core (first form), development half by index, questions 1 to {} asked at most; test limit {LIMIT_S} s",
                items.len()
            );
        }
        Some(d) => {
            let _ = writeln!(
                out,
                "TC1 scan dump for processor-only replay, development half by index, questions 1 to {}; no test limit (end-of-day run); dump folder {}",
                items.len(),
                d.display()
            );
        }
    }
    let org = open(home, &mut out)?;
    let sub = org.sub();
    let core = Settings { core: true, span, ..Settings::served() };
    let control = Settings::served();
    let _ = writeln!(
        out,
        "{}; decision rules: {}; core read: {}",
        settings_line(&control),
        nmbrain::g1::rules().text(),
        if span > 0 {
            format!(
                "span read, spans of about {span} tokens, best {} per span, final short read (test T6)",
                nmapi::tentacle_pass::SPAN_KEEP
            )
        } else {
            "full-window read (first form)".to_owned()
        }
    );
    if let Some(d) = dump {
        std::fs::create_dir_all(d).map_err(|e| Fault::new("tc1", nmtype::FaultKind::Execution, format!("{}: {e}", d.display())))?;
        let p = scan_dump::write_substrate(d, &sub)?;
        let _ = writeln!(
            out,
            "substrate written: {} ({} files, {} chunks, {} symbols, version {})",
            p.display(),
            sub.files.len(),
            sub.chunks.len(),
            sub.symbols.len(),
            sub.version
        );
        let _ = std::fs::write(rec, &out);
    }
    let pass = TentaclePass::new(home, core);
    let sets = nmapi::tentacle_pass::drones(home);
    let mut rows: Vec<Row> = Vec::new();
    let mut notes = Vec::new();
    for (i, x) in items.iter().enumerate() {
        let spent = started.elapsed().as_secs_f64();
        #[expect(clippy::cast_precision_loss, reason = "a count")]
        let mean =
            if rows.is_empty() { 0.0 } else { rows.iter().map(|r| r.scan_s + r.control_s + r.core_s).sum::<f64>() / rows.len() as f64 };
        if dump.is_none() && !rows.is_empty() && spent + mean > LIMIT_S {
            let _ = writeln!(
                out,
                "stopped before question {} (index {i}): {spent:.0} s spent, mean {mean:.0} s per question, limit {LIMIT_S} s",
                i + 1
            );
            break;
        }
        let g = gold(&sub, x);
        let right = |c: usize| g.contains(&c);
        let mut row = Row { id: x.id.clone(), ..Row::default() };
        let t = Instant::now();
        let (scan, _) = pass.scan(&sub, &x.q, &core)?;
        row.scan_s = t.elapsed().as_secs_f64();
        // Control arm.
        let t = Instant::now();
        let p = pass.read_as(&sub, &org.slots, &x.q, &control)?;
        row.control_s = t.elapsed().as_secs_f64();
        control_part(&mut row, &scan, &right, &p.order);
        // Core arm (with a recording tape when dumping).
        let tape = dump.map(|_| Arc::new(Tape::recorder()));
        let t = Instant::now();
        let (cp, cr) = nmapi::tentacle_core::judge_taped(home, &sub, &scan, &core, tape.clone())?;
        row.core_s = t.elapsed().as_secs_f64();
        core_part(&mut row, &cp, &cr, &right);
        row.memory = memory_line();
        let _ = writeln!(out, "{}", row_line(i, &row));
        if let (Some(d), Some(tape)) = (dump, tape) {
            let q = QuestionDump {
                index: i,
                id: x.id.clone(),
                gold: g.clone(),
                scan_s: row.scan_s,
                control: ControlArm { order: p.order.clone(), seconds: row.control_s, note: p.note.clone() },
                core: CoreArm {
                    order: cp.order.clone(),
                    ranked: cr.ranked.clone(),
                    rounds: row.rounds,
                    gate: row.gate,
                    stop: row.stop.clone(),
                    no_info: row.no_info,
                    reads: row.loop_reads,
                    seconds: row.core_s,
                    note: cp.note.clone(),
                    handover: nmhash::sha256_bytes(cr.outcome.handover.to_value().encode().unwrap_or_default().as_bytes()),
                },
                tape: tape.entries(),
                settings: settings_line(&control),
                memory: row.memory.clone(),
            };
            let heads = |t: usize| sets.for_tentacle(nmsubs::lens::NAMES.get(t).copied().unwrap_or("lens")).to_vec();
            let bytes = scan_dump::write_question(d, &q, &scan, &heads)?;
            let _ = writeln!(
                out,
                "  dumped {} ({} bytes; {} windows, {} tape reads); elapsed {:.0} s",
                scan_dump::stem(i),
                bytes,
                scan.windows.len(),
                q.tape.len(),
                started.elapsed().as_secs_f64()
            );
        }
        notes.push(format!("q{}: {}", i + 1, cp.note));
        rows.push(row);
        // The record so far, after every question, so a fault keeps what was measured.
        let _ = std::fs::write(rec, &out);
    }
    summary(&mut out, &rows);
    let _ = writeln!(out, "total {:.0} s; {}", started.elapsed().as_secs_f64(), memory_line());
    let _ = writeln!(out, "\ncore notes:");
    for nl in notes {
        let _ = writeln!(out, "{nl}");
    }
    Ok(out)
}

/// The newest dump folder.
fn newest_dump(home: &Path) -> Option<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(home.join(DUMP_DIR))
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("substrate.json").is_file())
        .collect();
    v.sort();
    v.pop()
}

/// The replay over a dump: processor only.
fn replay(home: &Path, dir: &Path) -> Result<String, Fault> {
    let started = Instant::now();
    let mut out = String::new();
    let _ = writeln!(out, "TC1 replay of a scan dump on the processor (no model loaded, no device opened): {}", dir.display());
    let items: Vec<Item> = crate::e2::suite(home)?;
    let sub = scan_dump::load_substrate(dir)?;
    let core = Settings { core: true, ..Settings::served() };
    // 2026-11-10 (refinement step 1): the equality with the live core arm is checked under the
    // first-form rules the dump run used; the Globalized Context Stack rules are reported after it.
    nmbrain::g1::set_rules(nmbrain::g1::Rules::first_form());
    let _ = writeln!(
        out,
        "substrate: {} files, {} chunks, {} symbols, version {} (loaded in {:.2} s)",
        sub.files.len(),
        sub.chunks.len(),
        sub.symbols.len(),
        sub.version,
        started.elapsed().as_secs_f64()
    );
    let mut rows: Vec<Row> = Vec::new();
    let mut equal = 0;
    let mut checks = Vec::new();
    let (mut hits, mut misses) = (0, 0);
    let mut replay_s = 0.0;
    for idx in scan_dump::indices(dir) {
        let Some(x) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(dir, idx, &x.q)?;
        if q.id != x.id {
            return Err(Fault::new(
                "tc1",
                nmtype::FaultKind::Execution,
                format!("dump question {idx} is {}, the question file has {} there", q.id, x.id),
            ));
        }
        let right = |c: usize| q.gold.contains(&c);
        let mut row = Row { id: q.id.clone(), scan_s: q.scan_s, control_s: q.control.seconds, ..Row::default() };
        control_part(&mut row, &scan, &right, &q.control.order);
        let tape = Arc::new(Tape::replayer(q.tape.clone()));
        let t = Instant::now();
        let (cp, cr) = nmapi::tentacle_core::judge_taped(home, &sub, &scan, &core, Some(Arc::clone(&tape)))?;
        row.core_s = t.elapsed().as_secs_f64();
        replay_s += row.core_s;
        core_part(&mut row, &cp, &cr, &right);
        row.memory = ram_line();
        let (h, m) = tape.counts();
        hits += h;
        misses += m;
        // Equality with the live core arm of the dump run.
        let bits = |l: &[(String, Vec<(usize, f32)>)]| {
            l.iter().map(|(id, v)| (id.clone(), v.iter().map(|(c, s)| (*c, s.to_bits())).collect::<Vec<_>>())).collect::<Vec<_>>()
        };
        let hand = nmhash::sha256_bytes(cr.outcome.handover.to_value().encode().unwrap_or_default().as_bytes());
        let same = cp.order == q.core.order
            && bits(&cr.ranked) == bits(&q.core.ranked)
            && cr.outcome.handover.rounds == q.core.rounds
            && cr.outcome.handover.gate_direct == q.core.gate
            && cr.no_info == q.core.no_info
            && cr.reads == q.core.reads
            && hand == q.core.handover;
        equal += usize::from(same);
        checks.push(format!(
            "q{} ({}): replay equals the live core arm: {same} (hand-over {}, order {}, rankings {}, rounds {} and {}, reads {} and {}); tape hits {h}, misses {m}; replay {:.3} s against live {:.1} s",
            idx + 1,
            q.id,
            if hand == q.core.handover { "same" } else { "differs" },
            if cp.order == q.core.order { "same" } else { "differs" },
            if bits(&cr.ranked) == bits(&q.core.ranked) { "same" } else { "differ" },
            cr.outcome.handover.rounds,
            q.core.rounds,
            cr.reads,
            q.core.reads,
            row.core_s,
            q.core.seconds
        ));
        let _ = writeln!(out, "{}", row_line(idx, &row));
        rows.push(row);
    }
    summary(&mut out, &rows);
    let _ = writeln!(out, "(scan and control seconds are the dump run's live figures; core seconds are the replay's)");
    let _ = writeln!(
        out,
        "\nreplay against the live core arm of the dump run: {equal} of {} equal; device reads answered from the tape {hits}, not in the tape {misses}; replay of the cores and G1 {replay_s:.2} s in all; total {:.1} s; {}",
        rows.len(),
        started.elapsed().as_secs_f64(),
        ram_line()
    );
    for c in checks {
        let _ = writeln!(out, "{c}");
    }
    // Refinement step 1: the same dump under the Globalized Context Stack rules (evidence combination, queue rule).
    let stack = nmbrain::g1::Rules::stack();
    nmbrain::g1::set_rules(stack);
    let (mut fwd_any, mut fwd_first, mut held_any, mut fwd_tok, mut held_tok, mut queued, mut nq) = (0, 0, 0, 0, 0, 0, 0);
    let t = Instant::now();
    for idx in scan_dump::indices(dir) {
        let Some(x) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(dir, idx, &x.q)?;
        let tape = Arc::new(Tape::replayer(q.tape.clone()));
        let (cp, cr) = nmapi::tentacle_core::judge_taped(home, &sub, &scan, &core, Some(tape))?;
        let set = &cr.outcome.evidence;
        let gold_key = |k: &str| q.gold.iter().any(|g| sub.chunks.get(*g).is_some_and(|c| c.key == k));
        fwd_any += usize::from(cp.order.iter().any(|c| q.gold.contains(c)));
        fwd_first += usize::from(cp.order.first().is_some_and(|c| q.gold.contains(c)));
        held_any += usize::from(set.all().any(|e| gold_key(&e.span.key)));
        let (ft, qt) = set.tokens();
        fwd_tok += ft;
        held_tok += ft + qt;
        queued += set.queue.len();
        nq += 1;
    }
    nmbrain::g1::set_rules(nmbrain::g1::Rules::first_form());
    let per = |x: usize| x.checked_div(nq.max(1)).unwrap_or(0);
    let _ = writeln!(out, "
stack rules ({}): right chunk forwarded in round 1 on {fwd_any} of {nq}, first on {fwd_first}; held by G1 (forwarded or queued) on {held_any}; per question {} tokens forwarded, {} tokens held, {} spans queued; {:.2} s", stack.text(), per(fwd_tok), per(held_tok), per(queued), t.elapsed().as_secs_f64());
    Ok(out)
}

/// `run tc1 [n]`, `run tc1 dump [n]`, `run tc1 replay [folder]`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let stamp = nmtype::now_ms();
    let mode = args.get(2).map(String::as_str);
    // 2026-11-10: `--rules chosen` sets the decision rules chosen by replay
    // (`runs/2026-11-10-neumor-note-decision-path-refinement-1.md`: first-form confidence, largest-gap
    // shortlist, G1 score 0.75 confidence plus 0.25 whole-code-base support) for a live
    // confirmation; the first form otherwise.
    if args.windows(2).any(|w| w.first().map(String::as_str) == Some("--rules") && w.get(1).map(String::as_str) == Some("chosen")) {
        use nmbrain::g1::{Combine, ConfidenceRule, Rules, ShortlistRule, Weights};
        nmbrain::g1::set_rules(Rules {
            confidence: ConfidenceRule::Logistic,
            shortlist: ShortlistRule::LargestGap,
            combine: Combine::Linear(Weights { confidence: 0.75, support: 0.25, agreement: 0.0, rank: 0.0 }),
            forward: nmbrain::g1::ForwardRule::Fixed,
        });
    }
    // 2026-11-10: `--span <tokens>`: the core reads each window as spans (test T6).
    let span = args
        .windows(2)
        .find(|w| w.first().map(String::as_str) == Some("--span"))
        .and_then(|w| w.get(1))
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(0);
    let (path, result) = match mode {
        Some("replay") => {
            let dir = args.get(3).map(PathBuf::from).map(|p| if p.is_absolute() { p } else { home.join(p) }).or_else(|| newest_dump(home));
            let path = home.join("runs").join(format!("TC1-replay-{stamp}.txt"));
            let r = dir.map_or_else(
                || Err(Fault::new("tc1", nmtype::FaultKind::Execution, format!("no dump under {DUMP_DIR}"))),
                |d| replay(home, &d),
            );
            (path, r)
        }
        Some("dump") => {
            let n = args.get(3).and_then(|x| x.parse().ok()).filter(|n: &usize| *n >= 1).unwrap_or(usize::MAX);
            let path = home.join("runs").join(format!("TC1-dump-{stamp}.txt"));
            let dir = home.join(DUMP_DIR).join(stamp.to_string());
            let r = test(home, n, &path, Some(&dir), span);
            (path, r)
        }
        _ => {
            let n = args.get(2).and_then(|x| x.parse().ok()).filter(|n: &usize| *n >= 1).unwrap_or(QUESTIONS);
            let path = home.join("runs").join(format!("TC1-dev-{stamp}.txt"));
            let r = test(home, n, &path, None, span);
            (path, r)
        }
    };
    let (text, ok) = match result {
        Ok(t) => (t, true),
        Err(e) => {
            // Keep the rows written so far and add the fault.
            let before = std::fs::read_to_string(&path).unwrap_or_default();
            (format!("{before}FAULT {}: {}\n", e.component, e.message), false)
        }
    };
    if let Err(e) = std::fs::write(&path, &text) {
        eprintln!("tc1: {}: {e}", path.display());
        return ExitCode::FAILURE;
    }
    println!("{text}\nreport: {}", path.display());
    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
