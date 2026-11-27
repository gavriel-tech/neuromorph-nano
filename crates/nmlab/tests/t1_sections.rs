//! Lane T1 section tests (tentacle master plan section 8; 2026-11-19), processor only, on the
//! Mini's TC1 dump (development half, read as `g1_settle.rs` reads it; no question text written):
//!
//! - `g0_r` (G0-R): first in window by the dump's calibrated read, raw against residualised on the
//!   window's nuisance basis (`nmbrain::g0::residualise`: place, tokens to the window end, log
//!   length, newline share); rule: up by 3 or more from 10 of 29; and the last-position penalty
//!   (the mean read of each window's last piece against the others, before and after).
//! - `team` (REACH-1, MAP-1, MAP-2, G1-T, G2-C, ISO): the served stack's team on the dump
//!   (`stack_run_arm`, Team; the arm's switches set by the caller's environment): answer in a
//!   round-1 window and in any window, tokens covered, rounds, overlap, unconserved and solo
//!   counts (each must be 0), G1's hand-over first, presence through G1, first sent up.
//!
//! Run: `cargo test --release -p nmlab --test t1_sections -- --ignored --nocapture <name>`;
//! `T1_LABEL` names the record `scratch/lane-t1/sections-<name>-<label>-<ms>.txt`.
use nmapi::drone_path::{self, Windows};
use nmapi::drone_windows::Keep;
use nmapi::scan_dump;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

type R<T> = Result<T, String>;
const DUMP: &str = "1794288995065";

/// The development half: (id, question text) by index, as the locked harness orders it.
fn dev_half(home: &Path) -> R<Vec<(String, String)>> {
    const TYPES: [&str; 5] = ["behaviour", "name", "value", "explain", "relation"];
    let mut out = Vec::new();
    for area in ["context", "engine", "tools"] {
        let p = home.join(format!("tests/capability-gates/cx1-context/questions-{area}.json"));
        let v = nmjson::parse(&std::fs::read(&p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let mut qs = Vec::new();
        for x in v.get("questions").and_then(|y| y.array()).map_err(|e| e.to_string())? {
            let s = |k: &str| x.get(k).and_then(|y| y.string()).map(str::to_owned).map_err(|e| e.to_string());
            let t = s("type")?;
            qs.push((TYPES.iter().position(|y| *y == t).unwrap_or(usize::MAX), s("id")?, s("q")?));
        }
        qs.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        out.extend(qs.into_iter().enumerate().filter(|(pos, _)| pos % 2 == 0).map(|(_, (_, id, q))| (id, q)));
    }
    Ok(out)
}

fn write(name: &str, rep: &str) -> R<()> {
    let home = nmtype::nano_root();
    let label = std::env::var("T1_LABEL").unwrap_or_else(|_| "run".into());
    let dir = home.join("scratch").join("lane-t1");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("sections-{name}-{label}-{}.txt", nmtype::now_ms()));
    std::fs::write(&path, rep).map_err(|e| e.to_string())?;
    eprintln!("{rep}\nrecord {}", path.display());
    Ok(())
}

#[test]
#[ignore = "G0-R on the Mini's TC1 dump; run by hand"]
fn g0_r() {
    if let Err(e) = run_g0_r() {
        panic!("g0-r: {e}");
    }
}

fn run_g0_r() -> R<()> {
    let home = nmtype::nano_root();
    let mini = home.join("..").join("neuromorph-mini");
    let dir = mini.join("scratch").join("tc1-dump").join(DUMP);
    let items = dev_half(&mini)?;
    let sub = scan_dump::load_substrate(&dir).map_err(|e| e.message)?;
    let mut rep = String::new();
    let _ = writeln!(rep, "G0-R, lane T1: dump {DUMP}; per question, is an answer piece first in the window holding it (best such window)");
    let _ = writeln!(rep, "q | id | windows with answer | raw first | residual first");
    let (mut n, mut raw1, mut res1) = (0_usize, 0_usize, 0_usize);
    let (mut last_raw, mut last_res, mut other_raw, mut other_res, mut nl, mut no) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0_usize, 0_usize);
    for idx in scan_dump::indices(&dir) {
        let Some((id, text)) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(&dir, idx, text).map_err(|e| e.message)?;
        let mut holding = 0;
        let (mut a, mut b) = (false, false);
        for w in &scan.windows {
            let cal = w.calibrated_parts();
            if cal.len() != w.parts.len() || cal.len() < 2 {
                continue;
            }
            let raw: Vec<f32> = cal.iter().skip(1).copied().collect();
            let toks: Vec<usize> = w.parts.iter().skip(1).map(|p| p.1).collect();
            let nls: Vec<f32> = w
                .place
                .iter()
                .map(|c| {
                    let t = sub.chunks.get(*c).map_or("", |ch| ch.text.as_str());
                    #[expect(clippy::cast_precision_loss, reason = "character counts")]
                    let s = t.matches('\n').count() as f32 / t.len().max(1) as f32;
                    s
                })
                .collect();
            let mut basis = if std::env::var("T1_G0R").is_ok_and(|v| v == "served") { nmbrain::g0::Nuisance::of_window(&toks, &nls) } else { nmbrain::g0::Nuisance::of_window_full(&toks, &nls) };
            // Variants (`T1_G0R`): `nolen` leaves log length out (answers may be longer), `place`
            // keeps the place and tokens to the end only.
            match std::env::var("T1_G0R").as_deref() {
                Ok("nolen") => basis.iter_mut().for_each(|b| b.log_len = 0.0),
                Ok("place") => basis.iter_mut().for_each(|b| {
                    b.log_len = 0.0;
                    b.newlines = 0.0;
                }),
                Ok("last") => {
                    let k = basis.len().saturating_sub(1);
                    for (i, b) in basis.iter_mut().enumerate() {
                        b.place = if i == k { 1.0 } else { 0.0 };
                        b.to_end = 0.0;
                        b.log_len = 0.0;
                        b.newlines = 0.0;
                    }
                }
                Ok("lastlen") => {
                    let k = basis.len().saturating_sub(1);
                    for (i, b) in basis.iter_mut().enumerate() {
                        b.place = if i == k { 1.0 } else { 0.0 };
                        b.to_end = 0.0;
                        b.newlines = 0.0;
                    }
                }
                _ => {}
            }
            let res = nmbrain::g0::residualise(&raw, &basis);
            let first = |v: &[f32]| (0..v.len()).max_by(|x, y| v[*x].total_cmp(&v[*y]).then(y.cmp(x)));
            // The last-position penalty: the last piece's read against the mean of the others.
            if raw.len() >= 2 {
                let k = raw.len() - 1;
                let mean = |v: &[f32]| f64::from(v.iter().take(k).sum::<f32>()) / k as f64;
                last_raw += f64::from(raw[k]);
                last_res += f64::from(res[k]);
                other_raw += mean(&raw);
                other_res += mean(&res);
                nl += 1;
                no += 1;
            }
            if !w.place.iter().any(|c| q.gold.contains(c)) {
                continue;
            }
            holding += 1;
            a |= first(&raw).and_then(|i| w.place.get(i)).is_some_and(|c| q.gold.contains(c));
            b |= first(&res).and_then(|i| w.place.get(i)).is_some_and(|c| q.gold.contains(c));
        }
        if holding == 0 {
            continue;
        }
        n += 1;
        raw1 += usize::from(a);
        res1 += usize::from(b);
        let _ = writeln!(rep, "{idx} | {id} | {holding} | {} | {}", u8::from(a), u8::from(b));
    }
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let (l0, l1) = ((last_raw - other_raw) / nl.max(1) as f64, (last_res - other_res) / no.max(1) as f64);
    let _ = writeln!(rep, "questions with an answer in a window {n}; first in window: raw {raw1}, residualised {res1}");
    let _ = writeln!(rep, "last-position penalty (last piece's read less the others' mean, over windows): raw {l0:.3}, residualised {l1:.3}");
    let _ = writeln!(rep, "rule (G0-R): residualised at least raw + 3 (from 10 of 29); penalty gone");
    write("g0r", &rep)
}

#[test]
#[ignore = "team sections on the Mini's TC1 dump; run by hand"]
fn team() {
    if let Err(e) = run_team() {
        panic!("team: {e}");
    }
}

fn run_team() -> R<()> {
    let home = nmtype::nano_root();
    let mini = home.join("..").join("neuromorph-mini");
    let dir = mini.join("scratch").join("tc1-dump").join(DUMP);
    let n_want: usize = std::env::var("T1_N").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let started = Instant::now();
    let items = dev_half(&mini)?;
    let sub = scan_dump::load_substrate(&dir).map_err(|e| e.message)?;
    let mut keep = Keep::open(&nmtype::models_dir(&home).join("qwen3-0.6b/files")).map_err(|e| e.message)?;
    keep.sync(&sub).map_err(|e| e.message)?;
    drone_path::set_pooled(Some(true));
    drone_path::set_g2(true);
    drone_path::set_settle(true);
    // On the card, the reader and the judge load in the background: wait until they settle (at
    // most 120 s) so every question is read as served.
    let tw = Instant::now();
    while tw.elapsed().as_secs() < 120
        && (drone_path::warm_reader() == drone_path::JudgeState::Loading || drone_path::warm_close_judge() == drone_path::JudgeState::Loading)
    {
        std::thread::park_timeout(std::time::Duration::from_millis(200));
    }
    let tk = |c: usize| keep.tokens(c).max(1);
    let mut rep = String::new();
    let _ = writeln!(rep, "Team sections, lane T1: dump {DUMP}, {} pieces, {} tokens, arm {}, processor only", sub.chunks.len(), (0..sub.chunks.len()).map(|c| keep.tokens(c)).sum::<usize>(), std::env::var("T1_ARM").unwrap_or_else(|_| "team".into()));
    let _ = writeln!(rep, "q | id | round-1 window | any window | rounds | tokens r1 | tokens all | overlap unconserved solo | g1 first | present | up first | distinct labels r1");
    let mut t = [0_usize; 9];
    let (mut tok1, mut tok_all) = (0_usize, 0_usize);
    let mut n = 0;
    for idx in scan_dump::indices(&dir) {
        if n >= n_want {
            break;
        }
        let Some((id, text)) = items.get(idx) else { continue };
        if std::env::var("T1_ONLY").is_ok_and(|v| !v.split(',').any(|x| x == id)) {
            continue;
        }
        let (q, _) = scan_dump::load_question(&dir, idx, text).map_err(|e| e.message)?;
        let arm = match std::env::var("T1_ARM").as_deref() {
            Ok("firstfit") => Windows::FirstFit,
            Ok("single") => Windows::Single,
            _ => Windows::Team,
        };
        let run = drone_path::stack_run_arm(&sub, &keep, text, None, arm, None).map_err(|e| e.message)?;
        let g = |c: &usize| q.gold.contains(c);
        let r1 = run.taken.first().is_some_and(|w| w.iter().flatten().any(g));
        let any = run.taken.iter().flatten().flatten().any(g);
        let held: Vec<usize> = if run.cands.is_empty() { run.handover.iter().chain(&run.queue).copied().collect() } else { run.cands.clone() };
        let t1: usize = run.taken.first().map_or(0, |w| w.iter().flatten().map(|c| tk(*c)).sum());
        let ta: usize = run.taken.iter().flatten().flatten().map(|c| tk(*c)).sum();
        let iso = run.overlap + run.unconserved + run.solo;
        let g1f = run.handover.first().is_some_and(g);
        let present = held.iter().any(g);
        let up1 = run.up.first().is_some_and(g);
        if std::env::var("T1_IDS").is_ok_and(|v| v.split(',').any(|x| x == id)) {
            let f = |v: &[usize]| v.iter().take(5).map(|c| format!("{c}{}", if g(c) { "*" } else { "" })).collect::<Vec<_>>().join(" ");
            let gref: Vec<String> = q.gold.iter().filter_map(|c| sub.chunks.get(*c).and_then(|ch| sub.files.get(ch.file).map(|fl| format!("{}:{}-{}", fl.path.replace('\\', "/"), ch.start, ch.end)))).collect();
            let upl = run.up.iter().position(g).map_or(0, |p| p + 1);
            let _ = writeln!(rep, "  {id}: gold {:?} {gref:?}; handover {}; up {} (gold at {upl}); rounds {}; labels {:?}", q.gold, f(&run.handover), f(&run.up), run.taken.len(), run.labels);
        }
        n += 1;
        for (x, b) in t.iter_mut().zip([r1, any, g1f, present, up1, iso == 0]) {
            *x += usize::from(b);
        }
        t[6] += run.taken.len();
        t[7] += run.overlap;
        t[8] += run.unconserved + run.solo;
        tok1 += t1;
        tok_all += ta;
        let _ = writeln!(
            rep,
            "{idx} | {id} | {} | {} | {} | {t1} | {ta} | {} {} {} | {} | {} | {}",
            u8::from(r1),
            u8::from(any),
            run.taken.len(),
            run.overlap,
            run.unconserved,
            run.solo,
            u8::from(g1f),
            u8::from(present),
            u8::from(up1)
        );
    }
    let _ = writeln!(rep, "questions {n}");
    let _ = writeln!(rep, "REACH-1 / MAP-1: answer in a round-1 window {} of {n}; in any window {} of {n}", t[0], t[1]);
    let _ = writeln!(rep, "MAP-2: rounds {} in all; tokens covered round 1 {tok1}, all rounds {tok_all} (ratio {:.2})", t[6], tok_all as f64 / tok1.max(1) as f64);
    let _ = writeln!(rep, "ISO: overlap {} (pieces in two windows of a round), unconserved and solo {}; questions clean {} of {n}", t[7], t[8], t[5]);
    let _ = writeln!(rep, "G1-T: G1's hand-over first {} of {n}; presence {} of {}", t[2], t[3], t[1]);
    let _ = writeln!(rep, "G2-C: first sent up {} of {n}", t[4]);
    let _ = writeln!(rep, "rules: REACH-1 27 of 30 in a window, overlap 0; MAP-1 reach 29; MAP-2 tokens nearly double after two rounds; G1-T forwarded first 18 of 30, presence all; total {:.1} s", started.elapsed().as_secs_f64());
    write("team", &rep)
}

#[test]
#[ignore = "deep check forms on the Mini's TC1 dump; run by hand"]
fn deep_forms() {
    if let Err(e) = run_deep_forms() {
        panic!("deep forms: {e}");
    }
}

/// Per question of the dump: the answer piece and the value's best wrong piece, each read by the
/// packed deep check in the shared-span form and in one span a form: how often each verdict is
/// above 0, and how often the answer's verdict beats the wrong piece's.
fn run_deep_forms() -> R<()> {
    let home = nmtype::nano_root();
    let mini = home.join("..").join("neuromorph-mini");
    let dir = mini.join("scratch").join("tc1-dump").join(DUMP);
    let n_want: usize = std::env::var("T1_N").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let items = dev_half(&mini)?;
    let sub = scan_dump::load_substrate(&dir).map_err(|e| e.message)?;
    let mut keep = Keep::open(&nmtype::models_dir(&home).join("qwen3-0.6b/files")).map_err(|e| e.message)?;
    keep.sync(&sub).map_err(|e| e.message)?;
    let model = nmapi::neural::model(&home).map_err(|e| e.message)?;
    let mut rep = String::new();
    let mut t = [[0_usize; 3]; 2];
    let mut n = 0;
    for idx in scan_dump::indices(&dir) {
        if n >= n_want {
            break;
        }
        let Some((id, text)) = items.get(idx) else { continue };
        let (q, _) = scan_dump::load_question(&dir, idx, text).map_err(|e| e.message)?;
        let Some(&gold) = q.gold.first() else { continue };
        let ev = keep.evidence_with(text, None).map_err(|e| e.message)?;
        let mut order: Vec<usize> = (0..sub.chunks.len()).collect();
        order.sort_by(|a, b| ev.settle_sum(*b).total_cmp(&ev.settle_sum(*a)).then(a.cmp(b)));
        let Some(&wrong) = order.iter().find(|c| !q.gold.contains(c)) else { continue };
        let reqs = vec![nmapi::drone_path::pieces(&sub, &[gold], &[]), nmapi::drone_path::pieces(&sub, &[wrong], &[])];
        let mut line = format!("{idx} | {id}");
        for (k, shared) in [true, false].into_iter().enumerate() {
            let v = nmapi::tentacle_core::deep_check_packed_with(&model, text, &reqs, shared);
            let g = v.first().and_then(|x| x.first()).copied().unwrap_or(f32::NAN);
            let w = v.get(1).and_then(|x| x.first()).copied().unwrap_or(f32::NAN);
            t[k][0] += usize::from(g > 0.0);
            t[k][1] += usize::from(w > 0.0);
            t[k][2] += usize::from(g > w);
            let _ = write!(line, " | {} gold {g:.2} wrong {w:.2}", if shared { "shared" } else { "per form" });
        }
        let _ = writeln!(rep, "{line}");
        n += 1;
    }
    for (k, name) in ["shared span", "one span a form"].iter().enumerate() {
        let _ = writeln!(rep, "{name}: answer above 0 {} of {n}, wrong above 0 {} of {n}, answer above wrong {} of {n}", t[k][0], t[k][1], t[k][2]);
    }
    write("deepforms", &rep)
}
