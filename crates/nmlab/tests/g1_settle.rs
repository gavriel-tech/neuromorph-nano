//! G1-T (tentacle master plan section 8, step G1; lane G, 2026-11-18): G1 forwarding by the plain
//! sum of the drones' settle chances (`nmapi::drone_path::settle_forward`, `NANO_G1_SETTLE`)
//! against G1's current order, on the Mini's TC1 dump (development half, 30 questions), processor
//! only. The served stack (team windows, pooled G1 on the common scale, G2 by cover, settle
//! chance) runs on the dump's substrate with the release tokenizer and table; no model runs and
//! the meaning lane is absent (the dump holds no stored vectors). Nothing is fitted, so every
//! figure is out of fold. Measures: forwarded first (G1's hand-over led by an answer piece),
//! presence (an answer piece held through G1, hand-over or queue, of the questions whose answer
//! is in some window), first in what went up, and the plain settle-sum order over the whole tree
//! as the baseline. Questions are named by index and id; no question text is written.
//!
//! Run: `cargo test --release -p nmlab --test g1_settle -- --ignored --nocapture`
//! (`NM_G1T_N=2` for the smoke run). Writes `scratch/lane-g/g1t-<ms>.txt`.
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

#[derive(Clone, Copy, Debug, Default)]
struct Out {
    in_window: bool,
    fwd_first: bool,
    present: bool,
    up_first: bool,
    sum_first: bool,
    place: usize,
}

fn one(sub: &nmsubs::Substrate, keep: &Keep, question: &str, gold: &[usize]) -> R<Out> {
    let run = drone_path::stack_run_arm(sub, keep, question, None, Windows::Team, None).map_err(|e| e.message)?;
    let g = |c: &usize| gold.contains(c);
    // What G1 holds: its hand-over then its queue, as G2 last read them (`StackRun::cands`;
    // `StackRun::queue` is the queue after G2's routing, not G1's).
    let held: Vec<usize> = if run.cands.is_empty() { run.handover.iter().chain(&run.queue).copied().collect() } else { run.cands.clone() };
    let sum_first = run.evidence.as_ref().is_some_and(|ev| {
        let mut idx: Vec<usize> = (0..sub.chunks.len()).collect();
        idx.sort_by(|a, b| ev.settle_sum(*b).total_cmp(&ev.settle_sum(*a)).then(a.cmp(b)));
        idx.first().is_some_and(g)
    });
    Ok(Out {
        in_window: run.taken.iter().flatten().flatten().any(g),
        fwd_first: run.handover.first().is_some_and(g),
        present: held.iter().any(g),
        up_first: run.up.first().is_some_and(g),
        sum_first,
        place: held.iter().position(g).map_or(0, |p| p + 1),
    })
}

#[test]
#[ignore = "G1-T replay: needs the Mini's TC1 dump and the release tokenizer and table; run by hand"]
fn g1_t_settle() {
    if let Err(e) = run() {
        panic!("g1-t: {e}");
    }
}

fn run() -> R<()> {
    let home = nmtype::nano_root();
    let mini = home.join("..").join("neuromorph-mini");
    let dir = mini.join("scratch").join("tc1-dump").join(DUMP);
    let n_want: usize = std::env::var("NM_G1T_N").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let started = Instant::now();
    let items = dev_half(&mini)?;
    let sub = scan_dump::load_substrate(&dir).map_err(|e| e.message)?;
    let mut keep = Keep::open(&nmtype::models_dir(&home).join("qwen3-0.6b/files")).map_err(|e| e.message)?;
    keep.sync(&sub).map_err(|e| e.message)?;
    // The served stack's switches (`--windows team --g1 pooled --g2 cover --r settle`).
    drone_path::set_pooled(Some(true));
    drone_path::set_g2(true);
    drone_path::set_settle(true);
    let mut rep = String::new();
    let _ = writeln!(rep, "G1-T, lane G: dump {DUMP}, {} pieces, processor only, setup {:.1} s", sub.chunks.len(), started.elapsed().as_secs_f64());
    let _ = writeln!(rep, "q | id | in window | off: fwd first, present, place, up first | on: fwd first, present, place, up first | settle sum first");
    let mut tot = [[0_usize; 4]; 2];
    let (mut n, mut win, mut sum1) = (0_usize, 0_usize, 0_usize);
    let mut pairs = (0_usize, 0_usize);
    for idx in scan_dump::indices(&dir) {
        if n >= n_want {
            break;
        }
        let Some((id, text)) = items.get(idx) else { continue };
        let (q, _) = scan_dump::load_question(&dir, idx, text).map_err(|e| e.message)?;
        drone_path::set_g1_settle(Some(false));
        let off = one(&sub, &keep, text, &q.gold)?;
        drone_path::set_g1_settle(Some(true));
        let on = one(&sub, &keep, text, &q.gold)?;
        drone_path::set_g1_settle(None);
        n += 1;
        win += usize::from(off.in_window);
        sum1 += usize::from(off.sum_first);
        for (t, o) in tot.iter_mut().zip([off, on]) {
            for (x, b) in t.iter_mut().zip([o.fwd_first, o.present && o.in_window, o.up_first, o.present]) {
                *x += usize::from(b);
            }
        }
        pairs.0 += usize::from(on.fwd_first && !off.fwd_first);
        pairs.1 += usize::from(off.fwd_first && !on.fwd_first);
        let _ = writeln!(
            rep,
            "{idx} | {id} | {} | {} {} {} {} | {} {} {} {} | {}",
            u8::from(off.in_window),
            u8::from(off.fwd_first),
            u8::from(off.present),
            off.place,
            u8::from(off.up_first),
            u8::from(on.fwd_first),
            u8::from(on.present),
            on.place,
            u8::from(on.up_first),
            u8::from(off.sum_first)
        );
    }
    let [a, b] = tot;
    let _ = writeln!(rep, "questions {n}; answer in a window {win} of {n}");
    let _ = writeln!(rep, "G1 today (off): forwarded first {} of {n}; presence {} of {win}; up first {} of {n}", a[0], a[1], a[2]);
    let _ = writeln!(rep, "G1 plain settle sum (on): forwarded first {} of {n}; presence {} of {win}; up first {} of {n}", b[0], b[1], b[2]);
    let _ = writeln!(rep, "on against off, forwarded first: +{} -{}", pairs.0, pairs.1);
    let _ = writeln!(rep, "baseline, plain settle sum over the whole tree: first {sum1} of {n}");
    let _ = writeln!(rep, "rule: forwarded first 18 or more of 30, presence 29 of 29, at least the plain sum baseline; total {:.1} s", started.elapsed().as_secs_f64());
    let dir_out = home.join("scratch").join("lane-g");
    let _ = std::fs::create_dir_all(&dir_out);
    let path = dir_out.join(format!("g1t-{}.txt", nmtype::now_ms()));
    std::fs::write(&path, &rep).map_err(|e| e.to_string())?;
    eprintln!("{rep}\nrecord {}", path.display());
    Ok(())
}
