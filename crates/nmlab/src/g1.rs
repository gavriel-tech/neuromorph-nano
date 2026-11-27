//! `nano-lab run g1 <stub|dev>` (2026-11-10): the tests of GCX build step 1, the bridge and
//! the window axis, and of the reading statistic (design
//! `docs/2026-11-10-neumor-design-g-layers.md`, Luis Jake Gabriel III; research note
//! `docs/2026-11-10-neumor-research-reading-statistic.md`; runbook
//! `runs/2026-11-10-G1-runbook.md`).
//!
//! One scan per question (eight lens windows, plain pose, each read with and without the
//! question) serves every arm: the four chunk scores read by the heads alone, and the
//! reranker choosing inside each tentacle from drone shortlists of 2, 4 and 6 under each
//! score. T1: the bridge keeps the judged order, and the heads arm under the E3.3 score holds
//! the E3.3 record. T2: every question through the live organism with the pass off, on with
//! the E3.3 settings, and on with the derived settings: the map's gates, one window row per
//! tentacle, the right code's place in the pack.
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use nmapi::forage::{Answer, Organism};
use nmapi::tentacle_pass::{Judge, Scan, Score, Settings, TentaclePass};
use nmjson::Value;
use nmtype::{Fault, FaultKind};

use crate::e2::Item;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("g1", FaultKind::Execution, m)
}

/// Drone shortlists the reranker chooses from, per tentacle.
const SHORTLISTS: [usize; 3] = [2, 4, 6];
/// The settings the research note derives, served in the third organism phase.
const DERIVED: (Score, usize) = (Score::Evidence, 6);
/// Chunks each tentacle forwards (as E3.3).
const PER: usize = 2;

/// `"{path}:{start} lines {a}..{b}"` as (path, a, b).
fn span(evidence: &str) -> Option<(String, usize, usize)> {
    let (src, lines) = evidence.split_once(" lines ")?;
    let (path, _) = src.rsplit_once(':')?;
    let (a, b) = lines.split_once("..")?;
    Some((path.replace('\\', "/"), a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// Place of the first pack packet whose evidence covers a key line of the question.
fn pack_place(x: &Item, a: &Answer) -> Option<usize> {
    a.pack.packets.iter().position(|p| {
        p.get_evidence().iter().filter_map(|e| span(e)).any(|(f, s, e)| x.keys.iter().any(|(kf, ka, kb)| *kf == f && s <= *kb && e >= *ka))
    })
}

fn gate(a: &Answer, k: &str) -> u64 {
    a.context_map.get("gates").and_then(|g| g.get(k)).and_then(Value::integer).unwrap_or(u64::MAX)
}

fn gates(a: &Answer) -> Vec<u64> {
    ["overlap", "unassigned", "split", "crossover"].iter().map(|k| gate(a, k)).collect()
}

fn windows(a: &Answer) -> usize {
    a.context_map.get("windows").and_then(Value::array).map_or(0, <[Value]>::len)
}

fn show(k: Option<usize>) -> String {
    k.map_or_else(|| "null".to_owned(), |x| x.to_string())
}

fn list(v: &[Option<usize>]) -> String {
    format!("[{}]", v.iter().map(|k| show(*k)).collect::<Vec<_>>().join(","))
}

/// (top, first three, mean reciprocal rank) of zero-based places.
fn tally(v: &[Option<usize>]) -> (usize, usize, f64) {
    let (mut top, mut three, mut rr) = (0, 0, 0.0);
    for k in v.iter().flatten() {
        top += usize::from(*k == 0);
        three += usize::from(*k < 3);
        #[expect(clippy::cast_precision_loss, reason = "places are small")]
        let inv = 1.0 / (*k as f64 + 1.0);
        rr += inv;
    }
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let mrr = rr / v.len().max(1) as f64;
    (top, three, mrr)
}

/// `a` is strictly better than `b`: top, then first three, then mean reciprocal rank.
fn better(a: (usize, usize, f64), b: (usize, usize, f64)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && (a.1 > b.1 || (a.1 == b.1 && a.2 > b.2 + 1e-9)))
}

/// One question of the E3.3 record: id, place under the heads on plain poses, place under the
/// reranker (which E3.3 ran on focused poses).
type Theirs = (String, Option<usize>, Option<usize>);

/// The newest E3.3 development record.
fn record(home: &Path) -> Option<(String, Vec<Theirs>)> {
    let mut names: Vec<String> = std::fs::read_dir(home.join("runs"))
        .ok()?
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with("E3-judge-dev-") && n.ends_with(".json"))
        .collect();
    names.sort();
    let name = names.pop()?;
    let v = nmjson::parse(&std::fs::read(home.join("runs").join(&name)).ok()?).ok()?;
    let rank = |q: &Value, k: &str| q.get(k).and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok());
    let rows = v
        .get("questions")
        .and_then(Value::array)
        .ok()?
        .iter()
        .filter_map(|q| Some((q.get("id").and_then(Value::string).ok()?.to_owned(), rank(q, "heads_plain"), rank(q, "reranker"))))
        .collect();
    Some((name, rows))
}

/// What one question produced.
#[derive(Default)]
struct Row {
    id: String,
    /// Place of the right code in the pack: pass off, on (E3.3 settings), on (derived settings).
    pack: Vec<Option<usize>>,
    /// The four gates in the same three phases.
    gates: Vec<Vec<u64>>,
    /// Window rows in the map, pass on (E3.3 settings, derived settings).
    windows: Vec<usize>,
    /// The pass was skipped in one of the two phases with the pass on.
    skipped: bool,
    /// The right code is in a window.
    held: bool,
    /// Per score: place of the right chunk inside its own window.
    in_window: Vec<Option<usize>>,
    /// Per score: place of the right chunk over the chunks of all windows together.
    global: Vec<Option<usize>>,
    /// The same for the evidence score without the anchor reference.
    global_raw: Option<usize>,
    /// Per score: place under the heads alone (two chunks per tentacle forwarded).
    heads: Vec<Option<usize>>,
    /// Per (score, shortlist): place under the reranker choosing inside each tentacle.
    rerank: Vec<Option<usize>>,
    /// Reads whose packets were not in the judged order.
    bridge_bad: usize,
    /// Chunks the reranker scored for this question.
    reranked: usize,
}

/// Where each tentacle's best chunk sits, per score, against all chunks (the note's first
/// prediction).
#[derive(Default)]
struct Skew {
    /// Per score: sum of the best chunk's relative centre in its window, and of its tokens.
    pos: Vec<f64>,
    len: Vec<f64>,
    windows: usize,
    /// All chunks: sum of relative centres, of tokens, and their number.
    all_pos: f64,
    all_len: f64,
    all: usize,
    /// The anchor's log ratio in every window read.
    anchors: Vec<f32>,
}

/// Relative centre of a span in a window of `of` tokens.
#[expect(clippy::cast_precision_loss, reason = "token positions for a report")]
fn centre(at: usize, len: usize, of: usize) -> f64 {
    (at as f64 + len as f64 / 2.0) / of.max(1) as f64
}

#[expect(clippy::cast_precision_loss, reason = "token counts for a report")]
fn measure(scan: &Scan, x: &Item, right: &dyn Fn(&Item, usize) -> bool, row: &mut Row, skew: &mut Skew) {
    if skew.pos.is_empty() {
        skew.pos = vec![0.0; Score::ALL.len()];
        skew.len = vec![0.0; Score::ALL.len()];
    }
    row.held = scan.windows.iter().any(|w| w.place.iter().any(|c| right(x, *c)));
    for w in &scan.windows {
        skew.windows += 1;
        skew.anchors.push(w.anchor());
        for &(at, len) in w.parts.iter().skip(1) {
            skew.all_pos += centre(at, len, w.len);
            skew.all_len += len as f64;
            skew.all += 1;
        }
    }
    for (si, score) in Score::ALL.iter().enumerate() {
        let mut inside: Option<usize> = None;
        let mut all: Vec<(f32, bool)> = Vec::new();
        let mut raw: Vec<(f32, bool)> = Vec::new();
        for w in &scan.windows {
            let ranked = w.ranked(*score);
            if let Some(&(at, len)) = ranked.first().and_then(|(k, _)| w.parts.get(k + 1))
                && let (Some(p), Some(l)) = (skew.pos.get_mut(si), skew.len.get_mut(si))
            {
                *p += centre(at, len, w.len);
                *l += len as f64;
            }
            let is_right = |k: usize| w.place.get(k).is_some_and(|c| right(x, *c));
            if let Some(r) = ranked.iter().position(|(k, _)| is_right(*k)) {
                inside = Some(inside.map_or(r, |b: usize| b.min(r)));
            }
            let anchor = w.anchor();
            for (k, s) in ranked {
                all.push((s, is_right(k)));
                if *score == Score::Evidence {
                    raw.push((s + anchor, is_right(k)));
                }
            }
        }
        let place = |v: &mut Vec<(f32, bool)>| {
            v.sort_by(|a, b| b.0.total_cmp(&a.0));
            v.iter().position(|x| x.1)
        };
        row.in_window.push(inside);
        row.global.push(place(&mut all));
        if *score == Score::Evidence {
            row.global_raw = place(&mut raw);
        }
    }
}

/// The reranker arms, in tie order: score (`Score::ALL`), then shortlist.
fn arms() -> Vec<(Score, usize)> {
    Score::ALL.iter().flat_map(|s| SHORTLISTS.iter().map(move |n| (*s, *n))).collect()
}

#[expect(clippy::too_many_lines, reason = "one run, phases in order")]
fn tests(home: &Path, stub: bool) -> Result<(String, String, bool), Fault> {
    let mut items = crate::e2::suite(home)?;
    if stub {
        items.truncate(2);
    }
    let n = items.len();
    let mut out = String::new();
    let org = Organism::open_with(home, home, &["neuromorph-nano".to_owned()], false, &home.join("memory"))?;
    let _ = writeln!(out, "neural: {}", nmapi::neural::bind(&org)?);
    let _ = writeln!(out, "dense: {}", nmapi::neural::start_dense(&org)?);
    let sub = org.sub();
    let path = |i: usize| sub.chunks.get(i).and_then(|c| sub.files.get(c.file)).map_or_else(String::new, |f| f.path.replace('\\', "/"));
    let right = |x: &Item, c: usize| {
        sub.chunks.get(c).is_some_and(|ch| x.keys.iter().any(|(f, a, b)| *f == path(c) && ch.start <= *b && ch.end >= *a))
    };
    let mut rows: Vec<Row> = items.iter().map(|x| Row { id: x.id.clone(), ..Row::default() }).collect();

    // Phase A: the live organism with the pass off (the served path with the qualified lane).
    let t = Instant::now();
    for (x, row) in items.iter().zip(&mut rows) {
        let a = org.ask(&x.q, 4000)?;
        row.pack.push(pack_place(x, &a));
        row.gates.push(gates(&a));
    }
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let off_s = t.elapsed().as_secs_f64() / n.max(1) as f64;

    // Phase B: the pass on. Per question: the organism under the E3.3 settings (this makes the
    // scan), every arm on that scan, then the organism under the derived settings.
    let base = Settings { per_tentacle: PER, ..Settings::default() };
    let _ = writeln!(out, "{}", nmapi::tentacle_pass::enable(&org, base));
    let served: Arc<TentaclePass> = org.tentacle_pass.read().ok().and_then(|p| p.clone()).ok_or_else(|| bad("the pass is not on"))?;
    let derived = Arc::new(served.retuned(Settings { score: DERIVED.0, shortlist: DERIVED.1, ..base }));
    let serve = |p: &Arc<TentaclePass>| -> Result<(), Fault> {
        *org.tentacle_pass.write().map_err(|_| bad("pass lock poisoned"))? = Some(Arc::clone(p));
        Ok(())
    };
    let note = || org.last_pass.lock().map(|s| s.clone()).unwrap_or_default();
    let mut skew = Skew::default();
    let (mut scan_s, mut arms_s, mut derived_s) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut enc_tokens, mut enc_ms, mut read_ms, mut reads) = (0_usize, 0.0_f64, 0.0_f64, 0_usize);
    let mut last_note = String::new();
    for (x, row) in items.iter().zip(&mut rows) {
        serve(&served)?;
        let t = Instant::now();
        let a = org.ask(&x.q, 4000)?;
        scan_s += t.elapsed().as_secs_f64();
        row.pack.push(pack_place(x, &a));
        row.gates.push(gates(&a));
        row.windows.push(windows(&a));
        row.skipped |= note().contains("skipped");

        let t = Instant::now();
        // The scan the organism just made (or a new one, were it not kept).
        let (scan, _) = served.scan(&sub, &x.q, &base)?;
        enc_tokens += scan.enc_tokens;
        enc_ms += scan.enc_ms;
        read_ms += scan.read_ms;
        reads += scan.windows.len() * 2;
        measure(&scan, x, &right, row, &mut skew);
        let read = |s: Settings, row: &mut Row| -> Result<Option<usize>, Fault> {
            let p = served.read_as(&sub, &org.slots, &x.q, &s)?;
            // The bridge keeps the judged order: packet k is chunk `order[k]`.
            let same = p.packets.len() == p.order.len()
                && p.packets.iter().zip(&p.order).enumerate().all(|(place, (pk, c))| {
                    sub.chunks.get(*c).is_some_and(|ch| pk.key() == format!("tentacle:{place:02}:{}:{}", path(*c), ch.start))
                });
            row.bridge_bad += usize::from(!same);
            Ok(p.order.iter().position(|c| right(x, *c)))
        };
        for score in Score::ALL {
            let place = read(Settings { judge: Judge::Heads, score, ..base }, row)?;
            row.heads.push(place);
        }
        for (score, shortlist) in arms() {
            let place = read(Settings { judge: Judge::Reranker, score, shortlist, ..base }, row)?;
            row.rerank.push(place);
        }
        row.reranked = scan.reranked();
        arms_s += t.elapsed().as_secs_f64();

        serve(&derived)?;
        let t = Instant::now();
        let a = org.ask(&x.q, 4000)?;
        derived_s += t.elapsed().as_secs_f64();
        row.pack.push(pack_place(x, &a));
        row.gates.push(gates(&a));
        row.windows.push(windows(&a));
        last_note = note();
        row.skipped |= last_note.contains("skipped");
    }
    serve(&served)?;

    // Report.
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let nf = n.max(1) as f64;
    let _ = writeln!(out, "G1 {} on {n} development questions (runbook runs/2026-11-10-G1-runbook.md)", if stub { "stub" } else { "dev" });
    let held = rows.iter().filter(|r| r.held).count();
    let _ = writeln!(out, "reach: right code in a window on {held}/{n} questions");

    let names: Vec<&str> = Score::ALL.iter().map(|s| s.name()).collect();
    let col = |f: &dyn Fn(&Row) -> Option<usize>| -> (usize, usize, f64) { tally(&rows.iter().map(f).collect::<Vec<_>>()) };
    let _ = writeln!(out, "\nreading statistic, heads alone (place of the right chunk; top, first three, mean reciprocal rank):");
    let mut forwarded = Vec::new();
    for (si, name) in names.iter().enumerate() {
        let w = col(&|r| r.in_window.get(si).copied().flatten());
        let two = rows.iter().filter(|r| r.in_window.get(si).copied().flatten().is_some_and(|k| k < PER)).count();
        let g = col(&|r| r.global.get(si).copied().flatten());
        let h = col(&|r| r.heads.get(si).copied().flatten());
        forwarded.push(two);
        let _ = writeln!(
            out,
            "  {name:<8} inside its window {}/{n}, {}/{n}, {:.3}; among its tentacle's best {PER}: {two}/{n}; over all windows together {}/{n}, {}/{n}, {:.3}; judged order of the forwarded chunks {}/{n}, {}/{n}, {:.3}",
            w.0, w.1, w.2, g.0, g.1, g.2, h.0, h.1, h.2
        );
    }
    let raw = col(&|r| r.global_raw);
    let _ = writeln!(out, "  evidence without the anchor reference, over all windows together: {}/{n}, {}/{n}, {:.3}", raw.0, raw.1, raw.2);

    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let (wn, an) = (skew.windows.max(1) as f64, skew.all.max(1) as f64);
    let _ = writeln!(
        out,
        "\nwhere each tentacle's best chunk sits ({} windows read; all chunks: mean centre {:.3} of the window, mean {:.0} tokens):",
        skew.windows,
        skew.all_pos / an,
        skew.all_len / an
    );
    for (si, name) in names.iter().enumerate() {
        let _ = writeln!(
            out,
            "  {name:<8} mean centre {:.3}, mean {:.0} tokens",
            skew.pos.get(si).copied().unwrap_or(0.0) / wn,
            skew.len.get(si).copied().unwrap_or(0.0) / wn
        );
    }
    if !skew.anchors.is_empty() {
        #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
        let count = skew.anchors.len() as f32;
        let mean = skew.anchors.iter().sum::<f32>() / count;
        let (lo, hi) = skew.anchors.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), a| (lo.min(*a), hi.max(*a)));
        let sd = (skew.anchors.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / count).sqrt();
        let _ = writeln!(
            out,
            "  anchor log ratio per window (summed over the drone heads): mean {mean:.3}, standard deviation {sd:.3}, least {lo:.3}, greatest {hi:.3}"
        );
    }

    let arm_list = arms();
    let arm_tally = |ai: usize, keep: &dyn Fn(usize) -> bool| -> (usize, usize, f64) {
        let v: Vec<Option<usize>> =
            rows.iter().enumerate().filter(|(qi, _)| keep(*qi)).map(|(_, r)| r.rerank.get(ai).copied().flatten()).collect();
        tally(&v)
    };
    let _ = writeln!(
        out,
        "\nreranker choosing inside each tentacle from a drone shortlist, {PER} forwarded (forwarded reach; top, first three, mean reciprocal rank):"
    );
    for (ai, (score, shortlist)) in arm_list.iter().enumerate() {
        let t = arm_tally(ai, &|_| true);
        let reach = rows.iter().filter(|r| r.rerank.get(ai).copied().flatten().is_some()).count();
        let _ = writeln!(out, "  {:<8} shortlist {shortlist}: reach {reach}/{n}; {}/{n}, {}/{n}, {:.3}", score.name(), t.0, t.1, t.2);
    }
    // The arm chosen on one fold, scored on the other (folds: even and odd places in suite order).
    let choose = |keep: &dyn Fn(usize) -> bool| -> usize {
        let mut best = 0;
        for ai in 1..arm_list.len() {
            if better(arm_tally(ai, keep), arm_tally(best, keep)) {
                best = ai;
            }
        }
        best
    };
    let (on_even, on_odd) = (choose(&|qi| qi % 2 == 0), choose(&|qi| qi % 2 == 1));
    let oos: Vec<Option<usize>> =
        rows.iter().enumerate().map(|(qi, r)| r.rerank.get(if qi % 2 == 0 { on_odd } else { on_even }).copied().flatten()).collect();
    let o = tally(&oos);
    let arm_name = |ai: usize| arm_list.get(ai).map_or_else(String::new, |(s, k)| format!("{} shortlist {k}", s.name()));
    let _ = writeln!(
        out,
        "  chosen out of sample (even fold chose {}, odd fold chose {}): {}/{n}, {}/{n}, {:.3}",
        arm_name(on_even),
        arm_name(on_odd),
        o.0,
        o.1,
        o.2
    );
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let mean_reranked = rows.iter().map(|r| r.reranked).sum::<usize>() as f64 / nf;
    let _ = writeln!(out, "  chunks the reranker scored per question, all arms together: {mean_reranked:.0}");

    // Decisions by the runbook's fixed rules.
    let mass_fwd = forwarded.first().copied().unwrap_or(0);
    let mut d1 = 0_usize;
    for si in [2_usize, 3, 1] {
        let f = forwarded.get(si).copied().unwrap_or(0);
        if f >= mass_fwd + 2 && f > forwarded.get(d1).copied().unwrap_or(0) {
            d1 = si;
        }
    }
    let d1_name = names.get(d1).copied().unwrap_or("mass");
    let at = |li: usize| arm_tally(d1 * SHORTLISTS.len() + li, &|_| true);
    let two = at(0);
    let mut d2 = PER;
    for (li, k) in SHORTLISTS.iter().enumerate().skip(1) {
        if at(li).0 >= two.0 + 2 {
            d2 = *k;
            break;
        }
    }
    let _ = writeln!(
        out,
        "\nD1 score: {d1_name} (among its tentacle's best {PER}: mass {mass_fwd}/{n}, {d1_name} {}/{n})",
        forwarded.get(d1).copied().unwrap_or(0)
    );
    let _ = writeln!(out, "D2 shortlist under {d1_name}: {d2}");

    // T1.
    let reads_total: usize = n * (Score::ALL.len() + arm_list.len());
    let bridge_bad: usize = rows.iter().map(|r| r.bridge_bad).sum();
    let mut t1 = bridge_bad == 0;
    let _ = writeln!(out, "\nT1a bridge order: {} of {reads_total} reads keep the judged order", reads_total.saturating_sub(bridge_bad));
    let rec = record(home);
    let heads_mass = col(&|r| r.heads.first().copied().flatten());
    let rr_mass = arm_tally(0, &|_| true);
    match &rec {
        Some((name, theirs)) if !stub => {
            let find = |id: &str| theirs.iter().find(|(i, _, _)| i == id);
            let their_heads = items.iter().filter(|x| find(&x.id).is_some_and(|r| r.1 == Some(0))).count();
            let their_rr = items.iter().filter(|x| find(&x.id).is_some_and(|r| r.2 == Some(0))).count();
            let same =
                items.iter().zip(&rows).filter(|(x, r)| find(&x.id).is_some_and(|t| t.1 == r.heads.first().copied().flatten())).count();
            let ok = heads_mass.0 + 2 >= their_heads;
            t1 &= ok;
            let _ = writeln!(
                out,
                "T1b heads, mass score, plain poses: top {}/{n}; E3.3 record top {their_heads}/{n}; same place on {same}/{n} questions: {}",
                heads_mass.0,
                if ok { "holds" } else { "below the record by more than 2" }
            );
            let _ = writeln!(
                out,
                "    reranker, mass score, shortlist {PER}: top {}/{n}; E3.3 record (focused poses) top {their_rr}/{n} (reported, no gate)",
                rr_mass.0
            );
            let _ = writeln!(
                out,
                "    E3.3 record: runs/{name} (the repository changed between that run and this one, so places are compared by count)"
            );
        }
        Some(_) => {
            let _ = writeln!(out, "T1b heads, mass score: top {}/{n} (stub: no record compared)", heads_mass.0);
        }
        None => {
            t1 &= stub;
            let _ = writeln!(out, "T1b: no E3.3 development record found");
        }
    }

    // T2.
    let zero = |r: &Row, phase: usize, k: usize| r.gates.get(phase).is_some_and(|g| g.iter().take(k).all(|x| *x == 0));
    let off_ok = rows.iter().filter(|r| zero(r, 0, 3)).count();
    let on_ok = rows.iter().filter(|r| zero(r, 1, 4)).count();
    let der_ok = rows.iter().filter(|r| zero(r, 2, 4)).count();
    let eight = nmsubs::lens::NAMES.len();
    let with_windows = rows.iter().filter(|r| r.windows.len() == 2 && r.windows.iter().all(|w| *w == eight)).count();
    let skipped = rows.iter().filter(|r| r.skipped).count();
    let t2 = on_ok == n && der_ok == n && with_windows == n && skipped == 0;
    let _ = writeln!(
        out,
        "T2 gates: all four at 0 on {on_ok}/{n} questions (E3.3 settings) and {der_ok}/{n} (derived settings: {} shortlist {}); a row for each of the {eight} windows in both on {with_windows}/{n}; pass skipped on {skipped}/{n}; pass off, the three segment gates at 0 on {off_ok}/{n}",
        DERIVED.0.name(),
        DERIVED.1
    );
    let _ = writeln!(out, "\npack (development, no gate; right code first, in the first three, anywhere):");
    for (phase, name) in ["pass off", "pass on, E3.3 settings", "pass on, derived settings"].iter().enumerate() {
        let v: Vec<Option<usize>> = rows.iter().map(|r| r.pack.get(phase).copied().flatten()).collect();
        let t = tally(&v);
        let any = v.iter().filter(|k| k.is_some()).count();
        let _ = writeln!(out, "  {name:<26} {}/{n}, {}/{n}, {any}/{n}", t.0, t.1);
    }
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let per_token = enc_ms / enc_tokens.max(1) as f64;
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let per_read = read_ms / reads.max(1) as f64;
    let _ = writeln!(
        out,
        "time per question: pass off {off_s:.1} s; pass on with the scan {:.1} s; all {} arms on the kept scan {:.1} s; derived settings on the kept scan {:.1} s; encode {per_token:.2} ms per token ({enc_tokens} tokens), {per_read:.0} ms per read ({reads} reads)",
        scan_s / nf,
        Score::ALL.len() + arm_list.len(),
        arms_s / nf,
        derived_s / nf
    );
    let _ = writeln!(out, "last pass: {last_note}");
    let _ = writeln!(out, "verdict: T1 {}, T2 {}", if t1 { "pass" } else { "FAIL" }, if t2 { "pass" } else { "FAIL" });

    let mut js = String::from(
        "{\"runbook\":\"runs/2026-11-10-G1-runbook.md\",\"scores\":[\"mass\",\"density\",\"evidence\",\"peak\"],\"shortlists\":[2,4,6],\"questions\":[\n",
    );
    for (i, r) in rows.iter().enumerate() {
        let _ = write!(
            js,
            "{}{{\"id\":\"{}\",\"held\":{},\"in_window\":{},\"global\":{},\"global_evidence_raw\":{},\"heads\":{},\"rerank\":{},\"pack\":{},\"gates\":{:?},\"windows\":{:?},\"reranked\":{}}}",
            if i == 0 { "" } else { ",\n" },
            r.id,
            r.held,
            list(&r.in_window),
            list(&r.global),
            show(r.global_raw),
            list(&r.heads),
            list(&r.rerank),
            list(&r.pack),
            r.gates,
            r.windows,
            r.reranked
        );
    }
    js.push_str("\n]}\n");
    Ok((out, js, t1 && t2))
}

/// `run g1 <stub|dev>`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let part = args.get(2).map_or("stub", String::as_str);
    if part != "stub" && part != "dev" {
        eprintln!("run g1 <stub|dev>");
        return ExitCode::FAILURE;
    }
    let stamp = nmtype::now_ms();
    let txt = home.join(format!("runs/G1-{part}-{stamp}.txt"));
    match tests(home, part == "stub") {
        Ok((text, js, ok)) => {
            let _ = std::fs::write(&txt, &text);
            let _ = std::fs::write(home.join(format!("runs/G1-{part}-{stamp}.json")), js);
            print!("{text}");
            println!("report: {}", txt.display());
            if ok || part == "stub" { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        Err(e) => {
            let _ = std::fs::write(&txt, format!("FAULT {e}\n"));
            eprintln!("g1 {part}: {e} (report {})", txt.display());
            ExitCode::FAILURE
        }
    }
}
