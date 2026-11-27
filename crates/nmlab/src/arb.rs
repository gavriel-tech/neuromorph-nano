//! `nano-lab run arb <record>...` (2026-11-12): processor replays of the Core Brain's merge
//! of its three judges (research file, Part 10 (e) arbitration). Input: the `judges q<n>:` lines
//! of `run dx whole` records under `runs/` (gold chunk ids; per piece in arrival order its chunk,
//! the reranker's P(yes) and the embedder's cosine). Output `runs/ARB-<ms>.txt`: right chunk
//! first per merge rule; rules with free numbers are chosen on one half of the questions and
//! scored on the other (two folds). Ids and numbers only.
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;

/// One question: gold chunks, and per piece (chunk, reranker, embedder), in arrival order.
struct Q {
    n: usize,
    gold: Vec<usize>,
    pieces: Vec<(usize, f32, f32)>,
}

fn parse(text: &str) -> Vec<Q> {
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("judges q") else { continue };
        let Some((n, rest)) = rest.split_once(": gold ") else { continue };
        let Some((gold, pieces)) = rest.split_once("; pieces ") else { continue };
        let gold = gold.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        let pieces = pieces
            .split_whitespace()
            .filter_map(|p| {
                let mut it = p.split(':');
                Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?, it.next()?.parse().ok()?))
            })
            .collect();
        if let Ok(n) = n.parse() {
            out.push(Q { n, gold, pieces });
        }
    }
    out
}

/// Indices best first by score (ties: arrival order).
fn by(s: &[f64]) -> Vec<usize> {
    let mut v: Vec<usize> = (0..s.len()).collect();
    v.sort_by(|a, b| s.get(*b).unwrap_or(&0.0).total_cmp(s.get(*a).unwrap_or(&0.0)).then(a.cmp(b)));
    v
}

/// Weighted reciprocal-rank merge: score = sum of w / (rank + k), ranks from 1.
fn rrf(orders: &[(Vec<usize>, f64)], n: usize, k: f64) -> Vec<usize> {
    let mut s = vec![0.0_f64; n];
    for (o, w) in orders {
        for (r, i) in o.iter().enumerate() {
            if let Some(x) = s.get_mut(*i) {
                #[expect(clippy::cast_precision_loss, reason = "ranks")]
                {
                    *x += w / (r as f64 + k);
                }
            }
        }
    }
    by(&s)
}

/// A judge's confidence on this question: the gap between its top two scores over the spread of
/// its scores (a margin in standard deviations).
fn margin(s: &[f64]) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let n = s.len().max(1) as f64;
    let mean = s.iter().sum::<f64>() / n;
    let sd = (s.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt().max(1e-9);
    let o = by(s);
    match (o.first().and_then(|i| s.get(*i)), o.get(1).and_then(|i| s.get(*i))) {
        (Some(a), Some(b)) => (a - b) / sd,
        _ => 0.0,
    }
}

/// The judges of one question: reranker log-odds, embedder cosine, arrival (higher first).
fn judges(q: &Q) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let logit = |p: f32| {
        let p = f64::from(p).clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln()
    };
    #[expect(clippy::cast_precision_loss, reason = "positions")]
    let arrival: Vec<f64> = (0..q.pieces.len()).map(|i| -(i as f64)).collect();
    (q.pieces.iter().map(|p| logit(p.1)).collect(), q.pieces.iter().map(|p| f64::from(p.2)).collect(), arrival)
}

fn first_right(q: &Q, order: &[usize]) -> bool {
    order.first().and_then(|i| q.pieces.get(*i)).is_some_and(|p| q.gold.contains(&p.0))
}

/// A merge rule with numbers: (weights reranker, embedder, arrival; k; margin-weighted).
#[derive(Clone, Copy, Debug)]
struct Rule {
    w: (f64, f64, f64),
    k: f64,
    margin: bool,
}
fn apply(q: &Q, r: Rule) -> Vec<usize> {
    let (a, b, c) = judges(q);
    let m = |s: &[f64]| if r.margin { margin(s).max(0.05) } else { 1.0 };
    rrf(&[(by(&a), r.w.0 * m(&a)), (by(&b), r.w.1 * m(&b)), (by(&c), r.w.2)], q.pieces.len(), r.k)
}
fn count(qs: &[&Q], r: Rule) -> usize {
    qs.iter().filter(|q| first_right(q, &apply(q, r))).count()
}

fn grid() -> Vec<Rule> {
    let ws = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
    let mut v = Vec::new();
    for k in [1.0, 2.0, 3.0, 5.0, 10.0, 60.0] {
        for margin in [false, true] {
            for a in ws {
                for b in ws {
                    for c in ws {
                        if a + b + c > 0.0 {
                            v.push(Rule { w: (a, b, c), k, margin });
                        }
                    }
                }
            }
        }
    }
    v
}

fn replay(home: &Path, files: &[String]) -> Result<String, String> {
    let mut qs = Vec::new();
    for f in files {
        let t = std::fs::read_to_string(home.join("runs").join(f)).map_err(|e| format!("{f}: {e}"))?;
        qs.extend(parse(&t));
    }
    qs.sort_by_key(|q| q.n);
    qs.dedup_by_key(|q| q.n);
    if qs.is_empty() {
        return Err("no `judges q` lines in the records given".into());
    }
    let all: Vec<&Q> = qs.iter().collect();
    let sent = all.iter().filter(|q| q.pieces.iter().any(|p| q.gold.contains(&p.0))).count();
    let mut out = format!(
        "ARB (2026-11-12): processor replay of the Core Brain's merge over {} questions ({sent} with the right chunk sent up). Right chunk first:\n",
        all.len()
    );
    let fixed = [
        ("today: equal weights, k = 1 (decide::agree)", Rule { w: (1.0, 1.0, 1.0), k: 1.0, margin: false }),
        ("reranker alone", Rule { w: (1.0, 0.0, 0.0), k: 1.0, margin: false }),
        ("embedder alone", Rule { w: (0.0, 1.0, 0.0), k: 1.0, margin: false }),
        ("arrival order alone (the context stack)", Rule { w: (0.0, 0.0, 1.0), k: 1.0, margin: false }),
        ("equal weights, k = 60 (common RRF)", Rule { w: (1.0, 1.0, 1.0), k: 60.0, margin: false }),
        ("equal weights, margin-weighted judges, k = 1", Rule { w: (1.0, 1.0, 1.0), k: 1.0, margin: true }),
    ];
    for (name, r) in fixed {
        let _ = writeln!(out, "  {name}: {}", count(&all, r));
    }
    // Upper bound: right chunk first in any one judge.
    let any = all
        .iter()
        .filter(|q| {
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
                .iter()
                .any(|w| first_right(q, &apply(q, Rule { w: *w, k: 1.0, margin: false })))
        })
        .count();
    let _ = writeln!(out, "  first in at least one judge (upper bound for any merge of ranks): {any}");
    // Two folds: odd and even question numbers.
    let g = grid();
    let folds: [Vec<&Q>; 2] =
        [all.iter().copied().filter(|q| q.n % 2 == 1).collect(), all.iter().copied().filter(|q| q.n % 2 == 0).collect()];
    let mut oof = 0;
    for (i, train) in folds.iter().enumerate() {
        let test = folds.get(1 - i).map_or(&[][..], Vec::as_slice);
        // Best on train; ties keep the rule closest to today's (equal weights, k = 1, no margin).
        let dist = |r: &Rule| {
            (r.w.0 - 1.0).abs() + (r.w.1 - 1.0).abs() + (r.w.2 - 1.0).abs() + (r.k - 1.0).abs() / 10.0 + if r.margin { 0.5 } else { 0.0 }
        };
        let best = g
            .iter()
            .max_by(|a, b| count(train, **a).cmp(&count(train, **b)).then(dist(b).total_cmp(&dist(a))))
            .copied()
            .unwrap_or(Rule { w: (1.0, 1.0, 1.0), k: 1.0, margin: false });
        let (tr, te) = (count(train, best), count(test, best));
        oof += te;
        let _ = writeln!(
            out,
            "  fold {} (chosen on {} questions): weights reranker {} embedder {} arrival {}, k {}, margin {} -> {tr} of {} there, {te} of {} on the other fold (today's rule there: {})",
            i + 1,
            train.len(),
            best.w.0,
            best.w.1,
            best.w.2,
            best.k,
            best.margin,
            train.len(),
            test.len(),
            count(test, Rule { w: (1.0, 1.0, 1.0), k: 1.0, margin: false })
        );
    }
    let _ = writeln!(out, "  out of fold, chosen rules: {oof} of {}", all.len());
    // Score merge: each judge's scores standardised per question (reranker as log-odds), then a
    // weighted sum; weights chosen on one fold, scored on the other.
    let zs = |s: &[f64]| -> Vec<f64> {
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let n = s.len().max(1) as f64;
        let m = s.iter().sum::<f64>() / n;
        let sd = (s.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n).sqrt().max(1e-9);
        s.iter().map(|x| (x - m) / sd).collect()
    };
    let fuse = |q: &Q, w: (f64, f64, f64)| -> Vec<usize> {
        let (a, b, c) = judges(q);
        let (a, b, c) = (zs(&a), zs(&b), zs(&c));
        by(&(0..q.pieces.len())
            .map(|i| w.0 * a.get(i).unwrap_or(&0.0) + w.1 * b.get(i).unwrap_or(&0.0) + w.2 * c.get(i).unwrap_or(&0.0))
            .collect::<Vec<_>>())
    };
    let sc = |qs: &[&Q], w: (f64, f64, f64)| qs.iter().filter(|q| first_right(q, &fuse(q, w))).count();
    let _ = writeln!(out, "  score merge, equal weights: {}", sc(&all, (1.0, 1.0, 1.0)));
    let ws = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0];
    let mut sgrid = Vec::new();
    for a in ws {
        for b in ws {
            for c in ws {
                if a + b + c > 0.0 {
                    sgrid.push((a, b, c));
                }
            }
        }
    }
    let mut soof = 0;
    for (i, train) in folds.iter().enumerate() {
        let test = folds.get(1 - i).map_or(&[][..], Vec::as_slice);
        let d = |w: &(f64, f64, f64)| (w.0 - 1.0).abs() + (w.1 - 1.0).abs() + (w.2 - 1.0).abs();
        let best =
            sgrid.iter().max_by(|x, y| sc(train, **x).cmp(&sc(train, **y)).then(d(y).total_cmp(&d(x)))).copied().unwrap_or((1.0, 1.0, 1.0));
        let te = sc(test, best);
        soof += te;
        let _ = writeln!(
            out,
            "  score merge fold {}: weights reranker {} embedder {} arrival {} -> {} of {} there, {te} of {} on the other fold",
            i + 1,
            best.0,
            best.1,
            best.2,
            sc(train, best),
            train.len(),
            test.len()
        );
    }
    let _ = writeln!(out, "  score merge out of fold: {soof} of {}", all.len());
    // 2026-11-12, continuous learning step 3: the L4 calibrator on these decisions. The decision
    // is the reranker's first; the label is whether it is right; the base probability is its
    // P(yes); the calibrator is left out one question at a time, with a shuffled-label control.
    let (mut xs, mut ys, mut base) = (Vec::new(), Vec::new(), Vec::new());
    for q in &all {
        let rr: Vec<f32> = q.pieces.iter().map(|p| p.1).collect();
        let Some(best) = (0..rr.len()).max_by(|a, b| rr.get(*a).unwrap_or(&0.0).total_cmp(rr.get(*b).unwrap_or(&0.0)).then(b.cmp(a)))
        else {
            continue;
        };
        xs.push(nmlearn::calibrate::features(&rr));
        ys.push(q.pieces.get(best).is_some_and(|p| q.gold.contains(&p.0)));
        base.push(f64::from(rr.get(best).copied().unwrap_or(0.5)));
    }
    // With agreement features: does the embedder's first match, and where the chosen piece arrived.
    let mut xa = Vec::new();
    for (q, x) in all.iter().zip(&xs) {
        let rr: Vec<f32> = q.pieces.iter().map(|p| p.1).collect();
        let em: Vec<f32> = q.pieces.iter().map(|p| p.2).collect();
        let top = |s: &[f32]| {
            (0..s.len()).max_by(|a, b| s.get(*a).unwrap_or(&0.0).total_cmp(s.get(*b).unwrap_or(&0.0)).then(b.cmp(a))).unwrap_or(0)
        };
        let (r, e) = (top(&rr), top(&em));
        let mut f = x.clone();
        f.push(f64::from(u8::from(r == e)));
        #[expect(clippy::cast_precision_loss, reason = "positions")]
        f.push(r as f64 / q.pieces.len().max(1) as f64);
        xa.push(f);
    }
    let ga = nmlearn::calibrate::gate(&xa, &ys, &base, 1.0);
    let _ = writeln!(
        out,
        "  L4 calibrator with agreement features: log-loss calibrated {:.3}, shuffled {:.3}; better on {}, worse on {}, p {:.4} -> {}",
        ga.calibrated,
        ga.shuffled,
        ga.better,
        ga.worse,
        ga.p,
        if ga.keep { "KEEP" } else { "NOT KEPT" }
    );
    let g = nmlearn::calibrate::gate(&xs, &ys, &base, 1.0);
    let _ = writeln!(
        out,
        "  L4 calibrator on {} decisions ({} right): mean log-loss base {:.3}, calibrated (left out one at a time) {:.3}, shuffled-label control {:.3}; better on {}, worse on {}, sign test p {:.4} -> {}",
        ys.len(),
        ys.iter().filter(|y| **y).count(),
        g.base,
        g.calibrated,
        g.shuffled,
        g.better,
        g.worse,
        g.p,
        if g.keep { "KEEP" } else { "NOT KEPT" }
    );
    let _ = writeln!(out, "Per question, place of the right chunk (today | reranker | embedder | arrival):");
    for q in &all {
        let place = |r: Rule| {
            apply(q, r)
                .iter()
                .position(|i| q.pieces.get(*i).is_some_and(|p| q.gold.contains(&p.0)))
                .map_or_else(|| "-".to_owned(), |p| (p + 1).to_string())
        };
        let _ = writeln!(
            out,
            "  q{}: {} | {} | {} | {}",
            q.n,
            place(Rule { w: (1.0, 1.0, 1.0), k: 1.0, margin: false }),
            place(Rule { w: (1.0, 0.0, 0.0), k: 1.0, margin: false }),
            place(Rule { w: (0.0, 1.0, 0.0), k: 1.0, margin: false }),
            place(Rule { w: (0.0, 0.0, 1.0), k: 1.0, margin: false })
        );
    }
    Ok(out)
}

/// `run arb <record>...` (file names under `runs/`).
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let files: Vec<String> = args.iter().skip(2).cloned().collect();
    match replay(home, &files) {
        Ok(text) => {
            let p = home.join(format!("runs/ARB-{}.txt", nmtype::now_ms()));
            let _ = std::fs::write(&p, &text);
            print!("{text}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("arb: {e}");
            ExitCode::FAILURE
        }
    }
}
