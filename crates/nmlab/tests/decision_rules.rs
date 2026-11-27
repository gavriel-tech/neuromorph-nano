//! Decision rules of the tentacle cores and G1 by replay of a scan dump (2026-11-10, Phase B).
//! Gavriel Technologies; architecture by Luis Jake Gabriel III.
//!
//! Processor only: the newest dump under `scratch/tc1-dump/` (written by `run tc1 dump`) is
//! replayed with the cores and the G1 loop under each rule set (`nmbrain::g1::Rules`), device
//! reads answered from the dump's tape (a read the tape lacks keeps the scan's scores, counted).
//! Two folds by question index parity inside the development half: rules are chosen on one fold
//! and reported on the other, both directions; the out-of-fold figures are the result, the
//! in-fold ones a diagnostic. The sealed claim half is never read (only even positions of the
//! sorted question files, the development half, as the locked harness orders it). Question text
//! is held in memory for the replay (G1's relevance and the tape keys need it) and never written.
//!
//! Run: `cargo test --release -p nmapi --test decision_rules -- --ignored --nocapture`.
//! Writes `runs/DR1-replay-<ms>.txt` (numbers, question indices and ids only).
use nmapi::scan_dump::{self, Tape};
use nmapi::tentacle_pass::Settings;
use nmbrain::g1::{self, Combine, ConfidenceRule, ForwardRule, Rules, ShortlistRule, Weights};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// A named per-window score (the T3 variants).
type ScoreFn = Box<dyn Fn(&nmapi::tentacle_pass::WindowScan) -> Vec<f32>>;

/// Every confidence rule of the report step (the T3 normalisations among them).
const NORMS: [ConfidenceRule; 6] = [
    ConfidenceRule::Logistic,
    ConfidenceRule::Window,
    ConfidenceRule::Standard,
    ConfidenceRule::Robust,
    ConfidenceRule::Margin,
    ConfidenceRule::NullMargin,
];

/// Kind of a chunk that beat the right chunk across windows, relative to the right chunk.
fn kind_of(sub: &nmsubs::Substrate, winner: usize, gold: &[usize]) -> &'static str {
    let path = |c: usize| sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file)).map_or_else(String::new, |f| f.path.replace('\\', "/"));
    let w = path(winner);
    let top = |p: &str| p.split('/').next().unwrap_or("").to_owned();
    let text = sub.chunks.get(winner).map_or("", |c| c.text.as_str());
    if gold.iter().any(|g| path(*g) == w) {
        "same file"
    } else if w.ends_with(".md") || w.ends_with(".txt") || w.ends_with(".json") {
        "document"
    } else if w.contains("/tests/") || w.contains("test") || text.contains("#[test]") || text.contains("#[cfg(test)]") {
        "test"
    } else if gold.iter().any(|g| top(&path(*g)) == top(&w)) {
        "same crate"
    } else {
        "other code"
    }
}

/// The development half: (id, question text) by index, as the locked harness orders it.
fn dev_half(home: &Path) -> Vec<(String, String)> {
    const TYPES: [&str; 5] = ["behaviour", "name", "value", "explain", "relation"];
    let mut out = Vec::new();
    for area in ["context", "engine", "tools"] {
        let p = home.join(format!("tests/capability-gates/cx1-context/questions-{area}.json"));
        let v = nmjson::parse(&std::fs::read(&p).expect("question file")).expect("json");
        let mut qs: Vec<(usize, String, String)> = v
            .get("questions")
            .and_then(|y| y.array())
            .expect("questions")
            .iter()
            .map(|x| {
                let s = |k: &str| x.get(k).and_then(|y| y.string()).expect("field").to_owned();
                let t = s("type");
                (TYPES.iter().position(|y| *y == t).expect("type"), s("id"), s("q"))
            })
            .collect();
        qs.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        out.extend(qs.into_iter().enumerate().filter(|(pos, _)| pos % 2 == 0).map(|(_, (_, id, q))| (id, q)));
    }
    out
}

/// The core's reported score: calibrated minus the anchor's.
fn above_anchor(w: &nmapi::tentacle_pass::WindowScan) -> Vec<f32> {
    let cal = w.calibrated_parts();
    let a = cal.first().copied().unwrap_or(0.0);
    cal.iter().skip(1).map(|x| x - a).collect()
}
/// Spread of the window's content-free read over its parts (as the tentacle core computes it).
fn null_spread(w: &nmapi::tentacle_pass::WindowScan) -> Option<f32> {
    let (heads, n) = (w.cal_heads(), w.parts.len());
    let parts: Vec<f32> = (0..n).map(|k| (0..heads).map(|h| w.cal_null.get(h * n + k).copied().unwrap_or(0.0)).sum::<f32>()).collect();
    #[expect(clippy::cast_precision_loss, reason = "a part count")]
    let mean = parts.iter().sum::<f32>() / n.max(1) as f32;
    #[expect(clippy::cast_precision_loss, reason = "a part count")]
    let sd = (parts.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n.max(1) as f32).sqrt();
    (n > 1 && sd.is_finite() && sd > 0.0).then_some(sd)
}

fn newest_dump(home: &Path) -> Option<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(home.join("scratch").join("tc1-dump"))
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("substrate.json").is_file())
        .collect();
    v.sort();
    v.pop()
}

/// One question's outcome under one rule set.
#[derive(Clone, Copy, Debug, Default)]
struct Out {
    first: bool,
    any: bool,
    spans: usize,
    misses: usize,
}

/// Totals over a set of questions: (right first, right anywhere, spans, tape misses, questions).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Tot {
    first: usize,
    any: usize,
    spans: usize,
    misses: usize,
    n: usize,
}
impl Tot {
    fn add(&mut self, o: Out) {
        self.first += usize::from(o.first);
        self.any += usize::from(o.any);
        self.spans += o.spans;
        self.misses += o.misses;
        self.n += 1;
    }
    fn of(outs: &[Out], idx: &[usize]) -> Self {
        let mut t = Self::default();
        for &i in idx {
            if let Some(o) = outs.get(i) {
                t.add(*o);
            }
        }
        t
    }
    /// Selection key: more right first, then more right anywhere, then fewer spans.
    fn key(&self) -> (usize, usize, std::cmp::Reverse<usize>) {
        (self.first, self.any, std::cmp::Reverse(self.spans))
    }
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    fn text(&self) -> String {
        format!(
            "first {} of {}, anywhere {} of {}, spans per question {:.1}, tape misses {}",
            self.first,
            self.n,
            self.any,
            self.n,
            self.spans as f64 / self.n.max(1) as f64,
            self.misses
        )
    }
}

/// Distance of a rule set from the first form, for ties (no change without evidence): 0 for
/// the first form, then the L1 distance of the weights from uniform plus one per changed rule.
fn distance(r: &Rules) -> f32 {
    let mut d = 0.0;
    if r.confidence != ConfidenceRule::Logistic {
        d += 1.0;
    }
    if r.shortlist != ShortlistRule::Entropy {
        d += 1.0;
    }
    if let Combine::Linear(w) = r.combine {
        d += 1.0 + [w.confidence, w.support, w.agreement, w.rank].iter().map(|x| (x - 0.25).abs()).sum::<f32>();
    }
    d
}

#[test]
#[ignore = "replays a scan dump (run on demand)"]
#[expect(clippy::too_many_lines, reason = "one replay study with its report")]
fn decision_rules_by_replay() {
    let home = nmtype::nano_root();
    let Some(dir) = newest_dump(&home) else {
        panic!("needs a scan dump under scratch/tc1-dump/ (write one with `nano-lab run tc1 dump`); none found");
    };
    let started = Instant::now();
    let items = dev_half(&home);
    let sub = scan_dump::load_substrate(&dir).expect("substrate");
    let settings = Settings { core: true, ..Settings::served() };
    // Load every question once.
    let mut qs = Vec::new();
    for idx in scan_dump::indices(&dir) {
        let Some((id, text)) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(&dir, idx, text).expect("question");
        assert_eq!(&q.id, id, "dump question {idx} does not match the development half");
        qs.push((idx, q, scan));
    }
    let n = qs.len();
    assert!(n > 0, "empty dump");
    // Folds by question index parity.
    let fold_a: Vec<usize> = (0..n).filter(|&i| qs.get(i).is_some_and(|(idx, _, _)| idx % 2 == 0)).collect();
    let fold_b: Vec<usize> = (0..n).filter(|&i| qs.get(i).is_some_and(|(idx, _, _)| idx % 2 == 1)).collect();
    let all_idx: Vec<usize> = (0..n).collect();
    // Grid resolution from the evidence: step 1 / m with m = ceil(sqrt(questions in the smaller
    // training fold)); finer steps than the questions can resolve would only fit noise.
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "a small count")]
    let m = ((fold_a.len().min(fold_b.len()).max(1) as f64).sqrt().ceil() as usize).max(1);
    let mut sets: Vec<Rules> = Vec::new();
    for confidence in NORMS {
        for shortlist in [ShortlistRule::Entropy, ShortlistRule::LargestGap] {
            sets.push(Rules { confidence, shortlist, combine: Combine::Product, forward: ForwardRule::Fixed });
            for w in g1::weight_grid(m) {
                sets.push(Rules { confidence, shortlist, combine: Combine::Linear(w), forward: ForwardRule::Fixed });
            }
        }
    }
    // Control arm and first form recorded in the dump.
    let right = |q: &scan_dump::QuestionDump, c: usize| q.gold.contains(&c);
    let control: Vec<Out> = qs
        .iter()
        .map(|(_, q, _)| Out {
            first: q.control.order.first().is_some_and(|c| right(q, *c)),
            any: q.control.order.iter().any(|c| right(q, *c)),
            spans: q.control.order.len(),
            misses: 0,
        })
        .collect();
    let live: Vec<Out> = qs
        .iter()
        .map(|(_, q, _)| Out {
            first: q.core.order.first().is_some_and(|c| right(q, *c)),
            any: q.core.order.iter().any(|c| right(q, *c)),
            spans: q.core.order.len(),
            misses: 0,
        })
        .collect();
    // Replay every rule set on every question.
    let mut table: Vec<Vec<Out>> = Vec::with_capacity(sets.len());
    for r in &sets {
        g1::set_rules(*r);
        let outs: Vec<Out> = qs
            .iter()
            .map(|(_, q, scan)| {
                let tape = Arc::new(Tape::replayer(q.tape.clone()));
                match nmapi::tentacle_core::judge_taped(&home, &sub, scan, &settings, Some(Arc::clone(&tape))) {
                    Ok((p, _)) => Out {
                        first: p.order.first().is_some_and(|c| right(q, *c)),
                        any: p.order.iter().any(|c| right(q, *c)),
                        spans: p.order.len(),
                        misses: tape.counts().1,
                    },
                    Err(e) => panic!("replay fault: {e}"),
                }
            })
            .collect();
        table.push(outs);
    }
    g1::set_rules(Rules::first_form());
    let all: Vec<usize> = (0..n).collect();
    let pick = |train: &[usize]| -> usize {
        let mut best = 0;
        for (i, outs) in table.iter().enumerate() {
            let (a, b) = (Tot::of(outs, train).key(), Tot::of(table.get(best).map_or(&[][..], Vec::as_slice), train).key());
            let closer = sets.get(i).map_or(f32::MAX, distance) < sets.get(best).map_or(f32::MAX, distance);
            if a > b || (a == b && closer) {
                best = i;
            }
        }
        best
    };
    let ff = 0; // sets[0] is the first form (Logistic, Entropy, Product).
    let mut rep = String::new();
    // ---- T3 (reported first): per-window normalisation of the report step. ----
    let _ = writeln!(
        rep,
        "T3 per-window normalisation (research note why-reads-degrade, cause 2), {n} development questions, dump {}",
        dir.display()
    );
    // (1) Right chunk first across all windows from the scan's scores, no G1 (static).
    let mut lines_t3: Vec<String> = Vec::new();
    let mut in_window_first = 0;
    let mut lost: Vec<&'static str> = Vec::new();
    let mut across_rows: Vec<(String, Vec<bool>)> = Vec::new();
    let mut variants: Vec<(String, ScoreFn)> = vec![
        ("mass (raw)".to_owned(), Box::new(|w| w.scores(nmapi::tentacle_pass::Score::Mass))),
        ("calibrated, no anchor".to_owned(), Box::new(|w| w.scores(nmapi::tentacle_pass::Score::Calibrated))),
        ("calibrated above the anchor (first form)".to_owned(), Box::new(above_anchor)),
    ];
    for r in [ConfidenceRule::Standard, ConfidenceRule::Robust, ConfidenceRule::Margin, ConfidenceRule::NullMargin] {
        variants.push((
            format!("{r:?} of calibrated above the anchor"),
            Box::new(move |w| {
                let sc = above_anchor(w);
                g1::normalised(r, &sc, w.cal_heads(), null_spread(w)).unwrap_or(sc)
            }),
        ));
    }
    for (name, f) in &variants {
        let mut hits = Vec::new();
        for (_, q, scan) in &qs {
            let mut best: Option<(f32, usize)> = None;
            for w in &scan.windows {
                for (k, v) in f(w).iter().enumerate() {
                    if let Some(c) = w.place.get(k)
                        && v.is_finite()
                        && best.is_none_or(|b| *v > b.0)
                    {
                        best = Some((*v, *c));
                    }
                }
            }
            hits.push(best.is_some_and(|b| q.gold.contains(&b.1)));
        }
        across_rows.push((name.clone(), hits));
    }
    for (_, q, scan) in &qs {
        // In-window first under the core's score, and what beat it across windows.
        let mut first_in = false;
        let mut best: Option<(f32, usize)> = None;
        for w in &scan.windows {
            let sc = above_anchor(w);
            let top = sc.iter().enumerate().filter(|(_, v)| v.is_finite()).max_by(|a, b| a.1.total_cmp(b.1));
            if let Some((k, v)) = top {
                if w.place.get(k).is_some_and(|c| q.gold.contains(c)) {
                    first_in = true;
                }
                if best.is_none_or(|b| *v > b.0) {
                    best = w.place.get(k).map(|c| (*v, *c));
                }
            }
        }
        in_window_first += usize::from(first_in);
        if let Some((_, c)) = best
            && first_in
            && !q.gold.contains(&c)
        {
            lost.push(kind_of(&sub, c, &q.gold));
        }
    }
    let _ = writeln!(
        rep,
        "right chunk first inside its window (calibrated above the anchor): {in_window_first} of {n}; first inside its window but beaten across windows: {} (by: {})",
        lost.len(),
        {
            let mut k: Vec<(&str, usize)> = Vec::new();
            for x in &lost {
                match k.iter_mut().find(|(a, _)| a == x) {
                    Some(e) => e.1 += 1,
                    None => k.push((x, 1)),
                }
            }
            k.iter().map(|(a, b)| format!("{a} {b}")).collect::<Vec<_>>().join(", ")
        }
    );
    let fold_count = |h: &[bool], idx: &[usize]| idx.iter().filter(|&&i| h.get(i).copied().unwrap_or(false)).count();
    let _ = writeln!(
        rep,
        "right chunk first across all windows (no G1; a fixed rule, nothing fitted, so all questions are out of sample): fold A / fold B / all"
    );
    for (name, h) in &across_rows {
        lines_t3.push(format!("  {name}: {} / {} / {} of {n}", fold_count(h, &fold_a), fold_count(h, &fold_b), fold_count(h, &all_idx)));
    }
    for l in &lines_t3 {
        let _ = writeln!(rep, "{l}");
    }
    // (2) Forwarded first through the G1 loop, G1 comparing the normalised values.
    let pick_in = |train: &[usize], keep: &dyn Fn(&Rules) -> bool| -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, outs) in table.iter().enumerate() {
            let Some(r) = sets.get(i).filter(|r| keep(r)) else { continue };
            let better = match best {
                None => true,
                Some(b) => {
                    let (ka, kb) = (Tot::of(outs, train).key(), Tot::of(table.get(b).map_or(&[][..], Vec::as_slice), train).key());
                    ka > kb || (ka == kb && distance(r) < sets.get(b).map_or(f32::MAX, distance))
                }
            };
            if better {
                best = Some(i);
            }
        }
        best
    };
    let _ = writeln!(
        rep,
        "forwarded through the G1 loop, per report-step normalisation (OUT OF FOLD: rules chosen on one parity fold among the sets with that normalisation, reported on the other, pooled):"
    );
    for norm in NORMS {
        let keep = |r: &Rules| r.confidence == norm;
        let mut oof = Tot::default();
        let mut chosen = Vec::new();
        for (train, test) in [(&fold_a, &fold_b), (&fold_b, &fold_a)] {
            if let Some(b) = pick_in(train, &keep) {
                chosen.push(sets.get(b).map(Rules::text).unwrap_or_default());
                let outs = table.get(b).map_or(&[][..], Vec::as_slice);
                for &i in test.iter() {
                    if let Some(o) = outs.get(i) {
                        oof.add(*o);
                    }
                }
            }
        }
        let pure = Rules {
            confidence: norm,
            shortlist: ShortlistRule::Entropy,
            combine: Combine::Linear(Weights { confidence: 1.0, support: 0.0, agreement: 0.0, rank: 0.0 }),
            forward: ForwardRule::Fixed,
        };
        let fixed = sets.iter().position(|r| *r == pure).map(|i| Tot::of(table.get(i).map_or(&[][..], Vec::as_slice), &all_idx));
        let _ = writeln!(
            rep,
            "  {norm:?}: out of fold {}; G1 ranking by the normalised value alone (fixed, nothing fitted, all questions) {}; chosen per fold: [{}]",
            oof.text(),
            fixed.map_or_else(String::new, |t| t.text()),
            chosen.join(" | ")
        );
    }
    let _ = writeln!(rep, "  control arm (reranker judge): {}", Tot::of(&control, &all_idx).text());
    let _ = writeln!(
        rep,
        "---- end of T3 ----
"
    );

    let _ = writeln!(rep, "DR1 decision rules of the tentacle cores and G1 by replay of a scan dump (processor only), {}", dir.display());
    let _ = writeln!(
        rep,
        "questions: {n} of the development half (indices {:?}); fold A (even index) {}, fold B (odd index) {}; rule sets replayed {} (confidence 6 x shortlist 2 x combine {} = product + simplex grid of four weights at step 1/{m}); replay time {:.1} s",
        qs.iter().map(|(i, _, _)| *i).collect::<Vec<_>>(),
        fold_a.len(),
        fold_b.len(),
        sets.len(),
        1 + g1::weight_grid(m).len(),
        started.elapsed().as_secs_f64()
    );
    let first_live_equal =
        table.get(ff).is_some_and(|o| o.iter().zip(&live).all(|(a, b)| a.first == b.first && a.any == b.any && a.spans == b.spans));
    let _ = writeln!(rep, "first-form replay reproduces the live core arm's forwarding on every question: {first_live_equal}");
    let mut oof = Tot::default();
    let mut oof_ff = Tot::default();
    let mut oof_ctl = Tot::default();
    for (name, train, test) in [("choose on A, report on B", &fold_a, &fold_b), ("choose on B, report on A", &fold_b, &fold_a)] {
        let b = pick(train);
        let r = sets.get(b).copied().unwrap_or_default();
        let outs = table.get(b).map_or(&[][..], Vec::as_slice);
        let (tr, te) = (Tot::of(outs, train), Tot::of(outs, test));
        let ffo = table.get(ff).map_or(&[][..], Vec::as_slice);
        let _ = writeln!(rep, "\n{name}: chosen {}", r.text());
        let _ = writeln!(
            rep,
            "  out of fold (result): chosen {}; first form {}; control arm {}",
            te.text(),
            Tot::of(ffo, test).text(),
            Tot::of(&control, test).text()
        );
        let _ = writeln!(rep, "  in fold (diagnostic): chosen {}; first form {}", tr.text(), Tot::of(ffo, train).text());
        for &i in test.iter() {
            if let Some(o) = outs.get(i) {
                oof.add(*o);
            }
            if let Some(o) = ffo.get(i) {
                oof_ff.add(*o);
            }
            if let Some(o) = control.get(i) {
                oof_ctl.add(*o);
            }
        }
    }
    let _ = writeln!(
        rep,
        "\nOUT OF FOLD, both folds pooled ({} questions): chosen rules {}; first-form rules {}; control arm {}",
        oof.n,
        oof.text(),
        oof_ff.text(),
        oof_ctl.text()
    );
    // Diagnostics: each rule family alone on all questions (in-sample).
    let _ = writeln!(rep, "\ndiagnostic, all questions in sample:");
    for (label, want) in [
        ("first form", Rules::first_form()),
        ("window confidence only", Rules { confidence: ConfidenceRule::Window, ..Rules::first_form() }),
        ("standard confidence only", Rules { confidence: ConfidenceRule::Standard, ..Rules::first_form() }),
        ("largest-gap shortlist only", Rules { shortlist: ShortlistRule::LargestGap, ..Rules::first_form() }),
        (
            "window + gap, product",
            Rules {
                confidence: ConfidenceRule::Window,
                shortlist: ShortlistRule::LargestGap,
                combine: Combine::Product,
                forward: ForwardRule::Fixed,
            },
        ),
        (
            "window + gap, uniform linear",
            Rules {
                confidence: ConfidenceRule::Window,
                shortlist: ShortlistRule::LargestGap,
                combine: Combine::Linear(Weights { confidence: 0.25, support: 0.25, agreement: 0.25, rank: 0.25 }),
                forward: ForwardRule::Fixed,
            },
        ),
    ] {
        if let Some(i) = sets.iter().position(|s| *s == want) {
            let _ = writeln!(rep, "  {label}: {}", Tot::of(table.get(i).map_or(&[][..], Vec::as_slice), &all).text());
        }
    }
    let best_all = pick(&all);
    let _ = writeln!(
        rep,
        "  best in sample on all questions (not a result): {} -> {}",
        sets.get(best_all).map(Rules::text).unwrap_or_default(),
        Tot::of(table.get(best_all).map_or(&[][..], Vec::as_slice), &all).text()
    );
    let _ =
        writeln!(rep, "  control arm: {}; live core arm of the dump run: {}", Tot::of(&control, &all).text(), Tot::of(&live, &all).text());
    let _ = writeln!(rep, "\nper question (index id: control first/any/spans | first form | chosen out of fold):");
    let chosen_a = pick(&fold_a);
    let chosen_b = pick(&fold_b);
    for (i, (idx, q, _)) in qs.iter().enumerate() {
        let c = control.get(i).copied().unwrap_or_default();
        let f = table.get(ff).and_then(|o| o.get(i)).copied().unwrap_or_default();
        let which = if idx % 2 == 0 { chosen_b } else { chosen_a };
        let o = table.get(which).and_then(|v| v.get(i)).copied().unwrap_or_default();
        let _ = writeln!(
            rep,
            "  q{} {}: {}/{}/{} | {}/{}/{} | {}/{}/{} (misses {})",
            idx + 1,
            q.id,
            c.first,
            c.any,
            c.spans,
            f.first,
            f.any,
            f.spans,
            o.first,
            o.any,
            o.spans,
            o.misses
        );
    }
    let _ = writeln!(rep, "\ntotal {:.1} s", started.elapsed().as_secs_f64());
    let path = home.join("runs").join(format!("DR1-replay-{}.txt", nmtype::now_ms()));
    std::fs::write(&path, &rep).expect("record");
    eprintln!("{rep}\nrecord {}", path.display());
}

/// Gate 1 shortlist sizes per window under the first-form and largest-gap rules, from the dump's
/// scores (no replay of G1): mean, smallest and largest, and how often the right chunk falls
/// inside the shortlist of its window.
#[test]
#[ignore = "reads a scan dump (run on demand)"]
fn shortlist_sizes_from_dump() {
    let home = nmtype::nano_root();
    let Some(dir) = newest_dump(&home) else { return };
    let items = dev_half(&home);
    let mut rows: Vec<(ShortlistRule, Vec<usize>, usize, usize)> =
        vec![(ShortlistRule::Entropy, Vec::new(), 0, 0), (ShortlistRule::LargestGap, Vec::new(), 0, 0)];
    for idx in scan_dump::indices(&dir) {
        let Some((_, text)) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(&dir, idx, text).expect("question");
        for w in &scan.windows {
            let sc = above_anchor(w);
            let mut order: Vec<usize> = (0..sc.len()).collect();
            order.sort_by(|a, b| {
                sc.get(*b).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&sc.get(*a).copied().unwrap_or(f32::NEG_INFINITY))
            });
            let has_gold = w.place.iter().any(|c| q.gold.contains(c));
            for (rule, sizes, inside, gold_windows) in &mut rows {
                let (k, _) = g1::shortlist_by(*rule, &sc, 6);
                sizes.push(k);
                if has_gold {
                    *gold_windows += 1;
                    if order.iter().take(k).any(|i| w.place.get(*i).is_some_and(|c| q.gold.contains(c))) {
                        *inside += 1;
                    }
                }
            }
        }
    }
    for (rule, sizes, inside, gold_windows) in &rows {
        #[expect(clippy::cast_precision_loss, reason = "small counts")]
        let mean = sizes.iter().sum::<usize>() as f64 / sizes.len().max(1) as f64;
        eprintln!(
            "{rule:?}: {} windows, shortlist mean {mean:.1}, min {}, max {}; right chunk inside the shortlist in {inside} of {gold_windows} windows holding one",
            sizes.len(),
            sizes.iter().min().unwrap_or(&0),
            sizes.iter().max().unwrap_or(&0)
        );
    }
}

/// Refinement step 2a (2026-11-10): the second look's shortlist is taken by the mass score of the
/// full read. From the dump's scans (processor only; the dump holds no second-look reads, so only
/// the shortlist is measured here): per window holding the right chunk, the size the rule gives and
/// whether the right chunk is inside it, for the entropy rule on the mass, the largest-gap rule on
/// the mass, the mass-share cover rule (`nmapi::core_read::look_size`) and the stated fallback 6;
/// and right chunk first by mass and by the calibrated score.
#[test]
#[ignore = "reads a scan dump (run on demand)"]
fn second_look_shortlist_from_dump() {
    let home = nmtype::nano_root();
    let Some(dir) = newest_dump(&home) else { return };
    let items = dev_half(&home);
    let t = Instant::now();
    let names = ["entropy of the mass", "largest gap of the mass", "look_size (derived)", "fixed 6 (stated fallback)"];
    let mut sizes: Vec<Vec<usize>> = vec![Vec::new(); names.len()];
    let mut inside = vec![0_usize; names.len()];
    let (mut gold_windows, mut first_mass, mut first_cal) = (0, 0, 0);
    for idx in scan_dump::indices(&dir) {
        let Some((_, text)) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(&dir, idx, text).expect("question");
        for w in &scan.windows {
            if !w.place.iter().any(|c| q.gold.contains(c)) {
                continue;
            }
            gold_windows += 1;
            let mass = w.scores(nmapi::tentacle_pass::Score::Mass);
            let cal = above_anchor(w);
            let order = |v: &[f32]| {
                let mut o: Vec<usize> = (0..v.len()).collect();
                o.sort_by(|a, b| {
                    v.get(*b)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY)
                        .total_cmp(&v.get(*a).copied().unwrap_or(f32::NEG_INFINITY))
                        .then(a.cmp(b))
                });
                o
            };
            let is_gold = |i: &usize| w.place.get(*i).is_some_and(|c| q.gold.contains(c));
            let om = order(&mass);
            first_mass += usize::from(om.first().is_some_and(is_gold));
            first_cal += usize::from(order(&cal).first().is_some_and(is_gold));
            let ks = [
                g1::shortlist_by(ShortlistRule::Entropy, &mass, 6).0,
                g1::shortlist_by(ShortlistRule::LargestGap, &mass, 6).0,
                nmapi::core_read::look_size(&mass).0,
                6.min(mass.len()),
            ];
            for (j, k) in ks.into_iter().enumerate() {
                sizes[j].push(k);
                inside[j] += usize::from(om.iter().take(k).any(&is_gold));
            }
        }
    }
    eprintln!(
        "second-look shortlist by the full read's mass, {gold_windows} windows holding the right chunk; right chunk first by mass {first_mass}, by the calibrated score {first_cal}; {:.2} s",
        t.elapsed().as_secs_f64()
    );
    for (j, name) in names.iter().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "small counts")]
        let mean = sizes[j].iter().sum::<usize>() as f64 / sizes[j].len().max(1) as f64;
        eprintln!(
            "{name}: size mean {mean:.1}, min {}, max {}; right chunk inside in {} of {gold_windows}",
            sizes[j].iter().min().unwrap_or(&0),
            sizes[j].iter().max().unwrap_or(&0),
            inside[j]
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Refinement step 1 (2026-11-10): the Globalized Context Stack carries evidence and drops nothing. Replay of the same
// dump on the processor through the cores, G1 (both forward rules), the bridge, the reduce step
// (tentacle packets only: the dump holds no exact-reader packets) and G2's selection.
// ---------------------------------------------------------------------------------------------

/// One question's Globalized Context Stack outcome under one rule set.
#[derive(Clone, Copy, Debug, Default)]
struct TrOut {
    report_any: bool,
    fwd_any: bool,
    fwd_first: bool,
    held_any: bool,
    bridge_first: bool,
    pack_any: bool,
    pack_first: bool,
    g2_any: bool,
    g2_first: bool,
    g2_set_any: bool,
    fwd_n: usize,
    fwd_tokens: usize,
    held_tokens: usize,
    queued: usize,
    released: usize,
    released_gold: bool,
    low: usize,
}

/// Is the citation line `path:line ...` inside a gold chunk?
fn line_is_gold(sub: &nmsubs::Substrate, gold: &[usize], line: &str) -> bool {
    let head = line.split_once(' ').map_or(line, |(a, _)| a);
    let Some((p, n)) = head.rsplit_once(':') else { return false };
    let Ok(n) = n.split('-').next().unwrap_or("").parse::<usize>() else { return false };
    let p = p.replace('\\', "/");
    gold.iter().any(|g| {
        sub.chunks
            .get(*g)
            .is_some_and(|c| c.start <= n && n <= c.end && sub.files.get(c.file).is_some_and(|f| f.path.replace('\\', "/") == p))
    })
}
fn key_is_gold(sub: &nmsubs::Substrate, gold: &[usize], key: &str) -> bool {
    gold.iter().any(|g| sub.chunks.get(*g).is_some_and(|c| c.key == key))
}

/// Replay one rule set on every question; with `downstream` also the reduce step and G2.
fn stack_run(
    home: &Path,
    sub: &nmsubs::Substrate,
    qs: &[(usize, scan_dump::QuestionDump, nmapi::tentacle_pass::Scan)],
    items: &[(String, String)],
    r: Rules,
    downstream: bool,
) -> Vec<TrOut> {
    g1::set_rules(r);
    let settings = Settings { core: true, ..Settings::served() };
    let slots = Arc::new(nmbrain::Slots::default());
    let book = nmbrain::chain::ChainBook::from_mini().expect("chains");
    let selector = nmbrain::g2::Selector::fit(Arc::clone(&slots), &nmbrain::g2::BuiltinCases).expect("g2");
    let out = qs
        .iter()
        .map(|(idx, q, scan)| {
            let tape = Arc::new(Tape::replayer(q.tape.clone()));
            let (p, cr) = nmapi::tentacle_core::judge_taped(home, sub, scan, &settings, Some(Arc::clone(&tape))).expect("replay");
            let gold = |c: usize| q.gold.contains(&c);
            let set = &cr.outcome.evidence;
            let gk = |k: &str| key_is_gold(sub, &q.gold, k);
            let mut o = TrOut {
                fwd_any: p.order.iter().any(|c| gold(*c)),
                fwd_first: p.order.first().is_some_and(|c| gold(*c)),
                held_any: set.all().any(|e| gk(&e.span.key)),
                report_any: set.all().any(|e| gk(&e.span.key) && !e.reasons.iter().any(|x| x.contains("below gate 1"))),
                fwd_n: p.order.len(),
                fwd_tokens: set.forwarded.iter().map(|e| e.facts.tokens).sum(),
                held_tokens: set.all().map(|e| e.facts.tokens).sum(),
                queued: set.queue.len(),
                ..TrOut::default()
            };
            if r.forward == ForwardRule::Fixed {
                // First form: what is not forwarded is dropped; nothing is held.
                o.held_any = o.fwd_any;
                o.held_tokens = o.fwd_tokens;
                o.queued = 0;
            }
            let best = p.packets.iter().max_by(|a, b| a.get_confidence().total_cmp(&b.get_confidence()));
            o.bridge_first = best.is_some_and(|b| line_is_gold(sub, &q.gold, b.provenance().source.as_str()));
            if downstream {
                let question = items.get(*idx).map_or("", |x| x.1.as_str());
                let mut ev = set.clone();
                let pack = match p.evidence.as_ref() {
                    Some(_) => nmbrain::synthesis::reduce_evidence(1, question, &p.packets, 4000, &slots, &mut ev),
                    None => nmbrain::synthesis::reduce(1, question, &p.packets, 4000, &slots),
                };
                let lines: Vec<String> = pack.packets.iter().map(nmapi::brain_side::part_line).collect();
                o.pack_any = lines.iter().any(|l| line_is_gold(sub, &q.gold, l));
                o.pack_first = lines.first().is_some_and(|l| line_is_gold(sub, &q.gold, l));
                let before_fwd: Vec<String> = ev.forwarded.iter().map(|e| e.span.key.clone()).collect();
                let sel = if p.evidence.is_some() {
                    selector.select_evidence(question, &lines, &book, &mut ev)
                } else {
                    selector.select(question, &lines, &book)
                };
                o.g2_any = sel.upstream.iter().any(|l| line_is_gold(sub, &q.gold, l));
                o.g2_first = sel.upstream.first().is_some_and(|l| line_is_gold(sub, &q.gold, l));
                o.g2_set_any = if p.evidence.is_some() { ev.all().any(|e| gk(&e.span.key)) } else { o.g2_any };
                let released: Vec<&nmbrain::evidence::SpanEvidence> =
                    ev.forwarded.iter().filter(|e| e.round > 1 && !before_fwd.contains(&e.span.key)).collect();
                o.released = released.len();
                o.released_gold = released.iter().any(|e| gk(&e.span.key));
                o.low = ev.queue.iter().filter(|e| e.priority == nmbrain::evidence::Priority::Low).count();
            }
            o
        })
        .collect();
    g1::set_rules(Rules::first_form());
    out
}

#[derive(Clone, Copy, Debug, Default)]
struct TrTot {
    n: usize,
    report_any: usize,
    fwd_any: usize,
    fwd_first: usize,
    held_any: usize,
    bridge_first: usize,
    pack_any: usize,
    pack_first: usize,
    g2_any: usize,
    g2_first: usize,
    g2_set_any: usize,
    fwd_n: usize,
    fwd_tokens: usize,
    held_tokens: usize,
    queued: usize,
    released: usize,
    released_gold: usize,
    low: usize,
}
impl TrTot {
    fn of(v: &[TrOut], idx: &[usize]) -> Self {
        let mut t = Self::default();
        for o in idx.iter().filter_map(|i| v.get(*i)) {
            t.n += 1;
            t.report_any += usize::from(o.report_any);
            t.fwd_any += usize::from(o.fwd_any);
            t.fwd_first += usize::from(o.fwd_first);
            t.held_any += usize::from(o.held_any);
            t.bridge_first += usize::from(o.bridge_first);
            t.pack_any += usize::from(o.pack_any);
            t.pack_first += usize::from(o.pack_first);
            t.g2_any += usize::from(o.g2_any);
            t.g2_first += usize::from(o.g2_first);
            t.g2_set_any += usize::from(o.g2_set_any);
            t.fwd_n += o.fwd_n;
            t.fwd_tokens += o.fwd_tokens;
            t.held_tokens += o.held_tokens;
            t.queued += o.queued;
            t.released += o.released;
            t.released_gold += usize::from(o.released_gold);
            t.low += o.low;
        }
        t
    }
    fn key(&self) -> (usize, usize) {
        (self.fwd_first, self.fwd_any)
    }
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    fn text(&self) -> String {
        let per = |x: usize| x as f64 / self.n.max(1) as f64;
        format!(
            "forwarded first {} / present {} of {}; held anywhere by G1 {}; spans forwarded per question {:.1} ({:.0} tokens); tokens held per question {:.0}; queued per question {:.1}",
            self.fwd_first,
            self.fwd_any,
            self.n,
            self.held_any,
            per(self.fwd_n),
            per(self.fwd_tokens),
            per(self.held_tokens),
            per(self.queued)
        )
    }
}

/// Distance of evidence weights from uniform over the active features (ties: no change without
/// evidence, the most even weighting wins).
fn ev_distance(w: &g1::EvWeights, active: &[usize]) -> f32 {
    #[expect(clippy::cast_precision_loss, reason = "a feature count")]
    let u = 1.0 / active.len().max(1) as f32;
    active.iter().map(|&f| (w.0.get(f).copied().unwrap_or(0.0) - u).abs()).sum()
}

#[test]
#[ignore = "replays a scan dump (run on demand)"]
#[expect(clippy::too_many_lines, reason = "one replay study with its report")]
fn stack_by_replay() {
    let home = nmtype::nano_root();
    let Some(dir) = newest_dump(&home) else {
        panic!("needs a scan dump under scratch/tc1-dump/ (write one with `nano-lab run tc1 dump`); none found");
    };
    let started = Instant::now();
    let items = dev_half(&home);
    let sub = scan_dump::load_substrate(&dir).expect("substrate");
    let mut qs = Vec::new();
    for idx in scan_dump::indices(&dir) {
        let Some((id, text)) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(&dir, idx, text).expect("question");
        assert_eq!(&q.id, id, "dump question {idx} does not match the development half");
        qs.push((idx, q, scan));
    }
    let n = qs.len();
    let g = sub.refs();
    let mut rep = String::new();
    let _ = writeln!(
        rep,
        "TR1 context stack replay (refinement step 1): {n} development questions, dump {}, processor only, no model loaded",
        dir.display()
    );
    let _ = writeln!(
        rep,
        "reference graph over items: {} chunks, {} item names, {} edges, {} bytes in RAM, built in {:.0} ms",
        g.keys.len(),
        g.names,
        g.edges,
        g.bytes(),
        g.build_ms
    );
    let lens_known = qs.iter().any(|(_, _, s)| s.windows.iter().any(|w| !w.lens_rank.is_empty()));
    // Active evidence features: lens rank only when the dump recorded it.
    let mut active: Vec<usize> = (0..7).collect();
    if lens_known {
        active.push(7);
    }
    let _ = writeln!(
        rep,
        "lens ranks in the dump: {lens_known}; features combined: {}",
        active.iter().filter_map(|i| g1::EV_FEATURES.get(*i)).copied().collect::<Vec<_>>().join(", ")
    );
    let fold_a: Vec<usize> = (0..n).filter(|&i| qs.get(i).is_some_and(|(idx, _, _)| idx % 2 == 0)).collect();
    let fold_b: Vec<usize> = (0..n).filter(|&i| qs.get(i).is_some_and(|(idx, _, _)| idx % 2 == 1)).collect();
    let all: Vec<usize> = (0..n).collect();
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "a small count")]
    let m = ((fold_a.len().min(fold_b.len()).max(1) as f64).sqrt().ceil() as usize).max(1);
    let grid = g1::EvWeights::grid(m, &active);
    let first = Rules::first_form();
    let queue = |combine: Combine| Rules { combine, forward: ForwardRule::Queue, ..first };
    // Arms.
    let ff = stack_run(&home, &sub, &qs, &items, first, true);
    let qp = stack_run(&home, &sub, &qs, &items, queue(Combine::Product), true);
    let t0 = Instant::now();
    let table: Vec<Vec<TrOut>> = grid.iter().map(|w| stack_run(&home, &sub, &qs, &items, queue(Combine::Evidence(*w)), false)).collect();
    let grid_s = t0.elapsed().as_secs_f64();
    let _ = writeln!(
        rep,
        "grid: {} weightings (simplex over {} features at step 1/{m}, m = ceil(sqrt(smaller fold))), replayed in {grid_s:.1} s",
        grid.len(),
        active.len()
    );
    let pick = |train: &[usize]| -> usize {
        let mut best = 0;
        for (i, outs) in table.iter().enumerate() {
            let (a, b) = (TrTot::of(outs, train).key(), TrTot::of(table.get(best).map_or(&[][..], Vec::as_slice), train).key());
            let closer =
                grid.get(i).map_or(f32::MAX, |w| ev_distance(w, &active)) < grid.get(best).map_or(f32::MAX, |w| ev_distance(w, &active));
            if a > b || (a == b && closer) {
                best = i;
            }
        }
        best
    };
    let mut oof: Vec<TrOut> = vec![TrOut::default(); n];
    let mut chosen = Vec::new();
    for (name, train, test) in
        [("choose on A (even), report on B (odd)", &fold_a, &fold_b), ("choose on B (odd), report on A (even)", &fold_b, &fold_a)]
    {
        let b = pick(train);
        let w = grid.get(b).copied().unwrap_or(g1::EvWeights([0.0; 8]));
        let outs = stack_run(&home, &sub, &qs, &items, queue(Combine::Evidence(w)), true);
        for &i in test.iter() {
            if let (Some(x), Some(o)) = (oof.get_mut(i), outs.get(i)) {
                *x = *o;
            }
        }
        let _ = writeln!(
            rep,
            "{name}: chosen {}; in fold {}; out of fold {}",
            w.text(),
            TrTot::of(&outs, train).text(),
            TrTot::of(&outs, test).text()
        );
        chosen.push(w);
    }
    let served_i = pick(&all);
    let served = grid.get(served_i).copied().unwrap_or(g1::EvWeights([0.0; 8]));
    let sv = stack_run(&home, &sub, &qs, &items, queue(Combine::Evidence(served)), true);
    let _ =
        writeln!(rep, "weights chosen on all {n} questions (the served rule; its in-sample figures are not the result): {}", served.text());
    let _ = writeln!(rep, "SERVED_EVIDENCE_WEIGHTS {:?}", served.0);
    let (tf, tq, toof, tsv) = (TrTot::of(&ff, &all), TrTot::of(&qp, &all), TrTot::of(&oof, &all), TrTot::of(&sv, &all));
    let _ = writeln!(rep, "\nRESULT (out of fold where weights are chosen):");
    let _ = writeln!(rep, "  first form (fixed forward, product): {}", tf.text());
    let _ = writeln!(rep, "  queue rule, first-form product combine (capacity alone, nothing fitted): {}", tq.text());
    let _ = writeln!(rep, "  queue rule, evidence combine, OUT OF FOLD: {}", toof.text());
    let _ = writeln!(rep, "  queue rule, evidence combine, served weights in sample (diagnostic): {}", tsv.text());
    let _ = writeln!(
        rep,
        "\n{}",
        loss_table(
            &qs,
            &[
                ("first form", tf),
                ("queue rule, product (served context stack rule)", tq),
                ("queue rule, evidence combine (out of fold)", toof)
            ]
        )
    );
    let _ = writeln!(
        rep,
        "\nqueue: spans queued per question first form 0 (dropped), evidence path {:.1}; released later by G2's NEXT: {} spans over {n} questions, holding the right chunk on {} questions; marked low (reducer floor or room, G2 word filter): {} records",
        toof.queued as f64 / n.max(1) as f64,
        toof.released,
        toof.released_gold,
        toof.low
    );
    let _ = writeln!(
        rep,
        "\nper question (index id: first form fwd any/first, held | evidence out of fold fwd any/first, held, pack first, G2 first, NEXT released):"
    );
    for (i, (idx, q, _)) in qs.iter().enumerate() {
        let (a, b) = (ff.get(i).copied().unwrap_or_default(), oof.get(i).copied().unwrap_or_default());
        let _ = writeln!(
            rep,
            "  q{} {}: {}/{} {} | {}/{} {} {} {} {}",
            idx + 1,
            q.id,
            a.fwd_any,
            a.fwd_first,
            a.held_any,
            b.fwd_any,
            b.fwd_first,
            b.held_any,
            b.pack_first,
            b.g2_first,
            b.released
        );
    }
    // Window fill (item 5): layout replay on the dump's substrate with the base tokenizer file.
    let _ = writeln!(rep, "\n{}", layout_replay(&home, &sub, &qs, &items));
    let _ = writeln!(rep, "\ntotal {:.1} s", started.elapsed().as_secs_f64());
    let path = home.join("runs").join(format!("TR1-replay-{}.txt", nmtype::now_ms()));
    std::fs::write(&path, &rep).expect("record");
    eprintln!("{rep}\nrecord {}", path.display());
}

/// The loss table: right chunk present and first at each hand-over, one column per arm.
fn loss_table(qs: &[(usize, scan_dump::QuestionDump, nmapi::tentacle_pass::Scan)], cols: &[(&str, TrTot)]) -> String {
    let (mut h0, mut h1, mut h3) = (0, 0, 0);
    for (_, q, scan) in qs {
        h0 += usize::from(scan.windows.iter().any(|w| w.place.iter().any(|c| q.gold.contains(c))));
        let mut best: Option<(f32, usize)> = None;
        let mut fin = false;
        for w in &scan.windows {
            let sc = above_anchor(w);
            if let Some((k, v)) = sc.iter().enumerate().filter(|(_, v)| v.is_finite()).max_by(|a, b| a.1.total_cmp(b.1)) {
                fin |= w.place.get(k).is_some_and(|c| q.gold.contains(c));
                if best.is_none_or(|b| *v > b.0) {
                    best = w.place.get(k).map(|c| (*v, *c));
                }
            }
        }
        h1 += usize::from(fin);
        h3 += usize::from(best.is_some_and(|b| q.gold.contains(&b.1)));
    }
    let mut out = format!(
        "loss table over {} questions (right chunk present / first at each hand-over)\n| hand-over | {} |\n|---|{}\n",
        qs.len(),
        cols.iter().map(|c| c.0).collect::<Vec<_>>().join(" | "),
        "---|".repeat(cols.len())
    );
    let mut row = |name: &str, f: &dyn Fn(&TrTot) -> String| {
        let cells: Vec<String> = cols.iter().map(|c| f(&c.1)).collect();
        let _ = writeln!(out, "| {name} | {} |", cells.join(" | "));
    };
    row("H0 window formation (dump windows)", &|_| format!("{h0} / -"));
    row("H1 drone read, first in its own window (calibrated)", &|_| format!("{h0} / {h1}"));
    row("H2 core reports (gate 1)", &|t| format!("{} / -", t.report_any));
    row("H3 across windows by drone score alone", &|_| format!("- / {h3}"));
    row("H4 G1 forwards (round 1)", &|t| format!("{} / {}", t.fwd_any, t.fwd_first));
    row("H4 held by G1 (forwarded or queued)", &|t| format!("{}", t.held_any));
    row("H5 bridge (first = highest packet relevance)", &|t| format!("{} / {}", t.fwd_any, t.bridge_first));
    row("H7 pack (reduce, tentacle packets only, 4,000 tokens)", &|t| format!("{} / {}", t.pack_any, t.pack_first));
    row("H9 G2 upstream parts (after NEXT)", &|t| format!("{} / {}", t.g2_any, t.g2_first));
    row("H9 present anywhere in what reaches G2 (parts or the set it holds)", &|t| format!("{}", t.g2_set_any));
    row("spans released later by NEXT (questions where one is the right chunk)", &|t| format!("{} ({})", t.released, t.released_gold));
    row("records marked low (reducer floor or room, G2 word filter)", &|t| format!("{}", t.low));
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    row("tokens forwarded / held by G1 per question", &|t| {
        format!("{:.0} / {:.0}", t.fwd_tokens as f64 / t.n.max(1) as f64, t.held_tokens as f64 / t.n.max(1) as f64)
    });
    out
}

/// The served Globalized Context Stack rule against the first form, with the reduce step and G2 (seconds; no grid).
#[test]
#[ignore = "replays a scan dump (run on demand)"]
fn stack_served_replay() {
    let home = nmtype::nano_root();
    let Some(dir) = newest_dump(&home) else { return };
    let started = Instant::now();
    let items = dev_half(&home);
    let sub = scan_dump::load_substrate(&dir).expect("substrate");
    let mut qs = Vec::new();
    for idx in scan_dump::indices(&dir) {
        let Some((_, text)) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(&dir, idx, text).expect("question");
        qs.push((idx, q, scan));
    }
    let all: Vec<usize> = (0..qs.len()).collect();
    let ff = TrTot::of(&stack_run(&home, &sub, &qs, &items, Rules::first_form(), true), &all);
    let tr = TrTot::of(&stack_run(&home, &sub, &qs, &items, Rules::stack(), true), &all);
    let ev = TrTot::of(&stack_run(&home, &sub, &qs, &items, Rules::evidence(), true), &all);
    let mut rep = format!(
        "TR2 served context stack rule replay: {} questions, dump {}\nserved context stack rule: {}\nevidence combine (weights refitted on all 30; in sample, a diagnostic): {}\n",
        qs.len(),
        dir.display(),
        Rules::stack().text(),
        g1::EvWeights(g1::STACK_WEIGHTS).text()
    );
    rep.push_str(&loss_table(&qs, &[("first form", ff), ("served context stack rule", tr), ("evidence combine, in sample", ev)]));
    let _ = writeln!(rep, "first form: {}\nserved context stack: {}\ntotal {:.1} s", ff.text(), tr.text(), started.elapsed().as_secs_f64());
    let path = home.join("runs").join(format!("TR2-served-{}.txt", nmtype::now_ms()));
    std::fs::write(&path, &rep).expect("record");
    eprintln!("{rep}\nrecord {}", path.display());
}

/// Window formation replayed on the processor (item 5): the same lens orders with and without the
/// fallback fill, tokens by the base tokenizer file. The replay substrate has no history, no
/// embedding order and no recalled chunks (the dump saved none), so the meaning, history and
/// memory lenses are empty in both arms; the comparison is the fill rule alone under equal
/// conditions, beside the windows the dump run actually held.
fn layout_replay(
    home: &Path,
    sub: &nmsubs::Substrate,
    qs: &[(usize, scan_dump::QuestionDump, nmapi::tentacle_pass::Scan)],
    items: &[(String, String)],
) -> String {
    let t = Instant::now();
    let Ok(tok) = nmengn::tokenizer::Tokenizer::open(&nmtype::models_dir(home).join("qwen3-0.6b").join("files").join("tokenizer.json"))
    else {
        return "layout replay: tokenizer file not readable".into();
    };
    let tokens: Vec<usize> = sub
        .chunks
        .iter()
        .map(|c| {
            sub.files.get(c.file).map_or(0, |f| tok.encode(&nmsubs::dense::doc_text(&f.path, &c.text)).map_or(usize::MAX, |v| v.len()))
        })
        .collect();
    let tok_s = t.elapsed().as_secs_f64();
    let (mut before_w, mut after_w, mut before_t, mut after_t, mut dump_w, mut dump_t, mut reach_b, mut reach_a) = (0, 0, 0, 0, 0, 0, 0, 0);
    let mut names_empty = 0;
    for (idx, q, scan) in qs {
        let question = items.get(*idx).map_or("", |x| x.1.as_str());
        let lists = nmsubs::lens::orders(sub, question, None, &[]);
        names_empty += usize::from(lists.get(2).is_some_and(Vec::is_empty));
        let (filled, _) = nmsubs::lens::fill_empty(&lists);
        let tk = |c: usize| tokens.get(c).copied().unwrap_or(usize::MAX);
        let wb = nmsubs::lens::assign(&lists, &tk, nmsubs::lens::BUDGET);
        let wa = nmsubs::lens::assign(&filled, &tk, nmsubs::lens::BUDGET);
        let sum = |w: &[Vec<usize>]| w.iter().flatten().map(|c| tk(*c)).sum::<usize>();
        before_w += wb.iter().filter(|w| !w.is_empty()).count();
        after_w += wa.iter().filter(|w| !w.is_empty()).count();
        before_t += sum(&wb);
        after_t += sum(&wa);
        reach_b += usize::from(wb.iter().flatten().any(|c| q.gold.contains(c)));
        reach_a += usize::from(wa.iter().flatten().any(|c| q.gold.contains(c)));
        dump_w += scan.windows.len();
        dump_t += scan.window_tokens;
    }
    let n = qs.len().max(1);
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let per = |x: usize| x as f64 / n as f64;
    format!(
        "window formation (item 5), layout replay over {} questions (tokenizer {tok_s:.1} s for {} chunks): names lens empty on {names_empty}; windows carrying context per question {:.2} without the fill, {:.2} with it (of 8); tokens of context held per question {:.0} without, {:.0} with; right chunk in some window {reach_b} without, {reach_a} with. The dump run itself held {:.2} windows and {:.0} tokens per question (with the embedding order and history the replay lacks).",
        qs.len(),
        sub.chunks.len(),
        per(before_w),
        per(after_w),
        per(before_t),
        per(after_t),
        per(dump_w),
        per(dump_t)
    )
}

/// One question of the T-E4 replay: (index, (first, present) per judge, reproduces control).
type JudgeRow = (usize, Vec<(bool, bool)>, bool);

/// T-E4 (2026-11-10, teardown patch 2, E-P4): the served judge by replay of the control arm.
/// Processor only: each window's shortlist of 6 by mass, the reranker's P(yes) recorded in the
/// dump chooses 2 inside each tentacle (as the served pass does), and the 12 candidates are
/// ordered by each judge (`reranker`, `rrf`, `sum`). Two folds by question index parity: the
/// judge is chosen on one fold (ties go to the present `reranker`) and reported on the other.
/// The replay of `reranker` must reproduce the dump's control order on every question.
/// Writes `runs/TE4-judge-replay-<ms>.txt`.
#[test]
#[ignore = "reads a scan dump (run on demand)"]
fn served_judge_replay() {
    use nmapi::tentacle_pass::{Cand, Judge, Score, judge_order};
    let home = nmtype::nano_root();
    let dir = std::env::var("NM_DUMP").map_or_else(|_| newest_dump(&home), |d| Some(home.join(d)));
    let Some(dir) = dir else { return };
    let limit: usize = std::env::var("NM_LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let started = Instant::now();
    let items = dev_half(&home);
    let judges = [Judge::Reranker, Judge::Rrf, Judge::Sum];
    // Per question: (index, [(first, present)] per judge, reproduces control).
    let mut rows: Vec<JudgeRow> = Vec::new();
    for idx in scan_dump::indices(&dir).into_iter().take(limit) {
        let Some((_, text)) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(&dir, idx, text).expect("question");
        let p = scan_dump::load_rerank(&dir, idx).expect("rerank");
        let mut cands = Vec::new();
        for w in &scan.windows {
            let mut chosen: Vec<(usize, f32)> = w.ranked(Score::Mass).into_iter().take(6).collect();
            let pv = |k: usize| w.place.get(k).and_then(|c| p.get(c)).copied().unwrap_or(0.0);
            chosen.sort_by(|a, b| pv(b.0).total_cmp(&pv(a.0)));
            chosen.truncate(2);
            for (place, &(k, score)) in chosen.iter().enumerate() {
                if let Some(c) = w.place.get(k).copied() {
                    cands.push(Cand { tentacle: w.tentacle, chunk: c, score, place, lines: Vec::new() });
                }
            }
        }
        let pc: Vec<f32> = cands.iter().map(|c| p.get(&c.chunk).copied().unwrap_or(0.0)).collect();
        let mut per = Vec::new();
        let mut same = false;
        for j in judges {
            let order: Vec<usize> = judge_order(&cands, Some(&pc), j).into_iter().filter_map(|i| cands.get(i).map(|c| c.chunk)).collect();
            if j == Judge::Reranker {
                same = order == q.control.order;
            }
            per.push((order.first().is_some_and(|c| q.gold.contains(c)), order.iter().any(|c| q.gold.contains(c))));
        }
        rows.push((idx, per, same));
    }
    let count = |j: usize, fold: Option<usize>| -> (usize, usize, usize) {
        let sel: Vec<&JudgeRow> = rows.iter().filter(|r| fold.is_none_or(|f| r.0 % 2 == f)).collect();
        (
            sel.iter().filter(|r| r.1.get(j).is_some_and(|x| x.0)).count(),
            sel.iter().filter(|r| r.1.get(j).is_some_and(|x| x.1)).count(),
            sel.len(),
        )
    };
    let mut rep = String::new();
    let _ = writeln!(
        rep,
        "T-E4 served judge by replay of the control arm (processor only), dump {}, {} questions; replay of reranker reproduces the control order on {} of {}",
        dir.display(),
        rows.len(),
        rows.iter().filter(|r| r.2).count(),
        rows.len()
    );
    for (j, judge) in judges.iter().enumerate() {
        let (a, b, t) = (count(j, Some(0)), count(j, Some(1)), count(j, None));
        let _ = writeln!(
            rep,
            "  {}: first fold A {} of {}, fold B {} of {}, all {} of {}; present {} of {}",
            judge.name(),
            a.0,
            a.2,
            b.0,
            b.2,
            t.0,
            t.2,
            t.1,
            t.2
        );
    }
    let (mut oof_first, mut oof_present, mut oof_n, mut chosen) = (0, 0, 0, Vec::new());
    for (train, test) in [(0, 1), (1, 0)] {
        let mut best = 0;
        for j in 1..judges.len() {
            if count(j, Some(train)).0 > count(best, Some(train)).0 {
                best = j;
            }
        }
        let t = count(best, Some(test));
        oof_first += t.0;
        oof_present += t.1;
        oof_n += t.2;
        chosen.push(judges.get(best).map_or("", |j| j.name()));
    }
    let ctl = count(0, None);
    let _ = writeln!(
        rep,
        "OUT OF FOLD (chosen on one parity fold, reported on the other, pooled): first {oof_first} of {oof_n}, present {oof_present} of {oof_n}; chosen per fold (on A, on B): {chosen:?}; present reranker judge first {} of {}",
        ctl.0, ctl.2
    );
    let _ = writeln!(rep, "per question (index: first per judge reranker/rrf/sum, present):");
    for (idx, per, _) in &rows {
        let _ = writeln!(
            rep,
            "  q{}: {} | {}",
            idx + 1,
            per.iter().map(|x| u8::from(x.0).to_string()).collect::<Vec<_>>().join("/"),
            u8::from(per.first().is_some_and(|x| x.1))
        );
    }
    let _ = writeln!(rep, "total {:.2} s", started.elapsed().as_secs_f64());
    if limit == usize::MAX {
        let path = home.join("runs").join(format!("TE4-judge-replay-{}.txt", nmtype::now_ms()));
        std::fs::write(&path, &rep).expect("record");
        eprintln!("record {}", path.display());
    }
    eprintln!("{rep}");
}
