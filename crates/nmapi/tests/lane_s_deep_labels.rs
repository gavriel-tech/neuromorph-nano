//! Lane S (2026-11-19): the tentacles' deep check as one packed card run. `nmengn::kvstore::
//! read_group_labels` over 8 groups (2 deep checks x the posed question and the three content-free
//! forms of `tentacle_core::NULL_FORMS`, each group the joint read's span for that form: the system
//! line, `Question: <form>`, the pieces as `joint_read::piece_text` places them; one tail per piece,
//! `Does piece [k] answer the question? Answer yes or no.`) against `joint_read::pass_with` called
//! one group at a time, as `tentacle_core::deep_check` calls it. Served switches on (as `serve`),
//! the reader as served (`neural::model`). Reports each tail's yes-minus-no log-odds difference,
//! whether every sign is the same, and the times.
//!
//! Run (card, under `scratch/LOCKS/card.lock`):
//! `cargo test --release -p nmapi --test lane_s_deep_labels -- --ignored --nocapture`.
use std::sync::atomic::Ordering;
use std::time::Instant;

// The joint read's prompt parts (`joint_read`'s private `SYSTEM` and `ASSISTANT`), copied; the test
// fails on any difference, since the packed reads must then differ from `pass_with`'s.
const SYSTEM: &str = "<|im_start|>system\nYou read pieces of one code repository and judge which pieces answer a question.<|im_end|>\n<|im_start|>user\n";
const ASSISTANT: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

#[test]
#[ignore = "card job: needs the reader"]
fn deep_check_packed() {
    if let Err(e) = run() {
        panic!("lane S deep labels: {e}");
    }
}

fn run() -> Result<(), String> {
    let home = nmtype::nano_root();
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let m = nmapi::neural::model(&home).map_err(|e| e.message)?;
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| e.to_string());
    let piece = |f: &str, first: usize, n: usize| -> Result<nmapi::joint_read::Piece, String> {
        let text = std::fs::read_to_string(home.join(f)).map_err(|e| format!("{f}: {e}"))?;
        let body = text.lines().skip(first - 1).take(n).collect::<Vec<_>>().join("\n");
        Ok(nmapi::joint_read::Piece { path: f.to_owned(), start: first, end: first + n - 1, text: body, ..Default::default() })
    };
    let checks: Vec<(&str, Vec<nmapi::joint_read::Piece>)> = vec![
        ("Where is an activation released after its last reader has run?", vec![piece("crates/nmengn/src/model.rs", 1188, 30)?, piece("crates/nmengn/src/kvstore.rs", 40, 12)?]),
        ("How does the engine choose the kernel for an 8-bit linear?", vec![piece("crates/nmengn/src/gpu.rs", 560, 45)?, piece("crates/nmengn/src/roles.rs", 300, 20)?]),
    ];
    let forms = |q: &str| -> Vec<String> { std::iter::once(q.to_owned()).chain(["N/A", "", "[MASK]"].iter().map(|s| (*s).to_owned())).collect() };
    let first = |s: &str| -> Result<usize, String> { tok(s)?.first().and_then(|x| usize::try_from(*x).ok()).ok_or_else(|| format!("no token for {s}")) };
    let rows = [first("yes")?, first("no")?];
    // Per-group reads, as the deep check does them today.
    let per_group = || -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::new();
        for (q, ps) in &checks {
            let refs: Vec<&nmapi::joint_read::Piece> = ps.iter().collect();
            let order: Vec<usize> = (0..ps.len()).collect();
            for f in forms(q) {
                out.push(nmapi::joint_read::pass_with(&m, nmengn::placement::Device::Gpu, &f, &refs, &order, &[], 17, true, false).map_err(|e| e.message)?.yes_no);
            }
        }
        Ok(out)
    };
    // The same groups in one packed run.
    let mut groups: Vec<(Vec<i32>, Vec<Vec<i32>>)> = Vec::new();
    for (q, ps) in &checks {
        for f in forms(q) {
            let mut span = tok(&format!("{SYSTEM}Question: {f}\n\n"))?;
            for (k, p) in ps.iter().enumerate() {
                span.extend(tok(&nmapi::joint_read::piece_text(k + 1, p))?);
            }
            let tails = (0..ps.len())
                .map(|k| tok(&format!("\nQuestion: {f}\nDoes piece [{}] answer the question? Answer yes or no.{ASSISTANT}", k + 1)))
                .collect::<Result<Vec<_>, _>>()?;
            groups.push((span, tails));
        }
    }
    let spans: Vec<usize> = groups.iter().map(|g| g.0.len()).collect();
    let packed = || -> Result<Vec<Vec<f32>>, String> {
        let _served = nmengn::gpu::ServedScope::new();
        let r = nmengn::kvstore::read_group_labels(&m, None, &groups, &rows).map_err(|e| e.message)?;
        Ok(r.iter().map(|g| g.iter().map(|t| t.first().copied().unwrap_or(0.0) - t.get(1).copied().unwrap_or(0.0)).collect()).collect())
    };
    let time = |f: &dyn Fn() -> Result<Vec<Vec<f32>>, String>| -> Result<(f64, Vec<Vec<f32>>), String> {
        let v = f()?;
        let mut t = Vec::new();
        for _ in 0..3 {
            let s = Instant::now();
            f()?;
            t.push(s.elapsed().as_secs_f64());
        }
        t.sort_by(f64::total_cmp);
        Ok((t[1], v))
    };
    let (t_old, old) = time(&per_group)?;
    let (t_new, new) = time(&packed)?;
    let diffs: Vec<f32> = old.iter().flatten().zip(new.iter().flatten()).map(|(a, b)| (a - b).abs()).collect();
    let worst = diffs.iter().copied().fold(0.0_f32, f32::max);
    let signs = old.iter().flatten().zip(new.iter().flatten()).filter(|(a, b)| (**a > 0.0) == (**b > 0.0)).count();
    // The deep check's value per piece: posed minus the mean of the three forms.
    let check = |v: &[Vec<f32>], c: usize, k: usize| v[4 * c][k] - (v[4 * c + 1][k] + v[4 * c + 2][k] + v[4 * c + 3][k]) / 3.0;
    let dc: Vec<String> = (0..2).flat_map(|c| (0..2).map(move |k| (c, k))).map(|(c, k)| format!("{:.3}/{:.3}", check(&old, c, k), check(&new, c, k))).collect();
    eprintln!(
        "8 groups (spans {spans:?} tokens, 16 tails): per-group pass_with {t_old:.3} s, packed read_group_labels {t_new:.3} s ({:.1} times); largest yes-minus-no difference {worst:.4}; signs the same on {signs} of {}; deep checks (per-group/packed) {dc:?}",
        t_old / t_new.max(1e-9),
        diffs.len()
    );
    // Lead's variant: one span a deep check (the posed header and the pieces), the three
    // content-free forms only in tails over it (the forms no longer in the span's header).
    let mut shared: Vec<(Vec<i32>, Vec<Vec<i32>>)> = Vec::new();
    for (q, ps) in &checks {
        let mut span = tok(&format!("{SYSTEM}Question: {q}

"))?;
        for (k, p) in ps.iter().enumerate() {
            span.extend(tok(&nmapi::joint_read::piece_text(k + 1, p))?);
        }
        let mut tails = Vec::new();
        for f in forms(q) {
            for k in 0..ps.len() {
                tails.push(tok(&format!("
Question: {f}
Does piece [{}] answer the question? Answer yes or no.{ASSISTANT}", k + 1))?);
            }
        }
        shared.push((span, tails));
    }
    let shared_run = || -> Result<Vec<Vec<f32>>, String> {
        let _served = nmengn::gpu::ServedScope::new();
        let r = nmengn::kvstore::read_group_labels(&m, None, &shared, &rows).map_err(|e| e.message)?;
        // Back to the per-group layout: [check * 4 + form][piece].
        let mut out = Vec::new();
        for g in &r {
            for f in 0..4 {
                out.push((0..2).map(|k| g[f * 2 + k][0] - g[f * 2 + k][1]).collect());
            }
        }
        Ok(out)
    };
    let (t_sh, sh) = time(&shared_run)?;
    let dsh: Vec<String> = (0..2).flat_map(|c| (0..2).map(move |k| (c, k))).map(|(c, k)| format!("{:.3}/{:.3}", check(&old, c, k), check(&sh, c, k))).collect();
    let posed_same = (0..2).all(|c| (0..2).all(|k| (old[4 * c][k] - sh[4 * c][k]).abs() < 0.25));
    eprintln!("shared span (forms only in tails): {t_sh:.3} s; posed tails equal within 0.25: {posed_same}; deep checks (per-group/shared) {dsh:?}");
    if worst > 0.25 || signs != diffs.len() {
        return Err(format!("packed label reads differ: largest {worst}, signs {signs} of {}", diffs.len()));
    }
    Ok(())
}
