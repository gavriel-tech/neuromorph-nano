//! E3 runs (design `docs/2026-11-10-neumor-design-e3-tentacles.md`, Luis Jake Gabriel III).
//! `reach`: on the CX1 development half, which tentacle windows hold the right code when eight
//! lenses each fill a 12,000-token window from the whole tree with no crossover
//! (`nmsubs::lens`), against one big window of the same total size from the lane's fused order,
//! one 12,000-token window, and the eight lenses filled without the no-crossover rule.
//! Selection only: nothing is encoded. Records hold question ids only.
use nmengn::placement::Device;
use nmsubs::lens::{self, BUDGET, NAMES};
use nmtype::{Fault, FaultKind};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("e3", FaultKind::Execution, m)
}

/// Fill one window from `order` up to `budget` tokens, skipping chunks too large for what is left.
fn fill(order: &[usize], tokens: &[usize], budget: usize) -> Vec<usize> {
    let mut used = 0;
    let mut out = Vec::new();
    for &c in order {
        let t = tokens.get(c).copied().unwrap_or(usize::MAX);
        if used + t <= budget {
            used += t;
            out.push(c);
        }
    }
    out
}

#[expect(clippy::too_many_lines, reason = "one run, in order")]
fn reach(home: &Path, want: Device) -> Result<(String, String), Fault> {
    let items = crate::e2::suite(home)?;
    let n = items.len();
    let mut out = String::new();
    let org = nmapi::forage::Organism::open_with(home, home, &["neuromorph-nano".to_owned()], false, &home.join("memory"))?;
    let sub = org.sub();
    let t0 = Instant::now();
    let (lane, line) = nmapi::neural::open_lane(home, &sub, want == Device::Gpu)?;
    let _ = writeln!(out, "{line} ({:.1} s)", t0.elapsed().as_secs_f64());
    let tok = nmtokz::Tokenizer::open(&nmtype::models_dir(home).join("qwen3-0.6b/files/tokenizer.json")).map_err(|e| bad(e.to_string()))?;
    let path = |i: usize| sub.files.get(sub.chunks[i].file).map_or_else(String::new, |f| f.path.replace('\\', "/"));
    let tokens: Vec<usize> = (0..sub.chunks.len())
        .map(|i| {
            let text =
                sub.files.get(sub.chunks[i].file).map_or_else(String::new, |f| nmsubs::dense::doc_text(&f.path, &sub.chunks[i].text));
            tok.encode(&text).map(|v| v.len()).unwrap_or(usize::MAX)
        })
        .collect();
    let total: usize = tokens.iter().filter(|t| **t != usize::MAX).sum();
    let _ = writeln!(
        out,
        "tree: {} chunks, {total} tokens; window {BUDGET} tokens per tentacle, {} in all",
        sub.chunks.len(),
        BUDGET * NAMES.len()
    );

    let (mut per_lens, mut union, mut big, mut one, mut free) = (vec![0_usize; NAMES.len()], 0_usize, 0_usize, 0_usize, 0_usize);
    let mut only = vec![0_usize; NAMES.len()];
    let (mut tok_lens, mut distinct_free) = (vec![0_usize; NAMES.len()], 0_usize);
    let mut js = String::from("{\"design\":\"docs/2026-11-10-neumor-design-e3-tentacles.md\",\"lenses\":[");
    let _ = writeln!(js, "{}],\"questions\":[", NAMES.iter().map(|x| format!("\"{x}\"")).collect::<Vec<_>>().join(","));
    for (qi, x) in items.iter().enumerate() {
        let right = |c: usize| {
            let (p, ch) = (path(c), &sub.chunks[c]);
            x.keys.iter().any(|(f, a, b)| *f == p && ch.start <= *b && ch.end >= *a)
        };
        let emb = lane.embedding_order(&sub, &x.q);
        let lists = lens::orders(&sub, &x.q, emb.as_deref(), &[]);
        let windows = lens::assign(&lists, &|c| tokens.get(c).copied().unwrap_or(usize::MAX), BUDGET);
        let hits: Vec<bool> = windows.iter().map(|w| w.iter().any(|c| right(*c))).collect();
        for (l, h) in hits.iter().enumerate() {
            per_lens[l] += usize::from(*h);
            tok_lens[l] += windows[l].iter().map(|c| tokens[*c]).sum::<usize>();
            if *h && hits.iter().filter(|y| **y).count() == 1 {
                only[l] += 1;
            }
        }
        let u = hits.iter().any(|h| *h);
        union += usize::from(u);
        let fused = lane.order_full(&sub, &x.q).ok_or_else(|| bad(format!("{}: no lane order", x.id)))?;
        let b = fill(&fused, &tokens, BUDGET * NAMES.len()).iter().any(|c| right(*c));
        let o = fill(&fused, &tokens, BUDGET).iter().any(|c| right(*c));
        big += usize::from(b);
        one += usize::from(o);
        // Without the no-crossover rule: each lens fills its own window from its own list.
        let mut seen = BTreeSet::new();
        let mut f = false;
        for l in &lists {
            let w = fill(l, &tokens, BUDGET);
            f |= w.iter().any(|c| right(*c));
            for c in w {
                if seen.insert(c) {
                    distinct_free += tokens[c];
                }
            }
        }
        free += usize::from(f);
        let _ = write!(
            js,
            "{}{{\"id\":\"{}\",\"lenses\":[{}],\"eight\":{u},\"one_big\":{b},\"one\":{o},\"eight_with_crossover\":{f}}}",
            if qi == 0 { "" } else { ",\n" },
            x.id,
            hits.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(",")
        );
    }
    js.push_str("\n]}\n");
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let mean = |t: usize| t as f64 / n.max(1) as f64;
    let _ = writeln!(out, "E3.2 reach on the CX1 development half: right code in a window, of {n}");
    let _ =
        writeln!(out, "eight lenses, no crossover         {union}/{n}  ({:.0} distinct tokens per question)", mean(tok_lens.iter().sum()));
    let _ = writeln!(out, "one window of {} (lane fused)   {big}/{n}", BUDGET * NAMES.len());
    let _ = writeln!(out, "one window of {BUDGET} (lane fused)    {one}/{n}");
    let _ = writeln!(out, "eight lenses with crossover        {free}/{n}  ({:.0} distinct tokens per question)", mean(distinct_free));
    for (l, name) in NAMES.iter().enumerate() {
        let _ = writeln!(out, "  lens {name:<10} {}/{n}, only this lens {}, mean {:.0} tokens", per_lens[l], only[l], mean(tok_lens[l]));
    }
    Ok((out, js))
}

/// `run e3 reach [gpu|cpu]`; `run e3 judge <stub|dev> [gpu|cpu]` (E3.3, `crate::e3j`).
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let part = args.get(2).map_or("reach", String::as_str);
    let (label, dev_arg) = match part {
        "reach" => ("reach".to_owned(), args.get(3)),
        "judge" if matches!(args.get(3).map(String::as_str), Some("stub" | "dev")) => (format!("judge-{}", args[3]), args.get(4)),
        _ => {
            eprintln!("run e3 reach [gpu|cpu] | run e3 judge <stub|dev> [gpu|cpu]");
            return ExitCode::FAILURE;
        }
    };
    let want = if dev_arg.map(String::as_str) == Some("cpu") { Device::Cpu } else { Device::Gpu };
    let stamp = nmtype::now_ms();
    let txt = home.join(format!("runs/E3-{label}-{stamp}.txt"));
    let result = if part == "reach" { reach(home, want) } else { crate::e3j::judge(home, label == "judge-stub", want) };
    match result {
        Ok((text, js)) => {
            let _ = std::fs::write(&txt, &text);
            let _ = std::fs::write(home.join(format!("runs/E3-{label}-{stamp}.json")), js);
            print!("{text}");
            println!("report: {}", txt.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = std::fs::write(
                &txt,
                format!(
                    "FAULT {e}
"
                ),
            );
            eprintln!("e3 {label}: {e} (report {})", txt.display());
            ExitCode::FAILURE
        }
    }
}
