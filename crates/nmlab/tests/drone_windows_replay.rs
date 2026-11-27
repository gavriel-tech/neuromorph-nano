//! Build step T1 by replay (2026-11-10; Gavriel Technologies, architecture by Luis Jake Gabriel
//! III): tentacle windows composed from drone flags plus relations (`nmapi::drone_windows`)
//! against today's lens windows, on the saved development scans. Processor only: the release
//! tokenizer and the token embedding table, no model run. Questions are named by index and id;
//! no question text is written. Encodings stay in RAM.
//!
//! Run: `cargo test --release -p nmapi --test drone_windows_replay -- --ignored --nocapture`
//! (`NM_T1_N=2` for the smoke run). Writes `runs/2026-11-10-T1-drone-windows-replay-<n>q-<ms>.txt`.
use nmapi::drone_windows::{Keep, compose};
use nmapi::scan_dump;
use nmsubs::lens::{BUDGET, NAMES};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

type R<T> = Result<T, String>;

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

/// Tokens, distinct files and overlap of a set of windows.
fn stats(sub: &nmsubs::Substrate, windows: &[Vec<usize>], tk: &dyn Fn(usize) -> usize) -> (usize, usize, usize) {
    let mut seen = BTreeSet::new();
    let overlap = windows.iter().flatten().filter(|c| !seen.insert(**c)).count();
    let tokens = seen.iter().map(|c| tk(*c)).sum();
    let files = seen.iter().filter_map(|c| sub.chunks.get(*c).map(|x| x.file)).collect::<BTreeSet<_>>().len();
    (tokens, files, overlap)
}

#[test]
#[ignore = "build step T1 replay: needs the saved scans and the release tokenizer and table; run by hand"]
#[expect(clippy::too_many_lines, reason = "one replay study with its report")]
fn drone_windows_t1() -> R<()> {
    nmapi::drone_windows::check()?;
    let home = nmtype::nano_root();
    let dir = home.join("scratch/tc1-dump/1794288995065");
    let n_want: usize = std::env::var("NM_T1_N").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let started = Instant::now();
    let items = dev_half(&home)?;
    let sub = scan_dump::load_substrate(&dir).map_err(|e| e.to_string())?;
    let nc = sub.chunks.len();
    // Chunk sizes as the live pass counts them: the document text (path line and chunk) by the
    // base tokenizer.
    let wdir = nmtype::models_dir(&home).join("qwen3-0.6b/files");
    let tok = nmtokz::Tokenizer::open(&wdir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let mut tokens = Vec::with_capacity(nc);
    for ch in &sub.chunks {
        let path = sub.files.get(ch.file).map_or("", |f| f.path.as_str());
        tokens.push(tok.encode(&nmsubs::dense::doc_text(path, &ch.text)).map_err(|e| e.to_string())?.len());
    }
    let sizes_s = t0.elapsed().as_secs_f64();
    let tk = |c: usize| tokens.get(c).copied().unwrap_or(usize::MAX);
    // The drone: intake of every chunk once (the code is unchanged across questions).
    let t1 = Instant::now();
    let mut keep = Keep::open(&wdir).map_err(|e| e.to_string())?;
    keep.sync(&sub).map_err(|e| e.to_string())?;
    let intake_s = t1.elapsed().as_secs_f64();
    let graph = sub.refs();

    let mut out = String::new();
    let _ = writeln!(
        out,
        "T1 drone windows by replay, processor only; dump 1794288995065; {nc} chunks, {} tokens; sizes {sizes_s:.2} s; drone open and intake {intake_s:.2} s (once while the code is unchanged)",
        tokens.iter().sum::<usize>()
    );
    let _ = writeln!(
        out,
        "mode --tentacle-windows drones: {} windows of {BUDGET} tokens, first fit from the drone order plus one-hop specific relations around strong flags (mean + 1 sd of the flag strengths); lens kinds not used for composition",
        NAMES.len()
    );
    let _ = writeln!(
        out,
        "q | id | drones: in some window | in window 1 | window | tokens | files | overlap | strong | related | not placed (referenced) | seconds || lens: in some window | tokens | files | overlap"
    );
    let (mut some, mut first, mut lens_some, mut n) = (0, 0, 0, 0_usize);
    let (mut sum_t, mut sum_f, mut sum_o, mut lt, mut lf, mut lo, mut secs) =
        (0_usize, 0_usize, 0_usize, 0_usize, 0_usize, 0_usize, 0.0_f64);
    let mut full = 0_usize;
    for idx in scan_dump::indices(&dir) {
        if n >= n_want {
            break;
        }
        let Some((id, text)) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(&dir, idx, text).map_err(|e| e.to_string())?;
        if &q.id != id {
            return Err(format!("dump question {idx} does not match the development half"));
        }
        let t = Instant::now();
        let fused = keep.sense(text).map_err(|e| e.to_string())?;
        let c = compose(&fused, &graph.specific, &tk, BUDGET, NAMES.len());
        let s = t.elapsed().as_secs_f64();
        secs += s;
        let win = c.windows.iter().position(|w| q.gold.iter().any(|g| w.contains(g)));
        let (dt, df, dov) = stats(&sub, &c.windows, &tk);
        full += c.windows.iter().filter(|w| w.iter().map(|x| tk(*x)).sum::<usize>() * 100 >= BUDGET * 95).count();
        let lens: Vec<Vec<usize>> = scan.windows.iter().map(|w| w.place.clone()).collect();
        let lin = lens.iter().any(|w| q.gold.iter().any(|g| w.contains(g)));
        let (ltk, lfi, lov) = stats(&sub, &lens, &tk);
        let placed: usize = c.windows.iter().map(Vec::len).sum();
        let _ = writeln!(
            out,
            "{idx} | {id} | {} | {} | {} | {dt} | {df} | {dov} | {} | {} | {} | {s:.2} || {} | {ltk} | {lfi} | {lov}",
            u8::from(win.is_some()),
            u8::from(win == Some(0)),
            win.map_or("-".to_owned(), |w| (w + 1).to_string()),
            c.strong,
            c.related,
            nc - placed,
            u8::from(lin)
        );
        some += usize::from(win.is_some());
        first += usize::from(win == Some(0));
        lens_some += usize::from(lin);
        sum_t += dt;
        sum_f += df;
        sum_o += dov;
        lt += ltk;
        lf += lfi;
        lo += lov;
        n += 1;
    }
    let k = n.max(1);
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let per = |x: usize| x as f64 / k as f64;
    let _ = writeln!(
        out,
        "summary over {n} questions: drones right chunk in some window {some} of {n}, in window 1 (strongest flag) {first} of {n}; mean tokens {:.0}, distinct files {:.1}, overlap total {sum_o}; windows at 95% of full or more {full} of {}",
        per(sum_t),
        per(sum_f),
        n * NAMES.len()
    );
    let _ = writeln!(
        out,
        "lens windows (today): right chunk in some window {lens_some} of {n}; mean tokens {:.0}, distinct files {:.1}, overlap total {lo}",
        per(lt),
        per(lf)
    );
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let mean = secs / k as f64;
    let _ = writeln!(
        out,
        "cost per question (sensing every chunk plus composition, processor): {mean:.2} s; one-time intake {intake_s:.2} s; whole test {:.1} s",
        started.elapsed().as_secs_f64()
    );
    print!("{out}");
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let p = home.join(format!("runs/2026-11-10-T1-drone-windows-replay-{n}q-{ms}.txt"));
    std::fs::write(&p, &out).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(())
}
