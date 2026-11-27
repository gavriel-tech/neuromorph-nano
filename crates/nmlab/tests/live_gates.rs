//! Capability gates C1 and C9 on the live path (2026-11-10): the organism that answers the
//! coding agent over MCP (`nmapi::forage::Organism`, the `nmpod` units and the `nmbrain` Core
//! Brain), with the suites and thresholds of the self gates (`gates.rs`, which run the earlier
//! `nmcore` brain and stay as their record). Runbook `runs/2026-11-10-live-gates-runbook.md`.
//! Run: `cargo test --release --test live_gates -- --ignored --test-threads 1 --nocapture`.
#![expect(clippy::unwrap_used, reason = "gate harness")]
use nmapi::forage::Organism;
use std::path::PathBuf;

fn mini() -> PathBuf {
    nmtype::nano_root()
}
fn memory(name: &str) -> PathBuf {
    let p = nmtype::tests_dir(&format!("capability-gates/{name}"));
    let _ = std::fs::remove_dir_all(&p);
    p
}
fn report(name: &str, text: &str) {
    println!("{text}");
    std::fs::write(mini().join(format!("runs/live-gate-{name}-{}.txt", nmtype::now_ms())), text).unwrap();
}
/// The C1 suite: (question, file) pairs and the decoys.
fn suite() -> (Vec<(String, String)>, Vec<String>) {
    let v = nmjson::parse(&std::fs::read(mini().join("suites/mini-context.json")).unwrap()).unwrap();
    let qs = v
        .get("questions")
        .unwrap()
        .array()
        .unwrap()
        .iter()
        .map(|q| (q.get("q").unwrap().string().unwrap().to_owned(), q.get("file").unwrap().string().unwrap().to_owned()))
        .collect();
    let ds = v.get("decoys").unwrap().array().unwrap().iter().map(|d| d.string().unwrap().to_owned()).collect();
    (qs, ds)
}
/// Right: not withheld and a cited answer line starts with the target file (the scoring of the
/// live-path harnesses).
fn right(a: &nmapi::forage::Answer, file: &str) -> bool {
    !a.pack.withheld && a.emitted.text().lines().skip(1).any(|l| l.starts_with(file))
}
fn open(dir: &std::path::Path, memory_on: bool) -> Organism {
    Organism::open_with(&mini(), &mini(), &[], memory_on, dir).unwrap()
}

#[test]
#[ignore = "capability gate"]
fn live_c1_context() {
    let (qs, decoys) = suite();
    let org = open(&memory("c1-context/live-off"), false);
    let mut lines = Vec::new();
    let (mut ok, mut tokens, mut whole) = (0, 0_u64, 0_u64);
    for (q, f) in &qs {
        let a = org.ask(q, 4000).unwrap();
        let r = right(&a, f);
        ok += usize::from(r);
        tokens += u64::from(a.pack.tokens);
        let mut files: Vec<String> =
            a.pack.packets.iter().filter_map(|p| p.provenance().source.rsplit_once(':').map(|(x, _)| x.to_owned())).collect();
        files.sort();
        files.dedup();
        whole += files.iter().filter_map(|x| std::fs::metadata(mini().join(x)).ok()).map(|m| m.len() / 4).sum::<u64>();
        lines.push(format!("{} | {q} | withheld {} | tokens {}", if r { "ok" } else { "MISS" }, a.pack.withheld, a.pack.tokens));
    }
    let held = decoys.iter().filter(|d| org.ask(d, 4000).unwrap().pack.withheld).count();
    #[expect(clippy::cast_precision_loss, reason = "token counts")]
    let ratio = tokens as f64 / whole.max(1) as f64;
    let pass = ok >= 15 && held >= 2 && ratio <= 0.5;
    let head = format!(
        "live C1 context (mini-context.json, live organism, memory off): correct {ok}/{} (gate 15), decoys withheld {held}/{} (gate 2), tokens {tokens} vs whole cited files {whole} = {ratio:.3} (gate 0.5): {}",
        qs.len(),
        decoys.len(),
        if pass { "PASS" } else { "FAIL" }
    );
    report("c1", &format!("{head}\n{}\n", lines.join("\n")));
    assert!(pass, "{head}");
}

#[test]
#[ignore = "capability gate"]
fn live_c9_memory() {
    let (qs, decoys) = suite();
    // Off: no memory.
    let off_org = open(&memory("c9-memory-gain/live-off"), false);
    let off = qs.iter().filter(|(q, f)| right(&off_org.ask(q, 4000).unwrap(), f)).count();
    let off_held = decoys.iter().filter(|d| off_org.ask(d, 4000).unwrap().pack.withheld).count();
    drop(off_org);
    // On: one session asks every question and marks right answers verified (feedback simulated
    // from ground truth, as the self gate does); a new session on the same memory asks again.
    let dir = memory("c9-memory-gain/live-on");
    {
        let fill = open(&dir, true);
        for (q, f) in &qs {
            let a = fill.ask(q, 4000).unwrap();
            if let (Some(id), true) = (a.episode.as_ref(), right(&a, f)) {
                let _ = fill.memory.lock().unwrap().episodic.verdict(id, true);
            }
        }
        nmapi::session::close(&fill);
    }
    let on_org = open(&dir, true);
    let on = qs.iter().filter(|(q, f)| right(&on_org.ask(q, 4000).unwrap(), f)).count();
    let on_held = decoys.iter().filter(|d| on_org.ask(d, 4000).unwrap().pack.withheld).count();
    let pass = on >= off && on_held >= off_held;
    let head = format!(
        "live C9 memory (live organism): off {off}/{} correct, {off_held}/{} decoys withheld; on {on}/{}, {on_held}/{}; delta {:+}: {}",
        qs.len(),
        decoys.len(),
        qs.len(),
        decoys.len(),
        i64::try_from(on).unwrap() - i64::try_from(off).unwrap(),
        if pass { "PASS" } else { "FAIL" }
    );
    report("c9", &head);
    assert!(pass, "{head}");
}
