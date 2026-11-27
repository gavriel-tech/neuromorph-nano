//! Core Brain side wiring (2026-11-10, first form): checks of the `--brain-chains` option.
//! Gavriel Technologies; architecture by Luis Jake Gabriel III.
//!
//! The organism studies a fixed part of this mini (`crates/` without the crates other work is
//! changing today, `nmapi`, `nmengn`, `nmffi`, `nmcore`), with memory off, so the same question
//! gives the same pack on every run. Questions are written here from crate names and public
//! items of this repository, not from any locked test file. Run with `--test-threads 1`
//! (the check folders under `tests/section-checks/` are cleared per test).
use nmapi::brain_side::{self, BrainRecord, Config};
use nmapi::forage::{Answer, Organism};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

/// Varied questions over the four task kinds (locate, explain, edit, verify).
pub const QUESTIONS: [&str; 12] = [
    "Where is the function mine defined in the nmmusc crate?",
    "How does Codebook assign a code to a behavioural message?",
    "Change BehaviourLog so that set_outcome refuses a negative value",
    "Is it true that MuscleStore refuses a folder named memory?",
    "Where is Envelope render implemented in nmtype?",
    "Explain how the Settings version changes when a value is set",
    "Verify that ChainBook validate refuses a reasoning node as the output",
    "Where is Selector fit defined in the nmbrain g2 module?",
    "Add a field for the loaded state to Label in the nmlearn cartridge module",
    "What does extract_rule return in the preference module?",
    "Where is sha256_bytes defined?",
    "Check whether the nmjson parser rejects a trailing comma",
];

fn home() -> PathBuf {
    nmtype::nano_root()
}
/// Folders left out of the studied tree: work in progress elsewhere today.
pub fn skip() -> Vec<String> {
    ["nmapi", "nmengn", "nmffi", "nmcore"].iter().map(|s| (*s).to_owned()).collect()
}
fn open(name: &str) -> Organism {
    let dir = nmtype::check_dir(name).expect("check folder");
    Organism::open_with(&home(), &home().join("crates"), &skip(), false, &dir).expect("organism opens")
}
/// Everything of an answer that the option must leave unchanged when it is off: the pack
/// (withheld, reason, tokens, every packet's key, kind, text, evidence) and the answer text.
fn fingerprint(q: &str, a: &Answer) -> String {
    let mut s = format!("Q {q}\nwithheld {} tokens {} reason {}\n", a.pack.withheld, a.pack.tokens, a.pack.reason);
    for p in &a.pack.packets {
        s.push_str(&format!("P {} | {} | {} | {}\n", p.kind(), p.key(), p.text(), p.get_evidence().join(" ; ")));
    }
    s.push_str(&format!("A {}\n", a.emitted.text()));
    s
}
fn run_all(o: &Organism) -> String {
    QUESTIONS.iter().map(|q| fingerprint(q, &o.ask(q, 4000).expect("ask"))).collect()
}
fn baseline_path() -> PathBuf {
    home().join("scratch").join("brain-chains").join("baseline-off.txt")
}
/// Per question: (block text, packet keys).
fn blocks(t: &str) -> BTreeMap<String, (String, BTreeSet<String>)> {
    t.split("Q ")
        .skip(1)
        .map(|b| {
            let q = b.lines().next().unwrap_or("").to_owned();
            let keys = b.lines().filter_map(|l| l.strip_prefix("P ")).filter_map(|l| l.split(" | ").nth(1)).map(str::to_owned).collect();
            (q, (b.to_owned(), keys))
        })
        .collect()
}
/// Files of the studied tree changed after the baseline was written (by other work).
fn changed_since_baseline() -> BTreeSet<String> {
    let Ok(t0) = std::fs::metadata(baseline_path()).and_then(|m| m.modified()) else { return BTreeSet::new() };
    let root = home().join("crates");
    let sub = nmsubs::Substrate::scan(&root, &skip(), std::sync::Arc::new(|_: &str| Vec::new()));
    sub.files
        .iter()
        .filter(|f| std::fs::metadata(root.join(&f.path)).and_then(|m| m.modified()).is_ok_and(|m| m > t0))
        .map(|f| f.path.clone())
        .collect()
}

/// With the option off, packs and answers are byte-identical to the path before the change:
/// `NM_BRAIN_BASELINE=write` records the baseline (run before the first edit); otherwise each
/// question is compared with it. A question may differ only through packets citing files that
/// other work changed after the baseline was written (checked by file time, listed).
/// Also: two runs in one organism agree, and an organism whose option was switched on, used
/// and switched off again answers exactly as one that never had it.
#[test]
fn off_is_identical() {
    let org = open("nmapi-brain-off");
    let first = run_all(&org);
    let second = run_all(&org);
    assert_eq!(first, second, "two runs in one organism differ with the option off");
    let path = baseline_path();
    if std::env::var("NM_BRAIN_BASELINE").as_deref() == Ok("write") {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("scratch folder");
        std::fs::write(&path, &first).expect("baseline written");
        eprintln!("baseline written: {} bytes, {} questions", first.len(), QUESTIONS.len());
        return;
    }
    let other = open("nmapi-brain-off-b");
    let side_dir = nmtype::check_dir("nmapi-brain-off-side").expect("folder");
    brain_side::enable_with(
        &other,
        Config {
            muscle_dir: Some(side_dir.join("muscle")),
            pref_home: side_dir.clone(),
            uses_path: side_dir.join("uses.jsonl"),
            first_form_order: false,
        },
    )
    .expect("enable");
    let _ = other.ask(QUESTIONS[0], 4000).expect("ask on");
    if let Ok(mut g) = other.brain_side.write() {
        *g = None;
    }
    assert_eq!(first, run_all(&other), "switching the option on and off again left a trace");
    let Ok(base) = std::fs::read_to_string(&path) else {
        eprintln!("no baseline at {}; in-process checks only", path.display());
        return;
    };
    if base == first {
        eprintln!(
            "option off: byte-identical to the baseline on {} of {} questions ({} bytes)",
            QUESTIONS.len(),
            QUESTIONS.len(),
            base.len()
        );
        return;
    }
    let _ = std::fs::write(path.with_file_name("latest-off.txt"), &first);
    let changed = changed_since_baseline();
    let (a, b) = (blocks(&base), blocks(&first));
    let mut same = 0;
    for (q, (text, keys)) in &a {
        let (now, now_keys) = b.get(q).cloned().unwrap_or_default();
        if *text == now {
            same += 1;
            continue;
        }
        let moved: Vec<&String> = keys.symmetric_difference(&now_keys).collect();
        let explained = !moved.is_empty() && moved.iter().all(|k| changed.iter().any(|f| k.contains(f.as_str())));
        assert!(
            explained,
            "question {q:?} differs from the baseline and not only through files changed since ({changed:?}); moved packets {moved:?}"
        );
        eprintln!("question {q:?}: differs only through packets of files changed after the baseline by other work: {moved:?}");
    }
    eprintln!(
        "option off: byte-identical to the baseline on {same} of {} questions; the rest differ only through files changed since ({changed:?})",
        a.len()
    );
}

/// The module's section check (no organism).
#[test]
fn section_check() {
    brain_side::check().expect("nmapi.brain_side");
}

fn planted_pack(dir: &std::path::Path, base: &str) -> String {
    let mut log = nmmusc::BehaviourLog::default();
    let steps = [
        (nmmusc::MessageKind::Route, "nmbrain.chain#locate-code", "select:locate-code"),
        (nmmusc::MessageKind::Report, brain_side::CHAIN, "hop:router:jobs"),
        (nmmusc::MessageKind::Report, brain_side::CHAIN, "hop:embedder:near"),
        (nmmusc::MessageKind::Report, "tentacle:3", "focus:tests"),
    ];
    let codes: Vec<u32> =
        steps.iter().map(|(k, t, a)| log.record("plant", &nmmusc::Message::behavioural(*k, t, a)).expect("code")).collect();
    let rec = nmmusc::GainRecord { with_n: 3, with_mean: 0.8, without_n: 3, without_mean: 0.4, diff: 0.4, se: 0.1 };
    let pack = nmmusc::Pack::build(&codes, &log.codebook, "chain locate-code: planted check pack", base, rec).expect("pack");
    nmmusc::MuscleStore::open(dir).expect("store").write(&pack).expect("write");
    pack.header.fingerprint
}

/// With the option on: the chain's final output for each of the four task kinds; boundary
/// faults on planted items; one behaviour sequence per question; a planted pack reloaded and
/// offered; preferences fed and recalled into the choice hops; settings edits recognised as
/// own; first measurements (time added, chain per question, share of escalations).
#[test]
#[expect(clippy::too_many_lines, reason = "one end-to-end check with its measurements")]
fn on_runs_the_core_brain_side() {
    let off = open("nmapi-brain-on-control");
    let t = Instant::now();
    let off_answers: Vec<Answer> = QUESTIONS.iter().map(|q| off.ask(q, 4000).expect("ask off")).collect();
    let off_ms = t.elapsed().as_secs_f64() * 1000.0;

    let org = open("nmapi-brain-on");
    let dir = nmtype::check_dir("nmapi-brain-on-stores").expect("folder");
    let base = nmlearn::cartridge::Loaded::population(&home(), brain_side::BASE, Vec::new()).map(|l| l.base_sha256).unwrap_or_default();
    let fp = planted_pack(&dir.join("muscle"), &base);
    let note = brain_side::enable_with(
        &org,
        Config {
            muscle_dir: Some(dir.join("muscle")),
            pref_home: dir.clone(),
            uses_path: dir.join("cartridge-uses.jsonl"),
            first_form_order: false,
        },
    )
    .expect("enable");
    eprintln!("{note}");
    let side = brain_side::side(&org).expect("side on");
    assert!(side.packs.iter().any(|(f, _)| *f == fp), "planted pack not loaded at session start");

    let t = Instant::now();
    let answers: Vec<Answer> = QUESTIONS.iter().map(|q| org.ask(q, 4000).expect("ask on")).collect();
    let on_ms = t.elapsed().as_secs_f64() * 1000.0;
    let recs: Vec<&BrainRecord> = answers.iter().map(|a| a.brain.as_ref().expect("record with the option on")).collect();
    let mut lines = Vec::new();
    for ((q, a), (r, o)) in QUESTIONS.iter().zip(&answers).zip(recs.iter().zip(&off_answers)) {
        // The pack is the same as without the option; only the answer text may change.
        assert_eq!(fingerprint(q, o).split("\nA ").next(), fingerprint(q, a).split("\nA ").next(), "the option changed the pack");
        if r.upstream > 0 {
            assert!(!r.final_output.is_empty(), "no final output for {q}");
        }
        if r.answer_replaced {
            let head = o.emitted.text().lines().next().unwrap_or("");
            assert_eq!(
                a.emitted.text(),
                brain_side::answer_text(head, &r.final_output, &r.citations),
                "final output and its citations are not the answer text"
            );
            assert!(!r.citations.is_empty(), "the final output carries no citations for {q}");
        } else {
            assert_eq!(a.emitted.text(), o.emitted.text(), "answer changed although the output was not used");
        }
        // The chain log never enters the answer text.
        assert!(!a.emitted.text().contains("hop "), "a hop report reached the answer");
        lines.push(format!(
            "{} | G2 {} conf {:.3} thr {:.3}{} | ran {} ({}) | up {} held {} | out {} lines | replaced {} | {:.3} ms",
            q,
            r.g2_choice,
            r.confidence,
            r.threshold,
            if r.escalated { " ESCALATE" } else { "" },
            r.kind,
            r.mode,
            r.upstream,
            r.held_back,
            r.final_output.len(),
            r.answer_replaced,
            r.ms
        ));
    }
    // One behaviour sequence per question.
    let traces = side.traces();
    assert_eq!(traces.len(), QUESTIONS.len(), "behaviour log sequences");
    assert!(traces.iter().all(|(_, k, _)| *k > 0), "an empty sequence");
    assert_eq!(traces.iter().map(|(id, _, _)| id.clone()).collect::<BTreeSet<_>>().len(), QUESTIONS.len());
    let symbols: usize = traces.iter().map(|(_, k, _)| k).sum();

    // Every task kind produces its chain's final output (G2's parts, the chain named).
    let a0 = &answers[0];
    for kind in nmbrain::chain::REQUIRED {
        let (_, r) = side.question(&org, QUESTIONS[0], &a0.pack, &a0.emitted, Some(kind)).expect("forced chain");
        assert_eq!(r.kind, kind);
        assert!(!r.final_output.is_empty(), "{kind}: no final output");
        assert_eq!(r.log.len(), side.book.get(kind).expect("chain").nodes.len());
        lines.push(format!(
            "forced {kind}: {} hops, output {:?}",
            r.log.len(),
            r.final_output.iter().map(|l| l.chars().take(60).collect::<String>()).collect::<Vec<_>>()
        ));
        if kind == "locate-code" {
            let o = r.offers.iter().find(|o| o.pack == fp).expect("planted pack offered for locate-code");
            assert_eq!(o.proposed, 4);
            assert_eq!((o.accepted, o.declined), (3, 1), "G2's decision on the planted steps");
            assert!(r.steps.iter().any(|e| e.description().starts_with("accept pack")), "G2's decision not recorded");
        }
    }

    // Boundary faults on planted cases.
    let sensory = a0.pack.packets.first().expect("a packet").clone();
    let err = side
        .record(&recs[0].question_id, &sensory, nmmusc::MessageKind::Report, "x", "y")
        .expect_err("sensory admitted to the behaviour log");
    assert!(err.message.starts_with(nmtype::WRONG_CLASS), "{err}");
    let mut bare_log = nmmusc::BehaviourLog::default();
    let e2 = bare_log
        .record("q", &brain_side::Symbol { item: &sensory, kind: nmmusc::MessageKind::Report, target: "x".into(), action: "y".into() })
        .expect_err("codebook took sensory");
    assert_eq!(e2.code, "nmmusc-sensory-refused");
    assert!(nmtype::class::boundaries::MUSCLE_MEMORY.admit(&sensory).is_err());
    for env in &recs[0].steps {
        assert!(nmtype::class::boundaries::CHAIN_INPUT.admit(env).is_err(), "behavioural step admitted as chain input");
        assert!(nmtype::class::boundaries::ANSWER_EVIDENCE.admit(env).is_err(), "behavioural step admitted as answer evidence");
    }
    let behavioural = nmtype::Packet::new_behavioural("question", "k", "t", nmtype::Provenance::new("x", "y", 1, 0));
    let mut planted = a0.pack.clone();
    planted.packets.push(behavioural);
    assert!(
        side.question(&org, QUESTIONS[0], &planted, &a0.emitted, None).is_err(),
        "a behavioural packet passed the chain-input boundary"
    );
    assert_eq!(side.traces().len(), QUESTIONS.len() + nmbrain::chain::REQUIRED.len(), "a refused question left a sequence");

    // Settings and action copies: one own edit per question run, recognised by G1 and G2.
    let stats = side.stats();
    assert_eq!(stats.own_edits_recognised, stats.questions);
    let set = recs[0].steps.iter().find(|e| e.target() == "broadcast").expect("settings edit");
    assert!(side.is_own(set));
    let foreign = nmtype::Envelope::new(
        nmtype::MessageKind::Instruction,
        "outside#0",
        "broadcast",
        1.0,
        "set chain_in_use to x",
        Vec::new(),
        brain_side::label_tokens("set chain_in_use to x"),
    )
    .expect("env");
    assert!(!side.is_own(&foreign), "a new input read as own");
    let edit = side
        .applied_edit("applied proposed patch", vec![nmtype::Reference::Span { path: "nmmusc/src/log.rs".into(), start: 60, end: 64 }])
        .expect("edit");
    assert!(side.is_own(&edit) && side.is_self_caused("nmmusc/src/log.rs", 62) && !side.is_self_caused("nmmusc/src/log.rs", 65));

    // Preferences (2026-11-12, signal router): a right or wrong verdict attaches the outcome and
    // never feeds preferences; a taste signal feeds them; the next question's choice hops get the rules.
    side.bind_episode("ep-check-1", &recs[0].question_id);
    let fb = side.on_feedback("ep-check-1", true);
    assert_eq!(fb.get("outcome_attached").ok(), Some(&nmjson::Value::Bool(true)), "{fb:?}");
    assert_eq!(fb.get("preference_logged").ok(), Some(&nmjson::Value::Bool(false)), "a verdict reached preferences: {fb:?}");
    let taste = side.on_taste("ep-check-1", true);
    assert_eq!(taste.get("preference_logged").ok(), Some(&nmjson::Value::Bool(true)), "{taste:?}");
    assert!(side.traces().iter().any(|(id, _, o)| *id == recs[0].question_id && *o == Some(1.0)));
    let again = org.ask(QUESTIONS[7], 4000).expect("ask");
    let r = again.brain.as_ref().expect("record");
    assert!(!r.preferences.is_empty(), "no rule recalled after feedback");
    assert!(
        r.log
            .iter()
            .filter(|h| brain_side::CHOICE_ROLES.contains(&h.role.as_str()))
            .all(|h| h.takes.iter().any(|t| t == brain_side::PREF_FIELD)),
        "rules not in a choice hop's input"
    );

    // G3 mining on request; trace rendering.
    let mined = side.mine_and_promote().expect("mine");
    let trace = side.trace_lines(&org.sub(), recs[0]);
    assert!(trace.iter().filter(|l| l.starts_with("nm-msg/1")).count() == recs[0].steps.len(), "every step renders");
    assert!(trace.iter().any(|l| l.contains("text ")), "no resolved block in the rendering");

    // First measurements.
    let esc = recs.iter().filter(|r| r.escalated).count();
    // Answer agreement with the control path: the control answer's first cited line kept
    // first, or kept anywhere, by the chain's output (only where the output was used).
    let cited = |a: &Answer| a.emitted.text().lines().nth(1).map(str::to_owned);
    let used: Vec<(Option<String>, &BrainRecord)> =
        off_answers.iter().zip(&recs).filter(|(_, r)| r.answer_replaced).map(|(o, r)| (cited(o), *r)).collect();
    let first_same = used.iter().filter(|(c, r)| c.is_some() && r.final_output.first() == c.as_ref()).count();
    let kept = used.iter().filter(|(c, r)| c.as_ref().is_some_and(|c| r.final_output.contains(c))).count();
    // The same against the chain's answer text (final output plus its citations): the first
    // cited line of the answer after its head line.
    let answers_used: Vec<(Option<String>, Option<String>)> = off_answers
        .iter()
        .zip(&answers)
        .zip(&recs)
        .filter(|(_, r)| r.answer_replaced)
        .map(|((o, a), _)| (cited(o), a.emitted.text().lines().skip(1).find(|l| l.contains(':')).map(str::to_owned)))
        .collect();
    let answer_first = answers_used.iter().filter(|(c, f)| c.is_some() && c == f).count();
    let held: usize = recs.iter().flat_map(|r| r.log.iter()).filter(|h| h.order == nmbrain::chain::OrderOutcome::Held).count();
    let reordered: usize = recs.iter().flat_map(|r| r.log.iter()).filter(|h| h.order == nmbrain::chain::OrderOutcome::Reordered).count();
    lines.push(format!("answers replaced {} of {}; control's first cited line first in the chain output {first_same}, anywhere in it {kept}; first cited line of the chain's answer text {answer_first}; ranking hops held {held}, reordered {reordered}", used.len(), recs.len()));
    assert_eq!(reordered, 0, "an exact ranking node with no track record reordered its input");
    // The first-form ordering on the same tree and questions (no order contract), for the
    // before and after comparison.
    let ff = open("nmapi-brain-on-first-form");
    let ff_dir = nmtype::check_dir("nmapi-brain-on-first-form-stores").expect("folder");
    brain_side::enable_with(
        &ff,
        Config {
            muscle_dir: Some(ff_dir.join("muscle")),
            pref_home: ff_dir.clone(),
            uses_path: ff_dir.join("uses.jsonl"),
            first_form_order: true,
        },
    )
    .expect("enable first form");
    let ff_answers: Vec<Answer> = QUESTIONS.iter().map(|q| ff.ask(q, 4000).expect("ask first form")).collect();
    let ff_used: Vec<(Option<String>, &BrainRecord, &Answer)> = off_answers
        .iter()
        .zip(&ff_answers)
        .filter_map(|(o, a)| a.brain.as_ref().filter(|r| r.answer_replaced).map(|r| (cited(o), r, a)))
        .collect();
    let ff_first = ff_used.iter().filter(|(c, r, _)| c.is_some() && r.final_output.first() == c.as_ref()).count();
    let ff_kept = ff_used.iter().filter(|(c, r, _)| c.as_ref().is_some_and(|c| r.final_output.contains(c))).count();
    let ff_answer_first = ff_used
        .iter()
        .filter(|(c, _, a)| c.is_some() && a.emitted.text().lines().skip(1).find(|l| l.contains(':')).map(str::to_owned) == *c)
        .count();
    lines.push(format!("first-form ordering, same tree: answers replaced {}; control's first cited line first in the chain output {ff_first}, anywhere in it {ff_kept}; first cited line of the chain's answer text {ff_answer_first}", ff_used.len()));
    assert!(
        first_same >= ff_first && answer_first >= ff_answer_first,
        "the order contract kept the control's best line first less often than the first form"
    );
    let side_ms: f64 = recs.iter().map(|r| r.ms).sum();
    let mut by_kind: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &recs {
        *by_kind.entry(r.kind.as_str()).or_default() += 1;
    }
    let report = format!(
        "{}\n{}\nquestions {}; ask total off {off_ms:.1} ms, on {on_ms:.1} ms; Core Brain side {side_ms:.2} ms total, {:.3} ms per question (G2 mean {:.3}, chain mean {:.3}); escalations {esc} of {} ({:.1}%); chains run {by_kind:?}; behaviour symbols {symbols} ({:.1} per question); settings v{}; mining {}\n--- trace of question 1\n{}\n",
        note,
        lines.join("\n"),
        QUESTIONS.len(),
        side_ms / recs.len() as f64,
        recs.iter().map(|r| r.g2_ms).sum::<f64>() / recs.len() as f64,
        recs.iter().map(|r| r.chain_ms).sum::<f64>() / recs.len() as f64,
        recs.len(),
        100.0 * esc as f64 / recs.len() as f64,
        symbols as f64 / QUESTIONS.len() as f64,
        side.settings().version(),
        mined.encode().unwrap_or_default(),
        trace.join("\n")
    );
    let out = nmtype::check_dir("nmapi-brain-measure").expect("folder").join("measure.txt");
    let _ = std::fs::write(&out, &report);
    eprintln!("{report}");
}
