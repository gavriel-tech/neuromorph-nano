//! G3 in the serving process (`serve --muscle on`): verdicts come from the evidence ledger and the
//! cycle runs at idle (`nmapi::g3::serve_idle_on`).
//!
//! Twenty real `nm_context_pack` calls go through the MCP handler on the small fixture repository
//! (processor only, memory on in a check folder). The test writes what the serving process's
//! observer and the hook's ledger would: one audit row per call carrying its configuration record,
//! and one verified `answered` outcome per call naming the place the reply cited first (plus one
//! refuted outcome and one for a pack never served, which must join nothing). The idle thread then
//! joins the verdicts and runs G3's cycle on its own; its replays write nothing to memory.
use nmapi::forage::Organism;
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::Path;

const QUESTIONS: [&str; 20] = [
    "Where is take_from_shelf defined?",
    "Where is total_units defined?",
    "Where is unit_price_cents defined?",
    "Where is checkout defined?",
    "Where is StockItem defined?",
    "Which function takes items from the shelf?",
    "Which function counts the total units?",
    "Which function gives the unit price in cents?",
    "Which function runs the checkout?",
    "Which struct holds a stock item?",
    "How does checkout compute the price?",
    "How does take_from_shelf decide it can take items?",
    "How does total_units add the stock?",
    "How does unit_price_cents pick a price?",
    "Why does take_from_shelf return false?",
    "Why does checkout return None?",
    "How does the shop crate use the inventory?",
    "Why does total_units return a u64?",
    "How does checkout change the stock item?",
    "Why does unit_price_cents take a code?",
];

/// Every file under `dir` with the hash of its content (sizes in a directory listing lag on
/// Windows while a file is held open for appending).
fn listing(dir: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.insert(p.display().to_string(), nmhash::sha256_bytes(&std::fs::read(&p).unwrap_or_default()));
            }
        }
    }
    out
}

fn outcome_row(pack: &str, verified: bool, place: &(String, usize, usize)) -> String {
    format!(
        "{{\"row\":\"outcome\",\"outcome\":\"answered:{pack}\",\"kind\":\"answered\",\"state\":\"{}\",\"strength\":\"direct\",\"rule\":\"edit\",\"session\":\"s1\",\"at\":1,\"places\":[{{\"path\":\"{}\",\"lines\":[{},{}],\"hash\":\"0\"}}]}}",
        if verified { "verified" } else { "refuted" },
        place.0,
        place.1,
        place.2
    )
}

#[test]
fn ledger_verdicts_reach_g3_and_the_idle_cycle_runs() {
    let home = nmtype::nano_root();
    let dir = nmtype::check_dir("nmapi-g3-ledger").expect("check folder");
    let memory = dir.join("memory");
    let root = home.join("tests").join("fixtures").join("small-repo");
    let org = Organism::open_with(&home, &root, &[], true, &memory).expect("organism opens");
    nmapi::tools::install(org).expect("install");
    let api = nmapi::tools::api();
    assert_eq!(nmapi::g3::enable(&home, Some(&dir.join("muscle"))).expect("muscle on"), 0);
    let audit = dir.join("audit.jsonl");
    let _ = std::fs::remove_file(&audit);
    assert!(nmapi::g3::serve_idle_on(Some(audit.clone())), "the idle cycle starts");

    // Twenty served calls through the MCP handler (each holds the gate), each with its audit row.
    let mut rows = String::new();
    let mut ledger = String::new();
    for (i, q) in QUESTIONS.iter().enumerate() {
        let line = format!("{{\"jsonrpc\":\"2.0\",\"id\":{i},\"method\":\"tools/call\",\"params\":{{\"name\":\"nm_context_pack\",\"arguments\":{{\"question\":\"{q}\"}}}}}}");
        let reply = nmapi::mcp::handle_line(&api, &line).expect("reply");
        let v = nmjson::parse(reply.as_bytes()).expect("json reply");
        let text = v.get("result").and_then(|r| r.get("content")).and_then(Value::array).ok().and_then(|c| c.first()).and_then(|c| c.get("text").ok()).and_then(|t| t.string().ok()).unwrap_or_default().to_owned();
        let d = nmapi::tools::take_diagnostics().expect("diagnostics");
        let call = d.get("config").and_then(|c| c.get("call")).and_then(Value::string).expect("configuration record").to_owned();
        let aid = format!("audit-{i}");
        rows.push_str(&format!("{{\"id\":\"{aid}\",\"kind\":\"tool\",\"tool\":\"nm_context_pack\",\"state\":\"done\",\"config\":{{\"call\":\"{call}\"}}}}\n"));
        let first = nmapi::g3::cited(&text).first().cloned().expect("the reply cites a place");
        ledger.push_str(&outcome_row(&aid, true, &first));
        ledger.push('\n');
    }
    // A refuted outcome and one for a pack never served join nothing.
    ledger.push_str(&outcome_row("audit-0", false, &("x.rs".to_owned(), 1, 1)));
    ledger.push('\n');
    ledger.push_str(&outcome_row("audit-99", true, &("x.rs".to_owned(), 1, 1)));
    ledger.push('\n');
    // Served calls finish their memory writes after the reply (`session::settle`, which the next
    // call would run); they land first, so only a replay's own writes would show.
    nmapi::session::settle(nmapi::tools::installed().expect("installed"));
    let before = listing(&memory);
    std::fs::write(&audit, rows).expect("audit rows");
    std::fs::write(memory.join("evidence.jsonl"), ledger).expect("ledger");

    // The idle thread joins the verdicts and runs the cycle on its own (it may already have run one
    // on the calls' own outcomes; the ledger's verdicts are joined at the next look).
    let t = std::time::Instant::now();
    let start = nmapi::g3::with(|st| st.cycles).unwrap_or(0);
    while nmapi::g3::verified_calls() < 20 || nmapi::g3::with(|st| st.cycles).unwrap_or(0) <= start {
        assert!(t.elapsed().as_secs() < 300, "no idle cycle within 300 s");
        std::thread::park_timeout(std::time::Duration::from_millis(200));
    }
    let verdicts = nmapi::g3::verified_calls();
    let cy = nmapi::g3::with(|st| st.last_cycle.clone()).expect("state");
    let cy = cy.expect("the cycle's report");
    eprintln!("idle cycle after {:.1} s: {verdicts} verdicts; {cy:?}", t.elapsed().as_secs_f64());
    assert_eq!(verdicts, 20, "every verified outcome joined its call, and nothing else");
    assert!(cy.replays > 0, "G3 replayed calls at idle");
    // Replays write nothing: memory holds what it held before the cycle, the ledger aside.
    let mut after = listing(&memory);
    after.retain(|k, _| !k.ends_with("evidence.jsonl"));
    let changed: Vec<String> = after.iter().filter(|(k, v)| before.get(*k) != Some(*v)).map(|(k, v)| format!("{k} {v}")).chain(before.keys().filter(|k| !after.contains_key(*k)).map(|k| format!("{k} removed"))).collect();
    assert!(changed.is_empty(), "a replay wrote to memory: {changed:?}");

    // A served call after the cycle still answers with its own configuration record.
    let line = r#"{"jsonrpc":"2.0","id":99,"method":"tools/call","params":{"name":"nm_context_pack","arguments":{"question":"Where is checkout defined?"}}}"#;
    assert!(nmapi::mcp::handle_line(&api, line).is_some());
    let d = nmapi::tools::take_diagnostics().expect("diagnostics");
    assert!(d.get("config").and_then(|c| c.get("call")).is_ok());
    let _ = nmapi::g3::disable();
}
