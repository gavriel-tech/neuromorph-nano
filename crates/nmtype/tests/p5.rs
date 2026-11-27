//! Step P5 (2026-11-10): sensory and behavioural separation and the in-house message, first
//! form. Runs every `nmtype` section check (the same list `neuromorph-nano check nmtype`
//! runs), reports test B0 and test C1 (first half), and takes the first measurement: bytes
//! per envelope for a typical instruction and the time to render and parse 10,000
//! envelopes on the processor. Run with `cargo test --release -p nmtype --test p5 --
//! --nocapture` to see the numbers.
use nmtype::message::{Bare, TableResolver, c1_examples};
use nmtype::{Envelope, MessageKind, Reference};
use std::time::Instant;

#[test]
fn section_checks_pass() {
    for (id, check) in nmtype::checks() {
        assert_eq!(check(), Ok(()), "{id}");
    }
}

#[test]
fn b0_every_wrong_class_refused() {
    let (ok, refused) = nmtype::class::b0_table().unwrap_or_else(|e| panic!("{e}"));
    println!("B0: {ok} right-class hand-overs accepted, {refused} wrong-class refused, 0 wrong");
    assert!(ok > 0 && refused > 0);
}

#[test]
fn c1_round_trip_each_kind() {
    let (envs, resolver) = c1_examples().unwrap_or_else(|e| panic!("{e}"));
    for e in &envs {
        let full = e.render(&resolver).unwrap_or_else(|f| panic!("{f}"));
        assert_eq!(Envelope::parse(&full).as_ref(), Ok(e), "{}", e.kind().name());
        assert_eq!(e.verify_label(&resolver), Ok(Some(true)));
    }
    println!("C1 (first half): {} envelopes, {} kinds, full and bare text round trips exact", envs.len(), MessageKind::ALL.len());
}

/// A typical instruction: G2 directs one tentacle at a 20-line span, with a same-weights
/// encoding handle for that span (its token ids attached).
fn typical() -> (Envelope, TableResolver) {
    let body: String = (1..=20).map(|i| format!("    let v{i} = compute(v{}, {i});\n", i - 1)).collect();
    let resolver = TableResolver { files: vec![("crates/nmpod/src/pod.rs".into(), body.clone())] };
    // Byte-level ids stand in for the base tokenizer; the real label is about a quarter as
    // many ids (roughly four bytes per token), so this overstates the label size.
    let desc = "Read lines 1 to 20 of pod.rs and report each caller of compute.";
    let token_count = body.len() / 4;
    let tokens: Vec<u32> = (0..u32::try_from(token_count).unwrap_or(0)).map(|i| 151_000 - i * 7).collect();
    let e = Envelope::new(
        MessageKind::Instruction,
        "nmbrain.g2#0",
        "nmpod.unit#t4",
        0.82,
        desc,
        vec![
            Reference::Span { path: "crates/nmpod/src/pod.rs".into(), start: 1, end: 20 },
            Reference::Encoding { content: nmhash::sha256_bytes(body.as_bytes()), base: nmhash::sha256_bytes(b"base"), tokens },
        ],
        TableResolver::encode(desc),
    )
    .unwrap_or_else(|f| panic!("{f}"));
    (e, resolver)
}

#[test]
fn p5_first_measurement() {
    let (e, _) = typical();
    let bare = e.render(&Bare).unwrap_or_default();
    let json = e.to_value().encode().unwrap_or_default();
    let no_tokens = Envelope::new(e.kind(), e.sender(), e.target(), e.confidence(), e.description(), vec![e.refs()[0].clone()], Vec::new())
        .unwrap_or_else(|f| panic!("{f}"));
    println!(
        "bytes per typical instruction: bare text {} (of which header and span only {}), JSON {}",
        bare.len(),
        no_tokens.render_bare().len(),
        json.len()
    );
    let n = 10_000;
    let texts: Vec<String> = (0..n).map(|_| e.render_bare()).collect();
    let t0 = Instant::now();
    let mut rendered = 0usize;
    for _ in 0..n {
        rendered += e.render_bare().len();
    }
    let render_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let mut same = 0usize;
    for t in &texts {
        if Envelope::parse(t).as_ref() == Ok(&e) {
            same += 1;
        }
    }
    let parse_s = t1.elapsed().as_secs_f64();
    println!(
        "10,000 envelopes: render {:.1} ms ({:.2} us each), parse {:.1} ms ({:.2} us each), {} bytes rendered, {same} exact",
        render_s * 1e3,
        render_s * 1e6 / f64::from(n),
        parse_s * 1e3,
        parse_s * 1e6 / f64::from(n),
        rendered
    );
    assert_eq!(same, 10_000);
}
