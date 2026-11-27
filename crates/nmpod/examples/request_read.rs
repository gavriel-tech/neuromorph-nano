//! Lane R (2026-11-18): print the request's drone read of each question in a file, one block per
//! question (`id<TAB>question` or `id<TAB>kind<TAB>question` lines). For the REQ-1 hand check.
//! usage: `cargo run -p nmpod --example request_read -- <file>`
fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: request_read <file>");
        return;
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("cannot read {path}");
        return;
    };
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (id, q) = match f.as_slice() {
            [id, q] | [id, _, q] => (*id, *q),
            _ => continue,
        };
        let r = nmpod::intake::Request::read(q, &[]);
        println!("{id} | {q}");
        println!("  kind {} | relations {}", r.kind, r.relations.iter().map(|a| a.name()).collect::<Vec<_>>().join(", "));
        for a in &r.aspects {
            let parts: Vec<String> = a.parts.iter().map(|p| format!("{}{}", p.text, if p.role == nmpod::intake::Role::Asked { "?" } else { "" })).collect();
            println!("  aspect [{}]", parts.join(" | "));
        }
    }
}
