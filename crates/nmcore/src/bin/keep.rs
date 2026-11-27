//! `neuromorph-nano keep [--apply] [--json]`: the retention pass over the audit log's tool rows
//! (`nmmem::keep`): rows older than 28 run-days go, every other row stays. Without `--apply` it only
//! shows what a pass would remove; with it the pass runs and one `retention` row is written to the
//! audit log. The training store keeps itself by the same run-day rule each time it is settled; the
//! memory journal is never pruned.
use nmjson::Value;
use std::path::Path;
use std::process::ExitCode;

fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
fn n(x: u64) -> Value {
    Value::Number(x.to_string())
}

fn report_value(r: &nmmem::keep::Report) -> Value {
    let removals = r
        .removals
        .iter()
        .map(|x| {
            Value::Object(
                [
                    ("file".to_owned(), s(&x.file.display().to_string())),
                    ("rows".to_owned(), n(x.rows as u64)),
                    ("bytes".to_owned(), n(x.bytes)),
                    ("why".to_owned(), s(&x.why)),
                ]
                .into_iter()
                .collect(),
            )
        })
        .collect();
    Value::Object(
        [
            ("applied".to_owned(), Value::Bool(r.applied)),
            ("bytes_before".to_owned(), n(r.before)),
            ("bytes_after".to_owned(), n(r.after)),
            ("rows".to_owned(), n(r.rows() as u64)),
            ("removals".to_owned(), Value::Array(removals)),
        ]
        .into_iter()
        .collect(),
    )
}

pub fn run(home: &Path, args: &[String]) -> ExitCode {
    let apply = args.iter().any(|a| a == "--apply");
    let r = match nmmem::keep::pass_days(home, nmtype::now_ms(), apply, crate::controls::audit_keep_days(home)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("keep: {e}");
            return ExitCode::FAILURE;
        }
    };
    if apply && r.rows() > 0 {
        let label = format!("Retention pass: {} audit rows removed, {} bytes freed", r.rows(), r.bytes());
        let fields = vec![
            ("label", s(&label)),
            ("rows", n(r.rows() as u64)),
            ("bytes_before", n(r.before)),
            ("bytes_after", n(r.after)),
            ("removed", Value::Array(r.removals.iter().map(|x| s(&format!("{} rows {} bytes: {}", x.rows, x.bytes, x.why))).collect())),
        ];
        if let Err(e) = crate::controls::audit(home, "retention", fields) {
            eprintln!("keep: the pass ran, its audit row was not written: {e}");
        }
    }
    if args.iter().any(|a| a == "--json") {
        println!("{}", report_value(&r).encode().unwrap_or_default());
        return ExitCode::SUCCESS;
    }
    println!(
        "retention pass{}: audit log {} bytes{}",
        if apply { "" } else { " (shown only; --apply runs it)" },
        r.before,
        if apply { format!(", now {}", r.after) } else { format!(", {} after it", r.after) }
    );
    for x in &r.removals {
        println!("  {:>6} rows {:>10} bytes  {}  {}", x.rows, x.bytes, x.why, x.file.display());
    }
    if r.removals.is_empty() {
        println!("  nothing past its lifetime");
    }
    ExitCode::SUCCESS
}
