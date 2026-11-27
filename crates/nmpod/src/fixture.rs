//! Tiny in-memory views for the section checks of this crate (no substrate crate needed).
use nmjson::Value;
use nmtype::{Fault, Hit, Packet, Provenance, SlotView, Status, SubstrateView};

pub(crate) struct Mem {
    files: Vec<(String, String)>,
}
pub(crate) struct NoSlots;

pub(crate) fn views() -> (Mem, NoSlots) {
    let files = vec![
        ("a/x.rs".to_owned(), "pub const LIMIT: u8 = 4;\nfn go() {}\n".to_owned()),
        ("b/y.rs".to_owned(), "const LIMIT: u8 = 9;\n".to_owned()),
    ];
    (Mem { files }, NoSlots)
}

impl SubstrateView for Mem {
    fn version(&self) -> u64 {
        1
    }
    fn files(&self) -> Vec<String> {
        self.files.iter().map(|f| f.0.clone()).collect()
    }
    fn view(&self, name: &str, q: &str) -> Vec<Packet> {
        if name != "structural" {
            return Vec::new();
        }
        let mut out = Vec::new();
        for n in nmlex::names(q) {
            for h in self.symbols(&n) {
                let src = format!("{}:{}", h.file, h.line);
                let value = h.text.split_once('=').map_or("", |(_, v)| v.trim().trim_end_matches(';'));
                out.push(
                    Packet::new("symbol", &format!("value:{n}"), &format!("{n} = {value}"), Provenance::new(&src, "fixture", 1, 0))
                        .status(Status::Observed)
                        .evidence(vec![format!("{src} {}", h.text)]),
                );
            }
        }
        out
    }
    fn grep(&self, terms: &[String], paths: &[String], limit: usize) -> Vec<Hit> {
        let mut out = Vec::new();
        for (p, t) in self.files.iter().filter(|(p, _)| paths.is_empty() || paths.iter().any(|x| p.starts_with(x.as_str()))) {
            for (i, l) in t.lines().enumerate() {
                if terms.iter().all(|x| l.to_lowercase().contains(&x.to_lowercase())) {
                    out.push(Hit { file: p.clone(), line: i + 1, text: l.trim().to_owned() });
                }
            }
        }
        out.truncate(limit);
        out
    }
    fn symbols(&self, name: &str) -> Vec<Hit> {
        self.grep(&[format!("const {name}")], &[], 9)
    }
}
impl SlotView for NoSlots {
    fn call(&self, _: &str, _: &Value) -> Result<Value, Fault> {
        Ok(Value::String(String::new()))
    }
}
