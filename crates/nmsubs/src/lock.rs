//! The code lock: proof that the code path is unchanged while documents are added. Over the code
//! files of a tree only (the files the format table reads as code), it hashes the sorted symbol
//! list, the chunk records (file, lines, content key), the crate list and the reference graph,
//! each built exactly as the substrate builds them. Documents never enter the lock, so a change to
//! how documents are read leaves it equal, and any change to how code is read breaks it.
use crate::{Substrate, from_files, refgraph::RefGraph, repo};
use std::path::Path;

/// One lock line per part: `name count sha256`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lock {
    pub lines: Vec<String>,
}

impl Lock {
    /// The lock as text, one part per line.
    #[must_use]
    pub fn text(&self) -> String {
        let mut s = self.lines.join("\n");
        s.push('\n');
        s
    }
}

fn part(name: &str, mut items: Vec<String>) -> String {
    items.sort();
    format!("{name} {} {}", items.len(), nmhash::sha256_bytes(items.join("\n").as_bytes()))
}

/// The lock of a substrate's code files and crate list.
#[must_use]
pub fn of_code(sub: &Substrate) -> Lock {
    let symbols = sub.symbols.iter().map(|s| format!("{}:{} {} {}", s.file, s.line, s.kind, s.name)).collect();
    let chunks = sub
        .chunks
        .iter()
        .map(|c| format!("{} {}-{} {}", sub.files.get(c.file).map_or("", |f| f.path.as_str()), c.start, c.end, c.key))
        .collect();
    let crates = sub.crates.iter().map(|c| format!("{} {} {}", c.name, c.dir, c.deps.join(","))).collect();
    let g = RefGraph::build(sub, None);
    let graph = vec![format!("names {} edges {}", g.names, g.edges), g.specific.describe(), g.keys.join(",")];
    Lock {
        lines: vec![
            format!("files {}", sub.files.len()),
            part("symbols", symbols),
            part("chunks", chunks),
            part("crates", crates),
            part("refgraph", graph),
        ],
    }
}

/// The lock of the code files under `root`: the Rust and TOML files the reader takes (git and
/// cargo are run as the substrate runs them).
#[must_use]
pub fn of_root(root: &Path) -> Lock {
    let root = repo::canon(root);
    let mut files = Vec::new();
    repo::walk(&root, &root, &[], &mut files);
    let code: Vec<(&str, &str)> =
        files.iter().filter(|f| f.path.ends_with(".rs") || f.path.ends_with(".toml")).map(|f| (f.path.as_str(), f.text.as_str())).collect();
    let mut sub = from_files(&code, std::sync::Arc::new(|_: &str| Vec::new()));
    sub.crates = repo::crates_of(&root);
    sub.root = root;
    of_code(&sub)
}
