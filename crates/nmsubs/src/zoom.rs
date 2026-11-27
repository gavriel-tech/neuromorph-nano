//! Context Zoom on the code axis (guide 5.1): the same subject at symbol, file, crate or
//! workspace level, in both directions. The anatomy axis (parameter cluster to organism)
//! would read the NTE snapshot and the NPCS binding table; it was planned with S5 and S6 and
//! is not built yet (2026-11-08).
use crate::Substrate;
use nmtype::{Packet, Provenance, Status};

/// Code-axis levels, finest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Symbol,
    File,
    Crate,
    Workspace,
}
impl Level {
    /// One step coarser (workspace stays).
    #[must_use]
    pub const fn out(self) -> Self {
        match self {
            Self::Symbol => Self::File,
            Self::File => Self::Crate,
            Self::Crate | Self::Workspace => Self::Workspace,
        }
    }
    /// One step finer (symbol stays).
    #[must_use]
    pub const fn r#in(self) -> Self {
        match self {
            Self::Workspace => Self::Crate,
            Self::Crate => Self::File,
            Self::File | Self::Symbol => Self::Symbol,
        }
    }
}

impl Substrate {
    fn zp(&self, kind: &str, key: &str, text: String, source: &str) -> Packet {
        Packet::new(kind, key, &text, Provenance::new(source, "nmsubs.zoom#g0", self.version, nmtype::now_ms()))
            .status(Status::Observed)
            .evidence(vec![source.to_owned()])
    }
    /// The subject at one level. A symbol name, a file path suffix or a crate name.
    #[must_use]
    pub fn zoom(&self, subject: &str, level: Level) -> Vec<Packet> {
        let q = subject.trim();
        match level {
            Level::Symbol => self
                .symbols
                .iter()
                .filter(|s| s.name == q)
                .map(|s| {
                    let src = format!("{}:{}", s.file, s.line);
                    self.zp("symbol", &format!("symbol:{}@{}", s.name, s.file), format!("{} {} at {src}: {}", s.kind, s.name, s.text), &src)
                })
                .collect(),
            Level::File => self
                .files
                .iter()
                .filter(|f| f.path.ends_with(q) || self.symbols.iter().any(|s| s.name == q && s.file == f.path))
                .take(3)
                .map(|f| {
                    let items: Vec<String> =
                        self.symbols.iter().filter(|s| s.file == f.path).map(|s| format!("{} {}:{}", s.kind, s.name, s.line)).collect();
                    self.zp(
                        "finding",
                        &format!("file:{}", f.path),
                        format!("{} ({} lines): {}", f.path, f.text.lines().count(), items.join("; ")),
                        &format!("{}:1", f.path),
                    )
                })
                .collect(),
            Level::Crate => {
                let by_file = self.symbols.iter().find(|s| s.name == q).map(|s| s.file.clone());
                let c = self.crate_named(q).or_else(|| {
                    let f = by_file?;
                    self.crates.iter().filter(|c| f.starts_with(&format!("{}/", c.dir))).max_by_key(|c| c.dir.len())
                });
                c.map(|c| {
                    let n = self.files.iter().filter(|f| f.path.starts_with(&format!("{}/", c.dir))).count();
                    vec![self.zp(
                        "deps",
                        &format!("crate:{}", c.name),
                        format!("crate {} at {} ({n} files), deps [{}]", c.name, c.dir, c.deps.join(", ")),
                        &format!("{}/Cargo.toml:1", c.dir),
                    )]
                })
                .unwrap_or_default()
            }
            Level::Workspace => self.view_of("global", q),
        }
    }
}

/// Self-check of the `zoom` sub-component: a symbol zooms out to its file and back in.
pub(crate) fn check() -> Result<(), String> {
    let embed: crate::Embed = std::sync::Arc::new(|_: &str| vec![1.0]);
    let s = crate::from_files(&[("a/x.rs", "pub struct Probe;\nfn go() {}\n")], embed);
    let sym = s.zoom("Probe", Level::Symbol);
    let file = s.zoom("Probe", Level::Symbol.out());
    if sym.len() != 1 || file.first().map(Packet::key) != Some("file:a/x.rs") || Level::File.r#in() != Level::Symbol {
        return Err("nmsubs.zoom: symbol to file and back".into());
    }
    Ok(())
}
