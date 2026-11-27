//! Content-addressed binary substrate cache (`substrate-cache.bin`, Issue C, 2026-11-22).
//!
//! Caches a scanned `Substrate` inside the project's `memory/` folder keyed by a fast
//! metadata-only SHA-256 fingerprint of every walked file (path, size, mtime), `.git/HEAD`,
//! `skip` list, and segmentation mode. On an unchanged repository, cold startup drops from
//! 10-32 s down to under ~200 ms with bit-identical `files`, `symbols`, `chunks`, `df`,
//! `imports`, `crates`, and `history`. Any added, deleted, or modified file invalidates the
//! fingerprint and falls back cleanly to a fresh `Substrate::scan`. Standard library + workspace
//! crates only; zero `unsafe`.
use crate::imports::Edge;
use crate::piece::Meta;
use crate::repo::{self, Crate, FileInfo, Skipped, Symbol};
use crate::tags::Import;
use crate::{Chunk, Embed, Substrate};
use nmtype::doc::{Block, BlockKind, Loc};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"NMSUBS01";
const FORMAT_VERSION: u32 = 1;

/// Whether the binary substrate cache is enabled (on by default; `NANO_SUBSTRATE_CACHE=off` turns
/// it off).
#[must_use]
pub fn enabled() -> bool {
    !std::env::var("NANO_SUBSTRATE_CACHE").is_ok_and(|v| v == "off")
}

fn config_tag() -> String {
    let cut = match crate::doc::cut() {
        crate::doc::Cut::Blocks(b) => format!("blocks:{}:{}", b.soft, b.hard),
        crate::doc::Cut::Windows => "windows".to_owned(),
    };
    // `binary:meta`: binary files are kept by their metadata only (`nmdoc::raw::holds_text`), so a
    // cache written by a reader that took text runs from them is not restored.
    format!("v{FORMAT_VERSION}|{cut}|piece:{}|struct:{}|binary:meta", crate::piece::on(), crate::STRUCTURAL_SEGMENTS)
}

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { buf: Vec::with_capacity(256 * 1024) }
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        let b = s.as_bytes();
        self.usize(b.len());
        self.buf.extend_from_slice(b);
    }
    fn opt_str(&mut self, s: Option<&str>) {
        match s {
            Some(v) => {
                self.u8(1);
                self.str(v);
            }
            None => self.u8(0),
        }
    }
    fn opt_usize(&mut self, v: Option<usize>) {
        match v {
            Some(x) => {
                self.u8(1);
                self.usize(x);
            }
            None => self.u8(0),
        }
    }
    fn str_vec(&mut self, xs: &[String]) {
        self.usize(xs.len());
        for s in xs {
            self.str(s);
        }
    }
    fn block(&mut self, b: &Block) {
        match b.kind {
            BlockKind::Heading(l) => {
                self.u8(0);
                self.u8(l);
            }
            BlockKind::Paragraph => self.u8(1),
            BlockKind::ListItem { depth, ordered } => {
                self.u8(2);
                self.u8(depth);
                self.u8(u8::from(ordered));
            }
            BlockKind::Table { head } => {
                self.u8(3);
                self.u16(head);
            }
            BlockKind::Code => self.u8(4),
            BlockKind::Caption => self.u8(5),
            BlockKind::Quote => self.u8(6),
            BlockKind::Note => self.u8(7),
            BlockKind::Definition => self.u8(8),
            BlockKind::FrontMatter => self.u8(9),
            BlockKind::Record => self.u8(10),
            BlockKind::LogEvent => self.u8(11),
            BlockKind::Item => self.u8(12),
            BlockKind::Other => self.u8(13),
        }
        self.usize(b.first);
        self.usize(b.last);
        self.opt_usize(b.parent);
        self.str(&b.label);
        match b.loc.page {
            Some((a, c)) => {
                self.u8(1);
                self.u32(a);
                self.u32(c);
            }
            None => self.u8(0),
        }
        self.opt_str(b.loc.page_label.as_deref());
        self.opt_str(b.loc.cells.as_deref());
        match b.loc.time {
            Some((a, c)) => {
                self.u8(1);
                self.u64(a);
                self.u64(c);
            }
            None => self.u8(0),
        }
        self.opt_str(b.loc.anchor.as_deref());
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let slice = self.data.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }
    fn u8(&mut self) -> Option<u8> {
        self.bytes(1).map(|b| b[0])
    }
    fn u16(&mut self) -> Option<u16> {
        let b: [u8; 2] = self.bytes(2)?.try_into().ok()?;
        Some(u16::from_le_bytes(b))
    }
    fn u32(&mut self) -> Option<u32> {
        let b: [u8; 4] = self.bytes(4)?.try_into().ok()?;
        Some(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Option<u64> {
        let b: [u8; 8] = self.bytes(8)?.try_into().ok()?;
        Some(u64::from_le_bytes(b))
    }
    fn usize(&mut self) -> Option<usize> {
        usize::try_from(self.u64()?).ok()
    }
    fn f32(&mut self) -> Option<f32> {
        let b: [u8; 4] = self.bytes(4)?.try_into().ok()?;
        Some(f32::from_le_bytes(b))
    }
    fn str_ref(&mut self) -> Option<&'a str> {
        let n = self.usize()?;
        std::str::from_utf8(self.bytes(n)?).ok()
    }
    fn string(&mut self) -> Option<String> {
        self.str_ref().map(str::to_owned)
    }
    fn opt_string(&mut self) -> Option<Option<String>> {
        match self.u8()? {
            0 => Some(None),
            1 => Some(Some(self.string()?)),
            _ => None,
        }
    }
    fn opt_usize(&mut self) -> Option<Option<usize>> {
        match self.u8()? {
            0 => Some(None),
            1 => Some(Some(self.usize()?)),
            _ => None,
        }
    }
    fn str_vec(&mut self) -> Option<Vec<String>> {
        let n = self.usize()?;
        let mut out = Vec::with_capacity(n.min(65_536));
        for _ in 0..n {
            out.push(self.string()?);
        }
        Some(out)
    }
    fn block(&mut self) -> Option<Block> {
        let kind = match self.u8()? {
            0 => BlockKind::Heading(self.u8()?),
            1 => BlockKind::Paragraph,
            2 => BlockKind::ListItem { depth: self.u8()?, ordered: self.u8()? != 0 },
            3 => BlockKind::Table { head: self.u16()? },
            4 => BlockKind::Code,
            5 => BlockKind::Caption,
            6 => BlockKind::Quote,
            7 => BlockKind::Note,
            8 => BlockKind::Definition,
            9 => BlockKind::FrontMatter,
            10 => BlockKind::Record,
            11 => BlockKind::LogEvent,
            12 => BlockKind::Item,
            13 => BlockKind::Other,
            _ => return None,
        };
        let first = self.usize()?;
        let last = self.usize()?;
        let parent = self.opt_usize()?;
        let label = self.string()?;
        let page = match self.u8()? {
            0 => None,
            1 => Some((self.u32()?, self.u32()?)),
            _ => return None,
        };
        let page_label = self.opt_string()?;
        let cells = self.opt_string()?;
        let time = match self.u8()? {
            0 => None,
            1 => Some((self.u64()?, self.u64()?)),
            _ => return None,
        };
        let anchor = self.opt_string()?;
        Some(Block { kind, first, last, parent, label, loc: Loc { page, page_label, cells, time, anchor } })
    }
}

fn intern_decoder(s: &str) -> Option<&'static str> {
    if let Some(l) = nmlex::lang::by_id(s) {
        return Some(l.id);
    }
    Some(match s {
        "rust" => "rust",
        "toml" => "toml",
        "code" => "code",
        "cuda" => "cuda",
        "go" => "go",
        "python" => "python",
        "typescript" => "typescript",
        "markdown" => "markdown",
        "text" => "text",
        "text?" => "text?",
        "csv" => "csv",
        "tsv" => "tsv",
        "html" => "html",
        "xml" => "xml",
        "json" => "json",
        "jsonl" => "jsonl",
        "notebook" => "notebook",
        "yaml" => "yaml",
        "log" => "log",
        "pdf" => "pdf",
        "docx" => "docx",
        "xlsx" => "xlsx",
        "pptx" => "pptx",
        "rtf" => "rtf",
        "odt" => "odt",
        "epub" => "epub",
        "general" => "general",
        _ => return None,
    })
}

fn intern_general(s: &str) -> Option<&'static str> {
    Some(match s {
        "container" => "container",
        "member" => "member",
        "gzip" => "gzip",
        "raw" => "raw",
        "binary" => "binary",
        _ => return None,
    })
}

fn intern_chunk_kind(s: &str) -> Option<&'static str> {
    Some(match s {
        "window" => "window",
        "heading" => "heading",
        "paragraph" => "paragraph",
        "list" => "list",
        "table" => "table",
        "code" => "code",
        "caption" => "caption",
        "quote" => "quote",
        "note" => "note",
        "definition" => "definition",
        "front-matter" => "front-matter",
        "record" => "record",
        "log" => "log",
        "item" => "item",
        "other" => "other",
        "block" => "block",
        "impl" => "impl",
        "schema" => "schema",
        "fn" => "fn",
        "function" => "function",
        "method" => "method",
        "struct" => "struct",
        "enum" => "enum",
        "trait" => "trait",
        "type" => "type",
        "const" => "const",
        "static" => "static",
        "mod" => "mod",
        "macro" => "macro",
        "class" => "class",
        "interface" => "interface",
        "imports" => "imports",
        "notes" => "notes",
        "group" => "group",
        "file" => "file",
        "body" => "body",
        _ => return None,
    })
}

fn intern_ecosystem(s: &str) -> Option<&'static str> {
    Some(match s {
        "cargo" => "cargo",
        "npm" => "npm",
        "python" => "python",
        _ => return None,
    })
}

fn intern_skip_reason(s: &str) -> Option<&'static str> {
    Some(match s {
        "unreadable" => "unreadable",
        "ignored by .gitignore" => repo::IGNORED,
        "build cache (generated): not read" => "build cache (generated): not read",
        "compiled kernel assembly (generated): not read" => "compiled kernel assembly (generated): not read",
        "generated file (marked in its first lines): not read" => "generated file (marked in its first lines): not read",
        "lock file (generated): not read" => "lock file (generated): not read",
        "minified file (generated): not read" => "minified file (generated): not read",
        "model weights or binary data: not read" => "model weights or binary data: not read",
        "Neuromorph runtime state: not read" => "Neuromorph runtime state: not read",
        "secret store (.env): never read" => "secret store (.env): never read",
        "secret store (keys or credentials): never read" => "secret store (keys or credentials): never read",
        "source map (generated): not read" => "source map (generated): not read",
        "too large: over 512 KB" => "too large: over 512 KB",
        "too large: over 64 MB" => "too large: over 64 MB",
        "unknown format, one long line of data: not read" => "unknown format, one long line of data: not read",
        "binary data: not read" => "binary data: not read",
        "empty" => "empty",
        "hidden file" => "hidden file",
        _ => return None,
    })
}

/// Attempts to load a cached `Substrate` from `cache_path` if its metadata manifest fingerprint
/// matches the current working tree at `root` identically. Returns `None` on miss or mismatch.
#[must_use]
pub fn load(root: &Path, skip: &[String], embed: &Embed, cache_path: &Path) -> Option<(Substrate, String)> {
    let root_canon = repo::canon(root);
    let fp = repo::manifest_fingerprint(&root_canon, skip, &config_tag());
    let data = std::fs::read(cache_path).ok()?;
    let mut r = Reader::new(&data);
    if r.bytes(8)? != MAGIC || r.u32()? != FORMAT_VERSION {
        return None;
    }
    if r.str_ref()? != fp {
        return None;
    }
    let version = r.u64()?;
    let n_files = r.usize()?;
    let mut files = Vec::with_capacity(n_files.min(65_536));
    for _ in 0..n_files {
        let path = r.string()?;
        let text = r.string()?;
        let hash = r.string()?;
        let decoder = intern_decoder(r.str_ref()?)?;
        let n_blocks = r.usize()?;
        let mut blocks = Vec::with_capacity(n_blocks.min(65_536));
        for _ in 0..n_blocks {
            blocks.push(r.block()?);
        }
        let general = match r.u8()? {
            0 => None,
            1 => Some(intern_general(r.str_ref()?)?),
            _ => return None,
        };
        files.push(FileInfo { path, text, hash, decoder, blocks, general });
    }
    let low = r.str_vec()?;
    let n_syms = r.usize()?;
    let mut symbols = Vec::with_capacity(n_syms.min(65_536));
    for _ in 0..n_syms {
        symbols.push(Symbol {
            name: r.string()?,
            kind: r.string()?,
            file: r.string()?,
            line: r.usize()?,
            end: r.usize()?,
            parent: r.string()?,
            text: r.string()?,
        });
    }
    let n_crates = r.usize()?;
    let mut crates = Vec::with_capacity(n_crates.min(4096));
    for _ in 0..n_crates {
        crates.push(Crate {
            name: r.string()?,
            dir: r.string()?,
            deps: r.str_vec()?,
            manifest: r.string()?,
            ecosystem: intern_ecosystem(r.str_ref()?)?,
        });
    }
    let n_chunks = r.usize()?;
    let mut chunks = Vec::with_capacity(n_chunks.min(131_072));
    for _ in 0..n_chunks {
        let file = r.usize()?;
        let start = r.usize()?;
        let end = r.usize()?;
        let text = r.string()?;
        let low_c = r.string()?;
        let n_vec = r.usize()?;
        let mut vec = Vec::with_capacity(n_vec.min(4096));
        for _ in 0..n_vec {
            vec.push(r.f32()?);
        }
        let key = r.string()?;
        let kind = intern_chunk_kind(r.str_ref()?)?;
        let heading = r.string()?;
        let section = match r.u8()? {
            0 => None,
            1 => Some((r.usize()?, r.usize()?)),
            _ => return None,
        };
        let tokens = r.usize()?;
        let name = r.string()?;
        let signature = r.string()?;
        let parent = r.string()?;
        let defines = r.str_vec()?;
        let assigns = r.str_vec()?;
        let n_uses = r.usize()?;
        let mut uses = Vec::with_capacity(n_uses.min(4096));
        for _ in 0..n_uses {
            uses.push((r.string()?, r.u32()?));
        }
        let n_paths = r.usize()?;
        let mut paths = Vec::with_capacity(n_paths.min(4096));
        for _ in 0..n_paths {
            paths.push((r.string()?, r.string()?));
        }
        let implements = r.str_vec()?;
        let members = r.str_vec()?;
        let parts = r.str_vec()?;
        let file_notes: Arc<str> = Arc::from(r.str_ref()?);
        let lead = r.string()?;
        chunks.push(Chunk {
            file,
            start,
            end,
            text,
            low: low_c,
            vec,
            key,
            kind,
            heading,
            section,
            tokens,
            piece: Meta { name, signature, parent, defines, assigns, uses, paths, implements, members, parts, file_notes },
            lead,
        });
    }
    let unit_budget = r.usize()?;
    let n_hist = r.usize()?;
    let mut history = BTreeMap::new();
    for _ in 0..n_hist {
        history.insert(r.string()?, r.str_vec()?);
    }
    let n_df = r.usize()?;
    let mut df = BTreeMap::new();
    for _ in 0..n_df {
        df.insert(r.string()?, r.usize()?);
    }
    let n_hdf = r.usize()?;
    let mut hidden_df = BTreeMap::new();
    for _ in 0..n_hdf {
        hidden_df.insert(r.string()?, r.usize()?);
    }
    let hidden_files = r.usize()?;
    let words = r.usize()?;
    let skip_saved = r.str_vec()?;
    let n_skipped = r.usize()?;
    let mut skipped = Vec::with_capacity(n_skipped.min(65_536));
    for _ in 0..n_skipped {
        skipped.push(Skipped { path: r.string()?, reason: intern_skip_reason(r.str_ref()?)? });
    }
    let n_imports = r.usize()?;
    let mut imports = BTreeMap::new();
    for _ in 0..n_imports {
        let k = r.string()?;
        let n_list = r.usize()?;
        let mut list = Vec::with_capacity(n_list.min(4096));
        for _ in 0..n_list {
            list.push(Import { spec: r.string()?, line: r.usize()?, names: r.str_vec()? });
        }
        imports.insert(k, list);
    }
    let n_edges = r.usize()?;
    let mut import_edges = Vec::with_capacity(n_edges.min(65_536));
    for _ in 0..n_edges {
        import_edges.push(Edge {
            from: r.string()?,
            line: r.usize()?,
            spec: r.string()?,
            to: r.opt_string()?,
            external: r.opt_string()?,
        });
    }
    if r.pos != data.len() {
        return None;
    }
    Some((
        Substrate {
            root: root_canon,
            version,
            files,
            low,
            symbols,
            crates,
            chunks,
            unit_budget,
            history,
            df,
            hidden_df,
            hidden_files,
            words,
            skip: skip_saved,
            skipped,
            imports,
            import_edges,
            ops: Arc::new(std::sync::RwLock::new(Vec::new())),
            dense: Arc::new(std::sync::RwLock::new(None)),
            ref_graph: Arc::new(std::sync::RwLock::new(None)),
            embed: Arc::clone(embed),
        },
        fp,
    ))
}

/// Atomically writes `sub` and its manifest fingerprint to `cache_path`. Silently skips on I/O
/// errors (the cache is strictly an acceleration layer).
pub fn save(sub: &Substrate, cache_path: &Path) {
    let fp = repo::manifest_fingerprint(&sub.root, &sub.skip, &config_tag());
    let mut w = Writer::new();
    w.buf.extend_from_slice(MAGIC);
    w.u32(FORMAT_VERSION);
    w.str(&fp);
    w.u64(sub.version);
    w.usize(sub.files.len());
    for f in &sub.files {
        w.str(&f.path);
        w.str(&f.text);
        w.str(&f.hash);
        w.str(f.decoder);
        w.usize(f.blocks.len());
        for b in &f.blocks {
            w.block(b);
        }
        w.opt_str(f.general);
    }
    w.str_vec(&sub.low);
    w.usize(sub.symbols.len());
    for s in &sub.symbols {
        w.str(&s.name);
        w.str(&s.kind);
        w.str(&s.file);
        w.usize(s.line);
        w.usize(s.end);
        w.str(&s.parent);
        w.str(&s.text);
    }
    w.usize(sub.crates.len());
    for c in &sub.crates {
        w.str(&c.name);
        w.str(&c.dir);
        w.str_vec(&c.deps);
        w.str(&c.manifest);
        w.str(c.ecosystem);
    }
    w.usize(sub.chunks.len());
    for c in &sub.chunks {
        w.usize(c.file);
        w.usize(c.start);
        w.usize(c.end);
        w.str(&c.text);
        w.str(&c.low);
        w.usize(c.vec.len());
        for &x in &c.vec {
            w.f32(x);
        }
        w.str(&c.key);
        w.str(c.kind);
        w.str(&c.heading);
        match c.section {
            Some((a, b)) => {
                w.u8(1);
                w.usize(a);
                w.usize(b);
            }
            None => w.u8(0),
        }
        w.usize(c.tokens);
        w.str(&c.piece.name);
        w.str(&c.piece.signature);
        w.str(&c.piece.parent);
        w.str_vec(&c.piece.defines);
        w.str_vec(&c.piece.assigns);
        w.usize(c.piece.uses.len());
        for (u, cnt) in &c.piece.uses {
            w.str(u);
            w.u32(*cnt);
        }
        w.usize(c.piece.paths.len());
        for (q, nm) in &c.piece.paths {
            w.str(q);
            w.str(nm);
        }
        w.str_vec(&c.piece.implements);
        w.str_vec(&c.piece.members);
        w.str_vec(&c.piece.parts);
        w.str(&c.piece.file_notes);
        w.str(&c.lead);
    }
    w.usize(sub.unit_budget);
    w.usize(sub.history.len());
    for (k, v) in &sub.history {
        w.str(k);
        w.str_vec(v);
    }
    w.usize(sub.df.len());
    for (k, &v) in &sub.df {
        w.str(k);
        w.usize(v);
    }
    w.usize(sub.hidden_df.len());
    for (k, &v) in &sub.hidden_df {
        w.str(k);
        w.usize(v);
    }
    w.usize(sub.hidden_files);
    w.usize(sub.words);
    w.str_vec(&sub.skip);
    w.usize(sub.skipped.len());
    for s in &sub.skipped {
        w.str(&s.path);
        w.str(s.reason);
    }
    w.usize(sub.imports.len());
    for (k, list) in &sub.imports {
        w.str(k);
        w.usize(list.len());
        for imp in list {
            w.str(&imp.spec);
            w.usize(imp.line);
            w.str_vec(&imp.names);
        }
    }
    w.usize(sub.import_edges.len());
    for e in &sub.import_edges {
        w.str(&e.from);
        w.usize(e.line);
        w.str(&e.spec);
        w.opt_str(e.to.as_deref());
        w.opt_str(e.external.as_deref());
    }
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = cache_path.with_extension("bin.tmp");
    if std::fs::write(&tmp, &w.buf).is_ok() {
        let _ = std::fs::rename(&tmp, cache_path);
    }
}

/// Section check (`nmsubs.cache`): verifies cold scan -> cache save -> exact cache hit parity ->
/// single-byte file edit invalidation and clean rescan.
/// # Errors
/// One fault line if any invariant fails.
pub fn check() -> Result<(), String> {
    let base = nmtype::nano_root().join("scratch").join(format!("nmsubs-cache-check-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let repo_dir = base.join("repo");
    let cache_file = base.join("mem").join("substrate-cache.bin");
    std::fs::create_dir_all(repo_dir.join("src")).map_err(|e| format!("nmsubs.cache: mkdir: {e}"))?;
    std::fs::write(repo_dir.join("src").join("lib.rs"), "pub fn alpha(x: i32) -> i32 { x + 1 }\n").map_err(|e| format!("nmsubs.cache: write: {e}"))?;
    std::fs::write(repo_dir.join("README.md"), "# Demo\n\nHello substrate cache.\n").map_err(|e| format!("nmsubs.cache: write md: {e}"))?;
    let embed: Embed = Arc::new(|t: &str| {
        #[expect(clippy::cast_precision_loss, reason = "tiny test fixture string length")]
        let l = t.len() as f32;
        vec![l]
    });
    let s1 = Substrate::scan_cached(&repo_dir, &[], Arc::clone(&embed), Some(&cache_file));
    if !cache_file.is_file() {
        let _ = std::fs::remove_dir_all(&base);
        return Err("nmsubs.cache: cold scan did not write cache file".into());
    }
    let (s2, _) = load(&repo_dir, &[], &embed, &cache_file).ok_or_else(|| {
        let _ = std::fs::remove_dir_all(&base);
        "nmsubs.cache: warm load missed on unchanged tree".to_owned()
    })?;
    let same = s1.files == s2.files
        && s1.low == s2.low
        && s1.symbols == s2.symbols
        && s1.chunks.len() == s2.chunks.len()
        && s1.chunks.iter().zip(&s2.chunks).all(|(a, b)| a.key == b.key && a.text == b.text && a.vec == b.vec && a.kind == b.kind)
        && s1.df == s2.df
        && s1.words == s2.words
        && s1.unit_budget == s2.unit_budget;
    if !same {
        let _ = std::fs::remove_dir_all(&base);
        return Err("nmsubs.cache: cached substrate differs from cold scan".into());
    }
    std::fs::write(repo_dir.join("src").join("lib.rs"), "pub fn alpha(x: i32) -> i32 { x + 2 }\npub fn beta() {}\n")
        .map_err(|e| format!("nmsubs.cache: edit: {e}"))?;
    if load(&repo_dir, &[], &embed, &cache_file).is_some() {
        let _ = std::fs::remove_dir_all(&base);
        return Err("nmsubs.cache: edited file did not invalidate cache".into());
    }
    let s3 = Substrate::scan_cached(&repo_dir, &[], embed, Some(&cache_file));
    let _ = std::fs::remove_dir_all(&base);
    if !s3.symbols.iter().any(|s| s.name == "beta") {
        return Err("nmsubs.cache: rescan after invalidation missed new symbol beta".into());
    }
    Ok(())
}
