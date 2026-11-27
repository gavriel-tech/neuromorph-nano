//! nmsubs: the Global Context Substrate (guide 5, 5.1, 6.2; PLAN 4c), thin form for S2.
//! One store, one version counter; the eight views are lenses over it (`views`), Context
//! Zoom moves along the code axis (`zoom`), and `sync` re-reads only changed files.
//! Implements `nmtype::SubstrateView`, the only face a unit sees.
pub mod cache;
pub mod dense;
pub mod doc;
pub mod lock;
pub mod general;
pub mod imports;
pub mod items;
pub mod lens;
pub mod piece;
pub mod refgraph;
pub mod repo;
pub mod sync;
pub mod tags;
pub mod tree_shape;
pub mod units;
pub mod views;
pub mod zoom;

use repo::{Crate, FileInfo, Symbol};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Lines per semantic chunk.
pub const CHUNK: usize = 30;

/// Definition size budget (non-whitespace characters) when the tree has no code windows to measure.
pub const UNIT_FALLBACK: usize = 700;

/// An embedder passed in by the Core Brain (exact control or a Qwen3 slot).
pub type Embed = Arc<dyn Fn(&str) -> Vec<f32> + Send + Sync>;

/// One semantic chunk: file index, first line, text, vector.
#[derive(Clone, Debug)]
pub struct Chunk {
    pub file: usize,
    pub start: usize,
    /// Last line of the segment (2026-11-10: segments follow the file's structure).
    pub end: usize,
    pub text: String,
    /// `text` lower-cased once at scan, for key-term coverage per question (2026-11-08).
    pub low: String,
    pub vec: Vec<f32>,
    /// SHA-256 of the text the dense lane embeds (path line and lines), its store key.
    pub key: String,
    /// Kind of the chunk's main block (`window` for code windows; `paragraph`, `list`, `table`,
    /// `code`, `record`, `log` ... for documents).
    pub kind: &'static str,
    /// Heading path of a document chunk (record key, table header, log time range); empty for code.
    pub heading: String,
    /// Lines of the section a document chunk belongs to (its parent for small-to-big hand-over).
    pub section: Option<(usize, usize)>,
    /// Tokens of the text (characters / 4).
    pub tokens: usize,
    /// What a code piece is about: name path, signature, identifiers (`piece::Meta`); empty for
    /// documents and for `CHUNK`-line windows.
    pub piece: piece::Meta,
    /// A code window's lead (`code_lead`): the attached doc comment and first line of the
    /// definition it sits in, when they lie above it; empty for documents.
    pub lead: String,
}

/// The substrate: repository graph, semantic store and history on one version.
#[derive(Clone)]
pub struct Substrate {
    pub root: PathBuf,
    pub version: u64,
    pub files: Vec<FileInfo>,
    /// Each file's text lower-cased once per index, same order as `files`, for grep and
    /// coverage (2026-11-08: grep lower-cased every line of the tree per term per question).
    pub low: Vec<String>,
    pub symbols: Vec<Symbol>,
    pub crates: Vec<Crate>,
    pub chunks: Vec<Chunk>,
    /// The size a served definition is kept within (non-whitespace characters, `units::budget`),
    /// derived from the tree at index.
    pub unit_budget: usize,
    pub history: BTreeMap<String, Vec<String>>,
    pub df: BTreeMap<String, usize>,
    /// The words of the files read from a project's own hidden configuration
    /// (`nmtype::in_hidden_folder`), with how many of those files hold each, and how many files
    /// that is: kept apart from `df`, `words` and the file count of `idf`, so reading them shifts no
    /// rarity; a word they hold is still known to the tree (`known`).
    pub hidden_df: BTreeMap<String, usize>,
    pub hidden_files: usize,
    /// Word occurrences over every file read (the length of the material in words; with the
    /// distinct words of `df`, how often the material uses each word it holds).
    pub words: usize,
    pub skip: Vec<String>,
    /// Files the format table did not read, each with its reason (shown by `nm_status`).
    pub skipped: Vec<repo::Skipped>,
    /// Import statements per file of the tagged languages (`tags`), by path.
    pub imports: BTreeMap<String, Vec<tags::Import>>,
    /// Those imports resolved to files of the tree or outside packages (`imports`).
    pub import_edges: Vec<imports::Edge>,
    /// Verified build and Step Check records for the operational view (2026-11-09), shared by
    /// every version of this substrate so a sync keeps them; reloaded after each build tool run.
    pub ops: Arc<std::sync::RwLock<Vec<OpRecord>>>,
    /// The dense lane when the embedder engine is bound (2026-11-10), shared by every version
    /// of this substrate so a sync keeps it.
    pub dense: Arc<std::sync::RwLock<Option<Arc<dense::DenseLane>>>>,
    /// The reference graph over items (2026-11-10, refinement step 1), held in RAM, built on
    /// first use and rebuilt on sync when in use (`refgraph`); shared by every version.
    pub ref_graph: Arc<std::sync::RwLock<Option<Arc<refgraph::RefGraph>>>>,
    embed: Embed,
}

/// One verified operational record: the file and line it bears on, and what happened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpRecord {
    pub source: String,
    pub text: String,
}

impl std::fmt::Debug for Substrate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Substrate")
            .field("root", &self.root)
            .field("version", &self.version)
            .field("files", &self.files.len())
            .field("symbols", &self.symbols.len())
            .field("crates", &self.crates.len())
            .finish_non_exhaustive()
    }
}

impl Substrate {
    /// Read a root folder into every graph at version 1. `skip` names extra folders to leave
    /// out (for example the mini itself when it studies its parent tree).
    #[must_use]
    pub fn scan(root: &Path, skip: &[String], embed: Embed) -> Self {
        let root = repo::canon(root);
        let (files, skipped, crates, history) = std::thread::scope(|sc| {
            let h_crates = sc.spawn(|| repo::crates_of(&root));
            let h_hist = sc.spawn(|| repo::history_of(&root));
            let mut files = Vec::new();
            let mut skipped = Vec::new();
            repo::walk_counted(&root, &root, skip, &mut files, &mut skipped);
            let crates = h_crates.join().unwrap_or_default();
            let history = h_hist.join().unwrap_or_default();
            (files, skipped, crates, history)
        });
        let mut s = Self {
            skipped,
            crates,
            history,
            root,
            version: 0,
            files: Vec::new(),
            low: Vec::new(),
            symbols: Vec::new(),
            chunks: Vec::new(),
            unit_budget: UNIT_FALLBACK,
            df: BTreeMap::new(),
            hidden_df: BTreeMap::new(),
            hidden_files: 0,
            words: 0,
            skip: skip.to_vec(),
            imports: BTreeMap::new(),
            import_edges: Vec::new(),
            ops: Arc::new(std::sync::RwLock::new(Vec::new())),
            dense: Arc::new(std::sync::RwLock::new(None)),
            ref_graph: Arc::new(std::sync::RwLock::new(None)),
            embed,
        };
        s.index(files);
        s.version = 1;
        s
    }
    /// Scan `root` with a content-addressed binary cache at `cache_path` (when `Some` and
    /// `cache::enabled()`). On an exact metadata manifest hit, restores the scanned substrate in
    /// milliseconds without re-reading or re-indexing unchanged files; on miss, performs a normal
    /// `scan` and atomically writes the cache file.
    #[must_use]
    pub fn scan_cached(root: &Path, skip: &[String], embed: Embed, cache_path: Option<&Path>) -> Self {
        if let Some(cp) = cache_path.filter(|_| cache::enabled()) {
            if let Some((sub, _)) = cache::load(root, skip, &embed, cp) {
                return sub;
            }
            let sub = Self::scan(root, skip, embed);
            cache::save(&sub, cp);
            return sub;
        }
        Self::scan(root, skip, embed)
    }
    /// Rebuild symbols, chunks and term counts from a file list.
    pub(crate) fn index(&mut self, mut files: Vec<FileInfo>) {
        struct PerFile {
            hidden: bool,
            word_count: usize,
            seen: Vec<String>,
            own: Vec<Symbol>,
            imports: Option<(String, Vec<tags::Import>)>,
            window_sizes: Vec<usize>,
            chunks: Vec<Chunk>,
            low: String,
        }
        self.symbols.clear();
        self.chunks.clear();
        self.df.clear();
        self.hidden_df.clear();
        self.hidden_files = 0;
        self.words = 0;
        self.imports.clear();
        let cut = doc::cut();
        let pieces = piece::on();
        let embed = &self.embed;
        let per_file: Vec<Option<PerFile>> = nmtype::budget::run(files.len(), |i| {
            let Some(f) = files.get(i) else {
                return PerFile {
                    hidden: false,
                    word_count: 0,
                    seen: Vec::new(),
                    own: Vec::new(),
                    imports: None,
                    window_sizes: Vec::new(),
                    chunks: Vec::new(),
                    low: String::new(),
                };
            };
            let mut seen = nmlex::words(&f.text);
            let hidden = nmtype::in_hidden_folder(&f.path);
            let word_count = if hidden { 0 } else { seen.len() };
            if nmlex::lang::by_id(f.decoder).is_some_and(|l| l.camel) {
                seen.extend(camel_words(&f.text));
            }
            let mut own: Vec<Symbol> = Vec::new();
            let mut blocks: Vec<repo::ImplBlock> = Vec::new();
            let mut rust_toks: Option<Vec<(String, usize)>> = None;
            let mut file_imports: Option<(String, Vec<tags::Import>)> = None;
            if f.path.ends_with(".rs") {
                let toks = nmlex::tokens(&f.text, true);
                let (syms, b) = repo::parse_items_with(&f.path, &f.text, &toks);
                own = syms;
                blocks = b;
                rust_toks = Some(toks);
            } else if f.is_code()
                && let Some(t) = tags::tag(&f.path, &f.text)
            {
                own = t.symbols;
                seen.extend(t.words);
                if !t.imports.is_empty() {
                    file_imports = Some((f.path.clone(), t.imports));
                }
            }
            seen.sort_unstable();
            seen.dedup();
            let lines: Vec<&str> = f.text.lines().collect();
            let window_sizes: Vec<usize> = if f.is_code() && !hidden {
                lines.chunks(CHUNK).map(|w| w.iter().map(|l| l.chars().filter(|c| !c.is_whitespace()).count()).sum::<usize>()).collect()
            } else {
                Vec::new()
            };
            let item_lines: Vec<usize> = if f.path.ends_with(".rs") { own.iter().map(|s| s.line).collect() } else { Vec::new() };
            let segs: Vec<(doc::Seg, piece::Meta)> = match (f.is_code(), cut) {
                (false, doc::Cut::Blocks(budget)) => doc::segments(&f.blocks, &f.text, budget).into_iter().map(|s| (s, piece::Meta::default())).collect(),
                (true, _) if pieces => {
                    let tables;
                    let defs: &[Symbol] = if f.decoder == "toml" {
                        tables = piece::toml_tables(&f.path, &f.text);
                        &tables
                    } else {
                        &own
                    };
                    let refs: Vec<&Symbol> = defs.iter().collect();
                    let toks = piece::tokens(&f.text, f.decoder, rust_toks.as_deref());
                    let mut cuts = piece::cut(&f.text, f.decoder, &refs, &blocks, &toks);
                    piece::fill(&mut cuts, &refs, &toks, f.decoder);
                    piece::file_notes(&mut cuts, &lines);
                    cuts.into_iter()
                        .map(|c| (doc::Seg { start: c.start, end: c.end, kind: c.kind, heading: c.heading, section: None }, c.meta))
                        .collect()
                }
                _ => {
                    let ranges: Vec<(usize, usize)> = if STRUCTURAL_SEGMENTS {
                        segments(&f.path, &lines, &item_lines)
                    } else {
                        tail_joined((1..=lines.len().max(1)).step_by(CHUNK).map(|s| (s, (s + CHUNK - 1).min(lines.len().max(1)))).collect())
                    };
                    let code = f.is_code();
                    ranges
                        .into_iter()
                        .map(|(start, end)| {
                            let seg = doc::Seg {
                                start,
                                end,
                                kind: if code { "window" } else { "paragraph" },
                                heading: if !code {
                                    doc::heading_path_of(f, start)
                                } else if matches!(f.decoder, "rust" | "toml") {
                                    String::new()
                                } else {
                                    nearest_definition(&own, &f.path, start)
                                },
                                section: None,
                            };
                            (seg, piece::Meta::default())
                        })
                        .collect()
                }
            };
            let mut chunks = Vec::with_capacity(segs.len());
            for (seg, meta) in segs {
                let (start, end) = (seg.start, seg.end);
                let text = lines.get(start - 1..end).map(|p| p.join("\n")).unwrap_or_default();
                let vec = if seg.heading.is_empty() {
                    embed(&format!("{} {text}", f.path))
                } else {
                    embed(&format!("{} {} {text}", f.path, seg.heading))
                };
                let low = text.to_lowercase();
                let key = nmhash::sha256_bytes(dense::doc_text_with(&f.path, &seg.heading, &text).as_bytes());
                let tokens = text.chars().count() / 4;
                chunks.push(Chunk {
                    file: i,
                    start,
                    end,
                    text,
                    low,
                    vec,
                    key,
                    kind: seg.kind,
                    heading: seg.heading,
                    section: seg.section,
                    tokens,
                    piece: meta,
                    lead: if f.is_code() { code_lead(&f.path, &lines, &own, start) } else { String::new() },
                });
            }
            let low = f.text.to_lowercase();
            PerFile { hidden, word_count, seen, own, imports: file_imports, window_sizes, chunks, low }
        });
        let mut window_sizes: Vec<usize> = Vec::new();
        let mut lows: Vec<String> = Vec::with_capacity(files.len());
        for pf in per_file.into_iter().flatten() {
            if pf.hidden {
                self.hidden_files += 1;
            } else {
                self.words += pf.word_count;
            }
            self.symbols.extend(pf.own);
            if let Some((path, imps)) = pf.imports {
                self.imports.insert(path, imps);
            }
            let counts = if pf.hidden { &mut self.hidden_df } else { &mut self.df };
            for w in pf.seen {
                *counts.entry(w).or_default() += 1;
            }
            window_sizes.extend(pf.window_sizes);
            self.chunks.extend(pf.chunks);
            lows.push(pf.low);
        }
        // Code files enter the document model: an item block per symbol.
        let mut by_file: BTreeMap<&str, Vec<&Symbol>> = BTreeMap::new();
        for s in &self.symbols {
            by_file.entry(s.file.as_str()).or_default().push(s);
        }
        // The size a served definition is kept within: the median size of this tree's code
        // windows, so a definition weighs about as a window does.
        self.unit_budget = units::budget(&window_sizes, UNIT_FALLBACK);
        for f in files.iter_mut().filter(|f| f.is_code()) {
            f.blocks = doc::code_blocks(f.decoder, &f.text, by_file.get(f.path.as_str()).map_or(&[][..], Vec::as_slice));
        }
        // Packages outside Cargo (npm, Python) and the resolved imports.
        self.crates.retain(|c| c.ecosystem == "cargo");
        self.crates.extend(imports::packages(&files));
        self.import_edges = imports::resolve(&files, &self.imports);
        self.low = lows;
        self.files = files;
    }
    /// The bytes the substrate's content holds in RAM, estimated from its parts (memory areas,
    /// 2026-11-27: the `content` area): each file's text and its lower-cased copy, each chunk's
    /// text, lower-cased copy, vector, key, heading, lead and piece facts, and each symbol's
    /// strings; each heap block its capacity rounded up to 16 bytes plus a 16-byte header.
    #[must_use]
    pub fn content_bytes(&self) -> u64 {
        let block = |cap: usize| if cap == 0 { 0 } else { cap.div_ceil(16) * 16 + 16 };
        let st = |s: &String| block(s.capacity());
        let names = |v: &Vec<String>| block(v.capacity() * 24) + v.iter().map(st).sum::<usize>();
        let files: usize = self
            .files
            .iter()
            .map(|f| std::mem::size_of::<FileInfo>() + st(&f.path) + st(&f.text) + st(&f.hash) + block(f.blocks.capacity() * std::mem::size_of::<nmtype::doc::Block>()))
            .sum();
        let low: usize = self.low.iter().map(st).sum();
        let chunks: usize = self
            .chunks
            .iter()
            .map(|c| {
                std::mem::size_of::<Chunk>()
                    + st(&c.text)
                    + st(&c.low)
                    + block(c.vec.capacity() * 4)
                    + st(&c.key)
                    + st(&c.heading)
                    + st(&c.lead)
                    + st(&c.piece.name)
                    + st(&c.piece.signature)
                    + st(&c.piece.parent)
                    + names(&c.piece.defines)
                    + names(&c.piece.assigns)
                    + names(&c.piece.implements)
                    + names(&c.piece.members)
                    + names(&c.piece.parts)
                    + block(c.piece.uses.capacity() * 32)
                    + c.piece.uses.iter().map(|(u, _)| st(u)).sum::<usize>()
                    + block(c.piece.paths.capacity() * 48)
                    + c.piece.paths.iter().map(|(a, b)| st(a) + st(b)).sum::<usize>()
            })
            .sum();
        let symbols: usize = self.symbols.iter().map(|s| std::mem::size_of::<Symbol>() + st(&s.name) + st(&s.kind) + st(&s.file) + st(&s.parent) + st(&s.text)).sum();
        let df: usize = self.df.keys().map(|k| st(k) + 48).sum();
        (files + low + chunks + symbols + df) as u64
    }
    /// Content digest of the studied tree: SHA-256 over the sorted `path hash` lines of every
    /// file read. Recorded in run reports so a result names exactly what it studied, even
    /// when the repository has uncommitted changes.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut lines: Vec<String> = self.files.iter().map(|f| format!("{} {}", f.path, f.hash)).collect();
        lines.sort();
        nmhash::sha256_bytes(lines.join("\n").as_bytes())
    }
    /// Query vector from the embedder this substrate was built with.
    #[must_use]
    pub fn embed(&self, text: &str) -> Vec<f32> {
        (self.embed)(text)
    }
    /// Rarity of a term: ln((files + 1) / (files containing it + 0.5)), at least 0.1.
    #[must_use]
    pub fn idf(&self, term: &str) -> f32 {
        #[expect(clippy::cast_precision_loss, reason = "file counts are small")]
        let (n, d) = (self.files.len().saturating_sub(self.hidden_files) as f32, self.df.get(term).copied().unwrap_or(0) as f32);
        ((n + 1.0) / (d + 0.5)).ln().max(0.1)
    }
    /// Key terms of a question that no file of the tree contains, and their rarity-weighted
    /// share of the question (the withhold rule in `nmbrain::synthesis`), judged without a
    /// vocabulary of general use (see `absent_terms_with`).
    #[must_use]
    pub fn absent_terms(&self, q: &str) -> (Vec<String>, f32) {
        self.absent_terms_with(q, None)
    }
    /// How likely the material is to hold a given word of general use if it held the answer: the
    /// weight an absent ordinary word carries as evidence of no answer. Measured on the material:
    /// `r`, the share of the common vocabulary (`general::GeneralUse::common_count`) that it
    /// holds, is the chance that one use of a word lands on a given common word; `k`, its word
    /// occurrences over its distinct words, is how often it uses each word it holds, so how many
    /// times an answer's concept is named in it. The chance that at least one of those uses is
    /// the question's word is `1 - (1 - r)^k`. A large tree of prose and code holds most common
    /// words many times (near 1: absence counts in full); a tree of a few small files holds few
    /// of them (near 0: absence of an ordinary word says little). Without a vocabulary, 1.
    #[must_use]
    pub fn expected_presence(&self, general: Option<&general::GeneralUse>) -> f32 {
        let Some(g) = general.filter(|g| g.common_count() > 0) else { return 1.0 };
        if self.df.is_empty() {
            return 1.0;
        }
        #[expect(clippy::cast_precision_loss, reason = "word counts are far below 2^24 per digit kept")]
        let (r, k) = (
            self.df.keys().filter(|w| g.frequent(w)).count() as f32 / g.common_count() as f32,
            self.words as f32 / self.df.len() as f32,
        );
        (1.0 - (1.0 - r.clamp(0.0, 1.0)).powf(k.max(1.0))).clamp(0.0, 1.0)
    }
    /// The question's terms (its names and key terms, code and document forms) that the tree holds
    /// in no form: no packet can contain them, so the reducer leaves them out when it measures how
    /// much of the question a packet covers.
    #[must_use]
    pub fn unmatched_terms(&self, q: &str) -> Vec<String> {
        let mut terms: Vec<String> = nmlex::names(q).iter().map(|n| n.to_lowercase()).collect();
        terms.extend(nmlex::key_terms(q));
        terms.extend(nmlex::key_terms_doc(q));
        terms.sort();
        terms.dedup();
        terms.retain(|t| !self.present_form(t));
        terms
    }
    /// True when the tree holds `t` or a near form of it (`absent_terms_with`'s first test).
    fn present_form(&self, t: &str) -> bool {
        [&self.df, &self.hidden_df].into_iter().any(|df| {
            df.contains_key(t)
                || df.range(t.to_owned()..).take_while(|(k, _)| k.starts_with(t)).any(|(k, _)| k.len() <= t.len() + 3)
                || (4..t.len()).rev().take(3).any(|n| t.get(..n).is_some_and(|p| df.contains_key(p)))
                || PREFIXES.iter().any(|x| t.strip_prefix(x).is_some_and(|rest| rest.len() >= 4 && df.contains_key(rest)))
        })
    }
    /// True when some file read holds the word `t`, its hidden configuration included.
    #[must_use]
    pub fn known(&self, t: &str) -> bool {
        self.df.contains_key(t) || self.hidden_df.contains_key(t)
    }
    /// Key terms of a question that name a concept no file of the tree contains, and their
    /// rarity-weighted share of the question. Only content terms count (`general::content_term`):
    /// function words and, given `general`, common verbs never make a question withheld, however
    /// absent they are. A `camelCase` term whose parts the tree holds counts half, like a
    /// hyphenated one. An absent word of general use, as written (not an identifier), counts by
    /// `expected_presence`: its absence is evidence of no answer only as far as the material would
    /// be expected to hold it; so does a bare number; a name, a code or a term outside general use
    /// counts in full.
    #[must_use]
    pub fn absent_terms_with(&self, q: &str, general: Option<&general::GeneralUse>) -> (Vec<String>, f32) {
        let terms = nmlex::key_terms(q);
        // The question's words as written, by their lower-case form, for the identifier test.
        let raw: std::collections::BTreeMap<String, String> = q
            .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == ':'))
            .map(|w| w.trim_matches(|c: char| c == '.' || c == '-' || c == '?' || c == ':'))
            .filter(|w| !w.is_empty())
            .map(|w| (w.to_lowercase(), w.to_owned()))
            .collect();
        let total: f32 = terms.iter().map(|t| self.idf(t)).sum();
        // A word form counts as present (`speak` in a tree that says `speaks`): a stored word
        // that extends the term by up to three letters, or that the term extends likewise.
        // A common prefix on a stored word counts likewise (`subgraph` where the tree says `graph`).
        let near = |t: &str| self.present_form(t);
        let presence = self.expected_presence(general);
        // 2026-11-10 (CX1 development half): an inflected form whose stem the tree holds
        // (`flagging` where it says `flag`: a common suffix, then a doubled last letter), and a
        // hyphenated term whose parts are all present and three letters or more (`non-test`).
        let stem = |t: &str| {
            SUFFIXES.iter().filter_map(|x| t.strip_suffix(x)).any(|s| {
                let undoubled = if s.len() > 3 && s.as_bytes().last() == s.as_bytes().get(s.len() - 2) { &s[..s.len() - 1] } else { s };
                s.len() >= 3 && (near(s) || near(undoubled))
            })
        };
        let present = |t: &str| near(t) || stem(t);
        let whole = |t: &str| {
            // Every part must be three letters or more: `top-p` names one concept and stays whole.
            present(t) || (t.contains('-') && t.split('-').all(|x| x.len() >= 3 && present(x)))
        };
        // A term found only through its stem or its parts is weaker evidence than the word
        // itself: it counts half absent (listed with a trailing `~`), so a question mixing real
        // words with an absent concept is still withheld (CX1 development decoy).
        let mut absent = Vec::new();
        let mut weight = 0.0_f32;
        for t in terms {
            if near(&t) {
                continue;
            }
            let as_written = raw.get(&t).map_or(t.as_str(), String::as_str);
            if !general::content_term(as_written, &t, general) {
                continue;
            }
            let camel = general::camel_parts(as_written);
            // A bare number is a quantity the question supplies (an order of 300 against steps
            // of 250), not a name the material must hold: weighed like a word of general use.
            let quantity = general.is_some() && t.chars().all(|c| c.is_ascii_digit() || c == '.');
            let ordinary = !general::identifier_like(as_written) && general.is_some_and(|g| g.general_word(&t));
            let evidence = if quantity || ordinary { presence } else { 1.0 };
            if whole(&t) || (!camel.is_empty() && camel.iter().all(|p| p.len() < 3 || present(p))) {
                weight += 0.5 * evidence * self.idf(&t);
                absent.push(format!("{t}~"));
            } else {
                weight += evidence * self.idf(&t);
                absent.push(t);
            }
        }
        let share = if total > 0.0 { weight / total } else { 0.0 };
        (absent, share)
    }
    /// Share of a question's key terms, weighted by rarity, that a text covers.
    #[must_use]
    pub fn coverage(&self, q: &str, text: &str) -> f32 {
        let terms = nmlex::key_terms(q);
        let d = text.to_lowercase();
        let total: f32 = terms.iter().map(|t| self.idf(t)).sum();
        if total <= 0.0 {
            return 0.0;
        }
        terms.iter().filter(|t| d.contains(t.as_str())).map(|t| self.idf(t)).sum::<f32>() / total
    }
    /// Crate by name: exact, then with `emma-` added or dropped, then by folder.
    #[must_use]
    pub fn crate_named(&self, q: &str) -> Option<&Crate> {
        let q = q.to_lowercase();
        self.crates
            .iter()
            .find(|c| c.name == q)
            .or_else(|| self.crates.iter().find(|c| c.name.strip_prefix("emma-") == Some(q.as_str()) || c.dir.ends_with(&format!("/{q}"))))
    }
    /// Crates that depend on `name`.
    #[must_use]
    pub fn dependents(&self, name: &str) -> Vec<String> {
        self.crates.iter().filter(|c| c.deps.iter().any(|d| d == name)).map(|c| c.name.clone()).collect()
    }
}

/// The lead of a code window (step 1l, change C): what the window's lines need to be read as
/// their definition is, written above them when the dense lane embeds the window: the doc comment
/// attached to the definition holding (or, in Rust, last opened before) the window's first line,
/// then that definition's first line, each only when it lies above the window. Empty for a window
/// that opens no definition's body and carries no comment above it.
#[must_use]
pub fn code_lead(path: &str, lines: &[&str], symbols: &[Symbol], start: usize) -> String {
    let here = symbols.iter().filter(|s| s.file == path && s.line <= start);
    let rust = path.ends_with(".rs");
    let holding = if rust { None } else { here.clone().filter(|s| start <= s.end).max_by_key(|s| s.line) };
    let Some(def) = holding.or_else(|| here.max_by_key(|s| s.line)) else { return String::new() };
    let line = |i: usize| lines.get(i.wrapping_sub(1)).map_or("", |l| l.trim());
    let comment = |t: &str| {
        t.starts_with("//") || t.starts_with("/*") || t.starts_with('*') || t.starts_with("--") || (!rust && t.starts_with('#') && !t.starts_with("#!"))
    };
    let mut doc: Vec<&str> = Vec::new();
    let mut i = def.line;
    while i > 1 {
        let t = line(i - 1);
        if t.starts_with("#[") || t.starts_with('@') {
            i -= 1;
            continue;
        }
        if t.is_empty() || !comment(t) || t.starts_with("//!") {
            break;
        }
        doc.push(t);
        i -= 1;
    }
    doc.reverse();
    // A docstring under the definition's first line (Python), when above the window.
    let mut under: Vec<&str> = Vec::new();
    let next = line(def.line + 1);
    if !rust && (next.starts_with("\"\"\"") || next.starts_with("'''")) {
        let quote = &next[..3];
        let mut j = def.line + 1;
        while j < start {
            let t = line(j);
            under.push(t);
            if (j > def.line + 1 || t.len() > 3) && t.ends_with(quote) {
                break;
            }
            j += 1;
        }
    }
    let mut out: Vec<&str> = doc;
    if def.line < start {
        out.push(line(def.line));
        out.extend(under);
    }
    out.join("\n")
}

/// The heading of a code window outside Rust: the innermost definition holding its first line,
/// else the last one above it (`method TaskStore.add`), empty when there is none.
fn nearest_definition(symbols: &[Symbol], path: &str, line: usize) -> String {
    let here = symbols.iter().filter(|s| s.file == path);
    let holding = here.clone().filter(|s| s.line <= line && line <= s.end).max_by_key(|s| s.line);
    let above = || here.clone().filter(|s| s.line <= line).max_by_key(|s| s.line);
    holding.or_else(above).map_or_else(String::new, |s| {
        if s.parent.is_empty() { format!("{} {}", s.kind, s.name) } else { format!("{} {}.{}", s.kind, s.parent, s.name) }
    })
}

/// The `camelCase` parts of every identifier in a text, lower case (`nmlex::lang::camel_parts`).
#[must_use]
pub fn camel_words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if w.len() >= 4 && w.chars().any(char::is_uppercase) && w.chars().any(char::is_lowercase) {
            out.extend(nmlex::lang::camel_parts(w));
        }
    }
    out
}

/// Cosine similarity of two vectors (0 when either is zero).
#[must_use]
pub fn cos(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (x, y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na.sqrt() * nb.sqrt()) }
}

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate, one per built sub-component.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmsubs.repo", repo::check),
        ("nmsubs.tags", tags::check),
        ("nmsubs.imports", imports::check),
        ("nmsubs.items", items::check),
        ("nmsubs.doc", doc::check),
        ("nmsubs.piece", piece::check),
        ("nmsubs.cache", cache::check),
        ("nmsubs.dense", dense::check),
        ("nmsubs.lens", lens::check),
        ("nmsubs.refgraph", refgraph::check),
        ("nmsubs.views", views::check),
        ("nmsubs.general", general::check),
        ("nmsubs.zoom", zoom::check),
        ("nmsubs.units", units::check),
        ("nmsubs.sync", sync::check),
        ("nmsubs.tree_shape", tree_shape::check),
    ]
}

/// English prefixes that make a compound of a stored word (absent-word check, 2026-11-09).
/// English suffixes stripped to find a stored stem (absent-word check, 2026-11-10).
const SUFFIXES: [&str; 7] = ["ing", "ed", "es", "s", "er", "ers", "ly"];
const PREFIXES: [&str; 12] = ["sub", "re", "pre", "un", "non", "multi", "inter", "super", "over", "under", "co", "de"];

impl Substrate {
    /// Load the verified records the build tools log (`build-outcomes.jsonl`: failed runs with
    /// their errors, fixes, Step Check verdicts) into the operational view, newest first. A
    /// missing file leaves the view empty. Returns how many records were loaded.
    pub fn load_operational(&self, path: &Path) -> usize {
        let text = std::fs::read_to_string(path).unwrap_or_default();
        let mut recs = Vec::new();
        let mut failed: std::collections::BTreeMap<String, (String, String)> = std::collections::BTreeMap::new();
        for line in text.lines() {
            let Ok(v) = nmjson::parse(line.as_bytes()) else { continue };
            let s = |k: &str| v.get(k).ok().and_then(|x| x.string().ok()).unwrap_or("").to_owned();
            match s("event").as_str() {
                "run" => {
                    for e in v.get("errors").ok().and_then(|x| x.array().ok()).unwrap_or(&[]) {
                        let g = |k: &str| e.get(k).ok().and_then(|x| x.string().ok()).unwrap_or("").to_owned();
                        let at = g("at");
                        recs.push(OpRecord {
                            text: format!(
                                "build error {} at {at} in package {} ({} run {}): {}",
                                g("code"),
                                s("package"),
                                s("kind"),
                                s("run"),
                                g("message")
                            ),
                            source: at,
                        });
                    }
                }
                "fixed" => {
                    let at = s("error_at");
                    recs.push(OpRecord {
                        text: format!(
                            "build error {} at {at} in package {} fixed by run {}: {}",
                            s("code"),
                            s("package"),
                            s("by_run"),
                            s("message")
                        ),
                        source: at,
                    });
                }
                "stepcheck" => {
                    let files: Vec<String> = v
                        .get("files")
                        .ok()
                        .and_then(|x| x.array().ok())
                        .unwrap_or(&[])
                        .iter()
                        .filter_map(|f| f.string().ok().map(str::to_owned))
                        .collect();
                    let added: Vec<String> = v
                        .get("added")
                        .ok()
                        .and_then(|x| x.array().ok())
                        .unwrap_or(&[])
                        .iter()
                        .filter_map(|f| f.string().ok().map(str::to_owned))
                        .collect();
                    let detail: String = s("detail").chars().take(300).collect();
                    let key = files.join(", ");
                    for f in &files {
                        recs.push(OpRecord {
                            source: format!("{f}:1"),
                            text: format!("Step Check {} on a diff touching {key}: {} {}", s("verdict"), s("failed_stage"), detail)
                                .trim_end()
                                .to_owned(),
                        });
                    }
                    // M2 (2026-11-09): a failed check followed by a passing one on the same files is
                    // a failed approach and its fix, returned when the same files come up again.
                    if s("verdict") == "fail" {
                        failed.insert(key, (s("failed_stage"), detail.chars().take(200).collect::<String>()));
                    } else if let Some((stage, why)) = failed.remove(&key) {
                        let fix = if added.is_empty() {
                            "no added lines recorded".to_owned()
                        } else {
                            added.iter().take(3).cloned().collect::<Vec<_>>().join(" / ")
                        };
                        for f in &files {
                            recs.push(OpRecord {
                                source: format!("{f}:1"),
                                text: format!(
                                    "Step Check failed on {key} ({stage}: {why}); fixed by a later diff that passed, adding: {fix}"
                                ),
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        recs.reverse();
        let n = recs.len();
        if let Ok(mut g) = self.ops.write() {
            *g = recs;
        }
        n
    }
}

/// Structural segments for the semantic store (2026-11-10): off. They kept the large-context
/// benchmark at 20 of 20 with 7 to 15% fewer tokens but lost two to three answers on the small
/// development sets (`runs/2026-11-10-neumor-note-pyramid-segments.md`), so files are still read
/// in `CHUNK`-line windows.
pub const STRUCTURAL_SEGMENTS: bool = false;

/// Longest segment, in lines, and shortest one kept on its own (a shorter one joins the next).
pub const MAX_SEGMENT: usize = 60;
pub const MIN_SEGMENT: usize = 6;

/// Fixed windows with a last window shorter than `MIN_SEGMENT` lines joined to the one before
/// (2026-11-27: a file of 1,801 lines left a window of its closing brace alone, which a pack cited
/// as a piece that carries no meaning).
fn tail_joined(mut w: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if w.len() >= 2
        && let Some(&(s, e)) = w.last()
        && e + 1 - s < MIN_SEGMENT
    {
        w.pop();
        if let Some(prev) = w.last_mut() {
            prev.1 = e;
        }
    }
    w
}

/// Segments of a file by its structure (2026-11-10; the Drones spec: whole units, never a
/// unit cut in two): a Rust file is cut where an item starts (moved up over the doc comments
/// and attributes above it), a Markdown file at headings, any other file every `CHUNK` lines.
/// A segment longer than `MAX_SEGMENT` is cut into equal pieces; one shorter than
/// `MIN_SEGMENT` joins the next. Returns 1-based inclusive (start, end) line ranges covering
/// the file.
#[must_use]
pub fn segments(path: &str, lines: &[&str], items: &[usize]) -> Vec<(usize, usize)> {
    let n = lines.len();
    if n == 0 {
        return vec![(1, 1)];
    }
    let mut cuts: Vec<usize> = vec![1];
    if path.ends_with(".rs") {
        for &l in items {
            let mut s = l;
            while s > 1
                && lines.get(s - 2).is_some_and(|x| {
                    let t = x.trim_start();
                    t.starts_with("///") || t.starts_with("#[") || t.starts_with("//!")
                })
            {
                s -= 1;
            }
            cuts.push(s);
        }
    } else if path.ends_with(".md") {
        cuts.extend(lines.iter().enumerate().filter(|(_, x)| x.starts_with('#')).map(|(i, _)| i + 1));
    } else {
        cuts.extend((1..=n).step_by(CHUNK));
    }
    cuts.retain(|c| *c >= 1 && *c <= n);
    cuts.sort_unstable();
    cuts.dedup();
    let mut raw: Vec<(usize, usize)> = cuts.iter().enumerate().map(|(i, &s)| (s, cuts.get(i + 1).map_or(n, |x| x - 1))).collect();
    // Join short segments into the next one.
    let mut joined: Vec<(usize, usize)> = Vec::new();
    let mut carry: Option<usize> = None;
    for (s, e) in raw.drain(..) {
        let s = carry.take().unwrap_or(s);
        if e + 1 - s < MIN_SEGMENT && e < n {
            carry = Some(s);
        } else {
            joined.push((s, e));
        }
    }
    if let Some(s) = carry {
        match joined.last_mut() {
            Some(last) => last.1 = n.max(last.1),
            None => joined.push((s, n)),
        }
    }
    // Cut long segments into equal pieces.
    let mut out = Vec::new();
    for (s, e) in joined {
        let len = e + 1 - s;
        let parts = len.div_ceil(MAX_SEGMENT).max(1);
        let size = len.div_ceil(parts);
        let mut a = s;
        while a <= e {
            let b = (a + size - 1).min(e);
            out.push((a, b));
            a = b + 1;
        }
    }
    out
}

/// Structural units within a size budget (2026-11-10; the split-then-merge of cAST, Zhang et al.
/// 2025, https://arxiv.org/html/2506.15655, code MIT, method only; not used by the live scan).
/// Size is non-whitespace characters. A Rust file is cut at its top-level items (an item start
/// moved up over its doc comments and attributes; `impl` blocks count as items); the lines
/// before the first item (module notes and imports) are their own unit and are never merged
/// with code. An item over `budget` is entered: it is cut at the items one indentation level
/// inside it, or failing those at blank lines between statement groups at that level, or
/// failing those into equal pieces; each piece is packed the same way. Adjacent small pieces
/// are then merged greedily while the sum stays within `budget`. A Markdown file is cut at
/// headings and packed the same way; any other file every `CHUNK` lines. Every line is in
/// exactly one unit. Returns 1-based inclusive (start, end) ranges covering the file.
#[must_use]
pub fn segments_budgeted(path: &str, lines: &[&str], items: &[usize], budget: usize) -> Vec<(usize, usize)> {
    let n = lines.len();
    if n == 0 {
        return vec![(1, 1)];
    }
    let rust = path.ends_with(".rs");
    if !rust && !path.ends_with(".md") {
        return (1..=n).step_by(CHUNK).map(|s| (s, (s + CHUNK - 1).min(n))).collect();
    }
    let budget = budget.max(1);
    // Prefix sums of non-whitespace characters; size(s, e) for 1-based inclusive lines.
    let mut pre = Vec::with_capacity(n + 1);
    pre.push(0_usize);
    for l in lines {
        pre.push(pre.last().copied().unwrap_or(0) + l.chars().filter(|c| !c.is_whitespace()).count());
    }
    let size = |s: usize, e: usize| pre.get(e).copied().unwrap_or(0).saturating_sub(pre.get(s - 1).copied().unwrap_or(0));
    let line = |i: usize| lines.get(i - 1).copied().unwrap_or("");
    let indent = |i: usize| line(i).len() - line(i).trim_start().len();
    // Cut starts with their level: Rust items by indentation, Markdown headings at level 0.
    let mut starts: Vec<(usize, usize)> = Vec::new();
    if rust {
        let mut at: Vec<usize> = items.iter().copied().filter(|l| *l >= 1 && *l <= n).collect();
        at.extend((1..=n).filter(|i| {
            let t = line(*i);
            t.starts_with("impl ") || t.starts_with("impl<") || t.starts_with("unsafe impl")
        }));
        at.sort_unstable();
        at.dedup();
        for l in at {
            let level = indent(l);
            let mut st = l;
            while st > 1 && {
                let t = line(st - 1).trim_start();
                t.starts_with("///") || t.starts_with("#[")
            } {
                st -= 1;
            }
            starts.push((st, level));
        }
    } else {
        starts.extend((1..=n).filter(|i| line(*i).starts_with('#')).map(|i| (i, 0)));
    }
    starts.sort_unstable();
    starts.dedup_by_key(|x| x.0);
    // Pack the range [s, e] whose own level is `level` into pieces within the budget.
    fn merge(pieces: Vec<(usize, usize)>, budget: usize, size: &dyn Fn(usize, usize) -> usize) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        for (s, e) in pieces {
            match out.last_mut() {
                Some(last) if size(last.0, e) <= budget => last.1 = e,
                _ => out.push((s, e)),
            }
        }
        out
    }
    fn equal(s: usize, e: usize, budget: usize, size: &dyn Fn(usize, usize) -> usize) -> Vec<(usize, usize)> {
        let parts = size(s, e).div_ceil(budget).max(1);
        let target = size(s, e).div_ceil(parts).max(1);
        let mut out = Vec::new();
        let mut a = s;
        for b in s..=e {
            if b == e || size(a, b) >= target {
                out.push((a, b));
                a = b + 1;
            }
        }
        out
    }
    #[expect(clippy::too_many_arguments, reason = "one recursive packer over shared tables")]
    fn pack(
        s: usize,
        e: usize,
        level: usize,
        step: usize,
        budget: usize,
        starts: &[(usize, usize)],
        size: &dyn Fn(usize, usize) -> usize,
        line: &dyn Fn(usize) -> String,
    ) -> Vec<(usize, usize)> {
        if size(s, e) <= budget || s >= e {
            return vec![(s, e)];
        }
        let inner = level + step;
        let mut cuts: Vec<usize> = starts.iter().filter(|(l, lv)| *l > s && *l <= e && *lv == inner).map(|x| x.0).collect();
        let by_items = !cuts.is_empty();
        if !by_items {
            // Statement groups: a line at the inner level that follows a blank line.
            cuts = ((s + 1)..=e)
                .filter(|i| {
                    line(i - 1).trim().is_empty() && !line(*i).trim().is_empty() && line(*i).len() - line(*i).trim_start().len() == inner
                })
                .collect();
        }
        if cuts.is_empty() {
            return equal(s, e, budget, size);
        }
        let mut bounds = vec![s];
        bounds.extend(cuts);
        let mut pieces = Vec::new();
        for (i, a) in bounds.iter().enumerate() {
            let b = bounds.get(i + 1).map_or(e, |x| x - 1);
            if i > 0 && by_items {
                pieces.extend(pack(*a, b, inner, step, budget, starts, size, line));
            } else if size(*a, b) > budget {
                pieces.extend(equal(*a, b, budget, size));
            } else {
                pieces.push((*a, b));
            }
        }
        merge(pieces, budget, size)
    }
    let own = |i: usize| line(i).to_owned();
    let tops: Vec<usize> = starts.iter().filter(|x| x.1 == 0).map(|x| x.0).collect();
    let mut out = Vec::new();
    let first = tops.first().copied().unwrap_or(n + 1);
    if first > 1 {
        // The file's opening notes and imports: their own unit (cut to the budget if large).
        out.extend(if size(1, first - 1) > budget { equal(1, first - 1, budget, &size) } else { vec![(1, first - 1)] });
    }
    let mut body = Vec::new();
    for (i, a) in tops.iter().enumerate() {
        let b = tops.get(i + 1).map_or(n, |x| x - 1);
        body.extend(pack(*a, b, 0, if rust { 4 } else { usize::MAX / 2 }, budget, &starts, &size, &own));
    }
    out.extend(merge(body, budget, &size));
    out
}

/// A substrate over in-memory files, for section checks (no git, no cargo).
#[must_use]
pub fn from_files(files: &[(&str, &str)], embed: Embed) -> Substrate {
    let infos = files
        .iter()
        .map(|(p, t)| FileInfo::new(p, t))
        .collect();
    let mut s = Substrate {
        root: PathBuf::new(),
        version: 0,
        files: Vec::new(),
        low: Vec::new(),
        symbols: Vec::new(),
        crates: Vec::new(),
        chunks: Vec::new(),
        unit_budget: UNIT_FALLBACK,
        history: BTreeMap::new(),
        df: BTreeMap::new(),
        hidden_df: BTreeMap::new(),
        hidden_files: 0,
        words: 0,
        skip: Vec::new(),
        skipped: Vec::new(),
        imports: BTreeMap::new(),
        import_edges: Vec::new(),
        ops: Arc::new(std::sync::RwLock::new(Vec::new())),
        dense: Arc::new(std::sync::RwLock::new(None)),
        ref_graph: Arc::new(std::sync::RwLock::new(None)),
        embed,
    };
    s.index(infos);
    s.version = 1;
    s
}
