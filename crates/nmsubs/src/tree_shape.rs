//! A clean structural tree over the substrate (WP-C4 item 4, 2026-11-23): the repository root,
//! then directories (nested), modules (a directory holding a package manifest or a `src`
//! folder), files and the files' top-level symbols. It is shaped by the cost model of PageIndex's
//! tree optimizer (VectifyAI/PageIndex, MIT, `pageindex/tree_optimize.py`; the cost model and the
//! merge, expand and `key_items` rules were taken as ideas and written fresh), measured in tokens
//! instead of pages.
//!
//! Cost model. Reading the notes of a node's children costs `entry_tokens` per child (the
//! routing cost of the node). `S(v)` is the tokens of the code under v. A leaf costs `S(v)`. An
//! inner node costs its routing cost plus the larger of its residual (the tokens of chunks it holds
//! itself, from folded leaves) and its costliest child. The worst-case search cost of a tree is
//! the cost of its root: the most one walk from the root down to a single leaf can read.
//!
//! Shaping rules, applied bottom-up in path order, so the same substrate always gives the same
//! tree:
//! 1. Split: a symbol over `split_tokens` holding more than one chunk gets one child per chunk,
//!    and no node over `split_tokens` is ever collapsed.
//! 2. Chain: a plain directory whose only content is one directory or module gives way to it.
//! 3. Fold: the leaf children under `merge_tokens` are folded into their parent (their chunks
//!    become the parent's own, their titles its `key_items`) when that does not raise the
//!    parent's cost.
//! 4. Collapse: an inner node at or under `split_tokens` whose whole text costs no more than
//!    routing into it (`S(v) <= cost(v)`) becomes a leaf, its descendants' titles kept as
//!    `key_items`.
//!
//! Merging rules 2 to 4 never raise the worst-case search cost; rule 1 can, because a window
//! must hold the node it reads.
use crate::{Chunk, Substrate};
use std::collections::{BTreeMap, BTreeSet};

/// Routing cost per child entry in tokens (PageIndex `ENTRY_TOKENS`).
pub const ENTRY_TOKENS: usize = 45;
/// A node over this many tokens is split by its children and never collapsed (about one
/// tentacle window).
pub const SPLIT_TOKENS: usize = 4000;
/// Leaves under this many tokens are folded into their parent.
pub const MERGE_TOKENS: usize = 4 * ENTRY_TOKENS;
/// Names a structural summary lists before it counts the rest.
const SUMMARY_ITEMS: usize = 8;
/// Characters a summary keeps.
const SUMMARY_CHARS: usize = 300;
/// Characters a symbol name keeps.
const NAME_CHARS: usize = 80;
/// Weight of a file at tree distance `d` from a found file is `PRIOR_DECAY` to the power `d / 2`
/// (a sibling file, `d = 2`, weighs `PRIOR_DECAY`).
pub const PRIOR_DECAY: f32 = 0.5;
/// File names that make their directory a module.
const MANIFESTS: [&str; 5] = ["Cargo.toml", "package.json", "pyproject.toml", "setup.py", "go.mod"];

/// What a node stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Root,
    Dir,
    Module,
    File,
    Symbol,
}

/// The shaping limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub entry_tokens: usize,
    pub split_tokens: usize,
    pub merge_tokens: usize,
    /// Rule 4 on (off only to measure the unshaped tree).
    pub collapse: bool,
}

impl Default for Shape {
    fn default() -> Self {
        Self { entry_tokens: ENTRY_TOKENS, split_tokens: SPLIT_TOKENS, merge_tokens: MERGE_TOKENS, collapse: true }
    }
}

/// One node of the shaped tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    /// Place in pre-order (the root is 0, children follow their parent).
    pub id: usize,
    pub kind: Kind,
    /// Path for the root, a directory, a module or a file; the name for a symbol.
    pub title: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub depth: usize,
    /// Every chunk under the node, in index order.
    pub chunks: Vec<usize>,
    /// The chunks the node holds itself (folded or collapsed leaves, or a leaf's own).
    pub own: Vec<usize>,
    pub tokens: usize,
    /// The note provider's note, else a structural signature.
    pub summary: String,
    /// Titles of the nodes merged into this one.
    pub key_items: Vec<String>,
}

/// Where a file sits in the shaped tree.
#[derive(Clone, Debug, PartialEq, Eq)]
struct FileAt {
    /// The node the file hangs from: a file node's parent, or the node it was merged into.
    anchor: usize,
    tokens: usize,
    chunks: Vec<usize>,
}

/// The shaped tree of one substrate version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tree {
    pub version: u64,
    pub shape: Shape,
    pub nodes: Vec<Node>,
    files: BTreeMap<String, FileAt>,
    /// Roadmap Phase 1 (2026-11-24): the dependency edges on the structure: per file path, the
    /// files whose items it uses (calls, types, imports; from the reference graph), set by
    /// `set_dependencies`; empty until then.
    pub depends: BTreeMap<String, BTreeSet<String>>,
}

/// A note provider: given a chunk key (symbols), a file hash (files) or `dir:` and a path
/// (directories, modules, the root), the note written for it, if any.
pub type Notes<'a> = &'a dyn Fn(&str) -> Option<String>;

/// A node while the tree is shaped.
struct Raw {
    kind: Kind,
    title: String,
    children: Vec<usize>,
    own: Vec<usize>,
    key_items: Vec<String>,
    /// Files anchored at this node (a file node holds its own file).
    files: Vec<usize>,
    /// The file a file node stands for.
    file: Option<usize>,
    tokens: usize,
    cost: usize,
}

impl Raw {
    fn new(kind: Kind, title: String) -> Self {
        Self { kind, title, children: Vec::new(), own: Vec::new(), key_items: Vec::new(), files: Vec::new(), file: None, tokens: 0, cost: 0 }
    }
}

struct Builder<'a> {
    sub: &'a Substrate,
    shape: Shape,
    raw: Vec<Raw>,
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_owned()
    } else {
        let mut t: String = s.chars().take(n.saturating_sub(3)).collect();
        t.push_str("...");
        t
    }
}

/// The name a definition line gives (`pub fn walk(` gives `walk`), if it is one.
fn definition_name(line: &str) -> Option<String> {
    let mut rest = line.trim();
    loop {
        let before = rest;
        if let Some(r) = rest.strip_prefix("pub(") {
            rest = r.split_once(')').map_or("", |x| x.1).trim_start();
        }
        for m in ["pub ", "export ", "default ", "async ", "unsafe ", "abstract ", "final ", "public ", "private ", "protected "] {
            if let Some(r) = rest.strip_prefix(m) {
                rest = r.trim_start();
            }
        }
        if let Some(r) = rest.strip_prefix("const fn ") {
            rest = r;
            return ident(rest);
        }
        if before == rest {
            break;
        }
    }
    if let Some(r) = rest.strip_prefix("impl")
        && (r.starts_with(' ') || r.starts_with('<'))
    {
        let head = rest.split('{').next().unwrap_or(rest).trim();
        return Some(clip(head, NAME_CHARS));
    }
    for k in ["fn ", "struct ", "enum ", "trait ", "mod ", "type ", "union ", "macro_rules! ", "class ", "def ", "function ", "interface ", "const ", "static "] {
        if let Some(r) = rest.strip_prefix(k) {
            return ident(r.trim_start());
        }
    }
    None
}

fn ident(s: &str) -> Option<String> {
    let n: String = s.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    if n.is_empty() { None } else { Some(n) }
}

/// The top-level symbol a chunk belongs to: the top of its heading path, the type it sits in,
/// its piece name, the definition it sits in, else its first unindented definition line.
fn symbol_name(c: &Chunk) -> String {
    let top = c.heading.split(" > ").next().unwrap_or("").trim();
    if !top.is_empty() {
        return clip(top, NAME_CHARS);
    }
    let parent = c.piece.parent.split("::").next().unwrap_or("").trim();
    if !parent.is_empty() {
        return clip(parent, NAME_CHARS);
    }
    let name = c.piece.name.trim();
    if !name.is_empty() {
        return clip(name.split("::").next().unwrap_or(name), NAME_CHARS);
    }
    if let Some(n) = c.lead.lines().find_map(definition_name) {
        return n;
    }
    let unindented = c.text.lines().filter(|l| !l.starts_with(char::is_whitespace)).find_map(definition_name);
    unindented.or_else(|| c.text.lines().find_map(definition_name)).unwrap_or_else(|| format!("lines {}-{}", c.start, c.end))
}

impl Builder<'_> {
    fn push(&mut self, r: Raw) -> usize {
        self.raw.push(r);
        self.raw.len() - 1
    }
    fn chunk_tokens(&self, c: usize) -> usize {
        self.sub.chunks.get(c).map_or(0, |x| x.tokens)
    }
    fn own_tokens(&self, v: usize) -> usize {
        self.raw.get(v).map_or(0, |r| r.own.iter().map(|c| self.chunk_tokens(*c)).sum())
    }
    fn title(&self, v: usize) -> String {
        self.raw.get(v).map(|r| r.title.clone()).unwrap_or_default()
    }
    fn kind(&self, v: usize) -> Kind {
        self.raw.get(v).map_or(Kind::Symbol, |r| r.kind)
    }
    fn children(&self, v: usize) -> Vec<usize> {
        self.raw.get(v).map(|r| r.children.clone()).unwrap_or_default()
    }
    fn tokens(&self, v: usize) -> usize {
        self.raw.get(v).map_or(0, |r| r.tokens)
    }
    fn cost(&self, v: usize) -> usize {
        self.raw.get(v).map_or(0, |r| r.cost)
    }
    /// Tokens of every node, bottom-up (shaping moves chunks only inside a subtree).
    fn total(&mut self, v: usize) -> usize {
        let mut t = self.own_tokens(v);
        for c in self.children(v) {
            t += self.total(c);
        }
        if let Some(r) = self.raw.get_mut(v) {
            r.tokens = t;
        }
        t
    }
    /// The cost of `v` with children `kids` and residual `own` tokens.
    fn cost_of(&self, kids: &[usize], own: usize, tokens: usize) -> usize {
        if kids.is_empty() {
            tokens
        } else {
            self.shape.entry_tokens * kids.len() + kids.iter().map(|c| self.cost(*c)).max().unwrap_or(0).max(own)
        }
    }
    /// Move every chunk, file and title under `c` into `into`.
    fn absorb(&mut self, into: usize, c: usize) {
        let kids = self.children(c);
        let (own, files, mut items, title) = match self.raw.get_mut(c) {
            Some(r) => (std::mem::take(&mut r.own), std::mem::take(&mut r.files), std::mem::take(&mut r.key_items), r.title.clone()),
            None => return,
        };
        let mut names = vec![title];
        names.append(&mut items);
        if let Some(r) = self.raw.get_mut(into) {
            r.own.extend(own);
            r.files.extend(files);
            r.key_items.extend(names);
        }
        for k in kids {
            self.absorb(into, k);
        }
        if let Some(r) = self.raw.get_mut(c) {
            r.children.clear();
        }
    }
    /// Shape `v` bottom-up; returns the node that takes its place.
    fn shaped(&mut self, v: usize) -> usize {
        let mut kids: Vec<usize> = self.children(v).into_iter().map(|c| self.shaped(c)).collect();
        kids.sort_by(|a, b| self.kind(*a).cmp(&self.kind(*b)).then_with(|| self.title(*a).cmp(&self.title(*b))));
        if let Some(r) = self.raw.get_mut(v) {
            r.children.clone_from(&kids);
        }
        let kind = self.kind(v);
        // Rule 2: chain.
        let bare = self.raw.get(v).is_some_and(|r| r.own.is_empty() && r.files.is_empty());
        if kind == Kind::Dir && bare && kids.len() == 1 && kids.first().is_some_and(|c| matches!(self.kind(*c), Kind::Dir | Kind::Module)) {
            return kids.first().copied().unwrap_or(v);
        }
        // Rule 3: fold the small leaves when the cost does not rise.
        let small: Vec<usize> = kids.iter().copied().filter(|c| self.children(*c).is_empty() && self.tokens(*c) < self.shape.merge_tokens).collect();
        let own = self.own_tokens(v);
        if !small.is_empty() {
            let kept: Vec<usize> = kids.iter().copied().filter(|c| !small.contains(c)).collect();
            let folded = own + small.iter().map(|c| self.tokens(*c)).sum::<usize>();
            let before = self.cost_of(&kids, own, self.tokens(v));
            let after = self.cost_of(&kept, folded, self.tokens(v));
            if after <= before {
                for c in &small {
                    self.absorb(v, *c);
                }
                kids = kept;
                if let Some(r) = self.raw.get_mut(v) {
                    r.children.clone_from(&kids);
                }
            }
        }
        // Rule 4: collapse.
        let own = self.own_tokens(v);
        let tokens = self.tokens(v);
        let cost = self.cost_of(&kids, own, tokens);
        if self.shape.collapse && kind != Kind::Root && !kids.is_empty() && tokens <= self.shape.split_tokens && tokens <= cost {
            for c in &kids {
                self.absorb(v, *c);
            }
            kids.clear();
            if let Some(r) = self.raw.get_mut(v) {
                r.children.clear();
            }
        }
        let cost = self.cost_of(&kids, self.own_tokens(v), tokens);
        if let Some(r) = self.raw.get_mut(v) {
            r.cost = cost;
        }
        v
    }
}

/// Build the shaped tree of `sub`. `notes` gives the written notes when there are any; every
/// other node gets a structural signature.
#[must_use]
pub fn build(sub: &Substrate, notes: Option<Notes<'_>>, shape: Shape) -> Tree {
    let mut b = Builder { sub, shape, raw: Vec::new() };
    let root = b.push(Raw::new(Kind::Root, ".".to_owned()));
    // Modules: a directory with a manifest, or with a `src` folder.
    let mut modules: BTreeSet<String> = BTreeSet::new();
    for f in &sub.files {
        let (dir, name) = f.path.rsplit_once('/').unwrap_or(("", f.path.as_str()));
        if MANIFESTS.contains(&name) {
            modules.insert(dir.to_owned());
        }
        let segs: Vec<&str> = f.path.split('/').collect();
        for (i, s) in segs.iter().enumerate() {
            if *s == "src" && i + 1 < segs.len() {
                modules.insert(segs.get(..i).map(|x| x.join("/")).unwrap_or_default());
            }
        }
    }
    let mut by_file: Vec<Vec<usize>> = vec![Vec::new(); sub.files.len()];
    for (i, c) in sub.chunks.iter().enumerate() {
        if let Some(v) = by_file.get_mut(c.file) {
            v.push(i);
        }
    }
    for v in &mut by_file {
        v.sort_by_key(|c| (sub.chunks.get(*c).map_or(0, |x| x.start), *c));
    }
    let mut dirs: BTreeMap<String, usize> = BTreeMap::new();
    dirs.insert(String::new(), root);
    let mut order: Vec<usize> = (0..sub.files.len()).collect();
    order.sort_by(|a, c| sub.files.get(*a).map(|f| &f.path).cmp(&sub.files.get(*c).map(|f| &f.path)));
    for fi in order {
        let Some(f) = sub.files.get(fi) else { continue };
        let dir = f.path.rsplit_once('/').map_or("", |x| x.0);
        // Make the directory chain down to `dir`.
        let mut parent = root;
        let mut at = String::new();
        for seg in dir.split('/').filter(|s| !s.is_empty()) {
            if !at.is_empty() {
                at.push('/');
            }
            at.push_str(seg);
            let id = if let Some(id) = dirs.get(&at) {
                *id
            } else {
                let kind = if modules.contains(&at) { Kind::Module } else { Kind::Dir };
                let id = b.push(Raw::new(kind, at.clone()));
                dirs.insert(at.clone(), id);
                if let Some(r) = b.raw.get_mut(parent) {
                    r.children.push(id);
                }
                id
            };
            parent = id;
        }
        let mut file = Raw::new(Kind::File, f.path.clone());
        file.file = Some(fi);
        file.files.push(fi);
        let file_id = b.push(file);
        if let Some(r) = b.raw.get_mut(parent) {
            r.children.push(file_id);
        }
        // Top-level symbols: runs of chunks with one name.
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        for c in by_file.get(fi).cloned().unwrap_or_default() {
            let name = sub.chunks.get(c).map(symbol_name).unwrap_or_default();
            match groups.last_mut() {
                Some((n, v)) if *n == name => v.push(c),
                _ => groups.push((name, vec![c])),
            }
        }
        for (name, cs) in groups {
            let tokens: usize = cs.iter().map(|c| b.chunk_tokens(*c)).sum();
            let mut sym = Raw::new(Kind::Symbol, name.clone());
            // Rule 1: split a symbol over the limit by its chunks.
            if tokens > shape.split_tokens && cs.len() > 1 {
                let sym_id = b.push(sym);
                for c in cs {
                    let (s, e) = sub.chunks.get(c).map_or((0, 0), |x| (x.start, x.end));
                    let mut part = Raw::new(Kind::Symbol, format!("{name} lines {s}-{e}"));
                    part.own.push(c);
                    let pid = b.push(part);
                    if let Some(r) = b.raw.get_mut(sym_id) {
                        r.children.push(pid);
                    }
                }
                if let Some(r) = b.raw.get_mut(file_id) {
                    r.children.push(sym_id);
                }
            } else {
                sym.own = cs;
                let sym_id = b.push(sym);
                if let Some(r) = b.raw.get_mut(file_id) {
                    r.children.push(sym_id);
                }
            }
        }
    }
    b.total(root);
    let top = b.shaped(root);
    finish(&b, top, sub, notes, &by_file)
}

/// Number the shaped nodes in pre-order and write their summaries.
fn finish(b: &Builder<'_>, top: usize, sub: &Substrate, notes: Option<Notes<'_>>, by_file: &[Vec<usize>]) -> Tree {
    let mut nodes: Vec<Node> = Vec::new();
    let mut raw_of: Vec<usize> = Vec::new();
    let mut stack: Vec<(usize, Option<usize>, usize)> = vec![(top, None, 0)];
    while let Some((r, parent, depth)) = stack.pop() {
        let Some(x) = b.raw.get(r) else { continue };
        let id = nodes.len();
        if let Some(p) = parent.and_then(|p| nodes.get_mut(p)) {
            p.children.push(id);
        }
        let mut own = x.own.clone();
        own.sort_unstable();
        nodes.push(Node {
            id,
            kind: x.kind,
            title: x.title.clone(),
            parent,
            children: Vec::new(),
            depth,
            chunks: Vec::new(),
            own,
            tokens: x.tokens,
            summary: String::new(),
            key_items: x.key_items.clone(),
        });
        raw_of.push(r);
        for c in x.children.iter().rev() {
            stack.push((*c, Some(id), depth + 1));
        }
    }
    // Chunks of every subtree, children before parents (children have larger ids).
    for i in (0..nodes.len()).rev() {
        let mut all: Vec<usize> = nodes.get(i).map(|n| n.own.clone()).unwrap_or_default();
        let kids = nodes.get(i).map(|n| n.children.clone()).unwrap_or_default();
        for k in kids {
            if let Some(n) = nodes.get(k) {
                all.extend_from_slice(&n.chunks);
            }
        }
        all.sort_unstable();
        if let Some(n) = nodes.get_mut(i) {
            n.chunks = all;
        }
    }
    // Summaries.
    for i in (0..nodes.len()).rev() {
        let Some(n) = nodes.get(i) else { continue };
        let key = match n.kind {
            Kind::File => raw_of.get(i).and_then(|r| b.raw.get(*r)).and_then(|r| r.file).and_then(|f| sub.files.get(f)).map(|f| f.hash.clone()),
            Kind::Symbol => n.chunks.first().and_then(|c| sub.chunks.get(*c)).map(|c| c.key.clone()),
            Kind::Root | Kind::Dir | Kind::Module => Some(format!("dir:{}", n.title)),
        };
        let written = key.and_then(|k| notes.and_then(|f| f(&k))).map(|s| s.trim().to_owned()).filter(|s| !s.is_empty());
        let summary = written.unwrap_or_else(|| {
            if n.kind == Kind::Symbol && n.children.is_empty() {
                let c = n.chunks.first().and_then(|c| sub.chunks.get(*c));
                let sig = c.map(|c| c.piece.signature.trim().to_owned()).filter(|s| !s.is_empty());
                sig.or_else(|| c.and_then(|c| c.text.lines().map(str::trim).find(|l| !l.is_empty()).map(str::to_owned))).unwrap_or_default()
            } else {
                let short = |t: &str| -> String {
                    match n.kind {
                        Kind::File | Kind::Symbol => t.to_owned(),
                        _ => t.strip_prefix(&format!("{}/", n.title)).unwrap_or(t).to_owned(),
                    }
                };
                let mut names: Vec<String> = n.children.iter().filter_map(|c| nodes.get(*c)).map(|c| short(&c.title)).collect();
                names.extend(n.key_items.iter().map(|t| short(t)));
                let more = names.len().saturating_sub(SUMMARY_ITEMS);
                names.truncate(SUMMARY_ITEMS);
                let mut s = names.join(", ");
                if more > 0 {
                    s.push_str(&format!(" (+{more} more)"));
                }
                s
            }
        });
        let summary = clip(&summary, SUMMARY_CHARS);
        if let Some(n) = nodes.get_mut(i) {
            n.summary = summary;
        }
    }
    // Where each file hangs.
    let mut files: BTreeMap<String, FileAt> = BTreeMap::new();
    for (i, n) in nodes.iter().enumerate() {
        let Some(r) = raw_of.get(i).and_then(|r| b.raw.get(*r)) else { continue };
        for fi in &r.files {
            let Some(f) = sub.files.get(*fi) else { continue };
            let anchor = if n.kind == Kind::File && r.file == Some(*fi) { n.parent.unwrap_or(i) } else { i };
            let chunks = by_file.get(*fi).cloned().unwrap_or_default();
            let tokens = chunks.iter().map(|c| sub.chunks.get(*c).map_or(0, |x| x.tokens)).sum();
            files.insert(f.path.clone(), FileAt { anchor, tokens, chunks });
        }
    }
    Tree { version: sub.version, shape: b.shape, nodes, files, depends: BTreeMap::new() }
}

impl Tree {
    /// The node titled `title` (a path, or a symbol name), first in pre-order.
    #[must_use]
    pub fn find(&self, title: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.title == title)
    }
    /// The chunks of a file, by line order (empty for a file the tree does not hold).
    #[must_use]
    pub fn file_chunks(&self, path: &str) -> &[usize] {
        self.files.get(path).map_or(&[], |f| f.chunks.as_slice())
    }
    /// The tokens of a file.
    #[must_use]
    pub fn file_tokens(&self, path: &str) -> usize {
        self.files.get(path).map_or(0, |f| f.tokens)
    }
    /// The files the tree holds, by path.
    pub fn file_paths(&self) -> impl Iterator<Item = &String> {
        self.files.keys()
    }
    /// The search cost of node `id`: its tokens for a leaf, else its routing cost plus the larger
    /// of its residual and its costliest child.
    #[must_use]
    pub fn cost(&self, id: usize) -> usize {
        let Some(n) = self.nodes.get(id) else { return 0 };
        if n.children.is_empty() {
            return n.tokens;
        }
        let own: usize = n.tokens - n.children.iter().filter_map(|c| self.nodes.get(*c)).map(|c| c.tokens).sum::<usize>().min(n.tokens);
        self.shape.entry_tokens * n.children.len() + n.children.iter().map(|c| self.cost(*c)).max().unwrap_or(0).max(own)
    }
    /// The worst-case search cost: the most one walk from the root to a single leaf reads, in
    /// tokens (routing reads of every node on the way plus the leaf's text).
    #[must_use]
    pub fn worst_cost(&self) -> usize {
        self.cost(0)
    }
    /// Steps up from `a` and from `b` to their lowest common ancestor.
    fn ups(&self, a: usize, b: usize) -> (usize, usize) {
        let depth = |x: usize| self.nodes.get(x).map_or(0, |n| n.depth);
        let up = |x: usize| self.nodes.get(x).and_then(|n| n.parent).unwrap_or(x);
        let (mut x, mut y, mut ua, mut ub) = (a, b, 0, 0);
        while depth(x) > depth(y) {
            x = up(x);
            ua += 1;
        }
        while depth(y) > depth(x) {
            y = up(y);
            ub += 1;
        }
        while x != y {
            let (nx, ny) = (up(x), up(y));
            if nx == x && ny == y {
                break;
            }
            x = nx;
            y = ny;
            ua += 1;
            ub += 1;
        }
        (ua, ub)
    }
    /// Tree distance between two files: two plus twice the height of the smallest subtree that
    /// holds both, counted from their anchors (files hanging from one node are at 2, a file in
    /// the parent's node at 4, files of sibling directories at 6). It ranks by the deepest
    /// shared ancestor, so a cousin in the same module comes before a file elsewhere at the
    /// same number of edges. 0 for the same file, `None` for a file the tree does not hold.
    #[must_use]
    pub fn file_distance(&self, a: &str, b: &str) -> Option<usize> {
        let (fa, fb) = (self.files.get(a)?, self.files.get(b)?);
        if a == b {
            return Some(0);
        }
        let (ua, ub) = self.ups(fa.anchor, fb.anchor);
        Some(2 + 2 * ua.max(ub))
    }
    /// Roadmap Phase 0 item 2 (W2 fallback): the project's structural outline, the nodes at most
    /// `max_depth` deep as indented lines (title, then its summary or its key items), at most
    /// `max_chars` characters (whole lines only). Used while the writer holds no project outline.
    #[must_use]
    pub fn structural_outline(&self, max_depth: usize, max_chars: usize) -> String {
        let mut s = String::new();
        for n in self.nodes.iter().filter(|n| n.depth <= max_depth) {
            let tail = if !n.summary.is_empty() {
                format!(": {}", n.summary)
            } else if n.key_items.is_empty() {
                String::new()
            } else {
                format!(" [{}]", n.key_items.join(", "))
            };
            let line = format!("{}{}{}\n", "  ".repeat(n.depth), n.title, tail);
            if s.len() + line.len() > max_chars {
                break;
            }
            s.push_str(&line);
        }
        s
    }
    /// Sets the file dependency edges (`depends`): `(from, to)` file paths, self edges left out.
    pub fn set_dependencies(&mut self, edges: impl IntoIterator<Item = (String, String)>) {
        self.depends.clear();
        for (a, b) in edges {
            if a != b {
                self.depends.entry(a).or_default().insert(b);
            }
        }
    }
    /// The dependency edges rolled up to folders: per folder, the other folders whose files its
    /// files use (the L3 level of the world model).
    #[must_use]
    pub fn module_depends(&self) -> BTreeMap<String, BTreeSet<String>> {
        let dir = |p: &str| p.rsplit_once('/').map_or_else(|| ".".to_owned(), |(d, _)| d.to_owned());
        let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (a, tos) in &self.depends {
            for b in tos {
                let (da, db) = (dir(a), dir(b));
                if da != db {
                    out.entry(da).or_default().insert(db);
                }
            }
        }
        out
    }
    /// The shaped tree as indented lines (kind, title, tokens, key items), for reports and checks.
    #[must_use]
    pub fn outline(&self) -> String {
        let mut s = String::new();
        for n in &self.nodes {
            let items = if n.key_items.is_empty() { String::new() } else { format!(" [{}]", n.key_items.join(", ")) };
            s.push_str(&format!("{}{:?} {} {}{}\n", "  ".repeat(n.depth), n.kind, n.title, n.tokens, items));
        }
        s
    }
}

/// The files nearest `file_path` in the tree, best first: siblings in the same module or
/// directory, then the nearest cousins; ties by more tokens, then path. Empty when the tree
/// does not hold the file.
#[must_use]
pub fn neighbours(tree: &Tree, file_path: &str, k: usize) -> Vec<String> {
    if !tree.files.contains_key(file_path) {
        return Vec::new();
    }
    let mut v: Vec<(usize, usize, &String)> = tree
        .files
        .iter()
        .filter(|(p, _)| p.as_str() != file_path)
        .filter_map(|(p, f)| tree.file_distance(file_path, p).map(|d| (d, f.tokens, p)))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(a.2.cmp(b.2)));
    v.into_iter().take(k).map(|x| x.2.clone()).collect()
}

/// Files near the found ones, best first, each with a weight in (0, 1]: per found file
/// `PRIOR_DECAY` to the power of half the tree distance, combined over the found files as a
/// noisy-or (a file near several found files weighs more). Found files are left out; ties by
/// more tokens, then path.
#[must_use]
pub fn prior(tree: &Tree, found_files: &[String], k: usize) -> Vec<(String, f32)> {
    let found: BTreeSet<&str> = found_files.iter().map(String::as_str).filter(|p| tree.files.contains_key(*p)).collect();
    if found.is_empty() {
        return Vec::new();
    }
    let mut v: Vec<(f32, usize, &String)> = tree
        .files
        .iter()
        .filter(|(p, _)| !found.contains(p.as_str()))
        .map(|(p, f)| {
            let miss: f32 = found
                .iter()
                .filter_map(|q| tree.file_distance(q, p))
                .map(|d| 1.0 - PRIOR_DECAY.powf(d as f32 / 2.0))
                .product();
            (1.0 - miss, f.tokens, p)
        })
        .filter(|x| x.0 > 0.0)
        .collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(b.2)));
    v.into_iter().take(k).map(|x| (x.2.clone(), x.0)).collect()
}

/// Synthetic files for the checks: a workspace with two crates and a guide.
#[cfg(test)]
fn sample_files() -> Vec<(&'static str, String)> {
    let mut lib = String::new();
    for i in 0..6 {
        lib.push_str(&format!("/// Step {i} of the walk.\npub fn step_{i}(x: usize) -> usize {{\n"));
        for j in 0..12 {
            lib.push_str(&format!("    let v{j} = x.wrapping_mul({j}).wrapping_add({i});\n"));
        }
        lib.push_str("    x\n}\n\n");
    }
    let mut guide = String::from("# Guide\n\n");
    for s in 0..3 {
        guide.push_str(&format!("## Part {s}\n\n"));
        for l in 0..40 {
            guide.push_str(&format!("Line {l} of part {s} explains how the walk reads the tree in order.\n\n"));
        }
    }
    vec![
        ("Cargo.toml", "[workspace]\nmembers = [\"crates/alpha\", \"crates/beta\"]\n".to_owned()),
        ("crates/alpha/Cargo.toml", "[package]\nname = \"alpha\"\n".to_owned()),
        ("crates/alpha/src/lib.rs", lib.clone()),
        ("crates/alpha/src/walk.rs", lib.replace("step_", "walk_")),
        ("crates/alpha/src/tiny.rs", "pub const T: u8 = 1;\n".to_owned()),
        ("crates/beta/Cargo.toml", "[package]\nname = \"beta\"\n".to_owned()),
        ("crates/beta/src/lib.rs", lib.replace("step_", "beta_")),
        ("docs/guide.md", guide),
    ]
}

#[cfg(test)]
fn sample(reversed: bool) -> Substrate {
    let mut files = sample_files();
    if reversed {
        files.reverse();
    }
    let refs: Vec<(&str, &str)> = files.iter().map(|(p, t)| (*p, t.as_str())).collect();
    let embed: crate::Embed = std::sync::Arc::new(|_: &str| vec![1.0]);
    crate::from_files(&refs, embed)
}

/// Self-check of the `tree_shape` sub-component: deterministic, siblings before cousins.
pub(crate) fn check() -> Result<(), String> {
    let embed: crate::Embed = std::sync::Arc::new(|_: &str| vec![1.0]);
    let s = crate::from_files(
        &[
            ("a/Cargo.toml", "[package]\nname = \"a\"\n"),
            ("a/src/one.rs", "pub fn one() -> u8 {\n    1\n}\n"),
            ("a/src/two.rs", "pub fn two() -> u8 {\n    2\n}\n"),
            ("b/src/three.rs", "pub fn three() -> u8 {\n    3\n}\n"),
        ],
        embed,
    );
    let t = build(&s, None, Shape::default());
    if t != build(&s, None, Shape::default()) {
        return Err("nmsubs.tree_shape: two builds differ".into());
    }
    let n = neighbours(&t, "a/src/one.rs", 1);
    if n.first().map(String::as_str) != Some("a/src/two.rs") {
        return Err(format!("nmsubs.tree_shape: neighbours {n:?}\n{}", t.outline()));
    }
    let p = prior(&t, &["a/src/one.rs".to_owned()], 3);
    if p.first().map(|x| x.0.as_str()) != Some("a/src/two.rs") || p.windows(2).any(|w| w.first().map(|x| x.1) < w.get(1).map(|x| x.1)) {
        return Err(format!("nmsubs.tree_shape: prior {p:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_check_passes() {
        assert_eq!(check(), Ok(()));
    }

    #[test]
    fn deterministic_over_input_order() {
        let a = sample(false);
        let b = sample(true);
        let ta = build(&a, None, Shape::default());
        assert_eq!(ta, build(&a, None, Shape::default()));
        assert_eq!(ta.outline(), build(&b, None, Shape::default()).outline());
        // Pre-order ids and parents agree.
        for n in &ta.nodes {
            assert!(n.children.iter().all(|c| ta.nodes.get(*c).is_some_and(|x| x.parent == Some(n.id) && x.depth == n.depth + 1)));
        }
    }

    #[test]
    fn modules_and_files_found() {
        let s = sample(false);
        let t = build(&s, None, Shape { collapse: false, merge_tokens: 0, ..Shape::default() });
        assert_eq!(t.find("crates/alpha").map(|n| n.kind), Some(Kind::Module));
        assert_eq!(t.find("crates/alpha/src/lib.rs").map(|n| n.kind), Some(Kind::File));
        assert_eq!(t.find("docs").map(|n| n.kind), Some(Kind::Dir));
        // Every chunk is covered exactly once by the root.
        let root = t.nodes.first().map(|n| n.chunks.clone()).unwrap_or_default();
        assert_eq!(root, (0..s.chunks.len()).collect::<Vec<_>>());
        let lib = t.find("crates/alpha/src/lib.rs").map(|n| n.summary.clone()).unwrap_or_default();
        assert!(lib.contains("step_0"), "file signature lists its symbols: {lib}");
    }

    #[test]
    fn dependencies_roll_up_to_folders() {
        let s = sample(false);
        let mut t = build(&s, None, Shape::default());
        t.set_dependencies([("crates/alpha/src/lib.rs".to_owned(), "crates/beta/src/lib.rs".to_owned()), ("crates/alpha/src/lib.rs".to_owned(), "crates/alpha/src/lib.rs".to_owned())]);
        assert_eq!(t.depends.get("crates/alpha/src/lib.rs").map(BTreeSet::len), Some(1));
        let m = t.module_depends();
        assert!(m.get("crates/alpha/src").is_some_and(|s| s.contains("crates/beta/src")));
    }

    #[test]
    fn structural_outline_lists_top_levels_within_its_room() {
        let s = sample(false);
        let t = build(&s, None, Shape::default());
        let o = t.structural_outline(2, 10_000);
        assert!(o.contains("crates/alpha"), "the outline names the crates: {o}");
        assert!(o.lines().all(|l| l.len() - l.trim_start().len() <= 4), "no line deeper than two levels: {o}");
        let short = t.structural_outline(2, 40);
        assert!(short.len() <= 40 && short.ends_with('\n') || short.is_empty(), "whole lines within the room: {short:?}");
    }

    #[test]
    fn split_rule() {
        let s = sample(false);
        let t = build(&s, None, Shape { split_tokens: 1, merge_tokens: 0, entry_tokens: ENTRY_TOKENS, collapse: true });
        let guide = t.nodes.iter().find(|n| n.kind == Kind::Symbol && n.title == "Guide");
        let parts = guide.map_or(0, |n| n.children.len());
        assert!(parts > 1, "the oversized guide symbol is split by its chunks:\n{}", t.outline());
        // Nothing over the limit is collapsed.
        assert!(t.nodes.iter().all(|n| n.key_items.is_empty()), "{}", t.outline());
    }

    #[test]
    fn merge_rules_never_raise_cost() {
        let s = sample(false);
        let flat = build(&s, None, Shape { split_tokens: usize::MAX, merge_tokens: 0, entry_tokens: ENTRY_TOKENS, collapse: false });
        let shaped = build(&s, None, Shape { split_tokens: usize::MAX, ..Shape::default() });
        assert!(shaped.worst_cost() <= flat.worst_cost(), "{} > {}", shaped.worst_cost(), flat.worst_cost());
        assert!(shaped.nodes.len() < flat.nodes.len());
        // The tiny file is merged away and kept as a key item.
        assert!(shaped.find("crates/alpha/src/tiny.rs").is_none());
        assert!(shaped.nodes.iter().any(|n| n.key_items.iter().any(|k| k == "crates/alpha/src/tiny.rs")), "{}", shaped.outline());
        // A file of one symbol is collapsed into a leaf.
        assert!(shaped.nodes.iter().all(|n| !(n.kind == Kind::File && n.children.len() == 1)), "{}", shaped.outline());
    }

    #[test]
    fn notes_replace_signatures() {
        let s = sample(false);
        let hash = s.files.iter().find(|f| f.path == "crates/beta/src/lib.rs").map(|f| f.hash.clone()).unwrap_or_default();
        let notes = move |k: &str| if k == hash { Some("Beta's walk steps.".to_owned()) } else { None };
        let t = build(&s, Some(&notes), Shape { collapse: false, merge_tokens: 0, ..Shape::default() });
        assert_eq!(t.find("crates/beta/src/lib.rs").map(|n| n.summary.as_str()), Some("Beta's walk steps."));
    }

    #[test]
    fn neighbours_and_prior_order() {
        let s = sample(false);
        let t = build(&s, None, Shape::default());
        let n = neighbours(&t, "crates/alpha/src/walk.rs", 10);
        let pos = |p: &str| n.iter().position(|x| x == p).unwrap_or(usize::MAX);
        assert!(pos("crates/alpha/src/lib.rs") < pos("crates/beta/src/lib.rs"), "{n:?}\n{}", t.outline());
        assert!(pos("crates/alpha/src/tiny.rs") < pos("crates/beta/src/lib.rs"), "{n:?}");
        assert!(pos("crates/beta/src/lib.rs") < pos("docs/guide.md"), "{n:?}");
        assert!(!n.iter().any(|x| x == "crates/alpha/src/walk.rs"));
        // Same distance: more tokens first.
        let lib = pos("crates/alpha/src/lib.rs");
        let tiny = pos("crates/alpha/src/tiny.rs");
        if t.file_distance("crates/alpha/src/walk.rs", "crates/alpha/src/lib.rs") == t.file_distance("crates/alpha/src/walk.rs", "crates/alpha/src/tiny.rs") {
            assert!(lib < tiny);
        }
        assert_eq!(neighbours(&t, "missing.rs", 3), Vec::<String>::new());
        let p = prior(&t, &["crates/alpha/src/walk.rs".to_owned()], 10);
        assert!(p.windows(2).all(|w| w.first().map(|x| x.1) >= w.get(1).map(|x| x.1)), "{p:?}");
        assert!(p.iter().all(|x| x.1 > 0.0 && x.1 <= 1.0));
        let w = |f: &str| p.iter().find(|x| x.0 == f).map_or(0.0, |x| x.1);
        assert!(w("crates/alpha/src/lib.rs") > w("crates/beta/src/lib.rs"));
        // Near two found files weighs more than near one.
        let two = prior(&t, &["crates/alpha/src/walk.rs".to_owned(), "crates/alpha/src/lib.rs".to_owned()], 10);
        let w2 = two.iter().find(|x| x.0 == "crates/alpha/src/tiny.rs").map_or(0.0, |x| x.1);
        assert!(w2 >= w("crates/alpha/src/tiny.rs"));
        assert!(!two.iter().any(|x| x.0 == "crates/alpha/src/lib.rs"));
    }

    #[test]
    fn definition_names() {
        assert_eq!(definition_name("pub(crate) async fn go_on(x: u8)"), Some("go_on".to_owned()));
        assert_eq!(definition_name("pub const fn k() {}"), Some("k".to_owned()));
        assert_eq!(definition_name("export default class Walker {"), Some("Walker".to_owned()));
        assert_eq!(definition_name("impl<T> Tree for Forest<T> {"), Some("impl<T> Tree for Forest<T>".to_owned()));
        assert_eq!(definition_name("let x = 1;"), None);
    }
}
