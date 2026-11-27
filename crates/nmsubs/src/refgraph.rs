//! Reference graph over items (2026-11-10, refinement step 1, item 4; architecture by Luis Jake
//! Gabriel III; analysis `docs/2026-11-10-neumor-analysis-c-routing-and-globalized-context-stack.md` sections 7.2 and
//! 9.3). The whole-code-base layer of G1 had file paths and item signatures but no edges, so the
//! structural relation between found spans (who refers to whom) was absent. This graph adds it,
//! from what the substrate already extracts (`Substrate::symbols`, `Substrate::chunks`): a chunk
//! that holds an item's name as an identifier refers to the chunk that defines the item.
//!
//! Held in RAM only (a few hundred kilobytes for this tree), never written to disk. It is built
//! when first asked for on a substrate version and rebuilt on sync when it was in use
//! (`Substrate::refresh_refs`), reusing the identifier sets of chunks whose content key did not
//! change. Nothing is encoded: identifiers are read from the chunk text.
//!
//! Edge weights follow the published repository-map method of Aider (`aider/repomap.py`,
//! `get_ranked_tags`, Apache-2.0, method only, no code taken): a name defined in more than five
//! files counts one tenth (common names such as `new` relate little), and a reference spreads
//! its weight over the name's definitions. The rank is personalised PageRank with the restart
//! vector from the seeds and the damping 0.85 of the original method (Brin and Page 1998; the
//! networkx default that Aider uses); iterations stop when the change falls below one millionth
//! of the mass, at most the count that bound implies for that damping.
//!
//! Specific relations (2026-11-10, refinement step 4a, fault F27; architecture by Luis Jake
//! Gabriel III; analysis `docs/2026-11-10-neumor-analysis-h-natural-systems-for-the-globalized-context-stack.md`
//! sections 0.2 and G). The graph above relates two chunks through any shared item name, which
//! on a real tree joins almost every chunk to every other, so a relation carries no locality.
//! `Specific` is the second form, built beside the first (the first stays as it was and stays
//! selectable): an edge is kept only through an identifier that few chunks hold, it is weighted
//! by how specific the identifier is (the natural logarithm of chunks over the chunks holding
//! it, spread over the name's definitions), and it is typed where the structural view can tell
//! the kinds apart (`EdgeKind`). The rarity limit is derived from the tree (`Limit::Derived`):
//! the median, over the item names followed, of the number of chunks holding the name, so a
//! name is specific when it is held by no more chunks than the typical defined name of this
//! tree; the stated fallback (`RARE_FALLBACK`, 10, the value measured in the analysis) is used
//! only when the tree defines no names.
use crate::Substrate;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

/// Damping of the personalised rank (the published PageRank value, stated above).
pub const DAMPING: f32 = 0.85;
/// A name defined in more files than this counts `COMMON_WEIGHT` (Aider's rule, stated above).
pub const COMMON_FILES: usize = 5;
pub const COMMON_WEIGHT: f32 = 0.1;
/// Names shorter than this are not followed (one- and two-letter names relate nothing).
pub const MIN_NAME: usize = 3;
/// Fallback rarity limit of the specific relations (analysis H section 0.2: names held by at
/// most 10 chunks), used only when the tree defines no names to derive the limit from.
pub const RARE_FALLBACK: usize = 10;

/// The kind of a specific relation, from the using chunk to the defining chunk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeKind {
    /// The kinds of the name's definitions disagree or are unknown: only the name relates.
    #[default]
    Names,
    /// A constant, static, module or macro and a chunk using it (definition to use).
    DefUse,
    /// A function and a chunk calling it (caller to callee).
    Call,
    /// A struct, enum, type alias or trait and a chunk using it (type use).
    TypeUse,
    /// A test chunk and the item it tests (test to tested item).
    Test,
    /// A piece implementing or extending a trait, interface or base (implementer to trait).
    Implements,
    /// A piece using a name another piece gives a value: a field default, a struct field, a
    /// binding (user to the piece that assigns it).
    DefinesUsed,
}
/// Edge kinds.
pub const KINDS: usize = 7;
impl EdgeKind {
    pub const ALL: [Self; KINDS] = [Self::Names, Self::DefUse, Self::Call, Self::TypeUse, Self::Test, Self::Implements, Self::DefinesUsed];
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Names => "names",
            Self::DefUse => "definition-use",
            Self::Call => "call",
            Self::TypeUse => "type-use",
            Self::Test => "test",
            Self::Implements => "implements",
            Self::DefinesUsed => "defines-what-it-uses",
        }
    }
    #[must_use]
    pub fn index(self) -> usize {
        Self::ALL.iter().position(|k| *k == self).unwrap_or(0)
    }
    /// From the kinds of a name's definitions (`repo::Symbol::kind`) and whether the using
    /// chunk is test code.
    fn of(kinds: &BTreeSet<&str>, user_is_test: bool) -> Self {
        if user_is_test {
            return Self::Test;
        }
        // Rust kinds, then the kinds of the generic tagger (`tags`).
        let class = |k: &str| match k {
            "fn" | "function" | "method" => Self::Call,
            "struct" | "enum" | "type" | "trait" | "class" | "interface" => Self::TypeUse,
            "const" | "static" | "mod" | "macro" | "variable" | "namespace" | "selector" | "keyframes" => Self::DefUse,
            _ => Self::Names,
        };
        let mut it = kinds.iter().map(|k| class(k));
        match it.next() {
            Some(first) if it.all(|k| k == first) => first,
            _ => Self::Names,
        }
    }
}

/// How the rarity limit of the specific relations is set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    /// From the tree: the median chunk count of the item names followed (module notes).
    Derived,
    /// A stated count.
    Fixed(usize),
    /// No limit: every name relates (edges still weighted by specificity).
    Unlimited,
}

/// One specific relation as a chunk sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    /// The other chunk.
    pub to: u32,
    pub kind: EdgeKind,
    /// `Uses`: this chunk uses an item the other defines; `UsedBy`: the reverse.
    pub dir: Relation,
    /// Specificity of the identifiers joining the two, summed.
    pub weight: f32,
}

/// The specific relations of a substrate version (module notes). Held in RAM only.
#[derive(Clone, Debug, Default)]
pub struct Specific {
    /// The rarity limit used (chunks holding a name; `usize::MAX` when unlimited) and how it
    /// was set.
    pub limit: usize,
    pub rule: String,
    /// Item names kept and names dropped as unspecific.
    pub names_kept: usize,
    pub names_dropped: usize,
    /// Undirected chunk pairs related, and directed relations by kind (`EdgeKind::ALL` order).
    pub edges: usize,
    pub by_kind: [usize; KINDS],
    start: Vec<u32>,
    adj: Vec<Edge>,
}
impl Specific {
    /// Chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.start.len().saturating_sub(1)
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The specific relations of chunk `i`, heaviest first.
    #[must_use]
    pub fn edges_of(&self, i: usize) -> &[Edge] {
        let (a, b) = (self.start.get(i).copied().unwrap_or(0) as usize, self.start.get(i + 1).copied().unwrap_or(0) as usize);
        self.adj.get(a..b).unwrap_or(&[])
    }
    /// Distinct chunks related to chunk `i`.
    #[must_use]
    pub fn degree(&self, i: usize) -> usize {
        self.edges_of(i).iter().map(|e| e.to).collect::<BTreeSet<_>>().len()
    }
    /// Degree distribution: (mean, median, largest, chunks with no relation).
    #[must_use]
    pub fn degrees(&self) -> (f64, usize, usize, usize) {
        let mut d: Vec<usize> = (0..self.len()).map(|i| self.degree(i)).collect();
        d.sort_unstable();
        #[expect(clippy::cast_precision_loss, reason = "chunk counts are small")]
        let mean = d.iter().sum::<usize>() as f64 / d.len().max(1) as f64;
        (mean, d.get(d.len() / 2).copied().unwrap_or(0), d.last().copied().unwrap_or(0), d.iter().filter(|x| **x == 0).count())
    }
    /// The weight each edge kind carries in a relation walk, derived from this graph: one plus
    /// the kind's information among the relations, ln(all / of this kind) / ln(all), so a rarer
    /// kind of relation counts more; the untyped kind `Names` and absent kinds count one, and a
    /// graph with no relations gives one for every kind (the stated fallback).
    #[must_use]
    pub fn kind_weights(&self) -> [f32; KINDS] {
        let total: usize = self.by_kind.iter().sum();
        let mut w = [1.0_f32; KINDS];
        if total < 2 {
            return w;
        }
        #[expect(clippy::cast_precision_loss, reason = "edge counts are small")]
        let ln_all = (total as f32).ln();
        for k in EdgeKind::ALL {
            let n = self.by_kind.get(k.index()).copied().unwrap_or(0);
            if k != EdgeKind::Names
                && n > 0
                && let Some(x) = w.get_mut(k.index())
            {
                #[expect(clippy::cast_precision_loss, reason = "edge counts are small")]
                let info = (total as f32 / n as f32).ln() / ln_all;
                *x = 1.0 + info;
            }
        }
        w
    }
    /// Hops from `from` to every chunk within `reach` hops over the specific relations.
    #[must_use]
    pub fn hops(&self, from: usize, reach: usize) -> BTreeMap<usize, usize> {
        let mut seen = BTreeMap::new();
        if from >= self.len() {
            return seen;
        }
        seen.insert(from, 0);
        let mut ring = vec![from];
        for h in 1..=reach {
            let mut next = Vec::new();
            for &i in &ring {
                for e in self.edges_of(i) {
                    if let std::collections::btree_map::Entry::Vacant(v) = seen.entry(e.to as usize) {
                        v.insert(h);
                        next.push(e.to as usize);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            ring = next;
        }
        seen
    }
    /// Personalised rank over the specific relations (the same walk as `RefGraph::rank`:
    /// both directions, restart at the seeds with probability `1 - DAMPING`).
    #[must_use]
    pub fn rank(&self, seeds: &[(usize, f32)]) -> (Vec<f32>, usize) {
        let n = self.len();
        let total: f32 = seeds.iter().filter(|s| s.0 < n).map(|s| s.1.max(0.0)).sum();
        if n == 0 || total <= 0.0 {
            return (vec![0.0; n], 0);
        }
        let mut p = vec![0.0_f32; n];
        for &(i, w) in seeds {
            if let Some(x) = p.get_mut(i) {
                *x += w.max(0.0) / total;
            }
        }
        let deg: Vec<f32> = (0..n).map(|i| self.edges_of(i).iter().map(|e| e.weight).sum()).collect();
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a small positive count")]
        let max_iter = ((1e-6_f32).ln() / DAMPING.ln()).ceil() as usize;
        let mut r = p.clone();
        let mut iters = 0;
        for _ in 0..max_iter {
            iters += 1;
            let mut next: Vec<f32> = p.iter().map(|x| (1.0 - DAMPING) * x).collect();
            let mut dangling = 0.0;
            for i in 0..n {
                let ri = r.get(i).copied().unwrap_or(0.0);
                let d = deg.get(i).copied().unwrap_or(0.0);
                if ri == 0.0 {
                    continue;
                }
                if d <= 0.0 {
                    dangling += ri;
                    continue;
                }
                for e in self.edges_of(i) {
                    if let Some(x) = next.get_mut(e.to as usize) {
                        *x += DAMPING * ri * e.weight / d;
                    }
                }
            }
            for (x, pi) in next.iter_mut().zip(&p) {
                *x += DAMPING * dangling * pi;
            }
            let change: f32 = next.iter().zip(&r).map(|(a, b)| (a - b).abs()).sum();
            r = next;
            if change < 1e-6 {
                break;
            }
        }
        (r, iters)
    }
    /// Bytes held in RAM.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.adj.len() * std::mem::size_of::<Edge>() + self.start.len() * 4
    }
    /// One line: what the specific relations hold.
    #[must_use]
    pub fn describe(&self) -> String {
        let (mean, median, max, alone) = self.degrees();
        let kinds: Vec<String> =
            EdgeKind::ALL.iter().map(|k| format!("{} {}", k.name(), self.by_kind.get(k.index()).copied().unwrap_or(0))).collect();
        let limit = if self.limit == usize::MAX { "none".to_owned() } else { self.limit.to_string() };
        format!(
            "specific relations: limit {limit} chunks per name ({}), names kept {} dropped {}, {} chunk pairs, degree mean {mean:.2} median {median} largest {max}, {alone} chunks unrelated, directed relations by kind: {}, {} bytes in RAM",
            self.rule,
            self.names_kept,
            self.names_dropped,
            self.edges,
            kinds.join(", "),
            self.bytes()
        )
    }
}

/// Item name to (chunks defining it, files defining it).
type Defs<'a> = BTreeMap<&'a str, (BTreeSet<usize>, BTreeSet<&'a str>)>;

/// Definitions of a substrate: item name to the chunks and files defining it, and the kinds of
/// its definitions.
fn definitions(sub: &Substrate) -> (Defs<'_>, BTreeMap<&str, BTreeSet<&str>>) {
    let mut by_file: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, c) in sub.chunks.iter().enumerate() {
        if let Some(f) = sub.files.get(c.file) {
            by_file.entry(f.path.as_str()).or_default().push(i);
        }
    }
    let mut defs: Defs<'_> = BTreeMap::new();
    let mut kinds: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for s in &sub.symbols {
        if s.name.len() < MIN_NAME {
            continue;
        }
        let Some(list) = by_file.get(s.file.as_str()) else { continue };
        for &i in list {
            if sub.chunks.get(i).is_some_and(|c| c.start <= s.line && s.line <= c.end) {
                let e = defs.entry(s.name.as_str()).or_default();
                e.0.insert(i);
                e.1.insert(s.file.as_str());
                kinds.entry(s.name.as_str()).or_default().insert(s.kind.as_str());
            }
        }
    }
    (defs, kinds)
}

/// Whether a chunk is test code (a test folder, or a test attribute in its text; in other
/// languages their test file names and test markers, from the language table).
fn is_test(sub: &Substrate, i: usize) -> bool {
    sub.chunks.get(i).is_some_and(|c| {
        let Some(f) = sub.files.get(c.file) else { return false };
        let path = f.path.replace('\\', "/");
        if c.text.contains("#[test]") || c.text.contains("#[cfg(test)]") || path.contains("/tests/") {
            return true;
        }
        match nmlex::lang::by_id(f.decoder).filter(|l| l.id != "rust") {
            Some(l) => nmlex::lang::is_test_path(&path) || nmlex::lang::has_test_mark(l, &c.text),
            None => false,
        }
    })
}

/// Build the specific relations from identifier sets and definitions (module notes).
fn specific_from(
    sub: &Substrate,
    per_chunk: &[Arc<Vec<String>>],
    defs: &Defs<'_>,
    kinds: &BTreeMap<&str, BTreeSet<&str>>,
    limit: Limit,
) -> Specific {
    let n = per_chunk.len();
    // Chunks holding each followed name.
    let mut users: BTreeMap<&str, usize> = defs.keys().map(|k| (*k, 0)).collect();
    for ids in per_chunk {
        for name in ids.iter() {
            if let Some(u) = users.get_mut(name.as_str()) {
                *u += 1;
            }
        }
    }
    let (cap, rule) = match limit {
        Limit::Fixed(k) => (k, "stated count".to_owned()),
        Limit::Unlimited => (usize::MAX, "no limit".to_owned()),
        Limit::Derived => {
            let mut counts: Vec<usize> = users.values().copied().collect();
            counts.sort_unstable();
            match counts.get(counts.len() / 2) {
                Some(&m) if m > 0 => (m, format!("derived: median chunk count of the {} item names followed", counts.len())),
                _ => (RARE_FALLBACK, "fallback: the tree defines no names to derive a limit from".to_owned()),
            }
        }
    };
    let tests: Vec<bool> = (0..n).map(|i| is_test(sub, i)).collect();
    let empty = BTreeSet::new();
    // Directed (user, definer) -> (weight, heaviest single contribution, its kind).
    let mut acc: BTreeMap<(u32, u32), (f32, f32, EdgeKind)> = BTreeMap::new();
    for (i, ids) in per_chunk.iter().enumerate() {
        for name in ids.iter() {
            let Some((chunks, _)) = defs.get(name.as_str()) else { continue };
            let u = users.get(name.as_str()).copied().unwrap_or(0);
            if u == 0 || u > cap || u >= n {
                continue;
            }
            #[expect(clippy::cast_precision_loss, reason = "chunk counts are small")]
            let w = (n as f32 / u as f32).ln() / chunks.len().max(1) as f32;
            let kind = EdgeKind::of(kinds.get(name.as_str()).unwrap_or(&empty), tests.get(i).copied().unwrap_or(false));
            for &d in chunks {
                if d != i
                    && let (Ok(a), Ok(b)) = (u32::try_from(i), u32::try_from(d))
                {
                    let e = acc.entry((a, b)).or_insert((0.0, 0.0, kind));
                    e.0 += w;
                    if w > e.1 {
                        e.1 = w;
                        e.2 = kind;
                    }
                }
            }
        }
    }
    let kept = users.values().filter(|u| **u > 0 && **u <= cap).count();
    assemble(n, &acc, cap, rule, kept, users.len() - kept)
}

/// Directed (user, definer) relations to `Specific`: both directions per chunk, heaviest first.
fn assemble(n: usize, acc: &BTreeMap<(u32, u32), (f32, f32, EdgeKind)>, limit: usize, rule: String, kept: usize, dropped: usize) -> Specific {
    let mut rows: Vec<Vec<Edge>> = vec![Vec::new(); n];
    let mut by_kind = [0_usize; KINDS];
    let mut pairs = BTreeSet::new();
    for (&(a, b), &(weight, _, kind)) in acc {
        if let Some(r) = rows.get_mut(a as usize) {
            r.push(Edge { to: b, kind, dir: Relation::Uses, weight });
        }
        if let Some(r) = rows.get_mut(b as usize) {
            r.push(Edge { to: a, kind, dir: Relation::UsedBy, weight });
        }
        if let Some(x) = by_kind.get_mut(kind.index()) {
            *x += 1;
        }
        pairs.insert((a.min(b), a.max(b)));
    }
    let mut start = Vec::with_capacity(n + 1);
    let mut adj = Vec::new();
    for mut r in rows {
        r.sort_by(|x, y| y.weight.total_cmp(&x.weight).then(x.to.cmp(&y.to)).then(x.dir.cmp(&y.dir)));
        start.push(u32::try_from(adj.len()).unwrap_or(u32::MAX));
        adj.extend(r);
    }
    start.push(u32::try_from(adj.len()).unwrap_or(u32::MAX));
    Specific { limit, rule, names_kept: kept, names_dropped: dropped, edges: pairs.len(), by_kind, start, adj }
}

/// True when the substrate's code is cut into pieces (their identifiers are recorded).
fn has_pieces(sub: &Substrate) -> bool {
    sub.chunks.iter().any(|c| !c.piece.name.is_empty())
}

/// The module a file stands for: its name without the extension, or its folder's name for a
/// module root (`mod.rs`, `lib.rs`, `main.rs`, `index.*`, `__init__.py`).
fn module_of(path: &str) -> &str {
    let mut parts = path.rsplit('/');
    let file = parts.next().unwrap_or(path);
    let stem = file.split('.').next().unwrap_or(file);
    if matches!(stem, "mod" | "lib" | "main" | "index" | "__init__") {
        let dir = parts.next().unwrap_or(stem);
        // A crate's `src/lib.rs` stands for the crate folder.
        if dir == "src" { parts.next().unwrap_or(stem) } else { dir }
    } else {
        stem
    }
}

/// Names a Rust file brings in by `use` with the module each comes from (`use a::b::{c, d}`:
/// `c` and `d` from `b`; `use a::b::c`: `c` from `b`).
fn rust_imports(text: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut stmt = String::new();
    let mut inside = false;
    for l in text.lines() {
        let t = l.trim_start();
        if !inside && (t.starts_with("use ") || t.starts_with("pub use ") || t.starts_with("pub(crate) use ")) {
            inside = true;
            stmt.clear();
        }
        if inside {
            stmt.push_str(t);
            stmt.push(' ');
            if t.contains(';') {
                inside = false;
                let toks: Vec<String> = nmlex::tokens(&stmt, true).into_iter().map(|x| x.0).collect();
                let mut prefix: Vec<String> = Vec::new();
                let mut last: Option<String> = None;
                for (i, x) in toks.iter().enumerate() {
                    match x.as_str() {
                        "{" => prefix.push(last.clone().unwrap_or_default()),
                        "}" => {
                            prefix.pop();
                        }
                        _ if x.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_') => {
                            let next = toks.get(i + 1).map_or("", String::as_str);
                            let after_sep = i >= 2 && toks.get(i - 1).is_some_and(|p| p == ":") && toks.get(i - 2).is_some_and(|p| p == ":");
                            if matches!(next, "," | "}" | ";" | "as") && x != "self" && x != "use" {
                                let module = if after_sep { last.clone() } else { prefix.last().cloned() };
                                if let Some(m) = module.filter(|m| !m.is_empty()) {
                                    out.insert(x.clone(), m);
                                }
                            }
                            if !matches!(x.as_str(), "use" | "pub" | "crate" | "as") {
                                last = Some(x.clone());
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    out
}

/// The specific relations of a substrate cut into pieces (module notes): typed, weighted edges
/// from the identifiers each piece uses to the pieces that define or assign them. A name is rare
/// by how many pieces define or assign it, after its use is resolved: a qualified path (`a::b`,
/// `Type::item`) to the definitions in module `a` or of type `Type`; an unqualified name to the
/// definitions in the user's own file, else to the module it is imported from. A piece that
/// assigns the name itself uses its own (no edge). A definition is preferred to an assignment.
/// Weight: the square root of the use count (Aider's repository map form) times the name's
/// specificity, the natural logarithm of pieces over the pieces using it, spread over the
/// definitions. A resolved edge is kept however many pieces use the name: a function defined once
/// and called from many places keeps every edge. Kinds: from the definition's kind (call, type use, definition use), `Implements` for an
/// implemented trait, `DefinesUsed` for a name only assigned, `Test` from test code.
fn specific_pieces(sub: &Substrate, limit: Limit) -> Specific {
    let n = sub.chunks.len();
    // The kind of each definition by file and name.
    let mut sym_kind: BTreeMap<(&str, &str), &str> = BTreeMap::new();
    let mut sym_parent: BTreeMap<(&str, &str), &str> = BTreeMap::new();
    for s in &sub.symbols {
        sym_kind.entry((s.file.as_str(), s.name.as_str())).or_insert(s.kind.as_str());
        if !s.parent.is_empty() {
            sym_parent.entry((s.file.as_str(), s.name.as_str())).or_insert(s.parent.as_str());
        }
    }
    let path_of = |i: usize| sub.chunks.get(i).and_then(|c| sub.files.get(c.file)).map_or("", |f| f.path.as_str());
    // Name to the pieces defining (true) or assigning (false) it.
    let mut definers: BTreeMap<&str, Vec<(u32, bool)>> = BTreeMap::new();
    for (i, c) in sub.chunks.iter().enumerate() {
        let Ok(i) = u32::try_from(i) else { continue };
        for d in &c.piece.defines {
            definers.entry(d.as_str()).or_default().push((i, true));
        }
        for a in &c.piece.assigns {
            definers.entry(a.as_str()).or_default().push((i, false));
        }
    }
    // Pieces using each name.
    let mut users: BTreeMap<&str, usize> = BTreeMap::new();
    for c in &sub.chunks {
        for (x, _) in &c.piece.uses {
            *users.entry(x.as_str()).or_default() += 1;
        }
    }
    // Imports by file: Rust `use` lines, the tagger's imports elsewhere.
    let mut imports: BTreeMap<usize, BTreeMap<String, String>> = BTreeMap::new();
    for (fi, f) in sub.files.iter().enumerate() {
        if f.decoder == "rust" {
            imports.insert(fi, rust_imports(&f.text));
        } else if let Some(list) = sub.imports.get(&f.path) {
            let mut m = BTreeMap::new();
            for imp in list {
                let spec = imp.spec.rsplit(['/', '.', ':']).find(|x| !x.is_empty()).unwrap_or(&imp.spec).to_owned();
                for name in &imp.names {
                    m.insert(name.clone(), spec.clone());
                }
            }
            imports.insert(fi, m);
        }
    }
    let (cap, rule) = match limit {
        Limit::Fixed(k) => (k, "stated count".to_owned()),
        Limit::Unlimited => (usize::MAX, "no limit".to_owned()),
        Limit::Derived => {
            // The count of defining pieces at the ninth decile of the names defined, at least 2.
            let mut counts: Vec<usize> = definers.values().map(|v| v.iter().map(|x| x.0).collect::<BTreeSet<_>>().len()).collect();
            counts.sort_unstable();
            match counts.get(counts.len() * 9 / 10) {
                Some(&m) => (m.max(2), format!("derived: ninth decile of defining pieces over the {} names defined or assigned", counts.len())),
                None => (RARE_FALLBACK, "fallback: the tree defines no names to derive a limit from".to_owned()),
            }
        }
    };
    let tests: Vec<bool> = (0..n).map(|i| is_test(sub, i)).collect();
    let mut acc: BTreeMap<(u32, u32), (f32, f32, EdgeKind)> = BTreeMap::new();
    // A pair keeps the kind of its heaviest typed relation; an untyped same-name link adds its
    // weight but names the pair only when nothing typed joins it.
    let mut add = |a: u32, b: u32, w: f32, kind: EdgeKind| {
        let e = acc.entry((a, b)).or_insert((0.0, 0.0, kind));
        e.0 += w;
        let typed = kind != EdgeKind::Names;
        if (typed && (e.2 == EdgeKind::Names || w > e.1)) || (!typed && e.2 == EdgeKind::Names && w > e.1) {
            e.1 = w;
            e.2 = kind;
        }
    };
    let class = |k: &str| match k {
        "fn" | "function" | "method" | "def" | "func" => EdgeKind::Call,
        "struct" | "enum" | "type" | "trait" | "class" | "interface" | "union" | "typedef" => EdgeKind::TypeUse,
        _ => EdgeKind::DefUse,
    };
    // The users cap of a match by name alone: the median count of pieces using a defined name.
    let users_cap = {
        let mut v: Vec<usize> = definers.keys().filter_map(|k| users.get(k).copied()).collect();
        v.sort_unstable();
        v.get(v.len() / 2).copied().unwrap_or(RARE_FALLBACK).max(2)
    };
    let mut used_names: BTreeSet<&str> = BTreeSet::new();
    let mut kept_names: BTreeSet<&str> = BTreeSet::new();
    for (i, c) in sub.chunks.iter().enumerate() {
        let Ok(a) = u32::try_from(i) else { continue };
        let file = path_of(i);
        let fi = c.file;
        let test = tests.get(i).copied().unwrap_or(false);
        for (x, count) in &c.piece.uses {
            if c.piece.assigns.contains(x) {
                continue;
            }
            let Some(all) = definers.get(x.as_str()) else { continue };
            used_names.insert(x.as_str());
            let mut cands: Vec<(u32, bool)> = all.iter().copied().filter(|d| d.0 != a).collect();
            if cands.is_empty() {
                continue;
            }
            let narrow = |cands: &mut Vec<(u32, bool)>, keep: &dyn Fn(u32) -> bool| {
                let v: Vec<(u32, bool)> = cands.iter().copied().filter(|d| keep(d.0)).collect();
                if !v.is_empty() {
                    *cands = v;
                    true
                } else {
                    false
                }
            };
            let quals: Vec<&str> = c.piece.paths.iter().filter(|(_, b)| b == x).map(|(q, _)| q.as_str()).collect();
            let resolved = if !quals.is_empty() {
                let local = quals.iter().any(|q| matches!(*q, "self" | "Self" | "crate" | "super"));
                narrow(&mut cands, &|d| {
                    let p = path_of(d as usize);
                    let parent = sub.chunks.get(d as usize).map_or("", |dc| dc.piece.parent.as_str());
                    quals.iter().any(|q| module_of(p) == *q || parent == *q || sym_parent.get(&(p, x.as_str())).is_some_and(|sp| sp == q))
                        || (local && p == file)
                })
            } else {
                false
            };
            // A member of the user's own type (`self.index` inside `impl Substrate`) resolves to
            // that type's definition first, wherever the type's `impl` sits.
            let own_type = !c.piece.parent.is_empty()
                && c.piece.members.contains(x)
                && narrow(&mut cands, &|d| sub.chunks.get(d as usize).is_some_and(|dc| dc.piece.parent == c.piece.parent));
            let resolved = resolved
                || own_type
                || narrow(&mut cands, &|d| path_of(d as usize) == file)
                || imports.get(&fi).and_then(|m| m.get(x.as_str())).is_some_and(|m| narrow(&mut cands, &|d| module_of(path_of(d as usize)) == m.as_str()));
            if cands.iter().any(|d| d.1) {
                cands.retain(|d| d.1);
            }
            cands.sort_unstable();
            cands.dedup_by_key(|d| d.0);
            let k = cands.len();
            if k == 0 || k > cap {
                continue;
            }
            // A use resolved to a definition by its path, its type, its file or its import is
            // specific by construction, as is a free use of a name only definitions give, and a
            // member use of a name one piece alone defines (resolved to that one piece, kept
            // whatever the number of callers). A use matched by name alone among several definers
            // (a field or method after `.` not resolved, or a name only assigned) keeps the rarity
            // cap on its users.
            let by_name = cands.iter().all(|d| !d.1) || (!resolved && c.piece.members.contains(x) && k > 1);
            if by_name && users.get(x.as_str()).copied().unwrap_or(0) > users_cap {
                continue;
            }
            kept_names.insert(x.as_str());
            #[expect(clippy::cast_precision_loss, reason = "counts are small")]
            let w = (*count as f32).sqrt() * (n as f32 / users.get(x.as_str()).copied().unwrap_or(1).max(1) as f32).ln().max(0.1) / k as f32;
            for (d, strong) in cands {
                let kind = if test {
                    EdgeKind::Test
                } else if strong {
                    class(sym_kind.get(&(path_of(d as usize), x.as_str())).copied().unwrap_or(""))
                } else {
                    EdgeKind::DefinesUsed
                };
                add(a, d, w, kind);
            }
        }
        for t in &c.piece.implements {
            let Some(all) = definers.get(t.as_str()) else { continue };
            let defs: Vec<u32> = all.iter().filter(|d| d.1 && d.0 != a).map(|d| d.0).collect();
            if defs.is_empty() || defs.len() > cap {
                continue;
            }
            #[expect(clippy::cast_precision_loss, reason = "counts are small")]
            let w = 1.0 / defs.len() as f32;
            for d in defs {
                add(a, d, w, if test { EdgeKind::Test } else { EdgeKind::Implements });
            }
        }
    }
    // Untyped same-name links, in code, documents and tables alike: two pieces naming the same
    // rare name (an identifier, a dashed key, a version) are joined. A name is rare when at most
    // `cap` files hold it, the limit the defining pieces gave (a file, like a definition, is one
    // place a name belongs to; pieces of one file repeat it).
    let mut holders: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    let mut files_of: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for (i, c) in sub.chunks.iter().enumerate() {
        let Ok(a) = u32::try_from(i) else { continue };
        for t in names_in(&c.text) {
            files_of.entry(t.clone()).or_default().insert(c.file);
            holders.entry(t).or_default().push(a);
        }
    }
    let mut linked = 0_usize;
    for (name, hs) in &holders {
        if hs.len() < 2 || files_of.get(name).map_or(0, BTreeSet::len) > cap {
            continue;
        }
        linked += 1;
        #[expect(clippy::cast_precision_loss, reason = "counts are small")]
        let w = 1.0 / hs.len() as f32;
        for (x, a) in hs.iter().enumerate() {
            for b in hs.iter().skip(x + 1) {
                add(*a, *b, w, EdgeKind::Names);
            }
        }
    }
    let kept = kept_names.len();
    assemble(
        n,
        &acc,
        cap,
        format!("{rule}; a match by name alone at most {users_cap} users; {linked} rare names linked untyped"),
        kept,
        used_names.len() - kept,
    )
}

/// The names a piece holds for untyped links, each once: identifiers written as names (an
/// underscore, a dash between letters, an inner capital, or all capitals) of `MIN_NAME` characters
/// or more, and versions (digits joined by dots). Plain words are not names.
fn names_in(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for w in text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '.')) {
        let w = w.trim_matches(|c: char| c == '.' || c == '-');
        if w.len() < MIN_NAME {
            continue;
        }
        let letters = w.chars().any(char::is_alphabetic);
        let version = !letters && w.contains('.') && w.split('.').all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()));
        if version {
            // A release (`1.99.1`) names its line (`1.99`) too.
            let parts: Vec<&str> = w.split('.').collect();
            if parts.len() > 2 {
                out.insert(parts.get(..2).map_or_else(String::new, |p| p.join(".")));
            }
            out.insert(w.to_owned());
            continue;
        }
        for part in w.split('.').filter(|p| p.len() >= MIN_NAME) {
            let cs: Vec<char> = part.chars().collect();
            let camel = cs.windows(2).any(|x| x.first().is_some_and(|a| a.is_lowercase()) && x.get(1).is_some_and(|b| b.is_uppercase()));
            let caps = cs.iter().filter(|c| c.is_alphabetic()).count() >= MIN_NAME && cs.iter().all(|c| !c.is_lowercase());
            let dashed = part.contains('-') && cs.iter().any(|c| c.is_alphabetic());
            if part.chars().any(char::is_alphabetic) && (part.contains('_') || dashed || camel || caps) {
                out.insert(part.to_lowercase());
            }
        }
    }
    out
}

/// How one chunk relates to another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Relation {
    /// This chunk uses an item the other chunk defines.
    Uses,
    /// The other chunk uses an item this chunk defines.
    UsedBy,
}
impl Relation {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Uses => "uses",
            Self::UsedBy => "used-by",
        }
    }
}

/// The graph: one node per substrate chunk, edges from a using chunk to a defining chunk.
#[derive(Clone, Debug, Default)]
pub struct RefGraph {
    /// Substrate version it was built on.
    pub version: u64,
    /// Content key of every chunk, by chunk index.
    pub keys: Vec<String>,
    key_index: BTreeMap<String, u32>,
    out_start: Vec<u32>,
    out: Vec<(u32, f32)>,
    in_start: Vec<u32>,
    inn: Vec<(u32, f32)>,
    /// Identifier sets by chunk content key (reused on rebuild).
    idents: BTreeMap<String, Arc<Vec<String>>>,
    /// Item names followed, edges, and the build time.
    pub names: usize,
    pub edges: usize,
    pub build_ms: f64,
    /// Chunks whose identifiers were read again on the last build (the rest were reused).
    pub reread: usize,
    /// The specific relations (second form, refinement step 4a), limit derived from the tree;
    /// built with the graph and rebuilt with it on sync.
    pub specific: Arc<Specific>,
}

/// Identifiers of a text: maximal runs of letters, digits and underscores starting with a letter
/// or underscore, each once.
#[must_use]
pub fn identifiers(text: &str) -> Vec<String> {
    let mut out = BTreeSet::new();
    let mut cur = String::new();
    for ch in text.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else if !cur.is_empty() {
            if cur.chars().next().is_some_and(|c| !c.is_ascii_digit()) && cur.len() >= MIN_NAME {
                out.insert(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    out.into_iter().collect()
}

impl RefGraph {
    /// Build over a substrate; `previous` lends the identifier sets of unchanged chunks.
    #[must_use]
    pub fn build(sub: &Substrate, previous: Option<&Self>) -> Self {
        let t = Instant::now();
        let n = sub.chunks.len();
        // Definitions: item name to the chunks defining it, and the files.
        let (defs, kinds) = definitions(sub);
        // Identifier sets, reused by content key.
        let mut idents: BTreeMap<String, Arc<Vec<String>>> = BTreeMap::new();
        let mut reread = 0;
        let mut per_chunk: Vec<Arc<Vec<String>>> = Vec::with_capacity(n);
        for c in &sub.chunks {
            let v = match previous.and_then(|p| p.idents.get(&c.key)).or_else(|| idents.get(&c.key)) {
                Some(v) => Arc::clone(v),
                None => {
                    reread += 1;
                    Arc::new(identifiers(&c.text))
                }
            };
            idents.insert(c.key.clone(), Arc::clone(&v));
            per_chunk.push(v);
        }
        // Edges: chunk -> defining chunk, weight spread over the definitions.
        let mut acc: BTreeMap<(u32, u32), f32> = BTreeMap::new();
        for (i, ids) in per_chunk.iter().enumerate() {
            for name in ids.iter() {
                let Some((chunks, files)) = defs.get(name.as_str()) else { continue };
                #[expect(clippy::cast_precision_loss, reason = "definition counts are small")]
                let w = if files.len() > COMMON_FILES { COMMON_WEIGHT } else { 1.0 } / chunks.len().max(1) as f32;
                for &d in chunks {
                    if d != i
                        && let (Ok(a), Ok(b)) = (u32::try_from(i), u32::try_from(d))
                    {
                        *acc.entry((a, b)).or_default() += w;
                    }
                }
            }
        }
        let (out_start, out) = csr(n, acc.iter().map(|(&(a, b), &w)| (a, b, w)));
        let (in_start, inn) = csr(n, acc.iter().map(|(&(a, b), &w)| (b, a, w)));
        let mut key_index = BTreeMap::new();
        for (i, c) in sub.chunks.iter().enumerate() {
            if let Ok(k) = u32::try_from(i) {
                key_index.entry(c.key.clone()).or_insert(k);
            }
        }
        let specific = Arc::new(if has_pieces(sub) { specific_pieces(sub, Limit::Derived) } else { specific_from(sub, &per_chunk, &defs, &kinds, Limit::Derived) });
        Self {
            version: sub.version,
            keys: sub.chunks.iter().map(|c| c.key.clone()).collect(),
            key_index,
            out_start,
            out,
            in_start,
            inn,
            idents,
            names: defs.len(),
            edges: acc.len(),
            build_ms: t.elapsed().as_secs_f64() * 1000.0,
            reread,
            specific,
        }
    }
    /// The specific relations under another limit (for comparison; the graph's own, derived
    /// from the tree, is `self.specific`). `sub` is the substrate this graph was built on.
    #[must_use]
    pub fn specific_with(&self, sub: &Substrate, limit: Limit) -> Specific {
        let (defs, kinds) = definitions(sub);
        let none = Arc::new(Vec::new());
        let per_chunk: Vec<Arc<Vec<String>>> = sub.chunks.iter().map(|c| Arc::clone(self.idents.get(&c.key).unwrap_or(&none))).collect();
        if has_pieces(sub) { specific_pieces(sub, limit) } else { specific_from(sub, &per_chunk, &defs, &kinds, limit) }
    }
    /// Distinct chunks related to chunk `i` in the first form (any shared item name).
    #[must_use]
    pub fn degree(&self, i: usize) -> usize {
        Self::row(&self.out_start, &self.out, i)
            .iter()
            .chain(Self::row(&self.in_start, &self.inn, i))
            .map(|e| e.0)
            .collect::<BTreeSet<_>>()
            .len()
    }
    /// Undirected chunk pairs related in the first form.
    #[must_use]
    pub fn pairs(&self) -> usize {
        (0..self.keys.len()).map(|i| self.degree(i)).sum::<usize>() / 2
    }
    /// Chunk index of a content key.
    #[must_use]
    pub fn index_of(&self, key: &str) -> Option<usize> {
        self.key_index.get(key).map(|i| *i as usize)
    }
    fn row<'a>(start: &[u32], v: &'a [(u32, f32)], i: usize) -> &'a [(u32, f32)] {
        let (a, b) = (start.get(i).copied().unwrap_or(0) as usize, start.get(i + 1).copied().unwrap_or(0) as usize);
        v.get(a..b).unwrap_or(&[])
    }
    /// Related chunks of chunk `i`: (chunk, relation, weight), heaviest first.
    #[must_use]
    pub fn neighbours(&self, i: usize) -> Vec<(usize, Relation, f32)> {
        let mut v: Vec<(usize, Relation, f32)> = Self::row(&self.out_start, &self.out, i)
            .iter()
            .map(|&(j, w)| (j as usize, Relation::Uses, w))
            .chain(Self::row(&self.in_start, &self.inn, i).iter().map(|&(j, w)| (j as usize, Relation::UsedBy, w)))
            .collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        v
    }
    /// Personalised rank of every chunk from `seeds` (chunk, weight): a walk along references in
    /// both directions (a found span relates to what it uses and to what uses it), restarting at
    /// the seeds with probability `1 - DAMPING`. Returns the rank per chunk (sums to 1) and the
    /// iterations run.
    #[must_use]
    pub fn rank(&self, seeds: &[(usize, f32)]) -> (Vec<f32>, usize) {
        let n = self.keys.len();
        let total: f32 = seeds.iter().filter(|s| s.0 < n).map(|s| s.1.max(0.0)).sum();
        if n == 0 || total <= 0.0 {
            return (vec![0.0; n], 0);
        }
        let mut p = vec![0.0_f32; n];
        for &(i, w) in seeds {
            if let Some(x) = p.get_mut(i) {
                *x += w.max(0.0) / total;
            }
        }
        // Out weight per node over both directions.
        let deg: Vec<f32> = (0..n)
            .map(|i| Self::row(&self.out_start, &self.out, i).iter().chain(Self::row(&self.in_start, &self.inn, i)).map(|e| e.1).sum())
            .collect();
        // Iteration bound: DAMPING^k below 1e-6.
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a small positive count")]
        let max_iter = ((1e-6_f32).ln() / DAMPING.ln()).ceil() as usize;
        let mut r = p.clone();
        let mut iters = 0;
        for _ in 0..max_iter {
            iters += 1;
            let mut next: Vec<f32> = p.iter().map(|x| (1.0 - DAMPING) * x).collect();
            let mut dangling = 0.0;
            for i in 0..n {
                let ri = r.get(i).copied().unwrap_or(0.0);
                let d = deg.get(i).copied().unwrap_or(0.0);
                if ri == 0.0 {
                    continue;
                }
                if d <= 0.0 {
                    dangling += ri;
                    continue;
                }
                for &(j, w) in Self::row(&self.out_start, &self.out, i).iter().chain(Self::row(&self.in_start, &self.inn, i)) {
                    if let Some(x) = next.get_mut(j as usize) {
                        *x += DAMPING * ri * w / d;
                    }
                }
            }
            for (x, pi) in next.iter_mut().zip(&p) {
                *x += DAMPING * dangling * pi;
            }
            let change: f32 = next.iter().zip(&r).map(|(a, b)| (a - b).abs()).sum();
            r = next;
            if change < 1e-6 {
                break;
            }
        }
        (r, iters)
    }
    /// Bytes held in RAM (edges, offsets, keys and identifier sets).
    #[must_use]
    pub fn bytes(&self) -> usize {
        let edges =
            (self.out.len() + self.inn.len()) * std::mem::size_of::<(u32, f32)>() + (self.out_start.len() + self.in_start.len()) * 4;
        let keys: usize = self.keys.iter().map(String::len).sum::<usize>() * 2;
        let ids: usize = self.idents.values().map(|v| v.iter().map(String::len).sum::<usize>()).sum();
        edges + keys + ids
    }
}

/// Compressed rows from (row, column, weight) triples.
fn csr(n: usize, triples: impl Iterator<Item = (u32, u32, f32)>) -> (Vec<u32>, Vec<(u32, f32)>) {
    let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    for (a, b, w) in triples {
        if let Some(r) = rows.get_mut(a as usize) {
            r.push((b, w));
        }
    }
    let mut start = Vec::with_capacity(n + 1);
    let mut out = Vec::new();
    for r in rows {
        start.push(u32::try_from(out.len()).unwrap_or(u32::MAX));
        out.extend(r);
    }
    start.push(u32::try_from(out.len()).unwrap_or(u32::MAX));
    (start, out)
}

impl Substrate {
    /// The reference graph of this substrate version, built on first use and kept in RAM (shared
    /// by every version of this substrate; a later version rebuilds it, reusing identifier sets).
    #[must_use]
    pub fn refs(&self) -> Arc<RefGraph> {
        if let Some(g) =
            self.ref_graph.read().ok().and_then(|g| g.clone()).filter(|g| g.version == self.version && g.keys.len() == self.chunks.len())
        {
            return g;
        }
        let prev = self.ref_graph.read().ok().and_then(|g| g.clone());
        let g = Arc::new(RefGraph::build(self, prev.as_deref()));
        if let Ok(mut slot) = self.ref_graph.write() {
            *slot = Some(Arc::clone(&g));
        }
        g
    }
    /// On sync: rebuild the graph for the new version when one was in use (nothing is built
    /// for a substrate that never asked for it, so the paths without it are unchanged).
    pub fn refresh_refs(&self) {
        if self.ref_graph.read().is_ok_and(|g| g.as_ref().is_some_and(|g| g.version != self.version)) {
            let _ = self.refs();
        }
    }
}

/// Self-check of `refgraph`: a chunk using a name refers to its definition and not to itself;
/// a name defined in more than five files counts one tenth; the rank from a user seed reaches
/// the definition before an unrelated chunk; a rebuild after one file changed rereads only that
/// file's chunks and keeps the other edges.
pub(crate) fn check() -> Result<(), String> {
    let embed: crate::Embed = Arc::new(|_: &str| vec![0.0]);
    let mut files: Vec<(String, String)> = vec![
        ("a/def.rs".into(), "pub fn derive_limit() -> usize { 4 }\n".into()),
        ("a/user.rs".into(), "pub fn use_it() -> usize { derive_limit() + 1 }\n".into()),
        ("a/far.rs".into(), "pub fn unrelated() -> u8 { 7 }\n".into()),
    ];
    for k in 0..6 {
        files.push((format!("c/common{k}.rs"), "pub fn new() -> u8 { 0 }\n".into()));
    }
    files.push(("a/newer.rs".into(), "pub fn make() -> u8 { new() }\n".into()));
    let borrowed: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let sub = crate::from_files(&borrowed, Arc::clone(&embed));
    let g = sub.refs();
    let idx = |p: &str| sub.chunks.iter().position(|c| sub.files.get(c.file).is_some_and(|f| f.path == p));
    let (Some(d), Some(u), Some(f), Some(m)) = (idx("a/def.rs"), idx("a/user.rs"), idx("a/far.rs"), idx("a/newer.rs")) else {
        return Err("nmsubs.refgraph: test chunks missing".into());
    };
    let nu = g.neighbours(u);
    if !nu.iter().any(|x| x.0 == d && x.1 == Relation::Uses)
        || nu.iter().any(|x| x.0 == u)
        || !g.neighbours(d).iter().any(|x| x.0 == u && x.1 == Relation::UsedBy)
    {
        return Err(format!("nmsubs.refgraph: user neighbours {nu:?}"));
    }
    let common: f32 = g.neighbours(m).iter().filter(|x| x.1 == Relation::Uses).map(|x| x.2).sum();
    if (common - COMMON_WEIGHT).abs() > 1e-4 {
        return Err(format!("nmsubs.refgraph: a name defined in six files weighs {common}, want {COMMON_WEIGHT}"));
    }
    let (r, iters) = g.rank(&[(u, 1.0)]);
    let (rd, rf) = (r.get(d).copied().unwrap_or(0.0), r.get(f).copied().unwrap_or(0.0));
    if rd <= rf || (r.iter().sum::<f32>() - 1.0).abs() > 1e-3 || iters == 0 {
        return Err(format!("nmsubs.refgraph: rank definition {rd} unrelated {rf} after {iters} iterations"));
    }
    // Sync: one file changes; only its chunks are read again; the version moves; edges kept.
    let mut next = sub.clone();
    let mut nf = next.files.clone();
    if let Some(x) = nf.iter_mut().find(|x| x.path == "a/far.rs") {
        x.text = "pub fn unrelated() -> usize { derive_limit() }\n".into();
        x.hash = nmhash::sha256_bytes(x.text.as_bytes());
    }
    let _ = next.apply(nf);
    next.refresh_refs();
    let g2 = next.refs();
    let f2 = next.chunks.iter().position(|c| next.files.get(c.file).is_some_and(|x| x.path == "a/far.rs")).unwrap_or(usize::MAX);
    if g2.version != next.version || g2.reread != 1 || !g2.neighbours(f2).iter().any(|x| x.1 == Relation::Uses) || g2.edges <= g.edges {
        return Err(format!(
            "nmsubs.refgraph: after sync version {} reread {} edges {} (before {})",
            g2.version, g2.reread, g2.edges, g.edges
        ));
    }
    // Specific relations (step 4a): a name held by more chunks than the limit relates nothing;
    // a rarer name weighs more; kinds are told apart; the first form is unchanged beside it.
    let s1 = g.specific_with(&sub, Limit::Fixed(2));
    let e_ud: Vec<&Edge> = s1.edges_of(u).iter().filter(|e| e.to as usize == d).collect();
    if e_ud.len() != 1
        || e_ud.first().is_none_or(|e| e.kind != EdgeKind::Call || e.dir != Relation::Uses)
        || s1.edges_of(m).iter().any(|e| e.dir == Relation::Uses)
        || !s1.edges_of(d).iter().any(|e| e.to as usize == u && e.dir == Relation::UsedBy)
    {
        return Err(format!(
            "nmsubs.refgraph: specific relations at limit 2: user {:?}, common-name user {:?}",
            s1.edges_of(u),
            s1.edges_of(m)
        ));
    }
    let all = g.specific_with(&sub, Limit::Unlimited);
    let w_rare = all.edges_of(u).iter().filter(|e| e.to as usize == d).map(|e| e.weight).sum::<f32>();
    let w_common = all.edges_of(m).iter().filter(|e| e.dir == Relation::Uses).map(|e| e.weight).fold(0.0_f32, f32::max);
    if w_rare <= w_common || all.edges <= s1.edges || s1.hops(u, 2).get(&d) != Some(&1) || s1.hops(u, 2).contains_key(&f) {
        return Err(format!(
            "nmsubs.refgraph: specificity weights rare {w_rare} common {w_common}; pairs {} unlimited against {} limited",
            all.edges, s1.edges
        ));
    }
    if g.specific.limit == 0
        || g.specific.rule.is_empty()
        || g.specific.len() != sub.chunks.len()
        || g.specific.kind_weights().iter().any(|w| !(1.0..=2.0).contains(w))
    {
        return Err(format!("nmsubs.refgraph: derived limit {} ({})", g.specific.limit, g.specific.rule));
    }
    let typed = crate::from_files(
        &[
            ("t/shape.rs", "pub struct RareShape { a: u8 }\n"),
            ("t/limit.rs", "pub const RARE_LIMIT: usize = 3;\n"),
            ("t/user.rs", "fn fit(x: RareShape) -> usize { 0 }\n"),
            ("t/reader.rs", "fn read_it() -> usize { RARE_LIMIT }\n"),
            ("t/tests/probe.rs", "#[test]\nfn probes() { let _ = RARE_LIMIT; }\n"),
            ("t/pad1.rs", "fn pad_one() {}\n"),
            ("t/pad2.rs", "fn pad_two() {}\n"),
        ],
        Arc::clone(&embed),
    );
    let ts = typed.refs().specific_with(&typed, Limit::Fixed(RARE_FALLBACK));
    if ts.by_kind.get(EdgeKind::Test.index()).copied().unwrap_or(0) == 0
        || ts.by_kind.get(EdgeKind::TypeUse.index()).copied().unwrap_or(0) == 0
        || ts.by_kind.get(EdgeKind::DefUse.index()).copied().unwrap_or(0) == 0
    {
        return Err(format!("nmsubs.refgraph: edge kinds {:?}", ts.by_kind));
    }
    let (sr, _) = s1.rank(&[(u, 1.0)]);
    if sr.get(d).copied().unwrap_or(0.0) <= sr.get(f).copied().unwrap_or(0.0) {
        return Err("nmsubs.refgraph: specific rank does not reach the definition first".into());
    }
    Ok(())
}
