//! The material's items: the unit of belief (lead's design 1.1, belief per item; approved by Luis
//! Jake Gabriel III 2026-11-17). The drones keep their intake cut (the pieces); the items are a
//! second view of the same lines, built once per material version and held in RAM.
//!
//! - Code with definitions (Rust items, the tagged languages' symbols, TOML tables): every line
//!   belongs to the innermost definition holding it, a definition starting at the comment and
//!   attribute lines attached directly above it. Lines before the first definition are the file's
//!   head (module comment, imports), its own item. Other lines no definition holds (blank lines,
//!   an `impl` header, a closing brace) belong to the next definition, after the last one to the
//!   last.
//! - Documents with headings: each heading's own lines, up to the next heading of any level; the
//!   lines before the first heading are the file's head.
//! - Anything else (code the parser cannot read, documents without headings, data): its pieces.
//!
//! Every line of a file is in exactly one item (`partition`); a piece links to every item its
//! lines meet and an item to every piece its lines meet.
use crate::Substrate;
use crate::repo::Symbol;

/// What an item is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A definition with its attached comment and attribute lines.
    Definition,
    /// A file's head: the lines before its first definition or heading.
    Head,
    /// A document's heading and its lines up to the next heading.
    Section,
    /// A piece taken as it is (the file has no definitions or headings).
    Piece,
}

/// One item: its file, its own lines (runs of lines, in order; an outer definition has holes
/// where its inner definitions are), the comment lines attached above its definition, and the
/// definition's first line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    pub file: usize,
    pub kind: Kind,
    /// Runs of its own lines, (first, last), 1-based, in order.
    pub runs: Vec<(usize, usize)>,
    /// The comment lines attached directly above the definition (first, last); `None` when there
    /// are none.
    pub described: Option<(usize, usize)>,
    /// The definition's first line (the heading line of a section); 0 for a head or a piece.
    pub def: usize,
    /// The definition's kind and name (`fn parse_items`), the heading, or empty.
    pub label: String,
}
impl Item {
    /// Its first own line.
    #[must_use]
    pub fn start(&self) -> usize {
        self.runs.first().map_or(0, |r| r.0)
    }
    /// Its last own line.
    #[must_use]
    pub fn end(&self) -> usize {
        self.runs.last().map_or(0, |r| r.1)
    }
    /// Whether line `n` is one of its own.
    #[must_use]
    pub fn owns(&self, n: usize) -> bool {
        self.runs.iter().any(|(a, b)| *a <= n && n <= *b)
    }
    /// How many own lines it has.
    #[must_use]
    pub fn lines(&self) -> usize {
        self.runs.iter().map(|(a, b)| b + 1 - a).sum()
    }
    /// Its own lines of `text` (the file's text), joined by newlines.
    #[must_use]
    pub fn text(&self, text: &str) -> String {
        let lines: Vec<&str> = text.lines().collect();
        let mut out: Vec<&str> = Vec::new();
        for (a, b) in &self.runs {
            out.extend(lines.get(a.saturating_sub(1)..(*b).min(lines.len())).unwrap_or_default());
        }
        out.join("\n")
    }
    /// The attached comment lines of `text`, joined by newlines (empty when none).
    #[must_use]
    pub fn described_text(&self, text: &str) -> String {
        let Some((a, b)) = self.described else { return String::new() };
        text.lines().skip(a.saturating_sub(1)).take((b + 1).saturating_sub(a)).collect::<Vec<_>>().join("\n")
    }
}

/// The items of a material version, with the links between its pieces and its items.
#[derive(Clone, Debug, Default)]
pub struct Items {
    pub items: Vec<Item>,
    /// Per piece (chunk index), the items its lines meet.
    pub of_piece: Vec<Vec<usize>>,
    /// Per item, the pieces its lines meet.
    pub pieces_of: Vec<Vec<usize>>,
    /// The substrate version they were built on.
    pub version: u64,
}

/// A line attached above a definition: a comment, a doc comment, an attribute or a decorator.
fn attached(line: &str) -> bool {
    let t = line.trim_start();
    !t.is_empty()
        && (t.starts_with("//")
            || t.starts_with("/*")
            || t.starts_with("* ")
            || t == "*"
            || t.starts_with("*/")
            || t.starts_with("#[")
            || t.starts_with('@')
            || (t.starts_with('#') && !t.starts_with("#include") && !t.starts_with("#define") && !t.starts_with("#!")))
}

/// A comment line (not an attribute or a decorator).
fn comment(line: &str) -> bool {
    attached(line) && !line.trim_start().starts_with("#[") && !line.trim_start().starts_with('@')
}

/// One item found in a file: its kind, runs of own lines, attached comment lines, definition
/// line and label.
type Found = (Kind, Vec<(usize, usize)>, Option<(usize, usize)>, usize, String);

/// A definition's extent in its file: its first attached line, its last line, its first line, its
/// attached comment lines and its label.
struct Extent {
    top: usize,
    end: usize,
    def: usize,
    described: Option<(usize, usize)>,
    label: String,
}

/// The items of one code file (`lines`) with definitions `defs`, in file order of their first line.
fn code_items(lines: &[&str], defs: &[&Symbol]) -> Vec<Found> {
    let n = lines.len();
    let mut ext: Vec<Extent> = Vec::new();
    for s in defs {
        if s.line == 0 || s.line > n {
            continue;
        }
        let mut top = s.line;
        while top > 1 && lines.get(top - 2).is_some_and(|l| attached(l)) {
            top -= 1;
        }
        // The attached comment lines: the first run of comment lines from the top (attributes and
        // decorators are part of the extent, not of the description).
        let mut a = top;
        while a < s.line && !lines.get(a - 1).is_some_and(|l| comment(l)) {
            a += 1;
        }
        let described = (a < s.line).then(|| {
            let mut b = a;
            while b + 1 < s.line && lines.get(b).is_some_and(|l| comment(l)) {
                b += 1;
            }
            (a, b)
        });
        let label = if s.parent.is_empty() { format!("{} {}", s.kind, s.name) } else { format!("{} {}.{}", s.kind, s.parent, s.name) };
        ext.push(Extent { top, end: s.end.max(s.line).min(n), def: s.line, described, label });
    }
    // Every line to the innermost extent holding it (the smallest; on equal size the later start).
    let mut owner: Vec<Option<usize>> = vec![None; n + 1];
    let mut by_size: Vec<usize> = (0..ext.len()).collect();
    by_size.sort_by_key(|i| ext.get(*i).map_or((0, 0), |e| (usize::MAX - (e.end - e.top), e.top)));
    for i in by_size {
        let Some(e) = ext.get(i) else { continue };
        for slot in owner.iter_mut().take(e.end + 1).skip(e.top) {
            *slot = Some(i);
        }
    }
    // The file's head (`usize::MAX`), then each line no definition holds to the next definition
    // (after the last, to the last).
    let head = usize::MAX;
    let first = (1..=n).find(|l| owner.get(*l).copied().flatten().is_some());
    let last_owner = (1..=n).rev().find_map(|l| owner.get(l).copied().flatten());
    let mut next: Option<usize> = None;
    for l in (1..=n).rev() {
        match owner.get(l).copied().flatten() {
            Some(o) => next = Some(o),
            None => {
                let to = if first.is_none_or(|f| l < f) { head } else { next.or(last_owner).unwrap_or(head) };
                if let Some(slot) = owner.get_mut(l) {
                    *slot = Some(to);
                }
            }
        }
    }
    // Runs per owner, in line order.
    let mut runs: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = std::collections::BTreeMap::new();
    for l in 1..=n {
        let Some(o) = owner.get(l).copied().flatten() else { continue };
        let r = runs.entry(o).or_default();
        match r.last_mut() {
            Some(last) if last.1 + 1 == l => last.1 = l,
            _ => r.push((l, l)),
        }
    }
    let mut out: Vec<Found> = runs
        .into_iter()
        .map(|(o, r)| match ext.get(o) {
            Some(e) => (Kind::Definition, r, e.described, e.def, e.label.clone()),
            None => (Kind::Head, r, None, 0, String::new()),
        })
        .collect();
    out.sort_by_key(|x| x.1.first().map_or(0, |r| r.0));
    out
}

/// The items of one document with heading blocks: the head, then each heading to the line before
/// the next heading.
fn section_items(n: usize, headings: &[(usize, String)]) -> Vec<Found> {
    let mut out = Vec::new();
    if let Some((h, _)) = headings.first()
        && *h > 1
    {
        out.push((Kind::Head, vec![(1, (h - 1).min(n))], None, 0, String::new()));
    }
    for (i, (h, label)) in headings.iter().enumerate() {
        let end = headings.get(i + 1).map_or(n, |x| x.0.saturating_sub(1)).min(n);
        if *h <= end {
            out.push((Kind::Section, vec![(*h, end)], None, *h, label.clone()));
        }
    }
    out
}

/// Build the items of `sub` and their links to its pieces.
#[must_use]
pub fn build(sub: &Substrate) -> Items {
    let mut items: Vec<Item> = Vec::new();
    let mut by_file: Vec<Vec<&Symbol>> = vec![Vec::new(); sub.files.len()];
    let index: std::collections::BTreeMap<&str, usize> = sub.files.iter().enumerate().map(|(i, f)| (f.path.as_str(), i)).collect();
    for s in &sub.symbols {
        if let Some(i) = index.get(s.file.as_str())
            && let Some(v) = by_file.get_mut(*i)
        {
            v.push(s);
        }
    }
    let mut pieces_by_file: Vec<Vec<usize>> = vec![Vec::new(); sub.files.len()];
    for (c, ch) in sub.chunks.iter().enumerate() {
        if let Some(v) = pieces_by_file.get_mut(ch.file) {
            v.push(c);
        }
    }
    for (fi, f) in sub.files.iter().enumerate() {
        let lines: Vec<&str> = f.text.lines().collect();
        let n = lines.len();
        let tables = if f.decoder == "toml" { crate::piece::toml_tables(&f.path, &f.text) } else { Vec::new() };
        let table_refs: Vec<&Symbol> = tables.iter().collect();
        let defs: &[&Symbol] = if f.decoder == "toml" { &table_refs } else { by_file.get(fi).map_or(&[][..], Vec::as_slice) };
        let headings: Vec<(usize, String)> =
            if f.is_code() { Vec::new() } else { f.blocks.iter().filter(|b| b.kind.level().is_some()).map(|b| (b.first, b.label.clone())).collect() };
        let found = if f.is_code() && !defs.is_empty() {
            code_items(&lines, defs)
        } else if !headings.is_empty() {
            section_items(n, &headings)
        } else {
            Vec::new()
        };
        if found.is_empty() {
            // The file's pieces, as they are.
            for c in pieces_by_file.get(fi).map_or(&[][..], Vec::as_slice) {
                if let Some(ch) = sub.chunks.get(*c) {
                    items.push(Item { file: fi, kind: Kind::Piece, runs: vec![(ch.start, ch.end)], described: None, def: 0, label: ch.heading.clone() });
                }
            }
            continue;
        }
        for (kind, runs, described, def, label) in found {
            items.push(Item { file: fi, kind, runs, described, def, label });
        }
    }
    // Links: a piece meets an item when one of its lines is one of the item's own.
    let mut of_piece: Vec<Vec<usize>> = vec![Vec::new(); sub.chunks.len()];
    let mut pieces_of: Vec<Vec<usize>> = vec![Vec::new(); items.len()];
    let mut items_by_file: Vec<Vec<usize>> = vec![Vec::new(); sub.files.len()];
    for (i, it) in items.iter().enumerate() {
        if let Some(v) = items_by_file.get_mut(it.file) {
            v.push(i);
        }
    }
    for (c, ch) in sub.chunks.iter().enumerate() {
        for i in items_by_file.get(ch.file).map_or(&[][..], Vec::as_slice) {
            let Some(it) = items.get(*i) else { continue };
            if it.runs.iter().any(|(a, b)| *a <= ch.end && *b >= ch.start) {
                if let Some(v) = of_piece.get_mut(c) {
                    v.push(*i);
                }
                if let Some(v) = pieces_of.get_mut(*i) {
                    v.push(c);
                }
            }
        }
    }
    Items { items, of_piece, pieces_of, version: sub.version }
}

/// The partition check over every file of `sub`: (code files, their lines, the lines in exactly
/// one item; document and other files, their lines, the lines in exactly one item; items by kind:
/// definitions, heads, sections, pieces).
#[must_use]
pub fn partition(sub: &Substrate, items: &Items) -> ([usize; 3], [usize; 3], [usize; 4]) {
    let mut code = [0_usize; 3];
    let mut other = [0_usize; 3];
    let mut per_file: Vec<Vec<&Item>> = vec![Vec::new(); sub.files.len()];
    for it in &items.items {
        if let Some(v) = per_file.get_mut(it.file) {
            v.push(it);
        }
    }
    for (fi, f) in sub.files.iter().enumerate() {
        let n = f.text.lines().count();
        let mine = per_file.get(fi).map_or(&[][..], Vec::as_slice);
        let mut count = vec![0_u32; n + 1];
        for it in mine {
            for (a, b) in &it.runs {
                for slot in count.iter_mut().take((*b).min(n) + 1).skip(*a) {
                    *slot += 1;
                }
            }
        }
        let once = (1..=n).filter(|l| count.get(*l) == Some(&1)).count();
        let t = if f.is_code() { &mut code } else { &mut other };
        t[0] += 1;
        t[1] += n;
        t[2] += once;
    }
    let mut kinds = [0_usize; 4];
    for it in &items.items {
        let k = match it.kind {
            Kind::Definition => 0,
            Kind::Head => 1,
            Kind::Section => 2,
            Kind::Piece => 3,
        };
        if let Some(x) = kinds.get_mut(k) {
            *x += 1;
        }
    }
    (code, other, kinds)
}

/// Section check: a small Rust file: the head holds the module comment and the import; a
/// function's doc comment is attached to it and named as its description; a constant inside a
/// function is its own item and leaves a hole in the function's lines; every line is in exactly
/// one item; a Markdown file is cut at its headings.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    let rs = "//! notes\nuse x;\n\n/// Grows the ledger.\n#[inline]\npub fn grow(x: u8) -> u8 {\n    const STEP: u8 = 2;\n    x + STEP\n}\n\nimpl A {\n    fn b() {}\n}\n";
    let md = "intro\n# Pump\ntext\n## Care\nmore\n";
    let embed: crate::Embed = std::sync::Arc::new(|_: &str| vec![0.0]);
    let sub = crate::from_files(&[("a.rs", rs), ("m.md", md)], embed);
    let it = build(&sub);
    let file = |p: &str| sub.files.iter().position(|f| f.path == p);
    let (Some(a), Some(m)) = (file("a.rs"), file("m.md")) else { return Err("nmsubs.items: check files".into()) };
    let of = |fi: usize| it.items.iter().filter(|x| x.file == fi).map(|x| (x.kind, x.runs.clone(), x.described, x.def)).collect::<Vec<_>>();
    let want_rs = vec![
        (Kind::Head, vec![(1, 3)], None, 0),
        (Kind::Definition, vec![(4, 6), (8, 9)], Some((4, 4)), 6),
        (Kind::Definition, vec![(7, 7)], None, 7),
        (Kind::Definition, vec![(10, 13)], None, 12),
    ];
    if of(a) != want_rs {
        return Err(format!("nmsubs.items: rust items {:?}", of(a)));
    }
    let want_md = vec![(Kind::Head, vec![(1, 1)], None, 0), (Kind::Section, vec![(2, 3)], None, 2), (Kind::Section, vec![(4, 5)], None, 4)];
    if of(m) != want_md {
        return Err(format!("nmsubs.items: markdown items {:?}", of(m)));
    }
    let (code, other, _) = partition(&sub, &it);
    if code[1] != code[2] || other[1] != other[2] || code[1] != 13 {
        return Err(format!("nmsubs.items: partition {code:?} {other:?}"));
    }
    if it.pieces_of.iter().any(Vec::is_empty) || it.of_piece.iter().any(Vec::is_empty) {
        return Err("nmsubs.items: a piece or an item with no link".into());
    }
    Ok(())
}
