//! The document model inside the substrate: code files mapped into it (one section per file, an
//! `Item` block per symbol, `Other` blocks between), heading paths and outlines of any file.
use crate::repo::{FileInfo, Symbol};
use nmtype::doc::{Block, BlockKind};

/// Blocks of a code file: for Rust an `Item` block per symbol, starting at the symbol's doc
/// comments and attributes, `Other` for the lines before the first; for TOML an `Item` block per
/// table header; for other code one `Other` block. Every line is in exactly one block.
#[must_use]
pub fn code_blocks(decoder: &str, text: &str, symbols: &[&Symbol]) -> Vec<Block> {
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    let mut blocks: Vec<Block> = Vec::new();
    match decoder {
        "rust" => {
            for s in symbols {
                let mut at = s.line;
                while at > 1
                    && lines.get(at - 2).is_some_and(|x| {
                        let t = x.trim_start();
                        t.starts_with("///") || t.starts_with("#[")
                    })
                {
                    at -= 1;
                }
                if at >= 1 && at <= n && !blocks.iter().any(|b| b.first == at) {
                    blocks.push(Block::new(BlockKind::Item, at, at, &format!("{} {}", s.kind, s.name)));
                }
            }
        }
        lang if nmlex::lang::by_id(lang).is_some() => {
            for s in symbols {
                if s.line >= 1 && s.line <= n && !blocks.iter().any(|b| b.first == s.line) {
                    let label = if s.parent.is_empty() { format!("{} {}", s.kind, s.name) } else { format!("{} {}.{}", s.kind, s.parent, s.name) };
                    blocks.push(Block::new(BlockKind::Item, s.line, s.line, &label));
                }
            }
        }
        "toml" => {
            for (i, l) in lines.iter().enumerate() {
                let t = l.trim();
                if t.starts_with('[') && t.ends_with(']') {
                    blocks.push(Block::new(BlockKind::Item, i + 1, i + 1, t.trim_matches(['[', ']'])));
                }
            }
        }
        _ => {}
    }
    blocks.sort_by_key(|b| b.first);
    if blocks.first().is_some_and(|b| b.first > 1) {
        blocks.insert(0, Block::new(BlockKind::Other, 1, 1, ""));
    }
    nmdoc::md::close_gaps(&mut blocks, n);
    if let Some(b) = blocks.first_mut()
        && b.kind == BlockKind::Paragraph
    {
        b.kind = BlockKind::Other;
    }
    blocks
}

/// The heading path of a line of a file, top first, joined by ` > ` (empty for code and for lines
/// above the first heading).
#[must_use]
pub fn heading_path_of(f: &FileInfo, line: usize) -> String {
    nmtype::doc::block_at(&f.blocks, line).map(|i| nmtype::doc::heading_path(&f.blocks, i).join(" > ")).unwrap_or_default()
}

/// What a pack names for a span of a document starting at `line`: for a paged document (a PDF)
/// its page locator `path:p12` first, then the heading path (`heading_path_of`).
#[must_use]
pub fn heading_of(f: &FileInfo, line: usize) -> String {
    let section = heading_path_of(f, line);
    let page = nmtype::doc::block_at(&f.blocks, line).and_then(|i| f.blocks.get(i)).and_then(|b| b.loc.page.map(|p| (p.0, b.loc.page_label.clone())));
    match page {
        Some((p, label)) => {
            let printed = label.filter(|l| *l != p.to_string()).map(|l| format!(" (printed page {l})")).unwrap_or_default();
            let place = format!("{}:p{p}{printed}", f.path);
            if section.is_empty() { place } else { format!("{place} | {section}") }
        }
        None => section,
    }
}

/// The outline of a file: (line, level, heading path).
#[must_use]
pub fn outline(f: &FileInfo) -> Vec<(usize, u8, String)> {
    f.blocks
        .iter()
        .enumerate()
        .filter_map(|(i, b)| b.kind.level().map(|l| (b.first, l, nmtype::doc::heading_path(&f.blocks, i).join(" > "))))
        .collect()
}

/// Size budget of a document chunk, in tokens (characters / 4): a chunk is filled with whole
/// blocks up to `soft`, never beyond `hard` unless one block alone is larger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub soft: usize,
    pub hard: usize,
}

/// How documents are cut into chunks: 30-line windows (as code) or structure-aware blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cut {
    Windows,
    Blocks(Budget),
}

/// The cut in use. Decided by any-data step D3 (blocks at soft 300, hard 500 tokens against
/// 30-line windows and a smaller budget, on the document development set). `NANO_DOC_CHUNKS`
/// (`windows` or `blocks:<soft>:<hard>`) overrides it for measurement.
#[must_use]
pub fn cut() -> Cut {
    let fallback = Cut::Blocks(Budget { soft: 300, hard: 500 });
    match std::env::var("NANO_DOC_CHUNKS").ok().as_deref() {
        Some("windows") => Cut::Windows,
        Some(v) if v.starts_with("blocks:") => {
            let n: Vec<usize> = v.split(':').skip(1).filter_map(|x| x.parse().ok()).collect();
            match n.as_slice() {
                [s, h] if *s > 0 && h >= s => Cut::Blocks(Budget { soft: *s, hard: *h }),
                _ => fallback,
            }
        }
        _ => fallback,
    }
}

/// One document chunk: its lines, the kind of its main block, its heading path (or record key,
/// table header, log time range) and the lines of the section it belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seg {
    pub start: usize,
    pub end: usize,
    pub kind: &'static str,
    pub heading: String,
    pub section: Option<(usize, usize)>,
}

/// Tokens of a run of lines (characters / 4, the Nano's rule).
fn tokens(lines: &[&str], s: usize, e: usize) -> usize {
    lines.get(s.saturating_sub(1)..e.min(lines.len())).map_or(0, |p| p.iter().map(|l| l.chars().count() + 1).sum::<usize>()) / 4
}

/// Cut lines `s..=e` into pieces of at most `budget` tokens at line ends (a piece of one line may
/// be larger); `prefer` marks good cut points (sentence ends), used when one lies in the last
/// third of a piece.
fn split_lines(lines: &[&str], s: usize, e: usize, budget: usize, prefer: &dyn Fn(usize) -> bool) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut a = s;
    while a <= e {
        let mut b = a;
        while b < e && tokens(lines, a, b + 1) <= budget {
            b += 1;
        }
        if b < e {
            let lo = a + (b - a) * 2 / 3;
            if let Some(c) = (lo..=b).rev().find(|x| prefer(*x)) {
                b = c;
            }
        }
        out.push((a, b));
        a = b + 1;
    }
    out
}

/// Structure-aware chunks of a document (any-data design 6.3): a chunk is whole consecutive
/// blocks up to the soft budget and never crosses a heading, except that a small section merges
/// with the sections after it under the same parent; a heading joins the text below it; a list
/// is kept whole when it fits the hard budget; a caption joins its table; a block larger than the
/// hard budget is split at sentence ends (paragraphs), at row groups (tables; the header is named
/// in every piece's heading) or at line ends. Every line is in exactly one chunk.
#[must_use]
pub fn segments(blocks: &[Block], text: &str, budget: Budget) -> Vec<Seg> {
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    if n == 0 || blocks.is_empty() {
        return vec![Seg { start: 1, end: n.max(1), kind: "paragraph", heading: String::new(), section: None }];
    }
    // Atoms: (first, last, starts with a heading of this level, main kind, table header lines).
    struct Atom {
        first: usize,
        last: usize,
        level: Option<u8>,
        kind: &'static str,
        head: Option<(usize, usize)>,
        label: Option<String>,
    }
    let mut atoms: Vec<Atom> = Vec::new();
    let mut i = 0;
    let sentence_end = |x: usize| lines.get(x - 1).is_some_and(|l| l.trim_end().ends_with(['.', '?', '!', ':', ';']));
    while i < blocks.len() {
        let Some(b) = blocks.get(i) else { break };
        let mut level = None;
        let first = b.first;
        let mut j = i;
        // A heading joins the block below it (several headings in a row join together).
        while let Some(h) = blocks.get(j).and_then(|x| x.kind.level()) {
            level = level.or(Some(h));
            if j + 1 >= blocks.len() {
                break;
            }
            j += 1;
        }
        let body = blocks.get(j).unwrap_or(b);
        if body.kind.level().is_some() {
            atoms.push(Atom { first, last: body.last, level, kind: "heading", head: None, label: None });
            i = j + 1;
            continue;
        }
        // A list: consecutive items, whole when it fits the hard budget.
        let mut last = body.last;
        let mut k = j;
        if matches!(body.kind, BlockKind::ListItem { .. }) {
            while blocks.get(k + 1).is_some_and(|x| matches!(x.kind, BlockKind::ListItem { .. })) {
                k += 1;
            }
            let list_last = blocks.get(k).map_or(last, |x| x.last);
            if tokens(&lines, first, list_last) <= budget.hard {
                last = list_last;
            } else {
                k = j;
            }
        }
        // A caption joins the table after it, and a table the caption after it.
        let is_table = |x: &Block| matches!(x.kind, BlockKind::Table { .. });
        if body.kind == BlockKind::Caption && blocks.get(k + 1).is_some_and(is_table) {
            k += 1;
            last = blocks.get(k).map_or(last, |x| x.last);
        }
        let main = blocks.get(k).unwrap_or(body);
        if is_table(main) && blocks.get(k + 1).is_some_and(|x| x.kind == BlockKind::Caption) {
            k += 1;
            last = blocks.get(k).map_or(last, |x| x.last);
        }
        let kind = main.kind.name();
        let data_table = matches!(main.kind, BlockKind::Table { head } if head > 0) && main.label.starts_with("columns:") && main.first == first;
        if let (true, BlockKind::Table { head }) = (data_table, main.kind) {
            // A data table (CSV, TSV): its header lines are the schema chunk, named by the schema;
            // its rows are cut into groups, each naming the header.
            let head_end = (main.first + usize::from(head)).saturating_sub(1).min(main.last);
            let header = Some((main.first, head_end));
            atoms.push(Atom { first, last: head_end, level, kind: "schema", head: header, label: Some(format!("table {}", main.label)) });
            if head_end < last {
                for (a, z) in split_lines(&lines, head_end + 1, last, budget.soft, &|_| false) {
                    atoms.push(Atom { first: a, last: z, level: None, kind, head: header, label: None });
                }
            }
        } else if tokens(&lines, first, last) <= budget.hard {
            atoms.push(Atom { first, last, level, kind, head: None, label: None });
        } else if let BlockKind::Table { head } = main.kind {
            // Row groups; the first group keeps the header lines and anything before them.
            let head_end = (main.first + usize::from(head)).saturating_sub(1).min(main.last);
            let header = (head > 0).then_some((main.first, head_end));
            let rows = split_lines(&lines, head_end + 1, last.max(head_end + 1), budget.soft, &|_| false);
            for (g, (a, z)) in rows.into_iter().enumerate() {
                let a = if g == 0 { first } else { a };
                atoms.push(Atom { first: a, last: z, level: if g == 0 { level } else { None }, kind, head: header, label: None });
            }
        } else {
            let prefer: &dyn Fn(usize) -> bool = if kind == "paragraph" { &sentence_end } else { &|_| false };
            for (g, (a, z)) in split_lines(&lines, first, last, budget.soft, prefer).into_iter().enumerate() {
                atoms.push(Atom { first: a, last: z, level: if g == 0 { level } else { None }, kind, head: None, label: None });
            }
        }
        i = k + 1;
    }
    // Pack atoms into chunks.
    type Packed = (usize, usize, Option<u8>, Vec<&'static str>, Option<(usize, usize)>, Option<String>);
    let mut segs: Vec<Packed> = Vec::new();
    for a in &atoms {
        let size = tokens(&lines, a.first, a.last);
        let merge = match segs.last() {
            None => false,
            Some((s, _, lvl, _, head, _)) => {
                let cur = tokens(&lines, *s, a.first - 1);
                let fits = cur + size <= budget.soft;
                if head.is_some() || a.head.is_some() {
                    false
                } else if let Some(l) = a.level {
                    // A new section: merge only a small current chunk with a same-level or deeper heading.
                    fits && cur < budget.soft / 3 && lvl.is_some_and(|c| l >= c)
                } else {
                    fits
                }
            }
        };
        if merge {
            if let Some(last) = segs.last_mut() {
                last.1 = a.last;
                last.3.push(a.kind);
            }
        } else {
            segs.push((a.first, a.last, a.level, vec![a.kind], a.head, a.label.clone()));
        }
    }
    segs.into_iter()
        .map(|(s, e, _, kinds, head, label)| {
            let kind = kinds.iter().copied().find(|k| *k != "heading").unwrap_or("heading");
            let at = nmtype::doc::block_at(blocks, s).unwrap_or(0);
            let mut heading = nmtype::doc::heading_path(blocks, at).join(" > ");
            if let Some(l) = label {
                heading = if heading.is_empty() { l } else { format!("{heading} | {l}") };
            }
            // A table piece without its header names the header in its heading.
            if let Some((h1, h2)) = head
                && s > h2
            {
                let header: Vec<&str> = lines.get(h1 - 1..h2).map(<[&str]>::to_vec).unwrap_or_default();
                let header = header.join(" ");
                heading = if heading.is_empty() { format!("header: {header}") } else { format!("{heading} | header: {header}") };
            }
            // A run of log events is named by its time range and event count.
            if kind == "log" && heading.is_empty() {
                let ev: Vec<&Block> = blocks.iter().filter(|b| b.kind == BlockKind::LogEvent && b.first >= s && b.first <= e).collect();
                if let (Some(a), Some(z)) = (ev.first(), ev.last()) {
                    heading = format!("log {} to {}, {} events", a.label, z.label, ev.len());
                }
            }
            let section = section_of(blocks, at, n);
            Seg { start: s, end: e, kind, heading, section }
        })
        .collect()
}

/// The lines of the section holding block `at`: from its heading to the line before the next
/// heading of the same or a higher level (or the end).
fn section_of(blocks: &[Block], at: usize, n: usize) -> Option<(usize, usize)> {
    let b = blocks.get(at)?;
    let h = if b.kind.level().is_some() { at } else { b.parent? };
    let hb = blocks.get(h)?;
    let level = hb.kind.level()?;
    let end = blocks.iter().skip(h + 1).find(|x| x.kind.level().is_some_and(|l| l <= level)).map_or(n, |x| x.first - 1);
    Some((hb.first, end))
}

pub(crate) fn check() -> Result<(), String> {
    let src = "//! notes\nuse x;\n/// doc\n#[derive(Debug)]\npub struct A;\n\nfn b() {}\n";
    let syms = crate::repo::parse_items("a.rs", src);
    let refs: Vec<&Symbol> = syms.iter().collect();
    let b = code_blocks("rust", src, &refs);
    nmtype::doc::check_cover(&b, 7).map_err(|e| format!("nmsubs.doc: {e}"))?;
    let got: Vec<(usize, usize, &str)> = b.iter().map(|x| (x.first, x.last, x.label.as_str())).collect();
    if got != [(1, 2, ""), (3, 6, "struct A"), (7, 7, "fn b")] {
        return Err(format!("nmsubs.doc: rust blocks {got:?}"));
    }
    let f = FileInfo::new("m.md", "# Pump\n\n## Care\n\ntext\n");
    if heading_of(&f, 5) != "Pump > Care" || outline(&f).len() != 2 {
        return Err(format!("nmsubs.doc: heading {:?}", heading_of(&f, 5)));
    }
    Ok(())
}
