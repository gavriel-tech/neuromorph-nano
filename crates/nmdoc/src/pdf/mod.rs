//! The PDF reader. A PDF's text sits in compressed content streams as glyph codes placed on the
//! page; this reader opens the file (`file`: cross-reference, object streams, repair,
//! encryption with an empty user password through `crypt`), decodes streams (`filters`), runs
//! each page's content (`content`) with its fonts mapped to characters (`font`), and puts the
//! glyphs back into words, lines and paragraphs in reading order (`layout`: by the tagged
//! structure when the file has one, else by geometry with columns). Over the whole document it
//! removes running headers and footers, joins words hyphenated at line ends, and marks headings
//! (from the tagged structure, else the outline, else font size, see `structure`).
//!
//! The derived text has one paragraph (or table row, or code line) per line, pages separated by
//! an empty line; every block carries its page (`Loc::page`) and printed page label. Limits, said
//! plainly: a scanned page holds no text and is reported, not indexed; a font with no way to map
//! its glyphs to characters gives unreadable text, which is detected per page and reported, not
//! indexed; a file that needs a real password is refused.
pub mod content;
pub mod crypt;
pub mod file;
pub mod filters;
pub mod fixtures;
pub mod font;
pub mod layout;
pub mod lex;
pub mod structure;
mod tables;

use layout::{Kind, Para};
use nmtype::doc::{Block, BlockKind, Doc};
use std::collections::{HashMap, HashSet};

/// Share of a page's glyphs that may lack a character before the page counts as unreadable.
pub const UNREADABLE_SHARE: f64 = 0.3;
/// Share of a page that pictures must cover for a page with no text to count as a scan.
pub const PICTURE_SHARE: f64 = 0.1;

/// How to read.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Use the tagged structure for reading order when the file has one.
    pub tags: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { tags: true }
    }
}

/// Read a PDF.
/// # Errors
/// Why it could not be read: not a PDF, no catalog, or encrypted with a password.
pub fn read(path: &str, bytes: &[u8]) -> Result<Doc, &'static str> {
    read_with(path, bytes, Options::default())
}

/// What one page gave.
struct PageOut {
    paras: Vec<Para>,
    height: f64,
}

/// Read a PDF with options.
/// # Errors
/// As `read`.
pub fn read_with(path: &str, bytes: &[u8], opts: Options) -> Result<Doc, &'static str> {
    let f = file::File::open(bytes)?;
    let pages = f.pages();
    let page_index: HashMap<u32, usize> = pages.iter().enumerate().filter_map(|(i, (n, _))| n.map(|n| (n, i))).collect();
    let marks = if opts.tags { structure::tags(&f, &page_index) } else { HashMap::new() };
    let bookmarks = structure::outline(&f, &page_index, pages.len());
    let labels = structure::page_labels(&f, pages.len());
    let mut notes: Vec<String> = Vec::new();
    if let Some(c) = &f.crypt {
        notes.push(c.describe());
    }
    if f.repaired {
        notes.push("cross-reference table missing or damaged: rebuilt by scanning the file".into());
    }
    let mut not_read: Vec<String> = Vec::new();
    let mut unmapped_dropped = 0usize;
    let mut skipped: Vec<&'static str> = Vec::new();
    let mut out: Vec<PageOut> = Vec::new();
    let mut tagged_pages = 0;
    for (i, (_, d)) in pages.iter().enumerate() {
        let pg = content::page(&f, d);
        for s in &pg.skipped {
            if !skipped.contains(s) {
                skipped.push(s);
            }
        }
        let visible: Vec<&content::Glyph> = pg.glyphs.iter().filter(|g| !g.text.trim().is_empty() || g.unmapped).collect();
        let unmapped = visible.iter().filter(|g| g.unmapped || g.text.chars().all(|c| ('\u{e000}'..='\u{f8ff}').contains(&c))).count();
        let n = i + 1;
        if visible.is_empty() {
            if pg.image_cover >= PICTURE_SHARE {
                not_read.push(format!("page {n} (no text layer: a picture only, such as a scan)"));
            }
            out.push(PageOut { paras: Vec::new(), height: pg.height });
            continue;
        }
        if unmapped as f64 > visible.len() as f64 * UNREADABLE_SHARE {
            not_read.push(format!("page {n} (its font has no map from glyphs to characters: text unreadable)"));
            out.push(PageOut { paras: Vec::new(), height: pg.height });
            continue;
        }
        unmapped_dropped += unmapped;
        let glyphs: Vec<content::Glyph> = pg.glyphs.into_iter().filter(|g| !g.unmapped).collect();
        let page_marks = marks.get(&i);
        let covered = page_marks.map_or(0, |m| glyphs.iter().filter(|g| !g.text.trim().is_empty() && (g.artifact || g.mcid.is_some_and(|c| m.contains_key(&c)))).count());
        let solid = glyphs.iter().filter(|g| !g.text.trim().is_empty()).count();
        let paras = match page_marks {
            Some(m) if covered * 10 >= solid * 8 => {
                tagged_pages += 1;
                layout::by_tags(&glyphs, m)
            }
            _ => layout::by_geometry(&glyphs),
        };
        out.push(PageOut { paras, height: pg.height });
    }
    if !skipped.is_empty() {
        notes.push(format!("streams not decoded: {}", skipped.join(", ")));
    }
    if unmapped_dropped > 0 {
        notes.push(format!("{unmapped_dropped} glyphs with no character map left out"));
    }
    for r in &not_read {
        notes.push(format!("not indexed: {r}"));
    }
    remove_running(&mut out);
    headings(&mut out, &bookmarks, tagged_pages > 0);
    let info = f.at(&f.trailer, "Info");
    let meta_title = info.as_dict().and_then(|d| f.at(d, "Title").as_str().map(file::text_string)).map(|t| t.trim().to_owned()).unwrap_or_default();
    Ok(assemble(path, &out, &labels, &not_read, notes, &meta_title))
}

fn norm_running(t: &str) -> String {
    let mut s = String::new();
    let mut last_digit = false;
    for c in t.chars() {
        if c.is_ascii_digit() || matches!(c, 'i' | 'v' | 'x' | 'l') && t.chars().all(|x| "ivxlcdm ".contains(x)) {
            if !last_digit {
                s.push('#');
            }
            last_digit = true;
        } else {
            last_digit = false;
            if !c.is_whitespace() {
                s.extend(c.to_lowercase());
            }
        }
    }
    s
}

/// Remove running headers and footers: paragraphs in the top or bottom tenth of the page whose
/// text, with numbers masked, recurs in the same band on at least half the pages with text (and
/// on two pages at least); and lone page numbers in those bands.
fn remove_running(pages: &mut [PageOut]) {
    let with_text = pages.iter().filter(|p| !p.paras.is_empty()).count();
    let band = |p: &Para, h: f64| -> Option<bool> {
        if h <= 0.0 || p.size == 0.0 {
            return None;
        }
        if p.bottom <= h * 0.1 {
            Some(true)
        } else if p.top >= h * 0.9 {
            Some(false)
        } else {
            None
        }
    };
    // Counted by part: a header row read as one line on one page (two pieces with a tab) and as
    // two lines on another (split by columns) still counts as the same parts.
    let parts = |p: &Para| -> Vec<String> { p.lines.iter().flat_map(|l| l.split('\t')).map(norm_running).filter(|k| !k.is_empty()).collect() };
    let mut count: HashMap<(bool, String), usize> = HashMap::new();
    for pg in pages.iter() {
        let mut seen = HashSet::new();
        for p in &pg.paras {
            if let Some(top) = band(p, pg.height) {
                for k in parts(p) {
                    if seen.insert((top, k.clone())) {
                        *count.entry((top, k)).or_default() += 1;
                    }
                }
            }
        }
    }
    let need = (with_text.div_ceil(2)).max(2);
    let lone_number = |k: &str| k.chars().all(|c| c == '#' || c == '-' || c == '.') || k.strip_prefix("page").is_some_and(|r| r.chars().all(|c| "#of/".contains(c)));
    for pg in pages.iter_mut() {
        let h = pg.height;
        pg.paras.retain(|p| {
            let Some(top) = band(p, h) else { return true };
            let ks = parts(p);
            ks.is_empty() || !ks.iter().all(|k| lone_number(k) || count.get(&(top, k.clone())).copied().unwrap_or(0) >= need)
        });
    }
}

fn norm_title(t: &str) -> String {
    t.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// Mark headings: the tagged structure's when it gave any; else the outline's titles matched on
/// their pages; else paragraphs set larger than the body text.
fn headings(pages: &mut [PageOut], bookmarks: &[structure::Bookmark], tagged: bool) {
    let tagged_heads = tagged && pages.iter().flat_map(|p| &p.paras).any(|p| matches!(p.kind, Kind::Heading(_)));
    if tagged_heads {
        return;
    }
    let mut matched = 0;
    for b in bookmarks {
        let want = norm_title(&b.title);
        if want.is_empty() {
            continue;
        }
        let range: Vec<usize> = match b.page {
            Some(p) => vec![p],
            None => (0..pages.len()).collect(),
        };
        'find: for p in range {
            let Some(pg) = pages.get_mut(p) else { continue };
            for para in &mut pg.paras {
                let got = norm_title(&para.lines.join(" "));
                if !matches!(para.kind, Kind::Heading(_)) && (got == want || (got.starts_with(&want) && got.len() <= want.len() + 12)) {
                    para.kind = Kind::Heading(b.level.min(6));
                    matched += 1;
                    break 'find;
                }
            }
        }
    }
    if matched > 0 {
        return;
    }
    // Body size: the size holding the most characters.
    let mut by_size: HashMap<i64, usize> = HashMap::new();
    for p in pages.iter().flat_map(|p| &p.paras) {
        *by_size.entry((p.size * 2.0).round() as i64).or_default() += p.lines.iter().map(|l| l.chars().count()).sum::<usize>();
    }
    let Some(body) = by_size.iter().max_by_key(|(_, n)| **n).map(|(s, _)| *s as f64 / 2.0) else { return };
    let is_head = |p: &Para| {
        let t = p.lines.join(" ");
        p.kind == Kind::Text && p.size >= body * 1.15 && p.lines.len() <= 3 && t.chars().count() <= 160 && !t.trim_end().ends_with('.')
    };
    let mut sizes: Vec<i64> = pages.iter().flat_map(|p| &p.paras).filter(|p| is_head(p)).map(|p| (p.size * 2.0).round() as i64).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes.dedup();
    for p in pages.iter_mut().flat_map(|p| &mut p.paras) {
        if is_head(p) {
            let k = (p.size * 2.0).round() as i64;
            let level = sizes.iter().position(|s| *s == k).unwrap_or(0).min(3) + 1;
            p.kind = Kind::Heading(u8::try_from(level).unwrap_or(4));
        }
    }
}

/// Words of the document (lowercase) with their counts, and its hyphenated compounds.
fn vocabulary(pages: &[PageOut]) -> (HashMap<String, usize>, HashSet<String>) {
    let mut words: HashMap<String, usize> = HashMap::new();
    let mut compounds = HashSet::new();
    for l in pages.iter().flat_map(|p| &p.paras).flat_map(|p| &p.lines) {
        let toks: Vec<&str> = l.split(|c: char| !(c.is_alphanumeric() || c == '-')).collect();
        let last = toks.len().saturating_sub(1);
        for (i, t) in toks.iter().enumerate() {
            let t = t.trim_matches('-');
            if t.is_empty() {
                continue;
            }
            if t.contains('-') {
                // A hyphen at the very end of a line is no compound.
                if i != last || !l.trim_end().ends_with('-') {
                    compounds.insert(t.to_lowercase());
                }
                for part in t.split('-') {
                    *words.entry(part.to_lowercase()).or_default() += 1;
                }
            } else {
                *words.entry(t.to_lowercase()).or_default() += 1;
            }
        }
    }
    (words, compounds)
}

/// Join the lines of a paragraph. A word broken by a hyphen at a line end is joined whole unless
/// the document writes it with a hyphen elsewhere, or both halves are words of the document and
/// the whole word is not.
fn join_lines(lines: &[String], vocab: &(HashMap<String, usize>, HashSet<String>)) -> String {
    let mut out = String::new();
    for l in lines {
        let l = l.trim();
        if l.is_empty() {
            continue;
        }
        if out.is_empty() {
            out.push_str(l);
            continue;
        }
        // A soft hyphen where the line broke (alone, or before the hyphen drawn for it): the word
        // was broken by the layout, so it is joined whole.
        if let Some(cut) = out.strip_suffix("\u{ad}-").or_else(|| out.strip_suffix('\u{ad}')).map(str::len) {
            out.truncate(cut);
            out.push_str(l);
            continue;
        }
        let ends_hyphen = out.ends_with(['-', '\u{2010}']) && out.chars().rev().nth(1).is_some_and(char::is_alphabetic);
        let next_lower = l.chars().next().is_some_and(char::is_lowercase);
        // A code or name broken after its own hyphen (`AQ-` then `40T`): joined, hyphen kept.
        if out.ends_with('-') && out.chars().rev().nth(1).is_some_and(char::is_alphanumeric) && l.chars().next().is_some_and(|c| c.is_ascii_digit() || c.is_uppercase()) {
            out.push_str(l);
            continue;
        }
        if ends_hyphen && next_lower {
            let left: String = out.chars().rev().skip(1).take_while(|c| c.is_alphabetic()).collect::<Vec<_>>().into_iter().rev().collect();
            let right: String = l.chars().take_while(|c| c.is_alphabetic()).collect();
            let (lw, rw) = (left.to_lowercase(), right.to_lowercase());
            let whole = format!("{lw}{rw}");
            let compound = format!("{lw}-{rw}");
            let (words, compounds) = vocab;
            let keep = !words.contains_key(&whole)
                && (compounds.contains(&compound) || (lw.len() >= 3 && rw.len() >= 3 && words.contains_key(&lw) && words.contains_key(&rw)));
            if !keep {
                out.pop();
            }
            out.push_str(l);
            continue;
        }
        out.push(' ');
        out.push_str(l);
    }
    // Soft hyphens inside a line mark where a word may break; they are not text.
    out.replace('\u{ad}', "")
}

fn file_name(path: &str) -> &str {
    path.rsplit(['/', '!']).next().unwrap_or(path)
}

/// The document: derived text, blocks with pages, title and notes.
fn assemble(path: &str, pages: &[PageOut], labels: &[Option<String>], not_read: &[String], notes: Vec<String>, meta_title: &str) -> Doc {
    let vocab = vocabulary(pages);
    let mut lines: Vec<String> = Vec::new();
    let mut blocks: Vec<Block> = Vec::new();
    if !not_read.is_empty() {
        lines.push(format!("PDF of {} pages; not read: {}.", pages.len(), not_read.join("; ")));
        blocks.push(Block::new(BlockKind::Note, 1, 1, ""));
    }
    for (i, pg) in pages.iter().enumerate() {
        if pg.paras.is_empty() {
            continue;
        }
        if !lines.is_empty() {
            lines.push(String::new());
        }
        let page = u32::try_from(i + 1).unwrap_or(u32::MAX);
        let label = labels.get(i).cloned().flatten();
        let mut prev_row = false;
        for p in &pg.paras {
            let first = lines.len() + 1;
            let (kind, label_text) = match p.kind {
                Kind::Code => {
                    lines.extend(p.lines.iter().map(|l| l.trim_end().replace('\u{ad}', "")));
                    (BlockKind::Code, String::new())
                }
                Kind::Row { head } => {
                    lines.push(join_lines(&p.lines, &vocab));
                    if prev_row && let Some(b) = blocks.last_mut() {
                        b.last = lines.len();
                        prev_row = true;
                        continue;
                    }
                    prev_row = true;
                    (BlockKind::Table { head: u16::from(head) }, String::new())
                }
                Kind::Heading(l) => {
                    let t = join_lines(&p.lines, &vocab);
                    lines.push(t.clone());
                    (BlockKind::Heading(l), t)
                }
                Kind::List => {
                    let t = join_lines(&p.lines, &vocab);
                    let ordered = t.chars().next().is_some_and(|c| c.is_ascii_digit());
                    lines.push(t);
                    (BlockKind::ListItem { depth: 0, ordered }, String::new())
                }
                Kind::Text => {
                    lines.push(join_lines(&p.lines, &vocab));
                    (BlockKind::Paragraph, String::new())
                }
            };
            if !matches!(p.kind, Kind::Row { .. }) {
                prev_row = false;
            }
            let mut b = Block::new(kind, first, lines.len().max(first), &label_text);
            b.loc.page = Some((page, page));
            b.loc.page_label.clone_from(&label);
            blocks.push(b);
        }
    }
    let n = lines.len();
    let mut text = lines.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    // Blank lines between pages join the block above them.
    for i in 0..blocks.len() {
        let next = blocks.get(i + 1).map_or(n + 1, |b| b.first);
        if let Some(b) = blocks.get_mut(i) {
            b.last = next - 1;
        }
    }
    nmtype::doc::link_parents(&mut blocks);
    let title = if !meta_title.is_empty() && !meta_title.eq_ignore_ascii_case("untitled") {
        meta_title.to_owned()
    } else {
        blocks.iter().find(|b| b.kind.level().is_some()).map_or_else(|| file_name(path).to_owned(), |b| b.label.clone())
    };
    Doc { path: path.to_owned(), decoder: "pdf", encoding: "binary", title, text, blocks, notes, general: None }
}

/// The page locator of line `line` of a PDF's derived text: `p12`, or `p12-13` for a span.
#[must_use]
pub fn locator(blocks: &[Block], first: usize, last: usize) -> Option<String> {
    let page = |l: usize| nmtype::doc::block_at(blocks, l).and_then(|i| blocks.get(i)).and_then(|b| b.loc.page).map(|p| p.0);
    let a = page(first).or_else(|| (first..=last).find_map(page))?;
    let b = page(last).unwrap_or(a);
    Some(if b > a { format!("p{a}-{b}") } else { format!("p{a}") })
}

pub(crate) fn check() -> Result<(), String> {
    lex::check()?;
    crypt::check()?;
    filters::check()?;
    file::check()?;
    font::check()?;
    layout::check()?;
    structure::check()?;
    // A one-page file through every layer: a simple font, two lines, a heading by size.
    let content = b"BT /F1 18 Tf 72 720 Td (Pump care) Tj ET BT /F1 10 Tf 72 690 Td 12 TL (Clean the strainer every month and check) Tj T* (the seal for leaks.) Tj ET";
    let mut f = b"%PDF-1.4\n".to_vec();
    let objs: [Vec<u8>; 4] = [
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".to_vec(),
        [format!("<< /Length {} >>\nstream\n", content.len()).as_bytes(), content, b"\nendstream"].concat(),
    ];
    let mut offs = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        offs.push(f.len());
        f.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        f.extend_from_slice(o);
        f.extend_from_slice(b"\nendobj\n");
    }
    offs.push(f.len());
    f.extend_from_slice(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n");
    let x = f.len();
    f.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for o in &offs {
        f.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    f.extend_from_slice(format!("trailer << /Size 6 /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n").as_bytes());
    let d = read("care.pdf", &f).map_err(|e| format!("nmdoc.pdf: {e}"))?;
    let want = "Pump care\nClean the strainer every month and check the seal for leaks.\n";
    if d.text != want || d.blocks.first().map(|b| b.kind) != Some(BlockKind::Heading(1)) || locator(&d.blocks, 2, 2).as_deref() != Some("p1") {
        return Err(format!("nmdoc.pdf: derived {:?} blocks {:?}", d.text, d.blocks));
    }
    nmtype::doc::check_cover(&d.blocks, d.text.lines().count()).map_err(|e| format!("nmdoc.pdf: {e}"))?;
    // Truncated and corrupted copies never panic.
    for cut in [10, f.len() / 3, f.len() / 2, f.len() - 20] {
        let _ = read("cut.pdf", f.get(..cut).unwrap_or(&[]));
    }
    let mut bad = f.clone();
    for i in (40..bad.len()).step_by(37) {
        if let Some(b) = bad.get_mut(i) {
            *b ^= 0x5A;
        }
    }
    let _ = read("bad.pdf", &bad);
    Ok(())
}
