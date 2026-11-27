//! From placed glyphs to paragraphs in reading order. Two ways:
//! - by the tagged structure, when the page is tagged: glyphs in the order of the structure tree's
//!   marked content, one paragraph per block element (paragraph, heading, list item, table row
//!   with its cells), running headers and page numbers left out as artifacts;
//! - by geometry otherwise: glyphs joined into pieces of text on one baseline (a space where the
//!   gap is wider than a fraction of the font size), pieces into rows, a column gutter found as an
//!   empty vertical band with text on both sides, rows that cross the gutter (titles, wide tables)
//!   kept whole and splitting the page into bands, each band read left column then right column;
//!   lines then grouped into paragraphs by line spacing, font size, list marks and short last
//!   lines.
//!
//! Joining a paragraph's lines (hyphens at line ends) and removing repeated headers and footers
//! happen over the whole document (`super::assemble`).
use super::content::Glyph;
use std::collections::HashMap;

/// What a paragraph is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Text,
    Heading(u8),
    List,
    /// A table row, its cells separated by tabs; `head` for a header row.
    Row {
        head: bool,
    },
    Code,
}

/// One paragraph of one page.
#[derive(Clone, Debug)]
pub struct Para {
    /// Its lines as shown (joined later; code keeps them apart).
    pub lines: Vec<String>,
    pub kind: Kind,
    pub size: f64,
    pub bold: bool,
    pub top: f64,
    pub bottom: f64,
}

/// What the structure tree says about one marked content id of a page: its place in reading
/// order, its block element, the block's kind, and its table cell.
#[derive(Clone, Debug)]
pub struct Mark {
    pub seq: usize,
    pub block: usize,
    pub kind: Kind,
    pub cell: Option<usize>,
}

/// A run of glyphs on one baseline.
#[derive(Clone, Debug)]
struct Piece {
    text: String,
    x0: f64,
    x1: f64,
    y: f64,
    size: f64,
    bold: f64,
    mono: f64,
    chars: f64,
}

impl Piece {
    fn top(&self) -> f64 {
        self.y - self.size * 0.8
    }
}

fn blank(g: &Glyph) -> bool {
    g.text.chars().all(char::is_whitespace)
}

/// Add a glyph to a run of text: a space when the gap is wide or a space glyph came between.
fn push_glyph(p: &mut Piece, g: &Glyph, pending_space: bool) {
    let gap = g.x0 - p.x1;
    if !p.text.is_empty() && !p.text.ends_with(' ') && (pending_space || gap > 0.15 * g.size.min(p.size).max(1.0)) {
        p.text.push(' ');
    }
    p.text.push_str(&g.text);
    p.x1 = p.x1.max(g.x1);
    p.x0 = p.x0.min(g.x0);
    let n = g.text.chars().count() as f64;
    p.bold += if g.bold { n } else { 0.0 };
    p.mono += if g.mono { n } else { 0.0 };
    p.chars += n;
    p.size = p.size.max(g.size);
}

fn piece_of(g: &Glyph) -> Piece {
    let n = g.text.chars().count() as f64;
    Piece { text: g.text.clone(), x0: g.x0, x1: g.x1, y: g.y, size: g.size, bold: if g.bold { n } else { 0.0 }, mono: if g.mono { n } else { 0.0 }, chars: n }
}

/// True when glyph `g` continues piece `p` on the same baseline.
fn continues(p: &Piece, g: &Glyph, max_gap: f64) -> bool {
    let s = p.size.min(g.size).max(1.0);
    (g.y - p.y).abs() <= 0.35 * s && g.x0 >= p.x1 - 0.6 * s && g.x0 - p.x1 <= max_gap * s
}

/// Pieces from glyphs in content order, then merged where two pieces on one row touch.
fn pieces(glyphs: &[&Glyph]) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::new();
    let mut pending = false;
    for g in glyphs {
        if blank(g) {
            pending = true;
            continue;
        }
        match out.last_mut() {
            Some(p) if continues(p, g, 0.8) => push_glyph(p, g, pending),
            _ => out.push(piece_of(g)),
        }
        pending = false;
    }
    // Merge pieces of one row that touch, whatever order they were drawn in.
    out.sort_by(|a, b| a.y.total_cmp(&b.y));
    let mut rows: Vec<Vec<Piece>> = Vec::new();
    for p in out {
        match rows.last_mut() {
            Some(r) if r.first().is_some_and(|f| (f.y - p.y).abs() <= 0.35 * f.size.min(p.size).max(1.0)) => r.push(p),
            _ => rows.push(vec![p]),
        }
    }
    let mut merged = Vec::new();
    for mut r in rows {
        r.sort_by(|a, b| a.x0.total_cmp(&b.x0));
        let mut cur: Option<Piece> = None;
        for p in r {
            cur = match cur {
                Some(mut c) if p.x0 - c.x1 <= 0.8 * c.size.min(p.size).max(1.0) && p.x0 >= c.x1 - 0.6 * c.size => {
                    if !c.text.ends_with(' ') && p.x0 - c.x1 > 0.15 * c.size.min(p.size) {
                        c.text.push(' ');
                    }
                    c.text.push_str(&p.text);
                    c.x1 = c.x1.max(p.x1);
                    c.bold += p.bold;
                    c.mono += p.mono;
                    c.chars += p.chars;
                    c.size = c.size.max(p.size);
                    Some(c)
                }
                Some(c) => {
                    merged.push(c);
                    Some(p)
                }
                None => Some(p),
            };
        }
        if let Some(c) = cur {
            merged.push(c);
        }
    }
    merged
}

/// One line in reading order: its pieces (more than one in a table row), its column.
#[derive(Clone, Debug)]
struct Line {
    parts: Vec<Piece>,
    col: u8,
}

impl Line {
    fn x0(&self) -> f64 {
        self.parts.iter().map(|p| p.x0).fold(f64::INFINITY, f64::min)
    }
    fn x1(&self) -> f64 {
        self.parts.iter().map(|p| p.x1).fold(f64::NEG_INFINITY, f64::max)
    }
    fn y(&self) -> f64 {
        self.parts.first().map_or(0.0, |p| p.y)
    }
    fn size(&self) -> f64 {
        self.parts.iter().map(|p| p.size).fold(0.0, f64::max)
    }
    fn chars(&self) -> f64 {
        self.parts.iter().map(|p| p.chars).sum()
    }
    fn bold(&self) -> bool {
        self.parts.iter().map(|p| p.bold).sum::<f64>() * 2.0 > self.chars()
    }
    fn mono(&self) -> bool {
        self.parts.iter().map(|p| p.mono).sum::<f64>() * 2.0 > self.chars()
    }
    fn text(&self) -> String {
        self.parts.iter().map(|p| p.text.trim()).collect::<Vec<_>>().join("\t")
    }
}

/// Rows of pieces (one baseline each), top first, pieces left to right.
fn rows_of(mut ps: Vec<Piece>) -> Vec<Vec<Piece>> {
    ps.sort_by(|a, b| a.y.total_cmp(&b.y));
    let mut rows: Vec<Vec<Piece>> = Vec::new();
    for p in ps {
        match rows.last_mut() {
            Some(r) if r.first().is_some_and(|f| (f.y - p.y).abs() <= 0.35 * f.size.min(p.size).max(1.0)) => r.push(p),
            _ => rows.push(vec![p]),
        }
    }
    for r in &mut rows {
        r.sort_by(|a, b| a.x0.total_cmp(&b.x0));
    }
    rows
}

/// The column gutter of a page: an x band in the middle half, covered by few rows, with rows of
/// text wholly on each side of it.
fn gutter(rows: &[Vec<Piece>]) -> Option<(f64, f64)> {
    let all: Vec<&Piece> = rows.iter().flatten().collect();
    if rows.len() < 8 {
        return None;
    }
    let left = all.iter().map(|p| p.x0).fold(f64::INFINITY, f64::min);
    let right = all.iter().map(|p| p.x1).fold(f64::NEG_INFINITY, f64::max);
    let w = right - left;
    let body = median(all.iter().map(|p| p.size).collect()).max(4.0);
    if w < body * 20.0 {
        return None;
    }
    // Rows of three or more pieces are table rows: they do not count against a gutter.
    let cover = |x: f64| rows.iter().filter(|r| r.len() < 3 && r.iter().any(|p| p.x0 < x && p.x1 > x)).count();
    let limit = (rows.len() as f64 * 0.12).max(1.0) as usize;
    let need = ((rows.len() as f64 * 0.2) as usize).max(4);
    // Coverage at each point of the middle half, one page unit apart.
    let from = left + w * 0.25;
    let n = (w * 0.5) as usize + 1;
    let cov: Vec<(f64, usize)> = (0..n).map(|i| from + i as f64).map(|x| (x, cover(x))).collect();
    // Candidate bands: runs covered by few rows, at least most of a font size wide; each narrowed
    // to its emptiest core (ragged line ends reach into the band's edges).
    let mut cores: Vec<(f64, f64, usize)> = Vec::new();
    let mut i = 0;
    while i < cov.len() {
        if cov.get(i).is_none_or(|c| c.1 > limit) {
            i += 1;
            continue;
        }
        let mut j = i;
        while cov.get(j + 1).is_some_and(|c| c.1 <= limit) {
            j += 1;
        }
        let band = cov.get(i..=j).unwrap_or(&[]);
        let lo = band.iter().map(|c| c.1).min().unwrap_or(0);
        let mut best: Option<(f64, f64)> = None;
        let mut k = 0;
        while k < band.len() {
            if band.get(k).is_some_and(|c| c.1 == lo) {
                let mut e = k;
                while band.get(e + 1).is_some_and(|c| c.1 == lo) {
                    e += 1;
                }
                let (x0, x1) = (band.get(k).map_or(0.0, |c| c.0), band.get(e).map_or(0.0, |c| c.0));
                if best.is_none_or(|(a, b)| x1 - x0 > b - a) {
                    best = Some((x0, x1));
                }
                k = e + 1;
            } else {
                k += 1;
            }
        }
        if let Some((x0, x1)) = best
            && x1 - x0 >= body * 0.8
        {
            cores.push((x0, x1, lo));
        }
        i = j + 1;
    }
    // A gutter has rows of text wholly on each side; the emptiest such core, then the widest.
    cores
        .into_iter()
        .filter(|(gl, gr, _)| {
            let lefts = rows.iter().filter(|r| r.iter().any(|p| p.x1 <= *gl)).count();
            let rights = rows.iter().filter(|r| r.iter().any(|p| p.x0 >= *gr)).count();
            lefts >= need && rights >= need
        })
        .min_by(|a, b| a.2.cmp(&b.2).then((b.1 - b.0).total_cmp(&(a.1 - a.0))))
        .map(|(gl, gr, _)| (gl, gr))
}

/// True when both pieces of a two-piece row lie on one side of `mid` (two cells of a table, not
/// a line of each column).
fn sides_of(r: &[Piece], mid: f64) -> bool {
    r.iter().all(|p| p.x1 <= mid) || r.iter().all(|p| p.x0 >= mid)
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// Lines in reading order by geometry.
fn read_order(ps: Vec<Piece>) -> Vec<Line> {
    let rows = rows_of(ps);
    let Some((gl, gr)) = gutter(&rows) else {
        return rows.into_iter().map(|parts| Line { parts, col: 0 }).collect();
    };
    let crosses = |r: &Vec<Piece>| r.iter().any(|p| p.x0 < gr && p.x1 > gl);
    let mut span: Vec<bool> = rows.iter().map(crosses).collect();
    // Rows of several pieces between two crossing rows close together (within about three lines)
    // belong to the same wide element, such as a table whose cells do not all reach across the
    // gutter.
    let body = median(rows.iter().flatten().map(|p| p.size).collect()).max(4.0);
    let ys: Vec<f64> = rows.iter().map(|r| r.first().map_or(0.0, |p| p.y)).collect();
    let idx: Vec<usize> = (0..rows.len()).filter(|i| span.get(*i) == Some(&true)).collect();
    for w in idx.windows(2) {
        let (a, b) = (w[0], w[1]);
        if b > a + 1 && ys.get(b).copied().unwrap_or(0.0) - ys.get(a).copied().unwrap_or(0.0) <= body * 3.6 {
            for (k, s) in span.iter_mut().enumerate().take(b).skip(a + 1) {
                *s = *s || rows.get(k).is_some_and(|r| r.len() >= 2);
            }
        }
    }
    // A row set larger than the body text, in a monospaced font, or of several cells, on one
    // side, with no text beside it on the other side above or below within a line and a half,
    // stands alone across the page (a short title, a code block or a narrow table over both
    // columns).
    let mid = (gl + gr) / 2.0;
    let side = |r: &Vec<Piece>| -> Option<bool> {
        let l = r.iter().all(|p| p.x1 <= mid);
        let rr = r.iter().all(|p| p.x0 >= mid);
        if l { Some(true) } else if rr { Some(false) } else { None }
    };
    let sides: Vec<Option<bool>> = rows.iter().map(side).collect();
    let alone: Vec<bool> = (0..rows.len())
        .map(|i| {
            let (Some(Some(me)), Some(&y)) = (sides.get(i), ys.get(i)) else { return false };
            let size = rows.get(i).map_or(0.0, |r| r.iter().map(|p| p.size).fold(0.0, f64::max));
            let mono = rows.get(i).is_some_and(|r| r.iter().all(|p| p.mono * 2.0 > p.chars));
            let table = rows.get(i).is_some_and(|r| r.len() >= 2);
            if span.get(i) == Some(&true) || (size < body * 1.15 && !mono && !table) {
                return false;
            }
            let others: Vec<f64> = (0..rows.len())
                .filter(|j| *j != i && span.get(*j) != Some(&true) && sides.get(*j) == Some(&Some(!*me)))
                .filter_map(|j| ys.get(j).copied())
                .collect();
            let above = others.iter().filter(|o| **o <= y).map(|o| y - o).fold(f64::INFINITY, f64::min);
            let below = others.iter().filter(|o| **o > y).map(|o| o - y).fold(f64::INFINITY, f64::min);
            (above.is_finite() || below.is_finite()) && above > body * 1.6 && below > body * 1.6
        })
        .collect();
    for (s, a) in span.iter_mut().zip(alone) {
        *s = *s || a;
    }
    // A wide table: rows of several cells next to a crossing row (within a line and a half) cross
    // with it, up and down the table.
    let cells = |r: &Vec<Piece>| r.len() >= 3 || (r.len() == 2 && sides_of(r, mid));
    for _ in 0..rows.len() {
        let mut changed = false;
        for i in 0..rows.len() {
            if span.get(i) == Some(&true) || !rows.get(i).is_some_and(cells) {
                continue;
            }
            let y = ys.get(i).copied().unwrap_or(0.0);
            let near = |j: usize| span.get(j) == Some(&true) && (ys.get(j).copied().unwrap_or(f64::MAX) - y).abs() <= body * 1.6;
            if (i > 0 && near(i - 1)) || near(i + 1) {
                if let Some(s) = span.get_mut(i) {
                    *s = true;
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut out = Vec::new();
    let (mut left, mut right): (Vec<Line>, Vec<Line>) = (Vec::new(), Vec::new());
    for (r, wide) in rows.into_iter().zip(span) {
        if wide {
            out.append(&mut left);
            out.append(&mut right);
            out.push(Line { parts: r, col: 3 });
            continue;
        }
        let (l, rr): (Vec<Piece>, Vec<Piece>) = r.into_iter().partition(|p| (p.x0 + p.x1) / 2.0 < (gl + gr) / 2.0);
        if !l.is_empty() {
            left.push(Line { parts: l, col: 1 });
        }
        if !rr.is_empty() {
            right.push(Line { parts: rr, col: 2 });
        }
    }
    out.append(&mut left);
    out.append(&mut right);
    out
}

/// True when a line starts with a list mark: a bullet, or a number or letter and `.` or `)`.
#[must_use]
pub fn list_mark(t: &str) -> bool {
    let t = t.trim_start();
    if t.starts_with(['\u{2022}', '\u{25e6}', '\u{25aa}', '\u{2023}', '\u{2043}', '\u{25cf}', '\u{25a0}', '\u{2013}']) {
        return true;
    }
    let head: String = t.chars().take_while(char::is_ascii_alphanumeric).collect();
    let rest = t.get(head.len()..).unwrap_or("");
    !head.is_empty() && head.len() <= 3 && (head.bytes().all(|b| b.is_ascii_digit()) || head.len() == 1) && (rest.starts_with(". ") || rest.starts_with(") "))
}

/// Paragraphs from lines in reading order.
fn paragraphs(lines: Vec<Line>) -> Vec<Para> {
    let pitch_of = |size: f64| -> f64 {
        let d: Vec<f64> = lines
            .windows(2)
            .filter(|w| w[0].col == w[1].col && (w[0].size() - size).abs() < 0.5 && (w[1].size() - size).abs() < 0.5)
            .map(|w| w[1].y() - w[0].y())
            .filter(|d| *d > 0.0 && *d < size * 3.0)
            .collect();
        let m = median(d);
        if m > 0.0 { m } else { size * 1.2 }
    };
    let mut pitches: HashMap<i64, f64> = HashMap::new();
    let mut col_right: HashMap<u8, f64> = HashMap::new();
    for l in &lines {
        let e = col_right.entry(l.col).or_insert(f64::NEG_INFINITY);
        *e = e.max(l.x1());
    }
    let mut col_left: HashMap<u8, f64> = HashMap::new();
    for l in &lines {
        let e = col_left.entry(l.col).or_insert(f64::INFINITY);
        *e = e.min(l.x0());
    }
    let mut out: Vec<Para> = Vec::new();
    let mut prev: Option<&Line> = None;
    for l in &lines {
        let text = l.text();
        let size = l.size();
        let key = (size * 2.0).round() as i64;
        let pitch = *pitches.entry(key).or_insert_with(|| pitch_of(size));
        let row = l.parts.len() > 1;
        let new = match prev {
            None => true,
            Some(p) => {
                let dy = l.y() - p.y();
                let width = col_right.get(&p.col).copied().unwrap_or(0.0) - col_left.get(&p.col).copied().unwrap_or(0.0);
                let short = p.x1() < col_right.get(&p.col).copied().unwrap_or(0.0) - width * 0.12 && p.text().trim_end().ends_with(['.', ':', '!', '?']);
                p.col != l.col
                    || dy <= 0.0
                    || dy > pitch * 1.25
                    || (size - p.size()).abs() > size * 0.12
                    || p.bold() != l.bold()
                    || p.mono() != l.mono()
                    || list_mark(&text)
                    || row
                    || p.parts.len() > 1
                    || short
            }
        };
        let kind = if row {
            Kind::Row { head: false }
        } else if l.mono() {
            Kind::Code
        } else if list_mark(&text) {
            Kind::List
        } else {
            Kind::Text
        };
        if new || out.is_empty() {
            out.push(Para { lines: vec![text], kind, size, bold: l.bold(), top: l.y() - size * 0.8, bottom: l.y() + size * 0.2 });
        } else if let Some(last) = out.last_mut() {
            last.lines.push(text);
            last.bottom = l.y() + size * 0.2;
        }
        prev = Some(l);
    }
    out
}

/// Paragraphs of a page by geometry. Glyphs whose baseline is not horizontal come last, each
/// run as its own paragraph.
#[must_use]
pub fn by_geometry(glyphs: &[Glyph]) -> Vec<Para> {
    let upright: Vec<&Glyph> = glyphs.iter().filter(|g| g.upright).collect();
    let ps = pieces(&upright);
    let mut out = paragraphs(read_order(ps));
    let mut other = String::new();
    for g in glyphs.iter().filter(|g| !g.upright) {
        other.push_str(&g.text);
    }
    let other = other.split_whitespace().collect::<Vec<_>>().join(" ");
    if !other.is_empty() {
        out.push(Para { lines: vec![other], kind: Kind::Text, size: 0.0, bold: false, top: 0.0, bottom: 0.0 });
    }
    out
}

/// Paragraphs of a page by its tagged structure: glyphs in structure order, a paragraph per
/// block element, cells of a row separated by tabs. Glyphs not in the structure (artifacts) are
/// left out.
#[must_use]
pub fn by_tags(glyphs: &[Glyph], marks: &HashMap<u32, Mark>) -> Vec<Para> {
    let mut tagged: Vec<(usize, usize, &Glyph, &Mark)> =
        glyphs.iter().enumerate().filter(|(_, g)| !g.artifact).filter_map(|(i, g)| g.mcid.and_then(|m| marks.get(&m)).map(|m| (m.seq, i, g, m))).collect();
    tagged.sort_by_key(|(s, i, _, _)| (*s, *i));
    let mut out: Vec<Para> = Vec::new();
    let mut cur_block: Option<usize> = None;
    let mut cur_cell: Option<usize> = None;
    let mut piece: Option<Piece> = None;
    let mut pending = false;
    let mut lines: Vec<String> = Vec::new();
    // Per paragraph: glyphs in a monospaced font, and all glyphs.
    let mut mono_share: Vec<(usize, usize)> = Vec::new();
    let flush_piece = |piece: &mut Option<Piece>, lines: &mut Vec<String>, para: Option<&mut Para>| {
        if let Some(p) = piece.take() {
            if let Some(para) = para {
                para.top = para.top.min(p.top());
                para.bottom = para.bottom.max(p.y + p.size * 0.2);
                para.size = para.size.max(p.size);
                para.bold = para.bold || p.bold * 2.0 > p.chars;
            }
            lines.push(p.text.trim().to_owned());
        }
    };
    for (_, _, g, m) in tagged {
        if cur_block != Some(m.block) {
            flush_piece(&mut piece, &mut lines, out.last_mut());
            if let Some(last) = out.last_mut() {
                last.lines = std::mem::take(&mut lines).into_iter().filter(|l| !l.is_empty()).collect();
            }
            out.push(Para { lines: Vec::new(), kind: m.kind, size: 0.0, bold: false, top: f64::INFINITY, bottom: f64::NEG_INFINITY });
            mono_share.push((0, 0));
            cur_block = Some(m.block);
            cur_cell = m.cell;
            pending = false;
        }
        if blank(g) {
            pending = true;
            continue;
        }
        if let Some(ms) = mono_share.last_mut() {
            ms.0 += usize::from(g.mono);
            ms.1 += 1;
        }
        let new_cell = m.cell.is_some() && m.cell != cur_cell;
        cur_cell = m.cell;
        match piece.as_mut() {
            Some(p) if new_cell => {
                p.text = p.text.trim_end().to_owned();
                p.text.push('\t');
                p.text.push_str(&g.text);
                p.x1 = g.x1;
            }
            Some(p) if continues(p, g, 6.0) => push_glyph(p, g, pending),
            _ => {
                flush_piece(&mut piece, &mut lines, out.last_mut());
                piece = Some(piece_of(g));
            }
        }
        pending = false;
    }
    flush_piece(&mut piece, &mut lines, out.last_mut());
    if let Some(last) = out.last_mut() {
        last.lines = lines.into_iter().filter(|l| !l.is_empty()).collect();
    }
    let monos: Vec<bool> = out.iter().zip(&mono_share).filter(|(p, _)| !p.lines.is_empty()).map(|(_, m)| m.0 * 2 > m.1).collect();
    out.retain(|p| !p.lines.is_empty());
    for (p, mono) in out.iter_mut().zip(monos) {
        if p.kind == Kind::Text && mono {
            p.kind = Kind::Code;
        }
        if p.kind == Kind::Text && p.lines.first().is_some_and(|l| list_mark(l)) {
            p.kind = Kind::List;
        }
    }
    out
}

pub(crate) fn check() -> Result<(), String> {
    let g = |t: &str, x0: f64, y: f64| Glyph {
        text: t.to_owned(),
        x0,
        x1: x0 + 5.0 * t.chars().count() as f64,
        y,
        size: 10.0,
        bold: false,
        mono: false,
        mcid: None,
        artifact: false,
        unmapped: false,
        upright: true,
    };
    // Two columns of eight lines each, drawn right column first; a wide title on top.
    let mut gl = vec![g("A wide title that crosses the middle of the page", 50.0, 40.0)];
    for i in 0..8 {
        gl.push(g(&format!("right {i}"), 320.0, 80.0 + 12.0 * f64::from(i)));
    }
    for i in 0..8 {
        gl.push(g(&format!("left {i}"), 50.0, 80.0 + 12.0 * f64::from(i)));
    }
    gl.push(g("x", 500.0, 80.0));
    let paras = by_geometry(&gl);
    let text: Vec<String> = paras.iter().map(|p| p.lines.join(" ")).collect();
    let joined = text.join(" | ");
    let l7 = joined.find("left 7").unwrap_or(usize::MAX);
    let r0 = joined.find("right 0").unwrap_or(0);
    if !joined.starts_with("A wide title") || l7 > r0 {
        return Err(format!("nmdoc.pdf.layout: two-column order {joined}"));
    }
    if !list_mark("\u{2022} item") || !list_mark("12. Step") || list_mark("1.5 bar") {
        return Err("nmdoc.pdf.layout: list marks".into());
    }
    Ok(())
}
