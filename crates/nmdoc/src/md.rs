//! Structure of Markdown and plain-text documents: blocks over the file's own lines.
//! Markdown: front matter, ATX and setext headings, fenced and indented code, block quotes (a
//! quote that opens with a bold Warning, Note, Caution or Important, or a `[!NOTE]` marker, is a
//! note box), list items, pipe tables, captions, bold-term definitions, paragraphs. Plain text:
//! headings underlined with `=`, `-`, `~` or `^`, numbered headings (`2.3 Measures of spread`),
//! list items and paragraphs. Blank lines join the block before them.
use nmtype::doc::{Block, BlockKind};

/// Leading spaces (a tab counts four).
fn indent(l: &str) -> usize {
    let mut n = 0;
    for c in l.chars() {
        match c {
            ' ' => n += 1,
            '\t' => n += 4,
            _ => break,
        }
    }
    n
}

/// An ATX heading: level and text.
fn atx(l: &str) -> Option<(u8, String)> {
    if indent(l) > 3 {
        return None;
    }
    let t = l.trim_start();
    let level = t.chars().take_while(|c| *c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = t.get(level..).unwrap_or("");
    if !(rest.is_empty() || rest.starts_with(' ') || rest.starts_with('\t')) {
        return None;
    }
    let mut text = rest.trim();
    // A closing run of #s preceded by a space is not part of the text.
    let stripped = text.trim_end_matches('#');
    if stripped.len() < text.len() && (stripped.is_empty() || stripped.ends_with(' ')) {
        text = stripped.trim_end();
    }
    Some((u8::try_from(level).unwrap_or(6), text.to_owned()))
}

/// A setext underline: 1 for `===`, 2 for `---`.
fn setext(l: &str) -> Option<u8> {
    if indent(l) > 3 {
        return None;
    }
    let t = l.trim();
    if t.is_empty() {
        return None;
    }
    if t.chars().all(|c| c == '=') {
        Some(1)
    } else if t.chars().all(|c| c == '-') {
        Some(2)
    } else {
        None
    }
}

fn fence(l: &str) -> Option<(char, usize)> {
    if indent(l) > 3 {
        return None;
    }
    let t = l.trim_start();
    let c = t.chars().next()?;
    if c != '`' && c != '~' {
        return None;
    }
    let n = t.chars().take_while(|x| *x == c).count();
    (n >= 3 && !(c == '`' && t.get(n..).is_some_and(|r| r.contains('`')))).then_some((c, n))
}

fn thematic(l: &str) -> bool {
    let t: String = l.chars().filter(|c| !c.is_whitespace()).collect();
    indent(l) <= 3 && t.len() >= 3 && (t.chars().all(|c| c == '-') || t.chars().all(|c| c == '*') || t.chars().all(|c| c == '_'))
}

/// A list item marker: (depth by indentation, ordered, content column).
fn list_item(l: &str) -> Option<(u8, bool, usize)> {
    let ind = indent(l);
    let t = l.trim_start();
    let (ordered, len) = if let Some(r) = t.strip_prefix(['-', '*', '+', '\u{2022}']) {
        (false, t.len() - r.len())
    } else {
        let digits = t.chars().take_while(char::is_ascii_digit).count();
        if digits == 0 || digits > 9 {
            return None;
        }
        let after = t.get(digits..).unwrap_or("");
        if !(after.starts_with('.') || after.starts_with(')')) {
            return None;
        }
        (true, digits + 1)
    };
    let rest = t.get(len..).unwrap_or("");
    if !(rest.starts_with(' ') || rest.starts_with('\t')) || rest.trim().is_empty() && !ordered {
        return None;
    }
    let depth = u8::try_from((ind / 2).min(8)).unwrap_or(8);
    Some((depth, ordered, ind + len + 1))
}

fn table_sep(l: &str) -> bool {
    let t = l.trim();
    t.contains('-') && t.chars().all(|c| matches!(c, '|' | '-' | ':' | ' ' | '\t')) && (t.contains('|') || t.starts_with(':'))
}

fn caption(l: &str) -> bool {
    let t = l.trim_start().trim_start_matches(['*', '_']);
    ["Figure ", "Fig. ", "Table ", "Chart ", "Diagram "]
        .iter()
        .any(|p| t.strip_prefix(p).is_some_and(|r| r.chars().next().is_some_and(|c| c.is_ascii_digit())))
}

fn definition(l: &str) -> bool {
    let t = l.trim_start();
    let Some(rest) = t.strip_prefix("**") else { return false };
    let Some(end) = rest.find("**") else { return false };
    let after = rest.get(end + 2..).unwrap_or("").trim_start();
    end > 0 && end < 60 && (after.starts_with(':') || after.starts_with(" is") || after.starts_with("is ") || after.starts_with("(") || after.starts_with('-'))
}

fn note_box(l: &str) -> bool {
    let t = l.trim_start().trim_start_matches('>').trim_start().to_lowercase();
    ["**warning", "**note", "**caution", "**important", "**danger", "**tip", "[!note]", "[!warning]", "[!caution]", "[!important]", "[!tip]"]
        .iter()
        .any(|p| t.starts_with(p))
}

/// Give every gap line (blank lines, thematic breaks) to the block before it, the leading ones to
/// the first block, so every line is in exactly one block.
pub fn close_gaps(blocks: &mut Vec<Block>, n: usize) {
    if n == 0 {
        blocks.clear();
        return;
    }
    if blocks.is_empty() {
        blocks.push(Block::new(BlockKind::Paragraph, 1, n, ""));
        return;
    }
    blocks.sort_by_key(|b| b.first);
    if let Some(f) = blocks.first_mut() {
        f.first = 1;
    }
    for i in 0..blocks.len() {
        let next = blocks.get(i + 1).map_or(n + 1, |b| b.first);
        if let Some(b) = blocks.get_mut(i) {
            b.last = next - 1;
        }
    }
}

/// Blocks of a Markdown document over its own lines.
#[must_use]
pub fn markdown(text: &str) -> Vec<Block> {
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    let line = |i: usize| lines.get(i).copied().unwrap_or("");
    let blank = |i: usize| line(i).trim().is_empty();
    let mut blocks: Vec<Block> = Vec::new();
    let mut i = 0;
    // Front matter: `---` on the first line closed by `---` or `...` within 80 lines.
    if line(0).trim_end() == "---"
        && let Some(end) = (1..n.min(80)).find(|j| matches!(line(*j).trim_end(), "---" | "..."))
    {
        blocks.push(Block::new(BlockKind::FrontMatter, 1, end + 1, ""));
        i = end + 1;
    }
    while i < n {
        let l = line(i);
        if blank(i) {
            i += 1;
            continue;
        }
        if let Some((c, len)) = fence(l) {
            let mut j = i + 1;
            while j < n && !fence(line(j)).is_some_and(|(c2, l2)| c2 == c && l2 >= len && line(j).trim().chars().all(|x| x == c)) {
                j += 1;
            }
            let end = j.min(n - 1);
            blocks.push(Block::new(BlockKind::Code, i + 1, end + 1, ""));
            i = end + 1;
            continue;
        }
        if let Some((level, t)) = atx(l) {
            blocks.push(Block::new(BlockKind::Heading(level), i + 1, i + 1, &t));
            i += 1;
            continue;
        }
        if thematic(l) {
            i += 1;
            continue;
        }
        if l.trim_start().starts_with('>') && indent(l) <= 3 {
            let mut j = i + 1;
            while j < n && !blank(j) && (line(j).trim_start().starts_with('>') || !starts_block(line(j))) {
                j += 1;
            }
            let kind = if note_box(l) { BlockKind::Note } else { BlockKind::Quote };
            blocks.push(Block::new(kind, i + 1, j, ""));
            i = j;
            continue;
        }
        if let Some((depth, ordered, content)) = list_item(l) {
            // The item runs to the next item, a blank line followed by a less indented line, or a
            // new block.
            let mut j = i + 1;
            while j < n {
                let lj = line(j);
                if blank(j) {
                    let k = (j..n).find(|k| !blank(*k)).unwrap_or(n);
                    if k < n && indent(line(k)) >= content && list_item(line(k)).is_none() {
                        j = k + 1;
                        continue;
                    }
                    break;
                }
                if list_item(lj).is_some() || atx(lj).is_some() || fence(lj).is_some() || (indent(lj) < content && starts_block(lj)) {
                    break;
                }
                j += 1;
            }
            blocks.push(Block::new(BlockKind::ListItem { depth, ordered }, i + 1, j, ""));
            i = j;
            continue;
        }
        if l.contains('|') && i + 1 < n && table_sep(line(i + 1)) {
            let mut j = i + 2;
            while j < n && !blank(j) && line(j).contains('|') {
                j += 1;
            }
            blocks.push(Block::new(BlockKind::Table { head: 2 }, i + 1, j, ""));
            i = j;
            continue;
        }
        if indent(l) >= 4 && (i == 0 || blank(i - 1)) && !blocks.last().is_some_and(|b| matches!(b.kind, BlockKind::ListItem { .. })) {
            let mut j = i + 1;
            while j < n && (indent(line(j)) >= 4 || blank(j)) {
                j += 1;
            }
            while j > i + 1 && blank(j - 1) {
                j -= 1;
            }
            blocks.push(Block::new(BlockKind::Code, i + 1, j, ""));
            i = j;
            continue;
        }
        // A paragraph: lines up to a blank line or a new block; a setext underline makes it a heading.
        let mut j = i + 1;
        let mut heading = None;
        while j < n && !blank(j) {
            if let Some(level) = setext(line(j)) {
                heading = Some(level);
                break;
            }
            if starts_block(line(j)) {
                break;
            }
            j += 1;
        }
        if let Some(level) = heading {
            let t: Vec<&str> = (i..j).map(|k| line(k).trim()).collect();
            blocks.push(Block::new(BlockKind::Heading(level), i + 1, j + 1, &t.join(" ")));
            i = j + 1;
            continue;
        }
        let kind = if caption(l) {
            BlockKind::Caption
        } else if definition(l) {
            BlockKind::Definition
        } else {
            BlockKind::Paragraph
        };
        blocks.push(Block::new(kind, i + 1, j, ""));
        i = j;
    }
    close_gaps(&mut blocks, n);
    nmtype::doc::link_parents(&mut blocks);
    blocks
}

/// A line that opens a new block inside running text.
fn starts_block(l: &str) -> bool {
    atx(l).is_some() || fence(l).is_some() || (l.trim_start().starts_with('>') && indent(l) <= 3) || list_item(l).is_some() || thematic(l)
}

/// A numbered heading of plain text: level and text.
fn numbered(l: &str) -> Option<(u8, String)> {
    let t = l.trim();
    if t.len() >= 80 || indent(l) > 0 || t.ends_with('.') || t.ends_with(':') || t.ends_with(',') {
        return None;
    }
    let num: String = t.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let rest = t.get(num.len()..).unwrap_or("");
    if num.is_empty() || !rest.starts_with(' ') || num.starts_with('.') || num.contains("..") {
        return None;
    }
    let parts: Vec<&str> = num.trim_end_matches('.').split('.').collect();
    // `2. Fill the tank` is a list item; `2 Introduction` and `2.3 Spread` are headings.
    if parts.len() == 1 && num.ends_with('.') {
        return None;
    }
    let title = rest.trim();
    if !title.chars().next().is_some_and(char::is_uppercase) || title.split_whitespace().count() > 12 {
        return None;
    }
    Some((u8::try_from(parts.len().min(6)).unwrap_or(6), t.to_owned()))
}

/// An underline of plain text and its level.
fn underline(l: &str) -> Option<u8> {
    let t = l.trim_end();
    if t.len() < 3 || indent(l) > 0 {
        return None;
    }
    let c = t.chars().next()?;
    if !t.chars().all(|x| x == c) {
        return None;
    }
    match c {
        '=' => Some(1),
        '-' => Some(2),
        '~' => Some(3),
        '^' => Some(4),
        _ => None,
    }
}

/// Blocks of a plain-text document over its own lines.
#[must_use]
pub fn plain(text: &str) -> Vec<Block> {
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    let line = |i: usize| lines.get(i).copied().unwrap_or("");
    let blank = |i: usize| line(i).trim().is_empty();
    let mut blocks: Vec<Block> = Vec::new();
    let mut i = 0;
    while i < n {
        if blank(i) {
            i += 1;
            continue;
        }
        let l = line(i);
        if i + 1 < n && !blank(i) && underline(line(i + 1)).is_some() && l.trim().len() < 100 && underline(l).is_none() {
            let level = underline(line(i + 1)).unwrap_or(1);
            blocks.push(Block::new(BlockKind::Heading(level), i + 1, i + 2, l.trim()));
            i += 2;
            continue;
        }
        if (i == 0 || blank(i - 1))
            && let Some((level, t)) = numbered(l)
        {
            blocks.push(Block::new(BlockKind::Heading(level), i + 1, i + 1, &t));
            i += 1;
            continue;
        }
        if let Some((depth, ordered, content)) = list_item(l) {
            let mut j = i + 1;
            while j < n && !blank(j) && list_item(line(j)).is_none() && indent(line(j)) >= content.min(2) {
                j += 1;
            }
            blocks.push(Block::new(BlockKind::ListItem { depth, ordered }, i + 1, j, ""));
            i = j;
            continue;
        }
        let mut j = i + 1;
        while j < n && !blank(j) && list_item(line(j)).is_none() && !(j + 1 < n && underline(line(j + 1)).is_some()) {
            j += 1;
        }
        let kind = if caption(l) { BlockKind::Caption } else { BlockKind::Paragraph };
        blocks.push(Block::new(kind, i + 1, j, ""));
        i = j;
    }
    close_gaps(&mut blocks, n);
    nmtype::doc::link_parents(&mut blocks);
    blocks
}

/// The headings of a block list: (line, level, text).
#[must_use]
pub fn outline(blocks: &[Block]) -> Vec<(usize, u8, String)> {
    blocks.iter().filter_map(|b| b.kind.level().map(|l| (b.first, l, b.label.clone()))).collect()
}

pub(crate) fn check() -> Result<(), String> {
    let md = "---\ndate: 2026-09-14\n---\n# Pump\n\nIntro text.\n\n## 3.2 Descaling ##\n\n1. Stop the pump.\n2. Fill the tank.\n   more\n\n```sh\n# not a heading\n```\n\n> **Warning:** hot.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nSetext\n------\nFigure 1.2: the pump\n";
    let b = markdown(md);
    nmtype::doc::check_cover(&b, md.lines().count()).map_err(|e| format!("nmdoc.md: {e}"))?;
    let got = outline(&b);
    let want = vec![(4, 1, "Pump".to_owned()), (8, 2, "3.2 Descaling".to_owned()), (24, 2, "Setext".to_owned())];
    if got != want {
        return Err(format!("nmdoc.md: markdown outline {got:?}"));
    }
    let kinds: Vec<&str> = b.iter().map(|x| x.kind.name()).collect();
    let want_kinds =
        ["front-matter", "heading", "paragraph", "heading", "list", "list", "code", "note", "table", "heading", "caption"];
    if kinds != want_kinds {
        return Err(format!("nmdoc.md: markdown kinds {kinds:?}"));
    }
    if nmtype::doc::heading_path(&b, 4) != ["Pump", "3.2 Descaling"] {
        return Err(format!("nmdoc.md: heading path {:?}", nmtype::doc::heading_path(&b, 4)));
    }
    let txt = "Simple statistics\n=================\n\nIntro.\n\n2.3 Measures of spread\n\nThe range is wide.\n1. Sort the values.\n2. Count them\n\n2 Introduction\nText\n";
    let b = plain(txt);
    nmtype::doc::check_cover(&b, txt.lines().count()).map_err(|e| format!("nmdoc.md: {e}"))?;
    let got = outline(&b);
    let want = vec![(1, 1, "Simple statistics".to_owned()), (6, 2, "2.3 Measures of spread".to_owned()), (12, 1, "2 Introduction".to_owned())];
    if got != want {
        return Err(format!("nmdoc.md: text outline {got:?}"));
    }
    Ok(())
}
