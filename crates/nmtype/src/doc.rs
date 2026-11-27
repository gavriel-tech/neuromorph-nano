//! The document model: one shape that every decoder produces and that code files map into.
//! A document is its derived text (plain lines; for a text file the file itself, so line numbers
//! are the file's own) and its blocks in document order. Every line of the derived text is in
//! exactly one block; headings form the section tree through `parent`.

/// What a block holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BlockKind {
    /// A heading of the given level (1 is the top).
    Heading(u8),
    Paragraph,
    /// One list item; `depth` 0 is the outer list.
    ListItem {
        depth: u8,
        ordered: bool,
    },
    /// A table: the first `head` lines are its header (a CSV header row, or a Markdown header and
    /// its separator line), the rest are rows.
    Table {
        head: u16,
    },
    /// Code inside a document (a fenced or indented block, a notebook code cell).
    Code,
    /// A figure or table caption.
    Caption,
    Quote,
    /// A warning or note box.
    Note,
    /// A term and its definition.
    Definition,
    /// Front matter at the top of a note.
    FrontMatter,
    /// One record of a data file (a JSON or YAML key, a JSON Lines line); its key path is the label.
    Record,
    /// One log event (a line and its continuation lines); its time is in `loc.time`.
    LogEvent,
    /// One item of a code file (a function, a type); its name is the label.
    Item,
    /// Lines of a code file outside any item.
    Other,
}

impl BlockKind {
    /// Short name for reports and chunk records.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Heading(_) => "heading",
            Self::Paragraph => "paragraph",
            Self::ListItem { .. } => "list",
            Self::Table { .. } => "table",
            Self::Code => "code",
            Self::Caption => "caption",
            Self::Quote => "quote",
            Self::Note => "note",
            Self::Definition => "definition",
            Self::FrontMatter => "front-matter",
            Self::Record => "record",
            Self::LogEvent => "log",
            Self::Item => "item",
            Self::Other => "other",
        }
    }
    /// The heading level, when this is a heading.
    #[must_use]
    pub fn level(self) -> Option<u8> {
        if let Self::Heading(l) = self { Some(l) } else { None }
    }
}

/// Where a block sits in its source beyond its lines.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Loc {
    /// Physical pages, from 1.
    pub page: Option<(u32, u32)>,
    /// The printed page number.
    pub page_label: Option<String>,
    /// Cell range: `rows 14-20` or `Sheet1!B4:D9`.
    pub cells: Option<String>,
    /// Log time range, milliseconds since 1970.
    pub time: Option<(u64, u64)>,
    /// HTML id, chapter file, slide number, notebook cell.
    pub anchor: Option<String>,
}

/// One block: its kind, its lines of the derived text (1-based, inclusive), the heading above it,
/// a label (heading text, record key path, item name) and its locator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub kind: BlockKind,
    pub first: usize,
    pub last: usize,
    /// Index of the heading block above (for a heading: the nearest heading of a lower level).
    pub parent: Option<usize>,
    pub label: String,
    pub loc: Loc,
}

impl Block {
    /// A block with no parent and no locator yet.
    #[must_use]
    pub fn new(kind: BlockKind, first: usize, last: usize, label: &str) -> Self {
        Self { kind, first, last, parent: None, label: label.to_owned(), loc: Loc::default() }
    }
}

/// One document: where it came from, how it was read, its derived text and its blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Doc {
    /// Path relative to the studied root, forward slashes.
    pub path: String,
    /// The decoder that read it: `markdown`, `text`, `csv`, `html`, `rust`, ...
    pub decoder: &'static str,
    /// The source encoding: `utf-8`, `utf-8-bom`, `utf-16le`, `utf-16be`, `windows-1252`.
    pub encoding: &'static str,
    /// First heading, metadata title, or the file name.
    pub title: String,
    /// Derived text: what chunks and citations index into.
    pub text: String,
    pub blocks: Vec<Block>,
    /// Decode warnings.
    pub notes: Vec<String>,
    /// How a file was read when no specific reader fits it as a whole: `None` for a specific
    /// reader; `container` for a container's own listing, `member` for a file inside a container
    /// (read by the reader that fits it), `gzip` for gzip data read once inflated, `raw` for runs of
    /// text recovered from the bytes.
    pub general: Option<&'static str>,
}

impl Doc {
    /// The chain of heading labels above block `i`, top first (the block's own label when it is
    /// a heading comes last).
    #[must_use]
    pub fn heading_path(&self, i: usize) -> Vec<&str> {
        heading_path(&self.blocks, i)
    }
}

/// Link each block to the heading above it: a heading's parent is the nearest earlier heading of
/// a lower level; any other block's parent is the nearest earlier heading.
pub fn link_parents(blocks: &mut [Block]) {
    let mut stack: Vec<(u8, usize)> = Vec::new();
    for i in 0..blocks.len() {
        let Some(b) = blocks.get(i) else { continue };
        if let Some(level) = b.kind.level() {
            while stack.last().is_some_and(|(l, _)| *l >= level) {
                stack.pop();
            }
            let parent = stack.last().map(|x| x.1);
            if let Some(b) = blocks.get_mut(i) {
                b.parent = parent;
            }
            stack.push((level, i));
        } else {
            let parent = stack.last().map(|x| x.1);
            if let Some(b) = blocks.get_mut(i) {
                b.parent = parent;
            }
        }
    }
}

/// The chain of heading labels above block `i`, top first; a heading's own label comes last.
#[must_use]
pub fn heading_path(blocks: &[Block], i: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut at = blocks.get(i).and_then(|b| if b.kind.level().is_some() { Some(i) } else { b.parent });
    let mut guard = 0;
    while let Some(j) = at {
        let Some(b) = blocks.get(j) else { break };
        out.push(b.label.as_str());
        at = b.parent;
        guard += 1;
        if guard > 64 {
            break;
        }
    }
    out.reverse();
    out
}

/// The block holding line `line`, by binary search over blocks in line order.
#[must_use]
pub fn block_at(blocks: &[Block], line: usize) -> Option<usize> {
    let i = blocks.partition_point(|b| b.last < line);
    blocks.get(i).filter(|b| b.first <= line).map(|_| i)
}

/// Every line from 1 to `lines` in exactly one block, blocks in order and not overlapping.
/// # Errors
/// The first line that is missing or held twice.
pub fn check_cover(blocks: &[Block], lines: usize) -> Result<(), String> {
    let mut next = 1;
    for b in blocks {
        if b.first != next || b.last < b.first {
            return Err(format!("block {} at lines {}-{} where line {next} was expected", b.kind.name(), b.first, b.last));
        }
        next = b.last + 1;
    }
    if next != lines + 1 && !(lines == 0 && blocks.is_empty()) {
        return Err(format!("blocks end at line {} of {lines}", next - 1));
    }
    Ok(())
}

pub(crate) fn check() -> Result<(), String> {
    let mut b = vec![
        Block::new(BlockKind::Heading(1), 1, 1, "Manual"),
        Block::new(BlockKind::Heading(2), 2, 2, "Care"),
        Block::new(BlockKind::Paragraph, 3, 4, ""),
        Block::new(BlockKind::Heading(1), 5, 5, "Parts"),
        Block::new(BlockKind::ListItem { depth: 0, ordered: true }, 6, 6, ""),
    ];
    link_parents(&mut b);
    if heading_path(&b, 2) != ["Manual", "Care"] || heading_path(&b, 4) != ["Parts"] || heading_path(&b, 1) != ["Manual", "Care"] {
        return Err(format!("nmtype.doc: heading paths {:?} {:?}", heading_path(&b, 2), heading_path(&b, 4)));
    }
    check_cover(&b, 6).map_err(|e| format!("nmtype.doc: {e}"))?;
    if check_cover(&b, 7).is_ok() || block_at(&b, 4) != Some(2) || block_at(&b, 9).is_some() {
        return Err("nmtype.doc: cover or block_at".into());
    }
    Ok(())
}
