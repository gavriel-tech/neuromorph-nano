//! Document structure beyond the pages: the tagged structure tree (`/StructTreeRoot`: reading
//! order of marked content and the role of each block), the outline (bookmarks, as headings with
//! their pages) and page labels (printed page numbers).
use super::file::{File, text_string};
use super::layout::{Kind, Mark};
use super::lex::{Dict, Obj};
use std::collections::HashMap;

/// Most structure elements and outline items visited.
const MAX_NODES: usize = 1_000_000;

/// The standard role a structure type maps to through `/RoleMap`.
fn standard_role(role: &str, map: &Dict) -> String {
    let mut r = role.to_owned();
    for _ in 0..8 {
        match map.name(&r) {
            Some(n) if n != r => r = n.to_owned(),
            _ => break,
        }
    }
    r
}

fn block_kind(role: &str) -> Option<Kind> {
    Some(match role {
        "H1" | "H" | "Title" => Kind::Heading(1),
        "H2" => Kind::Heading(2),
        "H3" => Kind::Heading(3),
        "H4" => Kind::Heading(4),
        "H5" => Kind::Heading(5),
        "H6" => Kind::Heading(6),
        "P" | "Caption" | "TOCI" | "Note" | "BlockQuote" | "Quote" | "Lbl" | "Index" | "BibEntry" | "Div" | "Sect" | "Art" | "Part" => Kind::Text,
        "LI" => Kind::List,
        "TR" => Kind::Row { head: false },
        "Code" => Kind::Code,
        _ => return None,
    })
}

/// Marked content of every page by the structure tree: page index to (MCID to its mark).
#[must_use]
pub fn tags(file: &File<'_>, page_index: &HashMap<u32, usize>) -> HashMap<usize, HashMap<u32, Mark>> {
    let mut out: HashMap<usize, HashMap<u32, Mark>> = HashMap::new();
    let Some(root) = file.root() else { return out };
    let tree = file.at(&root, "StructTreeRoot");
    let Some(tree) = tree.as_dict() else { return out };
    let map = file.at(tree, "RoleMap").as_dict().cloned().unwrap_or_default();
    let mut w = Walk { file, map, page_index, out: &mut out, seq: 0, nodes: 0, next_id: 0 };
    if let Some(k) = tree.get("K") {
        w.kids(k, None, &Ctx { block: None, kind: Kind::Text, cell: None, row_head: false }, 0);
    }
    out
}

#[derive(Clone)]
struct Ctx {
    block: Option<usize>,
    kind: Kind,
    cell: Option<usize>,
    row_head: bool,
}

struct Walk<'w, 'a> {
    file: &'w File<'a>,
    map: Dict,
    page_index: &'w HashMap<u32, usize>,
    out: &'w mut HashMap<usize, HashMap<u32, Mark>>,
    seq: usize,
    nodes: usize,
    next_id: usize,
}

impl Walk<'_, '_> {
    fn page_of(&self, pg: Option<&Obj>) -> Option<usize> {
        match pg {
            Some(Obj::Ref(n, _)) => self.page_index.get(n).copied(),
            _ => None,
        }
    }

    fn mcid(&mut self, page: Option<usize>, mcid: i64, ctx: &Ctx) {
        let (Some(p), Ok(m)) = (page, u32::try_from(mcid)) else { return };
        let block = ctx.block.unwrap_or_else(|| {
            self.next_id += 1;
            self.next_id
        });
        let kind = match ctx.kind {
            Kind::Row { .. } => Kind::Row { head: ctx.row_head },
            k => k,
        };
        self.seq += 1;
        self.out.entry(p).or_default().entry(m).or_insert(Mark { seq: self.seq, block, kind, cell: ctx.cell });
    }

    fn kids(&mut self, k: &Obj, page: Option<usize>, ctx: &Ctx, depth: usize) {
        if depth > 200 || self.nodes > MAX_NODES {
            return;
        }
        self.nodes += 1;
        match k {
            Obj::Int(m) => self.mcid(page, *m, ctx),
            Obj::Array(a) => {
                for x in a {
                    self.kids(x, page, ctx, depth + 1);
                }
            }
            Obj::Ref(..) => {
                let o = self.file.resolve(k);
                if matches!(o, Obj::Dict(_) | Obj::Stream(_)) {
                    self.kids(&o, page, ctx, depth + 1);
                }
            }
            Obj::Dict(d) => self.element(d, page, ctx, depth),
            _ => {}
        }
    }

    fn element(&mut self, d: &Dict, page: Option<usize>, ctx: &Ctx, depth: usize) {
        match d.name("Type") {
            Some("MCR") => {
                let p = self.page_of(d.get("Pg")).or(page);
                if let Some(m) = d.int("MCID") {
                    self.mcid(p, m, ctx);
                }
                return;
            }
            Some("OBJR") => return,
            _ => {}
        }
        let page = self.page_of(d.get("Pg")).or(page);
        let role = d.name("S").map(|r| standard_role(r, &self.map)).unwrap_or_default();
        let mut c = ctx.clone();
        if role == "Artifact" {
            return;
        }
        // Inside a table row, cells stay in the row's block; elsewhere the nearest block wins.
        let in_row = matches!(ctx.kind, Kind::Row { .. }) && ctx.block.is_some();
        if matches!(role.as_str(), "TD" | "TH") && in_row {
            self.next_id += 1;
            c.cell = Some(self.next_id);
            if role == "TH" {
                c.row_head = true;
            }
        } else if !in_row && let Some(kind) = block_kind(&role) {
            // A list item's label and body, a quote's paragraphs: the outer list item holds them.
            let keep_outer = matches!(ctx.kind, Kind::List) && ctx.block.is_some();
            if !keep_outer {
                self.next_id += 1;
                c.block = Some(self.next_id);
                c.kind = kind;
                c.cell = None;
                c.row_head = false;
            }
        } else if ctx.block.is_none() && !self.has_block(d, 0) {
            // A grouping element outside any block (`NonStruct`, `Div` holding no paragraph): the
            // text under it is one block.
            self.next_id += 1;
            c.block = Some(self.next_id);
            c.kind = Kind::Text;
        }
        if let Some(k) = d.get("K") {
            self.kids(k, page, &c, depth + 1);
        }
    }
}

impl Walk<'_, '_> {
    /// True when an element holds a block element (a paragraph, heading, list item or row) below it.
    fn has_block(&mut self, d: &Dict, depth: usize) -> bool {
        if depth > 12 || self.nodes > MAX_NODES {
            return false;
        }
        self.nodes += 1;
        let kids = match d.get("K") {
            Some(Obj::Array(a)) => a.clone(),
            Some(k) => vec![k.clone()],
            None => return false,
        };
        for k in kids {
            let o = self.file.resolve(&k);
            let Some(e) = o.as_dict() else { continue };
            if matches!(e.name("Type"), Some("MCR" | "OBJR")) {
                continue;
            }
            let role = e.name("S").map(|r| standard_role(r, &self.map)).unwrap_or_default();
            if block_kind(&role).is_some_and(|k| k != Kind::Text || role == "P") || self.has_block(e, depth + 1) {
                return true;
            }
        }
        false
    }
}

/// One outline item: its title, depth (1 is the top) and page index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bookmark {
    pub title: String,
    pub level: u8,
    pub page: Option<usize>,
}

/// Every named destination of the file (`/Dests` in the catalog and the `/Names` tree).
fn named_dests(file: &File<'_>, root: &Dict) -> HashMap<Vec<u8>, Obj> {
    let mut out = HashMap::new();
    if let Some(d) = file.at(root, "Dests").as_dict() {
        for (k, v) in &d.0 {
            out.insert(k.as_bytes().to_vec(), file.resolve(v));
        }
    }
    let names = file.at(root, "Names");
    if let Some(n) = names.as_dict() {
        let tree = file.at(n, "Dests");
        let mut stack = vec![(tree, 0usize)];
        let mut nodes = 0;
        while let Some((node, depth)) = stack.pop() {
            nodes += 1;
            if nodes > 100_000 || depth > 32 {
                break;
            }
            let Some(d) = node.as_dict() else { continue };
            if let Some(a) = file.at(d, "Names").as_array() {
                for p in a.chunks(2) {
                    if let [Obj::Str(k), v] = p {
                        out.insert(k.clone(), file.resolve(v));
                    }
                }
            }
            if let Some(kids) = file.at(d, "Kids").as_array() {
                for k in kids {
                    stack.push((file.resolve(k), depth + 1));
                }
            }
        }
    }
    out
}

fn dest_page(file: &File<'_>, dest: &Obj, names: &HashMap<Vec<u8>, Obj>, page_index: &HashMap<u32, usize>, pages: usize) -> Option<usize> {
    let dest = match dest {
        Obj::Name(n) => names.get(n.as_bytes()).cloned()?,
        Obj::Str(s) => names.get(s).cloned()?,
        o => file.resolve(o),
    };
    let dest = match &dest {
        Obj::Dict(d) => file.at(d, "D"),
        _ => dest,
    };
    match dest.as_array()?.first()? {
        Obj::Ref(n, _) => page_index.get(n).copied(),
        Obj::Int(i) => usize::try_from(*i).ok().filter(|i| *i < pages),
        _ => None,
    }
}

/// The outline, depth first.
#[must_use]
pub fn outline(file: &File<'_>, page_index: &HashMap<u32, usize>, pages: usize) -> Vec<Bookmark> {
    let mut out = Vec::new();
    let Some(root) = file.root() else { return out };
    let ol = file.at(&root, "Outlines");
    let Some(ol) = ol.as_dict() else { return out };
    let names = named_dests(file, &root);
    let mut seen: Vec<u32> = Vec::new();
    let mut stack: Vec<(Option<Obj>, u8)> = vec![(ol.get("First").cloned(), 1)];
    while let Some((item, level)) = stack.pop() {
        let Some(item) = item else { continue };
        if out.len() > 10_000 || level > 16 {
            break;
        }
        if let Obj::Ref(n, _) = item {
            if seen.contains(&n) {
                continue;
            }
            seen.push(n);
        }
        let o = file.resolve(&item);
        let Some(d) = o.as_dict() else { continue };
        let title = file.at(d, "Title").as_str().map(text_string).unwrap_or_default();
        let dest = d.get("Dest").cloned().or_else(|| {
            let a = file.at(d, "A");
            let a = a.as_dict()?;
            (a.name("S") == Some("GoTo")).then(|| a.get("D").cloned()).flatten()
        });
        let page = dest.and_then(|x| dest_page(file, &x, &names, page_index, pages));
        out.push(Bookmark { title: title.split_whitespace().collect::<Vec<_>>().join(" "), level, page });
        // Siblings after this item's children: push the next sibling first.
        stack.push((d.get("Next").cloned(), level));
        stack.push((d.get("First").cloned(), level.saturating_add(1)));
    }
    out
}

fn roman(mut n: usize, upper: bool) -> String {
    let table = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")];
    let mut s = String::new();
    for (v, r) in table {
        while n >= v && s.len() < 40 {
            s.push_str(r);
            n -= v;
        }
    }
    if upper { s.to_uppercase() } else { s }
}

fn letters(n: usize, upper: bool) -> String {
    let c = char::from(b'a' + u8::try_from((n.max(1) - 1) % 26).unwrap_or(0));
    let s: String = std::iter::repeat_n(c, (n.max(1) - 1) / 26 + 1).collect();
    if upper { s.to_uppercase() } else { s }
}

/// The printed label of each page from `/PageLabels`; `None` for a page with no label range.
#[must_use]
pub fn page_labels(file: &File<'_>, pages: usize) -> Vec<Option<String>> {
    let mut out = vec![None; pages];
    let Some(root) = file.root() else { return out };
    let tree = file.at(&root, "PageLabels");
    let mut ranges: Vec<(usize, Dict)> = Vec::new();
    let mut stack = vec![(tree, 0usize)];
    while let Some((node, depth)) = stack.pop() {
        if depth > 32 || ranges.len() > 100_000 {
            break;
        }
        let Some(d) = node.as_dict() else { continue };
        if let Some(a) = file.at(d, "Nums").as_array() {
            for p in a.chunks(2) {
                if let [k, v] = p
                    && let (Some(i), Some(l)) = (k.as_int().and_then(|i| usize::try_from(i).ok()), file.resolve(v).as_dict().cloned())
                {
                    ranges.push((i, l));
                }
            }
        }
        if let Some(kids) = file.at(d, "Kids").as_array() {
            for k in kids {
                stack.push((file.resolve(k), depth + 1));
            }
        }
    }
    if ranges.is_empty() {
        return out;
    }
    ranges.sort_by_key(|r| r.0);
    for (i, slot) in out.iter_mut().enumerate() {
        let Some((start, d)) = ranges.iter().rev().find(|(s, _)| *s <= i) else { continue };
        let n = usize::try_from(d.int("St").unwrap_or(1)).unwrap_or(1) + (i - start);
        let prefix = d.get("P").and_then(Obj::as_str).map(text_string).unwrap_or_default();
        let num = match d.name("S") {
            Some("D") => n.to_string(),
            Some("R") => roman(n, true),
            Some("r") => roman(n, false),
            Some("A") => letters(n, true),
            Some("a") => letters(n, false),
            _ => String::new(),
        };
        *slot = Some(format!("{prefix}{num}"));
    }
    out
}

pub(crate) fn check() -> Result<(), String> {
    if roman(14, false) != "xiv" || roman(1999, true) != "MCMXCIX" || letters(28, true) != "BB" {
        return Err("nmdoc.pdf.structure: page label numbering".into());
    }
    let mut map = Dict::default();
    map.set("Heading1", Obj::Name("H1".into()));
    map.set("Custom", Obj::Name("Heading1".into()));
    if standard_role("Custom", &map) != "H1" || block_kind("H2") != Some(Kind::Heading(2)) {
        return Err("nmdoc.pdf.structure: role map".into());
    }
    Ok(())
}
