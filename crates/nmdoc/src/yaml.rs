//! YAML, a subset read for structure: the file is its derived text; each top-level key (or each
//! top-level sequence item) is a `Record` labelled by its key, its value being the indented lines
//! below it; a key whose value runs `LARGE` lines or more becomes a heading with its own keys as
//! records; full-line comments directly above a key belong to it; `---` separates documents, each
//! a heading `document N` when there are several. Flow style, anchors and multi-line scalars are
//! read as the text they are, inside the record that holds them.
use nmtype::doc::{Block, BlockKind};

/// A key whose value runs at least this many lines becomes a heading over its own keys.
pub const LARGE: usize = 40;

fn indent(l: &str) -> usize {
    l.len() - l.trim_start_matches(' ').len()
}

/// The key of a mapping line (`name:` or `name: value`), not inside quotes or a comment.
fn key_of(l: &str) -> Option<&str> {
    let t = l.trim_start();
    if t.starts_with('#') || t.starts_with('-') || (t.starts_with('"') && !t.contains("\":")) {
        return None;
    }
    let (k, rest) = t.split_once(':')?;
    let k = k.trim().trim_matches(['"', '\'']);
    let plain = !k.is_empty() && !k.contains(' ');
    let words = !k.is_empty() && k.split_whitespace().count() <= 3 && (rest.is_empty() || rest.starts_with(' '));
    (plain || words).then_some(k)
}

/// Records of the lines `s..e` (0-based, end exclusive) at indentation `ind`.
fn records(lines: &[&str], (s, e): (usize, usize), ind: usize, path: &str, level: u8, out: &mut Vec<Block>) {
    let mut starts: Vec<(usize, String)> = Vec::new();
    let mut item = 0;
    for i in s..e {
        let l = lines.get(i).copied().unwrap_or("");
        if l.trim().is_empty() || indent(l) != ind || l.trim_start().starts_with('#') {
            continue;
        }
        let t = l.trim_start();
        if t.starts_with("- ") || t == "-" {
            let name = t
                .trim_start_matches('-')
                .trim()
                .split_once(':')
                .filter(|(k, _)| matches!(k.trim(), "name" | "id" | "key" | "title"))
                .map(|(_, v)| v.trim().trim_matches(['"', '\'']).to_owned());
            let at = format!("{path}[{item}]");
            starts.push((i, name.map_or(at.clone(), |n| format!("{at} {n}"))));
            item += 1;
        } else if let Some(k) = key_of(l) {
            starts.push((i, if path.is_empty() { k.to_owned() } else { format!("{path}.{k}") }));
        }
    }
    for (j, (i, label)) in starts.iter().enumerate() {
        let end = starts.get(j + 1).map_or(e, |x| x.0);
        // Comments directly above belong to the key.
        let mut first = *i;
        while first > s && lines.get(first - 1).is_some_and(|l| l.trim_start().starts_with('#') && indent(l) == ind) {
            first -= 1;
        }
        let span = end - i;
        let child = (i + 1..end).find_map(|k| lines.get(k).filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#')).map(|l| indent(l)));
        match child {
            Some(ci) if span >= LARGE && ci > ind && level < 3 => {
                out.push(Block::new(BlockKind::Heading(level), first + 1, i + 1, label));
                records(lines, (i + 1, end), ci, label, level + 1, out);
            }
            _ => out.push(Block::new(BlockKind::Record, first + 1, end, label)),
        }
    }
}

/// Blocks of a YAML file over its own lines.
#[must_use]
pub fn blocks(text: &str) -> Vec<Block> {
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    let seps: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.trim_end() == "---").map(|(i, _)| i).collect();
    let mut out = Vec::new();
    let docs: Vec<(usize, usize)> = {
        let mut cuts = vec![0];
        cuts.extend(seps.iter().map(|s| s + 1));
        cuts.iter().enumerate().map(|(j, c)| (*c, cuts.get(j + 1).map_or(n, |x| x - 1))).filter(|(a, b)| b > a).collect()
    };
    let several = docs.len() > 1;
    for (d, (a, b)) in docs.iter().enumerate() {
        // A document after a separator is a heading of its own (the separator line).
        if several && *a > 0 {
            out.push(Block::new(BlockKind::Heading(1), *a, *a, &format!("document {}", d + 1)));
        }
        records(&lines, (*a, *b), 0, "", if several { 2 } else { 1 }, &mut out);
    }
    out.sort_by_key(|b| b.first);
    out.dedup_by_key(|b| b.first);
    crate::md::close_gaps(&mut out, n);
    nmtype::doc::link_parents(&mut out);
    out
}

pub(crate) fn check() -> Result<(), String> {
    let y = "# service settings\nname: pump\nports:\n  - 80\n  - 443\n\n# the owners\nowners:\n  - name: ops\n    mail: ops@example.com\n";
    let b = blocks(y);
    nmtype::doc::check_cover(&b, y.lines().count()).map_err(|e| format!("nmdoc.yaml: {e}"))?;
    let got: Vec<(usize, &str)> = b.iter().map(|x| (x.first, x.label.as_str())).collect();
    if got != [(1, "name"), (3, "ports"), (7, "owners")] {
        return Err(format!("nmdoc.yaml: records {got:?}"));
    }
    let m = "a: 1\n---\nb: 2\n";
    let bm = blocks(m);
    let got: Vec<&str> = bm.iter().map(|x| x.label.as_str()).collect();
    if got != ["a", "document 2", "b"] {
        return Err(format!("nmdoc.yaml: documents {got:?}"));
    }
    Ok(())
}
