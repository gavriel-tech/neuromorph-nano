//! The PDF fixtures of `tests/documents/` against their expected text, by the pass rules of the
//! PDF reader's tests:
//! - per page, character similarity of the derived text to the expected text at least 0.98
//!   (similarity: one minus the character edit distance over the longer of the two, after white
//!   space is collapsed to single spaces);
//! - every gold passage on its page (`expected/pdf/gold.tsv`);
//! - two-column pages in the right reading order (they must pass the similarity rule, which an
//!   order error fails); also measured with the tagged structure set aside, as information;
//! - every encrypted copy reads exactly as the plain file;
//! - the picture-only page reported and not indexed; the page whose font has no character map
//!   reported and not indexed.
use super::{Options, read_with};
use nmtype::doc::Doc;
use std::collections::BTreeMap;
use std::path::Path;

/// Lowest similarity a page may have.
pub const MIN_SIMILARITY: f64 = 0.98;

/// Text with every run of white space made one space, trimmed.
#[must_use]
pub fn collapse(t: &str) -> String {
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One minus the character edit distance over the longer length.
#[must_use]
pub fn similarity(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let longer = a.len().max(b.len());
    if longer == 0 {
        return 1.0;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    1.0 - prev[b.len()] as f64 / longer as f64
}

/// The derived text of each page (page number to its lines, collapsed).
#[must_use]
pub fn page_texts(d: &Doc) -> BTreeMap<u32, String> {
    let lines: Vec<&str> = d.text.lines().collect();
    let mut out: BTreeMap<u32, String> = BTreeMap::new();
    for b in &d.blocks {
        let Some((p, _)) = b.loc.page else { continue };
        let t = lines.get(b.first.saturating_sub(1)..b.last.min(lines.len())).unwrap_or(&[]).join("\n");
        let e = out.entry(p).or_default();
        e.push(' ');
        e.push_str(&t);
    }
    out.into_iter().map(|(k, v)| (k, collapse(&v))).collect()
}

/// Expected text of each page from a file of `--- page N ---` sections.
fn expected(path: &Path) -> Result<BTreeMap<u32, String>, String> {
    let t = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = BTreeMap::new();
    let mut cur: Option<(u32, String)> = None;
    for l in t.lines() {
        if let Some(n) = l.strip_prefix("--- page ").and_then(|r| r.strip_suffix(" ---")).and_then(|n| n.parse::<u32>().ok()) {
            if let Some((p, s)) = cur.take() {
                out.insert(p, collapse(&s));
            }
            cur = Some((n, String::new()));
        } else if let Some((_, s)) = cur.as_mut() {
            s.push(' ');
            s.push_str(l);
        }
    }
    if let Some((p, s)) = cur {
        out.insert(p, collapse(&s));
    }
    Ok(out)
}

fn load(dir: &Path, name: &str, opts: Options) -> Result<Doc, String> {
    let bytes = std::fs::read(dir.join(name)).map_err(|e| format!("pdf: {name}: {e}"))?;
    read_with(name, &bytes, opts).map_err(|e| format!("pdf: {name}: {e}"))
}

/// Per page similarities of a document against its expected text: (page, similarity), pages
/// expected empty included; and the lowest.
fn compare(d: &Doc, want: &BTreeMap<u32, String>) -> (Vec<(u32, f64)>, f64) {
    let got = page_texts(d);
    let mut rows = Vec::new();
    let pages: std::collections::BTreeSet<u32> = want.keys().chain(got.keys()).copied().collect();
    for p in pages {
        let w = want.get(&p).map_or("", String::as_str);
        let g = got.get(&p).map_or("", String::as_str);
        rows.push((p, similarity(g, w)));
    }
    let min = rows.iter().map(|r| r.1).fold(1.0, f64::min);
    (rows, min)
}

/// Every PDF fixture check; the number of cases. Prints one line per file.
/// # Errors
/// The first rule broken.
pub fn check(base: &Path) -> Result<usize, String> {
    let dir = base.join("pdf");
    let exp = base.join("expected/pdf");
    let mut cases = 0;
    let mut docs: BTreeMap<String, Doc> = BTreeMap::new();
    for (name, want) in [
        ("manual-1col.pdf", "manual-1col.txt"),
        ("manual-2col.pdf", "manual-2col.txt"),
        ("manual-msprint.pdf", "manual-msprint.txt"),
        ("inspection-scan.pdf", "inspection-scan.txt"),
        ("crafted-features.pdf", "crafted-features.txt"),
    ] {
        let d = load(&dir, name, Options::default())?;
        let w = expected(&exp.join(want))?;
        let (rows, min) = compare(&d, &w);
        let mean = rows.iter().map(|r| r.1).sum::<f64>() / rows.len().max(1) as f64;
        for (p, s) in &rows {
            if *s < MIN_SIMILARITY {
                return Err(format!("pdf: {name} page {p}: similarity {s:.4} under {MIN_SIMILARITY}"));
            }
        }
        println!("ok   documents.pdf: {name}: {} pages, similarity lowest {min:.4}, mean {mean:.4}", rows.len());
        cases += rows.len();
        docs.insert(name.to_owned(), d);
    }
    // Two columns with the tagged structure set aside: reading order by geometry alone.
    let geo = load(&dir, "manual-2col.pdf", Options { tags: false })?;
    let (_, gmin) = compare(&geo, &expected(&exp.join("manual-2col.txt"))?);
    let geo1 = load(&dir, "manual-1col.pdf", Options { tags: false })?;
    let (_, gmin1) = compare(&geo1, &expected(&exp.join("manual-1col.txt"))?);
    println!("info documents.pdf: geometry only (tags set aside): manual-2col lowest {gmin:.4}, manual-1col lowest {gmin1:.4}");
    // Gold passages on their pages.
    let gold = std::fs::read_to_string(exp.join("gold.tsv")).map_err(|e| format!("pdf: gold.tsv: {e}"))?;
    for row in gold.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let cols: Vec<&str> = row.split('\t').collect();
        let (Some(f), Some(p), Some(passage)) = (cols.first(), cols.get(1).and_then(|p| p.parse::<u32>().ok()), cols.get(2)) else { continue };
        let d = docs.get(*f).ok_or_else(|| format!("pdf: gold names {f}, not read"))?;
        let pages = page_texts(d);
        let on: Vec<u32> = pages.iter().filter(|(_, t)| t.contains(passage)).map(|(k, _)| *k).collect();
        if on != [p] {
            return Err(format!("pdf: {f}: gold passage {passage:?} found on pages {on:?}, want {p}"));
        }
        cases += 1;
    }
    println!("ok   documents.pdf: every gold passage on its page");
    // Encrypted copies read exactly as the plain file.
    let plain = docs.get("manual-1col.pdf").map(|d| d.text.clone()).unwrap_or_default();
    for name in ["manual-1col-locked-rc4-40.pdf", "manual-1col-locked-rc4-128.pdf", "manual-1col-locked-aes-128.pdf", "manual-1col-locked-aes-256.pdf"] {
        let d = load(&dir, name, Options::default())?;
        if d.text != plain || !d.notes.iter().any(|n| n.contains("empty user password")) {
            return Err(format!("pdf: {name} does not read as the plain file ({} against {} bytes)", d.text.len(), plain.len()));
        }
        cases += 1;
    }
    println!("ok   documents.pdf: four encrypted copies (RC4 40 and 128 bit, AES-128, AES-256) read as the plain file");
    // The picture page: reported, nothing indexed on it.
    let scan = docs.get("inspection-scan.pdf").ok_or("pdf: inspection-scan.pdf not read")?;
    if !scan.notes.iter().any(|n| n.contains("page 2") && n.contains("no text layer")) || page_texts(scan).contains_key(&2) {
        return Err(format!("pdf: inspection-scan.pdf: picture page not reported or indexed (notes {:?})", scan.notes));
    }
    // The font with no character map: reported, nothing indexed.
    let nomap = load(&dir, "crafted-nomap.pdf", Options::default())?;
    if !nomap.notes.iter().any(|n| n.contains("page 1") && n.contains("unreadable")) || !page_texts(&nomap).is_empty() {
        return Err(format!("pdf: crafted-nomap.pdf: unreadable page not reported or indexed (notes {:?})", nomap.notes));
    }
    cases += 2;
    println!("ok   documents.pdf: picture-only page and unmappable font reported, not indexed");
    for d in docs.values() {
        nmtype::doc::check_cover(&d.blocks, d.text.lines().count()).map_err(|e| format!("pdf: {}: {e}", d.path))?;
    }
    Ok(cases)
}
