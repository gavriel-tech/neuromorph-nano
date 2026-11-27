//! nmdoc: the document decoders of the Neuromorph Nano. The format table (`format`) says which
//! files are read and why others are skipped; the text decoder (`text`) turns bytes into text in
//! any common encoding; the structure readers turn a document into the one document model of
//! `nmtype::doc` (derived text plus blocks). Files no specific reader fits are read generally
//! (`general`): containers opened in memory (`zip`, `tar`, `inflate`) with each member read by
//! the reader that fits it, anything else as runs of text recovered from its bytes (`raw`). PDF has
//! its own reader (`pdf`): its text sits in compressed streams as placed glyphs that only a reader
//! of the format can reach and put in order. Depends only on `nmtype`, `nmjson` and `nmlex` (the
//! language table); standard library
//! only, no `unsafe`; every decoder returns a document or a named reason and never panics on bad
//! input.
pub mod csv;
pub mod format;
pub mod general;
pub mod html;
pub mod inflate;
pub mod json;
pub mod log;
pub mod markup;
pub mod md;
pub mod pdf;
pub mod raw;
pub mod tar;
pub mod text;
pub mod xml;
pub mod yaml;
pub mod zip;

pub use format::Class;
use nmtype::doc::{Block, Doc};

/// What reading one file gave: a document, or the reason it was skipped.
pub type Read = Result<Doc, &'static str>;

/// The file name of a path.
fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Blocks of already decoded text for a decoder (the text is the derived text).
#[must_use]
pub fn structure(decoder: &str, text: &str) -> Vec<Block> {
    match decoder {
        // HTML, XML and notebooks are derived into Markdown form (`from_text`).
        "markdown" | "html" | "xml" | "notebook" => md::markdown(text),
        "text" => md::plain(text),
        "json" => json::blocks(text),
        "jsonl" => json::lines_blocks(text),
        "yaml" => yaml::blocks(text),
        "log" => log::blocks(text),
        "csv" => csv::blocks(text, ','),
        "tsv" => csv::blocks(text, '\t'),
        "raw" | "container" => paragraphs(text),
        _ => {
            let mut b = Vec::new();
            md::close_gaps(&mut b, text.lines().count());
            b
        }
    }
}

/// Paragraph blocks split at blank lines, a long paragraph cut every 40 lines; no headings (for
/// text read generally, which has none).
#[must_use]
pub fn paragraphs(text: &str) -> Vec<Block> {
    let mut b = Vec::new();
    let (mut start, mut len) = (None::<usize>, 0);
    for (i, l) in text.lines().enumerate() {
        if l.trim().is_empty() {
            start = None;
            continue;
        }
        if start.is_none() || len >= 40 {
            b.push(Block::new(nmtype::doc::BlockKind::Paragraph, i + 1, i + 1, ""));
            start = Some(i + 1);
            len = 0;
        }
        len += 1;
    }
    md::close_gaps(&mut b, text.lines().count());
    b
}

/// A document from decoded text: derived text, blocks and title.
#[must_use]
pub fn from_text(path: &str, decoder: &'static str, encoding: &'static str, text: String) -> Doc {
    let mut notes = Vec::new();
    let (text, named) = match decoder {
        "html" => html::derive(&text),
        "xml" => xml::derive(&text),
        "notebook" => match json::notebook(&text) {
            Ok(t) => (t, String::new()),
            Err(e) => {
                notes.push(e.to_owned());
                (text, String::new())
            }
        },
        _ => (text, String::new()),
    };
    let blocks = structure(decoder, &text);
    let title = if named.is_empty() {
        blocks.iter().find(|b| b.kind.level().is_some()).map_or_else(|| file_name(path).to_owned(), |b| b.label.clone())
    } else {
        named
    };
    Doc { path: path.to_owned(), decoder, encoding, title, text, blocks, notes, general: None }
}

/// Read one file's bytes as a document of the given class (from `format::by_name`), by a
/// specific reader only.
/// # Errors
/// The skip reason, or `read generally` when only the general reading (`read`) fits it.
pub fn decode(path: &str, class: Class, bytes: &[u8]) -> Read {
    let decoder = match format::by_bytes(class, bytes) {
        Class::Skip(why) => return Err(why),
        Class::Doc(general::GENERAL) => return Err("read generally"),
        Class::Doc("pdf") => return pdf::read(path, bytes),
        Class::Code(lang) => lang,
        Class::Doc(d) => d,
    };
    let (text, encoding) = text::decode(bytes)?;
    Ok(from_text(path, decoder, encoding, text))
}

/// Read one file's bytes by every means that fits: the specific reader of its class, or the
/// general reading (a container's listing and members, or the runs of text in its bytes). One
/// file gives one document, or for a container one per member read, plus a skip reason for each
/// member not read.
#[must_use]
pub fn read(path: &str, class: Class, bytes: &[u8]) -> Vec<general::Item> {
    match format::by_bytes(class, bytes) {
        Class::Skip(why) => vec![Err((path.to_owned(), why))],
        Class::Doc(general::GENERAL) => general::read(path, bytes),
        Class::Doc("pdf") => vec![read_pdf(path, bytes)],
        Class::Code(d) | Class::Doc(d) => match text::decode(bytes) {
            Ok((t, enc)) => vec![Ok(from_text(path, d, enc, t))],
            Err(_) => general::read(path, bytes),
        },
    }
}

/// A PDF by its own reader. A file that needs a password, or uses another security handler, is
/// skipped with that reason; a file the reader cannot open at all is read raw with a note, so it is
/// never unreadable.
fn read_pdf(path: &str, bytes: &[u8]) -> general::Item {
    match pdf::read(path, bytes) {
        Ok(d) => Ok(d),
        Err(why) if why == pdf::crypt::NEEDS_PASSWORD || why == pdf::crypt::UNSUPPORTED => Err((path.to_owned(), why)),
        Err(why) => Ok(general::raw_doc(path, bytes, Some(&format!("the PDF could not be read by the PDF reader ({why})")))),
    }
}

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate, in order.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmdoc.format", format::check),
        ("nmdoc.text", text::check),
        ("nmdoc.inflate", inflate::check),
        ("nmdoc.zip", zip::check),
        ("nmdoc.tar", tar::check),
        ("nmdoc.raw", raw::check),
        ("nmdoc.general", general::check),
        ("nmdoc.pdf", pdf::check),
        ("nmdoc.md", md::check),
        ("nmdoc.csv", csv::check),
        ("nmdoc.markup", markup::check),
        ("nmdoc.html", html::check),
        ("nmdoc.xml", xml::check),
        ("nmdoc.json", json::check),
        ("nmdoc.yaml", yaml::check),
        ("nmdoc.log", log::check),
    ]
}
