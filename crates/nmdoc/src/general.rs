//! General reading, for files no specific reader fits as a whole. A container (zip, gzip, tar)
//! is opened in memory and each member is read by whichever reader fits it, by the same rules as
//! a file on disk, and located as `container!member`; a Word, Excel, PowerPoint, EPUB or
//! OpenDocument file is a zip of XML, so its XML members reach the XML reader with no knowledge of
//! those formats. Anything else is read raw (`raw`). Every document made here carries its mark
//! (`Doc::general`), so a pack can say how well it was read.
use crate::format::{self, Class};
use crate::inflate::{self, Limits};
use crate::{raw, tar, zip};
use nmtype::doc::Doc;

/// The class the format table gives a file to be read generally.
pub const GENERAL: &str = "general";
/// Largest member read from a container, uncompressed. Larger than the limit for a text file on
/// disk, since office members are verbose markup whose derived text is a fraction of their size.
pub const MAX_MEMBER: usize = 8 << 20;
/// Most bytes inflated from one file on disk, all members and nesting together.
pub const MAX_TOTAL: usize = 64 << 20;
/// Deepest nesting of containers opened (a zip in a zip in a zip).
pub const MAX_DEPTH: usize = 3;
/// Why a member is not read.
pub const NO_TEXT: &str = "member holds no text (image, font or binary part)";
pub const ESCAPE: &str = "member path escapes the container: not read";
pub const BUDGET: &str = "container limit reached: member not opened";
pub const TOO_DEEP: &str = "containers nested too deep: not opened";
pub const MEMBER_LARGE: &str = "too large: member over 8 MB";

/// One result of reading: a document, or a path and the reason it was not read.
pub type Item = Result<Doc, (String, &'static str)>;

fn file_name(path: &str) -> &str {
    path.rsplit(['/', '!']).next().unwrap_or(path)
}

/// One general reading in progress: the bytes still allowed to be inflated for the file on disk,
/// and what was read so far.
struct Cx {
    left: usize,
    out: Vec<Item>,
}

impl Cx {
    fn limits(&self) -> Limits {
        Limits { max_out: self.left.min(MAX_MEMBER), ..Limits::default() }
    }
}

/// True when a member name would leave its container if used as a path.
fn escapes(name: &str) -> bool {
    name.starts_with('/') || name.as_bytes().get(1) == Some(&b':') || name.split('/').any(|s| s == "..")
}

fn mark(mut d: Doc, how: &'static str) -> Doc {
    d.general = Some(how);
    d
}

/// A file read raw, with an optional note on why (a container that would not open, say). Binary
/// content (`raw::holds_text` false: a program, image, archive the readers do not open, a
/// cartridge or weights file, any head with NUL bytes or mostly control bytes) is kept by its
/// metadata only, so its bytes never become text chunks or citations.
#[must_use]
pub fn raw_doc(path: &str, bytes: &[u8], note: Option<&str>) -> Doc {
    if raw::holds_text(bytes) {
        mark(crate::from_text(path, "raw", "binary", raw::derive(file_name(path), bytes, note)), "raw")
    } else {
        mark(crate::from_text(path, "raw", "binary", raw::describe(file_name(path), bytes, note)), "binary")
    }
}

/// Read a file of the general class: open it as a container when it is one, else read it raw.
#[must_use]
pub fn read(path: &str, bytes: &[u8]) -> Vec<Item> {
    let mut cx = Cx { left: MAX_TOTAL, out: Vec::new() };
    open(path, bytes, 0, &mut cx);
    cx.out
}

fn is_container(b: &[u8]) -> bool {
    zip::is_zip(b) || inflate::is_gzip(b) || tar::is_tar(b)
}

fn open(path: &str, bytes: &[u8], depth: usize, cx: &mut Cx) {
    if zip::is_zip(bytes) {
        match zip::entries(bytes) {
            Ok(list) => zip_members(path, bytes, &list, depth, cx),
            Err(e) => cx.out.push(Ok(raw_doc(path, bytes, Some(&format!("the zip container could not be opened ({e})"))))),
        }
        return;
    }
    if inflate::is_gzip(bytes) {
        match inflate::gzip(bytes, &cx.limits()) {
            Ok(g) => {
                cx.left = cx.left.saturating_sub(g.data.len());
                if tar::is_tar(&g.data) {
                    tar_members(path, &g.data, depth, cx);
                    return;
                }
                // The inflated bytes are the file itself: read by the reader its inner name fits,
                // located by the gzip file's own path.
                let base = file_name(path);
                let inner = if g.name.is_empty() {
                    base.strip_suffix(".gz").or_else(|| base.strip_suffix(".gzip")).unwrap_or(base).to_owned()
                } else {
                    g.name.clone()
                };
                let at = cx.out.len();
                member(path, &inner, &g.data, depth, cx, "gzip");
                for it in cx.out.iter_mut().skip(at) {
                    if matches!(it, Err((p, why)) if *why == NO_TEXT && p == path) {
                        *it = Ok(raw_doc(path, &g.data, Some("read once inflated from gzip")));
                    }
                }
            }
            Err(f) => cx.out.push(Ok(raw_doc(path, bytes, Some(&format!("the gzip data could not be opened ({})", f.name()))))),
        }
        return;
    }
    if tar::is_tar(bytes) {
        tar_members(path, bytes, depth, cx);
        return;
    }
    cx.out.push(Ok(raw_doc(path, bytes, None)));
}

/// Read one member by the format table, as a file on disk would be read. Returns what happened,
/// for the container's listing.
fn member(path: &str, name: &str, data: &[u8], depth: usize, cx: &mut Cx, how: &'static str) -> String {
    let class = format::by_name(name, 0);
    let class = if let Class::Skip(_) = class { class } else { format::by_bytes(class, data) };
    match class {
        Class::Skip(why) => {
            cx.out.push(Err((path.to_owned(), why)));
            why.to_owned()
        }
        Class::Doc(GENERAL) => {
            if is_container(data) {
                if depth + 1 > MAX_DEPTH {
                    cx.out.push(Err((path.to_owned(), TOO_DEEP)));
                    return TOO_DEEP.to_owned();
                }
                open(path, data, depth + 1, cx);
                "opened as a container".to_owned()
            } else {
                cx.out.push(Err((path.to_owned(), NO_TEXT)));
                NO_TEXT.to_owned()
            }
        }
        Class::Code(d) | Class::Doc(d) => match crate::text::decode(data) {
            Ok((t, enc)) => {
                cx.out.push(Ok(mark(crate::from_text(path, d, enc, t), how)));
                format!("read as {d}")
            }
            Err(_) => {
                cx.out.push(Err((path.to_owned(), NO_TEXT)));
                NO_TEXT.to_owned()
            }
        },
    }
}

/// The container's own document: its kind, size, and each member with what happened to it.
fn listing(path: &str, kind: &str, size: usize, rows: &[(String, u64, String)], note: Option<&str>) -> Doc {
    let mut text = format!(
        "container: {kind} (by its first bytes), {} members, {size} bytes\nread generally: each member is read by the reader that fits it, located as {path}!<member>\n",
        rows.len()
    );
    if let Some(n) = note {
        text.push_str(&format!("note: {n}\n"));
    }
    for (i, (name, bytes, what)) in rows.iter().enumerate() {
        if i % 20 == 0 {
            text.push('\n');
        }
        text.push_str(&format!("- {name} ({bytes} bytes): {what}\n"));
    }
    mark(crate::from_text(path, "container", "binary", text), "container")
}

fn zip_members(path: &str, bytes: &[u8], list: &[zip::Entry], depth: usize, cx: &mut Cx) {
    let mut rows = Vec::new();
    // The listing goes first; its place is held until every member has been read.
    let at = cx.out.len();
    cx.out.push(Err((path.to_owned(), BUDGET)));
    let files: Vec<&zip::Entry> = list.iter().filter(|e| !e.is_dir()).collect();
    let encrypted = files.iter().filter(|e| e.encrypted()).count();
    for e in &files {
        let mpath = format!("{path}!{}", e.name);
        let what: String = if escapes(&e.name) {
            cx.out.push(Err((mpath, ESCAPE)));
            ESCAPE.to_owned()
        } else if let Class::Skip(why) = format::by_name(file_name(&e.name), 0) {
            cx.out.push(Err((mpath, why)));
            why.to_owned()
        } else if e.encrypted() {
            cx.out.push(Err((mpath, "encrypted member: name only")));
            "encrypted: name only".to_owned()
        } else if e.size > MAX_MEMBER as u64 {
            cx.out.push(Err((mpath, MEMBER_LARGE)));
            MEMBER_LARGE.to_owned()
        } else if e.size > cx.left as u64 {
            cx.out.push(Err((mpath, BUDGET)));
            BUDGET.to_owned()
        } else {
            match zip::data(bytes, e, &cx.limits()) {
                Ok(data) => {
                    cx.left = cx.left.saturating_sub(data.len());
                    member(&mpath, file_name(&e.name), &data, depth, cx, "member")
                }
                Err(why) => {
                    cx.out.push(Err((mpath, why)));
                    why.to_owned()
                }
            }
        };
        rows.push((e.name.clone(), e.size, what));
    }
    let note = match encrypted {
        0 => None,
        n if n == files.len() => Some("encrypted or password-protected: only the member names are read".to_owned()),
        n => Some(format!("{n} encrypted members: only their names are read")),
    };
    if let Some(slot) = cx.out.get_mut(at) {
        *slot = Ok(listing(path, "zip", bytes.len(), &rows, note.as_deref()));
    }
}

fn tar_members(path: &str, bytes: &[u8], depth: usize, cx: &mut Cx) {
    let list = match tar::members(bytes) {
        Ok(l) => l,
        Err(e) => {
            cx.out.push(Ok(raw_doc(path, bytes, Some(&format!("the tar archive could not be read ({e})")))));
            return;
        }
    };
    let mut rows = Vec::new();
    let at = cx.out.len();
    cx.out.push(Err((path.to_owned(), BUDGET)));
    for m in &list {
        let mpath = format!("{path}!{}", m.name);
        let data = bytes.get(m.start..m.start + m.len).unwrap_or(&[]);
        let what = if escapes(&m.name) {
            cx.out.push(Err((mpath, ESCAPE)));
            ESCAPE.to_owned()
        } else if m.len > MAX_MEMBER {
            cx.out.push(Err((mpath, MEMBER_LARGE)));
            MEMBER_LARGE.to_owned()
        } else {
            member(&mpath, file_name(&m.name), data, depth, cx, "member")
        };
        rows.push((m.name.clone(), m.len as u64, what));
    }
    if let Some(slot) = cx.out.get_mut(at) {
        *slot = Ok(listing(path, "tar", bytes.len(), &rows, None));
    }
}

pub(crate) fn check() -> Result<(), String> {
    let doc_xml = b"<?xml version=\"1.0\"?><w:document><w:body><w:p><w:r><w:t>The pump seal is rated for forty bar.</w:t></w:r></w:p></w:body></w:document>";
    let inner = zip::stored_zip(&[("notes.txt", b"inner note about valves\n")]);
    let z = zip::stored_zip(&[
        ("word/document.xml", doc_xml),
        ("word/media/image1.png", b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"),
        ("../escape.txt", b"outside\n"),
        ("nested.zip", &inner),
        (".env", b"KEY=1\n"),
    ]);
    let items = read("a.docx", &z);
    let read_paths: Vec<(&str, Option<&str>)> = items.iter().filter_map(|i| i.as_ref().ok()).map(|d| (d.path.as_str(), d.general)).collect();
    let want = [
        ("a.docx", Some("container")),
        ("a.docx!word/document.xml", Some("member")),
        ("a.docx!nested.zip", Some("container")),
        ("a.docx!nested.zip!notes.txt", Some("member")),
    ];
    if read_paths != want {
        return Err(format!("nmdoc.general: read {read_paths:?}"));
    }
    let skipped: Vec<(&str, &str)> = items.iter().filter_map(|i| i.as_ref().err()).map(|(p, w)| (p.as_str(), *w)).collect();
    let want_skip = [("a.docx!word/media/image1.png", NO_TEXT), ("a.docx!../escape.txt", ESCAPE), ("a.docx!.env", "secret store (.env): never read")];
    if skipped != want_skip {
        return Err(format!("nmdoc.general: skipped {skipped:?}"));
    }
    let body = items.iter().filter_map(|i| i.as_ref().ok()).find(|d| d.path.ends_with("document.xml")).map(|d| d.text.clone()).unwrap_or_default();
    if !body.contains("The pump seal is rated for forty bar.") {
        return Err(format!("nmdoc.general: member text {body:?}"));
    }
    let broken = read("b.zip", b"PK\x03\x04 the rest is not a zip, but has readable words in it");
    if !matches!(broken.as_slice(), [Ok(d)] if matches!(d.general, Some("raw" | "binary")) && d.text.contains("could not be opened")) {
        return Err("nmdoc.general: a broken zip was not read raw".into());
    }
    Ok(())
}
