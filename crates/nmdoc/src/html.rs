//! HTML to derived text in Markdown form, so the Markdown structure reader gives its blocks:
//! `h1` to `h6` become `#` headings, paragraphs one line each, list items `-` or `1.` (nested by
//! two spaces), tables pipe rows (a separator after a header row), `pre` fenced code, block quotes
//! `>`, captions their own line. Scripts, styles, navigation, page header and footer are dropped;
//! the `title` names the document and becomes the top heading when the page has no `h1`.
use crate::markup::{Tok, collapse, tokens};

/// Elements whose content is dropped.
const DROP: &[&str] = &["script", "style", "noscript", "template", "svg", "nav", "header", "footer", "iframe", "object", "button", "select"];
/// Elements that start a block of text.
const BLOCK: &[&str] = &[
    "p", "div", "section", "article", "main", "aside", "body", "html", "address", "figure", "form", "fieldset", "dl", "details", "summary",
    "center",
];

struct Out {
    lines: Vec<String>,
    inline: String,
    /// Kind of the last block written, to place blank lines: `list`, `row`, `quote` or `para`.
    last: &'static str,
}

impl Out {
    fn block(&mut self, kind: &'static str, line: String) {
        let joins = (kind == "list" && self.last == "list") || (kind == "row" && self.last == "row") || (kind == "quote" && self.last == "quote");
        if !self.lines.is_empty() && !joins {
            self.lines.push(String::new());
        }
        self.lines.push(line);
        self.last = kind;
    }
    fn flush(&mut self, quote: bool) {
        let t = collapse(&self.inline);
        self.inline.clear();
        if !t.is_empty() {
            if quote { self.block("quote", format!("> {t}")) } else { self.block("para", t) }
        }
    }
}

/// The derived text of an HTML page and its title.
#[must_use]
pub fn derive(src: &str) -> (String, String) {
    let toks = tokens(src, true, &["script", "style", "textarea", "title"]);
    let mut out = Out { lines: Vec::new(), inline: String::new(), last: "" };
    let mut title = String::new();
    let mut drop = 0_usize;
    let mut in_title = false;
    let mut heading: Option<usize> = None;
    let mut lists: Vec<(bool, usize)> = Vec::new();
    let mut item: Option<String> = None;
    let mut row: Option<(Vec<String>, bool)> = None;
    let mut cell: Option<String> = None;
    let mut rows_seen = 0_usize;
    let mut pre: Option<String> = None;
    let mut quote = 0_usize;
    let mut has_h1 = false;
    for t in &toks {
        match t {
            Tok::Start(n, _, selfc) => {
                if DROP.contains(&n.as_str()) && !selfc {
                    drop += 1;
                    continue;
                }
                if drop > 0 {
                    continue;
                }
                match n.as_str() {
                    "title" => in_title = true,
                    "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                        out.flush(quote > 0);
                        heading = n.get(1..).and_then(|d| d.parse().ok());
                        has_h1 |= n == "h1";
                    }
                    "br" if pre.is_none() => {
                        if let Some(c) = cell.as_mut() {
                            c.push(' ');
                        } else if let Some(i) = item.as_mut() {
                            i.push(' ');
                        } else if heading.is_none() {
                            out.flush(quote > 0);
                        }
                    }
                    "ul" | "ol" | "menu" => {
                        if let Some(i) = item.take() {
                            emit_item(&mut out, &lists, &i);
                        }
                        out.flush(quote > 0);
                        lists.push((n == "ol", 0));
                    }
                    "li" => {
                        if let Some(i) = item.take() {
                            emit_item(&mut out, &lists, &i);
                        }
                        out.flush(quote > 0);
                        if let Some(l) = lists.last_mut() {
                            l.1 += 1;
                        }
                        item = Some(String::new());
                    }
                    "table" => {
                        out.flush(quote > 0);
                        rows_seen = 0;
                    }
                    "tr" => row = Some((Vec::new(), false)),
                    "td" | "th" => {
                        if let Some(r) = row.as_mut() {
                            r.1 |= n == "th";
                        }
                        cell = Some(String::new());
                    }
                    "pre" => {
                        out.flush(quote > 0);
                        pre = Some(String::new());
                    }
                    "blockquote" => {
                        out.flush(quote > 0);
                        quote += 1;
                    }
                    "caption" | "figcaption" | "dt" | "dd" | "hr" => out.flush(quote > 0),
                    x if BLOCK.contains(&x) => out.flush(quote > 0),
                    _ => {}
                }
            }
            Tok::End(n) => {
                if DROP.contains(&n.as_str()) {
                    drop = drop.saturating_sub(1);
                    continue;
                }
                if drop > 0 {
                    continue;
                }
                match n.as_str() {
                    "title" => in_title = false,
                    "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                        let t = collapse(&out.inline);
                        out.inline.clear();
                        if !t.is_empty() {
                            let level = heading.take().unwrap_or(1);
                            out.block("para", format!("{} {t}", "#".repeat(level)));
                        }
                        heading = None;
                    }
                    "li" => {
                        if let Some(i) = item.take() {
                            emit_item(&mut out, &lists, &i);
                        }
                    }
                    "ul" | "ol" | "menu" => {
                        if let Some(i) = item.take() {
                            emit_item(&mut out, &lists, &i);
                        }
                        lists.pop();
                        // Back in the outer item, if any.
                        if !lists.is_empty() {
                            item = None;
                        }
                    }
                    "td" | "th" => {
                        if let (Some(c), Some(r)) = (cell.take(), row.as_mut()) {
                            r.0.push(collapse(&c).replace('|', "\\|"));
                        }
                    }
                    "tr" => {
                        if let Some((cells, head)) = row.take()
                            && !cells.is_empty()
                        {
                            let n = cells.len();
                            out.block("row", format!("| {} |", cells.join(" | ")));
                            if head && rows_seen == 0 {
                                out.block("row", format!("|{}", " --- |".repeat(n)));
                            }
                            rows_seen += 1;
                        }
                    }
                    "pre" => {
                        if let Some(p) = pre.take() {
                            let body = p.trim_matches('\n');
                            out.block("para", format!("```\n{body}\n```"));
                        }
                    }
                    "blockquote" => {
                        out.flush(true);
                        quote = quote.saturating_sub(1);
                    }
                    "caption" | "figcaption" | "dt" | "dd" => out.flush(quote > 0),
                    x if BLOCK.contains(&x) || x == "table" => out.flush(quote > 0),
                    _ => {}
                }
            }
            Tok::Text(s) => {
                if in_title {
                    title.push_str(s);
                    continue;
                }
                if drop > 0 {
                    continue;
                }
                if let Some(p) = pre.as_mut() {
                    p.push_str(s);
                } else if let Some(c) = cell.as_mut() {
                    c.push_str(s);
                } else if let Some(i) = item.as_mut() {
                    i.push_str(s);
                } else {
                    out.inline.push_str(s);
                }
            }
        }
    }
    if let Some(i) = item.take() {
        emit_item(&mut out, &lists, &i);
    }
    out.flush(false);
    let title = collapse(&title);
    let mut lines = out.lines;
    if !has_h1 && !title.is_empty() {
        lines.insert(0, format!("# {title}"));
        if lines.len() > 1 {
            lines.insert(1, String::new());
        }
    }
    let mut text = lines.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    (text, title)
}

fn emit_item(out: &mut Out, lists: &[(bool, usize)], text: &str) {
    let t = collapse(text);
    if t.is_empty() {
        return;
    }
    let depth = lists.len().saturating_sub(1);
    let (ordered, n) = lists.last().copied().unwrap_or((false, 1));
    let marker = if ordered { format!("{n}.") } else { "-".to_owned() };
    out.block("list", format!("{}{marker} {t}", "  ".repeat(depth)));
}

pub(crate) fn check() -> Result<(), String> {
    let page = "<html><head><title>Pump care</title><style>p{}</style></head><body><nav><a>Home</a></nav>\
        <h2>Descaling</h2><p>Use a <b>mild</b> acid &amp; water.</p><ol><li>Stop the pump.</li><li>Fill it.</li></ol>\
        <table><tr><th>Part</th><th>Torque</th></tr><tr><td>Nut</td><td>12 N m</td></tr></table>\
        <pre>x = 1\n  y = 2</pre><footer>c</footer></body></html>";
    let (text, title) = derive(page);
    let want = "# Pump care\n\n## Descaling\n\nUse a mild acid & water.\n\n1. Stop the pump.\n2. Fill it.\n\n| Part | Torque |\n| --- | --- |\n| Nut | 12 N m |\n\n```\nx = 1\n  y = 2\n```\n";
    if text != want || title != "Pump care" {
        return Err(format!("nmdoc.html: derived {text:?}"));
    }
    Ok(())
}
