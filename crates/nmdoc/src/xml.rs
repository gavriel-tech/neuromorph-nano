//! XML to derived text in Markdown form. An element with a `title` or `heading` child is a
//! section: its title becomes a `#` heading (one level deeper per enclosing section). A leaf
//! element's text becomes a line: prose (four words or more) as it is, a short value as
//! `name: value`. A record (an element whose children are all short leaves) becomes one line
//! `name: child=value; child=value`. Attributes are kept only on elements without text.
//! No DTD is read and no external entity is resolved.
use crate::markup::{Tok, collapse, tokens};

#[derive(Debug, Default)]
struct Node {
    name: String,
    attrs: Vec<(String, String)>,
    text: String,
    kids: Vec<Node>,
}

fn tree(src: &str) -> Node {
    let mut stack: Vec<Node> = vec![Node::default()];
    for t in tokens(src, false, &[]) {
        match t {
            Tok::Start(name, attrs, selfc) => {
                let n = Node { name, attrs, ..Node::default() };
                if selfc {
                    if let Some(top) = stack.last_mut() {
                        top.kids.push(n);
                    }
                } else {
                    stack.push(n);
                }
            }
            Tok::End(name) => {
                // Close up to the matching element; an unmatched end tag is ignored.
                if stack.iter().skip(1).any(|n| n.name == name) {
                    while stack.len() > 1 {
                        let Some(n) = stack.pop() else { break };
                        let done = n.name == name;
                        if let Some(top) = stack.last_mut() {
                            top.kids.push(n);
                        }
                        if done {
                            break;
                        }
                    }
                }
            }
            Tok::Text(s) => {
                if let Some(top) = stack.last_mut() {
                    top.text.push_str(&s);
                    top.text.push(' ');
                }
            }
        }
    }
    while stack.len() > 1 {
        let Some(n) = stack.pop() else { break };
        if let Some(top) = stack.last_mut() {
            top.kids.push(n);
        }
    }
    stack.pop().unwrap_or_default()
}

fn local(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

fn is_title(n: &Node) -> bool {
    matches!(local(&n.name).to_lowercase().as_str(), "title" | "heading" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6")
}

fn emit(n: &Node, depth: usize, lines: &mut Vec<String>) {
    let text = collapse(&n.text);
    let name = local(&n.name);
    if n.kids.is_empty() {
        if !text.is_empty() {
            let line = if text.split_whitespace().count() >= 4 { text } else { format!("{name}: {text}") };
            lines.push(line);
        } else if !n.attrs.is_empty() {
            let a: Vec<String> = n.attrs.iter().map(|(k, v)| format!("{k}={v}")).collect();
            lines.push(format!("{name}: {}", a.join(", ")));
        }
        return;
    }
    if let Some(t) = n.kids.iter().find(|k| is_title(k)) {
        let level = (depth + 1).min(6);
        lines.push(format!("{} {}", "#".repeat(level), collapse(&t.text)));
        if !text.is_empty() {
            lines.push(text);
        }
        for k in n.kids.iter().filter(|k| !is_title(k)) {
            emit(k, depth + 1, lines);
        }
        return;
    }
    let record = n.kids.len() >= 2 && n.kids.iter().all(|k| k.kids.is_empty() && collapse(&k.text).split_whitespace().count() < 4);
    if record && text.is_empty() {
        let mut parts: Vec<String> = n.attrs.iter().map(|(k, v)| format!("{k}={v}")).collect();
        parts.extend(n.kids.iter().map(|k| format!("{}={}", local(&k.name), collapse(&k.text))));
        lines.push(format!("{name}: {}", parts.join("; ")));
        return;
    }
    if !text.is_empty() {
        lines.push(text);
    }
    for k in &n.kids {
        emit(k, depth, lines);
    }
}

/// The derived text of an XML document and its title (the root's title, if any).
#[must_use]
pub fn derive(src: &str) -> (String, String) {
    let root = tree(src);
    let mut lines = Vec::new();
    for k in &root.kids {
        emit(k, 0, &mut lines);
    }
    let title = root.kids.first().and_then(|r| r.kids.iter().find(|k| is_title(k))).map(|t| collapse(&t.text)).unwrap_or_default();
    // A blank line before each heading and each prose line keeps blocks apart.
    let mut out: Vec<String> = Vec::new();
    for l in lines {
        let record = l.split_once(": ").is_some_and(|(k, _)| !k.contains(' '));
        let prev_record = out.last().is_some_and(|p| p.split_once(": ").is_some_and(|(k, _)| !k.contains(' ')));
        if !out.is_empty() && !(record && prev_record) {
            out.push(String::new());
        }
        out.push(l);
    }
    let mut text = out.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    (text, title)
}

pub(crate) fn check() -> Result<(), String> {
    let doc = "<?xml version=\"1.0\"?><!DOCTYPE m [<!ENTITY x SYSTEM \"file:///etc/passwd\">]><manual><title>Pump</title>\
        <section><title>Care</title><para>Flush the pump with clean water each month.</para></section>\
        <parts><part><no>P-1</no><name>Seal</name></part><part><no>P-2</no><name>Nut</name></part></parts><rev date=\"2026\"/></manual>";
    let (text, title) = derive(doc);
    let want = "# Pump\n\n## Care\n\nFlush the pump with clean water each month.\n\npart: no=P-1; name=Seal\npart: no=P-2; name=Nut\nrev: date=2026\n";
    if text != want || title != "Pump" || text.contains("passwd") {
        return Err(format!("nmdoc.xml: derived {text:?}"));
    }
    Ok(())
}
