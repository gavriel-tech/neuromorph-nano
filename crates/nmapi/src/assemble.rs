//! Context delivery (guide 17.4 steps 9 to 11, feature baseline 15.4, Comprehensive Vision
//! 18.5): the reply Claude Code reads is built from the Core Brain's pack in one of the three
//! forms of decision 7 (2026-11-06 decision list), chosen adaptively per query:
//! C `answer`, the validated value packets of the question's named symbol, nothing else
//! (sibling definitions with other values are conflicts and are all kept);
//! B `cited`, the packets the conclusion cites plus every contradicted or obsolete one;
//! A `full`, every admitted packet.
//! Forms B and C use a compact packet encoding (source, text, evidence only when it adds to
//! the text, status only when it is not the reply's default). Every reply states how many
//! packets it left out and how to get them (Context Zoom, guide 5.1). `since` returns only
//! what is newer than the caller's version (guide 17.3).
use crate::forage::Answer;
use nmjson::Value;
use nmsubs::repo::FileInfo;
use nmtype::{Packet, Status};

/// The delivery form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    Adaptive,
    Full,
    Cited,
    Answer,
}
impl Form {
    /// From a tool argument; adaptive when empty.
    /// # Errors
    /// Any other word.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "" | "adaptive" => Ok(Self::Adaptive),
            "full" => Ok(Self::Full),
            "cited" => Ok(Self::Cited),
            "answer" => Ok(Self::Answer),
            o => Err(format!("form {o:?}: adaptive, answer, cited or full")),
        }
    }
    const fn word(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Full => "full",
            Self::Cited => "cited",
            Self::Answer => "answer",
        }
    }
}

/// Packets the cited form sends when a caller asks for it by name: the first of the ranked pack
/// (the emitted answer lists the first three of them). The adaptive form, what a caller naming no
/// form gets, sends the whole pack its token budget selected (`choose`).
/// 2026-11-13 (context compression, measured on CX1 and CX2 from full replies): the first 7
/// packets carry every right line the full pack carries on both sets (45 of 60 and 49 of 60) at
/// about 80% of its tokens; 3 lost 4 and 5 of them.
const CITED: usize = 7;

fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
fn n(x: impl ToString) -> Value {
    Value::Number(x.to_string())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// The obsolete note a recalled packet carries, if any.
#[must_use]
pub fn obsolete(p: &Packet) -> Option<&str> {
    p.get_constraints().iter().find_map(|c| c.strip_prefix("obsolete: "))
}

/// Compact encoding: source, text, evidence only when it adds to the text, status only when
/// it is not Observed, the obsolete note when there is one.
#[must_use]
pub fn compact(p: &Packet) -> Value {
    let mut f = vec![("source", s(&p.provenance().source)), ("text", s(p.text()))];
    let extra: Vec<Value> = p
        .get_evidence()
        .iter()
        .filter(|e| {
            let body = e.split_once(' ').map_or(e.as_str(), |x| x.1);
            !p.text().contains(body.trim()) && **e != p.provenance().source
        })
        .take(2)
        .map(|e| s(e))
        .collect();
    if !extra.is_empty() {
        f.push(("evidence", Value::Array(extra)));
    }
    if p.get_status() != Status::Observed {
        f.push(("status", s(p.get_status().name())));
    }
    if let Some(o) = obsolete(p) {
        f.push(("obsolete", s(o)));
    }
    obj(f)
}

/// The value packets of a symbol the question names (whole-word match on the name after
/// `value:`).
fn values<'a>(question: &str, packets: &'a [Packet]) -> Vec<&'a Packet> {
    let words: Vec<&str> = question.split(|c: char| !(c.is_alphanumeric() || c == '_')).collect();
    packets.iter().filter(|p| p.key().strip_prefix("value:").is_some_and(|name| words.contains(&name))).collect()
}

/// Packets that must never be dropped from a compact reply: contradicted or obsolete.
fn must_keep(p: &Packet) -> bool {
    p.get_status() == Status::Contradicted || obsolete(p).is_some()
}

/// What one reply carries, shared by the JSON and the text renderings.
struct Chosen<'a> {
    word: &'static str,
    full: bool,
    picked: Vec<&'a Packet>,
    answer: String,
    left: usize,
    conflict: bool,
}
enum Outcome<'a> {
    Withheld,
    Unchanged(usize),
    Reply(Chosen<'a>),
}

/// Picks the packets a reply carries: the form, the `since` filter and the cited cut.
fn choose<'a>(question: &str, a: &'a Answer, form: Form, since: Option<u64>) -> Outcome<'a> {
    let all = &a.pack.packets;
    if a.pack.withheld {
        return Outcome::Withheld;
    }
    // Incremental: only newer packets, plus obsolete marks (a stale claim must be seen); a kept
    // conflict was delivered with the reply that first carried it and is not resent.
    let fresh: Vec<&Packet> = all.iter().filter(|p| since.is_none_or(|v| p.provenance().version > v) || obsolete(p).is_some()).collect();
    if since.is_some() && fresh.is_empty() {
        return Outcome::Unchanged(all.len());
    }
    let vals = values(question, all);
    let chosen = match form {
        Form::Adaptive if !vals.is_empty() => Form::Answer,
        Form::Adaptive => Form::Cited,
        f => f,
    };
    let picked: Vec<&Packet> = match chosen {
        Form::Full => fresh.clone(),
        // Adaptive: every packet the caller's budget selected, the named values first; the pack
        // is not cut again by a count.
        _ if form == Form::Adaptive => {
            let mut v: Vec<&Packet> = fresh.iter().copied().filter(|p| vals.iter().any(|x| std::ptr::eq(*x, *p))).collect();
            let rest: Vec<&Packet> = fresh.iter().copied().filter(|p| !v.iter().any(|x| std::ptr::eq(*x, *p))).collect();
            v.extend(rest);
            v
        }
        Form::Answer if !vals.is_empty() => {
            fresh.iter().copied().filter(|p| vals.iter().any(|v| std::ptr::eq(*v, *p)) || must_keep(p)).collect()
        }
        _ => fresh.iter().copied().enumerate().filter(|(i, p)| *i < CITED || must_keep(p)).map(|(_, p)| p).collect(),
    };
    let answer = if chosen == Form::Answer && !vals.is_empty() {
        let mut texts: Vec<&str> = vals.iter().map(|p| p.text()).collect();
        texts.dedup();
        texts.join(" | ")
    } else {
        a.emitted.text().lines().next().unwrap_or("").to_owned()
    };
    let conflict = picked.iter().any(|p| p.get_status() == Status::Contradicted);
    let left = all.len().saturating_sub(picked.len());
    let word = if chosen == Form::Answer && vals.is_empty() { "cited" } else { chosen.word() };
    Outcome::Reply(Chosen { word, full: chosen == Form::Full, picked, answer, left, conflict })
}

/// Build the reply for `form` as JSON (the earlier formatter, kept for programs and for
/// comparison). `version` is the substrate version the caller passes back as `since`; `full`
/// is the full packet encoding, passed in by the caller.
#[must_use]
pub fn reply(question: &str, a: &Answer, form: Form, since: Option<u64>, version: u64, full: &dyn Fn(&Packet) -> Value) -> Value {
    let mut head = vec![("version", n(version))];
    if let Some(e) = &a.episode {
        head.push(("episode", s(e)));
    }
    match choose(question, a, form, since) {
        Outcome::Withheld => {
            head.extend([("withheld", Value::Bool(true)), ("reason", s(&a.pack.reason))]);
        }
        Outcome::Unchanged(all) => {
            head.extend([("unchanged", Value::Bool(true)), ("left_out", n(all))]);
        }
        Outcome::Reply(c) => {
            head.extend([
                ("form", s(c.word)),
                ("answer", s(&c.answer)),
                ("status", s(if c.conflict { "conflict kept" } else { "observed" })),
            ]);
            if !a.pack.floor_met {
                head.extend([("confidence", s("low")), ("floor_met", Value::Bool(false))]);
            }
            let enc: Vec<Value> = c.picked.iter().map(|p| if c.full { full(p) } else { compact(p) }).collect();
            head.push(("packets", Value::Array(enc)));
            if c.left > 0 {
                head.extend([("left_out", n(c.left)), ("more", s("form=full"))]);
            }
        }
    }
    obj(head)
}

/// The reply format: plain text (default) or the earlier JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Text,
    Json,
}
impl Format {
    /// From a tool argument; text when empty.
    /// # Errors
    /// Any other word.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "" | "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            o => Err(format!("format {o:?}: text or json")),
        }
    }
}

/// `path:line` of a packet's source, when it names a line.
fn place(source: &str) -> Option<(&str, usize)> {
    let (path, line) = source.rsplit_once(':')?;
    line.parse().ok().filter(|l| *l > 0).map(|l| (path, l))
}

fn file_text<'a>(files: &'a [FileInfo], path: &str) -> Option<&'a str> {
    files.iter().find(|f| f.path == path).map(|f| f.text.as_str())
}

/// What a span of a document names after its place: its heading path and, for a slice of a table
/// that leaves out the header, the header line itself (` | header line 1: part_no,name,...`), so
/// every table slice carries its header, and for a file read generally how it was read. Empty for
/// code.
fn doc_head(files: &[FileInfo], path: &str, first: usize) -> String {
    let Some(f) = files.iter().find(|f| f.path == path).filter(|f| !f.is_code()) else { return String::new() };
    let mut out = String::new();
    if let Some(how) = f.general {
        out.push_str(match how {
            "raw" => " | read generally: text runs recovered from the bytes, no headings or pages",
            "binary" => " | binary file: kept by its metadata only (name, kind, size), no text from its bytes",
            "container" => " | read generally: container listing",
            "gzip" => " | read generally: inflated from gzip",
            _ => " | read generally: container member",
        });
    }
    let heading = nmsubs::doc::heading_of(f, first);
    if !heading.is_empty() {
        out.push_str(" | ");
        out.push_str(&heading);
    }
    if let Some(b) = nmtype::doc::block_at(&f.blocks, first).and_then(|i| f.blocks.get(i))
        && let nmtype::doc::BlockKind::Table { head } = b.kind
        && head > 0
    {
        let head_end = b.first + usize::from(head) - 1;
        if first > head_end {
            let header = file_lines(&f.text, b.first, head_end).join(" ");
            let lines = if head == 1 { format!("line {}", b.first) } else { format!("lines {}-{head_end}", b.first) };
            out.push_str(&format!(" | header {lines}: {header}"));
        }
    }
    out
}

/// The file lines `first..=last` (1-based), without carriage returns.
fn file_lines(text: &str, first: usize, last: usize) -> Vec<&str> {
    text.split('\n').skip(first.saturating_sub(1)).take((last + 1).saturating_sub(first)).map(|l| l.trim_end_matches('\r')).collect()
}

/// The span a packet carries when its text is the file's own lines from its source line on:
/// (path, first, last), lines compared without surrounding spaces (a use or grep line is sent
/// trimmed; the span then shows the file's own line). Anything else (a symbol summary, a value,
/// a memory) is not a span.
fn verbatim(p: &Packet, files: &[FileInfo]) -> Option<(String, usize, usize)> {
    let (path, first) = place(&p.provenance().source)?;
    // Split on every newline: a text ending in one carries an empty last line of the file.
    if p.text().is_empty() {
        return None;
    }
    let want: Vec<&str> = p.text().split('\n').map(|l| l.trim_end_matches('\r')).collect();
    let last = first + want.len() - 1;
    let have = file_lines(file_text(files, path)?, first, last);
    (have.len() == want.len() && have.iter().zip(&want).all(|(h, w)| h.trim() == w.trim())).then(|| (path.to_owned(), first, last))
}

/// An evidence line that only restates a place in `path` (`path:12`, `path:12 lines 12..41`,
/// `path:12 line 15`); returns the single line it names, if any, as a match mark.
fn restates(e: &str, path: &str) -> Option<Option<usize>> {
    let rest = e.strip_prefix(path)?.strip_prefix(':')?;
    let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 {
        return None;
    }
    let rest = rest.get(digits..)?.trim();
    if rest.is_empty() {
        return Some(None);
    }
    if let Some(r) = rest.strip_prefix("lines ") {
        return r.chars().all(|c| c.is_ascii_digit() || c == '.').then_some(None);
    }
    rest.strip_prefix("line ").and_then(|r| r.parse::<usize>().ok()).map(Some)
}

/// Evidence that adds to what the reply shows (as `compact`), with place restatements dropped;
/// the single lines those restatements name come back as match marks.
fn evidence_of(p: &Packet, path: &str) -> (Vec<String>, Vec<usize>) {
    let mut marks = Vec::new();
    let mut extra = Vec::new();
    for e in p.get_evidence() {
        // A piece's fields are read by the judge, not shown.
        if e.starts_with(nmbrain::synthesis::PIECE) {
            continue;
        }
        if let Some(mark) = restates(e, path) {
            marks.extend(mark);
            continue;
        }
        let body = e.split_once(' ').map_or(e.as_str(), |x| x.1);
        if !p.text().contains(body.trim()) && *e != p.provenance().source && extra.len() < 2 {
            extra.push(e.clone());
        }
    }
    (extra, marks)
}

/// Display order of statuses: the strongest claim a merged span holds is shown.
const fn rank(st: Status) -> u8 {
    match st {
        Status::Contradicted => 0,
        Status::Verified => 1,
        Status::Observed => 2,
        Status::Inferred => 3,
        Status::Hypothesized => 4,
        Status::Unknown => 5,
    }
}

/// One merged run of file lines: every packet of the same file whose lines overlap or touch.
struct Span {
    path: String,
    first: usize,
    last: usize,
    status: Status,
    marks: Vec<usize>,
    evidence: Vec<String>,
    obsolete: Vec<String>,
}
impl Span {
    fn take(&mut self, o: Self) {
        self.first = self.first.min(o.first);
        self.last = self.last.max(o.last);
        if rank(o.status) < rank(self.status) {
            self.status = o.status;
        }
        self.marks.extend(o.marks);
        for e in o.evidence {
            if !self.evidence.contains(&e) {
                self.evidence.push(e);
            }
        }
        for e in o.obsolete {
            if !self.obsolete.contains(&e) {
                self.obsolete.push(e);
            }
        }
    }
    fn touches(&self, path: &str, first: usize, last: usize) -> bool {
        self.path == path && first <= self.last + 1 && last + 1 >= self.first
    }
}
enum Item<'a> {
    Span(Span),
    Note(&'a Packet),
}

/// Packets in order, the same file's overlapping, contained or touching line runs merged into
/// one span at the place of its first packet; a one-line packet inside a longer span becomes a
/// match mark on that line.
fn spans<'a>(picked: &[&'a Packet], files: &[FileInfo]) -> Vec<Item<'a>> {
    let mut items: Vec<Item<'a>> = Vec::new();
    for p in picked {
        let Some((path, first, last)) = verbatim(p, files) else {
            items.push(Item::Note(p));
            continue;
        };
        let (evidence, mut marks) = evidence_of(p, &path);
        if first == last {
            marks.push(first);
        }
        let new = Span {
            path,
            first,
            last,
            status: p.get_status(),
            marks,
            evidence,
            obsolete: obsolete(p).map(str::to_owned).into_iter().collect(),
        };
        let at = items.iter().position(|it| matches!(it, Item::Span(x) if x.touches(&new.path, new.first, new.last)));
        let Some(i) = at else {
            items.push(Item::Span(new));
            continue;
        };
        if let Some(Item::Span(x)) = items.get_mut(i) {
            x.take(new);
        }
        // The grown span may now reach later spans of the same file: fold them in too.
        while let Some(Item::Span(x)) = items.get(i) {
            let (path, first, last) = (x.path.clone(), x.first, x.last);
            let Some(j) = items
                .iter()
                .enumerate()
                .skip(i + 1)
                .find_map(|(j, it)| matches!(it, Item::Span(y) if y.touches(&path, first, last)).then_some(j))
            else {
                break;
            };
            if let Item::Span(y) = items.remove(j)
                && let Some(Item::Span(x)) = items.get_mut(i)
            {
                x.take(y);
            }
        }
    }
    items
}

const fn status_of(it: &Item) -> Status {
    match it {
        Item::Span(x) => x.status,
        Item::Note(p) => p.get_status(),
    }
}

/// The reply as plain text: a head line, the answer, the reply's status, then each span as an
/// `@@ path:first-last` header followed by exactly those file lines, and each other packet as a
/// `~~ source kind` header followed by its text (a line count on the header when it is not one).
/// A status is printed on a header only when it differs from the reply's. `debug` adds the
/// diagnostics as one JSON line at the end.
#[must_use]
#[expect(clippy::too_many_arguments, reason = "the JSON reply's arguments plus the files and the diagnostics")]
pub fn reply_text(
    question: &str,
    a: &Answer,
    form: Form,
    since: Option<u64>,
    version: u64,
    files: &[FileInfo],
    debug: Option<&Value>,
) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let episode = a.episode.as_deref().map(|e| format!("; episode {e}")).unwrap_or_default();
    let c = match choose(question, a, form, since) {
        Outcome::Withheld => {
            let _ = write!(out, "nm-pack v{version} withheld{episode}\nreason: {}\n", a.pack.reason);
            return with_debug(out, debug);
        }
        Outcome::Unchanged(all) => {
            let _ = writeln!(out, "nm-pack v{version} unchanged since v{}; {all} packets left out{episode}", since.unwrap_or_default());
            return with_debug(out, debug);
        }
        Outcome::Reply(c) => c,
    };
    let items = spans(&c.picked, files);
    let left = if c.left > 0 { format!("; {} left out: form full", c.left) } else { String::new() };
    let _ = writeln!(out, "nm-pack v{version} {} {} of {} packets{left}{episode}", c.word, c.picked.len(), a.pack.packets.len());
    let _ = writeln!(out, "answer: {}", c.answer);
    if c.full {
        for l in a.emitted.text().lines().skip(1) {
            let _ = writeln!(out, "  {l}");
        }
    }
    // The reply's status is the one most items hold; headers name only the others.
    let mut counts: Vec<(Status, usize)> = Vec::new();
    for it in &items {
        let st = status_of(it);
        match counts.iter_mut().find(|x| x.0 == st) {
            Some(x) => x.1 += 1,
            None => counts.push((st, 1)),
        }
    }
    let default = counts
        .iter()
        .fold(None::<(Status, usize)>, |best, x| if best.is_none_or(|b| x.1 > b.1) { Some(*x) } else { best })
        .map_or(Status::Observed, |x| x.0);
    let _ = writeln!(out, "status: {}{}", default.name(), if c.conflict { "; conflict kept" } else { "" });
    if !a.pack.floor_met {
        let _ = writeln!(out, "confidence: low; floor_met: false");
    }
    if c.full {
        if a.pack.reason != c.answer {
            let _ = writeln!(out, "reason: {}", a.pack.reason);
        }
        let _ = writeln!(out, "tokens: {}", a.pack.tokens);
        for q in &a.questions {
            let _ = writeln!(out, "? {}", q.text().replace('\n', " "));
        }
    }
    for it in &items {
        let st = status_of(it);
        let st = if st == default { String::new() } else { format!(" {}", st.name()) };
        match it {
            Item::Span(x) => {
                let mut marks: Vec<usize> =
                    x.marks.iter().copied().filter(|m| *m >= x.first && *m <= x.last && x.first != x.last).collect();
                marks.sort_unstable();
                marks.dedup();
                let marks = if marks.is_empty() {
                    String::new()
                } else {
                    format!(" (matched {})", marks.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
                };
                // A code piece's heading (`in <signature>` for a part after the first, `in <type>`
                // for a member) is shown after the span, as a document's heading is.
                let (ins, rest): (Vec<&String>, Vec<&String>) = x.evidence.iter().partition(|e| e.starts_with("in "));
                let ev = if rest.is_empty() { String::new() } else { format!("; evidence: {}", rest.iter().map(|e| e.as_str()).collect::<Vec<_>>().join(" | ")) };
                let ob = if x.obsolete.is_empty() { String::new() } else { format!("; obsolete: {}", x.obsolete.join(" | ")) };
                let mut head = doc_head(files, &x.path, x.first);
                if let Some(i) = ins.first() {
                    head.push_str(" | ");
                    head.push_str(i);
                }
                let _ = writeln!(out, "@@ {}:{}-{}{st}{marks}{head}{ev}{ob}", x.path, x.first, x.last);
                for l in file_text(files, &x.path).map(|t| file_lines(t, x.first, x.last)).unwrap_or_default() {
                    let _ = writeln!(out, "{l}");
                }
            }
            Item::Note(p) => {
                let src = &p.provenance().source;
                let (ev, _) = evidence_of(p, place(src).map_or(src.as_str(), |x| x.0));
                let ev = if ev.is_empty() { String::new() } else { format!("; evidence: {}", ev.join(" | ")) };
                let ob = obsolete(p).map(|o| format!("; obsolete: {o}")).unwrap_or_default();
                let lines: Vec<&str> = p.text().split('\n').collect();
                let count = if lines.len() == 1 { String::new() } else { format!(" ({} lines)", lines.len()) };
                let _ = writeln!(out, "~~ {src} {}{st}{count}{ev}{ob}", p.kind());
                for l in lines {
                    let _ = writeln!(out, "{l}");
                }
            }
        }
    }
    with_debug(out, debug)
}

fn with_debug(mut out: String, debug: Option<&Value>) -> String {
    if let Some(d) = debug {
        out.push_str("debug: ");
        out.push_str(&d.encode().unwrap_or_default());
        out.push('\n');
    }
    out
}

/// Section check on a built-in pack.
/// # Errors
/// One line naming what failed.
pub fn check() -> Result<(), String> {
    use nmtype::{Pack, Provenance};
    let mk = |k: &str, t: &str, src: &str, v: u64| {
        Packet::new("symbol", k, t, Provenance::new(src, "x", v, 0)).status(Status::Observed).evidence(vec![format!("{src} {t}")])
    };
    let packets = vec![
        mk("symbol:gb@a.rs", "fn gb defined at a.rs:3", "a.rs:3", 1),
        mk("value:GB", "GB = 1_000", "b.rs:7", 1),
        mk("value:GB", "GB = 2_000", "c.rs:9", 2).status(Status::Contradicted),
        mk("grep:x", "x", "d.rs:1", 1),
        mk("grep:y", "y", "e.rs:1", 1),
        Packet::new("memory", "memory:e1", "earlier question", Provenance::new("memory:e1", "m", 1, 0))
            .status(Status::Unknown)
            .constraints(vec!["obsolete: b.rs changed at version 2".into()]),
    ];
    let emitted = Packet::new("pack", "k", "answered from observed packets\nb.rs:7 GB", Provenance::new("x", "x", 1, 0));
    let a = Answer::for_check(Pack { task: 1, packets, withheld: false, reason: "answered".into(), tokens: 0, floor_met: true }, emitted);
    let full = |p: &Packet| s(p.text());
    let c = reply("What value is the constant GB set to?", &a, Form::Adaptive, None, 2, &full);
    let count = |v: &Value| v.get("packets").and_then(Value::array).map_or(0, <[Value]>::len);
    // Adaptive with a named value: the value packets lead and every other packet follows.
    let first = c.get("packets").and_then(Value::array).ok().and_then(|p| p.first()).and_then(|p| p.get("text").ok()).and_then(|t| t.string().ok()).unwrap_or_default().to_owned();
    if c.get("form").and_then(Value::string).ok() != Some("answer") || count(&c) != 6 || !first.starts_with("GB = ") {
        return Err(format!("form C: {}", c.encode().unwrap_or_default()));
    }
    let ans = reply("What value is the constant GB set to?", &a, Form::Answer, None, 2, &full);
    if count(&ans) != 3 {
        return Err(format!("form answer: {}", ans.encode().unwrap_or_default()));
    }
    let b = reply("Where is parse_thing used?", &a, Form::Adaptive, None, 2, &full);
    if b.get("form").and_then(Value::string).ok() != Some("cited") || count(&b) != 6 {
        return Err(format!("form B: {}", b.encode().unwrap_or_default()));
    }
    let f = reply("Where is parse_thing used?", &a, Form::Full, None, 2, &full);
    if count(&f) != 6 || f.get("left_out").is_ok() {
        return Err("form A must carry every packet".into());
    }
    let inc = reply("What value is the constant GB set to?", &a, Form::Answer, Some(1), 2, &full);
    if count(&inc) != 2 {
        return Err(format!("since keeps only newer or must-keep packets: {}", inc.encode().unwrap_or_default()));
    }
    Form::parse("brief").err().ok_or("unknown form accepted")?;
    Format::parse("yaml").err().ok_or("unknown format accepted")?;
    text_check()
}

/// Text form: a trimmed one-line packet inside a chunk of the same file and a touching chunk merge
/// into one span at the first packet's place with the line marked; every carried line keeps its
/// number; a packet that is not file lines keeps its own text; a withheld pack says why.
fn text_check() -> Result<(), String> {
    use nmtype::{Pack, Provenance};
    let lines = |a: usize, b: usize| (a..=b).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
    let mut file = FileInfo::new("s.rs", &format!("{}\n", lines(1, 12)));
    file.hash = String::new();
    let chunk = |a: usize, b: usize| {
        Packet::new("chunk", &format!("chunk:s.rs:{a}"), &lines(a, b), Provenance::new(&format!("s.rs:{a}"), "x", 1, 0))
            .status(Status::Inferred)
            .evidence(vec![format!("s.rs:{a} lines {a}..{b}")])
    };
    let grep = Packet::new("grep", "grep:s.rs:3", "  line 3", Provenance::new("s.rs:3", "x", 1, 0))
        .status(Status::Observed)
        .evidence(vec!["s.rs:3 line 3".into()]);
    let symbol = Packet::new("symbol", "symbol:gb@a.rs", "fn gb defined at a.rs:3", Provenance::new("a.rs:3", "x", 1, 0))
        .status(Status::Observed)
        .evidence(vec!["a.rs:3 fn gb".into()]);
    let packets = vec![chunk(1, 5), symbol, grep, chunk(6, 8), chunk(11, 12)];
    let emitted = Packet::new("pack", "k", "answered", Provenance::new("x", "x", 1, 0));
    let a = Answer::for_check(
        Pack { task: 1, packets, withheld: false, reason: "answered".into(), tokens: 0, floor_met: true },
        emitted.clone(),
    );
    let t = reply_text("where is line", &a, Form::Full, None, 2, std::slice::from_ref(&file), None);
    let want = format!(
        "@@ s.rs:1-8 (matched 3)\n{}\n~~ a.rs:3 symbol\nfn gb defined at a.rs:3\n@@ s.rs:11-12 inferred\n{}\n",
        lines(1, 8),
        lines(11, 12)
    );
    if !t.starts_with("nm-pack v2 full 5 of 5 packets\n") || !t.ends_with(&want) {
        return Err(format!("text form: {t}"));
    }
    let held = Answer::for_check(
        Pack { task: 1, packets: Vec::new(), withheld: true, reason: "withheld: nothing observed".into(), tokens: 0, floor_met: true },
        emitted,
    );
    if reply_text("q", &held, Form::Adaptive, None, 2, &[file], None) != "nm-pack v2 withheld\nreason: withheld: nothing observed\n" {
        return Err("text form: withheld reply".into());
    }
    Ok(())
}
