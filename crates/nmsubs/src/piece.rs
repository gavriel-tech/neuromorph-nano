//! Pieces: the one unit every reader takes (the drones, the dense lane, the judge and the pack).
//! A drone takes one piece of the raw input; a piece of code is one definition: its doc comments,
//! its signature and its body. Kinds: function, method (named by type and item), type, constant
//! (a run of small constants and statics is one constant group), module, macro, imports, file
//! notes; code with no definitions keeps its blocks (statements at the top level, cut where a
//! statement ends). Documents keep their structure pieces (`doc::segments`).
//!
//! Sizes come from the two windows that read pieces: a piece is at most the judge's window
//! (`JUDGE_WINDOW`, smaller than a tentacle's window, so a tentacle always holds whole pieces). A
//! longer definition is split at its top-level statement boundaries (inside a statement too large
//! alone, at the boundaries one level in; at line ends only when no boundary is left), and every
//! part after the first is headed by the definition's signature. Neighbours under `SMALL` tokens
//! with one parent are merged. No piece crosses a definition: each line belongs to exactly one
//! piece and each piece to at most one definition, or to a run of small ones under one parent.
//!
//! Each piece records its name path, signature and kind, and the identifiers it defines, assigns
//! and uses (with use counts and the qualifier a path names, `connect::record`), whole and split
//! into stemmed parts (`Meta`). The reference graph reads them (`refgraph`).
use crate::repo::{ImplBlock, Symbol};

/// Tokens the judge reads of one piece (characters / 4, the Nano's token rule).
pub const JUDGE_WINDOW: usize = 384;
/// Tokens of one tentacle's window.
pub const TENTACLE_WINDOW: usize = crate::lens::BUDGET;
/// Largest piece: the smaller of the two windows that read a piece whole.
pub const MAX: usize = if JUDGE_WINDOW < TENTACLE_WINDOW { JUDGE_WINDOW } else { TENTACLE_WINDOW };
/// Neighbours under this many tokens with one parent are merged: a tenth of the judge's window,
/// below which a piece carries too little to be judged alone.
pub const SMALL: usize = JUDGE_WINDOW / 10;
/// Deepest nesting a split looks into for statement boundaries before it cuts at line ends.
const MAX_LEVELS: usize = 6;

/// Whether code is cut into pieces: off unless `NANO_PIECES=on`, so code keeps `CHUNK`-line
/// windows until the piece model passes its measure against them on the plain path.
#[must_use]
pub fn on() -> bool {
    std::env::var("NANO_PIECES").is_ok_and(|v| v == "on")
}

/// What a piece is about, read from its text and the definitions in it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Meta {
    /// Name path: `Type::item` for a member, the item's name at the top level, names joined by
    /// `, ` for a merged group, the file's place (`notes`, `imports`) for the rest.
    pub name: String,
    /// The definition's head (its first lines up to the body), trimmed.
    pub signature: String,
    /// The type, trait, class or module the piece sits in (empty at the top of a file).
    pub parent: String,
    /// Names of the definitions whose first line is in the piece.
    pub defines: Vec<String>,
    /// Names given a value in the piece: fields, defaults, bindings, keyword arguments.
    pub assigns: Vec<String>,
    /// Identifiers the piece uses (comments and strings left out) with their counts, names it
    /// defines left out.
    pub uses: Vec<(String, u32)>,
    /// Qualified uses: (qualifier, name) for `a::b` (and `a.b` outside Rust).
    pub paths: Vec<(String, String)>,
    /// Traits or bases the piece implements or extends.
    pub implements: Vec<String>,
    /// Names the piece uses only after a `.` (a field or method of a value whose type it does
    /// not name).
    pub members: Vec<String>,
    /// Stemmed parts (split on case and underscores) of what it defines, then of what it uses.
    pub parts: Vec<String>,
    /// The file's own notes (its opening comment lines), shared by every piece of the file: the
    /// file's field in field-weighted scoring, so a file is weighed first and its pieces inside it.
    pub file_notes: std::sync::Arc<str>,
}

/// One piece of a code file: its lines, its kind, its heading (the signature for a part after
/// the first, the name path for a member) and what it is about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cut {
    pub start: usize,
    pub end: usize,
    pub kind: &'static str,
    pub heading: String,
    pub meta: Meta,
}

/// One token of a code file: text, line, and whether it is an identifier, punctuation or other
/// (number, string).
#[derive(Clone, Debug)]
pub struct Tk {
    pub text: String,
    pub line: usize,
    pub class: TkClass,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TkClass {
    Ident,
    Punct,
    Other,
}

/// Tokens of a code file: the Rust scanner for Rust, the generic lexer for every other language.
#[must_use]
pub fn tokens(text: &str, decoder: &str, rust: Option<&[(String, usize)]>) -> Vec<Tk> {
    let class_of = |t: &str| match t.chars().next() {
        Some(c) if c.is_alphabetic() || c == '_' => TkClass::Ident,
        Some(c) if c.is_ascii_digit() => TkClass::Other,
        _ => TkClass::Punct,
    };
    if decoder == "rust" {
        let own;
        let toks = match rust {
            Some(t) => t,
            None => {
                own = nmlex::tokens(text, true);
                &own
            }
        };
        return toks.iter().map(|(t, l)| Tk { class: class_of(t), text: t.clone(), line: *l }).collect();
    }
    let lang = nmlex::lang::by_id(decoder).unwrap_or(&nmlex::lang::GENERIC);
    nmlex::lang_lex::lex(text, lang)
        .into_iter()
        .map(|t| {
            let class = match t.kind {
                nmlex::lang_lex::Kind::Ident => TkClass::Ident,
                nmlex::lang_lex::Kind::Punct => TkClass::Punct,
                _ => TkClass::Other,
            };
            Tk { text: t.text, line: t.line, class }
        })
        .collect()
}

/// Per line (1-based): bracket depth before its first token (`None`: no token on the line), depth
/// after its last token, and its indentation.
struct Shape<'a> {
    lines: Vec<&'a str>,
    first: Vec<Option<i32>>,
    end: Vec<i32>,
    /// Character count + 1 of each line, prefix-summed (index 0 is 0).
    pre: Vec<usize>,
}

impl<'a> Shape<'a> {
    fn new(text: &'a str, toks: &[Tk]) -> Self {
        let lines: Vec<&str> = text.lines().collect();
        let n = lines.len();
        let mut first = vec![None; n + 2];
        let mut end = vec![i32::MIN; n + 2];
        let mut depth = 0_i32;
        for t in toks {
            if t.line > n {
                break;
            }
            if let Some(f) = first.get_mut(t.line)
                && f.is_none()
            {
                *f = Some(depth);
            }
            if t.class == TkClass::Punct {
                match t.text.as_str() {
                    "{" | "(" | "[" => depth += 1,
                    "}" | ")" | "]" => depth -= 1,
                    _ => {}
                }
            }
            if let Some(e) = end.get_mut(t.line) {
                *e = depth;
            }
        }
        let mut carry = 0;
        for e in end.iter_mut() {
            if *e == i32::MIN {
                *e = carry;
            } else {
                carry = *e;
            }
        }
        let mut pre = Vec::with_capacity(n + 1);
        pre.push(0);
        for l in &lines {
            pre.push(pre.last().copied().unwrap_or(0) + l.chars().count() + 1);
        }
        Self { lines, first, end, pre }
    }
    fn n(&self) -> usize {
        self.lines.len()
    }
    fn line(&self, i: usize) -> &str {
        self.lines.get(i.wrapping_sub(1)).copied().unwrap_or("")
    }
    fn indent(&self, i: usize) -> usize {
        let l = self.line(i);
        l.len() - l.trim_start().len()
    }
    fn blank(&self, i: usize) -> bool {
        self.line(i).trim().is_empty()
    }
    fn first(&self, i: usize) -> Option<i32> {
        self.first.get(i).copied().flatten()
    }
    fn end(&self, i: usize) -> i32 {
        self.end.get(i).copied().unwrap_or(0)
    }
    /// Tokens of lines `a..=b` (characters / 4).
    fn tokens(&self, a: usize, b: usize) -> usize {
        if b < a {
            return 0;
        }
        (self.pre.get(b.min(self.n())).copied().unwrap_or(0).saturating_sub(self.pre.get(a.saturating_sub(1)).copied().unwrap_or(0))) / 4
    }
    /// The last line at or before `i` holding a token, not before `floor`.
    fn code_before(&self, i: usize, floor: usize) -> Option<usize> {
        (floor..i).rev().find(|j| self.first(*j).is_some())
    }
}

/// Statement boundaries in `a..=b` at bracket depth `d` and indentation `ind`: a line starting at
/// that depth and indentation right after a line that ended at that depth.
fn boundaries(sh: &Shape<'_>, a: usize, b: usize, d: i32, ind: usize) -> Vec<usize> {
    ((a + 1)..=b)
        .filter(|&l| {
            sh.first(l) == Some(d) && sh.indent(l) == ind && sh.code_before(l, a).is_some_and(|p| sh.end(p) == d)
        })
        .collect()
}

/// Lines `a..=b` cut at line ends into parts of at most `budget` tokens and `CHUNK` lines.
fn by_lines(sh: &Shape<'_>, a: usize, b: usize, budget: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut s = a;
    while s <= b {
        let mut e = s;
        while e < b && e + 1 - s < crate::CHUNK && sh.tokens(s, e + 1) <= budget {
            e += 1;
        }
        out.push((s, e));
        s = e + 1;
    }
    out
}

/// The line that opens the body of a definition whose head is line `head` at depth `d0`: the first
/// line from `head` on that leaves one bracket open and holds `{`, or ends with `:` at `d0` (an
/// indentation language); `head` when there is none. A signature spread over several lines
/// (parameters one per line) stays whole above it.
fn body_open(sh: &Shape<'_>, head: usize, b: usize, d0: i32) -> usize {
    (head..=b)
        .find(|l| {
            let t = sh.line(*l).trim_end();
            (sh.end(*l) == d0 + 1 && t.contains('{')) || (sh.end(*l) == d0 && t.ends_with(':'))
        })
        .unwrap_or(head)
}

/// Cut lines `a..=b` into parts of at most `budget` tokens at statement boundaries: the body's
/// statements first (one bracket level in from the line that opens the body, or one indentation in
/// for an indentation language); `same` looks at the range's own level first (a block of
/// top-level statements). `def` is the definition's own line when the range is one definition: it
/// heads the range (doc comments and attributes above it are not its head) and every cut falls
/// below the line that opens its body, so the first part holds the whole signature. Groups are
/// packed greedily; a group too large alone is cut one level in.
fn split(sh: &Shape<'_>, (a, b): (usize, usize), budget: usize, same: bool, def: Option<usize>, level: usize) -> Vec<(usize, usize)> {
    if sh.tokens(a, b) <= budget || a >= b {
        return vec![(a, b)];
    }
    if level >= MAX_LEVELS {
        return by_lines(sh, a, b, budget);
    }
    let first_code = def.filter(|d| *d >= a && *d <= b && sh.first(*d).is_some()).or_else(|| (a..=b).find(|l| sh.first(*l).is_some()));
    let Some(head) = first_code else { return by_lines(sh, a, b, budget) };
    let d0 = sh.first(head).unwrap_or(0);
    let open = if def.is_some() { body_open(sh, head, b, d0) } else { head };
    let inner = || -> Vec<usize> {
        // One bracket level in: the first line inside at that depth gives the indentation.
        let start = ((open + 1)..=b).find(|l| sh.first(*l) == Some(d0 + 1));
        let mut v = start.map_or_else(Vec::new, |s| boundaries(sh, s, b, d0 + 1, sh.indent(s)));
        if let Some(s) = start {
            v.insert(0, s);
        }
        if v.len() > 1 {
            return v;
        }
        // One indentation in at the same depth (indentation languages).
        let start = ((open + 1)..=b).find(|l| sh.first(*l) == Some(d0) && sh.indent(*l) > sh.indent(head));
        let mut v = start.map_or_else(Vec::new, |s| boundaries(sh, s, b, d0, sh.indent(s)));
        if let Some(s) = start {
            v.insert(0, s);
        }
        v
    };
    let mut cuts = if same { boundaries(sh, head, b, d0, sh.indent(head)) } else { Vec::new() };
    if cuts.is_empty() {
        cuts = inner();
    }
    cuts.retain(|c| *c > a && *c <= b && *c > open);
    cuts.dedup();
    if cuts.is_empty() {
        return by_lines(sh, a, b, budget);
    }
    let mut groups = Vec::new();
    let mut s = a;
    for c in cuts {
        groups.push((s, c - 1));
        s = c;
    }
    groups.push((s, b));
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut cur: Option<(usize, usize)> = None;
    for (gs, ge) in groups {
        if sh.tokens(gs, ge) > budget {
            if let Some(c) = cur.take() {
                out.push(c);
            }
            out.extend(split(sh, (gs, ge), budget, true, None, level + 1));
            continue;
        }
        cur = match cur {
            Some((cs, _)) if sh.tokens(cs, ge) <= budget => Some((cs, ge)),
            Some(c) => {
                out.push(c);
                Some((gs, ge))
            }
            None => Some((gs, ge)),
        };
    }
    if let Some(c) = cur {
        out.push(c);
    }
    out
}

/// True for kinds whose body is one definition however much it holds (inner items included).
fn fn_like(kind: &str) -> bool {
    matches!(kind, "fn" | "function" | "method" | "macro" | "def" | "filter" | "procedure" | "constructor" | "func" | "sub")
}

/// The piece kind of a definition kind.
fn kind_of(kind: &str, member: bool) -> &'static str {
    match kind {
        "fn" | "function" | "def" | "func" | "sub" | "procedure" | "filter" => {
            if member {
                "method"
            } else {
                "function"
            }
        }
        "method" | "constructor" => "method",
        "struct" | "enum" | "union" | "type" | "typedef" | "trait" | "class" | "interface" | "impl" | "record" | "object" => "type",
        "const" | "static" | "define" | "variable" | "var" | "let" | "constant" => "constant",
        "mod" | "module" | "namespace" | "package" => "module",
        "macro" => "macro",
        "table" => "table",
        "selector" | "keyframes" => "rule",
        _ => "definition",
    }
}

/// A line that names the file's purpose (a note) or brings in other code (an import).
fn line_class(l: &str) -> &'static str {
    let t = l.trim_start();
    let import = ["use ", "pub use ", "pub(crate) use ", "import ", "from ", "#include", "extern crate", "require(", "package ", "using "]
        .iter()
        .any(|p| t.starts_with(p))
        || (t.starts_with("mod ") || t.starts_with("pub mod ") || t.starts_with("pub(crate) mod ")) && t.ends_with(';');
    if import {
        "imports"
    } else if ["//", "#", "/*", "*", "\"\"\"", "--", ";"].iter().any(|p| t.starts_with(p)) && !t.starts_with("#[") {
        "notes"
    } else {
        "block"
    }
}

/// One definition as the cutter sees it.
struct Def {
    start: usize,
    line: usize,
    end: usize,
    kind: String,
    name: String,
    container: bool,
}

/// What owns a run of lines.
#[derive(Clone)]
struct Seg {
    kind: &'static str,
    name: String,
    parent: String,
    signature: String,
    /// The definition's own line (0 for lines no definition owns).
    line: usize,
}

/// The head of a definition: its first line up to the line that opens its body (at most four
/// lines), joined and trimmed.
fn signature(sh: &Shape<'_>, line: usize, end: usize) -> String {
    let mut parts = Vec::new();
    for i in line..=end.min(line + 3) {
        let t = sh.line(i).trim();
        parts.push(t);
        if t.contains('{') || t.ends_with(':') || t.ends_with(';') || t.ends_with('=') || t.ends_with("=>") {
            break;
        }
    }
    let s = parts.join(" ");
    let s = s.trim_end_matches('{').trim_end().to_owned();
    s.chars().take(200).collect()
}

/// Cut one code file into pieces. `symbols` are its definitions (Rust items or the tagger's),
/// `blocks` the Rust `impl` and `trait` blocks, `toks` its tokens (`tokens`). Every line is in
/// exactly one piece.
#[must_use]
pub fn cut(text: &str, decoder: &str, symbols: &[&Symbol], blocks: &[ImplBlock], toks: &[Tk]) -> Vec<Cut> {
    let sh = Shape::new(text, toks);
    let n = sh.n();
    if n == 0 {
        return vec![Cut { start: 1, end: 1, kind: "block", heading: String::new(), meta: Meta::default() }];
    }
    let hash = matches!(decoder, "python" | "powershell" | "shell" | "ruby" | "toml" | "yaml" | "perl" | "r");
    // Doc comments and attributes above a definition lead it; an attribute spread over several
    // lines (its inner lines deeper in brackets) leads it whole.
    let lead = |line: usize| {
        let mut s = line;
        let base = sh.first(line);
        while s > 1 {
            if crate::units::leads(sh.line(s - 1), hash) && sh.indent(s - 1) == sh.indent(line) {
                s -= 1;
                continue;
            }
            let mut j = s - 1;
            while j > 1 && sh.first(j).is_some_and(|d| base.is_some_and(|b| d > b)) {
                j -= 1;
            }
            let opens = sh.line(j).trim_start();
            if j < s - 1 && (opens.starts_with("#[") || opens.starts_with('@')) && sh.indent(j) == sh.indent(line) {
                s = j;
                continue;
            }
            break;
        }
        s
    };
    // A definition whose last line was not found (a macro, a constant array) ends where the
    // bracket its first line leaves open closes.
    let close = |line: usize| {
        let base = sh.first(line).unwrap_or(0);
        if sh.end(line) <= base {
            return line;
        }
        ((line + 1)..=n).find(|j| sh.end(*j) <= base).unwrap_or(n)
    };
    let mut defs: Vec<Def> = Vec::new();
    for s in symbols.iter().filter(|s| s.line >= 1 && s.line <= n) {
        let end = if s.end > s.line { s.end.min(n) } else { close(s.line) };
        defs.push(Def { start: lead(s.line), line: s.line, end, kind: s.kind.clone(), name: s.name.clone(), container: false });
    }
    for (a, b, name) in blocks {
        if *a >= 1 && *a <= n && !defs.iter().any(|d| d.line == *a) {
            defs.push(Def { start: lead(*a), line: *a, end: (*b).min(n), kind: "impl".into(), name: name.clone(), container: true });
        }
    }
    defs.sort_by(|x, y| x.start.cmp(&y.start).then(y.end.cmp(&x.end)));
    // Containers: an impl block, or a definition that is not function-like holding another's line.
    let spans: Vec<(usize, usize)> = defs.iter().map(|d| (d.line, d.end)).collect();
    for d in &mut defs {
        if !d.container && !fn_like(&d.kind) {
            d.container = spans.iter().any(|(l, _)| *l > d.line && *l <= d.end);
        }
    }
    // The innermost container holding a line.
    let container_at = |line: usize| defs.iter().filter(|d| d.container && d.start <= line && line <= d.end).min_by_key(|d| d.end - d.start);
    const NONE: usize = usize::MAX;
    let mut owner = vec![NONE; n + 2];
    let mut segs: Vec<Seg> = Vec::new();
    for d in defs.iter().filter(|d| !d.container) {
        if owner.get(d.line).is_none_or(|o| *o != NONE) {
            continue;
        }
        let parent = container_at(d.line).map(|c| c.name.clone()).unwrap_or_default();
        let name = if parent.is_empty() { d.name.clone() } else { format!("{parent}::{}", d.name) };
        segs.push(Seg { kind: kind_of(&d.kind, !parent.is_empty()), name, parent, signature: signature(&sh, d.line, d.end), line: d.line });
        let id = segs.len() - 1;
        for o in owner.iter_mut().take(d.end + 1).skip(d.start) {
            if *o == NONE {
                *o = id;
            }
        }
    }
    // Glue: lines no leaf definition owns.
    let ctx_of = |line: usize| container_at(line).map(|c| (c.start, c.name.clone(), c.kind.clone()));
    let mut i = 1;
    while i <= n {
        if owner.get(i).is_some_and(|o| *o != NONE) {
            i += 1;
            continue;
        }
        let a = i;
        // A run of glue lies inside one container (or none): a container's head starts a new run.
        let key = |l: usize| container_at(l).map(|c| (c.start, c.end));
        while i <= n && owner.get(i).is_some_and(|o| *o == NONE) && key(i) == key(a) {
            i += 1;
        }
        let b = i - 1;
        let ctx = ctx_of(a);
        let parent = ctx.as_ref().map(|c| c.1.clone()).unwrap_or_default();
        let next = owner.get(b + 1).copied().filter(|o| *o != NONE);
        let prev = if a > 1 { owner.get(a - 1).copied().filter(|o| *o != NONE) } else { None };
        let same = |o: Option<usize>| o.filter(|x| segs.get(*x).is_some_and(|s| s.parent == parent));
        let attach = same(next).or_else(|| same(prev)).or(next).or(prev);
        let all_blank = (a..=b).all(|l| sh.blank(l));
        let head_of_file = ctx.is_none() && prev.is_none() && !all_blank;
        if (all_blank || sh.tokens(a, b) < SMALL) && !head_of_file && attach.is_some() {
            let id = attach.unwrap_or(NONE);
            for o in owner.iter_mut().take(b + 1).skip(a) {
                *o = id;
            }
            continue;
        }
        if head_of_file {
            // The file's opening lines: notes and imports apart; blank lines go with what follows.
            let mut s = a;
            while s <= b {
                let Some(f) = (s..=b).find(|l| !sh.blank(*l)) else { break };
                let class = line_class(sh.line(f));
                let mut e = f;
                while let Some(l) = ((e + 1)..=b).find(|l| !sh.blank(*l)).filter(|l| line_class(sh.line(*l)) == class) {
                    e = l;
                }
                if ((e + 1)..=b).all(|l| sh.blank(l)) {
                    e = b;
                }
                segs.push(Seg { kind: class, name: class.to_owned(), parent: String::new(), signature: String::new(), line: 0 });
                let id = segs.len() - 1;
                for o in owner.iter_mut().take(e + 1).skip(s) {
                    *o = id;
                }
                s = e + 1;
            }
            continue;
        }
        let (kind, name) = match &ctx {
            Some((start, name, kind)) if *start == a => (kind_of(kind, false), name.clone()),
            _ => ("block", if parent.is_empty() { "block".to_owned() } else { parent.clone() }),
        };
        segs.push(Seg { kind, name, parent: parent.clone(), signature: String::new(), line: 0 });
        let id = segs.len() - 1;
        for o in owner.iter_mut().take(b + 1).skip(a) {
            *o = id;
        }
    }
    // Runs of one owner, split when larger than the judge's window.
    let mut out: Vec<(Cut, bool, bool)> = Vec::new(); // (piece, part after the first, small)
    let mut s = 1;
    while s <= n {
        let id = owner.get(s).copied().unwrap_or(NONE);
        let mut e = s;
        while e < n && owner.get(e + 1).copied().unwrap_or(NONE) == id {
            e += 1;
        }
        let seg = segs.get(id).cloned().unwrap_or(Seg { kind: "block", name: "block".into(), parent: String::new(), signature: String::new(), line: 0 });
        let member = if seg.parent.is_empty() || seg.kind == "block" { String::new() } else { seg.name.clone() };
        let cont = if seg.signature.is_empty() {
            member.clone()
        } else if member.is_empty() {
            seg.signature.clone()
        } else {
            format!("{member}: {}", seg.signature)
        };
        let budget = MAX.saturating_sub(cont.chars().count().div_ceil(4)).max(MAX / 2);
        let def = (seg.line > 0).then_some(seg.line);
        let parts = split(&sh, (s, e), budget, seg.signature.is_empty(), def, 0);
        for (k, (ps, pe)) in parts.into_iter().enumerate() {
            let meta = Meta { name: seg.name.clone(), signature: seg.signature.clone(), parent: seg.parent.clone(), ..Meta::default() };
            let heading = if k == 0 { member.clone() } else { cont.clone() };
            let small = sh.tokens(ps, pe) < SMALL;
            out.push((Cut { start: ps, end: pe, kind: seg.kind, heading, meta }, k > 0, small));
        }
        s = e + 1;
    }
    // Neighbours under `SMALL` tokens with one parent are merged while within the judge's window:
    // definitions with definitions (a run of constants stays a constant group), notes with notes,
    // imports with imports, blocks with blocks.
    let family = |k: &str| match k {
        "notes" | "imports" | "block" | "group" => k.to_owned(),
        _ => "definition".to_owned(),
    };
    let mut merged: Vec<(Cut, bool)> = Vec::new(); // (piece, a run of small ones)
    for (c, part, small) in out {
        if let Some((last, run)) = merged.last_mut()
            && small
            && !part
            && *run
            && last.meta.parent == c.meta.parent
            && (family(last.kind) == family(c.kind) || (last.kind == "group" && family(c.kind) == "definition"))
            && sh.tokens(last.start, c.end) <= MAX
        {
            last.end = c.end;
            if last.kind != c.kind {
                last.kind = "group";
            }
            if !last.meta.name.split(", ").any(|x| x == c.meta.name) {
                last.meta.name = format!("{}, {}", last.meta.name, c.meta.name);
            }
            last.heading = if last.meta.parent.is_empty() { String::new() } else { last.meta.parent.clone() };
            continue;
        }
        merged.push((c, small && !part));
    }
    merged.into_iter().map(|(c, _)| c).collect()
}

/// Fill each piece's identifiers: what it defines (the symbols whose first line it holds), assigns
/// and uses, the qualified paths it names, what it implements, and the stemmed parts.
pub fn fill(cuts: &mut [Cut], symbols: &[&Symbol], toks: &[Tk], decoder: &str) {
    let rust = decoder == "rust";
    let mut at = 0;
    for c in cuts.iter_mut() {
        let m = &mut c.meta;
        m.defines = symbols.iter().filter(|s| s.line >= c.start && s.line <= c.end && s.name.len() >= crate::refgraph::MIN_NAME).map(|s| s.name.clone()).collect();
        m.defines.dedup();
        while toks.get(at).is_some_and(|t| t.line < c.start) {
            at += 1;
        }
        let mut b = at;
        while toks.get(b).is_some_and(|t| t.line <= c.end) {
            b += 1;
        }
        let ts = toks.get(at..b).unwrap_or(&[]);
        let mut uses: std::collections::BTreeMap<&str, u32> = std::collections::BTreeMap::new();
        let mut free: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        let mut assigns: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut paths: std::collections::BTreeSet<(String, String)> = std::collections::BTreeSet::new();
        let t = |i: usize| ts.get(i).map_or("", |x| x.text.as_str());
        let ident = |i: usize| ts.get(i).is_some_and(|x| x.class == TkClass::Ident);
        for (i, tk) in ts.iter().enumerate() {
            if tk.class != TkClass::Ident || tk.text.len() < crate::refgraph::MIN_NAME {
                continue;
            }
            *uses.entry(tk.text.as_str()).or_default() += 1;
            if !(i >= 1 && t(i - 1) == ".") {
                free.insert(tk.text.as_str());
            }
            let path_sep = t(i + 1) == ":" && t(i + 2) == ":" || t(i + 1) == "::";
            let after_sep = (i >= 2 && t(i - 1) == ":" && t(i - 2) == ":") || (i >= 1 && t(i - 1) == "::");
            if (t(i + 1) == ":" && !path_sep && !after_sep && t(i + 2) != ":")
                || (t(i + 1) == "=" && !matches!(t(i + 2), "=" | ">") && !(i >= 1 && matches!(t(i - 1), "=" | "!" | "<" | ">")))
            {
                assigns.insert(tk.text.clone());
            }
            let skip = if t(i + 1) == "::" { 2 } else { 3 };
            if path_sep && ident(i + skip) {
                paths.insert((tk.text.clone(), t(i + skip).to_owned()));
            } else if !rust && t(i + 1) == "." && ident(i + 2) && !matches!(tk.text.as_str(), "self" | "this" | "cls" | "super") {
                paths.insert((tk.text.clone(), t(i + 2).to_owned()));
            }
        }
        // What the piece implements: `impl Trait for Type`, `extends`, `implements`.
        let mut implements = Vec::new();
        for (i, tk) in ts.iter().enumerate() {
            if rust && tk.text == "impl" {
                let mut depth = 0_i32;
                let mut last = None;
                for x in ts.iter().skip(i + 1) {
                    match x.text.as_str() {
                        "<" => depth += 1,
                        ">" => depth -= 1,
                        "for" if depth == 0 => {
                            if let Some(l) = last.take() {
                                implements.push(l);
                            }
                            break;
                        }
                        "{" | ";" | "where" => break,
                        _ if depth == 0 && x.class == TkClass::Ident => last = Some(x.text.clone()),
                        _ => {}
                    }
                }
            } else if !rust && matches!(tk.text.as_str(), "extends" | "implements") && ident(i + 1) {
                implements.push(t(i + 1).to_owned());
            }
        }
        implements.sort();
        implements.dedup();
        m.uses = uses.into_iter().filter(|(k, _)| !m.defines.iter().any(|d| d == k)).map(|(k, v)| (k.to_owned(), v)).collect();
        m.assigns = assigns.into_iter().filter(|a| !m.defines.contains(a)).collect();
        m.paths = paths.into_iter().collect();
        m.implements = implements;
        m.members = m.uses.iter().filter(|(k, _)| !free.contains(k.as_str())).map(|(k, _)| k.clone()).collect();
        let mut parts: Vec<String> = Vec::new();
        for name in m.defines.iter().chain(m.uses.iter().map(|u| &u.0)) {
            for p in nmlex::name_parts(name) {
                let s = nmlex::stem(&p);
                if s.len() >= 2 && !parts.contains(&s) {
                    parts.push(s);
                }
            }
        }
        m.parts = parts;
    }
}

/// Give every piece of a file the file's notes: the text of its opening notes piece (the comment
/// lines that open the file, before any definition), empty when the file opens otherwise.
pub fn file_notes(cuts: &mut [Cut], lines: &[&str]) {
    let notes: std::sync::Arc<str> = match cuts.first() {
        Some(c) if c.kind == "notes" => lines.get(c.start.saturating_sub(1)..c.end.min(lines.len())).map_or_else(String::new, |l| l.join("
")).into(),
        _ => std::sync::Arc::from(""),
    };
    for c in cuts.iter_mut() {
        c.meta.file_notes = std::sync::Arc::clone(&notes);
    }
}

/// Definitions of a TOML file for the cutter: one per `[table]`, to the line before the next.
#[must_use]
pub fn toml_tables(path: &str, text: &str) -> Vec<Symbol> {
    let lines: Vec<&str> = text.lines().collect();
    let heads: Vec<(usize, String)> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, l)| {
            let t = l.trim();
            (t.starts_with('[') && t.ends_with(']')).then(|| (i + 1, t.trim_matches(['[', ']']).to_owned()))
        })
        .collect();
    heads
        .iter()
        .enumerate()
        .map(|(k, (line, name))| {
            let end = heads.get(k + 1).map_or(lines.len(), |x| x.0 - 1);
            let end = (*line..=end).rev().find(|j| lines.get(j - 1).is_some_and(|l| !l.trim().is_empty())).unwrap_or(*line);
            Symbol { name: name.clone(), kind: "table".into(), file: path.to_owned(), line: *line, end, parent: String::new(), text: format!("[{name}]") }
        })
        .collect()
}

/// Piece sizes for a report: (count, histogram over token bounds `<SMALL`, `<96`, `<192`,
/// `<288`, `<=MAX`, over), by kind.
#[must_use]
pub fn histogram(tokens: &[usize]) -> [usize; 6] {
    let mut h = [0_usize; 6];
    for t in tokens {
        let k = if *t < SMALL {
            0
        } else if *t < 96 {
            1
        } else if *t < 192 {
            2
        } else if *t < 288 {
            3
        } else if *t <= MAX {
            4
        } else {
            5
        };
        if let Some(x) = h.get_mut(k) {
            *x += 1;
        }
    }
    h
}

/// Section check: pieces cover every line once and never cross a definition; a long function is
/// split at its statements with its signature as the later parts' heading; small constants with
/// one parent merge; a method is named by its type and keeps its `impl` line; identifiers are read.
pub(crate) fn check() -> Result<(), String> {
    let mut src = String::from("//! Notes.\nuse std::fmt;\n\n/// The cap.\npub const CAP: usize = 4;\npub const FLOOR: usize = 1;\n\n/// The limits a growth step keeps: the largest ratio of one size to the next, read by the guard\n/// before every step, and set by the defaults below.\npub struct Limits { pub max_ratio: u32 }\n\nimpl Default for Limits {\n    fn default() -> Self {\n        Self { max_ratio: CAP as u32 }\n    }\n}\n\n/// A long one.\npub fn long_one(x: usize) -> usize {\n");
    for k in 0..60 {
        src.push_str(&format!("    let value_{k} = x + {k} * CAP + helper_function_name(x, {k});\n"));
    }
    src.push_str("    x\n}\n\nfn helper_function_name(a: usize, b: usize) -> usize {\n    a + b + connect::record(a)\n}\n");
    // A definition led by a doc comment and an attribute spread over lines, long enough to split.
    src.push_str("\n/// Led by an attribute over three lines.\n#[expect(\n    clippy::too_many_lines,\n    reason = \"a check\"\n)]\npub fn attributed(\n    first: usize,\n    second: usize,\n) -> usize {\n");
    for k in 0..60 {
        src.push_str(&format!("    let other_{k} = first + second * {k} + helper_function_name(first, {k});\n"));
    }
    src.push_str("    first\n}\n");
    let (syms, blocks) = crate::repo::parse_items_with("a.rs", &src, &nmlex::tokens(&src, true));
    let refs: Vec<&Symbol> = syms.iter().collect();
    let toks = tokens(&src, "rust", None);
    let mut cuts = cut(&src, "rust", &refs, &blocks, &toks);
    fill(&mut cuts, &refs, &toks, "rust");
    let n = src.lines().count();
    let mut next = 1;
    for c in &cuts {
        if c.start != next || c.end < c.start {
            return Err(format!("nmsubs.piece: pieces do not cover the file in order: {:?}", cuts.iter().map(|c| (c.start, c.end)).collect::<Vec<_>>()));
        }
        next = c.end + 1;
        let defs = syms.iter().filter(|s| s.line >= c.start && s.line <= c.end && s.kind == "fn").count();
        if defs > 1 {
            return Err(format!("nmsubs.piece: piece {}-{} crosses a definition", c.start, c.end));
        }
    }
    if next != n + 1 {
        return Err("nmsubs.piece: lines left out".into());
    }
    let longs: Vec<&Cut> = cuts.iter().filter(|c| c.meta.name == "long_one").collect();
    if longs.len() < 2
        || longs.iter().any(|c| (c.start..=c.end).map(|l| src.lines().nth(l - 1).map_or(0, |x| x.chars().count() + 1)).sum::<usize>() / 4 > MAX)
        || longs.iter().skip(1).any(|c| !c.heading.contains("pub fn long_one(x: usize) -> usize"))
        || longs.first().is_some_and(|c| !c.heading.is_empty())
    {
        return Err(format!("nmsubs.piece: long function parts {:?}", longs.iter().map(|c| (c.start, c.end, &c.heading)).collect::<Vec<_>>()));
    }
    // The attribute and the doc comment lead their definition, and its first part holds the whole
    // signature.
    let attributed: Vec<&Cut> = cuts.iter().filter(|c| c.meta.name == "attributed").collect();
    let lead_line = src.lines().position(|l| l.starts_with("/// Led by")).map_or(0, |i| i + 1);
    let body_line = src.lines().position(|l| l.starts_with(") -> usize {")).map_or(0, |i| i + 1);
    if attributed.len() < 2 || attributed.first().is_none_or(|c| c.start > lead_line || c.end < body_line) {
        return Err(format!("nmsubs.piece: attributed parts {:?}", attributed.iter().map(|c| (c.start, c.end)).collect::<Vec<_>>()));
    }
    let consts = cuts.iter().find(|c| c.meta.defines.contains(&"CAP".to_owned()));
    if !consts.is_some_and(|c| c.meta.defines.contains(&"FLOOR".to_owned()) && c.kind == "constant") {
        return Err(format!("nmsubs.piece: small constants not merged: {consts:?}"));
    }
    let m = cuts.iter().find(|c| c.meta.name == "Limits::default");
    if !m.is_some_and(|c| c.kind == "method" && src.lines().nth(c.start - 1).is_some_and(|l| l.starts_with("impl Default")) && c.meta.implements == ["Default"] && c.meta.assigns.contains(&"max_ratio".to_owned())) {
        return Err(format!("nmsubs.piece: method piece {m:?}"));
    }
    let h = cuts.iter().find(|c| c.meta.name == "helper_function_name");
    if !h.is_some_and(|c| c.meta.paths.contains(&("connect".to_owned(), "record".to_owned())) && c.meta.parts.contains(&"help".to_owned())) {
        return Err(format!("nmsubs.piece: identifiers {h:?}"));
    }
    if cuts.first().is_none_or(|c| c.kind != "notes") || cuts.get(1).is_none_or(|c| c.kind != "imports") {
        return Err(format!("nmsubs.piece: file head {:?}", cuts.iter().take(2).map(|c| (c.kind, c.start, c.end)).collect::<Vec<_>>()));
    }
    Ok(())
}
