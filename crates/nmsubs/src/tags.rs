//! The generic tagger: definitions (name, kind, first and last line, parent) and import
//! statements of a source file in any language of the table (`nmlex::lang`), read from the tokens
//! of the generic lexer. Brace languages find a definition by its head word (or, in C, by the
//! shape `type name ( ... ) {`), take its last line from the matching closing brace and its parent
//! from the enclosing class or namespace; Python reads indentation. Rust is not tagged here: it
//! keeps `repo::parse_items`.
//!
//! What it cannot see: calls bound to their class, names made by macros or metaprogramming,
//! definitions without a head word other than the forms below (module-level assignments, class
//! fields holding arrow functions).
use crate::repo::Symbol;
use nmlex::lang::{Block, Lang};
use nmlex::lang_lex::{Kind, Tok, lex};

/// One import statement: the module named, its line, and the names it brings in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Import {
    pub spec: String,
    pub line: usize,
    pub names: Vec<String>,
}

/// What the tagger found in one file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tags {
    pub symbols: Vec<Symbol>,
    pub imports: Vec<Import>,
    /// Plain words for what the file's syntax holds though its text may never say them: a
    /// stylesheet holds `styles`, a regular expression literal is a `regular expression`, a
    /// `__global__` function is a `kernel`. They count as present in the tree's vocabulary.
    pub words: Vec<String>,
}

/// The plain words for what a file's syntax holds (`Tags::words`).
fn syntax_words(path: &str, lang: &Lang, toks: &[Tok]) -> Vec<String> {
    let mut out: Vec<&str> = Vec::new();
    if lang.id == "css" {
        out.extend(["style", "styles", "stylesheet", "css", "selector", "selectors"]);
    }
    if toks.iter().any(|t| t.kind == Kind::Regex) {
        out.extend(["regex", "regular", "expression", "expressions", "pattern"]);
    }
    let low = path.to_lowercase();
    if low.ends_with(".tsx") || low.ends_with(".jsx") {
        out.extend(["component", "components", "jsx"]);
    }
    let has = |w: &str| toks.iter().any(|t| t.kind == Kind::Ident && t.text == w);
    if has("__global__") {
        out.extend(["kernel", "kernels", "gpu"]);
    }
    if lang.ecosystem == "c"
        && toks.windows(2).any(|w| w.first().is_some_and(|a| a.text == "#") && w.get(1).is_some_and(|b| b.text == "define"))
    {
        out.extend(["macro", "macros"]);
    }
    out.into_iter().map(str::to_owned).collect()
}

/// Words that are never a function name in C-like code.
const C_NOT_NAMES: &[&str] =
    &["if", "for", "while", "switch", "return", "sizeof", "catch", "do", "else", "case", "defined", "alignof", "decltype"];
/// Words after which a JavaScript `const`, `let` or `var` value is a function.
const JS_FN_VALUE: &[&str] = &["function", "async"];

/// Tags of one file, or `None` when it is not code or is Rust. Code in a language without its
/// own table entry is read through `nmlex::lang::GENERIC`.
#[must_use]
pub fn tag(path: &str, text: &str) -> Option<Tags> {
    let name = path.rsplit('/').next().unwrap_or(path);
    let lang = nmlex::lang::lang_of(path)
        .or_else(|| matches!(nmdoc::format::by_name(name, 0), nmdoc::Class::Code("code")).then_some(&nmlex::lang::GENERIC))?;
    if lang.id == "rust" {
        return None;
    }
    let toks = lex(text, lang);
    let lines: Vec<&str> = text.lines().collect();
    let pre = directives(&toks, lang);
    let mut t =
        Tagger { lang, toks: &toks, close: closers(&toks), pre, src: text.chars().collect(), path, lines: &lines, out: Tags::default() };
    match lang.block {
        Block::Indent => t.python(),
        Block::Neutral => t.neutral(),
        Block::Brace => match lang.id {
            "css" => t.css(),
            _ => t.braces(),
        },
    }
    let mut out = t.out;
    out.words = syntax_words(path, lang, &toks);
    out.symbols.sort_by(|a, b| a.line.cmp(&b.line).then(a.name.cmp(&b.name)));
    out.symbols.dedup_by(|a, b| a.line == b.line && a.name == b.name);
    Some(out)
}

/// For each token, whether it belongs to a C preprocessor line (`#` first on its line, and the
/// lines a trailing backslash joins to it).
fn directives(toks: &[Tok], lang: &Lang) -> Vec<bool> {
    let mut out = vec![false; toks.len()];
    if lang.ecosystem != "c" {
        return out;
    }
    let mut i = 0;
    while i < toks.len() {
        let first = i == 0 || toks.get(i - 1).zip(toks.get(i)).is_some_and(|(a, b)| a.last_line < b.line);
        if first && toks.get(i).is_some_and(|t| t.text == "#" && t.kind == Kind::Punct) {
            let mut cur = toks.get(i).map_or(0, |t| t.line);
            while let Some(t) = toks.get(i) {
                let joined = i > 0 && toks.get(i - 1).is_some_and(|p| p.text == "\\") && t.line == cur + 1;
                if t.line != cur && !joined {
                    break;
                }
                cur = t.last_line;
                if let Some(x) = out.get_mut(i) {
                    *x = true;
                }
                i += 1;
            }
            continue;
        }
        i += 1;
    }
    out
}

/// For each opening bracket token, the index of its closing token (the last token when unclosed).
fn closers(toks: &[Tok]) -> Vec<usize> {
    let mut close = vec![usize::MAX; toks.len()];
    let mut stack: Vec<(usize, char)> = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        if t.kind != Kind::Punct {
            continue;
        }
        match t.text.as_str() {
            "{" => stack.push((i, '}')),
            "(" => stack.push((i, ')')),
            "[" => stack.push((i, ']')),
            c @ ("}" | ")" | "]") => {
                // Close the nearest opener of the same shape; openers left above it are unclosed.
                if let Some(at) = stack.iter().rposition(|(_, want)| c.starts_with(*want)) {
                    for (o, _) in stack.drain(at..) {
                        if let Some(x) = close.get_mut(o) {
                            *x = i;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let last = toks.len().saturating_sub(1);
    for (o, _) in stack {
        if let Some(x) = close.get_mut(o) {
            *x = last;
        }
    }
    close
}

/// One open scope of the brace tagger.
#[derive(Clone, Debug)]
struct Scope {
    close: usize,
    /// `class` (members are methods), `transparent` (namespace, `extern "C"`, `@media`: the
    /// inside counts as top level), `body` (anything else).
    kind: &'static str,
    name: String,
}

struct Tagger<'a> {
    lang: &'static Lang,
    toks: &'a [Tok],
    close: Vec<usize>,
    /// Tokens of preprocessor lines (C family).
    pre: Vec<bool>,
    src: Vec<char>,
    path: &'a str,
    lines: &'a [&'a str],
    out: Tags,
}

impl Tagger<'_> {
    fn tok(&self, i: usize) -> Option<&Tok> {
        self.toks.get(i)
    }
    fn is(&self, i: usize, s: &str) -> bool {
        self.tok(i).is_some_and(|t| t.text == s && t.kind != Kind::Str)
    }
    fn ident(&self, i: usize) -> Option<&str> {
        self.tok(i).filter(|t| t.kind == Kind::Ident).map(|t| t.text.as_str())
    }
    fn line_text(&self, line: usize) -> String {
        self.lines.get(line.saturating_sub(1)).map_or("", |l| l.trim()).to_owned()
    }
    /// True when token `i` is the first on its line.
    fn first_on_line(&self, i: usize) -> bool {
        i == 0 || self.tok(i - 1).zip(self.tok(i)).is_some_and(|(a, b)| a.last_line < b.line)
    }
    fn end_of(&self, i: usize) -> usize {
        self.close.get(i).copied().filter(|c| *c != usize::MAX).unwrap_or(i)
    }
    fn add(&mut self, name: &str, kind: &str, line: usize, end: usize, parent: &str) {
        self.out.symbols.push(Symbol {
            name: name.to_owned(),
            kind: kind.to_owned(),
            file: self.path.to_owned(),
            line,
            end: end.max(line),
            parent: parent.to_owned(),
            text: self.line_text(line),
        });
    }
    /// The source text between two token places, white space collapsed.
    fn between(&self, a: usize, b: usize) -> String {
        let s: String = self.src.get(a..b.min(self.src.len())).map(|x| x.iter().collect()).unwrap_or_default();
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// From token `i` (just past a name), the token that opens the body `{` at this nesting, or
    /// `None` when a `;` (a declaration) or a closing bracket comes first. Bracket groups `( )` and
    /// `[ ]` are stepped over; `<` ... `>` need no care because they are not brackets here.
    fn body_from(&self, mut i: usize, limit: usize) -> Option<usize> {
        while i < limit {
            let t = self.tok(i)?;
            if t.kind == Kind::Punct {
                match t.text.as_str() {
                    "{" => return Some(i),
                    "(" | "[" => {
                        i = self.end_of(i) + 1;
                        continue;
                    }
                    ";" | "}" | ")" | "]" => return None,
                    // An arrow function's expression body has no braces.
                    "=" if !self.lang.id.contains("script") => return None,
                    _ => {}
                }
            }
            i += 1;
        }
        None
    }
    /// The last token of a statement starting at token `s`: the `;` that ends it at this nesting,
    /// or the token before the next line that starts at this nesting at or left of column `col`.
    fn statement_end(&self, s: usize, col: usize, limit: usize) -> usize {
        let mut i = s;
        let mut last = s;
        while i < limit {
            let Some(t) = self.tok(i) else { break };
            if i > s
                && self.first_on_line(i)
                && t.col <= col
                && !matches!(t.text.as_str(), "." | "?" | ":" | "+" | "|" | "&" | ")" | "]" | "}")
            {
                return last;
            }
            if t.kind == Kind::Punct {
                match t.text.as_str() {
                    ";" => return i,
                    "{" | "(" | "[" => {
                        last = self.end_of(i);
                        i = last + 1;
                        continue;
                    }
                    "}" | ")" | "]" => return last,
                    _ => {}
                }
            }
            last = i;
            i += 1;
        }
        last
    }

    /// Brace languages: JavaScript, TypeScript, C, C++, CUDA, PowerShell.
    fn braces(&mut self) {
        let js = self.lang.id == "typescript" || self.lang.id == "javascript";
        let c_like = self.lang.ecosystem == "c";
        let mut scopes: Vec<Scope> = Vec::new();
        let n = self.toks.len();
        let mut i = 0;
        let mut directive_end_line = 0;
        while i < n {
            scopes.retain(|s| s.close > i);
            let Some(t) = self.tok(i).cloned() else { break };
            let top = scopes.iter().rev().find(|s| s.kind != "transparent");
            let at_top = top.is_none();
            let in_class = top.is_some_and(|s| s.kind == "class");
            let parent = scopes.iter().rev().find(|s| !s.name.is_empty()).map(|s| s.name.clone()).unwrap_or_default();
            let limit = scopes.last().map_or(n, |s| s.close);
            // Preprocessor directives (C family): `#define`, `#include`; the rest is skipped.
            if c_like && t.text == "#" && self.first_on_line(i) {
                i = self.directive(i);
                directive_end_line = self.tok(i.saturating_sub(1)).map_or(0, |x| x.last_line);
                continue;
            }
            if t.line <= directive_end_line && c_like {
                i += 1;
                continue;
            }
            if js {
                self.js_import(i);
            }
            if t.kind != Kind::Ident {
                if t.text == "{" {
                    scopes.push(Scope { close: self.end_of(i), kind: "body", name: String::new() });
                }
                i += 1;
                continue;
            }
            let prev_dot = i > 0 && (self.is(i - 1, ".") || self.is(i - 1, "::") || self.is(i - 1, "->"));
            if prev_dot {
                i += 1;
                continue;
            }
            // `extern "C" {` and C++ `namespace X {`: the inside counts as top level.
            if c_like && t.text == "extern" && self.tok(i + 1).is_some_and(|x| x.kind == Kind::Str) && self.is(i + 2, "{") {
                scopes.push(Scope { close: self.end_of(i + 2), kind: "transparent", name: parent.clone() });
                i += 3;
                continue;
            }
            if let Some(&(_, kind)) = self.lang.defs.iter().find(|(h, _)| *h == t.text)
                && let Some(next) = self.head(i, kind, &parent, at_top || in_class, &mut scopes)
            {
                i = next;
                continue;
            }
            if js
                && at_top
                && matches!(t.text.as_str(), "const" | "let" | "var")
                && let Some(next) = self.js_assign(i, &parent, limit)
            {
                i = next;
                continue;
            }
            if in_class && let Some(next) = self.member(i, &parent, js, limit) {
                i = next;
                continue;
            }
            // Methods without a head word (`int size() {`): the C family, Java, C#, Dart.
            let shaped = c_like || matches!(self.lang.ecosystem, "jvm" | "dotnet" | "dart");
            if shaped
                && (at_top || in_class)
                && let Some(next) = self.c_function(i, &parent, in_class, &mut scopes)
            {
                i = next;
                continue;
            }
            if c_like && at_top && let Some(next) = self.c_const(i, &parent, limit) {
                i = next;
                continue;
            }
            i += 1;
        }
    }

    /// A definition head word at `i` with the kind it gives; returns where to continue.
    fn head(&mut self, i: usize, kind: &str, parent: &str, place_ok: bool, scopes: &mut Vec<Scope>) -> Option<usize> {
        let word = self.tok(i)?.text.clone();
        let line = self.tok(i)?.line;
        let js = self.lang.id == "typescript" || self.lang.id == "javascript";
        let mut at = i + 1;
        if word == "function" && self.is(at, "*") {
            at += 1;
        }
        // Go methods: `func (r *Store) Add(...)`: the receiver's type is the parent.
        let mut owner = String::new();
        if word == "func" && self.is(at, "(") {
            let close = self.end_of(at);
            owner = (at..close).rev().find_map(|k| self.ident(k)).unwrap_or("").to_owned();
            at = close + 1;
        }
        let parent = if owner.is_empty() { parent } else { owner.as_str() };
        // PowerShell `function global:Name`.
        if self.lang.id == "powershell" && self.is(at + 1, ":") && self.ident(at + 2).is_some() {
            at += 2;
        }
        if word == "typedef" {
            return self.typedef(i, parent, scopes);
        }
        if (word == "struct" || word == "union" || word == "class") && self.ident(at) == Some("class") {
            at += 1;
        }
        if word == "enum" && (self.ident(at) == Some("class") || self.ident(at) == Some("struct")) {
            at += 1;
        }
        let name = self.ident(at)?.to_owned();
        let after = at + 1;
        // Contextual words of TypeScript: `type X =`, `interface X {`, `namespace X {`.
        if js {
            if word == "type" && !(self.is(after, "=") || self.is(after, "<")) {
                return None;
            }
            if (word == "interface" || word == "namespace" || word == "enum")
                && !(self.is(after, "{") || self.ident(after).is_some() || self.is(after, "<"))
            {
                return None;
            }
        }
        if !place_ok && word != "function" {
            return None;
        }
        let limit = scopes.last().map_or(self.toks.len(), |s| s.close);
        if js && word == "type" {
            let col = self.tok(i)?.col;
            let start = self.statement_start(i);
            let end = self.statement_end(i, self.tok(start).map_or(col, |t| t.col), limit);
            self.add(&name, kind, line, self.tok(end)?.last_line, parent);
            return Some(end + 1);
        }
        match self.body_from(after, limit) {
            Some(b) => {
                let close = self.end_of(b);
                // `struct x y;` is a use, not a definition; a C struct needs its body right after.
                if matches!(word.as_str(), "struct" | "union" | "enum") && self.ecosystem_c() && b != after && !self.is(after, ":") {
                    return None;
                }
                self.add(&name, kind, line, self.tok(close)?.last_line, parent);
                let scope_kind = match word.as_str() {
                    "class" | "struct" | "union" | "interface" | "enum" => "class",
                    "namespace" => "transparent",
                    _ => "body",
                };
                // Members are named under the class (or the namespace, for its items).
                let scope_name = if word == "function" || word == "filter" { String::new() } else { name.clone() };
                scopes.push(Scope { close, kind: scope_kind, name: scope_name });
                Some(b + 1)
            }
            None => {
                // A declaration without a body: a C `struct x;` is skipped, a TypeScript
                // overload or `declare function` ends at its `;`.
                if self.ecosystem_c() || word == "class" {
                    return None;
                }
                let start = self.statement_start(i);
                let end = self.statement_end(i, self.tok(start).map_or(0, |t| t.col), limit);
                self.add(&name, kind, line, self.tok(end)?.last_line, parent);
                Some(end + 1)
            }
        }
    }
    fn ecosystem_c(&self) -> bool {
        self.lang.ecosystem == "c"
    }
    /// The first token of the statement holding token `i` (stepping back over modifiers).
    fn statement_start(&self, i: usize) -> usize {
        let mut s = i;
        while s > 0 && self.tok(s - 1).is_some_and(|t| t.kind == Kind::Ident && self.lang.modifiers.contains(&t.text.as_str())) {
            s -= 1;
        }
        s
    }

    /// TypeScript and JavaScript `const`, `let` or `var` at module level.
    fn js_assign(&mut self, i: usize, parent: &str, limit: usize) -> Option<usize> {
        let word = self.tok(i)?.text.clone();
        let name = self.ident(i + 1)?.to_owned();
        let line = self.tok(i)?.line;
        // Type annotation up to `=`.
        let mut at = i + 2;
        if self.is(at, ":") {
            while at < limit && !self.is(at, "=") && !self.is(at, ";") {
                at = if self.is(at, "(") || self.is(at, "{") || self.is(at, "[") { self.end_of(at) + 1 } else { at + 1 };
            }
        }
        let start = self.statement_start(i);
        let col = self.tok(start)?.col;
        let end = self.statement_end(i, col, limit);
        let end_line = self.tok(end)?.last_line;
        if !self.is(at, "=") {
            // `declare const X: T;` and `let x;`: a declaration, still a module-level name.
            self.add(&name, if word == "const" { "const" } else { "variable" }, line, end_line, parent);
            return Some(end + 1);
        }
        let v = at + 1;
        let kind = if self.ident(v).is_some_and(|w| JS_FN_VALUE.contains(&w)) {
            "function"
        } else if self.ident(v) == Some("class") {
            "class"
        } else if self.is(v, "(") && self.is(self.end_of(v) + 1, "=>") {
            "function"
        } else if self.is(v, "(") && self.is(self.end_of(v) + 1, ":") {
            // `(x: T): R => ...`: an arrow function with a return type.
            let mut k = self.end_of(v) + 1;
            while k < end && !self.is(k, "=>") {
                k += 1;
            }
            if self.is(k, "=>") { "function" } else { "const" }
        } else if (self.ident(v).is_some() && self.is(v + 1, "=>"))
            || (self.is(v, "<") && (v..end).any(|k| self.is(k, "=>")) && (v..end).any(|k| self.is(k, "(")))
        {
            "function"
        } else if word == "const" {
            "const"
        } else {
            "variable"
        };
        self.add(&name, kind, line, end_line, parent);
        Some(end + 1)
    }

    /// True when an arrow `=>` follows a parameter list ending just before `k`, directly or after
    /// a return type (`(x): T => ...`), before the statement ends.
    fn arrow_after(&self, mut k: usize, limit: usize) -> bool {
        if self.is(k, "=>") {
            return true;
        }
        if !self.is(k, ":") {
            return false;
        }
        while k < limit && !self.is(k, ";") && !self.is(k, "{") && !self.is(k, "=") {
            if self.is(k, "=>") {
                return true;
            }
            k = if self.is(k, "(") || self.is(k, "[") { self.end_of(k) + 1 } else { k + 1 };
        }
        false
    }

    /// A class member at `i`: a method `name(...) {`, or a field holding an arrow function.
    fn member(&mut self, i: usize, parent: &str, js: bool, limit: usize) -> Option<usize> {
        // Only at the start of a member: first on its line or after `;`, `{` or `}`.
        let mut s = i;
        // PowerShell methods may start with their return type: `[string] Name() {`.
        let typed = |t: &Self, k: usize| {
            t.lang.id == "powershell"
                && k > 0
                && t.is(k - 1, "]")
                && (0..k - 1).rev().find(|o| t.is(*o, "[") && t.end_of(*o) == k - 1).is_some_and(|o| t.first_on_line(o))
        };
        let starts_member =
            |t: &Self, k: usize| k == 0 || t.first_on_line(k) || t.is(k - 1, ";") || t.is(k - 1, "{") || t.is(k - 1, "}") || typed(t, k);
        while s > 0 && self.tok(s - 1).is_some_and(|t| t.kind == Kind::Ident && self.lang.modifiers.contains(&t.text.as_str())) {
            s -= 1;
        }
        if !starts_member(self, s) {
            return None;
        }
        let tok = self.tok(i)?;
        if self.lang.modifiers.contains(&tok.text.as_str()) && self.ident(i + 1).is_some() && !self.is(i + 1, "(") {
            return None;
        }
        let name = tok.text.clone();
        let line = self.tok(s)?.line;
        let mut at = i + 1;
        if js && self.is(at, "<") {
            while at < limit && !self.is(at, "(") {
                at += 1;
            }
        }
        if self.is(at, "(") {
            let b = self.body_from(self.end_of(at) + 1, limit)?;
            if self.ecosystem_c()
                && !self.is(self.end_of(at) + 1, "{")
                && !self.ident(self.end_of(at) + 1).is_some_and(|w| matches!(w, "const" | "override" | "noexcept" | "final"))
            {
                return None;
            }
            let close = self.end_of(b);
            self.add(&name, "method", line, self.tok(close)?.last_line, parent);
            return Some(close + 1);
        }
        if js && self.is(at, "=") {
            let v = at + 1;
            let arrow = (self.is(v, "(") && self.arrow_after(self.end_of(v) + 1, limit))
                || (self.ident(v).is_some() && self.is(v + 1, "=>"))
                || (self.ident(v) == Some("async"));
            if arrow {
                let end = self.statement_end(i, self.tok(s)?.col, limit);
                self.add(&name, "method", line, self.tok(end)?.last_line, parent);
                return Some(end + 1);
            }
        }
        None
    }

    /// A C, C++ or CUDA function definition at `i`: `name ( ... ) {` after a type.
    fn c_function(&mut self, i: usize, parent: &str, in_class: bool, scopes: &mut Vec<Scope>) -> Option<usize> {
        let t = self.tok(i)?;
        if C_NOT_NAMES.contains(&t.text.as_str()) || self.lang.modifiers.contains(&t.text.as_str()) {
            return None;
        }
        // `Type::name(` (C++ out-of-class method).
        let (name, owner, open) = if self.is(i + 1, "::") && self.ident(i + 2).is_some() && self.is(i + 3, "(") {
            (self.ident(i + 2)?.to_owned(), t.text.clone(), i + 3)
        } else if self.is(i + 1, "(") {
            (t.text.clone(), String::new(), i + 1)
        } else {
            return None;
        };
        // A type (or `*`, `&`, `>`) must come before the name, unless it is a constructor.
        let before_ok = i > 0
            && self.tok(i - 1).is_some_and(|p| {
                (p.kind == Kind::Ident && !C_NOT_NAMES.contains(&p.text.as_str()) && p.text != "return")
                    || matches!(p.text.as_str(), "*" | "&" | ">")
            });
        if !before_ok && !(in_class && name == *parent) && owner.is_empty() {
            return None;
        }
        let after = self.end_of(open) + 1;
        let mut k = after;
        while self.ident(k).is_some_and(|w| matches!(w, "const" | "override" | "noexcept" | "final" | "volatile")) {
            k += 1;
        }
        if self.is(k, "->") {
            while k < self.toks.len() && !self.is(k, "{") && !self.is(k, ";") {
                k += 1;
            }
        }
        if self.is(k, ":") && !owner.is_empty() {
            // Constructor initialiser list.
            while k < self.toks.len() && !self.is(k, "{") && !self.is(k, ";") {
                k = if self.is(k, "(") { self.end_of(k) + 1 } else { k + 1 };
            }
        }
        if !self.is(k, "{") {
            return None;
        }
        let close = self.end_of(k);
        let start = self.c_start(i);
        let line = self.tok(start)?.line;
        let (kind, par) = if !owner.is_empty() {
            ("method", owner)
        } else if in_class {
            ("method", parent.to_owned())
        } else {
            ("function", parent.to_owned())
        };
        self.add(&name, kind, line, self.tok(close)?.last_line, &par);
        scopes.push(Scope { close, kind: "body", name: String::new() });
        Some(k + 1)
    }
    /// The first token of a C declaration ending at the name `i`: back over type words, `*`,
    /// template headers, on the same line or lines that hold only the type.
    fn c_start(&self, i: usize) -> usize {
        let mut s = i;
        while s > 0 {
            let Some(p) = self.tok(s - 1) else { break };
            let typeish = p.kind == Kind::Ident || matches!(p.text.as_str(), "*" | "&" | "<" | ">" | "," | "::");
            if !typeish || matches!(p.text.as_str(), ";" | "}" | "{") || self.pre.get(s - 1).copied().unwrap_or(false) {
                break;
            }
            s -= 1;
        }
        s
    }
    /// A module-level C constant: `static const int NAME = ...;` or `constexpr ... NAME = ...;`.
    fn c_const(&mut self, i: usize, parent: &str, limit: usize) -> Option<usize> {
        let w = self.ident(i)?;
        if w != "const" && w != "constexpr" {
            return None;
        }
        let s = if i > 0 && self.ident(i - 1) == Some("static") { i - 1 } else { i };
        let statement_start = s == 0 || self.is(s - 1, ";") || self.is(s - 1, "}") || self.is(s - 1, "{") || self.first_on_line(s);
        if !statement_start {
            return None;
        }
        let mut k = i + 1;
        while k < limit && self.ident(k).is_some() && !(self.is(k + 1, "=") || self.is(k + 1, "[")) {
            k += 1;
            while self.is(k, "*") {
                k += 1;
            }
        }
        let name = self.ident(k)?.to_owned();
        if !self.is(k + 1, "=") && !self.is(k + 1, "[") {
            return None;
        }
        let end = (k..limit).find(|x| self.is(*x, ";"))?;
        self.add(&name, "const", self.tok(s)?.line, self.tok(end)?.last_line, parent);
        Some(end + 1)
    }
    /// `typedef ... NAME;`, with a struct, union or enum body or a function pointer.
    fn typedef(&mut self, i: usize, parent: &str, scopes: &[Scope]) -> Option<usize> {
        let line = self.tok(i)?.line;
        let limit = scopes.last().map_or(self.toks.len(), |s| s.close);
        let mut k = i + 1;
        let mut name: Option<String> = None;
        let mut struct_name: Option<(String, usize)> = None;
        while k < limit && !self.is(k, ";") {
            if self.is(k, "{") {
                if let Some(w) = self.ident(k - 1).filter(|_| self.ident(k - 2).is_some_and(|s| matches!(s, "struct" | "union" | "enum"))) {
                    struct_name = Some((w.to_owned(), k));
                }
                k = self.end_of(k) + 1;
                continue;
            }
            if self.is(k, "(") && self.is(k + 1, "*") && self.ident(k + 2).is_some() {
                name = self.ident(k + 2).map(str::to_owned);
                k = self.end_of(k) + 1;
                continue;
            }
            if let Some(w) = self.ident(k)
                && (name.is_none() || self.is(k + 1, ";") || self.is(k + 1, "["))
            {
                name = Some(w.to_owned());
            }
            k += 1;
        }
        let end_line = self.tok(k)?.last_line;
        if let Some((sname, open)) = struct_name {
            let kind = if self.ident(open - 2) == Some("enum") { "enum" } else { "struct" };
            self.add(&sname, kind, line, end_line, parent);
        }
        let name = name?;
        self.add(&name, "type", line, end_line, parent);
        Some(k + 1)
    }
    /// A preprocessor directive at `#` (token `i`): records `#define` and `#include`; returns the
    /// first token after the directive (its lines joined by trailing backslashes).
    fn directive(&mut self, i: usize) -> usize {
        let line = self.tok(i).map_or(0, |t| t.line);
        let mut end = i + 1;
        let mut cur = line;
        while let Some(t) = self.tok(end) {
            if t.line == cur || (t.line == cur + 1 && self.is(end - 1, "\\")) {
                cur = t.last_line;
                end += 1;
            } else {
                break;
            }
        }
        let last_line = self.tok(end.saturating_sub(1)).map_or(line, |t| t.last_line);
        match self.ident(i + 1) {
            Some("define") => {
                if let Some(name) = self.ident(i + 2).map(str::to_owned) {
                    self.add(&name, "macro", line, last_line, "");
                }
            }
            Some("include") => {
                if let Some(t) = self.tok(i + 2).filter(|t| t.kind == Kind::Str) {
                    let spec = t.text.clone();
                    self.out.imports.push(Import { spec, line, names: Vec::new() });
                } else if self.is(i + 2, "<")
                    && let Some(close) = (i + 3..end).find(|k| self.is(*k, ">"))
                {
                    let (a, b) = (self.tok(i + 3).map_or(0, |t| t.start), self.tok(close).map_or(0, |t| t.start));
                    let spec = format!("<{}>", self.between(a, b).replace(' ', ""));
                    self.out.imports.push(Import { spec, line, names: Vec::new() });
                }
            }
            _ => {}
        }
        end
    }

    /// A JavaScript or TypeScript import at token `i`: `import ... from "x"`, `import "x"`,
    /// `export ... from "x"`, `require("x")`, `import("x")`.
    fn js_import(&mut self, i: usize) {
        let Some(t) = self.tok(i) else { return };
        if t.kind != Kind::Ident || (i > 0 && self.is(i - 1, ".")) {
            return;
        }
        let line = t.line;
        let str_at = |s: &Self, k: usize| s.tok(k).filter(|x| x.kind == Kind::Str).map(|x| x.text.clone());
        match t.text.as_str() {
            "require" | "import" if self.is(i + 1, "(") => {
                if let Some(spec) = str_at(self, i + 2) {
                    self.out.imports.push(Import { spec, line, names: Vec::new() });
                }
            }
            "import" | "export" => {
                if t.text == "import"
                    && let Some(spec) = str_at(self, i + 1)
                {
                    self.out.imports.push(Import { spec, line, names: Vec::new() });
                    return;
                }
                let mut names = Vec::new();
                let mut k = i + 1;
                while k < i + 200 {
                    let Some(x) = self.tok(k) else { return };
                    if x.text == ";" && x.kind == Kind::Punct {
                        return;
                    }
                    if x.kind == Kind::Ident && x.text == "from" {
                        if let Some(spec) = str_at(self, k + 1) {
                            self.out.imports.push(Import { spec, line, names });
                        }
                        return;
                    }
                    if x.kind == Kind::Ident {
                        if matches!(x.text.as_str(), "type" | "as" | "typeof") {
                            k += 1;
                            continue;
                        }
                        if t.text == "export" && k == i + 1 && !self.is(k, "{") {
                            // `export const x`, `export function f`: not a re-export.
                            return;
                        }
                        if !self.is(k - 1, "as") {
                            names.push(x.text.clone());
                        }
                    } else if x.kind == Kind::Punct && !matches!(x.text.as_str(), "{" | "}" | "," | "*") {
                        return;
                    }
                    k += 1;
                }
            }
            _ => {}
        }
    }

    /// CSS: one `selector` symbol per selector of a rule, `variable` for custom properties,
    /// `keyframes` for animations; `@media` and `@supports` count as top level; `@import`.
    fn css(&mut self) {
        let n = self.toks.len();
        let mut i = 0;
        let mut scopes: Vec<usize> = Vec::new();
        while i < n {
            scopes.retain(|c| *c > i);
            let Some(t) = self.tok(i).cloned() else { break };
            if t.text == "}" || t.text == ";" {
                i += 1;
                continue;
            }
            if t.text == "@" {
                let word = self.ident(i + 1).unwrap_or("").to_owned();
                let semi = (i..n).find(|k| self.is(*k, ";") || self.is(*k, "{")).unwrap_or(n);
                if word == "import" {
                    if let Some(s) = self.tok(i + 2).filter(|x| x.kind == Kind::Str) {
                        let spec = s.text.clone();
                        self.out.imports.push(Import { spec, line: t.line, names: Vec::new() });
                    }
                    i = semi + 1;
                    continue;
                }
                if self.is(semi, "{") {
                    let close = self.end_of(semi);
                    if word == "keyframes" || word.ends_with("-keyframes") {
                        if let Some(name) = self.ident(i + 2).map(str::to_owned) {
                            self.add(&name, "keyframes", t.line, self.tok(close).map_or(t.line, |x| x.last_line), "");
                        }
                        i = close + 1;
                    } else if matches!(word.as_str(), "media" | "supports" | "layer" | "container" | "document") {
                        scopes.push(close);
                        i = semi + 1;
                    } else {
                        i = close + 1;
                    }
                    continue;
                }
                i = semi + 1;
                continue;
            }
            // A rule: selectors up to `{`.
            let Some(open) = (i..n).find(|k| self.is(*k, "{") || self.is(*k, "}") || self.is(*k, ";")) else { break };
            if !self.is(open, "{") {
                i = open + 1;
                continue;
            }
            let close = self.end_of(open);
            let end_line = self.tok(close).map_or(t.line, |x| x.last_line);
            let mut part = i;
            let mut first_sel = String::new();
            let mut k = i;
            while k <= open {
                let split = k == open || (self.is(k, ",") && !(part..k).any(|x| self.is(x, "(") && self.end_of(x) > k));
                if split && k > part {
                    let (a, b) = (self.tok(part).map_or(0, |x| x.start), self.tok(k).map_or(0, |x| x.start));
                    let name = self.between(a, b);
                    let line = self.tok(part).map_or(t.line, |x| x.line);
                    if first_sel.is_empty() {
                        first_sel.clone_from(&name);
                    }
                    self.add(&name, "selector", line, end_line, "");
                    part = k + 1;
                } else if split {
                    part = k + 1;
                }
                k = if self.is(k, "(") && k != open { self.end_of(k).max(k + 1) } else { k + 1 };
            }
            // Custom properties declared in the rule.
            let mut d = open + 1;
            while d < close {
                if let Some(name) = self.ident(d).filter(|w| w.starts_with("--")).map(str::to_owned)
                    && self.is(d + 1, ":")
                    && (self.is(d - 1, "{") || self.is(d - 1, ";"))
                {
                    let end = (d..close).find(|x| self.is(*x, ";")).unwrap_or(close.saturating_sub(1));
                    let line = self.tok(d).map_or(0, |x| x.line);
                    self.add(&name, "variable", line, self.tok(end).map_or(line, |x| x.last_line), &first_sel);
                }
                d = if self.is(d, "{") || self.is(d, "(") { self.end_of(d) + 1 } else { d + 1 };
            }
            i = close + 1;
        }
    }

    /// Neutral blocks (`Block::Neutral`): a head word of the language at the start of a line
    /// (after its modifiers) and a name, or a shell function `name() {`. It ends at the matching
    /// `}` when it opens one on its own line, else before the next line indented as little as it,
    /// that line included when it is a closing word (`end`, `fi`, `done`, `}`).
    fn neutral(&mut self) {
        let starts: Vec<usize> = (0..self.toks.len()).filter(|&i| self.first_on_line(i)).collect();
        let n = self.toks.len();
        let mut open: Vec<(usize, String, bool)> = Vec::new();
        for (li, &s) in starts.iter().enumerate() {
            let Some(t) = self.tok(s).cloned() else { continue };
            let indent = t.col;
            while open.last().is_some_and(|o| o.0 >= indent) {
                open.pop();
            }
            let mut k = s;
            while self.ident(k).is_some_and(|w| self.lang.modifiers.contains(&w)) {
                k += 1;
            }
            let next_start = starts.get(li + 1).copied().unwrap_or(n);
            let head = self.ident(k).unwrap_or("").to_owned();
            let (name, kind) = if let Some(&(_, kind)) = self.lang.defs.iter().find(|(h, _)| *h == head) {
                // The name, or the last part of a dotted one (`self.name`, `M.name`, `M:name`).
                let mut j = k + 1;
                let mut name = None;
                while j < next_start {
                    if let Some(w) = self.ident(j) {
                        name = Some(w.to_owned());
                        if self.is(j + 1, ".") || self.is(j + 1, ":") || self.is(j + 1, "::") {
                            j += 2;
                            continue;
                        }
                    }
                    break;
                }
                let Some(name) = name else { continue };
                if self.is(k + 1, "=") {
                    continue;
                }
                (name, kind)
            } else if matches!(self.lang.id, "shell" | "code")
                && !head.is_empty()
                && self.is(k + 1, "(")
                && self.is(k + 2, ")")
                && (self.is(k + 3, "{") || self.tok(k + 3).is_some_and(|x| x.line > t.line))
            {
                (head.clone(), "function")
            } else {
                continue;
            };
            let brace = (k..next_start).find(|&j| self.is(j, "{"));
            let end = match brace {
                Some(b) => self.tok(self.end_of(b)).map_or(t.line, |x| x.last_line),
                None => {
                    let stop = starts.iter().skip(li + 1).find(|&&x| self.tok(x).is_some_and(|y| y.col <= indent)).copied();
                    match stop {
                        Some(x)
                            if self.tok(x).is_some_and(|y| {
                                y.col == indent && matches!(y.text.as_str(), "end" | "fi" | "done" | "esac" | "}" | "END")
                            }) =>
                        {
                            self.tok(x).map_or(t.line, |y| y.last_line)
                        }
                        Some(x) => self.tok(x.saturating_sub(1)).map_or(t.line, |y| y.last_line),
                        None => self.tok(n.saturating_sub(1)).map_or(t.line, |y| y.last_line),
                    }
                }
            };
            let parent = open.last().map(|o| o.1.clone()).unwrap_or_default();
            let kind = if kind == "function" && open.last().is_some_and(|o| o.2) { "method" } else { kind };
            self.add(&name, kind, t.line, end, &parent);
            open.push((indent, name, matches!(kind, "class" | "struct" | "module" | "interface" | "trait")));
        }
    }

    /// Python: `def` and `class` by indentation, module-level assignments, imports.
    fn python(&mut self) {
        // Logical line starts: first token of a physical line at bracket depth zero, not after a
        // backslash continuation.
        let mut starts: Vec<usize> = Vec::new();
        let mut depth = 0_i32;
        for (i, t) in self.toks.iter().enumerate() {
            if depth == 0 && self.first_on_line(i) && !(i > 0 && self.is(i - 1, "\\")) {
                starts.push(i);
            }
            if t.kind == Kind::Punct {
                match t.text.as_str() {
                    "(" | "[" | "{" => depth += 1,
                    ")" | "]" | "}" => depth = (depth - 1).max(0),
                    _ => {}
                }
            }
        }
        let n = self.toks.len();
        // Open definitions: (indent, name, is class).
        let mut open: Vec<(usize, String, bool)> = Vec::new();
        for (li, &s) in starts.iter().enumerate() {
            let Some(t) = self.tok(s).cloned() else { continue };
            let indent = t.col;
            while open.last().is_some_and(|o| o.0 >= indent) {
                open.pop();
            }
            let next_line_start = starts.get(li + 1).copied().unwrap_or(n);
            let mut k = s;
            if self.ident(k) == Some("async") {
                k += 1;
            }
            let head = self.ident(k).unwrap_or("").to_owned();
            if head == "import" || head == "from" {
                self.py_import(k, next_line_start);
                continue;
            }
            if head == "def" || head == "class" {
                let Some(name) = self.ident(k + 1).map(str::to_owned) else { continue };
                // The body ends before the next logical line at this indent or less.
                let stop = starts.iter().skip(li + 1).find(|&&x| self.tok(x).is_some_and(|y| y.col <= indent)).copied().unwrap_or(n);
                let end = self.tok(stop.saturating_sub(1)).map_or(t.line, |x| x.last_line);
                let parent = open.last().map(|o| o.1.clone()).unwrap_or_default();
                let kind = if head == "class" {
                    "class"
                } else if open.last().is_some_and(|o| o.2) {
                    "method"
                } else {
                    "function"
                };
                self.add(&name, kind, t.line, end, &parent);
                open.push((indent, name, head == "class"));
                continue;
            }
            // Module-level `NAME = ...` or `NAME: T = ...`.
            if indent == 0
                && t.kind == Kind::Ident
                && !["if", "for", "while", "with", "try", "elif", "else", "return", "del", "assert", "raise", "global", "print"]
                    .contains(&t.text.as_str())
            {
                let assign = (self.is(s + 1, "=") && !self.is(s + 2, "="))
                    || (self.is(s + 1, ":")
                        && (s + 2..next_line_start).any(|x| self.is(x, "=") && !self.is(x + 1, "=") && !self.is(x - 1, "=")));
                if assign {
                    let end = self.tok(next_line_start.saturating_sub(1)).map_or(t.line, |x| x.last_line);
                    let upper = t.text.chars().all(|c| c.is_uppercase() || c.is_ascii_digit() || c == '_')
                        && t.text.chars().any(char::is_alphabetic);
                    self.add(&t.text, if upper { "const" } else { "variable" }, t.line, end, "");
                }
            }
        }
    }
    /// A Python `import a.b as c, d` or `from .x import y, z` line between tokens `k` and `stop`.
    fn py_import(&mut self, k: usize, stop: usize) {
        let line = self.tok(k).map_or(0, |t| t.line);
        let dotted = |s: &Self, mut j: usize| -> (String, usize) {
            let mut name = String::new();
            while j < stop {
                let Some(t) = s.tok(j) else { break };
                if t.text == "import" || t.text == "as" {
                    break;
                }
                if t.text == "." || (t.kind == Kind::Ident && (name.is_empty() || name.ends_with('.'))) {
                    name.push_str(&t.text);
                    j += 1;
                } else {
                    break;
                }
            }
            (name, j)
        };
        if self.ident(k) == Some("import") {
            let mut j = k + 1;
            while j < stop {
                let (name, next) = dotted(self, j);
                if name.is_empty() {
                    j += 1;
                    continue;
                }
                self.out.imports.push(Import { spec: name, line, names: Vec::new() });
                j = next;
                if self.ident(j) == Some("as") {
                    j += 2;
                }
            }
            return;
        }
        let (module, j) = dotted(self, k + 1);
        if self.ident(j) != Some("import") {
            return;
        }
        let mut names = Vec::new();
        let mut x = j + 1;
        while x < stop {
            if let Some(w) = self.ident(x)
                && !self.is(x - 1, "as")
                && w != "as"
            {
                names.push(w.to_owned());
            }
            x += 1;
        }
        self.out.imports.push(Import { spec: module, line, names });
    }
}

pub(crate) fn check() -> Result<(), String> {
    let got = |path: &str, src: &str| -> Vec<String> {
        tag(path, src)
            .map(|t| {
                t.symbols.iter().map(|s| format!("{} {} {}-{} {}", s.kind, s.name, s.line, s.end, s.parent).trim_end().to_owned()).collect()
            })
            .unwrap_or_default()
    };
    let ts = "import { a } from \"./a\";\nexport interface Shape {\n  area(): number;\n}\nexport class Box implements Shape {\n  area(): number {\n    return 1;\n  }\n  scale = (k: number) => k * 2;\n}\nexport const make = (n: number): Box => {\n  return new Box();\n};\nconst LIMIT = 4;\ntype Id = string | number;\n";
    let want = [
        "interface Shape 2-4",
        "class Box 5-10",
        "method area 6-8 Box",
        "method scale 9-9 Box",
        "function make 11-13",
        "const LIMIT 14-14",
        "type Id 15-15",
    ];
    if got("x.ts", ts) != want {
        return Err(format!("nmsubs.tags: typescript {:?}", got("x.ts", ts)));
    }
    let py = "import os\nfrom .util import load, save as keep\nMAX = 3\n\nclass Store:\n    def get(self, k):\n        return k\n\n    async def put(self, k):\n        pass\n\ndef main():\n    s = Store()\n";
    let want = ["const MAX 3-3", "class Store 5-10", "method get 6-7 Store", "method put 9-10 Store", "function main 12-13"];
    if got("x.py", py) != want {
        return Err(format!("nmsubs.tags: python {:?}", got("x.py", py)));
    }
    let imports: Vec<String> =
        tag("x.py", py).map(|t| t.imports.iter().map(|i| format!("{} {}", i.spec, i.names.join(","))).collect()).unwrap_or_default();
    if imports != ["os ", ".util load,save"] {
        return Err(format!("nmsubs.tags: python imports {imports:?}"));
    }
    let c = "#include \"a.h\"\n#define BLOCK 256\ntypedef struct {\n  int n;\n} Pair;\nstatic int add(int a, int b) {\n  return a + b;\n}\n__global__ void scale_k(float* x, int n) {\n  x[0] *= n;\n}\n";
    let want = ["macro BLOCK 2-2", "type Pair 3-5", "function add 6-8", "function scale_k 9-11"];
    if got("k.cu", c) != want {
        return Err(format!("nmsubs.tags: cuda {:?}", got("k.cu", c)));
    }
    let ps = "function Get-Thing {\n  param($a)\n  $a\n}\n";
    if got("t.ps1", ps) != ["function Get-Thing 1-4"] {
        return Err(format!("nmsubs.tags: powershell {:?}", got("t.ps1", ps)));
    }
    let css = ":root {\n  --accent: #fff;\n}\n.panel, .panel > h2 {\n  color: red;\n}\n";
    if got("a.css", css) != ["selector :root 1-3", "variable --accent 2-2 :root", "selector .panel 4-6", "selector .panel > h2 4-6"] {
        return Err(format!("nmsubs.tags: css {:?}", got("a.css", css)));
    }
    Ok(())
}
