//! Text and token helpers shared by the substrate scanner, the drone tip readers and the code
//! tools. `lex.rs` is the Rust and C family scanner the Rust path has always used (a byte copy of
//! emma-labs `tools/stdchk/src/lex.rs`); `lang.rs` is the language table and `lang_lex.rs` the
//! generic lexer it drives for every other language; this file adds the term helpers.
pub mod lang;
pub mod lang_lex;
mod lex;

/// One token: text and 1-based line. Comments and string contents are skipped.
#[must_use]
pub fn tokens(source: &str, rust: bool) -> Vec<(String, usize)> {
    lex::tokens(source, rust).into_iter().map(|t| (t.text, t.line)).collect()
}
/// Lines that contain the `unsafe` keyword outside comments and literals.
#[must_use]
pub fn unsafe_lines(source: &str) -> Vec<usize> {
    lex::unsafe_lines(source)
}

/// Words a question is made of that carry no lookup meaning.
pub const STOP: &[&str] = &[
    "what",
    "where",
    "which",
    "who",
    "whom",
    "does",
    "did",
    "the",
    "and",
    "for",
    "are",
    "how",
    "why",
    "when",
    "this",
    "that",
    "with",
    "from",
    "into",
    "its",
    "use",
    "uses",
    "used",
    "using",
    "defined",
    "define",
    "defines",
    "definition",
    "crate",
    "crates",
    "function",
    "functions",
    "file",
    "files",
    "struct",
    "type",
    "module",
    "called",
    "name",
    "named",
    "there",
    "any",
    "depend",
    "depends",
    "dependency",
    "dependencies",
    "on",
    "is",
    "in",
    "of",
    "to",
    "a",
    "an",
    "do",
    "has",
    "have",
    "list",
    "show",
    "find",
    "all",
    "repo",
    "repository",
    "code",
    "about",
    "say",
    "says",
    "mean",
    "means",
    "work",
    "works",
    "happen",
    "get",
    "set",
    "can",
    "should",
    "would",
    "will",
    "not",
    "you",
    "our",
    "their",
    "it",
    "be",
    "by",
    "at",
    "as",
    "or",
    "if",
    "value",
    "constant",
    "line",
    "trait",
    "enum",
    "directly",
    "normal",
    "workspace",
    "emma-labs",
    // Words that say what kind of answer is wanted, not what it is about (2026-11-09): a line
    // naming the item answers "where is it tested" without containing "tested".
    "tested",
    "recently",
    "changed",
];

/// Lower-case identifier words, plus the parts of `snake_case` and `kebab-case` words.
#[must_use]
pub fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '.')) {
        let w = w.trim_matches(|c: char| c == '.' || c == '-').to_lowercase();
        if w.is_empty() {
            continue;
        }
        let parts: Vec<&str> = w.split(['_', '-', '.']).filter(|p| !p.is_empty()).collect();
        if parts.len() > 1 {
            out.extend(parts.iter().map(|p| (*p).to_owned()));
        }
        out.push(w);
    }
    out
}
/// A two-character word that looks like an identifier: an upper-case letter followed by an
/// upper-case letter or a digit (`GB`, `MB`, `K1`). Shorter words are never names; longer
/// ones follow the usual rules. (S3 found these dropped by the length-three rule; fix at S6.)
#[must_use]
pub fn short_name(w: &str) -> bool {
    let mut c = w.chars();
    matches!((c.next(), c.next(), c.next()), (Some(a), Some(b), None) if a.is_ascii_uppercase() && (b.is_ascii_uppercase() || b.is_ascii_digit()))
}
/// A word's stem for matching name parts: lower case, a common ending taken off (`releases`,
/// `forking`, `spans`), and a final `e` (`release` and `releases` meet).
#[must_use]
pub fn stem(w: &str) -> String {
    let w = w.to_lowercase();
    let mut s = w.as_str();
    for suf in ["ings", "ing", "ers", "er", "ed", "es", "s", "ly"] {
        if let Some(r) = s.strip_suffix(suf).filter(|r| r.len() >= 3) {
            s = r;
            break;
        }
    }
    s.strip_suffix('e').filter(|r| r.len() >= 3).unwrap_or(s).to_owned()
}
/// The parts of an identifier: split at underscores, dashes, path separators and lower-to-upper
/// case changes, lower-cased, empty parts left out.
#[must_use]
pub fn name_parts(name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for c in name.chars() {
        if c == '_' || c == '-' || c == ':' {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            prev_lower = false;
            continue;
        }
        if c.is_uppercase() && prev_lower && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        prev_lower = c.is_lowercase() || c.is_ascii_digit();
        cur.extend(c.to_lowercase());
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}
/// The words of a question that a correct answer must cover (stop words dropped).
#[must_use]
pub fn key_terms(q: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for w in q.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '.')) {
        let raw = w.trim_matches(|c: char| c == '.' || c == '-' || c == '?');
        let w = raw.to_lowercase();
        if (w.len() >= 3 || short_name(raw)) && !STOP.contains(&w.as_str()) && !out.contains(&w) {
            out.push(w);
        }
    }
    out
}
/// Words of `STOP` that are code words in a code question but content words in prose ("the
/// function of the bypass valve", "the value on the label", "the type of seal"). They stay out of
/// a question's terms when it names a code identifier or is scored against code. `file` and `line`
/// are not among them: "which file", "which line" ask where, in prose as in code.
pub const CODE_WORDS: &[&str] = &[
    "type",
    "value",
    "list",
    "name",
    "named",
    "function",
    "functions",
    "module",
    "constant",
    "definition",
    "work",
    "works",
];

/// The words of a question that a correct passage of a document must cover: as `key_terms`, but
/// the code words (`CODE_WORDS`) count as content words unless the question names a code
/// identifier, and numbers of two digits or more count (`25`, `3.2`, `1,200`, `25%`).
#[must_use]
pub fn key_terms_doc(q: &str) -> Vec<String> {
    let code_question = !names(q).is_empty();
    let mut out: Vec<String> = Vec::new();
    for w in q.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == ',' || c == '%')) {
        let raw = w.trim_matches(|c: char| c == '.' || c == '-' || c == '?' || c == ',');
        let w = raw.to_lowercase();
        let number = w.chars().filter(char::is_ascii_digit).count() >= 2 && w.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | ',' | '%'));
        if w.contains(',') && !number {
            for part in w.split(',') {
                if (part.len() >= 3 || short_name(part)) && !STOP.contains(&part) && !out.iter().any(|x| x == part) {
                    out.push(part.to_owned());
                }
            }
            continue;
        }
        let stop = STOP.contains(&w.as_str()) && (code_question || !CODE_WORDS.contains(&w.as_str()));
        if (number || ((w.len() >= 3 || short_name(raw)) && !stop)) && !out.contains(&w) {
            out.push(w);
        }
    }
    out
}

/// The document kind a question asks for, by its wording, when it names no code identifier:
/// `define` (what is X, what does X mean), `procedure` (how do I, steps to, walk me through),
/// `summarise` (summarise, overview, overall) or `compare` (difference between, compare, versus).
#[must_use]
pub fn doc_kind(q: &str) -> Option<&'static str> {
    if !names(q).is_empty() {
        return None;
    }
    let l = format!(" {} ", q.to_lowercase().replace(['?', ',', '.'], " "));
    let has = |xs: &[&str]| xs.iter().any(|x| l.contains(x));
    if has(&[" summar", " overview", " overall ", " in short ", " outline of "]) {
        return Some("summarise");
    }
    if has(&[" difference between ", " differences between ", " compare ", " differ ", " versus ", " vs "]) {
        return Some("compare");
    }
    if has(&[" how do i ", " how do you ", " how to ", " steps ", " step by step ", " walk me through ", " procedure for "]) {
        return Some("procedure");
    }
    if has(&[" what is meant by ", " meaning of ", " define ", " definition of "])
        || (l.contains(" what does ") && (l.contains(" mean ") || l.ends_with(" mean ")))
    {
        return Some("define");
    }
    // `what is X` with a short subject and no article: a definition, not a value lookup.
    if let Some(rest) = l.split(" what is ").nth(1).or_else(|| l.split(" what are ").nth(1)) {
        let words: Vec<&str> = rest.split_whitespace().collect();
        if !words.is_empty() && words.len() <= 3 && !words.iter().any(|w| matches!(*w, "the" | "this" | "that" | "it" | "our")) {
            return Some("define");
        }
    }
    None
}

/// Identifiers as written in the question (case kept): words with an upper-case letter past
/// the first character, an underscore or a dash, the likeliest names of symbols and crates.
#[must_use]
pub fn names(q: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for w in q.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-')) {
        let w = w.trim_matches('-');
        let looks = w.chars().skip(1).any(|c| c.is_uppercase()) || w.contains(['_', '-']);
        let looks = (looks && w.len() >= 3) || short_name(w);
        if looks && !out.iter().any(|x| x == w) {
            out.push(w.to_owned());
        }
    }
    out
}

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

fn check_lex() -> Result<(), String> {
    let t = tokens("// note fn x\nfn real() { let s = \"fn fake\"; }", true);
    let fns = t.iter().filter(|(x, _)| x == "fn").count();
    if fns != 1 || t.first().map(|x| x.1) != Some(2) {
        return Err(format!("nmlex.lex: comments or strings not skipped: {t:?}"));
    }
    if unsafe_lines("// unsafe\nlet a = 1;\nunsafe { }").as_slice() != [3] {
        return Err("nmlex.lex: unsafe_lines".into());
    }
    Ok(())
}
fn check_terms() -> Result<(), String> {
    if key_terms("Where is the struct ParMap defined?") != ["parmap"] {
        return Err("nmlex.terms: key_terms".into());
    }
    if names("What does emma-tokenz use for MAX_LEN?") != ["emma-tokenz", "MAX_LEN"] {
        return Err("nmlex.terms: names".into());
    }
    if names("What is GB set to, and K1?") != ["GB", "K1"] || key_terms("What is MB?") != ["mb"] || !names("Is it ok?").is_empty() {
        return Err("nmlex.terms: two-character names".into());
    }
    if key_terms_doc("What is the function of the bypass valve at 25%?") != ["function", "bypass", "valve", "25%"]
        || key_terms_doc("Which function calls MAX_LEN?") != ["calls", "max_len"]
        || key_terms_doc("Stock of 1,200 parts in section 3.2") != ["stock", "1,200", "parts", "section", "3.2"]
    {
        return Err(format!("nmlex.terms: document terms {:?}", key_terms_doc("Stock of 1,200 parts in section 3.2")));
    }
    let kinds = [
        ("What is emissivity?", Some("define")),
        ("What does priming the pump mean?", Some("define")),
        ("Walk me through descaling the pump.", Some("procedure")),
        ("Summarise the chapter.", Some("summarise")),
        ("What is the difference between A and B?", Some("compare")),
        ("What is the default decoy cut on the reranker's best score?", None),
        ("Which function calls zeta_pack?", None),
    ];
    for (q, want) in kinds {
        if doc_kind(q) != want {
            return Err(format!("nmlex.terms: doc_kind({q:?}) = {:?}", doc_kind(q)));
        }
    }
    Ok(())
}

/// Section checks of this crate, in order.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![("nmlex.lex", check_lex), ("nmlex.terms", check_terms), ("nmlex.lang", lang::check), ("nmlex.lang_lex", lang_lex::check)]
}
