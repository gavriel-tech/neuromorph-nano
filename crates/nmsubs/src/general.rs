//! Which words of a question can name a concept the repository lacks.
//!
//! The withholding rule (`Substrate::absent_terms`) counts a question term as an absent concept
//! only when it is a content term. A term is a content term when it is written like an
//! identifier (`camelCase`, `snake_case`, digits, `::`, an inner capital, an acronym), or when it
//! is an ordinary word that is neither a function word nor a common verb. Function words and
//! irregular verb forms are closed classes and listed here. Common verbs are recognised by the
//! system's own measure of general use: the base model's tokenizer vocabulary. Byte-level BPE
//! ids follow the merge order, and merges follow frequency in the model's training text, so a
//! word that is one early token is frequent in general use. A word counts as a common verb when
//! one of its stems and that stem's `-ing` form are both such early tokens (`happens`: `happen`
//! and `happening`; `give`: `give` and `giving`). Frequent nouns without an `-ing` form
//! (`password`, `payment`) stay content terms.
use std::collections::BTreeMap;

/// Tokens merged before this id count as frequent in general use. Qwen3's vocabulary holds about
/// 25,000 lower-case English words as single tokens; the first 30,000 ids cover everyday English
/// (`happening` 12,482, `stored` 9,768, `parse` 4,715) while technical names come later
/// (`tokenizer` 45,958, `websocket` 58,943) or are not single tokens at all.
pub const COMMON_RANK: u32 = 30_000;

/// Function words a question is built from: articles, pronouns, prepositions, conjunctions,
/// auxiliaries, determiners, question words and common adverbs.
const FUNCTION: &[&str] = &[
    "a",
    "about",
    "above",
    "across",
    "after",
    "again",
    "against",
    "all",
    "almost",
    "along",
    "already",
    "also",
    "although",
    "always",
    "am",
    "among",
    "an",
    "and",
    "another",
    "any",
    "anything",
    "are",
    "around",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "behind",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "down",
    "during",
    "each",
    "either",
    "else",
    "enough",
    "even",
    "ever",
    "every",
    "everything",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "however",
    "i",
    "if",
    "in",
    "inside",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "least",
    "less",
    "many",
    "may",
    "me",
    "might",
    "more",
    "most",
    "much",
    "must",
    "my",
    "near",
    "neither",
    "never",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "often",
    "on",
    "once",
    "one",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "our",
    "ours",
    "out",
    "outside",
    "over",
    "own",
    "per",
    "perhaps",
    "please",
    "plenty",
    "quite",
    "rather",
    "same",
    "shall",
    "she",
    "should",
    "several",
    "since",
    "so",
    "some",
    "something",
    "somewhere",
    "still",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "therefore",
    "these",
    "they",
    "thing",
    "things",
    "this",
    "those",
    "though",
    "through",
    "thus",
    "to",
    "together",
    "too",
    "toward",
    "towards",
    "under",
    "unless",
    "until",
    "up",
    "upon",
    "us",
    "very",
    "various",
    "via",
    "was",
    "we",
    "were",
    "what",
    "whatever",
    "when",
    "whenever",
    "where",
    "whereas",
    "wherever",
    "whether",
    "which",
    "while",
    "who",
    "whoever",
    "whole",
    "whom",
    "whose",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "yes",
    "yet",
    "you",
    "your",
    "yours",
    "yourself",
];

/// Irregular verb forms (past and participle), a closed class the `-ing` test cannot reach.
const IRREGULAR: &[&str] = &[
    "ate",
    "became",
    "become",
    "began",
    "begun",
    "bent",
    "bit",
    "bitten",
    "blew",
    "blown",
    "bought",
    "bound",
    "broke",
    "broken",
    "brought",
    "built",
    "came",
    "caught",
    "chose",
    "chosen",
    "come",
    "dealt",
    "drawn",
    "drew",
    "driven",
    "drove",
    "eaten",
    "fed",
    "fell",
    "felt",
    "fled",
    "flew",
    "flown",
    "forgot",
    "forgotten",
    "found",
    "froze",
    "frozen",
    "gave",
    "given",
    "gone",
    "got",
    "gotten",
    "grew",
    "grown",
    "held",
    "hid",
    "hidden",
    "hung",
    "kept",
    "knew",
    "known",
    "laid",
    "lain",
    "led",
    "left",
    "lent",
    "lost",
    "made",
    "meant",
    "met",
    "paid",
    "ran",
    "rang",
    "rode",
    "rose",
    "risen",
    "rung",
    "said",
    "sang",
    "sat",
    "saw",
    "seen",
    "sent",
    "shook",
    "shot",
    "shown",
    "slept",
    "sold",
    "sought",
    "spent",
    "spoke",
    "spoken",
    "stood",
    "stole",
    "stolen",
    "struck",
    "sung",
    "swept",
    "taken",
    "taught",
    "thought",
    "threw",
    "thrown",
    "told",
    "took",
    "tore",
    "torn",
    "understood",
    "went",
    "woke",
    "woken",
    "won",
    "wore",
    "worn",
    "wound",
    "wrote",
    "written",
];

/// Words that ask for a form of answer (a summary, an explanation, a comparison) rather than name a
/// concept of the material.
const REQUEST: &[&str] = &[
    "compare",
    "describe",
    "explain",
    "outline",
    "overview",
    "summarise",
    "summarize",
    "summary",
];

/// Inflection endings stripped to find a verb's stem.
const ENDINGS: [&str; 7] = ["ing", "es", "ed", "en", "s", "d", "n"];

/// Frequency of words in general use, read from the base model's tokenizer vocabulary.
#[derive(Clone, Debug, Default)]
pub struct GeneralUse {
    /// Lower-case ASCII words that are one token with a leading space, by token id.
    ranks: BTreeMap<String, u32>,
}

impl GeneralUse {
    /// Reads a byte-level BPE vocabulary (`vocab.json`: token to id). Only lower-case ASCII
    /// words with the leading-space marker are kept.
    /// # Errors
    /// The bytes are not a JSON object of token ids.
    pub fn from_vocab(bytes: &[u8]) -> Result<Self, String> {
        let v = nmjson::parse(bytes).map_err(|e| format!("vocabulary: {e}"))?;
        let nmjson::Value::Object(map) = v else { return Err("vocabulary: not an object".into()) };
        let ranks = map
            .into_iter()
            .filter_map(|(k, id)| {
                let w = k.strip_prefix('\u{120}')?;
                let id = id.integer().ok().and_then(|n| u32::try_from(n).ok())?;
                (w.len() >= 2 && w.bytes().all(|b| b.is_ascii_lowercase())).then(|| (w.to_owned(), id))
            })
            .collect();
        Ok(Self { ranks })
    }
    /// A vocabulary from (word, id) pairs, for checks.
    #[must_use]
    pub fn from_pairs(pairs: &[(&str, u32)]) -> Self {
        Self { ranks: pairs.iter().map(|(w, r)| ((*w).to_owned(), *r)).collect() }
    }
    /// Words read.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranks.len()
    }
    /// True when no word was read.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }
    /// Words of general use merged before `COMMON_RANK` (the size of the common vocabulary).
    #[must_use]
    pub fn common_count(&self) -> usize {
        self.ranks.values().filter(|r| **r < COMMON_RANK).count()
    }
    /// True when `w` is a word of general use: it, or one of its stems, is one token of the
    /// vocabulary (at any rank). Such a word describes; a word outside general use (a name, a code,
    /// a term of art the vocabulary splits) names a thing.
    #[must_use]
    pub fn general_word(&self, w: &str) -> bool {
        stems(w).iter().any(|s| self.ranks.contains_key(s))
    }
    /// True when `w` is one token merged before `COMMON_RANK`.
    #[must_use]
    pub fn frequent(&self, w: &str) -> bool {
        self.ranks.get(w).is_some_and(|r| *r < COMMON_RANK)
    }
    /// True when `w` is a form of a frequent verb: one of its stems and that stem's `-ing` form
    /// are both frequent. A frequent `-ly` adverb counts likewise (it never names a concept).
    #[must_use]
    pub fn common_verb(&self, w: &str) -> bool {
        if w.ends_with("ly") && self.frequent(w) {
            return true;
        }
        stems(w).iter().any(|s| self.frequent(s) && ing_forms(s).iter().any(|f| self.frequent(f)))
    }
}

/// Candidate stems of `w`: the word itself and, for each inflection ending, the rest, the rest
/// with a final `e`, and both again with a doubled last letter undone (`running`, `written`).
fn stems(w: &str) -> Vec<String> {
    let mut out = vec![w.to_owned()];
    for e in ENDINGS {
        let Some(base) = w.strip_suffix(e).filter(|b| b.len() >= 2) else { continue };
        let b = base.as_bytes();
        let undoubled = (b.len() >= 3 && b.last() == b.get(b.len() - 2)).then(|| base.get(..base.len() - 1)).flatten();
        for s in std::iter::once(base).chain(undoubled) {
            for c in [s.to_owned(), format!("{s}e")] {
                if !out.contains(&c) {
                    out.push(c);
                }
            }
        }
    }
    out
}

/// The `-ing` forms a stem can take: added, with a final `e` dropped, or with the last letter
/// doubled.
fn ing_forms(s: &str) -> Vec<String> {
    let mut out = vec![format!("{s}ing")];
    if let Some(b) = s.strip_suffix('e') {
        out.push(format!("{b}ing"));
    }
    if let Some(c) = s.chars().last() {
        out.push(format!("{s}{c}ing"));
    }
    out
}

/// True for a function word or an irregular verb form.
#[must_use]
pub fn closed_class(w: &str) -> bool {
    FUNCTION.contains(&w) || IRREGULAR.contains(&w) || REQUEST.contains(&w)
}

/// True when a word as written in the question looks like an identifier: an underscore, `::`, a
/// digit, a capital after the first letter (`camelCase`, `GraphQL`), or two or more capitals
/// (`GPU`).
#[must_use]
pub fn identifier_like(raw: &str) -> bool {
    raw.contains('_')
        || raw.contains("::")
        || raw.chars().any(|c| c.is_ascii_digit())
        || raw.chars().skip(1).any(char::is_uppercase)
        || raw.chars().filter(|c| c.is_uppercase()).count() >= 2
}

/// True when a question term can name a concept: identifier-like as written, or an ordinary word
/// that is neither a function word nor (with a vocabulary) a common verb. Without a vocabulary,
/// only the closed classes are excluded.
#[must_use]
pub fn content_term(raw: &str, low: &str, general: Option<&GeneralUse>) -> bool {
    if identifier_like(raw) {
        return true;
    }
    if closed_class(low) {
        return false;
    }
    !general.is_some_and(|g| g.common_verb(low))
}

/// The parts of a `camelCase` or `PascalCase` word, lower-cased (`takeFromShelf`: take, from,
/// shelf); a word with no inner capital gives nothing.
#[must_use]
pub fn camel_parts(raw: &str) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = raw.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        let next_lower = chars.get(i + 1).is_some_and(|n| n.is_lowercase());
        let prev_lower = i > 0 && chars.get(i - 1).is_some_and(|p| p.is_lowercase() || p.is_ascii_digit());
        if c.is_uppercase() && !cur.is_empty() && (prev_lower || next_lower) {
            parts.push(cur.to_lowercase());
            cur.clear();
        }
        if c.is_alphanumeric() {
            cur.push(*c);
        }
    }
    if !cur.is_empty() {
        parts.push(cur.to_lowercase());
    }
    if parts.len() > 1 { parts } else { Vec::new() }
}

/// Section check: common verbs and function words are not content terms, frequent nouns and
/// identifiers are, and camel case splits.
pub(crate) fn check() -> Result<(), String> {
    let g = GeneralUse::from_pairs(&[
        ("happen", 3537),
        ("happening", 12482),
        ("give", 2968),
        ("giving", 7086),
        ("write", 3270),
        ("writing", 4378),
        ("run", 1598),
        ("running", 4303),
        ("password", 3552),
        ("payment", 8160),
        ("frame", 4034),
        ("framing", 57949),
        ("quickly", 6157),
    ]);
    for w in ["happens", "give", "given", "written", "running", "quickly", "the", "kept", "many", "summarise"] {
        if content_term(w, w, Some(&g)) {
            return Err(format!("nmsubs.general: {w} counted as a content term"));
        }
    }
    for (raw, low) in [
        ("password", "password"),
        ("payment", "payment"),
        ("frame", "frame"),
        ("GraphQL", "graphql"),
        ("take_from_shelf", "take_from_shelf"),
        ("zqxv", "zqxv"),
    ] {
        if !content_term(raw, low, Some(&g)) {
            return Err(format!("nmsubs.general: {raw} not counted as a content term"));
        }
    }
    if content_term("given", "given", None) || !content_term("happens", "happens", None) {
        return Err("nmsubs.general: closed classes without a vocabulary".into());
    }
    if camel_parts("takeFromShelf") != ["take", "from", "shelf"]
        || camel_parts("HTTPServer") != ["http", "server"]
        || !camel_parts("shelf").is_empty()
    {
        return Err(format!("nmsubs.general: camel parts {:?} {:?}", camel_parts("takeFromShelf"), camel_parts("HTTPServer")));
    }
    let v = GeneralUse::from_vocab("{\"\u{120}give\":2968,\"give\":46430,\"\u{120}Give\":5,\"\u{120}giving\":7086}".as_bytes())
        .map_err(|e| format!("nmsubs.general: {e}"))?;
    if !g.general_word("payments") || g.general_word("zqxv") || g.common_count() != 12 {
        return Err("nmsubs.general: general words and the common count".into());
    }
    if v.len() != 2 || !v.common_verb("gives") {
        return Err("nmsubs.general: vocabulary read".into());
    }
    Ok(())
}
