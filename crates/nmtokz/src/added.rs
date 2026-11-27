//! Added tokens, matched literally and leftmost-longest as the oracle's added vocabulary
//! does: tokens that are not normalized match the raw text, the normalized ones match each
//! remaining span after normalization. `lstrip` and `rstrip` take the adjacent whitespace.
use crate::{Error, Result, json::Value};
use std::cmp::Reverse;
use std::collections::HashMap;

/// One added token.
#[derive(Debug, Clone)]
pub(crate) struct Added {
    pub text: String,
    pub id: i32,
    lstrip: bool,
    rstrip: bool,
    special: bool,
}
/// The added tokens, split by the pass that matches them.
#[derive(Debug, Default)]
pub(crate) struct Vocabulary {
    /// Matched on the raw text.
    pub raw: Vec<Added>,
    /// Matched after normalization.
    pub normalized: Vec<Added>,
}
/// A span of text left for the pre-tokenizer, or a matched token id.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Piece<'a> {
    Text(&'a str),
    Token(i32),
}
fn enabled(item: &Value, key: &str) -> Result<bool> {
    Ok(matches!(item.object()?.get(key), Some(Value::Bool(true))))
}
impl Vocabulary {
    /// Read `added_tokens`; bytes of each token go to the decode table.
    pub(crate) fn load(data: &Value, bytes: &mut HashMap<i32, Vec<u8>>) -> Result<Self> {
        let mut out = Self::default();
        let mut seen = Vec::new();
        for item in data.get("added_tokens")?.array()? {
            if enabled(item, "single_word")? {
                return Err(Error::Invalid("unsupported added-token flag single_word".into()));
            }
            let text = item.get("content")?.string()?.to_owned();
            let id = i32::try_from(item.get("id")?.integer()?).map_err(|e| Error::Invalid(format!("token id: {e}")))?;
            if text.is_empty() || seen.contains(&text) {
                return Err(Error::Invalid("empty or duplicate added token".into()));
            }
            seen.push(text.clone());
            bytes.insert(id, text.as_bytes().to_vec());
            let token =
                Added { text, id, lstrip: enabled(item, "lstrip")?, rstrip: enabled(item, "rstrip")?, special: enabled(item, "special")? };
            if enabled(item, "normalized")? {
                out.normalized.push(token);
            } else {
                out.raw.push(token);
            }
        }
        // Longest first gives leftmost-longest; the stable sort keeps asset order on ties.
        out.raw.sort_by_key(|t| Reverse(t.text.chars().count()));
        out.normalized.sort_by_key(|t| Reverse(t.text.chars().count()));
        Ok(out)
    }
    /// Every token in one list, strip flags off: the one-pass decoy.
    pub(crate) fn merged_plain(&self) -> Vec<Added> {
        let mut all: Vec<Added> =
            self.raw.iter().chain(&self.normalized).map(|t| Added { lstrip: false, rstrip: false, ..t.clone() }).collect();
        all.sort_by_key(|t| Reverse(t.text.chars().count()));
        all
    }
    /// Ordinary added tokens remain literal when special tokens are treated as text.
    pub(crate) fn without_special(&self) -> Self {
        let keep = |tokens: &[Added]| tokens.iter().filter(|t| !t.special).cloned().collect();
        Self { raw: keep(&self.raw), normalized: keep(&self.normalized) }
    }
}
fn slice(text: &str, from: usize, to: usize) -> Result<&str> {
    text.get(from..to).ok_or_else(|| Error::Invalid("invalid text boundary".into()))
}
/// Split `text` on `tokens`, leftmost-longest.
pub(crate) fn split<'a>(text: &'a str, tokens: &[Added]) -> Result<Vec<Piece<'a>>> {
    let mut out = Vec::new();
    let (mut start, mut cursor) = (0, 0);
    while cursor < text.len() {
        let tail = slice(text, cursor, text.len())?;
        let Some(token) = tokens.iter().find(|t| tail.starts_with(&t.text)) else {
            cursor += tail.chars().next().map_or(1, char::len_utf8);
            continue;
        };
        let before = slice(text, start, cursor)?;
        let begin = if token.lstrip { start + before.trim_end().len() } else { cursor };
        let mut stop = cursor + token.text.len();
        if token.rstrip {
            let rest = slice(text, stop, text.len())?;
            stop += rest.len() - rest.trim_start().len();
        }
        if begin > start {
            out.push(Piece::Text(slice(text, start, begin)?));
        }
        out.push(Piece::Token(token.id));
        (start, cursor) = (stop, stop);
    }
    if start < text.len() {
        out.push(Piece::Text(slice(text, start, text.len())?));
    }
    Ok(out)
}
#[test]
fn strip_and_longest() -> Result<()> {
    let data = crate::json::parse(
        br#"{"added_tokens":[{"id":1,"content":"[M]","lstrip":true,"normalized":false},
        {"id":2,"content":"  ","normalized":true},{"id":3,"content":"   ","normalized":true}]}"#,
    )?;
    let vocab = Vocabulary::load(&data, &mut HashMap::new())?;
    assert_eq!(split("a  [M]b", &vocab.raw)?, [Piece::Text("a"), Piece::Token(1), Piece::Text("b")], "lstrip takes the spaces");
    assert_eq!(
        split("x     y", &vocab.normalized)?,
        [Piece::Text("x"), Piece::Token(3), Piece::Token(2), Piece::Text("y")],
        "longest first"
    );
    Ok(())
}
