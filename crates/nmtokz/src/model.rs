//! Tokenizer assets, literal splitting and ranked byte pair merges.
use crate::{
    Error, Result,
    added::{self, Piece, Vocabulary},
    json::{self, Value},
    normalize, split,
};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

/// The supported pre-tokenizers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Split {
    /// Qwen: the Unicode `Split` pattern, then `ByteLevel` without its regex.
    Qwen,
    /// Qwen3.5: the same with marks joining letter runs.
    QwenMarks,
    /// GPT-2 `ByteLevel` with its regex and no prefix space (`ModernBERT`).
    Gpt2,
}
/// Planted faults for the gate decoys; `None` is the tokenizer itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenFault {
    /// No fault.
    None,
    /// The Qwen split in place of the asset's own.
    QwenSplit,
    /// Every added token in one pass on the raw text, strip flags ignored.
    OnePass,
}
/// A byte BPE tokenizer loaded from the model's own assets.
#[derive(Debug)]
pub struct Tokenizer {
    vocab: HashMap<String, i32>,
    ranks: HashMap<(String, String), usize>,
    added: Vocabulary,
    bytes: HashMap<i32, Vec<u8>>,
    encoder: Vec<char>,
    normalize: bool,
    split: Split,
}
fn id(value: &Value) -> Result<i32> {
    i32::try_from(value.integer()?).map_err(|e| Error::Invalid(format!("token id: {e}")))
}
fn disabled(model: &Value, key: &str) -> Result<bool> {
    Ok(model
        .object()?
        .get(key)
        .is_none_or(|v| matches!(v, Value::Null | Value::Bool(false)) || matches!(v,Value::String(s) if s.is_empty())))
}
fn pre_tokenizer(data: &Value) -> Result<Split> {
    let pre = data.get("pre_tokenizer")?;
    if pre.get("type")?.string()? == "ByteLevel" {
        let fields = pre.object()?;
        let regex = matches!(fields.get("use_regex"), Some(Value::Bool(true)));
        let prefix = !matches!(fields.get("add_prefix_space"), Some(Value::Bool(false)));
        let known = fields.keys().all(|k| ["type", "add_prefix_space", "trim_offsets", "use_regex"].contains(&k.as_str()));
        if !regex || prefix || !known {
            return Err(Error::Invalid("ByteLevel needs use_regex true and add_prefix_space false".into()));
        }
        return Ok(Split::Gpt2);
    }
    qwen_pre_tokenizer(pre)
}
fn validate(data: &Value) -> Result<(bool, Split)> {
    let model = data.get("model")?;
    if model.get("type")?.string()? != "BPE" {
        return Err(Error::Invalid("model must be BPE".into()));
    }
    for key in ["dropout", "byte_fallback", "ignore_merges", "continuing_subword_prefix", "end_of_word_suffix"] {
        if !disabled(model, key)? {
            return Err(Error::Invalid(format!("unsupported BPE option {key}")));
        }
    }
    let split = pre_tokenizer(data)?;
    normalizer(data).map(|n| (n, split))
}
fn qwen_pre_tokenizer(pre: &Value) -> Result<Split> {
    let stages = pre.get("pretokenizers")?.array()?;
    if pre.get("type")?.string()? != "Sequence" || stages.len() != 2 {
        return Err(Error::Invalid("expected Split then ByteLevel sequence".into()));
    }
    let expected = |pattern: &str| {
        format!("{{\"type\":\"Split\",\"pattern\":{{\"Regex\":{}}},\"behavior\":\"Isolated\",\"invert\":false}}", quote_pattern(pattern))
    };
    let byte = json::parse(br#"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}"#)?;
    if stages[1] != byte {
        return Err(Error::Invalid("unsupported pre-tokenizer pattern or flags".into()));
    }
    if stages[0] == json::parse(expected(split::PATTERN).as_bytes())? {
        return Ok(Split::Qwen);
    }
    if stages[0] == json::parse(expected(split::PATTERN_MARKS).as_bytes())? {
        return Ok(Split::QwenMarks);
    }
    Err(Error::Invalid("unsupported pre-tokenizer pattern or flags".into()))
}
fn normalizer(data: &Value) -> Result<bool> {
    match data.object()?.get("normalizer") {
        None | Some(Value::Null) => Ok(false),
        Some(v) if v.object().is_ok_and(|m| m.len() == 1) && v.get("type").and_then(Value::string).is_ok_and(|s| s == "NFC") => Ok(true),
        _ => Err(Error::Invalid("unsupported normalizer, require NFC or null".into())),
    }
}
fn quote_pattern(pattern: &str) -> String {
    format!("\"{}\"", pattern.replace('\\', "\\\\").replace('"', "\\\""))
}
fn byte_map() -> Result<Vec<char>> {
    let mut bytes: Vec<u32> = (33..127).chain(161..173).chain(174..256).collect();
    let mut points = bytes.clone();
    for byte in 0..256 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            points.push(256 + u32::try_from(points.len() - 188).map_err(|e| Error::Invalid(e.to_string()))?);
        }
    }
    let mut out = vec!['\0'; 256];
    for (b, p) in bytes.into_iter().zip(points) {
        out[usize::try_from(b).map_err(|e| Error::Invalid(e.to_string()))?] =
            char::from_u32(p).ok_or_else(|| Error::Invalid("invalid byte map".into()))?;
    }
    Ok(out)
}
impl Tokenizer {
    /// Read tokenizer.json in place and validate the supported pipeline.
    /// # Errors
    /// Returns errors for unreadable, malformed or unsupported assets.
    pub fn open(path: &Path) -> Result<Self> {
        let path = if path.is_dir() { path.join("tokenizer.json") } else { path.to_owned() };
        let raw = std::fs::read(&path).map_err(|source| Error::Io { context: path.display().to_string(), source })?;
        let data = json::parse(&raw)?;
        let (normalize, split) = validate(&data)?;
        let encoder = byte_map()?;
        let decoder: HashMap<_, _> = encoder
            .iter()
            .enumerate()
            .map(|(i, c)| u8::try_from(i).map(|b| (*c, b)))
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::Invalid(e.to_string()))?;
        let mut vocab = HashMap::new();
        let mut bytes = HashMap::new();
        // Entries outside the byte map must be added tokens of the same text (`ModernBERT`
        // keeps its space-run tokens in the vocabulary as plain text).
        let mut plain = Vec::new();
        for (token, value) in data.get("model")?.get("vocab")?.object()? {
            let id = id(value)?;
            let Some(raw) = token.chars().map(|c| decoder.get(&c).copied()).collect::<Option<Vec<_>>>() else {
                plain.push((token.clone(), id));
                vocab.insert(token.clone(), id);
                continue;
            };
            if bytes.insert(id, raw).is_some() {
                return Err(Error::Invalid(format!("duplicate vocabulary id {id}")));
            }
            vocab.insert(token.clone(), id);
        }
        let ranks = merges(data.get("model")?.get("merges")?)?;
        let added = Vocabulary::load(&data, &mut bytes)?;
        for (token, id) in plain {
            if bytes.get(&id).map(Vec::as_slice) != Some(token.as_bytes()) {
                return Err(Error::Invalid(format!("unknown byte-map characters in vocabulary entry {token:?}")));
            }
        }
        Ok(Self { vocab, ranks, added, bytes, encoder, normalize, split })
    }
    /// Encode text with literal added tokens and optional NFC normalization.
    /// # Errors
    /// Fails when an asset lacks a byte symbol or merged vocabulary entry.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        self.encode_with(text, TokenFault::None)
    }
    /// [`Self::encode`] with a planted fault (gate decoys only).
    /// # Errors
    /// As [`Self::encode`].
    pub fn encode_with(&self, text: &str, fault: TokenFault) -> Result<Vec<i32>> {
        self.encode_tokens(text, fault, &self.added)
    }
    /// Encode special-token spellings as ordinary text, preserving ordinary added tokens.
    /// # Errors
    /// As [`Self::encode`].
    pub fn encode_plain(&self, text: &str) -> Result<Vec<i32>> {
        self.encode_tokens(text, TokenFault::None, &self.added.without_special())
    }
    /// Whether `text` holds no added-token spelling, raw or after normalization, so that its
    /// ids come from the ordinary vocabulary alone (used by the token-id hand-off).
    /// # Errors
    /// As [`Self::encode`].
    pub fn is_plain(&self, text: &str) -> Result<bool> {
        for piece in added::split(text, &self.added.raw)? {
            let Piece::Text(span) = piece else {
                return Ok(false);
            };
            let normalized = if self.normalize { normalize::nfc(span) } else { span.to_owned() };
            if added::split(&normalized, &self.added.normalized)?.iter().any(|p| matches!(p, Piece::Token(_))) {
                return Ok(false);
            }
        }
        Ok(true)
    }
    /// Rows of the ordinary (byte-level BPE) vocabulary; added tokens sit above them.
    #[must_use]
    pub fn ordinary_rows(&self) -> usize {
        self.vocab.len()
    }
    fn encode_tokens(&self, text: &str, fault: TokenFault, added: &Vocabulary) -> Result<Vec<i32>> {
        let mut result = Vec::new();
        let one_pass = (fault == TokenFault::OnePass).then(|| added.merged_plain());
        let first = one_pass.as_deref().unwrap_or(&added.raw);
        for piece in added::split(text, first)? {
            let span = match piece {
                Piece::Token(id) => {
                    result.push(id);
                    continue;
                }
                Piece::Text(span) => span,
            };
            let normalized = if self.normalize { normalize::nfc(span) } else { span.to_owned() };
            let second: &[added::Added] = if one_pass.is_some() { &[] } else { &added.normalized };
            for inner in added::split(&normalized, second)? {
                match inner {
                    Piece::Token(id) => result.push(id),
                    Piece::Text(text) => self.ordinary(text, fault, &mut result)?,
                }
            }
        }
        Ok(result)
    }
    fn ordinary(&self, text: &str, fault: TokenFault, result: &mut Vec<i32>) -> Result<()> {
        let pieces = if self.split == Split::Gpt2 && fault != TokenFault::QwenSplit {
            split::gpt2_pieces(text)
        } else {
            split::pieces(text, self.split == Split::QwenMarks && fault != TokenFault::QwenSplit)
        };
        for piece in pieces {
            result.extend(self.bpe(&piece)?);
        }
        Ok(())
    }
    /// Decode UTF-8 with replacement, preserving added tokens and ignoring unknown IDs.
    pub fn decode(&self, ids: &[i32]) -> String {
        let raw: Vec<u8> = ids.iter().filter_map(|id| self.bytes.get(id)).flatten().copied().collect();
        String::from_utf8_lossy(&raw).into_owned()
    }
    fn offer(
        &self,
        left: usize,
        symbols: &[String],
        next: &[usize],
        alive: &[bool],
        heap: &mut BinaryHeap<Reverse<(usize, usize, usize)>>,
    ) {
        if left < symbols.len()
            && alive[left]
            && next[left] < symbols.len()
            && let Some(rank) = self.ranks.get(&(symbols[left].clone(), symbols[next[left]].clone()))
        {
            heap.push(Reverse((*rank, left, next[left])));
        }
    }
    fn bpe(&self, piece: &str) -> Result<Vec<i32>> {
        let mut symbols: Vec<String> = piece.bytes().map(|b| self.encoder[usize::from(b)].to_string()).collect();
        let size = symbols.len();
        let mut previous: Vec<_> = (0..size).map(|i| i.checked_sub(1).unwrap_or(size)).collect();
        let mut next: Vec<_> = (1..=size).collect();
        let mut alive = vec![true; size];
        let mut heap = BinaryHeap::new();
        for i in 0..size {
            self.offer(i, &symbols, &next, &alive, &mut heap);
        }
        while let Some(Reverse((rank, left, right))) = heap.pop() {
            if !alive[left]
                || !alive[right]
                || next[left] != right
                || self.ranks.get(&(symbols[left].clone(), symbols[right].clone())) != Some(&rank)
            {
                continue;
            }
            let right_symbol = symbols[right].clone();
            symbols[left].push_str(&right_symbol);
            alive[right] = false;
            next[left] = next[right];
            if next[right] < size {
                previous[next[right]] = left;
            }
            self.offer(previous[left], &symbols, &next, &alive, &mut heap);
            self.offer(left, &symbols, &next, &alive, &mut heap);
        }
        symbols
            .iter()
            .zip(alive)
            .filter(|(_, live)| *live)
            .map(|(s, _)| self.vocab.get(s).copied().ok_or_else(|| Error::Invalid(format!("BPE symbol missing from vocabulary: {s:?}"))))
            .collect()
    }
}
fn merges(value: &Value) -> Result<HashMap<(String, String), usize>> {
    let mut out = HashMap::new();
    for (rank, item) in value.array()?.iter().enumerate() {
        let pair = match item {
            Value::String(s) => s.split(' ').map(str::to_owned).collect::<Vec<_>>(),
            Value::Array(a) => a.iter().map(|v| v.string().map(str::to_owned)).collect::<Result<Vec<_>>>()?,
            _ => return Err(Error::Invalid("invalid BPE merge".into())),
        };
        if pair.len() != 2 {
            return Err(Error::Invalid("merge requires two symbols".into()));
        }
        if out.insert((pair[0].clone(), pair[1].clone()), rank).is_some() {
            return Err(Error::Invalid("duplicate merge pair".into()));
        }
    }
    Ok(out)
}
