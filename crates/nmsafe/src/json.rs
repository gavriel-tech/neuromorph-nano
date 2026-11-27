//! Strict, bounded JSON reader for local model assets.
use crate::{Error, Result};
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub(crate) enum Value {
    Object(BTreeMap<String, Self>),
    Array(Vec<Self>),
    String(String),
    Number(String),
    Bool(bool),
    Null,
}
impl Value {
    pub(crate) fn get(&self, key: &str) -> Result<&Self> {
        self.object()?.get(key).ok_or_else(|| Error::Invalid(format!("missing JSON key {key}")))
    }
    pub(crate) fn object(&self) -> Result<&BTreeMap<String, Self>> {
        if let Self::Object(v) = self { Ok(v) } else { Err(Error::Invalid("expected JSON object".into())) }
    }
    pub(crate) fn array(&self) -> Result<&[Self]> {
        if let Self::Array(v) = self { Ok(v) } else { Err(Error::Invalid("expected JSON array".into())) }
    }
    pub(crate) fn string(&self) -> Result<&str> {
        if let Self::String(v) = self { Ok(v) } else { Err(Error::Invalid("expected JSON string".into())) }
    }
    pub(crate) fn integer(&self) -> Result<u64> {
        if let Self::Number(v) = self {
            v.parse().map_err(|e| Error::Invalid(format!("invalid unsigned integer {v}: {e}")))
        } else {
            Err(Error::Invalid("expected JSON integer".into()))
        }
    }
}
pub(crate) fn parse(bytes: &[u8]) -> Result<Value> {
    // A leading UTF-8 byte-order mark is ignored (RFC 8259 section 8.1 allows it).
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    let text = std::str::from_utf8(bytes).map_err(|e| Error::Invalid(format!("JSON UTF-8: {e}")))?;
    let mut reader = Reader { chars: text.chars().collect(), at: 0 };
    let value = reader.value(0)?;
    reader.space();
    if reader.at != reader.chars.len() {
        return Err(Error::Invalid("trailing JSON input".into()));
    }
    Ok(value)
}
struct Reader {
    chars: Vec<char>,
    at: usize,
}
impl Reader {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.at).copied()
    }
    fn take(&mut self) -> Result<char> {
        let c = self.peek().ok_or_else(|| Error::Invalid("truncated JSON".into()))?;
        self.at += 1;
        Ok(c)
    }
    fn space(&mut self) {
        while self.peek().is_some_and(|c| matches!(c, ' ' | '\t' | '\r' | '\n')) {
            self.at += 1;
        }
    }
    fn eat(&mut self, c: char) -> bool {
        self.space();
        if self.peek() == Some(c) {
            self.at += 1;
            true
        } else {
            false
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > 64 {
            return Err(Error::Invalid("JSON depth exceeds 64".into()));
        }
        self.space();
        match self.peek() {
            Some('{') => self.object(depth + 1),
            Some('[') => self.array(depth + 1),
            Some('"') => self.string().map(Value::String),
            Some('-' | '0'..='9') => self.number(),
            Some('t') => self.literal("true", Value::Bool(true)),
            Some('f') => self.literal("false", Value::Bool(false)),
            Some('n') => self.literal("null", Value::Null),
            _ => Err(Error::Invalid("invalid JSON value".into())),
        }
    }
    fn literal(&mut self, text: &str, value: Value) -> Result<Value> {
        for c in text.chars() {
            if self.take()? != c {
                return Err(Error::Invalid("invalid JSON literal".into()));
            }
        }
        Ok(value)
    }
    fn object(&mut self, depth: usize) -> Result<Value> {
        self.at += 1;
        let mut map = BTreeMap::new();
        if self.eat('}') {
            return Ok(Value::Object(map));
        }
        loop {
            self.space();
            let key = self.string()?;
            if !self.eat(':') {
                return Err(Error::Invalid("missing JSON colon".into()));
            }
            if map.insert(key.clone(), self.value(depth)?).is_some() {
                return Err(Error::Invalid(format!("duplicate JSON key {key}")));
            }
            if self.eat('}') {
                break;
            }
            if !self.eat(',') {
                return Err(Error::Invalid("missing JSON comma".into()));
            }
        }
        Ok(Value::Object(map))
    }
    fn array(&mut self, depth: usize) -> Result<Value> {
        self.at += 1;
        let mut items = Vec::new();
        if self.eat(']') {
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.value(depth)?);
            if self.eat(']') {
                break;
            }
            if !self.eat(',') {
                return Err(Error::Invalid("missing JSON comma".into()));
            }
        }
        Ok(Value::Array(items))
    }
    fn digits(&mut self) -> Result<()> {
        let start = self.at;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.at += 1;
        }
        if start == self.at {
            return Err(Error::Invalid("missing JSON digits".into()));
        }
        Ok(())
    }
    fn number(&mut self) -> Result<Value> {
        let start = self.at;
        if self.peek() == Some('-') {
            self.at += 1;
        }
        if self.peek() == Some('0') {
            self.at += 1;
        } else {
            self.digits()?;
        }
        if self.peek() == Some('.') {
            self.at += 1;
            self.digits()?;
        }
        if self.peek().is_some_and(|c| matches!(c, 'e' | 'E')) {
            self.at += 1;
            if self.peek().is_some_and(|c| matches!(c, '+' | '-')) {
                self.at += 1;
            }
            self.digits()?;
        }
        Ok(Value::Number(self.chars[start..self.at].iter().collect()))
    }
    fn string(&mut self) -> Result<String> {
        if self.take()? != '"' {
            return Err(Error::Invalid("expected JSON string".into()));
        }
        let mut text = String::new();
        loop {
            match self.take()? {
                '"' => return Ok(text),
                '\\' => match self.take()? {
                    '"' => text.push('"'),
                    '\\' => text.push('\\'),
                    '/' => text.push('/'),
                    'b' => text.push('\u{8}'),
                    'f' => text.push('\u{c}'),
                    'n' => text.push('\n'),
                    'r' => text.push('\r'),
                    't' => text.push('\t'),
                    'u' => text.push(self.unicode()?),
                    _ => return Err(Error::Invalid("invalid JSON escape".into())),
                },
                c if c.is_control() => return Err(Error::Invalid("JSON control character".into())),
                c => text.push(c),
            }
        }
    }
    fn hex(&mut self) -> Result<u32> {
        let mut value = 0;
        for _ in 0..4 {
            value = value * 16 + self.take()?.to_digit(16).ok_or_else(|| Error::Invalid("invalid hex escape".into()))?;
        }
        Ok(value)
    }
    fn unicode(&mut self) -> Result<char> {
        let mut point = self.hex()?;
        if (0xd800..=0xdbff).contains(&point) {
            if self.take()? != '\\' || self.take()? != 'u' {
                return Err(Error::Invalid("missing low surrogate".into()));
            }
            let low = self.hex()?;
            if !(0xdc00..=0xdfff).contains(&low) {
                return Err(Error::Invalid("invalid low surrogate".into()));
            }
            point = 0x10000 + (point - 0xd800) * 1024 + low - 0xdc00;
        }
        char::from_u32(point).ok_or_else(|| Error::Invalid("invalid Unicode scalar".into()))
    }
}
