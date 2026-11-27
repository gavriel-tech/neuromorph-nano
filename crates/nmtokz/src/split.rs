//! Ordered alternatives of the supported Qwen Unicode Split pattern.
use crate::normalize::{letter, mark, number};
pub(crate) const PATTERN: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// Qwen3.5's pattern: letter runs also take marks (`[\p{L}\p{M}]+`); symbol runs exclude them.
pub(crate) const PATTERN_MARKS: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
fn space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | ' ' | '\u{85}' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}'
    )
}
fn newline(c: char) -> bool {
    matches!(c, '\r' | '\n')
}
fn contraction(chars: &[char]) -> Option<usize> {
    if chars.first() != Some(&'\'') {
        return None;
    }
    for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
        if chars.len() > suffix.len()
            && chars[1..].iter().take(suffix.len()).map(|c| if *c == '\u{17f}' { 's' } else { c.to_ascii_lowercase() }).eq(suffix.chars())
        {
            return Some(1 + suffix.len());
        }
    }
    None
}
fn end(chars: &[char], start: usize, marks: bool) -> usize {
    let size = chars.len();
    let c = chars[start];
    // Qwen3.5 (`marks`): letter runs are [\p{L}\p{M}]+ and symbol runs exclude marks.
    let run = |x: char| letter(x) || (marks && mark(x));
    if let Some(n) = contraction(&chars[start..]) {
        return start + n;
    }
    let next = start + usize::from(!newline(c) && !letter(c) && !number(c));
    if next < size && run(chars[next]) {
        let mut i = next + 1;
        while i < size && run(chars[i]) {
            i += 1;
        }
        return i;
    }
    if number(c) {
        return start + 1;
    }
    let begin = start + usize::from(c == ' ');
    let mut i = begin;
    while i < size && !space(chars[i]) && !run(chars[i]) && !number(chars[i]) {
        i += 1;
    }
    if i > begin {
        while i < size && newline(chars[i]) {
            i += 1;
        }
        return i;
    }
    let mut last = None;
    i = start;
    while i < size && space(chars[i]) {
        if newline(chars[i]) {
            last = Some(i);
        }
        i += 1;
    }
    if let Some(last) = last {
        last + 1
    } else if i == size {
        i
    } else if i - start > 1 {
        i - 1
    } else {
        start + 1
    }
}
pub(crate) fn pieces(text: &str, marks: bool) -> Vec<String> {
    let chars: Vec<_> = text.chars().collect();
    let mut start = 0;
    let mut out = Vec::new();
    while start < chars.len() {
        let next = end(&chars, start, marks);
        out.push(chars[start..next].iter().collect());
        start = next;
    }
    out
}
fn other(c: char) -> bool {
    !space(c) && !letter(c) && !number(c)
}
/// End of one GPT-2 `ByteLevel` piece: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
/// ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, alternatives tried in order, contractions case-sensitive.
fn gpt2_end(chars: &[char], start: usize) -> usize {
    let size = chars.len();
    let c = chars[start];
    if c == '\'' {
        for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
            if chars[start + 1..].iter().take(suffix.len()).copied().eq(suffix.chars()) {
                return start + 1 + suffix.len();
            }
        }
    }
    let lead = usize::from(c == ' ');
    if let Some(&first) = chars.get(start + lead) {
        let kinds: [fn(char) -> bool; 3] = [letter, number, other];
        for kind in kinds {
            if kind(first) {
                let mut i = start + lead + 1;
                while i < size && kind(chars[i]) {
                    i += 1;
                }
                return i;
            }
        }
    }
    let mut i = start;
    while i < size && space(chars[i]) {
        i += 1;
    }
    if i == size || i - start == 1 { i } else { i - 1 }
}
/// Pieces of the GPT-2 `ByteLevel` pre-tokenizer (`use_regex` true).
pub(crate) fn gpt2_pieces(text: &str) -> Vec<String> {
    let chars: Vec<_> = text.chars().collect();
    let mut start = 0;
    let mut out = Vec::new();
    while start < chars.len() {
        let next = gpt2_end(&chars, start);
        out.push(chars[start..next].iter().collect());
        start = next;
    }
    out
}
#[test]
fn gpt2_cases() {
    assert_eq!(
        gpt2_pieces("I'M we're  x1 2!?\n\nend "),
        ["I", "'", "M", " we", "'re", " ", " x", "1", " 2", "!?", "\n", "\n", "end", " "],
        "GPT-2 ordered alternatives"
    );
}
#[test]
fn split_cases() {
    assert_eq!(pieces("WE'RE 123\r\n hi", false), ["WE", "'RE", " ", "1", "2", "3", "\r\n", " hi"], "ordered alternatives");
    assert_eq!(pieces("a   b  ", false), ["a", "  ", " b", "  "], "whitespace lookahead");
    // A combining acute (U+0301) stays in the letter run only with marks.
    assert_eq!(pieces("e\u{301}x", true), ["e\u{301}x"], "marks join letters");
    assert_eq!(pieces("e\u{301}x", false), ["e", "\u{301}x"], "without marks the mark leads the next letter run");
}
