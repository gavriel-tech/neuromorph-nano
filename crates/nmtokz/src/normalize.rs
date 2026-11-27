//! Canonical Unicode normalisation including algorithmic Hangul composition.
use crate::unicode_data::{CCC, COMPOSE, DECOMP, LETTER, NUMBER};
fn class(point: u32) -> u8 {
    CCC.binary_search_by_key(&point, |x| x.0).map_or(0, |i| CCC[i].1)
}
fn in_ranges(c: char, ranges: &[(u32, u32)]) -> bool {
    let n = u32::from(c);
    let i = ranges.partition_point(|x| x.0 <= n);
    i > 0 && n <= ranges[i - 1].1
}
pub(crate) fn letter(c: char) -> bool {
    in_ranges(c, LETTER)
}
/// A Unicode mark (Mn, Mc or Me).
pub(crate) fn mark(c: char) -> bool {
    in_ranges(c, crate::unicode_marks::MARK)
}
pub(crate) fn number(c: char) -> bool {
    in_ranges(c, NUMBER)
}
fn decompose(n: u32, out: &mut Vec<u32>) {
    if (0xac00..0xd7a4).contains(&n) {
        let index = n - 0xac00;
        out.push(0x1100 + index / 588);
        out.push(0x1161 + (index % 588) / 28);
        if !index.is_multiple_of(28) {
            out.push(0x11a7 + index % 28);
        }
    } else if let Ok(i) = DECOMP.binary_search_by_key(&n, |x| x.0) {
        for &part in DECOMP[i].1 {
            decompose(part, out);
        }
    } else {
        out.push(n);
    }
}
fn compose(a: u32, b: u32) -> Option<u32> {
    if (0x1100..0x1113).contains(&a) && (0x1161..0x1176).contains(&b) {
        return Some(0xac00 + (a - 0x1100) * 588 + (b - 0x1161) * 28);
    }
    if (0xac00..0xd7a4).contains(&a) && (a - 0xac00).is_multiple_of(28) && (0x11a8..0x11c3).contains(&b) {
        return Some(a + b - 0x11a7);
    }
    COMPOSE.binary_search_by_key(&(a, b), |x| (x.0, x.1)).ok().map(|i| COMPOSE[i].2)
}
pub(crate) fn nfc(text: &str) -> String {
    let mut parts = Vec::new();
    for c in text.chars() {
        decompose(u32::from(c), &mut parts);
    }
    // Stable insertion within each starter's combining sequence.
    for i in 1..parts.len() {
        let mut j = i;
        while j > 0 && class(parts[j]) != 0 && class(parts[j - 1]) > class(parts[j]) {
            parts.swap(j, j - 1);
            j -= 1;
        }
    }
    let mut out = Vec::new();
    let mut starter = 0;
    let mut previous = 0;
    for point in parts {
        let order = class(point);
        let merged = out.get(starter).copied().and_then(|a| if previous == 0 || previous < order { compose(a, point) } else { None });
        if let Some(n) = merged {
            out[starter] = n;
        } else {
            if order == 0 {
                starter = out.len();
            }
            out.push(point);
            previous = order;
        }
    }
    out.into_iter().filter_map(char::from_u32).collect()
}
#[test]
fn canonical_cases() {
    for (input, want) in [
        ("e\u{301}", "\u{e9}"),
        ("\u{212b}", "\u{c5}"),
        ("\u{1100}\u{1161}\u{11a8}", "\u{ac01}"),
        ("a\u{315}\u{300}", "\u{e0}\u{315}"),
        ("\u{344}", "\u{308}\u{301}"),
    ] {
        assert_eq!(nfc(input), want, "NFC canonical case");
    }
}
