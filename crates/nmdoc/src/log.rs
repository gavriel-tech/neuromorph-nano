//! Logs: the file is its derived text; each event is a `LogEvent` block, a line that starts with
//! a time stamp plus the lines after it that do not (a stack trace, a wrapped message). Time
//! stamps: ISO 8601 (`2026-10-03T14:02:11Z`, a space for the `T`, fractions, offsets), bracketed
//! (`[2026-10-03 14:02:11]`), syslog (`Oct  3 14:02:11`, year taken as 1970), and Windows event
//! text (`3/10/2026 2:02:11 PM` is not read: day and month order is not known). An event's time is
//! in `loc.time` (milliseconds since 1970, UTC; an offset is applied), its label is the time
//! stamp as written and its level word when there is one.
use nmtype::doc::{Block, BlockKind};

fn num(s: &str) -> Option<u64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// Days from 1970-01-01 to a civil date (proleptic Gregorian).
fn days(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

const MONTHS: [&str; 12] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

/// The time stamp at the start of a line: (milliseconds since 1970 UTC, length in bytes).
#[must_use]
pub fn stamp(line: &str) -> Option<(u64, usize)> {
    let (l, skip) = match line.strip_prefix('[') {
        Some(r) => (r, 1),
        None => (line, 0),
    };
    let b = l.as_bytes();
    // ISO 8601 date and time.
    if b.len() >= 19 && b.get(4) == Some(&b'-') && b.get(7) == Some(&b'-') && matches!(b.get(10), Some(b'T' | b' ')) && b.get(13) == Some(&b':') {
        let y = num(l.get(..4)?)?;
        let mo = num(l.get(5..7)?)?;
        let d = num(l.get(8..10)?)?;
        let h = num(l.get(11..13)?)?;
        let mi = num(l.get(14..16)?)?;
        let s = num(l.get(17..19)?)?;
        if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || s > 60 {
            return None;
        }
        let mut end = 19;
        let mut ms = 0_u64;
        if matches!(b.get(end), Some(b'.' | b',')) {
            let digits: String = l.get(end + 1..)?.chars().take_while(char::is_ascii_digit).collect();
            ms = num(&format!("{:0<3}", digits.get(..digits.len().min(3))?)).unwrap_or(0);
            end += 1 + digits.len();
        }
        let mut offset: i64 = 0;
        match b.get(end) {
            Some(b'Z') => end += 1,
            Some(c @ (b'+' | b'-')) if b.len() >= end + 6 && b.get(end + 3) == Some(&b':') => {
                let oh = num(l.get(end + 1..end + 3)?)? as i64;
                let om = num(l.get(end + 4..end + 6)?)? as i64;
                offset = (oh * 60 + om) * if *c == b'+' { 1 } else { -1 };
                end += 6;
            }
            _ => {}
        }
        let secs = days(y as i64, mo as i64, d as i64) * 86_400 + (h * 3600 + mi * 60 + s) as i64 - offset * 60;
        let t = u64::try_from(secs).ok()? * 1000 + ms;
        if skip == 1 && b.get(end) == Some(&b']') {
            end += 1;
        }
        return Some((t, end + skip));
    }
    // Syslog: `Oct  3 14:02:11`.
    let mon = l.get(..3)?.to_lowercase();
    let m = MONTHS.iter().position(|x| *x == mon)?;
    let rest = l.get(3..)?;
    let rest_t = rest.trim_start();
    let day_len = rest_t.chars().take_while(char::is_ascii_digit).count();
    let d = num(rest_t.get(..day_len)?)?;
    let time = rest_t.get(day_len..)?.trim_start();
    let tb = time.as_bytes();
    if tb.len() < 8 || tb.get(2) != Some(&b':') || tb.get(5) != Some(&b':') {
        return None;
    }
    let (h, mi, s) = (num(time.get(..2)?)?, num(time.get(3..5)?)?, num(time.get(6..8)?)?);
    let secs = days(1970, m as i64 + 1, d as i64) * 86_400 + (h * 3600 + mi * 60 + s) as i64;
    let used = line.len() - time.len() + 8;
    Some((u64::try_from(secs).ok()? * 1000, used))
}

/// Blocks of a log over its own lines.
#[must_use]
pub fn blocks(text: &str) -> Vec<Block> {
    let n = text.lines().count();
    let mut out: Vec<Block> = Vec::new();
    for (i, l) in text.lines().enumerate() {
        if let Some((t, len)) = stamp(l) {
            let written = l.get(..len).unwrap_or("").trim_matches(['[', ']']);
            let level = l
                .get(len..)
                .unwrap_or("")
                .split_whitespace()
                .next()
                .filter(|w| matches!(w.trim_matches(['[', ']', ':']).to_uppercase().as_str(), "TRACE" | "DEBUG" | "INFO" | "NOTICE" | "WARN" | "WARNING" | "ERROR" | "ERR" | "FATAL" | "CRITICAL"))
                .map(|w| format!(" {}", w.trim_matches(['[', ']', ':']).to_uppercase()))
                .unwrap_or_default();
            let mut b = Block::new(BlockKind::LogEvent, i + 1, i + 1, &format!("{written}{level}"));
            b.loc.time = Some((t, t));
            out.push(b);
        }
    }
    if out.first().is_some_and(|b| b.first > 1) {
        out.insert(0, Block::new(BlockKind::Paragraph, 1, 1, ""));
    }
    crate::md::close_gaps(&mut out, n);
    out
}

pub(crate) fn check() -> Result<(), String> {
    let log = "boot banner\n2026-10-03T14:02:11Z INFO started\n2026-10-03 14:02:12.5+02:00 ERROR failed\n  at pump.read (x.rs:4)\n  at main\n[2026-10-03 14:03:00] WARN slow\nOct  3 14:04:05 host cron: ran\n";
    let b = blocks(log);
    nmtype::doc::check_cover(&b, log.lines().count()).map_err(|e| format!("nmdoc.log: {e}"))?;
    let got: Vec<(usize, usize, &str)> = b.iter().map(|x| (x.first, x.last, x.label.as_str())).collect();
    let want = [
        (1, 1, ""),
        (2, 2, "2026-10-03T14:02:11Z INFO"),
        (3, 5, "2026-10-03 14:02:12.5+02:00 ERROR"),
        (6, 6, "2026-10-03 14:03:00 WARN"),
        (7, 7, "Oct  3 14:04:05"),
    ];
    if got != want {
        return Err(format!("nmdoc.log: events {got:?}"));
    }
    let times: Vec<u64> = b.iter().filter_map(|x| x.loc.time.map(|t| t.0)).collect();
    if times != [1_791_036_131_000, 1_791_028_932_500, 1_791_036_180_000, 23_810_645_000] {
        return Err(format!("nmdoc.log: times {times:?}"));
    }
    Ok(())
}
