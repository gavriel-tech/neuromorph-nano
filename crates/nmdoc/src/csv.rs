//! Tables in CSV and TSV: RFC 4180 records (quoted fields, doubled quotes, line breaks inside
//! quotes), the delimiter sniffed (comma, semicolon, tab, pipe), the header row detected, a type
//! per column. The derived text is the file itself (a row's line number is its line in the file);
//! the table is one `Table` block whose label is its schema (column names, types, a few distinct
//! values), so the schema can be found by a question that names a column, and every slice of rows
//! can name its header.
use nmtype::doc::{Block, BlockKind};

/// One record: its fields and its first and last line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub fields: Vec<String>,
    pub first: usize,
    pub last: usize,
}

/// The delimiter: the candidate that splits the first lines into the same number of fields (more
/// than one), most fields first; `default` when none does.
#[must_use]
pub fn sniff(text: &str, default: char) -> char {
    let sample: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).take(20).collect();
    let mut best = (default, 0_usize);
    for d in [',', '\t', ';', '|'] {
        let counts: Vec<usize> = sample.iter().map(|l| split_line(l, d).len()).collect();
        let Some(&first) = counts.first() else { continue };
        if first > 1 && counts.iter().all(|c| *c == first) && first > best.1 {
            best = (d, first);
        }
    }
    best.0
}

fn split_line(l: &str, d: char) -> Vec<String> {
    parse(l, d).into_iter().next().map(|r| r.fields).unwrap_or_default()
}

/// Records of a delimited text (RFC 4180; a quote opens a quoted field only at its start).
#[must_use]
pub fn parse(text: &str, d: char) -> Vec<Record> {
    let mut out = Vec::new();
    let mut fields: Vec<String> = Vec::new();
    let mut field = String::new();
    let (mut quoted, mut at_start) = (false, true);
    let (mut line, mut first) = (1_usize, 1_usize);
    let mut chars = text.chars().peekable();
    let mut any = false;
    while let Some(c) = chars.next() {
        any = true;
        if quoted {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    chars.next();
                    field.push('"');
                }
                '"' => quoted = false,
                '\n' => {
                    line += 1;
                    field.push('\n');
                }
                '\r' => {}
                _ => field.push(c),
            }
            continue;
        }
        match c {
            '"' if at_start => {
                quoted = true;
                at_start = false;
            }
            '\r' => {}
            '\n' => {
                fields.push(std::mem::take(&mut field));
                out.push(Record { fields: std::mem::take(&mut fields), first, last: line });
                line += 1;
                first = line;
                at_start = true;
                any = false;
            }
            _ if c == d => {
                fields.push(std::mem::take(&mut field));
                at_start = true;
            }
            _ => {
                field.push(c);
                at_start = false;
            }
        }
    }
    if any || !field.is_empty() || !fields.is_empty() {
        fields.push(field);
        out.push(Record { fields, first, last: line });
    }
    out
}

/// The type of one value: integer, number, date, boolean or text.
fn type_of(v: &str) -> &'static str {
    let t = v.trim();
    if t.is_empty() {
        return "";
    }
    let digits = t.trim_start_matches(['-', '+']);
    if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
        return "integer";
    }
    if digits.parse::<f64>().is_ok() {
        return "number";
    }
    let b = t.as_bytes();
    if b.len() >= 10 && b.get(4) == Some(&b'-') && b.get(7) == Some(&b'-') && t.get(..4).is_some_and(|y| y.chars().all(|c| c.is_ascii_digit())) {
        return "date";
    }
    if matches!(t.to_lowercase().as_str(), "true" | "false" | "yes" | "no") {
        return "boolean";
    }
    "text"
}

/// True when the first record is a header: every field is text and none repeats, and some later
/// column is not text, or the first record shares no value with the second.
fn has_header(records: &[Record]) -> bool {
    let Some(h) = records.first() else { return false };
    if h.fields.iter().any(|f| f.trim().is_empty() || type_of(f) != "text") {
        return false;
    }
    let mut seen: Vec<&str> = h.fields.iter().map(String::as_str).collect();
    seen.sort_unstable();
    seen.dedup();
    if seen.len() != h.fields.len() {
        return false;
    }
    let rows = records.get(1..records.len().min(30)).unwrap_or(&[]);
    rows.is_empty() || (0..h.fields.len()).any(|c| rows.iter().any(|r| r.fields.get(c).is_some_and(|v| type_of(v) != "text" && !v.trim().is_empty())))
}

/// The schema of a table: `columns: name (type, e.g. a, b, c); ...` over the data rows.
#[must_use]
pub fn schema(records: &[Record], header: bool) -> String {
    let width = records.iter().map(|r| r.fields.len()).max().unwrap_or(0);
    let names: Vec<String> = match records.first() {
        Some(h) if header => h.fields.iter().map(|f| f.trim().to_owned()).collect(),
        _ => (1..=width).map(|i| format!("column {i}")).collect(),
    };
    let rows = if header { records.get(1..).unwrap_or(&[]) } else { records };
    let mut cols = Vec::new();
    for (c, name) in names.iter().enumerate() {
        let vals: Vec<&str> = rows.iter().filter_map(|r| r.fields.get(c).map(|v| v.trim())).filter(|v| !v.is_empty()).collect();
        let mut types: Vec<&str> = vals.iter().map(|v| type_of(v)).collect();
        types.sort_unstable();
        types.dedup();
        let ty = match types.as_slice() {
            [one] => *one,
            two if two.iter().all(|t| matches!(*t, "integer" | "number")) && !two.is_empty() => "number",
            [] => "empty",
            _ => "text",
        };
        let mut distinct: Vec<&str> = Vec::new();
        for v in &vals {
            if !distinct.contains(v) {
                distinct.push(v);
            }
        }
        let range = if matches!(ty, "integer" | "number") {
            let nums: Vec<f64> = vals.iter().filter_map(|v| v.parse::<f64>().ok()).collect();
            let lo = nums.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = nums.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if nums.is_empty() { String::new() } else { format!(" {lo} to {hi}") }
        } else {
            let few: Vec<&str> = distinct.iter().take(4).copied().collect();
            let more = if distinct.len() > 4 { format!(", ... {} distinct", distinct.len()) } else { String::new() };
            format!(" e.g. {}{more}", few.join(", "))
        };
        cols.push(format!("{name} ({ty}{range})"));
    }
    format!("columns: {}; {} rows", cols.join("; "), rows.len())
}

/// The blocks of a delimited table: one `Table` block over every line, its label the schema.
#[must_use]
pub fn blocks(text: &str, default: char) -> Vec<Block> {
    let n = text.lines().count();
    if n == 0 {
        return Vec::new();
    }
    let d = sniff(text, default);
    let records = parse(text, d);
    let header = has_header(&records);
    let head = if header { records.first().map_or(1, |r| r.last - r.first + 1) } else { 0 };
    let mut b = Block::new(BlockKind::Table { head: u16::try_from(head).unwrap_or(1) }, 1, n, &schema(&records, header));
    b.loc.cells = Some(format!("rows 1-{} of {}", records.len().saturating_sub(usize::from(header)), records.len().saturating_sub(usize::from(header))));
    vec![b]
}

/// Words that ask for the largest or the smallest value of a column.
const MOST: &[&str] = &["most", "highest", "largest", "biggest", "maximum", "max", "latest", "newest", "longest", "slowest", "top"];
const LEAST: &[&str] = &["least", "lowest", "smallest", "minimum", "min", "cheapest", "fastest", "quickest", "earliest", "oldest", "shortest", "soonest"];
/// Words that point at a column by its meaning when its name is not said.
const MEANS: &[(&str, &[&str])] = &[
    ("expensive", &["price", "cost"]),
    ("cheap", &["price", "cost"]),
    ("cheapest", &["price", "cost"]),
    ("costly", &["price", "cost"]),
    ("fastest", &["time", "days", "lead"]),
    ("quickest", &["time", "days", "lead"]),
    ("slowest", &["time", "days", "lead"]),
    ("delivers", &["time", "days", "lead"]),
    ("newest", &["date", "counted", "updated"]),
    ("latest", &["date", "counted", "updated"]),
    ("oldest", &["date", "counted", "updated"]),
];

/// The rows of a table that a question about a column's extreme or empty values points at: the
/// row or rows holding the largest or smallest value of the numeric column the question names
/// (by name, or by meaning: "most expensive" is the highest price), or the rows where that column
/// is zero ("out of stock", "none in stock"). The rows are context (the lines that hold the
/// answer), not a computed answer. Returns (line, why) for each row; empty when the question names
/// no column or no extreme.
#[must_use]
pub fn select(text: &str, default: char, question: &str) -> Vec<(usize, String)> {
    let q = question.to_lowercase();
    let words: Vec<&str> = q.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let most = words.iter().any(|w| MOST.contains(w)) || q.contains("most expensive");
    let least = words.iter().any(|w| LEAST.contains(w));
    let zero = q.contains("out of ") || (q.contains(" no ") && q.contains(" in ")) || q.contains("none in") || q.contains("zero");
    if !(most || least || zero) {
        return Vec::new();
    }
    let d = sniff(text, default);
    let records = parse(text, d);
    if !has_header(&records) {
        return Vec::new();
    }
    let Some((head, rows)) = records.split_first() else { return Vec::new() };
    let numeric = |c: usize| rows.iter().filter_map(|r| r.fields.get(c)).filter(|v| !v.trim().is_empty()).all(|v| v.trim().parse::<f64>().is_ok());
    let name_words = |c: usize| -> Vec<String> {
        head.fields.get(c).map(|h| h.to_lowercase().split(|x: char| !x.is_alphanumeric()).filter(|w| w.len() >= 3).map(str::to_owned).collect()).unwrap_or_default()
    };
    let said = |c: usize| name_words(c).iter().any(|w| words.iter().any(|q| q == w || (q.len() > 3 && w.starts_with(q)) || (w.len() > 3 && q.starts_with(w.as_str()))));
    let meant = |c: usize| {
        let nw = name_words(c);
        words.iter().any(|w| MEANS.iter().any(|(k, cols)| k == w && cols.iter().any(|x| nw.iter().any(|n| n == x))))
    };
    let cols: Vec<usize> = (0..head.fields.len()).filter(|c| numeric(*c)).collect();
    let Some(col) = cols.iter().copied().find(|c| said(*c)).or_else(|| cols.iter().copied().find(|c| meant(*c))) else { return Vec::new() };
    let name = head.fields.get(col).map_or("", |h| h.trim());
    let vals: Vec<(f64, usize)> =
        rows.iter().filter_map(|r| r.fields.get(col).and_then(|v| v.trim().parse::<f64>().ok()).map(|v| (v, r.first))).collect();
    let (target, why) = if zero {
        (Some(0.0), format!("{name} is 0"))
    } else if least && !most {
        (vals.iter().map(|v| v.0).reduce(f64::min), format!("lowest {name}"))
    } else {
        (vals.iter().map(|v| v.0).reduce(f64::max), format!("highest {name}"))
    };
    let Some(t) = target else { return Vec::new() };
    vals.iter().filter(|v| (v.0 - t).abs() < 1e-9).map(|v| (v.1, why.clone())).collect()
}

pub(crate) fn check() -> Result<(), String> {
    let text = "part_no,name,stock\r\nP-1,\"Seal, rubber\",12\r\nP-2,\"Bolt \"\"M8\"\"\",0\r\nP-3,\"two\nlines\",5\r\n";
    let r = parse(text, ',');
    let want = [vec!["part_no", "name", "stock"], vec!["P-1", "Seal, rubber", "12"], vec!["P-2", "Bolt \"M8\"", "0"], vec!["P-3", "two\nlines", "5"]];
    let got: Vec<Vec<&str>> = r.iter().map(|x| x.fields.iter().map(String::as_str).collect()).collect();
    if got != want || r.get(3).map(|x| (x.first, x.last)) != Some((4, 5)) {
        return Err(format!("nmdoc.csv: records {got:?}"));
    }
    if sniff("a\tb\tc\n1\t2\t3\n", ',') != '\t' || sniff("a;b\n1;2\n", ',') != ';' {
        return Err("nmdoc.csv: delimiter".into());
    }
    let b = blocks(text, ',');
    let s = b.first().map(|x| x.label.clone()).unwrap_or_default();
    if b.len() != 1 || !matches!(b.first().map(|x| x.kind), Some(BlockKind::Table { head: 1 })) || !s.contains("stock (integer 0 to 12)") || !s.contains("3 rows")
    {
        return Err(format!("nmdoc.csv: blocks {b:?}"));
    }
    if has_header(&parse("1,2\n3,4\n", ',')) {
        return Err("nmdoc.csv: a numeric first row is not a header".into());
    }
    let t = "part,unit_price,stock\nA,4.5,0\nB,9.0,3\nC,9.0,0\n";
    let lines = |q: &str| select(t, ',', q).iter().map(|x| x.0).collect::<Vec<_>>();
    if lines("Which is the most expensive part?") != [3, 4] || lines("Which parts are out of stock?") != [2, 4] || !lines("Which part is red?").is_empty() {
        return Err(format!("nmdoc.csv: select {:?}", lines("Which is the most expensive part?")));
    }
    Ok(())
}
