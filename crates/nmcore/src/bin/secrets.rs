//! Secret-dropping: the one routine applied to every piece of prompt or tool text before it is
//! stored. Anything that looks like a key, token, password, private key block or connection
//! string credential is replaced by a marker naming its kind, `[secret dropped: <kind>]`.
//! It errs on the side of dropping: a false hit loses a few words, a miss keeps a secret.
//! Known misses are listed in `docs/privacy.md`.

/// The marker put in place of a dropped secret.
#[must_use]
pub fn marker(kind: &str) -> String {
    format!("[secret dropped: {kind}]")
}

/// The text with every secret-looking span replaced by its marker.
#[must_use]
pub fn drop_secrets(text: &str) -> String {
    let spans = find(text);
    if spans.is_empty() {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for (a, b, kind) in spans {
        out.push_str(text.get(at..a).unwrap_or_default());
        out.push_str(&marker(kind));
        at = b;
    }
    out.push_str(text.get(at..).unwrap_or_default());
    out
}

/// The kinds of secret found in the text, in order, one per span.
#[must_use]
pub fn kinds(text: &str) -> Vec<&'static str> {
    find(text).into_iter().map(|s| s.2).collect()
}

type Span = (usize, usize, &'static str);

/// Every span to drop, sorted, with overlaps merged (the earlier and longer span wins).
fn find(text: &str) -> Vec<Span> {
    let mut all: Vec<Span> = Vec::new();
    pem(text, &mut all);
    url_credentials(text, &mut all);
    prefixed(text, &mut all);
    jwt(text, &mut all);
    authorization(text, &mut all);
    named(text, &mut all);
    flags(text, &mut all);
    short_flags(text, &mut all);
    prose(text, &mut all);
    random(text, &mut all);
    all.sort_by(|x, y| x.0.cmp(&y.0).then(y.1.cmp(&x.1)));
    let mut out: Vec<Span> = Vec::new();
    for s in all {
        match out.last_mut() {
            Some(last) if s.0 < last.1 => last.1 = last.1.max(s.1),
            _ => out.push(s),
        }
    }
    out
}

fn token_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'-'
}
fn at_word_start(b: &[u8], i: usize) -> bool {
    i == 0 || b.get(i - 1).is_none_or(|c| !c.is_ascii_alphanumeric() && *c != b'_')
}
fn run_end(b: &[u8], from: usize, ok: impl Fn(u8) -> bool) -> usize {
    let mut j = from;
    while b.get(j).is_some_and(|c| ok(*c)) {
        j += 1;
    }
    j
}

/// PEM private key blocks, to their END line or, cut short, to the end of the text.
fn pem(text: &str, out: &mut Vec<Span>) {
    let b = text.as_bytes();
    for (i, _) in text.match_indices("-----BEGIN ") {
        let line_end = text.get(i..).and_then(|t| t.find('\n')).map_or(text.len(), |n| i + n);
        if !text.get(i..line_end).is_some_and(|l| l.contains("PRIVATE KEY")) {
            continue;
        }
        let end = text.get(line_end..).and_then(|t| t.find("-----END ")).map_or(text.len(), |n| {
            let s = line_end + n + 9;
            let close = text.get(s..).and_then(|t| t.find("-----")).map_or(text.len(), |m| s + m + 5);
            close.min(b.len())
        });
        out.push((i, end, "private key"));
    }
}

/// The password of `scheme://user:password@host` (connection strings and URLs).
fn url_credentials(text: &str, out: &mut Vec<Span>) {
    let b = text.as_bytes();
    for (i, _) in text.match_indices("://") {
        let start = i + 3;
        let stop = run_end(b, start, |c| !matches!(c, b'/' | b'@' | b'"' | b'\'' | b'`' | b'<' | b'>') && !c.is_ascii_whitespace());
        if b.get(stop) != Some(&b'@') {
            continue;
        }
        let userinfo = text.get(start..stop).unwrap_or_default();
        if let Some(colon) = userinfo.find(':') {
            if colon + 1 < userinfo.len() {
                out.push((start + colon + 1, stop, "password in a connection string"));
            }
        } else if userinfo.len() >= 16 {
            out.push((start, stop, "access token"));
        }
    }
}

/// Keys and tokens recognised by their published prefixes.
const PREFIXES: &[(&str, usize, &str)] = &[
    ("sk-", 20, "API key"),
    ("sk_live_", 10, "API key"),
    ("sk_test_", 10, "API key"),
    ("rk_live_", 10, "API key"),
    ("rk_test_", 10, "API key"),
    ("ghp_", 30, "access token"),
    ("gho_", 30, "access token"),
    ("ghu_", 30, "access token"),
    ("ghs_", 30, "access token"),
    ("ghr_", 30, "access token"),
    ("github_pat_", 20, "access token"),
    ("glpat-", 20, "access token"),
    ("xoxb-", 10, "access token"),
    ("xoxp-", 10, "access token"),
    ("xoxa-", 10, "access token"),
    ("xoxr-", 10, "access token"),
    ("xoxs-", 10, "access token"),
    ("xapp-", 10, "access token"),
    ("AIza", 30, "API key"),
    ("hf_", 30, "access token"),
    ("npm_", 30, "access token"),
    ("pypi-", 50, "access token"),
    ("shpat_", 20, "access token"),
    ("shpss_", 20, "access token"),
    ("dop_v1_", 40, "access token"),
    ("SG.", 20, "API key"),
];

fn prefixed(text: &str, out: &mut Vec<Span>) {
    let b = text.as_bytes();
    for (p, min, kind) in PREFIXES {
        for (i, _) in text.match_indices(p) {
            if !at_word_start(b, i) {
                continue;
            }
            let end = run_end(b, i + p.len(), |c| token_char(c) || c == b'.');
            let end = trim_dots(b, i + p.len(), end);
            if end - (i + p.len()) >= *min {
                out.push((i, end, kind));
            }
        }
    }
    for p in ["AKIA", "ASIA"] {
        for (i, _) in text.match_indices(p) {
            let end = run_end(b, i + 4, |c| c.is_ascii_uppercase() || c.is_ascii_digit());
            if at_word_start(b, i) && end - (i + 4) >= 16 && !b.get(end).is_some_and(|c| c.is_ascii_alphanumeric()) {
                out.push((i, end, "cloud access key"));
            }
        }
    }
}

/// A run's end with trailing dots left out (a sentence's full stop is not part of a key).
fn trim_dots(b: &[u8], from: usize, mut end: usize) -> usize {
    while end > from && b.get(end - 1) == Some(&b'.') {
        end -= 1;
    }
    end
}

/// JSON web tokens: `eyJ` and three base64url parts joined by dots.
fn jwt(text: &str, out: &mut Vec<Span>) {
    let b = text.as_bytes();
    for (i, _) in text.match_indices("eyJ") {
        if !at_word_start(b, i) {
            continue;
        }
        let mut j = i;
        let mut parts = Vec::new();
        for _ in 0..3 {
            let e = run_end(b, j, token_char);
            parts.push(e - j);
            j = e;
            if b.get(j) == Some(&b'.') && parts.len() < 3 {
                j += 1;
            } else {
                break;
            }
        }
        if parts.len() == 3 && parts.first().is_some_and(|n| *n >= 10) && parts.get(1).is_some_and(|n| *n >= 10) {
            out.push((i, j, "JWT"));
        }
    }
}

/// `Authorization:` header values, and `Bearer <token>` anywhere.
fn authorization(text: &str, out: &mut Vec<Span>) {
    let low = text.to_ascii_lowercase();
    let b = text.as_bytes();
    for (i, _) in low.match_indices("authorization:") {
        let s = run_end(b, i + 14, |c| c == b' ' || c == b'\t');
        let e = run_end(b, s, |c| !matches!(c, b'\r' | b'\n' | b'"' | b'\''));
        if e > s {
            out.push((s, e, "authorization header"));
        }
    }
    for (i, _) in low.match_indices("bearer ") {
        let s = run_end(b, i + 7, |c| c == b' ');
        let e = run_end(b, s, |c| token_char(c) || matches!(c, b'.' | b'~' | b'+' | b'/' | b'='));
        let tok = text.get(s..e).unwrap_or_default();
        if tok.len() >= 16 && tok.bytes().any(|c| c.is_ascii_digit()) {
            out.push((s, e, "access token"));
        }
    }
}

/// The words that make a name a secret's name, compared with the name's parts.
const SECRET_PARTS: &[(&str, &str)] = &[
    ("password", "password"),
    ("passwd", "password"),
    ("passphrase", "password"),
    ("pwd", "password"),
    ("pass", "password"),
    ("secret", "secret"),
    ("token", "access token"),
    ("apikey", "API key"),
    ("auth", "access token"),
    ("credential", "credential"),
    ("credentials", "credential"),
    ("cookie", "cookie"),
    ("privatekey", "private key"),
    ("sig", "signature"),
    ("signature", "signature"),
];
/// Parts that, followed by `key`, make a key's name (`api_key`, `AccountKey`, `x-api-key`).
const KEY_QUALIFIERS: &[&str] =
    &["api", "access", "secret", "private", "account", "client", "signing", "encryption", "master", "auth", "subscription", "storage", "license"];

/// The name's parts: split at `_`, `-`, `.` and lower-to-upper case changes, lower-cased.
fn name_parts(name: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for c in name.chars() {
        if matches!(c, '_' | '-' | '.' | '$' | ':') {
            if !cur.is_empty() {
                parts.push(std::mem::take(&mut cur));
            }
            prev_lower = false;
            continue;
        }
        if c.is_ascii_uppercase() && prev_lower && !cur.is_empty() {
            parts.push(std::mem::take(&mut cur));
        }
        prev_lower = c.is_ascii_lowercase() || c.is_ascii_digit();
        cur.push(c.to_ascii_lowercase());
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

/// The kind of secret a name stands for, if it names one.
fn secret_name(name: &str) -> Option<&'static str> {
    let parts = name_parts(name);
    for (i, p) in parts.iter().enumerate() {
        if let Some((_, kind)) = SECRET_PARTS.iter().find(|(w, _)| w == p) {
            return Some(kind);
        }
        // Names written as one word: `PGPASSWORD`, `DBSECRET`, `GITHUBTOKEN`, `MYAPIKEY`.
        if p.contains("password") || p.contains("passwd") || (p.contains("secret") && !p.contains("secretar")) {
            return Some(if p.contains("secret") { "secret" } else { "password" });
        }
        if p.len() > 5 && (p.ends_with("token") || p.ends_with("apikey")) {
            return Some(if p.ends_with("token") { "access token" } else { "API key" });
        }
        if p == "key" && i > 0 && parts.get(i - 1).is_some_and(|q| KEY_QUALIFIERS.contains(&q.as_str())) {
            return Some("API key");
        }
    }
    None
}

/// Values that stand for no secret: placeholders, variable references, type names.
fn placeholder(v: &str, name_kind: &str) -> bool {
    let low = v.to_ascii_lowercase();
    const WORDS: &[&str] = &[
        "null", "none", "nil", "true", "false", "undefined", "string", "str", "&str", "number", "int", "integer", "bool",
        "boolean", "any", "usize", "u64", "u32", "i64", "i32", "option", "value", "required", "optional", "yes", "no",
        "on", "off", "redacted", "hidden",
    ];
    v.is_empty()
        || v.len() < 4 && name_kind != "password"
        || v.len() < 3
        || WORDS.contains(&low.as_str())
        || v.starts_with('<')
        || v.starts_with('$')
        || v.starts_with('%')
        || v.starts_with("{{")
        || v.starts_with('[')
        || v.chars().all(|c| matches!(c, '*' | 'x' | 'X' | '.' | '#' | '-'))
        || v.contains('(')
        || (v.chars().all(|c| c.is_ascii_digit()) && v.len() < 8 && name_kind != "password")
        || low.starts_with("process.env")
        || low.starts_with("os.environ")
        || low.starts_with("env.")
}

/// `name=value` and `name: value` where the name is a secret's name (shell variables, settings,
/// headers, connection string fields). The value is dropped, the name kept.
fn named(text: &str, out: &mut Vec<Span>) {
    let b = text.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        if c != b'=' && c != b':' {
            continue;
        }
        // `==`, `=>`, `::`, `:=` and URL schemes are not assignments of a value to a name.
        let next = b.get(i + 1).copied();
        let prev = if i > 0 { b.get(i - 1).copied() } else { None };
        if matches!(next, Some(b'=' | b'>' | b':')) || matches!(prev, Some(b'=' | b':' | b'!' | b'<' | b'>')) {
            continue;
        }
        if c == b':' && text.get(i..).is_some_and(|t| t.starts_with("://")) {
            continue;
        }
        // The name: skip spaces and a closing quote to the left, then the name's characters.
        let mut j = i;
        while j > 0 && b.get(j - 1).is_some_and(|x| *x == b' ' || *x == b'\t') {
            j -= 1;
        }
        if j > 0 && b.get(j - 1).is_some_and(|x| *x == b'"' || *x == b'\'') {
            j -= 1;
        }
        let name_end = j;
        while j > 0 && b.get(j - 1).is_some_and(|x| x.is_ascii_alphanumeric() || matches!(x, b'_' | b'-' | b'.')) {
            j -= 1;
        }
        let Some(kind) = text.get(j..name_end).filter(|n| !n.is_empty()).and_then(secret_name) else { continue };
        // The value: quoted, or a run to the next space or separator.
        let s = run_end(b, i + 1, |x| x == b' ' || x == b'\t');
        let (vs, ve) = match b.get(s) {
            Some(&q @ (b'"' | b'\'')) => {
                let e = run_end(b, s + 1, |x| x != q && x != b'\n');
                (s + 1, e)
            }
            _ if kind == "cookie" || kind == "authorization header" => (s, run_end(b, s, |x| x != b'\n' && x != b'\r')),
            _ => (s, run_end(b, s, |x| !x.is_ascii_whitespace() && !matches!(x, b';' | b',' | b'"' | b'\'' | b'`' | b')' | b'}' | b']'))),
        };
        let v = text.get(vs..ve).unwrap_or_default();
        if !placeholder(v, kind) {
            out.push((vs, ve, kind));
        }
    }
}

/// `--password value`, `--token value`, `--api-key value`, `--secret value` (value after a space).
fn flags(text: &str, out: &mut Vec<Span>) {
    let b = text.as_bytes();
    for (i, _) in text.match_indices("--") {
        let ne = run_end(b, i + 2, |c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_');
        let Some(kind) = text.get(i + 2..ne).and_then(secret_name) else { continue };
        if b.get(ne) != Some(&b' ') {
            continue;
        }
        let s = run_end(b, ne, |c| c == b' ');
        if b.get(s) == Some(&b'-') {
            continue;
        }
        let (vs, ve) = match b.get(s) {
            Some(&q @ (b'"' | b'\'')) => (s + 1, run_end(b, s + 1, |x| x != q && x != b'\n')),
            _ => (s, run_end(b, s, |x| !x.is_ascii_whitespace())),
        };
        let v = text.get(vs..ve).unwrap_or_default();
        if !placeholder(v, kind) {
            out.push((vs, ve, kind));
        }
    }
}

/// Passwords given as short flags: `curl -u user:pass`, `mysql -pPASS`, `docker login -p PASS`,
/// `sshpass -p PASS`.
fn short_flags(text: &str, out: &mut Vec<Span>) {
    let b = text.as_bytes();
    let low = text.to_ascii_lowercase();
    let value_after = |at: usize| -> Option<(usize, usize)> {
        let s = run_end(b, at, |c| c == b' ');
        match b.get(s) {
            Some(&q @ (b'"' | b'\'')) => Some((s + 1, run_end(b, s + 1, |x| x != q && x != b'\n'))),
            Some(c) if *c != b'-' => Some((s, run_end(b, s, |x| !x.is_ascii_whitespace()))),
            _ => None,
        }
    };
    for flag in [" -u ", " --user "] {
        for (i, _) in text.match_indices(flag) {
            if let Some((vs, ve)) = value_after(i + flag.len())
                && let Some(colon) = text.get(vs..ve).and_then(|v| v.find(':'))
                && vs + colon + 1 < ve
            {
                out.push((vs + colon + 1, ve, "password"));
            }
        }
    }
    let mysql = ["mysql", "mariadb", "mysqldump", "mysqladmin"].iter().any(|w| low.contains(w));
    let separate = ["sshpass", "docker login", "podman login", "helm registry login"].iter().any(|w| low.contains(w));
    for (i, _) in text.match_indices(" -p") {
        let after = i + 3;
        match b.get(after) {
            Some(c) if mysql && !c.is_ascii_whitespace() => {
                let e = run_end(b, after, |x| !x.is_ascii_whitespace());
                out.push((after, e, "password"));
            }
            Some(b' ') if separate => {
                if let Some((vs, ve)) = value_after(after) {
                    out.push((vs, ve, "password"));
                }
            }
            _ => {}
        }
    }
}

/// Prose that names a secret and gives it: `the password is hunter2`, `my token was ab12cd34`.
/// The value must hold a digit or a symbol, so `the password is wrong` is left alone.
fn prose(text: &str, out: &mut Vec<Span>) {
    let low = text.to_ascii_lowercase();
    let b = text.as_bytes();
    for word in ["password", "passwd", "passphrase", "token", "secret", "api key", "apikey"] {
        for (i, _) in low.match_indices(word) {
            let rest = &low[i + word.len()..];
            let Some(skip) = [" is ", " was ", " = "].iter().find(|w| rest.starts_with(**w)).map(|w| w.len()) else { continue };
            let s = i + word.len() + skip;
            let e = run_end(b, s, |c| !c.is_ascii_whitespace() && !matches!(c, b',' | b';' | b'"' | b'\''));
            let e = trim_dots(b, s, e);
            let v = text.get(s..e).unwrap_or_default();
            if v.len() >= 4 && v.bytes().any(|c| c.is_ascii_digit() || c.is_ascii_punctuation()) && !placeholder(v, "password") {
                out.push((s, e, if word.starts_with("pass") { "password" } else { "secret" }));
            }
        }
    }
}

/// Shannon entropy of a word in bits per character.
fn entropy(w: &str) -> f64 {
    let mut counts = [0_u32; 256];
    for c in w.bytes() {
        if let Some(n) = counts.get_mut(usize::from(c)) {
            *n += 1;
        }
    }
    let n = w.len() as f64;
    counts.iter().filter(|c| **c > 0).map(|c| f64::from(*c) / n).map(|p| -p * p.log2()).sum()
}

/// Long random-looking words: 32 or more base64 characters with digits, upper and lower case
/// letters and high entropy. Pure hexadecimal (commit ids, hashes) is left alone.
fn random(text: &str, out: &mut Vec<Span>) {
    let b = text.as_bytes();
    let base64 = |c: u8| c.is_ascii_alphanumeric() || matches!(c, b'+' | b'/' | b'=' | b'_' | b'-');
    let mut i = 0;
    while i < b.len() {
        if !b.get(i).is_some_and(|c| base64(*c)) {
            i += 1;
            continue;
        }
        let e = run_end(b, i, base64);
        let w = text.get(i..e).unwrap_or_default();
        let digits = w.bytes().filter(u8::is_ascii_digit).count();
        let ok_edges = !b.get(e).is_some_and(|c| matches!(c, b'.' | b'\\')) || b.get(e + 1).is_none_or(u8::is_ascii_whitespace);
        let cloud_secret = w.len() == 40 && (w.contains('/') || w.contains('+')) && digits >= 1;
        let sk_hex = w.len() == 34 && w.starts_with("SK") && w.bytes().skip(2).all(|c| c.is_ascii_hexdigit());
        if sk_hex {
            out.push((i, e, "API key"));
        } else if w.len() >= 32
            && ok_edges
            && (digits >= 2 || cloud_secret)
            && w.bytes().any(|c| c.is_ascii_uppercase())
            && w.bytes().any(|c| c.is_ascii_lowercase())
            && !w.contains("//")
            && entropy(w) >= 4.3
        {
            out.push((i, e, "long random string"));
        }
        i = e.max(i + 1);
    }
}

/// The test list: (text, must something be dropped, a word that must survive or vanish).
/// True cases must lose the secret; false cases must come back unchanged.
pub const CASES: &[(&str, bool, &str)] = &[
    ("export OPENAI_API_KEY=sk-proj-abcDEF1234567890ghijKLMNOP", true, "abcDEF1234567890"),
    ("key sk-ant-api03-AbCdEf0123456789AbCdEf0123456789 here", true, "AbCdEf0123456789"),
    ("git push https://ghp_1234567890abcdefghijABCDEFGHIJ123456@github.com/o/r", true, "ghp_1234567890"),
    ("github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuv", true, "11ABCDEFG"),
    ("aws configure set aws_access_key_id AKIAIOSFODNN7EXAMPLE", true, "AKIAIOSFODNN7EXAMPLE"),
    ("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", true, "wJalrXUtnFEMI"),
    ("SLACK=xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx", true, "AbCdEfGhIjKl"),
    ("maps key AIzaSyA1234567890abcdefghijklmnopqrstu", true, "AIzaSyA1234567890"),
    (
        "jwt eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
        true,
        "dozjgNryP4J3",
    ),
    ("curl -H \"Authorization: Bearer abc123def456ghi789\" https://x.test", true, "abc123def456ghi789"),
    ("use password=hunter22 for the test account", true, "hunter22"),
    ("{\"db_password\": \"s3cr3t-Value!\"}", true, "s3cr3t-Value!"),
    ("mysql --password=Tr0ub4dor&3 -u root", true, "Tr0ub4dor"),
    ("psql postgres://admin:SuperSecret99@db.example.com:5432/shop", true, "SuperSecret99"),
    ("Server=tcp:x.database.windows.net;Database=shop;User ID=me;Password=Pa55word!;", true, "Pa55word!"),
    ("DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5MGFiY2RlZg==;", true, "Zm9vYmFy"),
    (
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA1234567890abcdef\nQWERTYuiop\n-----END RSA PRIVATE KEY-----\nafter",
        true,
        "MIIEpAIBAAKCAQEA",
    ),
    ("-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ", true, "b3BlbnNzaC1rZXkt"),
    ("token glpat-AbCdEfGhIjKlMnOpQrSt in CI", true, "glpat-AbCdEf"),
    ("HF hf_AbCdEfGhIjKlMnOpQrStUvWxYz01234567", true, "hf_AbCdEf"),
    ("//registry.npmjs.org/:_authToken=npm_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", true, "npm_AbCdEf"),
    ("stripe sk_live_51HxAbCdEfGhIjKl0123456789", true, "51HxAbCdEf"),
    ("the token is 9fJ2kLmN0pQrS7tUvWxYz1A2b3C4d5E6f7G8h, keep it", true, "9fJ2kLmN0pQr"),
    ("git clone https://user:pa55w0rd@internal.example.com/repo.git", true, "pa55w0rd"),
    ("redis://:mypassword123@localhost:6379", true, "mypassword123"),
    ("set SECRET_TOKEN=abc123xyz", true, "abc123xyz"),
    ("$env:API_KEY = \"abcd1234efgh5678\"", true, "abcd1234efgh5678"),
    ("gh auth login --token abcdef123456", true, "abcdef123456"),
    ("Cookie: session=abc123def456", true, "abc123def456"),
    ("my password: correct horse", true, "correct"),
    ("client_secret: 'q8Xv2LpN7rTz'", true, "q8Xv2LpN7rTz"),
    ("the password is hunter2", true, "hunter2"),
    ("export PGPASSWORD=Secret123", true, "Secret123"),
    ("https://acct.blob.core.windows.net/c?sv=2022-11-02&sig=AbCdEf%2FGhIjKl0123456789%3D", true, "AbCdEf"),
    ("curl -u admin:s3cretPass http://x.test", true, "s3cretPass"),
    ("mysql -uroot -pMyPass123 shop", true, "MyPass123"),
    ("docker login -p MyPass123 registry", true, "MyPass123"),
    ("sshpass -p 'Secr3t!' ssh host", true, "Secr3t!"),
    ("key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY end", true, "wJalrXUtnFEMI"),
    ("twilio SKabcdef0123456789abcdef0123456789", true, "SKabcdef0123"),
    ("cargo test -p shop", false, "cargo test -p shop"),
    ("git checkout 3f2a9c1d8e7b6a5f4e3d2c1b0a9f8e7d6c5b4a39", false, "3f2a9c1d8e7b"),
    ("Where is take_from_shelf defined?", false, "take_from_shelf"),
    ("the password field in the login form should be hidden", false, "password field"),
    ("fn token_count(s: &str) -> usize", false, "token_count"),
    ("let key = map.get(k);", false, "map.get(k)"),
    ("author: Jane Doe", false, "Jane"),
    ("the task-1234567890abcdefghijk id", false, "task-1234567890"),
    ("id 550e8400-e29b-41d4-a716-446655440000", false, "550e8400"),
    ("open crates/nmcore/src/bin/project.rs line 40", false, "project.rs"),
    ("password: <your password>", false, "<your password>"),
    ("API_KEY=${API_KEY} cargo run", false, "${API_KEY}"),
    ("token: string;", false, "string"),
    ("monkey=banana", false, "banana"),
    ("https://example.com/docs?page=2", false, "page=2"),
    ("Basic usage of the API", false, "Basic usage"),
    ("Bearer tokens expire after an hour", false, "tokens expire"),
    ("sha256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", false, "9f86d081"),
    ("const MAX_TOKENS: usize = 4096;", false, "4096"),
    ("max_tokens=4096", false, "4096"),
    ("tokenizer: qwen3", false, "qwen3"),
    ("mail me at user@example.com", false, "user@example.com"),
    ("see https://docs.rs/serde/1.0.0/serde/", false, "docs.rs"),
    ("git@github.com:org/repo.git", false, "org/repo.git"),
    ("ThisIsAVeryLongCamelCaseIdentifierName", false, "CamelCase"),
    ("C:\\Users\\me\\AppData\\Local\\Temp\\claude\\bba35aeb099ddd6d\\x", false, "bba35aeb"),
    ("ghp_short", false, "ghp_short"),
    ("if a == b && c != d { x = y }", false, "x = y"),
    ("the password is wrong", false, "is wrong"),
    ("cargo build -p nmcore -u x", false, "-p nmcore"),
    ("curl -u admin http://x.test", false, "-u admin"),
    ("the secretary signed it", false, "secretary"),
];

/// Section check: every true case loses its secret word and gains a marker; every false case
/// comes back unchanged.
/// # Errors
/// The cases that failed, one line.
pub fn check() -> Result<(), String> {
    let bad: Vec<String> = CASES
        .iter()
        .filter_map(|(text, secret, word)| {
            let got = drop_secrets(text);
            let ok = if *secret { !got.contains(word) && got.contains("[secret dropped: ") } else { got == *text };
            (!ok).then(|| format!("{text:?} gave {got:?}"))
        })
        .collect();
    if bad.is_empty() { Ok(()) } else { Err(format!("secrets: {} of {} cases wrong: {}", bad.len(), CASES.len(), bad.join(" | "))) }
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_case() {
        super::check().unwrap();
    }
    #[test]
    fn marker_names_kind() {
        let got = super::drop_secrets("password=hunter22");
        assert_eq!(got, "password=[secret dropped: password]");
        assert_eq!(super::kinds("psql postgres://a:b1c2d3@h/db"), vec!["password in a connection string"]);
    }
}
