//! The multi-session continual learning soak (roadmap 2026-11-23, Phase 2 item 3):
//! `nano-lab run soak --questions <tuning set> [--root <repo>] [--sessions N] [--n limit] [--mode m]`.
//!
//! Sessions 1 to N each ask every question of a tuning set through the served call, score the
//! replies against the answer key (right first, in the top 3, in the pack, every answer in the pack,
//! decoys withheld or marked low confidence), and write the verified outcomes a user's work would
//! have given: one learning example per answered question (`training/examples.jsonl`, features only,
//! no question or passage text), preference signals (an edit kept in an answer's file, an edit
//! reverted in a wrongly first-cited file) and G3's verdict on the call (the answer key's reciprocal
//! place). Between sessions the unified idle learning cycle runs (G3's prior replay, the preference
//! cartridge fit, the engine's gated Cycle B; `nmlearn::idle`).
//!
//! It then checks for forgetting (decoys held and right first never below session 1) and that the
//! sensory boundary checks stay green after every session (the data-class tables of `nmtype::class`,
//! G3's decision-site and configuration checks, and no question or passage text in the store or the
//! cycle records).
//!
//! Everything it writes stays under `scratch/soak/<stamp>/` (its own memory folder, muscle store and
//! library), never the project's memory or the Nano's library. Unseen verification sets are refused
//! as input: `tests/dev-large-2` and `tests/real-use-2` are never tuned or fitted on.
use nmjson::Value;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Folder names of unseen verification sets the soak refuses (never tuned or fitted on).
pub const REFUSED: [&str; 2] = ["dev-large-2", "real-use-2"];
/// Sessions run by default.
pub const DEFAULT_SESSIONS: usize = 5;

const USAGE: &str = "usage: nano-lab run soak --questions <tuning question set> [--root <repo>] [--sessions N] [--n limit] [--mode plain|fast|full|flow]";

/// The soak's arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoakArgs {
    pub questions: PathBuf,
    pub root: PathBuf,
    pub sessions: usize,
    pub limit: Option<usize>,
    pub mode: Option<String>,
}

/// Why a path is refused as soak input: it names an unseen verification set (a folder named
/// `dev-large-2` or `real-use-2`, as written or once resolved).
#[must_use]
pub fn refused_input(p: &Path) -> Option<String> {
    let named = |q: &Path| {
        q.components().find_map(|c| {
            let name = c.as_os_str().to_string_lossy().to_ascii_lowercase();
            REFUSED.iter().find(|r| **r == name).map(|r| (*r).to_owned())
        })
    };
    let hit = named(p).or_else(|| std::fs::canonicalize(p).ok().and_then(|c| named(&c)));
    hit.map(|r| format!("{} is in tests/{r}, an unseen verification set: it is never tuned or fitted on, so the soak refuses it", p.display()))
}

/// Reads the soak's arguments (`args` from `run soak` on), with relative paths taken under `home`.
/// # Errors
/// A missing or malformed value, or an input in an unseen verification set.
pub fn parse_args(args: &[String], home: &Path) -> Result<SoakArgs, String> {
    let opt = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
    let abs = |p: String| {
        let p = PathBuf::from(p);
        if p.is_absolute() { p } else { home.join(p) }
    };
    let questions = abs(opt("--questions").ok_or_else(|| format!("--questions is required\n{USAGE}"))?);
    let root = opt("--root").map_or_else(|| home.to_path_buf(), abs);
    let sessions = match opt("--sessions") {
        None => DEFAULT_SESSIONS,
        Some(v) => v.parse::<usize>().ok().filter(|x| *x >= 2).ok_or_else(|| format!("--sessions takes a whole number of 2 or more, got {v}"))?,
    };
    let limit = match opt("--n") {
        None => None,
        Some(v) => Some(v.parse::<usize>().ok().filter(|x| *x > 0).ok_or_else(|| format!("--n takes a whole number above 0, got {v}"))?),
    };
    let mode = opt("--mode");
    if let Some(m) = &mode
        && !matches!(m.as_str(), "plain" | "fast" | "full" | "flow")
    {
        return Err(format!("--mode takes plain, fast, full or flow, got {m}"));
    }
    for p in [&questions, &root] {
        if let Some(why) = refused_input(p) {
            return Err(why);
        }
    }
    Ok(SoakArgs { questions, root, sessions, limit, mode })
}

/// One answer: a file and its line range (whole file when `whole`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Answer {
    pub file: String,
    pub first: usize,
    pub last: usize,
    pub whole: bool,
    pub anchor: Option<String>,
}

/// One question of the tuning set (no answers: a decoy).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Question {
    pub id: String,
    pub kind: String,
    pub text: String,
    pub answers: Vec<Answer>,
}

/// The questions of a question set file's JSON (`{"questions": [...]}`, each with `id`, `kind`,
/// `question` and `answers` of `file`, `lines`, optional `anchor` and `match`).
/// # Errors
/// Not a question set.
pub fn parse_questions(v: &Value) -> Result<Vec<Question>, String> {
    let text = |x: &Value, k: &str| x.get(k).and_then(Value::string).ok().map(str::to_owned);
    let list = v.get("questions").and_then(Value::array).map_err(|e| format!("questions: {e}"))?;
    let mut out = Vec::new();
    for q in list {
        let id = text(q, "id").ok_or("a question has no id")?;
        let question = text(q, "question").or_else(|| text(q, "q")).ok_or_else(|| format!("{id}: no question"))?;
        let mut answers = Vec::new();
        for a in q.get("answers").and_then(Value::array).unwrap_or_default() {
            let file = text(a, "file").ok_or_else(|| format!("{id}: an answer has no file"))?.replace('\\', "/");
            let lines: Vec<usize> = a.get("lines").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.integer().ok()).filter_map(|x| usize::try_from(x).ok()).collect();
            let whole = text(a, "match").as_deref() == Some("file") || lines.is_empty();
            let first = lines.first().copied().unwrap_or(0);
            answers.push(Answer { file, first, last: lines.get(1).copied().unwrap_or(first), whole, anchor: text(a, "anchor") });
        }
        out.push(Question { id, kind: text(q, "kind").unwrap_or_else(|| if answers.is_empty() { "decoy".into() } else { "unknown".into() }), text: question, answers });
    }
    Ok(out)
}

/// Moves each answer's range to where its anchor text now is in `root` (the range keeps its
/// length); an anchor not found leaves the range as written.
pub fn locate(root: &Path, qs: &mut [Question]) {
    let mut cache: BTreeMap<String, Option<String>> = BTreeMap::new();
    for a in qs.iter_mut().flat_map(|q| q.answers.iter_mut()) {
        let (Some(anchor), false) = (a.anchor.clone(), a.whole) else { continue };
        let body = cache.entry(a.file.clone()).or_insert_with(|| std::fs::read_to_string(root.join(&a.file)).ok());
        if let Some(i) = body.as_deref().and_then(|b| b.lines().position(|l| l.contains(anchor.as_str()))) {
            let len = a.last.saturating_sub(a.first);
            a.first = i + 1;
            a.last = i + 1 + len;
        }
    }
}

/// How one reply scored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mark {
    pub first: bool,
    pub top3: bool,
    pub pack: bool,
    pub every: bool,
    /// Withheld, or marked low confidence.
    pub held_back: bool,
    /// Mean reciprocal place of the answers in the cited places.
    pub rr: f64,
    pub cited: Vec<(String, usize, usize)>,
}

fn hits(a: &Answer, x: &(String, usize, usize)) -> bool {
    a.file == x.0 && (a.whole || (x.1 <= a.last && x.2 >= a.first))
}

/// Scores a text reply against a question's answers.
#[must_use]
pub fn score(q: &Question, reply: &str) -> Mark {
    let c = nmapi::g3::cited(reply);
    let hit = |x: &(String, usize, usize)| q.answers.iter().any(|a| hits(a, x));
    #[expect(clippy::cast_precision_loss, reason = "places and counts")]
    let rr = if q.answers.is_empty() {
        0.0
    } else {
        q.answers.iter().map(|a| c.iter().position(|x| hits(a, x)).map_or(0.0, |p| 1.0 / (p + 1) as f64)).sum::<f64>() / q.answers.len() as f64
    };
    Mark {
        first: c.first().is_some_and(hit),
        top3: c.iter().take(3).any(hit),
        pack: c.iter().any(hit),
        every: !q.answers.is_empty() && q.answers.iter().all(|a| c.iter().any(|x| hits(a, x))),
        held_back: reply.lines().next().is_some_and(|l| l.contains(" withheld")) || reply.lines().any(|l| l.starts_with("confidence: low")),
        rr,
        cited: c,
    }
}

/// One session's scores.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionScore {
    pub session: usize,
    pub answered: usize,
    pub decoys: usize,
    pub first: usize,
    pub top3: usize,
    pub pack: usize,
    pub every: usize,
    pub decoys_held: usize,
}

/// The session's counts over its questions and marks.
#[must_use]
pub fn summarise(session: usize, qs: &[Question], marks: &[Mark]) -> SessionScore {
    let mut s = SessionScore { session, ..SessionScore::default() };
    for (q, m) in qs.iter().zip(marks) {
        if q.answers.is_empty() {
            s.decoys += 1;
            s.decoys_held += usize::from(m.held_back);
        } else {
            s.answered += 1;
            s.first += usize::from(m.first);
            s.top3 += usize::from(m.top3);
            s.pack += usize::from(m.pack);
            s.every += usize::from(m.every);
        }
    }
    s
}

/// Forgetting found across the sessions: any later session with fewer decoys held or fewer right
/// first than session 1. Empty when there is none.
#[must_use]
pub fn forgetting(scores: &[SessionScore]) -> Vec<String> {
    let Some(base) = scores.first() else { return Vec::new() };
    let mut out = Vec::new();
    for s in scores.iter().skip(1) {
        if s.decoys_held < base.decoys_held {
            out.push(format!("session {}: decoys held {}/{} below session 1 ({}/{})", s.session, s.decoys_held, s.decoys, base.decoys_held, base.decoys));
        }
        if s.first < base.first {
            out.push(format!("session {}: right first {}/{} below session 1 ({}/{})", s.session, s.first, s.answered, base.first, base.answered));
        }
    }
    out
}

fn sha16(t: &str) -> String {
    nmhash::sha256_bytes(t.as_bytes()).chars().take(16).collect()
}
fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn num(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// The learning example a verified outcome gives (features only: hashes, places, labels; no
/// question or passage text).
#[must_use]
pub fn example_row(session: usize, project: &str, q: &Question, m: &Mark, at: u64) -> Value {
    let ask = sha16(&format!("{}|{}", q.id, q.text));
    let anchor = q.answers.first().map_or("", |a| a.file.as_str());
    let split = nmtype::examples::split_of(&ask, anchor).name();
    let mut cands: Vec<Value> = m
        .cited
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, x)| {
            let ok = q.answers.iter().any(|a| hits(a, x));
            obj(vec![
                ("place", s(&format!("{}:{}-{}", x.0, x.1, x.2))),
                ("judged", Value::Bool(true)),
                ("served", Value::Bool(true)),
                ("rank", num(i as u64)),
                ("label", s(if ok { "verified" } else { "passed-over" })),
                ("strength", s("direct")),
            ])
        })
        .collect();
    for a in q.answers.iter().filter(|a| !m.cited.iter().any(|x| hits(a, x))) {
        let place = if a.whole { a.file.clone() } else { format!("{}:{}-{}", a.file, a.first, a.last) };
        cands.push(obj(vec![("place", s(&place)), ("judged", Value::Bool(false)), ("served", Value::Bool(false)), ("label", s("went")), ("strength", s("indirect"))]));
    }
    obj(vec![
        ("v", num(nmtype::examples::FORMAT)),
        ("row", s("learning")),
        ("family", s("learning")),
        ("id", s(&format!("soak-s{session}-{}", q.id))),
        ("pack", s(&format!("soak-s{session}-{}", q.id))),
        ("verdict", s(if m.pack { "verified" } else { "corrected" })),
        ("strength", s("direct")),
        ("split", s(split)),
        ("ask_sha", s(&ask)),
        ("domains", Value::Array(vec![s(&format!("project:{project}")), s(&format!("kind:{}", q.kind))])),
        ("path", s("soak")),
        ("project", s(project)),
        ("at", num(at)),
        ("candidates", Value::Array(cands)),
    ])
}

/// The preference signals a verified outcome gives: an edit kept in the first answer's file, and
/// an edit reverted in the first cited file when that was not an answer.
#[must_use]
pub fn preference_rows(session: usize, project: &str, q: &Question, m: &Mark, at: u64) -> Vec<Value> {
    let ask = sha16(&format!("{}|{}", q.id, q.text));
    let mut out = Vec::new();
    let row = |kind: &str, strength: &str, path: &str, tag: &str| {
        let split = nmtype::examples::split_of(&ask, path).name();
        obj(vec![
            ("v", num(nmtype::examples::FORMAT)),
            ("row", s("preference")),
            ("family", s("preference")),
            ("id", s(&format!("p-soak-s{session}-{}-{tag}", q.id))),
            ("kind", s(kind)),
            ("strength", s(strength)),
            ("session", s(&format!("soak-{session}"))),
            ("at", num(at)),
            ("paths", Value::Array(vec![s(path)])),
            ("scope", s(&format!("project:{project}"))),
            ("split", s(split)),
        ])
    };
    if let Some(a) = q.answers.first() {
        out.push(row("edit-kept", "indirect", &a.file, "k"));
    }
    if let Some(x) = m.cited.first()
        && !q.answers.is_empty()
        && !q.answers.iter().any(|a| hits(a, x))
    {
        out.push(row("edit-reverted", "weak", &x.0, "r"));
    }
    out
}

fn append_rows(file: &Path, rows: &[Value]) -> Result<(), String> {
    if let Some(d) = file.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(file).map_err(|e| format!("{}: {e}", file.display()))?;
    for r in rows {
        writeln!(f, "{}", r.encode().map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", file.display()))?;
    }
    Ok(())
}

/// Keys that would carry sensory text into a behavioural record.
const TEXT_KEYS: [&str; 4] = ["question", "text", "passage", "prompt"];

/// Sensory text found in the soak's behavioural records: a text key in a training row, or a
/// question's words in the cycle records. Empty when clean.
#[must_use]
pub fn leaks(memory: &Path, qs: &[Question]) -> Vec<String> {
    let mut out = Vec::new();
    for file in [nmtype::examples::EXAMPLES, nmtype::examples::PREFERENCES] {
        for r in nmtype::examples::read_rows(memory, file).unwrap_or_default() {
            if let Some(k) = TEXT_KEYS.iter().find(|k| r.get(k).is_ok()) {
                out.push(format!("{file}: a row carries {k}"));
                break;
            }
        }
    }
    let cycles = std::fs::read_to_string(memory.join(nmlearn::idle::DIR).join(nmlearn::idle::CYCLES)).unwrap_or_default();
    if let Some(q) = qs.iter().find(|q| q.text.len() >= 12 && cycles.contains(q.text.as_str())) {
        out.push(format!("cycle records carry the words of question {}", q.id));
    }
    out
}

/// The sensory boundary checks: the data-class and origin tables, G3's decision-site and
/// configuration checks, and `leaks`. Returns the faults (empty: green).
#[must_use]
pub fn boundary_checks(memory: &Path, qs: &[Question]) -> Vec<String> {
    let mut out = Vec::new();
    if let Err(e) = nmtype::class::b0_table() {
        out.push(format!("nmtype.class b0: {e}"));
    }
    if let Err(e) = nmtype::class::origin_table() {
        out.push(format!("nmtype.class origin: {e}"));
    }
    for (name, check) in nmmusc::checks().into_iter().filter(|(n, _)| n.contains("sites") || n.contains("config")) {
        if let Err(e) = check() {
            out.push(format!("{name}: {e}"));
        }
    }
    out.extend(leaks(memory, qs));
    out
}

/// The text of a served reply.
fn as_text(r: Result<Value, nmapi::ApiError>) -> String {
    match r {
        Ok(Value::String(t)) => t,
        Ok(v) => v.encode().unwrap_or_default(),
        Err(e) => format!("error {}: {}", e.code, e.message),
    }
}

/// `nano-lab run soak ...` (module note).
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let own: Vec<String> = args.iter().skip(2).cloned().collect();
    let a = match parse_args(&own, home) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("soak: {e}");
            return ExitCode::FAILURE;
        }
    };
    match soak(&a, home) {
        Ok((report, clean)) => {
            let p = home.join(format!("runs/SOAK-{}.txt", nmtype::now_ms()));
            let _ = std::fs::write(&p, &report);
            print!("{report}");
            println!("report: {}", p.display());
            if clean { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        Err(e) => {
            eprintln!("soak: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The soak itself: (the report, whether it passed: no forgetting and every boundary check green).
fn soak(a: &SoakArgs, home: &Path) -> Result<(String, bool), String> {
    let v = nmjson::parse(&std::fs::read(&a.questions).map_err(|e| format!("{}: {e}", a.questions.display()))?).map_err(|e| format!("{}: {e}", a.questions.display()))?;
    let mut qs = parse_questions(&v)?;
    if let Some(n) = a.limit {
        qs.truncate(n);
    }
    let root = nmsubs::repo::canon(&a.root);
    locate(&root, &mut qs);
    let stamp = nmtype::now_ms();
    let dir = home.join("scratch").join("soak").join(stamp.to_string());
    let memory = dir.join("project-memory");
    let muscle = dir.join("muscle");
    let audit = dir.join("audit.jsonl");
    std::fs::create_dir_all(&memory).map_err(|e| format!("{}: {e}", memory.display()))?;
    let project = nmtype::project::label(home, &root);
    let org = nmapi::forage::Organism::open_with(home, &root, &["neuromorph-nano".to_owned()], false, &dir.join("organism-memory")).map_err(|e| format!("organism: {e}"))?;
    nmapi::tools::install(org).map_err(|e| format!("{}: {}", e.code, e.message))?;
    nmapi::g3::enable(home, Some(&muscle)).map_err(|e| format!("muscle: {e}"))?;
    let api = nmapi::tools::api();
    let mut out = String::new();
    let _ = writeln!(out, "soak: {} questions from {} on {}, {} sessions, everything under {}", qs.len(), a.questions.display(), root.display(), a.sessions, dir.display());
    let mut scores: Vec<SessionScore> = Vec::new();
    let mut green = true;
    let mut audits = 0_u64;
    for session in 1..=a.sessions {
        let mut marks = Vec::new();
        for q in &qs {
            let mut input = BTreeMap::from([("question".to_owned(), s(&q.text)), ("form".to_owned(), s("full"))]);
            if let Some(m) = a.mode.as_ref().filter(|m| m.as_str() != "plain") {
                input.insert("mode".to_owned(), s(m));
            }
            let before = nmapi::g3::last_call();
            let reply = as_text(api.dispatch("nm_context_pack", &Value::Object(input)));
            let _ = nmapi::tools::take_diagnostics();
            let m = score(q, &reply);
            if !q.answers.is_empty() {
                if let Some(id) = nmapi::g3::last_call().filter(|id| Some(id) != before.as_ref()) {
                    let _ = nmapi::g3::verdict(&id, m.rr);
                }
                let at = nmtype::now_ms();
                let tdir = nmtype::examples::dir(&memory);
                append_rows(&tdir.join(nmtype::examples::EXAMPLES), &[example_row(session, &project, q, &m, at)])?;
                append_rows(&tdir.join(nmtype::examples::PREFERENCES), &preference_rows(session, &project, q, &m, at))?;
            }
            marks.push(m);
        }
        let sc = summarise(session, &qs, &marks);
        let faults = boundary_checks(&memory, &qs);
        green &= faults.is_empty();
        let _ = writeln!(
            out,
            "session {session}: first {}/{a_n}  top3 {}/{a_n}  pack {}/{a_n}  every {}/{a_n}  decoys {}/{}  boundary {}",
            sc.first,
            sc.top3,
            sc.pack,
            sc.every,
            sc.decoys_held,
            sc.decoys,
            if faults.is_empty() { "green".to_owned() } else { format!("RED: {}", faults.join("; ")) },
            a_n = sc.answered
        );
        scores.push(sc);
        if session < a.sessions {
            let mut learners: Vec<Box<dyn nmlearn::idle::Learner>> = vec![Box::new(nmapi::g3::G3Learner { audit: audit.clone(), memory: memory.clone() })];
            learners.push(Box::new(nmlearn::pref_fit::PreferenceLearner::new(&dir, &memory, &project)));
            learners.push(Box::new(nmlearn::idle::EngineLearner { memory: memory.clone(), sync: true }));
            let mut write_audit = |rec: &Value| {
                audits += 1;
                let id = format!("soak-a{audits}");
                let row = obj(vec![("id", s(&id)), ("kind", s("learning-cycle")), ("cycle", rec.get("id").cloned().unwrap_or(Value::Null)), ("time_ms", num(nmtype::now_ms()))]);
                append_rows(&audit, &[row]).ok().map(|()| id)
            };
            match nmlearn::idle::run_cycle(&memory, &mut learners, "soak", "session", &mut write_audit) {
                Ok(rec) => {
                    let _ = writeln!(out, "  cycle {} after session {session}:", rec.id);
                    for l in &rec.learners {
                        let f = |x: Option<f64>| x.map_or_else(|| "-".to_owned(), |v| format!("{v:.4}"));
                        let _ = writeln!(out, "    {}: ran {}, held out {}, current {}, candidate {}, promoted {}; {}", l.name, l.ran, l.held_out, f(l.current), f(l.candidate), l.promoted, l.note);
                    }
                }
                Err(e) => {
                    let _ = writeln!(out, "  cycle after session {session}: {e}");
                }
            }
        }
    }
    let _ = nmapi::g3::disable();
    let lost = forgetting(&scores);
    if let (Some(f), Some(l)) = (scores.first(), scores.last()) {
        let d = |x: usize, y: usize| i64::try_from(y).unwrap_or(0) - i64::try_from(x).unwrap_or(0);
        let _ = writeln!(
            out,
            "lift session 1 to {}: first {:+}, top3 {:+}, pack {:+}, every {:+}, decoys held {:+}",
            l.session,
            d(f.first, l.first),
            d(f.top3, l.top3),
            d(f.pack, l.pack),
            d(f.every, l.every),
            d(f.decoys_held, l.decoys_held)
        );
    }
    let _ = writeln!(out, "forgetting: {}", if lost.is_empty() { "none".to_owned() } else { lost.join("; ") });
    let _ = writeln!(out, "sensory boundary: {}", if green { "green in every session" } else { "RED in at least one session" });
    Ok((out, lost.is_empty() && green))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_owned()).collect()
    }

    #[test]
    fn unseen_verification_sets_are_refused() {
        let home = Path::new("D:/nano");
        for bad in ["tests/dev-large-2/questions.json", "tests\\real-use-2\\q.json", "D:/x/tests/DEV-LARGE-2/a.json"] {
            let e = parse_args(&args(&["--questions", bad]), home);
            assert!(e.as_ref().is_err_and(|m| m.contains("unseen verification set")), "{bad}: {e:?}");
        }
        let e = parse_args(&args(&["--questions", "tests/dev/closure.json", "--root", "tests/dev-large-2/material-code"]), home);
        assert!(e.is_err_and(|m| m.contains("dev-large-2")), "a root in an unseen set is refused too");
        assert!(refused_input(Path::new("tests/dev-large/questions.json")).is_none(), "dev-large is a tuning set");
    }

    #[test]
    fn arguments_parse_with_defaults_and_refuse_bad_values() {
        let home = Path::new("D:/nano");
        let a = parse_args(&args(&["--questions", "tests/dev/closure.json"]), home).expect("parse");
        assert_eq!(a.questions, home.join("tests/dev/closure.json"));
        assert_eq!(a.root, home.to_path_buf());
        assert_eq!((a.sessions, a.limit, a.mode), (DEFAULT_SESSIONS, None, None));
        let b = parse_args(&args(&["--questions", "q.json", "--sessions", "3", "--n", "7", "--mode", "full", "--root", "D:/repo"]), home).expect("parse");
        assert_eq!((b.sessions, b.limit, b.mode.as_deref(), b.root), (3, Some(7), Some("full"), PathBuf::from("D:/repo")));
        assert!(parse_args(&args(&[]), home).is_err());
        assert!(parse_args(&args(&["--questions", "q.json", "--sessions", "1"]), home).is_err(), "one session cannot show lift");
        assert!(parse_args(&args(&["--questions", "q.json", "--n", "0"]), home).is_err());
        assert!(parse_args(&args(&["--questions", "q.json", "--mode", "turbo"]), home).is_err());
    }

    fn set() -> Value {
        nmjson::parse(
            br#"{"questions":[
              {"id":"d01","kind":"name","question":"Where is tide_level?","answers":[{"file":"src\\lib.rs","anchor":"pub fn tide_level","lines":[10,12],"match":"lines"}]},
              {"id":"d02","kind":"file","question":"Which file holds the docs?","answers":[{"file":"README.md","match":"file"}]},
              {"id":"c29","kind":"decoy","question":"Which function writes to PostgreSQL?","answers":[]}
            ]}"#,
        )
        .expect("json")
    }

    #[test]
    fn question_sets_parse_with_decoys_and_whole_file_answers() {
        let qs = parse_questions(&set()).expect("parse");
        assert_eq!(qs.len(), 3);
        let a = qs.first().and_then(|q| q.answers.first()).expect("answer");
        assert_eq!((a.file.as_str(), a.first, a.last, a.whole), ("src/lib.rs", 10, 12, false));
        assert!(qs.get(1).and_then(|q| q.answers.first()).is_some_and(|a| a.whole));
        assert!(qs.get(2).is_some_and(|q| q.answers.is_empty() && q.kind == "decoy"));
        assert!(parse_questions(&nmjson::parse(b"{\"x\":1}").expect("json")).is_err());
    }

    #[test]
    fn replies_score_first_top3_pack_every_and_decoys() {
        let qs = parse_questions(&set()).expect("parse");
        let (q, decoy) = (qs.first().expect("q"), qs.get(2).expect("decoy"));
        let reply = "@@ src/other.rs:1-5\nx\n@@ src/lib.rs:11-11\ny\n";
        let m = score(q, reply);
        assert!(!m.first && m.top3 && m.pack && m.every);
        assert!((m.rr - 0.5).abs() < 1e-12);
        let held = score(decoy, "pack withheld: nothing answers this\n");
        assert!(held.held_back);
        let low = score(decoy, "pack\nconfidence: low\n@@ a.rs:1-2\n");
        assert!(low.held_back);
        let s1 = summarise(1, &qs, &[m.clone(), Mark::default(), held]);
        assert_eq!((s1.answered, s1.decoys, s1.first, s1.top3, s1.pack, s1.every, s1.decoys_held), (2, 1, 0, 1, 1, 1, 1));
        let s2 = summarise(2, &qs, &[m, Mark::default(), Mark::default()]);
        let lost = forgetting(&[s1.clone(), s2]);
        assert_eq!(lost.len(), 1, "{lost:?}");
        assert!(forgetting(&[s1.clone(), s1]).is_empty());
    }

    #[test]
    fn scripted_outcomes_carry_no_text_and_keep_the_split_across_sessions() {
        let qs = parse_questions(&set()).expect("parse");
        let q = qs.first().expect("q");
        let m = score(q, "@@ src/other.rs:1-5\nx\n");
        let a = example_row(1, "p", q, &m, 5);
        let b = example_row(2, "p", q, &m, 6);
        assert_eq!(a.get("split").ok(), b.get("split").ok(), "the same question keeps its split");
        assert_ne!(a.get("id").ok(), b.get("id").ok());
        assert_eq!(a.get("verdict").and_then(Value::string).ok(), Some("corrected"));
        let enc = a.encode().expect("encode");
        assert!(!enc.contains("tide_level") && TEXT_KEYS.iter().all(|k| a.get(k).is_err()));
        let prefs = preference_rows(1, "p", q, &m, 5);
        let kinds: Vec<&str> = prefs.iter().filter_map(|r| r.get("kind").and_then(Value::string).ok()).collect();
        assert_eq!(kinds, vec!["edit-kept", "edit-reverted"]);
        assert!(preference_rows(1, "p", qs.get(2).expect("decoy"), &Mark::default(), 5).is_empty());
    }

    #[test]
    fn anchors_move_answer_ranges_and_leaks_are_found() {
        let dir = nmtype::nano_root().join("scratch").join("unit-tests").join(format!("nmlab-soak-{}-{}", std::process::id(), nmtype::now_ms()));
        std::fs::create_dir_all(dir.join("src")).expect("dir");
        std::fs::write(dir.join("src/lib.rs"), "// a\n// b\npub fn tide_level() {}\n").expect("write");
        let mut qs = parse_questions(&set()).expect("parse");
        locate(&dir, &mut qs);
        let a = qs.first().and_then(|q| q.answers.first()).expect("answer");
        assert_eq!((a.first, a.last), (3, 5));
        let mem = dir.join("mem");
        assert!(leaks(&mem, &qs).is_empty());
        std::fs::create_dir_all(nmtype::examples::dir(&mem)).expect("dir");
        std::fs::write(nmtype::examples::dir(&mem).join(nmtype::examples::EXAMPLES), "{\"id\":\"x\",\"question\":\"Where is tide_level?\"}\n").expect("write");
        assert_eq!(leaks(&mem, &qs).len(), 1);
    }
}
