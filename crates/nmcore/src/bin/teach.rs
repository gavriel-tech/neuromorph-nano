//! A live teacher for distillation: the user's agent (Claude Code, Antigravity or any MCP client,
//! with a cloud or a local model behind it) or a local open-weights model answers the Nano's
//! teaching prompts directly, with no file in between.
//!
//! - `start`: the user (or the agent) opens a teaching set: the prompts to be answered, the
//!   instruction the teacher follows, and the licence of what the teacher writes.
//! - `next`: the prompts of the set not yet answered, with the instruction, for the teacher.
//! - `answer`: the teacher's answers (a completion, and optionally a verdict of accept or reject
//!   with the paths and tool it concerns, for a preference cartridge). They go into the teacher
//!   data through the same import as a file (`distil::import`): split by the prompt's hash,
//!   labelled `used`, listed as reference material with the teacher's name and the licence, audited
//!   and undone in Audit. The teacher is named by the caller (an MCP client's name, or `local`).
//! - A local open-weights model answers a set's open prompts through the lab (`lab run distil
//!   teach --set <set>`), started from the interface, MCP or the API as a lab build.
//!
//! Teaching sets are kept in `memory/training/teach/<set>.json` (prompts, instruction, licence,
//! answered prompt ids). Nothing leaves the machine through the Nano: what an agent sends to its
//! own model is the agent's.
use nmjson::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Prompts one teaching set holds at most.
const MAX_PROMPTS: usize = 2000;
/// Prompts one `next` call gives at most.
const MAX_NEXT: usize = 50;

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}

fn dir(home: &Path) -> PathBuf {
    nmtype::examples::dir(&home.join("memory")).join("teach")
}
fn set_file(home: &Path, set: &str) -> PathBuf {
    dir(home).join(format!("{}.json", nmapi::distil::slug(set)))
}

/// A prompt's id within a set: `p-` and 12 hex digits of its SHA-256.
#[must_use]
pub fn prompt_id(prompt: &str) -> String {
    format!("p-{}", nmhash::sha256_bytes(prompt.as_bytes()).get(..12).unwrap_or(""))
}

/// One teaching set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Set {
    pub name: String,
    pub instruction: String,
    pub licence: String,
    pub prompts: Vec<String>,
    pub answered: BTreeSet<String>,
    pub made_ms: u64,
}

impl Set {
    fn to_value(&self) -> Value {
        obj(vec![
            ("schema", s("nm-teach-set/1")),
            ("name", s(&self.name)),
            ("instruction", s(&self.instruction)),
            ("licence", s(&self.licence)),
            ("prompts", Value::Array(self.prompts.iter().map(|p| s(p)).collect())),
            ("answered", Value::Array(self.answered.iter().map(|p| s(p)).collect())),
            ("made_ms", n(self.made_ms)),
        ])
    }
    fn from_value(v: &Value) -> Self {
        let strs = |k: &str| -> Vec<String> { v.get(k).and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect() };
        Self {
            name: text(v, "name"),
            instruction: text(v, "instruction"),
            licence: text(v, "licence"),
            prompts: strs("prompts"),
            answered: strs("answered").into_iter().collect(),
            made_ms: v.get("made_ms").and_then(Value::integer).unwrap_or(0),
        }
    }
    fn open(&self) -> Vec<(String, &String)> {
        self.prompts.iter().map(|p| (prompt_id(p), p)).filter(|(id, _)| !self.answered.contains(id)).collect()
    }
}

/// A teaching set, read.
/// # Errors
/// No such set.
pub fn read(home: &Path, set: &str) -> Result<Set, String> {
    let f = set_file(home, set);
    let b = std::fs::read(&f).map_err(|_| format!("no teaching set {set}: start one with its prompts"))?;
    Ok(Set::from_value(&nmjson::parse(&b).map_err(|e| format!("{}: {e}", f.display()))?))
}

fn write(home: &Path, set: &Set) -> Result<(), String> {
    let f = set_file(home, &set.name);
    if let Some(d) = f.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    std::fs::write(&f, nmjson::pretty::pretty(&set.to_value())).map_err(|e| format!("{}: {e}", f.display()))
}

/// Every teaching set with how many prompts are open.
#[must_use]
pub fn list(home: &Path) -> Value {
    let mut rows = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir(home)) {
        let mut names: Vec<String> = rd.flatten().filter_map(|e| e.file_name().to_str().and_then(|x| x.strip_suffix(".json")).map(str::to_owned)).collect();
        names.sort();
        for name in names {
            if let Ok(st) = read(home, &name) {
                rows.push(obj(vec![("set", s(&st.name)), ("prompts", n(st.prompts.len() as u64)), ("open", n(st.open().len() as u64)), ("instruction", s(&st.instruction)), ("licence", s(&st.licence))]));
            }
        }
    }
    obj(vec![("sets", Value::Array(rows))])
}

/// Opens (or adds prompts to) a teaching set.
/// # Errors
/// No prompts, or too many, or the set not writable.
pub fn start(home: &Path, set: &str, prompts: &[String], instruction: &str, licence: &str, by: &str) -> Result<Value, String> {
    let name = nmapi::distil::slug(set);
    let mut st = read(home, &name).unwrap_or(Set { name: name.clone(), made_ms: nmtype::now_ms(), ..Set::default() });
    let have: BTreeSet<String> = st.prompts.iter().map(|p| prompt_id(p)).collect();
    let fresh: Vec<String> = prompts.iter().map(|p| p.trim().to_owned()).filter(|p| !p.is_empty() && p.chars().count() <= nmtype::examples::DISTIL_PROMPT_MAX).filter(|p| !have.contains(&prompt_id(p))).collect();
    if fresh.is_empty() && st.prompts.is_empty() {
        return Err("a teaching set needs prompts: the questions or tasks the teacher answers".to_owned());
    }
    if st.prompts.len() + fresh.len() > MAX_PROMPTS {
        return Err(format!("a teaching set holds at most {MAX_PROMPTS} prompts"));
    }
    let added = fresh.len();
    st.prompts.extend(fresh);
    if !instruction.trim().is_empty() {
        instruction.trim().clone_into(&mut st.instruction);
    }
    if !licence.trim().is_empty() {
        licence.trim().clone_into(&mut st.licence);
    }
    write(home, &st)?;
    let audit = crate::controls::audit(
        home,
        "teach",
        vec![("action", s("start")), ("set", s(&name)), ("added", n(added as u64)), ("by", s(by)), ("label", s(&format!("Teaching set {name}: {added} prompts added ({} open)", st.open().len())))],
    )?;
    Ok(obj(vec![("set", s(&name)), ("added", n(added as u64)), ("open", n(st.open().len() as u64)), ("audit_id", s(&audit))]))
}

/// The open prompts of a set, for the teacher, with the instruction it follows.
/// # Errors
/// No such set.
pub fn next(home: &Path, set: &str, count: usize) -> Result<Value, String> {
    let st = read(home, set)?;
    let open = st.open();
    let rows: Vec<Value> = open.iter().take(count.clamp(1, MAX_NEXT)).map(|(id, p)| obj(vec![("prompt_id", s(id)), ("prompt", s(p))])).collect();
    Ok(obj(vec![
        ("set", s(&st.name)),
        ("instruction", s(&st.instruction)),
        ("open", n(open.len() as u64)),
        ("prompts", Value::Array(rows)),
        ("how", s("answer each prompt as the teacher, following the instruction, and send the answers with teach_answer: prompt_id (or prompt) and completion; for a preference cartridge also verdict (accept or reject), paths and tool; for the Reranker the prompt is a query, the completion a document, and score your judgment from 0 to 1 that the document answers the query")),
    ]))
}

/// The teacher's answers, written into the teacher data through the import (audited, undone in
/// Audit), and the answered prompts marked.
/// # Errors
/// No answers that read, or the import refusing.
pub fn answer(home: &Path, set: &str, answers: &[Value], teacher: &str, licence: &str, by: &str) -> Result<Value, String> {
    let mut st = read(home, set)?;
    let by_id: std::collections::BTreeMap<String, String> = st.prompts.iter().map(|p| (prompt_id(p), p.clone())).collect();
    let teacher = if teacher.trim().is_empty() { "the user's agent" } else { teacher.trim() };
    let licence = if licence.trim().is_empty() {
        if st.licence.is_empty() { "teacher outputs under the user's own terms".to_owned() } else { st.licence.clone() }
    } else {
        licence.trim().to_owned()
    };
    let mut lines = Vec::new();
    let mut ids = Vec::new();
    for a in answers {
        let prompt = match text(a, "prompt_id") {
            id if !id.is_empty() => by_id.get(&id).cloned().ok_or_else(|| format!("{id}: not a prompt of set {}", st.name))?,
            _ => text(a, "prompt"),
        };
        let completion = text(a, "completion");
        if prompt.trim().is_empty() || completion.trim().is_empty() {
            return Err("each answer needs a prompt (or prompt_id) and a completion".to_owned());
        }
        let mut row = vec![("prompt", s(&prompt)), ("completion", s(&completion)), ("teacher", s(teacher)), ("licence", s(&licence))];
        let verdict = text(a, "verdict");
        if !verdict.is_empty() {
            if verdict != "accept" && verdict != "reject" {
                return Err(format!("verdict {verdict}: accept or reject"));
            }
            row.push(("verdict", s(&verdict)));
            row.push(("paths", a.get("paths").cloned().unwrap_or(Value::Array(Vec::new()))));
            row.push(("tool", s(&text(a, "tool"))));
        }
        // A teacher's score of the completion for the prompt, 0 to 1 (the Reranker's teacher data:
        // the prompt a query, the completion a document).
        match a.get("score") {
            Ok(Value::Number(x) | Value::String(x)) => match x.trim().parse::<f64>() {
                Ok(v) if v.is_finite() && (0.0..=1.0).contains(&v) => row.push(("score", Value::Number(format!("{v}")))),
                _ => return Err(format!("score {x}: a number from 0 to 1")),
            },
            Ok(Value::Null) | Err(_) => {}
            Ok(_) => return Err("score: a number from 0 to 1".to_owned()),
        }
        lines.push(obj(row).encode().map_err(|e| e.to_string())?);
        ids.push(prompt_id(&prompt));
    }
    if lines.is_empty() {
        return Err("no answers".to_owned());
    }
    // The answers go through the same import as a file, kept beside the set as the record of what
    // the teacher sent.
    let rec = dir(home).join(format!("{}-{}.jsonl", st.name, nmtype::now_ms()));
    std::fs::create_dir_all(dir(home)).map_err(|e| e.to_string())?;
    std::fs::write(&rec, lines.join("\n") + "\n").map_err(|e| format!("{}: {e}", rec.display()))?;
    let imported = crate::distil::import(home, &rec, Some(&st.name), Some(&licence), Some(teacher), by)?;
    st.answered.extend(ids);
    write(home, &st)?;
    Ok(obj(vec![("set", s(&st.name)), ("open", n(st.open().len() as u64)), ("import", imported)]))
}

/// The prompts file and output file of a local teacher's run on a set's open prompts (prompts
/// joined by `---` lines, as the lab reads them).
/// # Errors
/// No such set, no open prompts, or the folder not writable.
pub fn local_files(home: &Path, set: &str) -> Result<(PathBuf, PathBuf, String, usize), String> {
    let st = read(home, set)?;
    let open = st.open();
    if open.is_empty() {
        return Err(format!("teaching set {set} has no open prompts"));
    }
    std::fs::create_dir_all(dir(home)).map_err(|e| e.to_string())?;
    let stamp = nmtype::now_ms();
    let pf = dir(home).join(format!("{}-prompts-{stamp}.txt", st.name));
    let of = dir(home).join(format!("{}-local-{stamp}.jsonl", st.name));
    let text: Vec<&str> = open.iter().map(|(_, p)| p.as_str()).collect();
    std::fs::write(&pf, text.join("\n---\n")).map_err(|e| format!("{}: {e}", pf.display()))?;
    Ok((pf, of, st.instruction.clone(), open.len()))
}

/// After a local teacher's run: its answers imported into the teacher data (their top
/// log-probabilities kept) and the answered prompts marked.
/// # Errors
/// No output, or the import refusing.
pub fn local_done(home: &Path, set: &str, out: &Path, teacher: &str, by: &str) -> Result<Value, String> {
    let mut st = read(home, set)?;
    let body = std::fs::read_to_string(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let ids: Vec<String> = body.lines().filter_map(|l| nmjson::parse(l.trim().as_bytes()).ok()).map(|v| prompt_id(&text(&v, "prompt"))).collect();
    let licence = if st.licence.is_empty() { "Apache-2.0 (teacher outputs of the Qwen3 open weights)".to_owned() } else { st.licence.clone() };
    let imported = crate::distil::import(home, out, Some(&st.name), Some(&licence), Some(teacher), by)?;
    st.answered.extend(ids);
    write(home, &st)?;
    Ok(obj(vec![("set", s(&st.name)), ("open", n(st.open().len() as u64)), ("import", imported)]))
}

/// `POST /api/training/distil/teach`.
/// # Errors
/// As each action.
pub fn api(home: &Path, v: &Value, by: &str) -> Result<Value, String> {
    let set = text(v, "set");
    let need_set = || if set.is_empty() { Err("set: the teaching set's name".to_owned()) } else { Ok(set.as_str()) };
    match text(v, "action").as_str() {
        "" | "list" => Ok(list(home)),
        "start" => {
            let prompts: Vec<String> = v.get("prompts").and_then(Value::array).unwrap_or_default().iter().filter_map(|p| p.string().ok().map(str::to_owned)).collect();
            start(home, need_set()?, &prompts, &text(v, "instruction"), &text(v, "licence"), by)
        }
        "next" => next(home, need_set()?, v.get("count").and_then(Value::integer).ok().and_then(|c| usize::try_from(c).ok()).unwrap_or(10)),
        "answer" => {
            let answers = v.get("answers").and_then(Value::array).map_err(|_| "answers: a list of {prompt_id or prompt, completion}".to_owned())?;
            answer(home, need_set()?, answers, &text(v, "teacher"), &text(v, "licence"), by)
        }
        other => Err(format!("action {other}: use list, start, next or answer")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_live_teacher_answers_a_set_and_the_pairs_reach_the_teacher_data() {
        let h = std::env::temp_dir().join(format!("nano-teach-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&h);
        std::fs::create_dir_all(h.join("memory")).unwrap();
        let prompts: Vec<String> = (0..6).map(|i| format!("Write a one-line note of passage {i}")).collect();
        start(&h, "notes", &prompts, "one line, the main identifiers first", "own", "test").unwrap();
        let nx = next(&h, "notes", 4).unwrap();
        let open: Vec<Value> = nx.get("prompts").and_then(Value::array).unwrap().to_vec();
        assert_eq!(open.len(), 4);
        let answers: Vec<Value> = open.iter().map(|p| obj(vec![("prompt_id", s(&text(p, "prompt_id"))), ("completion", s("Covers the note"))])).collect();
        let r = answer(&h, "notes", &answers, "claude-code", "", "test").unwrap();
        assert_eq!(r.get("open").and_then(Value::integer).unwrap(), 2, "four answered, two open");
        let rows = nmtype::examples::read_distil(&h.join("memory")).unwrap();
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|r| r.teacher == "claude-code" && r.licence == "own" && r.set == "notes" && r.healthy()));
        assert!(answer(&h, "notes", &[obj(vec![("prompt_id", s("p-000000000000")), ("completion", s("x"))])], "x", "", "test").is_err(), "an unknown prompt is refused");
        let _ = std::fs::remove_dir_all(&h);
    }
}
