//! What session open hands the agent: the memories of `1-core` (the most used, at most 32), read
//! from the core view the journal writes (`1-core/items.json`), each anchor re-read fresh from its
//! source. The source decides: a memory is handed on only when every rereadable anchor reads back
//! with the content it was made from (at its lines, or moved to other lines); one whose source
//! changed or is gone is held back, and the serving process reconsolidates it at its next recall.
//! A memory handed on says where to look; it is never evidence by itself.
use crate::record::{Part, Record, Relocation, State};
use nmjson::Value;
use std::path::Path;

/// One core memory as handed on: its gist and the places it rests on now (`path:a-b`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Core {
    pub id: String,
    pub text: String,
    pub places: Vec<String>,
}

/// The core memories of memory folder `dir` whose anchors read back fresh under `root`, most used
/// first (the view's order). Missing or unreadable view: none.
#[must_use]
pub fn for_session(dir: &Path, root: &Path) -> Vec<Core> {
    let Some(v) = crate::store::get(&dir.join("1-core").join("items.json")) else { return Vec::new() };
    let mut out = Vec::new();
    for item in v.get("items").and_then(Value::array).unwrap_or_default() {
        let Ok(r) = Record::from_value(item) else { continue };
        if r.state != State::Active {
            continue;
        }
        let mut places = Vec::new();
        let holds = r.anchors.iter().filter(|a| a.source.rereadable()).all(|a| {
            let file = a.location.split('/').fold(root.to_path_buf(), |p, c| p.join(c));
            let Ok(text) = std::fs::read_to_string(file) else { return false };
            let at = match (crate::record::relocate_lines(a, &text), &a.part) {
                (Relocation::Valid, Part::Lines { start, end }) => Some((*start, *end)),
                (Relocation::Moved { start, end }, _) => Some((start, end)),
                _ => None,
            };
            at.inspect(|(start, end)| places.push(format!("{}:{start}-{end}", a.location))).is_some()
        });
        if holds {
            out.push(Core { id: r.id.clone(), text: r.text.clone(), places });
        }
    }
    out
}

/// Section check: of three core memories, one whose lines read back is handed on at its place, one
/// whose lines moved is handed on at their new place, and one whose lines changed is held back; a
/// folder with no core view hands on nothing.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    let root = nmtype::check_dir("nmmem-session-open-root").map_err(|e| e.to_string())?;
    let dir = nmtype::check_dir("nmmem-session-open").map_err(|e| e.to_string())?;
    if !for_session(&dir, &root).is_empty() {
        return Err("nmmem.session_open: a folder with no core view handed something on".into());
    }
    std::fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    let write = |p: &str, t: &str| std::fs::write(root.join(p), t).map_err(|e| e.to_string());
    write("src/a.rs", "pub fn alpha() -> u8 {\n    1\n}\n")?;
    write("src/b.rs", "pub fn beta() -> u8 {\n    2\n}\n")?;
    write("src/c.rs", "pub fn gamma() -> u8 {\n    3\n}\n")?;
    let rec = |q: &str, cite: &str| {
        let ep = crate::episodic::Episode::new("neuromorph", "check", q, cite, vec![format!("{cite} lines 1..3")], 1);
        crate::migrate::record_of(&ep, crate::record::Tier::Core, 1, &root).to_value()
    };
    let items = vec![rec("where is alpha", "src/a.rs:1"), rec("where is beta", "src/b.rs:1"), rec("where is gamma", "src/c.rs:1")];
    std::fs::create_dir_all(dir.join("1-core")).map_err(|e| e.to_string())?;
    let view = crate::store::obj(vec![("record", crate::store::s("memory-items")), ("tier", crate::store::s("1-core")), ("items", Value::Array(items))]);
    crate::store::put(&dir.join("1-core").join("items.json"), &view).map_err(|e| e.to_string())?;
    // beta moves down two lines; gamma changes.
    write("src/b.rs", "// moved\n\npub fn beta() -> u8 {\n    2\n}\n")?;
    write("src/c.rs", "pub fn gamma() -> u8 {\n    4\n}\n")?;
    let got: Vec<Vec<String>> = for_session(&dir, &root).into_iter().map(|c| c.places).collect();
    if got != [vec!["src/a.rs:1-3".to_owned()], vec!["src/b.rs:3-5".to_owned()]] {
        return Err(format!("nmmem.session_open: handed on {got:?}"));
    }
    Ok(())
}
