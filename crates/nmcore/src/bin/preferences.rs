//! Stated preference cartridges from the program's side: the `/api/preferences` routes (lent to
//! MCP as `nm_learning op preference_*`), the `preferences` command, and the undo of each write.
//! The rules, the parser, the transforms and the library writes are `nmlearn::stated`; this module
//! adds the library lock, the two-step confirmation of every write (a preview first, then the same
//! request with `confirm: true`), the audit rows and their undo.
//!
//! Every write goes through the cartridge library: a new version of a `preference` cartridge
//! (`Library::add_cartridge`), the stacks that held the version before moved to a new version, and
//! the uses that ran them switched on. Plugging in and out is a library use switch
//! (`library::switch_use`), the same as on the Cartridges page's decks. Audit rows are `library`
//! rows (`preference-say`, `preference-edit`, `preference-remove`, and `use` for a plug), so the
//! Audit page undoes them like any library change. Removing a cartridge is a delete: it goes
//! through the delete gate (`forget::gate`, a preview and a confirmation token), takes every
//! version out of the catalogue and keeps the files, so undo puts it back.
use nmjson::Value;
use nmlearn::library::Library;
use nmlearn::stated::{self, Rule, Scope};
use std::path::Path;
use std::process::ExitCode;

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn text<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::string).ok().filter(|x| !x.trim().is_empty())
}
fn flag(v: &Value, k: &str, default: bool) -> bool {
    match v.get(k) {
        Ok(Value::Bool(b)) => *b,
        _ => default,
    }
}

/// What a person reads about how each kind of rule is held (the side panel and the docs).
pub const ENFORCEMENT: [(&str, &str); 4] = [
    ("enforced", "applied by the Nano to text it writes (Writer notes, summaries, outlines) and by preference_apply to a draft's prose; code, inline code, quoted and evidence lines are never changed"),
    ("checked", "reported by preference_apply with the line (avoided words in prose, and indentation, line length and naming in code); the text is left as it is"),
    ("biased", "a wanted term: a bounded push toward it in the Writer's decoding (strength 0.1 to 2.0, capped per token; an avoided word's mask always wins), listed to your agent as prefer: <term>, and noted by preference_apply where a draft lacks it (information, not a violation)"),
    ("guidance", "passed to your agent in every nm_context_pack reply whose project and cited files fit the rule's scope; a wanted style (concise, formal, numbered lists, ...) is listed as prefer: <style>"),
];
/// The forms the parser reads, as examples.
pub const FORMS: [&str; 11] = [
    "no em dashes / remove smart quotes / replace em dashes with ' - '",
    "use \"use\" instead of \"utilize\" / prefer colour over color",
    "want: prefer 'colour' / always use 'Result' / I like concise replies / more numbered lists",
    "want strength: strongly prefer (1.0), prefer (0.5), slightly prefer (0.25); editable per rule",
    "avoid: never say 'leverage' / avoid synergy",
    "indent with 4 spaces / use tabs for indentation",
    "max line length 100",
    "snake_case for functions / PascalCase for types",
    "for .kicad_pcb files, <any rule> (scope by file kind)",
    "under docs/, <any rule> (scope by folder)",
    "anything else is kept word for word as guidance",
];

fn in_text(v: &Value) -> Result<&str, String> {
    let t = text(v, "text").ok_or("text: give the preference in your own words")?;
    if t.chars().count() > stated::MAX_TEXT {
        return Err(format!("text: at most {} characters", stated::MAX_TEXT));
    }
    Ok(t)
}

fn rules_value(rules: &[Rule]) -> Value {
    Value::Array(rules.iter().map(Rule::view).collect())
}

/// `GET /api/preferences`: every cartridge holding stated rules (newest version, name, scope,
/// rules with what each does and how it is held, plug state with the use and stack that plug it,
/// whether it also holds learned values, its versions), the rules in force now, and how each kind
/// of rule is held.
#[must_use]
pub fn list(home: &Path) -> Value {
    let lib = match Library::open(home) {
        Ok(l) => l,
        Err(e) => return obj(vec![("error", s(&e.to_string()))]),
    };
    let carts: Vec<Value> = stated::stated_ids(&lib).iter().filter_map(|id| stated::view(&lib, id)).collect();
    let force = stated::InForce::from_library(&lib);
    obj(vec![
        ("cartridges", Value::Array(carts)),
        ("in_force", n(force.held.len() as u64)),
        ("enforcement", Value::Object(ENFORCEMENT.iter().map(|(k, d)| ((*k).to_owned(), s(d))).collect())),
        ("forms", Value::Array(FORMS.iter().map(|f| s(f)).collect())),
        ("applies", s("a plugged cartridge applies in a serving process within a second of the switch (the library is watched); trial cartridges are recorded, never applied")),
    ])
}

/// The cartridge a write goes to: its id, name and scope, the rules it holds now and its newest
/// version's key (none for a new cartridge).
struct Target {
    id: String,
    name: String,
    scope: Scope,
    rules: Vec<Rule>,
    prev: Option<String>,
}

/// The target of a `say`: the cartridge id (given, or made from the name), its name, its scope,
/// and the rules it holds now.
fn target(lib: &Library, v: &Value) -> Result<Target, String> {
    let name_given = text(v, "name").map(str::trim);
    if name_given.is_some_and(|nm| nm.chars().count() > 80 || nm.chars().any(char::is_control)) {
        return Err("name: at most 80 plain characters".into());
    }
    let id = match text(v, "cartridge") {
        Some(c) => c.trim().to_owned(),
        None => stated::id_from_name(name_given.unwrap_or("my preferences")),
    };
    let prev = lib.cartridge(&id).cloned();
    let scope = match text(v, "scope") {
        Some(t) => Scope::parse(t)?,
        None => prev.as_ref().and_then(|p| Scope::parse(&p.scope).ok()).unwrap_or_default(),
    };
    let name = name_given.map(str::to_owned).or_else(|| prev.as_ref().map(|p| p.name.clone())).unwrap_or_else(|| "My preferences".to_owned());
    let (rules, key) = match &prev {
        Some(p) => (stated::contents(lib, &p.key())?.0, Some(p.key())),
        None if text(v, "cartridge").is_some() => return Err(format!("no cartridge {id}; leave cartridge out to make a new one")),
        None => (Vec::new(), None),
    };
    Ok(Target { id, name, scope, rules, prev: key })
}

/// The reply of a preview: what would be written, nothing written.
fn preview(t: &Target, version: u64, new: &[Rule], kept: usize) -> Value {
    obj(vec![
        ("step", s("preview")),
        ("written", Value::Bool(false)),
        ("cartridge", obj(vec![("id", s(&t.id)), ("name", s(&t.name)), ("scope", s(&t.scope.to_text())), ("scope_words", s(&t.scope.words())), ("version", n(version)), ("adds_to", t.prev.as_deref().map_or(Value::Null, s))])),
        ("rules", rules_value(new)),
        ("kept", n(kept as u64)),
        ("next", s("Nothing has been written. Send the same request again with confirm: true to write this version.")),
    ])
}

/// Writes a version under the library lock, saves the catalogue and audits it.
fn commit(home: &Path, w: &stated::Write, action: &str, label: &str, extra: Vec<(&str, Value)>) -> Result<(String, stated::Written), String> {
    let (mut lib, _g) = crate::library::for_change(home)?;
    let out = stated::write(&mut lib, w)?;
    lib.save().map_err(|e| e.to_string())?;
    let mut fields = vec![("action", s(action))];
    fields.extend(out.fields());
    fields.extend(extra);
    fields.extend([("by", s(&w.by)), ("label", s(label))]);
    let audit = crate::controls::audit(home, "library", fields)?;
    Ok((audit, out))
}

fn written_reply(home: &Path, audit: &str, out: &stated::Written, new: &[Rule]) -> Value {
    let id = out.key.split('@').next().unwrap_or("");
    let lib = Library::open(home).ok();
    obj(vec![
        ("step", s("written")),
        ("written", Value::Bool(true)),
        ("audit_id", s(audit)),
        ("key", s(&out.key)),
        ("previous", out.previous.as_deref().map_or(Value::Null, s)),
        ("rules", rules_value(new)),
        ("cartridge", lib.as_ref().and_then(|l| stated::view(l, id)).unwrap_or(Value::Null)),
        ("undo", s("Audit page, or nm_settings op undo with this audit_id")),
    ])
}

/// `POST /api/preferences/say {text, name?, scope?, cartridge?, project?, plug?, trial?, confirm?}`:
/// turns the user's words into rules. Without `confirm` it answers a preview of each rule (what
/// it does, enforced, checked or guidance) and writes nothing; with `confirm: true` it writes them
/// as a new version of the cartridge (`cartridge`, or the one the name gives, made when new), plugs
/// it in unless `plug: false`, and audits it (`library` `preference-say`, undone by the Audit page).
/// # Errors
/// No text, a bad scope or name, nothing to parse, or a write failure.
pub fn say(home: &Path, v: &Value, by: &str) -> Result<Value, String> {
    let words = in_text(v)?;
    let lib = crate::library::open(home)?;
    let t = target(&lib, v)?;
    let project = text(v, "project");
    let new = stated::with_ids(&t.rules, stated::parse(words, &t.scope, project));
    if new.is_empty() {
        return Err("text: no statement found in it".into());
    }
    if !flag(v, "confirm", false) {
        return Ok(preview(&t, stated::next_version(&lib, &t.id), &new, t.rules.len()));
    }
    drop(lib);
    let Target { id, name, scope, rules: existing, .. } = t;
    let mut rules = existing;
    rules.extend(new.iter().cloned());
    let w = stated::Write { id: id.clone(), name, scope, rules, made_from: format!("the user's words, said through {by}"), by: by.to_owned(), trial: flag(v, "trial", false), plug: flag(v, "plug", true) };
    let label = format!("Stated preferences {id}: {} rule{} added", new.len(), if new.len() == 1 { "" } else { "s" });
    let (audit, out) = commit(home, &w, "preference-say", &label, vec![("said", s(words))])?;
    Ok(written_reply(home, &audit, &out, &new))
}

/// `POST /api/preferences/edit {id, add?, rule?, text?, strength?, drop?, name?, scope?, confirm?}`:
/// adds rules from new words (`add`), rewrites one rule from new words (`rule` with `text`; words
/// that give several rules replace it with all of them), sets a want's strength (`rule` with
/// `strength`, 0.1 to 2.0, a number or its text), drops rules (`drop`: rule ids), renames the
/// cartridge or changes its scope (rules that had the cartridge's scope take the new one). A
/// preview first, then `confirm: true` writes a new version (audited as `preference-edit`).
/// # Errors
/// An unknown cartridge or rule, no change, no rule left, or a write failure.
pub fn edit(home: &Path, v: &Value, by: &str) -> Result<Value, String> {
    let id = text(v, "id").ok_or("id: the cartridge's id as preference_list gives it")?.trim().to_owned();
    let lib = crate::library::open(home)?;
    let c = lib.cartridge(&id).cloned().ok_or_else(|| format!("no cartridge {id}"))?;
    let (old, _) = stated::contents(&lib, &c.key())?;
    let old_scope = Scope::parse(&c.scope).unwrap_or_default();
    let scope = match text(v, "scope") {
        Some(t) => Scope::parse(t)?,
        None => old_scope.clone(),
    };
    let name = text(v, "name").map_or_else(|| c.name.clone(), |x| x.trim().to_owned());
    let project = text(v, "project");
    let mut rules: Vec<Rule> = old.iter().cloned().map(|mut r| {
        if r.scope == old_scope {
            r.scope = scope.clone();
        }
        r
    }).collect();
    let drop_ids: Vec<String> = v.get("drop").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect();
    if let Some(bad) = drop_ids.iter().find(|d| !rules.iter().any(|r| r.id == **d)) {
        return Err(format!("drop: no rule {bad} in {id}"));
    }
    rules.retain(|r| !drop_ids.contains(&r.id));
    let mut changed_rules: Vec<Rule> = Vec::new();
    let strength = match v.get("strength") {
        Ok(Value::Null) | Err(_) => None,
        Ok(x) => Some(stated::strength_of(x).ok_or("strength: a number from 0.1 to 2.0")?),
    };
    if strength.is_some() && text(v, "rule").is_none() {
        return Err("strength: name the want with rule (r1, r2, ...)".into());
    }
    if let Some(rid) = text(v, "rule") {
        let at = rules.iter().position(|r| r.id == rid).ok_or_else(|| format!("rule: no rule {rid} in {id}"))?;
        if strength.is_none() || text(v, "text").is_some() {
            let words = in_text(v)?;
            let mut parsed = stated::parse(words, &scope, project);
            if parsed.is_empty() {
                return Err("text: no statement found in it".into());
            }
            let first = parsed.remove(0);
            let rest = stated::with_ids(&rules, parsed);
            if let Some(slot) = rules.get_mut(at) {
                *slot = Rule { id: rid.to_owned(), ..first };
            }
            changed_rules.extend(rest.iter().cloned());
            rules.extend(rest);
        }
        if let (Some(c), Some(slot)) = (strength, rules.get_mut(at)) {
            slot.set_strength(c)?;
        }
        if let Some(slot) = rules.get(at)
            && old.iter().all(|o| o != slot)
        {
            changed_rules.insert(0, slot.clone());
        }
    }
    if let Some(add) = text(v, "add") {
        if add.chars().count() > stated::MAX_TEXT {
            return Err(format!("add: at most {} characters", stated::MAX_TEXT));
        }
        let more = stated::with_ids(&rules, stated::parse(add, &scope, project));
        changed_rules.extend(more.iter().cloned());
        rules.extend(more);
    }
    if rules == old && name == c.name && scope == old_scope {
        return Ok(obj(vec![("step", s("unchanged")), ("written", Value::Bool(false)), ("key", s(&c.key()))]));
    }
    if rules.is_empty() {
        return Err(format!("no rule would be left in {id}; remove the cartridge instead (preference_remove)"));
    }
    if !flag(v, "confirm", false) {
        let t = Target { id: id.clone(), name: name.clone(), scope: scope.clone(), rules: Vec::new(), prev: Some(c.key()) };
        let mut p = preview(&t, stated::next_version(&lib, &id), &changed_rules, rules.len());
        if let Value::Object(m) = &mut p {
            m.insert("dropped".into(), Value::Array(drop_ids.iter().map(|d| s(d)).collect()));
            m.insert("all_rules".into(), rules_value(&rules));
        }
        return Ok(p);
    }
    drop(lib);
    let w = stated::Write { id: id.clone(), name, scope, rules, made_from: format!("{} edited through {by}", c.key()), by: by.to_owned(), trial: c.trial, plug: false };
    let label = match (strength, text(v, "rule")) {
        (Some(c), Some(rid)) => format!("Stated preferences {id}: want {rid} strength set to {}", stated::strength_text(c)),
        _ => format!("Stated preferences {id} edited ({} changed, {} dropped)", changed_rules.len(), drop_ids.len()),
    };
    let mut extra = vec![("dropped", Value::Array(drop_ids.iter().map(|d| s(d)).collect()))];
    if let (Some(c), Some(rid)) = (strength, text(v, "rule")) {
        extra.push(("strength", obj(vec![("rule", s(rid)), ("to", Value::Number(stated::strength_text(c)))])));
    }
    let (audit, out) = commit(home, &w, "preference-edit", &label, extra)?;
    Ok(written_reply(home, &audit, &out, &changed_rules))
}

/// The preview of removing cartridge `id` for the delete gate: what goes, and the word to type.
/// # Errors
/// As `nmlearn::stated::removal`.
pub fn remove_preview(home: &Path, id: &str) -> Result<(Value, String), String> {
    let lib = Library::open(home).map_err(|e| e.to_string())?;
    let r = stated::removal(&lib, id)?;
    let mut items: Vec<Value> = r.cartridges.iter().map(|c| obj(vec![("label", s(&format!("{} ({})", c.key(), c.name))), ("path", Value::Null), ("bytes", n(c.bytes())), ("files", n(c.files.len() as u64))])).collect();
    items.extend(r.stacks.iter().map(|st| obj(vec![("label", s(&format!("stack {} for {}", st.key(), st.use_for))), ("path", Value::Null), ("bytes", Value::Null), ("files", Value::Null)])));
    let bytes: u64 = r.cartridges.iter().map(nmlearn::library::Cartridge::bytes).sum();
    let preview = obj(vec![
        ("title", s(&format!("Remove the preference cartridge {id}"))),
        ("items", Value::Array(items)),
        ("bytes", n(bytes)),
        ("files", n(r.files.len() as u64)),
        ("undo", s("can be undone from the Audit page: the versions leave the library, their files stay on the drive")),
        ("note", s(if r.active.is_empty() { "it is not plugged in" } else { "it is plugged in now; removing it unplugs it" })),
    ]);
    Ok((preview, id.to_owned()))
}

/// `POST /api/preferences/remove {id}` (after the delete gate): takes every version of the
/// cartridge, the stacks that hold only it and the uses switched to them out of the library; the
/// files stay. Audited as `preference-remove`, which undo puts back.
/// # Errors
/// As `nmlearn::stated::removal`, or a write failure.
pub fn remove(home: &Path, id: &str, by: &str) -> Result<Value, String> {
    let (mut lib, _g) = crate::library::for_change(home)?;
    let r = stated::removal(&lib, id)?;
    stated::remove(&mut lib, &r);
    lib.save().map_err(|e| e.to_string())?;
    let audit = crate::controls::audit(
        home,
        "library",
        vec![("action", s("preference-remove")), ("cartridge", s(id)), ("removed", r.to_value()), ("by", s(by)), ("label", s(&format!("Preference cartridge {id} removed ({} versions; files kept)", r.cartridges.len())))],
    )?;
    Ok(obj(vec![("step", s("removed")), ("deleted", Value::Bool(true)), ("audit_id", s(&audit)), ("id", s(id)), ("versions", n(r.cartridges.len() as u64)), ("files_kept", Value::Array(r.files.iter().map(|f| s(f)).collect()))]))
}

/// `POST /api/preferences/plug {id, on}`: plugs the newest version of the cartridge in or out by
/// switching its stack's use (`library::switch_use`, the decks' plug): audited as a `library`
/// `use` row, undone like any switch.
/// # Errors
/// An unknown cartridge, one with no stack, or the switch failing.
pub fn plug(home: &Path, id: &str, on: bool, by: &str) -> Result<Value, String> {
    let lib = crate::library::open(home)?;
    let (use_for, stack) = stated::plug_target(&lib, id).ok_or_else(|| format!("{id}: no stated preference cartridge with a stack (see preference_list)"))?;
    drop(lib);
    let audit = crate::library::switch_use(home, &use_for, on.then_some(stack.as_str()), by)?;
    let lib = Library::open(home).map_err(|e| e.to_string())?;
    Ok(obj(vec![("audit_id", audit.map_or(Value::Null, |a| s(&a))), ("id", s(id)), ("use", s(&use_for)), ("stack", s(&stack)), ("plugged", Value::Bool(stated::plugged(&lib, id)))]))
}

/// `POST /api/preferences/apply {text, path?, project?}`: the rules in force applied to a draft
/// (`nmlearn::stated::apply`): the transformed text, the places changed, each violation with its
/// line, and the guidance in scope. Writes nothing.
/// # Errors
/// No text, or an unreadable library.
pub fn apply(home: &Path, v: &Value) -> Result<Value, String> {
    let t = v.get("text").and_then(Value::string).map_err(|_| "text: give the draft".to_owned())?;
    if t.len() > 2 * 1024 * 1024 {
        return Err("text: at most 2 MB".into());
    }
    let lib = Library::open(home).map_err(|e| e.to_string())?;
    let force = stated::InForce::from_library(&lib);
    let out = stated::apply(t, &force, text(v, "project"), text(v, "path"));
    let mut r = out.to_value();
    if let Value::Object(m) = &mut r {
        m.insert("in_force".into(), n(force.held.len() as u64));
        m.insert("in_scope".into(), n(out.rules.len() as u64));
    }
    Ok(r)
}

/// The undo of a `preference-say`, `preference-edit` or `preference-remove` row (called by
/// `library::undo`); audited as `preference-undone`.
/// # Errors
/// The library cannot be written.
pub fn undo(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let (mut lib, _g) = crate::library::for_change(home)?;
    let action = text(row, "action").unwrap_or("");
    let done = if action == "preference-remove" { stated::restore(&mut lib, row.get("removed").unwrap_or(&Value::Null))? } else { stated::unwrite(&mut lib, row) };
    if done.is_empty() {
        return Ok(None);
    }
    lib.save().map_err(|e| e.to_string())?;
    let label = format!("Undone: {action} of {} ({})", text(row, "cartridge").unwrap_or(""), done.join("; "));
    crate::controls::audit(home, "library", vec![("action", s("preference-undone")), ("cartridge", s(text(row, "cartridge").unwrap_or(""))), ("undo_of", s(id)), ("by", s(by)), ("label", s(&label))]).map(Some)
}

// ---------------------------------------------------------------------------------------------
// The command: the same routes as the interface and MCP.

const USAGE: &str = "usage: neuromorph-nano preferences <list|say|edit|remove|plug|unplug|undo|apply> ...
  list                                   the stated preference cartridges and their rules
  say <words> [--name N] [--scope S] [--cartridge ID] [--no-plug] [--trial] [--yes]
                                         parse your words into rules; without --yes, a preview
  edit <id> [--add WORDS] [--rule rN --text WORDS] [--rule rN --strength 0.1..2.0] [--drop r1,r2] [--name N] [--scope S] [--yes]
  remove <id> [--confirm <id>]           a preview; with --confirm and the id typed again, removes
  plug <id> | unplug <id>                switch the cartridge's stack on or off
  undo <audit id>                        undo a say, edit, remove, plug or unplug (as the Audit page)
  apply [--path P] [--project K] [--text T]   the rules in force over a draft (stdin without --text)
  scope: global, project:<key>, kinds:<ext>,<ext>, folders:<path>,<path> (joined by ;)";

fn arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w.first().map(String::as_str) == Some(flag)).and_then(|w| w.get(1)).map(String::as_str)
}

fn body(pairs: Vec<(&str, Value)>) -> Vec<u8> {
    obj(pairs).encode().unwrap_or_default().into_bytes()
}

/// `preferences remove`: the delete gate's two steps. Without `confirm` it prints the preview and
/// removes nothing; with the cartridge id typed again as `confirm` (the preview's confirm word), the
/// preview's token is taken and the same request is sent with it, in this one process (a token
/// lives in the process that made it).
fn remove_command(home: &Path, id: &str, confirm: Option<&str>) -> Result<Value, String> {
    let first = crate::ui::lent_route(home, "/api/preferences/remove", "", Some(&body(vec![("id", s(id))])), "cli")?;
    let Some(word) = confirm else { return Ok(first) };
    if word != id {
        return Err(format!("--confirm: type the cartridge id ({id}) to remove it; nothing was removed"));
    }
    let token = first.get("confirm_token").and_then(Value::string).map_err(|_| "the preview gave no confirmation token; nothing was removed".to_owned())?;
    crate::ui::lent_route(home, "/api/preferences/remove", "", Some(&body(vec![("id", s(id)), ("confirm_token", s(token))])), "cli")
}

/// `neuromorph-nano preferences ...`: runs the interface's own routes (`ui::lent_route`), so the
/// command, MCP and the interface do the same thing.
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    let rest = args.get(1..).unwrap_or_default();
    let sub = rest.first().map(String::as_str);
    let pos = rest.get(1).filter(|x| !x.starts_with("--")).map(String::as_str);
    let opt = |k: &'static str, f: &str, pairs: &mut Vec<(&'static str, Value)>| {
        if let Some(x) = arg(rest, f) {
            pairs.push((k, s(x)));
        }
    };
    let call = match (sub, pos) {
        (Some("list"), _) | (None, _) => crate::ui::lent_route(home, "/api/preferences", "", None, "cli"),
        (Some("say"), Some(words)) => {
            let mut p = vec![("text", s(words)), ("plug", Value::Bool(!rest.iter().any(|a| a == "--no-plug"))), ("trial", Value::Bool(rest.iter().any(|a| a == "--trial"))), ("confirm", Value::Bool(rest.iter().any(|a| a == "--yes")))];
            opt("name", "--name", &mut p);
            opt("scope", "--scope", &mut p);
            opt("cartridge", "--cartridge", &mut p);
            opt("project", "--project", &mut p);
            crate::ui::lent_route(home, "/api/preferences/say", "", Some(&body(p)), "cli")
        }
        (Some("edit"), Some(id)) => {
            let mut p = vec![("id", s(id)), ("confirm", Value::Bool(rest.iter().any(|a| a == "--yes")))];
            for (k, f) in [("add", "--add"), ("rule", "--rule"), ("text", "--text"), ("strength", "--strength"), ("name", "--name"), ("scope", "--scope")] {
                opt(k, f, &mut p);
            }
            if let Some(d) = arg(rest, "--drop") {
                p.push(("drop", Value::Array(d.split(',').map(str::trim).filter(|x| !x.is_empty()).map(s).collect())));
            }
            crate::ui::lent_route(home, "/api/preferences/edit", "", Some(&body(p)), "cli")
        }
        (Some("remove"), Some(id)) => remove_command(home, id, arg(rest, "--confirm")),
        (Some(w @ ("plug" | "unplug")), Some(id)) => crate::ui::lent_route(home, "/api/preferences/plug", "", Some(&body(vec![("id", s(id)), ("on", Value::Bool(w == "plug"))])), "cli"),
        (Some("undo"), Some(aid)) => crate::ui::lent_route(home, "/api/undo", "", Some(&body(vec![("id", s(aid))])), "cli"),
        (Some("apply"), _) => {
            let draft = match arg(rest, "--text") {
                Some(t) => t.to_owned(),
                None => {
                    let mut t = String::new();
                    let _ = std::io::Read::read_to_string(&mut std::io::stdin(), &mut t);
                    t
                }
            };
            let mut p = vec![("text", s(&draft))];
            opt("path", "--path", &mut p);
            opt("project", "--project", &mut p);
            crate::ui::lent_route(home, "/api/preferences/apply", "", Some(&body(p)), "cli")
        }
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    match call {
        Ok(v) => {
            println!("{}", nmjson::pretty::pretty(&v));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("preferences: {e}");
            ExitCode::FAILURE
        }
    }
}
