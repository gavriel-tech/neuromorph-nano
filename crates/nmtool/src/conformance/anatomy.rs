//! Detectors over the anatomy: C-BLOCK (bound assemblies that are the released layer order),
//! C-CORE (a neural core per tentacle), C-DRONE (drone sets from data), C-ONE-TENTACLE (one
//! tentacle system) and C-CROSS (no tentacle-to-tentacle path). Each reads the repository
//! under `root` and returns its violations keyed for the rule's stand-ins.
use super::Finding;
use super::text::{crate_sources, has_word, is_comment, list, read, rel, s, served_assemblies, so, strs, walk};
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::Path;

fn parse_file(p: &Path) -> Result<Value, String> {
    let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    nmjson::parse(&bytes).map_err(|e| format!("{}: {e}", p.display()))
}

/// The notes along an assembly's base chain: (root assembly name, every note joined).
fn chain_notes(dir: &Path, name: &str) -> Result<(String, String), String> {
    let mut n = name.to_owned();
    let mut notes = String::new();
    for _ in 0..8 {
        let v = parse_file(&dir.join(format!("{n}.json")))?;
        notes.push_str(v.get("note").and_then(Value::string).unwrap_or(""));
        notes.push('\n');
        match v.get("base").and_then(Value::string) {
            Ok(b) => n = b.to_owned(),
            Err(_) => return Ok((n, notes)),
        }
    }
    Err(format!("{name}: base chain deeper than 8"))
}

/// C-BLOCK.
pub(super) fn block(root: &Path, p: &Value) -> Result<Finding, String> {
    let dir = root.join(s(p, "assembly_dir")?);
    let marks = strs(p, "conventional_marks");
    let mut f = Finding::default();
    let bound = served_assemblies(root, &dir, &strs(p, "served_paths"));
    if bound.is_empty() {
        f.notes.push("the served path names no engine assembly".into());
    }
    for (name, file) in &bound {
        let (base, notes) = chain_notes(&dir, name)?;
        let conv = marks.iter().any(|m| notes.contains(m.as_str()));
        let over = if base == *name { String::new() } else { format!(" (addition to {base})") };
        f.notes.push(format!("{name}{over} bound in {file}{}", if conv { ": released layer order" } else { "" }));
        if conv {
            f.add(name, format!("the served path binds {name}{over}, the released layer order written by the reference generator"));
        }
    }
    Ok(f)
}

/// C-CORE.
///
/// 2026-11-10 (refinement step 1; directions of Luis Jake Gabriel III of 08:00 and 08:07 UTC: the
/// weights are not placed in the tentacles; each tentacle keeps its own core and drone tip; one
/// copy of the weights is held centrally and used by reference). The rule passes when every
/// tentacle's core and drone tip exist as functions on the served path (the core type implements
/// the `g0_tx` boundary and is built once per window; the tip's attention-head read is called),
/// the cores hold the weights only by a shared reference, no served file binds the set-aside
/// per-tentacle assemblies, and those set-aside modules carry their mark. Until this date the rule
/// required one bound assembly per tentacle, which put a body of weights in each tentacle.
pub(super) fn core(root: &Path, p: &Value) -> Result<Finding, String> {
    let mut f = Finding::default();
    let core_file = root.join(s(p, "core_file")?);
    let text = read(&core_file);
    if text.is_empty() {
        f.add("tentacle-without-core", format!("{} not present", rel(root, &core_file)));
        return Ok(f);
    }
    let code: String = text.lines().filter(|l| !is_comment(l)).collect::<Vec<_>>().join(
        "
",
    );
    let mut served: Vec<std::path::PathBuf> = Vec::new();
    for d in strs(p, "served_paths") {
        walk(&root.join(d), &["rs".to_owned()], &mut served);
    }
    let set_aside = strs(p, "set_aside");
    let in_set_aside = |file: &Path| set_aside.iter().any(|x| rel(root, file) == *x);
    let code_of = |file: &Path| {
        read(file).lines().filter(|l| !is_comment(l)).collect::<Vec<_>>().join(
            "
",
        )
    };
    let core_impl = s(p, "core_impl")?;
    let per_window = s(p, "per_window")?;
    let callers: Vec<String> = served
        .iter()
        .filter(|file| !in_set_aside(file) && code_of(file).contains(per_window.as_str()))
        .map(|file| rel(root, file))
        .collect();
    if !code.contains(core_impl.as_str()) || callers.is_empty() {
        f.add(
            "tentacle-without-core",
            format!(
                "the core type does not implement the g0_tx boundary ({core_impl}) or no served file builds it per window ({per_window})"
            ),
        );
    }
    let tip = s(p, "tip_call")?;
    let tips: Vec<String> =
        served.iter().filter(|file| !in_set_aside(file) && code_of(file).contains(tip.as_str())).map(|file| rel(root, file)).collect();
    if tips.is_empty() {
        f.add("tip-not-a-function", format!("no served file calls the drone tip read {tip}"));
    }
    let shared = s(p, "shared_weights")?;
    if !code.contains(shared.as_str()) {
        f.add("weights-not-shared", format!("{} does not hold the weights by a shared reference ({shared})", rel(root, &core_file)));
    }
    for call in strs(p, "set_aside_calls") {
        for file in served.iter().filter(|file| !in_set_aside(file)) {
            if code_of(file).contains(call.as_str()) {
                f.add("weights-not-shared", format!("{} binds a per-tentacle assembly ({call})", rel(root, file)));
            }
        }
    }
    let mark = s(p, "set_aside_mark")?;
    for x in &set_aside {
        let t = read(&root.join(x));
        if !t.is_empty() && !t.lines().take(6).any(|l| l.contains(mark.as_str())) {
            f.add(&format!("set-aside-not-marked:{x}"), format!("{x} is kept but its module comment does not say {mark}"));
        }
    }
    let tentacles = p.get("tentacles").and_then(Value::integer).map_or(8, |n| usize::try_from(n).unwrap_or(8));
    f.notes.push(format!("{tentacles} tentacles: core {core_impl} built per window by {}; drone tip read called in {}; weights by shared reference ({shared}); set aside and unused: {}", callers.join(", "), tips.join(", "), set_aside.join(", ")));
    Ok(f)
}

/// C-DRONE.
pub(super) fn drone(root: &Path, p: &Value) -> Result<Finding, String> {
    let call = s(p, "reader_call")?;
    let prefix = s(p, "constant_prefix")?;
    let mut f = Finding::default();
    for file in strs(p, "served_files") {
        let text = read(&root.join(&file));
        if text.is_empty() {
            f.notes.push(format!("{file}: not present"));
            continue;
        }
        if !text.contains(call.as_str()) {
            f.add(&format!("no-reader:{file}"), format!("{file} never calls {call}"));
        }
        let lines: Vec<&str> = text.lines().collect();
        let end = lines.iter().position(|l| l.contains("fn check(")).unwrap_or(lines.len());
        for (i, line) in lines.iter().enumerate().take(end) {
            if is_comment(line) || !line.contains(prefix.as_str()) {
                continue;
            }
            let ctx = lines[i.saturating_sub(4)..=i].join("\n");
            if !(ctx.contains("unwrap_or") || ctx.contains("fallback")) {
                f.add(
                    &format!("constant:{file}:{}", i + 1),
                    format!("{file}:{} uses a {prefix} constant outside a stated fallback", i + 1),
                );
            }
        }
    }
    let d = parse_file(&root.join(s(p, "design")?))?;
    let pops = d.get("populations").and_then(Value::object).map_err(|e| format!("drones design: {e}"))?;
    for (pop, v) in pops {
        let n = list(v, "default").len();
        if n == 0 {
            f.add(&format!("design:{pop}"), format!("{pop} has no default drone set"));
        }
        f.notes.push(format!(
            "{pop}: {n} heads by default, {} tentacle sets",
            v.get("tentacles").and_then(Value::object).map_or(0, std::collections::BTreeMap::len)
        ));
    }
    Ok(f)
}

/// C-ONE-TENTACLE.
pub(super) fn one_tentacle(root: &Path, p: &Value) -> Result<Finding, String> {
    let mark = s(p, "name_mark")?;
    let mut systems = Vec::new();
    let mut listed = Vec::new();
    for sys in list(p, "systems") {
        let file = s(sys, "file")?;
        // A system named by two files (2026-11-10: the windowed pass and its tentacle core) is
        // counted once.
        let name = s(sys, "name")?;
        if read(&root.join(&file)).contains(s(sys, "mark")?.as_str()) && !systems.contains(&name) {
            systems.push(name);
        }
        listed.push(file);
    }
    // 2026-11-10 (refinement step 1): a module whose head carries the set-aside mark is compiled
    // and kept but used by no path (the per-tentacle assemblies, set aside on the direction of
    // Luis Jake Gabriel III that the weights are not placed in the tentacles); it is reported,
    // not counted as a tentacle system.
    let aside_mark = so(p, "set_aside_mark");
    let mut aside = Vec::new();
    for src in crate_sources(root) {
        let r = rel(root, &src);
        let name = src.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        if name.contains(mark.as_str()) && !listed.contains(&r) {
            if !aside_mark.is_empty() && read(&src).lines().take(6).any(|l| l.contains(aside_mark.as_str())) {
                aside.push(r);
                continue;
            }
            systems.push(r);
        }
    }
    let mut f = Finding::default();
    if !aside.is_empty() {
        f.notes.push(format!("set aside, not counted: {}", aside.join(", ")));
    }
    f.notes.push(format!("tentacle systems: {}", if systems.is_empty() { "none".into() } else { systems.join(", ") }));
    if systems.len() > 1 {
        f.add(&systems.join("+"), format!("{} tentacle systems in source: {}", systems.len(), systems.join(", ")));
    } else if systems.is_empty() {
        f.add("none", "no tentacle system found in source");
    }
    Ok(f)
}

/// The unit (node of kind tentacle) a node belongs to, and whether it sits in the Core Brain,
/// by its parent chain inside one assembly.
fn place(kinds: &BTreeMap<String, (String, String)>, id: &str) -> (Option<String>, bool) {
    let (mut unit, mut brain, mut cur) = (None, false, id.to_owned());
    for _ in 0..32 {
        let Some((kind, parent)) = kinds.get(&cur) else { break };
        if kind == "tentacle" && unit.is_none() {
            unit = Some(cur.clone());
        }
        brain |= kind == "brain";
        if parent.is_empty() {
            break;
        }
        cur.clone_from(parent);
    }
    (unit, brain)
}

/// Assembly edges that bypass the Core Brain (the same relation as
/// `nmnte::topology::bypasses_brain`, read from the files). Edges to nodes a file does not
/// define (templates) are skipped.
fn assembly_edges(root: &Path, dir: &Path, f: &mut Finding) -> Result<usize, String> {
    let mut files = Vec::new();
    walk(dir, &["json".to_owned()], &mut files);
    let mut edges = 0;
    for file in files {
        let v = parse_file(&file)?;
        let kinds: BTreeMap<String, (String, String)> =
            list(&v, "nodes").iter().map(|n| (so(n, "id"), (so(n, "kind"), so(n, "parent")))).collect();
        if !kinds.values().any(|(k, _)| k == "tentacle") {
            continue;
        }
        for e in list(&v, "edges") {
            let (a, b) = (so(e, "from"), so(e, "to"));
            if !kinds.contains_key(&a) || !kinds.contains_key(&b) {
                continue;
            }
            edges += 1;
            let ((ua, ba), (ub, bb)) = (place(&kinds, &a), place(&kinds, &b));
            let bypass = match (&ua, &ub) {
                (None, None) => false,
                (Some(x), Some(y)) => x != y,
                (Some(_), None) => !bb,
                (None, Some(_)) => !ba,
            };
            if bypass {
                let r = rel(root, &file);
                f.add(&format!("assembly:{r}:{a}->{b}"), format!("{r}: edge {a} -> {b} passes between tentacles or around the Core Brain"));
            }
        }
    }
    Ok(edges)
}

/// Parameter types of a signature text between its first parenthesis and the matching one.
fn param_types(sig: &str) -> Vec<String> {
    let Some(open) = sig.find('(') else { return Vec::new() };
    let (mut depth, mut cur, mut out) = (0i32, String::new(), Vec::new());
    for c in sig[open + 1..].chars() {
        match c {
            '(' | '<' | '[' => depth += 1,
            ')' if depth == 0 => break,
            ')' | '>' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    out.push(cur);
    out.into_iter().map(|p| p.trim().to_owned()).filter(|p| !p.is_empty()).collect()
}

/// Struct fields and function parameters that hand one unit or window to another.
fn source_scan(root: &Path, p: &Value, f: &mut Finding) -> Result<usize, String> {
    let coord = strs(p, "coordinating");
    let mut scanned = 0;
    for ut in list(p, "unit_types") {
        let (ty, defined, module) = (s(ut, "type")?, s(ut, "defined")?, s(ut, "module")?);
        let home = defined.rsplit_once("/src/").map(|x| format!("{}/", x.0)).unwrap_or_default();
        let krate = home.trim_end_matches('/').rsplit('/').next().unwrap_or_default().to_owned();
        for src in crate_sources(root) {
            let r = rel(root, &src);
            if coord.iter().any(|c| r.starts_with(c.as_str())) {
                continue;
            }
            let text = read(&src);
            let imports = text.lines().any(|l| {
                l.trim_start().starts_with("use ") && l.contains(krate.as_str()) && l.contains(module.as_str()) && has_word(l, &ty)
            });
            if !(r.starts_with(&home) || imports) {
                continue;
            }
            scanned += 1;
            scan_text(&r, &text, &ty, f);
        }
    }
    Ok(scanned)
}

fn scan_text(r: &str, text: &str, ty: &str, f: &mut Finding) {
    let lines: Vec<&str> = text.lines().collect();
    let (mut in_struct, mut in_impl) = (false, false);
    for (i, line) in lines.iter().enumerate() {
        if is_comment(line) {
            continue;
        }
        let t = line.trim();
        if !line.starts_with(' ') {
            if t.starts_with("impl") {
                in_impl = has_word(t.split(" for ").last().unwrap_or(t), ty);
            } else if t == "}" {
                in_impl = false;
            }
        }
        if t.contains("struct ")
            && has_word(t.split('{').next().unwrap_or(""), ty)
            && t.ends_with('{')
            && t.split_whitespace().any(|w| w == ty)
        {
            in_struct = true;
            continue;
        }
        if in_struct {
            if t.starts_with('}') {
                in_struct = false;
            } else if let Some((field, kind)) = t.split_once(':')
                && (has_word(kind, "Unit") || has_word(kind, "Window"))
            {
                f.add(
                    &format!("source:{r}:{ty}.{}", field.trim().trim_start_matches("pub ")),
                    format!("{r}:{} {ty} holds a handle to another unit or window", i + 1),
                );
            }
            continue;
        }
        if let Some(at) = t.find("fn ") {
            let sig: String = lines[i..lines.len().min(i + 8)].join(" ");
            let sig = &sig[sig.find("fn ").unwrap_or(at)..];
            let params = param_types(sig.split('{').next().unwrap_or(sig));
            let selfish = in_impl && params.first().is_some_and(|p| p.contains("self"));
            let typed = params.iter().filter(|p| p.split_once(':').is_some_and(|(_, k)| has_word(k, ty))).count();
            if typed + usize::from(selfish) >= 2 {
                let name: String = sig[3..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
                f.add(&format!("source:{r}:fn {name}"), format!("{r}:{} fn {name} takes two values of {ty}", i + 1));
            }
        }
    }
}

/// C-CROSS.
pub(super) fn cross(root: &Path, p: &Value) -> Result<Finding, String> {
    let mut f = Finding::default();
    for g in list(p, "gates") {
        let (name, file, mark) = (s(g, "name")?, s(g, "file")?, s(g, "mark")?);
        if read(&root.join(&file)).contains(mark.as_str()) {
            f.notes.push(format!("gate present: {name} ({file})"));
        } else {
            f.add(&format!("gate-missing:{name}"), format!("the existing gate {name} is no longer in {file}"));
        }
    }
    let edges = assembly_edges(root, &root.join(s(p, "assembly_dir")?), &mut f)?;
    let files = source_scan(root, p, &mut f)?;
    f.notes.push(format!("{edges} assembly edges and {files} source files searched"));
    Ok(f)
}
