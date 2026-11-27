//! Imports and packages outside Cargo. Each import statement the tagger found (`tags::Import`) is
//! resolved to the file of the tree it names, or recorded as an outside package:
//! - JavaScript and TypeScript: relative paths with the extension list and `index` files, the
//!   `paths` and `baseUrl` of the nearest `tsconfig.json` (one `extends` level followed), and the
//!   packages of the tree by their `package.json` name (npm workspaces).
//! - Python: relative imports from the importing package; absolute imports from the importing
//!   file's own folder, the tree's root, `src/`, and the folders named in `pyproject.toml`.
//! - C, C++, CUDA: `#include "x"` from the including folder, then from every `include` folder,
//!   then any file of the tree whose path ends with the name; `<x>` only when the tree holds it.
//!
//! Packages: every `package.json` with a name (npm) and every `pyproject.toml` with a project name
//! (Python), each with its dependencies on the other packages of the tree.
use crate::repo::{Crate, FileInfo};
use crate::tags::Import;
use std::collections::{BTreeMap, BTreeSet};

/// One import statement resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    pub from: String,
    pub line: usize,
    pub spec: String,
    /// The file of the tree it names, when found.
    pub to: Option<String>,
    /// The outside package it names (npm or Python package name, or the header), when not found.
    pub external: Option<String>,
}

/// Normalise a `/` path: drop `.` parts and apply `..`.
fn norm(p: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for part in p.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            x => out.push(x),
        }
    }
    out.join("/")
}

fn dir_of(p: &str) -> &str {
    p.rsplit_once('/').map_or("", |(d, _)| d)
}

fn join(dir: &str, rest: &str) -> String {
    if dir.is_empty() { norm(rest) } else { norm(&format!("{dir}/{rest}")) }
}

/// JSON with comments and trailing commas (as `tsconfig.json` allows) made strict.
#[must_use]
pub fn strict_json(text: &str) -> String {
    let c: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while let Some(&ch) = c.get(i) {
        if ch == '"' {
            let start = i;
            i += 1;
            while let Some(&x) = c.get(i) {
                i += if x == '\\' { 2 } else { 1 };
                if x == '"' {
                    break;
                }
            }
            out.extend(c.get(start..i.min(c.len())).unwrap_or(&[]));
        } else if ch == '/' && c.get(i + 1) == Some(&'/') {
            while c.get(i).is_some_and(|x| *x != '\n') {
                i += 1;
            }
        } else if ch == '/' && c.get(i + 1) == Some(&'*') {
            i += 2;
            while i < c.len() && !(c.get(i) == Some(&'*') && c.get(i + 1) == Some(&'/')) {
                i += 1;
            }
            i += 2;
        } else if ch == ',' {
            let mut j = i + 1;
            while c.get(j).is_some_and(|x| x.is_whitespace()) {
                j += 1;
            }
            if !matches!(c.get(j), Some('}' | ']')) {
                out.push(',');
            }
            i += 1;
        } else {
            out.push(ch);
            i += 1;
        }
    }
    out
}

/// What one `tsconfig.json` says about module names: (base folder, patterns and their targets).
#[derive(Clone, Debug, Default)]
struct TsPaths {
    base: Option<String>,
    paths: Vec<(String, Vec<String>)>,
}

/// The tree's files by path, and what the resolvers read once.
struct Tree<'a> {
    files: BTreeSet<&'a str>,
    text: BTreeMap<&'a str, &'a str>,
    ts: BTreeMap<String, TsPaths>,
    npm: Vec<(String, String)>,
    py_roots: Vec<String>,
}

const TS_EXTS: [&str; 8] = ["ts", "tsx", "d.ts", "js", "jsx", "mjs", "cjs", "json"];

impl<'a> Tree<'a> {
    fn new(files: &'a [FileInfo]) -> Self {
        let mut t = Self {
            files: files.iter().map(|f| f.path.as_str()).collect(),
            text: files.iter().map(|f| (f.path.as_str(), f.text.as_str())).collect(),
            ts: BTreeMap::new(),
            npm: Vec::new(),
            py_roots: Vec::new(),
        };
        for f in files {
            let name = f.path.rsplit('/').next().unwrap_or(&f.path);
            if name == "tsconfig.json" {
                let p = t.tsconfig(&f.path, 0);
                t.ts.insert(dir_of(&f.path).to_owned(), p);
            }
        }
        t.npm = files
            .iter()
            .filter(|f| f.path.rsplit('/').next() == Some("package.json"))
            .filter_map(|f| {
                let v = nmjson::parse(f.text.as_bytes()).ok()?;
                Some((v.get("name").ok()?.string().ok()?.to_owned(), dir_of(&f.path).to_owned()))
            })
            .collect();
        t.npm.sort_by_key(|p| std::cmp::Reverse(p.0.len()));
        t.py_roots = python_roots(files);
        t
    }
    fn tsconfig(&self, path: &str, depth: usize) -> TsPaths {
        let Some(text) = self.text.get(path) else { return TsPaths::default() };
        let Ok(v) = nmjson::parse(strict_json(text).as_bytes()) else { return TsPaths::default() };
        let dir = dir_of(path);
        let mut out = match v.get("extends").and_then(nmjson::Value::string) {
            Ok(e) if depth < 2 && e.starts_with('.') => {
                let target = join(dir, e);
                let target = if target.ends_with(".json") { target } else { format!("{target}.json") };
                self.tsconfig(&target, depth + 1)
            }
            _ => TsPaths::default(),
        };
        if let Ok(opts) = v.get("compilerOptions") {
            if let Ok(b) = opts.get("baseUrl").and_then(nmjson::Value::string) {
                out.base = Some(join(dir, b));
            }
            if let Ok(paths) = opts.get("paths").and_then(nmjson::Value::object) {
                let base = out.base.clone().unwrap_or_else(|| dir.to_owned());
                out.paths = paths
                    .iter()
                    .map(|(k, v)| {
                        let targets = v
                            .array()
                            .map(|a| a.iter().filter_map(|x| x.string().ok()).map(|x| join(&base, x)).collect())
                            .unwrap_or_default();
                        (k.clone(), targets)
                    })
                    .collect();
                out.paths.sort_by_key(|p| std::cmp::Reverse(p.0.len()));
            }
        }
        out
    }
    /// A path tried as a module file: itself, with each extension, then as a folder's `index`.
    fn module_file(&self, base: &str) -> Option<String> {
        if self.files.contains(base) {
            return Some(base.to_owned());
        }
        let stem = [".js", ".jsx", ".mjs", ".cjs"].iter().find_map(|e| base.strip_suffix(e)).unwrap_or(base);
        for e in TS_EXTS {
            let p = format!("{stem}.{e}");
            if self.files.contains(p.as_str()) {
                return Some(p);
            }
        }
        for e in TS_EXTS {
            let p = format!("{base}/index.{e}");
            if self.files.contains(p.as_str()) {
                return Some(p);
            }
        }
        None
    }
    /// A package folder's entry file: `types`, `module`, `main` or `exports` of its
    /// `package.json`, else `index` or `src/index`.
    fn package_entry(&self, dir: &str) -> Option<String> {
        let pj = join(dir, "package.json");
        if let Some(v) = self.text.get(pj.as_str()).and_then(|t| nmjson::parse(t.as_bytes()).ok()) {
            let mut fields: Vec<String> =
                ["types", "module", "main"].iter().filter_map(|k| v.get(k).ok()?.string().ok().map(str::to_owned)).collect();
            if let Ok(e) = v.get("exports") {
                if let Ok(s) = e.string() {
                    fields.insert(0, s.to_owned());
                } else if let Ok(dot) = e.get(".") {
                    if let Ok(s) = dot.string() {
                        fields.insert(0, s.to_owned());
                    } else {
                        fields.extend(
                            ["types", "import", "default", "require"]
                                .iter()
                                .filter_map(|k| dot.get(k).ok()?.string().ok().map(str::to_owned)),
                        );
                    }
                }
            }
            for f in fields {
                if let Some(m) = self.module_file(&join(dir, &f)) {
                    return Some(m);
                }
            }
        }
        self.module_file(&join(dir, "index")).or_else(|| self.module_file(&join(dir, "src/index")))
    }
    fn nearest_ts(&self, from: &str) -> Option<&TsPaths> {
        let mut d = dir_of(from).to_owned();
        loop {
            if let Some(t) = self.ts.get(&d) {
                return Some(t);
            }
            if d.is_empty() {
                return None;
            }
            d = dir_of(&d).to_owned();
        }
    }
    fn js(&self, from: &str, spec: &str) -> (Option<String>, Option<String>) {
        if spec.starts_with('.') || spec.starts_with('/') {
            return (self.module_file(&join(dir_of(from), spec)), None);
        }
        if let Some(ts) = self.nearest_ts(from) {
            for (pat, targets) in &ts.paths {
                let hit = match pat.split_once('*') {
                    Some((a, b)) => spec.strip_prefix(a).and_then(|r| r.strip_suffix(b)).map(str::to_owned),
                    None => (spec == pat).then(String::new),
                };
                if let Some(star) = hit {
                    for t in targets {
                        if let Some(m) = self.module_file(&t.replace('*', &star)) {
                            return (Some(m), None);
                        }
                    }
                }
            }
            if let Some(m) = ts.base.as_ref().and_then(|b| self.module_file(&join(b, spec))) {
                return (Some(m), None);
            }
        }
        for (name, dir) in &self.npm {
            if spec == name {
                return (self.package_entry(dir), None);
            }
            if let Some(rest) = spec.strip_prefix(&format!("{name}/")) {
                let m = self.module_file(&join(dir, rest)).or_else(|| self.module_file(&join(dir, &format!("src/{rest}"))));
                return (m, None);
            }
        }
        let mut parts = spec.split('/');
        let first = parts.next().unwrap_or(spec);
        let pkg = if first.starts_with('@') { format!("{first}/{}", parts.next().unwrap_or("")) } else { first.to_owned() };
        (None, Some(pkg))
    }
    fn python(&self, from: &str, imp: &Import) -> (Option<String>, Option<String>) {
        let spec = imp.spec.as_str();
        let dots = spec.chars().take_while(|c| *c == '.').count();
        let rest = spec.get(dots..).unwrap_or("");
        let try_mod = |base: &str, module: &str| -> Option<String> {
            let rel = module.replace('.', "/");
            let p = join(base, &rel);
            let file = format!("{p}.py");
            let init = join(&p, "__init__.py");
            if self.files.contains(file.as_str()) {
                Some(file)
            } else if self.files.contains(init.as_str()) {
                Some(init)
            } else {
                None
            }
        };
        if dots > 0 {
            let mut base = dir_of(from).to_owned();
            for _ in 1..dots {
                base = dir_of(&base).to_owned();
            }
            if rest.is_empty() {
                // `from . import x`: the submodule `x`, else the package itself.
                let sub = imp.names.first().and_then(|n| try_mod(&base, n));
                return (sub.or_else(|| try_mod(&base, "").filter(|_| self.files.contains(join(&base, "__init__.py").as_str()))), None);
            }
            return (try_mod(&base, rest), None);
        }
        let mut roots = vec![dir_of(from).to_owned()];
        roots.extend(self.py_roots.iter().cloned());
        for r in &roots {
            if let Some(m) = try_mod(r, rest) {
                return (Some(m), None);
            }
        }
        (None, Some(rest.split('.').next().unwrap_or(rest).to_owned()))
    }
    fn include(&self, from: &str, spec: &str) -> (Option<String>, Option<String>) {
        let angle = spec.starts_with('<');
        let name = spec.trim_matches(['<', '>']);
        if !angle {
            let p = join(dir_of(from), name);
            if self.files.contains(p.as_str()) {
                return (Some(p), None);
            }
        }
        let inc: Vec<&str> = self.files.iter().filter(|f| f.contains("include/")).copied().collect();
        if let Some(f) = inc.iter().find(|f| f.ends_with(&format!("include/{name}"))) {
            return (Some((*f).to_owned()), None);
        }
        let ends: Vec<&&str> = self.files.iter().filter(|f| **f == name || f.ends_with(&format!("/{name}"))).collect();
        if let [one] = ends.as_slice() {
            return (Some((**one).to_owned()), None);
        }
        (None, Some(name.to_owned()))
    }
}

/// Python import roots: the root, `src/` when it holds Python, and the `where` folders of
/// `[tool.setuptools.packages.find]` or the `from` of `[tool.poetry] packages` in any
/// `pyproject.toml`.
fn python_roots(files: &[FileInfo]) -> Vec<String> {
    let mut roots = vec![String::new()];
    if files.iter().any(|f| f.path.starts_with("src/") && f.path.ends_with(".py")) {
        roots.push("src".to_owned());
    }
    for f in files.iter().filter(|f| f.path.rsplit('/').next() == Some("pyproject.toml")) {
        let dir = dir_of(&f.path);
        for line in f.text.lines() {
            let t = line.trim();
            for key in ["where", "from"] {
                if let Some(v) = t.strip_prefix(key).map(str::trim_start).and_then(|r| r.strip_prefix('=')) {
                    for part in v.split(['"', '\'']).skip(1).step_by(2) {
                        let r = join(dir, part);
                        if !roots.contains(&r) {
                            roots.push(r);
                        }
                    }
                }
            }
        }
    }
    roots
}

/// Every import of the tree resolved.
#[must_use]
pub fn resolve(files: &[FileInfo], imports: &BTreeMap<String, Vec<Import>>) -> Vec<Edge> {
    let tree = Tree::new(files);
    let mut out = Vec::new();
    for (from, list) in imports {
        let lang = nmlex::lang::lang_of(from).map_or("", |l| l.ecosystem);
        for imp in list {
            let (to, external) = match lang {
                "npm" => tree.js(from, &imp.spec),
                "python" => tree.python(from, imp),
                "c" => tree.include(from, &imp.spec),
                "css" => (tree.module_file(&join(dir_of(from), &imp.spec)), None),
                _ => (None, None),
            };
            out.push(Edge { from: from.clone(), line: imp.line, spec: imp.spec.clone(), to, external });
        }
    }
    out
}

/// Normalised Python package name (`My_Pkg` and `my-pkg` are one name).
fn py_name(s: &str) -> String {
    s.to_lowercase().replace(['_', '.'], "-")
}

/// npm and Python packages of the tree, each with its dependencies on the others.
#[must_use]
pub fn packages(files: &[FileInfo]) -> Vec<Crate> {
    let mut npm: Vec<(String, String, Vec<String>, String)> = Vec::new();
    for f in files.iter().filter(|f| f.path.rsplit('/').next() == Some("package.json")) {
        let Ok(v) = nmjson::parse(f.text.as_bytes()) else { continue };
        let Some(name) = v.get("name").ok().and_then(|x| x.string().ok()).map(str::to_owned) else { continue };
        let mut deps = Vec::new();
        for k in ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"] {
            if let Ok(o) = v.get(k).and_then(nmjson::Value::object) {
                deps.extend(o.keys().cloned());
            }
        }
        npm.push((name, dir_of(&f.path).to_owned(), deps, f.path.clone()));
    }
    let mut py: Vec<(String, String, Vec<String>, String)> = Vec::new();
    for f in files.iter().filter(|f| f.path.rsplit('/').next() == Some("pyproject.toml")) {
        let (mut name, mut deps, mut section, mut in_deps) = (None, Vec::new(), String::new(), false);
        for line in f.text.lines() {
            let t = line.trim();
            if t.starts_with('[') {
                section = t.to_owned();
                in_deps = false;
                continue;
            }
            let project = section == "[project]" || section == "[tool.poetry]";
            if project
                && name.is_none()
                && let Some(v) = t.strip_prefix("name").map(str::trim_start).and_then(|r| r.strip_prefix('='))
            {
                name = Some(v.trim().trim_matches(['"', '\'']).to_owned());
            }
            if (section == "[project]" && t.starts_with("dependencies")) || in_deps {
                in_deps = !t.contains(']');
                for part in t.split(['"', '\'']).skip(1).step_by(2) {
                    let n: String = part.chars().take_while(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.')).collect();
                    deps.push(n);
                }
            }
            if section == "[tool.poetry.dependencies]"
                && let Some((k, _)) = t.split_once('=')
            {
                deps.push(k.trim().to_owned());
            }
        }
        if let Some(n) = name {
            py.push((n, dir_of(&f.path).to_owned(), deps, f.path.clone()));
        }
    }
    let npm_names: BTreeSet<String> = npm.iter().map(|p| p.0.clone()).collect();
    let py_names: BTreeSet<String> = py.iter().map(|p| py_name(&p.0)).collect();
    let mut out = Vec::new();
    for (name, dir, deps, manifest) in npm {
        let mut d: Vec<String> = deps.into_iter().filter(|x| npm_names.contains(x) && *x != name).collect();
        d.sort();
        d.dedup();
        out.push(Crate { name, dir, deps: d, manifest, ecosystem: "npm" });
    }
    for (name, dir, deps, manifest) in py {
        let mut d: Vec<String> = deps.into_iter().filter(|x| py_names.contains(&py_name(x)) && py_name(x) != py_name(&name)).collect();
        d.sort();
        d.dedup();
        out.push(Crate { name, dir, deps: d, manifest, ecosystem: "python" });
    }
    out
}

pub(crate) fn check() -> Result<(), String> {
    let files: Vec<FileInfo> = [
        ("package.json", r#"{"name":"root","workspaces":["packages/*"]}"#),
        ("packages/core/package.json", r#"{"name":"@x/core","main":"src/index.ts"}"#),
        ("packages/core/src/index.ts", "export const a = 1;\n"),
        ("packages/app/package.json", r#"{"name":"@x/app","dependencies":{"@x/core":"1","react":"18"}}"#),
        ("packages/app/tsconfig.json", "{ // paths\n \"compilerOptions\": { \"baseUrl\": \".\", \"paths\": { \"@/*\": [\"src/*\"] }, },\n}\n"),
        ("packages/app/src/util/fmt.ts", "export function f() {}\n"),
        ("packages/app/src/main.tsx", "import { a } from \"@x/core\";\nimport { f } from \"@/util/fmt\";\nimport x from \"./util/fmt.js\";\nimport React from \"react\";\n"),
        ("pkg/__init__.py", ""),
        ("pkg/io.py", "from . import util\nfrom .util import g\nimport os\n"),
        ("pkg/util.py", "def g():\n    pass\n"),
    ]
    .iter()
    .map(|(p, t)| FileInfo::new(p, t))
    .collect();
    let mut imports = BTreeMap::new();
    for f in &files {
        if let Some(t) = crate::tags::tag(&f.path, &f.text)
            && !t.imports.is_empty()
        {
            imports.insert(f.path.clone(), t.imports);
        }
    }
    let got: Vec<String> = resolve(&files, &imports)
        .iter()
        .map(|e| {
            format!("{}:{} {}", e.from, e.line, e.to.clone().or_else(|| e.external.clone().map(|x| format!("ext {x}"))).unwrap_or_default())
        })
        .collect();
    let want = [
        "packages/app/src/main.tsx:1 packages/core/src/index.ts",
        "packages/app/src/main.tsx:2 packages/app/src/util/fmt.ts",
        "packages/app/src/main.tsx:3 packages/app/src/util/fmt.ts",
        "packages/app/src/main.tsx:4 ext react",
        "pkg/io.py:1 pkg/util.py",
        "pkg/io.py:2 pkg/util.py",
        "pkg/io.py:3 ext os",
    ];
    if got != want {
        return Err(format!("nmsubs.imports: resolved {got:?}"));
    }
    let pk: Vec<String> = packages(&files).iter().map(|c| format!("{} {} [{}]", c.name, c.dir, c.deps.join(","))).collect();
    if pk != ["root  []", "@x/core packages/core []", "@x/app packages/app [@x/core]"] {
        return Err(format!("nmsubs.imports: packages {pk:?}"));
    }
    Ok(())
}
