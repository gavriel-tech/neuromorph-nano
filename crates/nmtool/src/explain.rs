//! Explain a build run from exact sources only (law 8): the location is rustc's primary
//! span; the cause is the error code's family from a fixed table of common rustc codes; the
//! fix is rustc's machine-applicable suggestion (the exact control fixed at S7), then
//! rustc's own help line, then the table's generic remedy. When none applies the fix says
//! so (law 6). A writer can be passed as a hook and is labelled when used; the default
//! path never calls it while it is unqualified.
use crate::build::{BuildRun, Diagnostic};
use std::path::Path;

/// Where the fix came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixSource {
    MachineApplicable,
    RustcHelp,
    RuleTable,
    Writer,
    None,
}
impl FixSource {
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            Self::MachineApplicable => "rustc-machine-applicable",
            Self::RustcHelp => "rustc-help",
            Self::RuleTable => "rule-table",
            Self::Writer => "writer",
            Self::None => "none",
        }
    }
}

/// The explanation of the first error (or failed test) of a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Explanation {
    pub file: String,
    pub line: usize,
    pub col: usize,
    /// Error code, or `test` for a failed test.
    pub code: String,
    pub message: String,
    /// The code family's cause, from the table; empty when the code is not in it.
    pub cause: String,
    pub fix: String,
    pub fix_source: FixSource,
    /// Up to two lines either side of the error line: (1-based line, text).
    pub around: Vec<(usize, String)>,
    /// Other errors in the same run, not explained here.
    pub others: usize,
}

/// A writer hook: given the diagnostic and the lines around it, a fixed line or nothing.
pub type Writer<'a> = &'a dyn Fn(&Diagnostic, &[(usize, String)]) -> Option<String>;

/// Cause and generic remedy for common rustc error codes.
#[must_use]
pub fn rule(code: &str) -> Option<(&'static str, &'static str)> {
    Some(match code {
        "E0425" => {
            ("unresolved name: no value or function with this name is in scope", "define it, import it with `use`, or fix the spelling")
        }
        "E0412" => ("unknown type name in this scope", "import the type or fix its path or spelling"),
        "E0433" => ("a path segment does not resolve (module, crate or type)", "add the `use` or the dependency, or fix the path"),
        "E0432" => ("unresolved import", "fix the `use` path or add the dependency to Cargo.toml"),
        "E0599" => (
            "no method or associated item with this name for the type",
            "call a method the type has, import the trait that provides it, or convert the value first",
        ),
        "E0609" => ("no field with this name on the type", "use an existing field or a method"),
        "E0560" => ("struct literal names a field the struct does not have", "remove or rename the field"),
        "E0063" => ("struct literal misses a field", "add the missing field or use `..Default::default()`"),
        "E0308" => (
            "mismatched types: the value's type differs from the expected type",
            "change the value, the annotation, or convert with `into`, `parse`, `as` or a constructor",
        ),
        "E0277" => ("a trait bound is not satisfied for the type", "implement or derive the trait, or pass a type that has it"),
        "E0282" | "E0283" => ("the type cannot be inferred", "add a type annotation or a turbofish"),
        "E0061" => ("wrong number of arguments in a call", "pass the arguments the signature lists"),
        "E0107" => ("wrong number of generic arguments", "match the item's generic parameters"),
        "E0106" => ("a reference type misses its lifetime", "add a named lifetime parameter"),
        "E0382" => ("use of a moved value", "clone before the move, borrow instead, or restructure ownership"),
        "E0499" | "E0502" => ("conflicting borrows of the same value", "shorten one borrow's scope or split the borrow"),
        "E0596" => ("mutable borrow of a value not declared mutable", "declare it `mut` or borrow it mutably at the source"),
        "E0384" => ("assignment to an immutable binding", "declare the binding `mut`"),
        "E0369" => ("binary operator not supported for these types", "implement the operator trait or convert the operands"),
        "E0614" => ("dereference of a value that is not a pointer", "remove the `*` or borrow first"),
        "E0603" | "E0616" | "E0624" => {
            ("the item, field or method is private", "make it `pub` where it is defined or use a public accessor")
        }
        "E0004" => ("the match is not exhaustive", "add the missing arms or a wildcard arm"),
        "E0023" | "E0026" | "E0027" => ("the pattern does not match the variant's fields", "list the variant's fields as declared"),
        "E0428" => ("the name is defined twice in one scope", "rename or remove one definition"),
        "E0423" => ("a struct or type name used as a value", "use the constructor or struct literal"),
        "E0015" => ("a non-const call in a constant context", "make the function `const` or compute the value at run time"),
        "E0728" => ("`await` outside an async context", "make the enclosing function `async`"),
        _ => return None,
    })
}

fn around(root: &Path, file: &str, line: usize) -> Vec<(usize, String)> {
    let Ok(text) = std::fs::read_to_string(root.join(file)) else { return Vec::new() };
    text.lines().enumerate().skip(line.saturating_sub(3)).take(5).map(|(i, l)| (i + 1, l.trim_end().chars().take(160).collect())).collect()
}

fn fix_of(d: &Diagnostic) -> (String, FixSource) {
    if let Some(c) = d.children.iter().find(|c| c.applicability == "MachineApplicable" && c.replacement.is_some()) {
        let rep = c.replacement.clone().unwrap_or_default();
        let at = c.at.as_ref().map_or_else(String::new, |(f, l, a, b)| format!(" at {f}:{l}:{a}-{b}"));
        return (format!("replace{at} with `{rep}` ({})", c.message), FixSource::MachineApplicable);
    }
    if let Some(c) = d.children.iter().find(|c| c.level == "help") {
        let rep = c.replacement.as_ref().filter(|r| !r.is_empty()).map_or_else(String::new, |r| format!(": `{r}`"));
        return (format!("{}{rep}", c.message), FixSource::RustcHelp);
    }
    rule(&d.code).map_or_else(
        || ("no exact fix known; read the lines around the error".to_owned(), FixSource::None),
        |(_, remedy)| (remedy.to_owned(), FixSource::RuleTable),
    )
}

/// Explain the first error of `run` (files read under `root`), or its first failed test;
/// `None` when the run has neither (the build is clean: withhold).
#[must_use]
pub fn explain(root: &Path, run: &BuildRun, writer: Option<Writer<'_>>) -> Option<Explanation> {
    let errors = run.errors();
    if let Some(d) = errors.first() {
        let ctx = around(root, &d.file, d.line);
        let (mut fix, mut source) = fix_of(d);
        if source != FixSource::MachineApplicable
            && let Some(line) = writer.and_then(|w| w(d, &ctx))
        {
            fix = format!("line {} becomes `{line}` (writer, unqualified)", d.line);
            source = FixSource::Writer;
        }
        return Some(Explanation {
            file: d.file.clone(),
            line: d.line,
            col: d.col,
            code: d.code.clone(),
            message: d.message.clone(),
            cause: rule(&d.code).map(|r| r.0.to_owned()).unwrap_or_default(),
            fix,
            fix_source: source,
            around: ctx,
            others: errors.len() - 1,
        });
    }
    let t = run.tests.first()?;
    let (file, line) = t.at.clone().unwrap_or_default();
    Some(Explanation {
        around: around(root, &file, line),
        file,
        line,
        col: 0,
        code: "test".into(),
        message: format!("test {} failed: {}", t.name, t.message),
        cause: "a test assertion or panic failed at this line".into(),
        fix: "read the assertion and the values it compared".into(),
        fix_source: FixSource::None,
        others: run.tests.len() - 1,
    })
}

/// Section check on a built-in fixture.
/// # Errors
/// One line naming what failed.
pub fn check() -> Result<(), String> {
    use crate::build::Child;
    let dir = crate::scratch_dir("explain")?;
    std::fs::create_dir_all(dir.join("src")).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("src/lib.rs"), "fn a() {}\nfn b() {\n    let x = y;\n}\nfn c() {}\n").map_err(|e| e.to_string())?;
    let mut d = Diagnostic {
        level: "error".into(),
        code: "E0425".into(),
        message: "cannot find value `y`".into(),
        file: "src/lib.rs".into(),
        line: 3,
        col: 13,
        label: String::new(),
        children: Vec::new(),
    };
    let mut run = BuildRun { diags: vec![d.clone()], ..BuildRun::default() };
    let e = explain(&dir, &run, None).ok_or("no explanation")?;
    if e.line != 3 || e.fix_source != FixSource::RuleTable || e.around.len() != 5 || e.cause.is_empty() {
        return Err(format!("rule-table explanation {e:?}"));
    }
    d.children.push(Child {
        level: "help".into(),
        message: "use `x`".into(),
        replacement: Some("x".into()),
        applicability: "MachineApplicable".into(),
        at: Some(("src/lib.rs".into(), 3, 13, 14)),
    });
    run.diags = vec![d];
    let e = explain(&dir, &run, Some(&|_, _| Some("let x = 0;".into()))).ok_or("no explanation")?;
    if e.fix_source != FixSource::MachineApplicable {
        return Err("the machine-applicable fix must win over the writer".into());
    }
    if explain(&dir, &BuildRun::default(), None).is_some() {
        return Err("a clean run must withhold".into());
    }
    Ok(())
}
