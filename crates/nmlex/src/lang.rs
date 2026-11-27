//! The language table: one entry per programming language the reader takes, with what the
//! generic lexer (`lang_lex`) and the generic tagger need to know about it. Rust is listed so that
//! `lang_of` names it, but Rust keeps its own scanner (`tokens` with `rust = true`).
//!
//! Entries are thin: comment and string forms, definition head words, block style. A source file
//! in a language without an entry is read through `GENERIC` (language-neutral comments, strings,
//! head words and blocks), so no code is skipped for its language.

/// How a language marks the extent of a definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Block {
    /// `{ ... }` bodies (C family, JavaScript, PowerShell, CSS).
    Brace,
    /// Indentation (Python).
    Indent,
    /// Unknown or mixed: a definition is a head word at the start of a line followed by a name;
    /// it ends at its matching `}` when it opens one on its line, else before the next line
    /// indented as little as it (an `end`, `fi` or `done` there closes it).
    Neutral,
}

/// What the lexer and tagger know about one language.
#[derive(Debug)]
pub struct Lang {
    /// Short id, also the reader's decoder name: `typescript`, `python`, `c`, ...
    pub id: &'static str,
    /// File extensions, lower case, without the dot.
    pub exts: &'static [&'static str],
    pub block: Block,
    /// Line comment openers.
    pub line_comments: &'static [&'static str],
    /// Block comment pairs.
    pub block_comments: &'static [(&'static str, &'static str)],
    /// Quote characters of ordinary strings.
    pub quotes: &'static [char],
    /// The escape character inside ordinary strings.
    pub escape: char,
    /// A quote doubled inside a string stands for itself (PowerShell `'it''s'`).
    pub doubled_quote: bool,
    /// Python: triple-quoted strings and the prefixes `r b f u` and their pairs.
    pub python_strings: bool,
    /// JavaScript template literals with `${ }`.
    pub template: bool,
    /// Go: backtick raw strings.
    pub backtick_raw: bool,
    /// C#: verbatim `@"..."`, interpolated `$"..."` and raw `"""..."""` strings.
    pub csharp_strings: bool,
    /// PowerShell: here-strings `@" ... "@` and `@' ... '@`.
    pub here_strings: bool,
    /// C++: raw strings `R"x(...)x"`.
    pub cpp_raw: bool,
    /// JavaScript: `/` may open a regular expression literal.
    pub regex_literal: bool,
    /// Characters allowed inside identifiers besides letters, digits and `_`. A `-` is taken only
    /// between a letter or digit and a letter (`Get-Item`, `--accent-color`).
    pub ident_extra: &'static str,
    /// Characters that may start an identifier besides letters and `_`.
    pub ident_start: &'static str,
    /// Words skipped before a definition head.
    pub modifiers: &'static [&'static str],
    /// Definition head word and the kind it gives.
    pub defs: &'static [(&'static str, &'static str)],
    /// Path parts that mark test code (`.test.`, `__tests__/`, `test_`).
    pub test_paths: &'static [&'static str],
    /// Text that marks test code inside a file.
    pub test_marks: &'static [&'static str],
    /// Words that mark an item as part of the public interface.
    pub public_marks: &'static [&'static str],
    /// Split `camelCase` words for term counts.
    pub camel: bool,
    /// Package ecosystem: `npm`, `python`, `c`, `powershell`, `css`, `cargo`.
    pub ecosystem: &'static str,
}

const NONE: &[&str] = &[];

const JS_MODIFIERS: &[&str] = &[
    "export",
    "default",
    "declare",
    "async",
    "abstract",
    "public",
    "private",
    "protected",
    "static",
    "readonly",
    "override",
    "accessor",
    "get",
    "set",
];
const JS_DEFS: &[(&str, &str)] = &[
    ("function", "function"),
    ("class", "class"),
    ("interface", "interface"),
    ("type", "type"),
    ("enum", "enum"),
    ("namespace", "namespace"),
];
const JS_TEST_PATHS: &[&str] = &[".test.", ".spec.", "__tests__/", "/test/", "/tests/", "test/", "tests/"];
const JS_TEST_MARKS: &[&str] = &["describe(", "it(", "test("];

const C_MODIFIERS: &[&str] = &[
    "static",
    "inline",
    "extern",
    "const",
    "volatile",
    "unsigned",
    "signed",
    "__global__",
    "__device__",
    "__host__",
    "__forceinline__",
    "__inline__",
    "__noinline__",
    "constexpr",
];
const C_DEFS: &[(&str, &str)] = &[("struct", "struct"), ("union", "struct"), ("enum", "enum"), ("typedef", "type"), ("class", "class")];

/// Every language in the table.
pub const LANGS: &[Lang] = &[
    Lang {
        id: "rust",
        exts: &["rs"],
        block: Block::Brace,
        line_comments: &["//"],
        block_comments: &[("/*", "*/")],
        quotes: &['"'],
        escape: '\\',
        doubled_quote: false,
        python_strings: false,
        template: false,
        backtick_raw: false,
        csharp_strings: false,
        here_strings: false,
        cpp_raw: false,
        regex_literal: false,
        ident_extra: "",
        ident_start: "",
        modifiers: &["pub", "async", "unsafe", "const", "extern"],
        defs: &[("fn", "fn"), ("struct", "struct"), ("enum", "enum"), ("trait", "trait"), ("mod", "mod"), ("type", "type")],
        test_paths: &["/tests/"],
        test_marks: &["#[test]", "#[cfg(test)]"],
        public_marks: &["pub"],
        camel: false,
        ecosystem: "cargo",
    },
    Lang {
        id: "typescript",
        exts: &["ts", "tsx", "mts", "cts"],
        block: Block::Brace,
        line_comments: &["//"],
        block_comments: &[("/*", "*/")],
        quotes: &['"', '\''],
        escape: '\\',
        doubled_quote: false,
        python_strings: false,
        template: true,
        backtick_raw: false,
        csharp_strings: false,
        here_strings: false,
        cpp_raw: false,
        regex_literal: true,
        ident_extra: "$",
        ident_start: "$",
        modifiers: JS_MODIFIERS,
        defs: JS_DEFS,
        test_paths: JS_TEST_PATHS,
        test_marks: JS_TEST_MARKS,
        public_marks: &["export"],
        camel: true,
        ecosystem: "npm",
    },
    Lang {
        id: "javascript",
        exts: &["js", "jsx", "mjs", "cjs"],
        block: Block::Brace,
        line_comments: &["//"],
        block_comments: &[("/*", "*/")],
        quotes: &['"', '\''],
        escape: '\\',
        doubled_quote: false,
        python_strings: false,
        template: true,
        backtick_raw: false,
        csharp_strings: false,
        here_strings: false,
        cpp_raw: false,
        regex_literal: true,
        ident_extra: "$",
        ident_start: "$",
        modifiers: &["export", "default", "async", "static", "get", "set"],
        defs: &[("function", "function"), ("class", "class")],
        test_paths: JS_TEST_PATHS,
        test_marks: JS_TEST_MARKS,
        public_marks: &["export", "module.exports", "exports."],
        camel: true,
        ecosystem: "npm",
    },
    Lang {
        id: "python",
        exts: &["py", "pyi"],
        block: Block::Indent,
        line_comments: &["#"],
        block_comments: &[],
        quotes: &['"', '\''],
        escape: '\\',
        doubled_quote: false,
        python_strings: true,
        template: false,
        backtick_raw: false,
        csharp_strings: false,
        here_strings: false,
        cpp_raw: false,
        regex_literal: false,
        ident_extra: "",
        ident_start: "",
        modifiers: &["async"],
        defs: &[("def", "function"), ("class", "class")],
        test_paths: &["test_", "_test.py", "/tests/", "tests/", "conftest.py"],
        test_marks: &["def test_", "unittest.TestCase", "import pytest"],
        public_marks: NONE,
        camel: true,
        ecosystem: "python",
    },
    Lang {
        id: "c",
        exts: &["c", "h"],
        block: Block::Brace,
        line_comments: &["//"],
        block_comments: &[("/*", "*/")],
        quotes: &['"', '\''],
        escape: '\\',
        doubled_quote: false,
        python_strings: false,
        template: false,
        backtick_raw: false,
        csharp_strings: false,
        here_strings: false,
        cpp_raw: false,
        regex_literal: false,
        ident_extra: "",
        ident_start: "",
        modifiers: C_MODIFIERS,
        defs: C_DEFS,
        test_paths: &["/tests/", "tests/", "test_", "_test."],
        test_marks: NONE,
        public_marks: NONE,
        camel: true,
        ecosystem: "c",
    },
    Lang {
        id: "cpp",
        exts: &["cpp", "cc", "cxx", "hpp", "hh", "hxx"],
        block: Block::Brace,
        line_comments: &["//"],
        block_comments: &[("/*", "*/")],
        quotes: &['"', '\''],
        escape: '\\',
        doubled_quote: false,
        python_strings: false,
        template: false,
        backtick_raw: false,
        csharp_strings: false,
        here_strings: false,
        cpp_raw: true,
        regex_literal: false,
        ident_extra: "",
        ident_start: "",
        modifiers: C_MODIFIERS,
        defs: &[
            ("struct", "struct"),
            ("union", "struct"),
            ("enum", "enum"),
            ("typedef", "type"),
            ("class", "class"),
            ("namespace", "namespace"),
        ],
        test_paths: &["/tests/", "tests/", "test_", "_test."],
        test_marks: &["TEST(", "TEST_F("],
        public_marks: NONE,
        camel: true,
        ecosystem: "c",
    },
    Lang {
        id: "cuda",
        exts: &["cu", "cuh"],
        block: Block::Brace,
        line_comments: &["//"],
        block_comments: &[("/*", "*/")],
        quotes: &['"', '\''],
        escape: '\\',
        doubled_quote: false,
        python_strings: false,
        template: false,
        backtick_raw: false,
        csharp_strings: false,
        here_strings: false,
        cpp_raw: true,
        regex_literal: false,
        ident_extra: "",
        ident_start: "",
        modifiers: C_MODIFIERS,
        defs: &[
            ("struct", "struct"),
            ("union", "struct"),
            ("enum", "enum"),
            ("typedef", "type"),
            ("class", "class"),
            ("namespace", "namespace"),
        ],
        test_paths: &["/tests/", "tests/", "test_", "_test."],
        test_marks: NONE,
        public_marks: NONE,
        camel: true,
        ecosystem: "c",
    },
    Lang {
        id: "powershell",
        exts: &["ps1", "psm1", "psd1"],
        block: Block::Brace,
        line_comments: &["#"],
        block_comments: &[("<#", "#>")],
        quotes: &['"', '\''],
        escape: '`',
        doubled_quote: true,
        python_strings: false,
        template: false,
        backtick_raw: false,
        csharp_strings: false,
        here_strings: true,
        cpp_raw: false,
        regex_literal: false,
        ident_extra: "-",
        ident_start: "$",
        modifiers: NONE,
        defs: &[("function", "function"), ("filter", "function"), ("class", "class"), ("enum", "enum")],
        test_paths: &[".tests.ps1", "/tests/", "tests/"],
        test_marks: &["Describe ", "It "],
        public_marks: &["Export-ModuleMember"],
        camel: true,
        ecosystem: "powershell",
    },
    Lang {
        id: "css",
        exts: &["css"],
        block: Block::Brace,
        line_comments: NONE,
        block_comments: &[("/*", "*/")],
        quotes: &['"', '\''],
        escape: '\\',
        doubled_quote: false,
        python_strings: false,
        template: false,
        backtick_raw: false,
        csharp_strings: false,
        here_strings: false,
        cpp_raw: false,
        regex_literal: false,
        ident_extra: "-",
        ident_start: "-",
        modifiers: NONE,
        defs: &[],
        test_paths: NONE,
        test_marks: NONE,
        public_marks: NONE,
        camel: false,
        ecosystem: "css",
    },
    Lang {
        id: "go",
        exts: &["go"],
        quotes: &['"', '\''],
        backtick_raw: true,
        defs: GO_DEFS,
        test_paths: &["_test.go"],
        test_marks: &["func Test"],
        ecosystem: "go",
        ..BRACE
    },
    Lang {
        id: "java",
        exts: &["java"],
        modifiers: JVM_MODIFIERS,
        defs: &[("class", "class"), ("interface", "interface"), ("enum", "enum"), ("record", "class")],
        test_paths: &["/test/", "Test.java"],
        test_marks: &["@Test"],
        ecosystem: "jvm",
        ..BRACE
    },
    Lang {
        id: "kotlin",
        exts: &["kt", "kts"],
        modifiers: JVM_MODIFIERS,
        defs: &[("fun", "function"), ("class", "class"), ("interface", "interface"), ("object", "class"), ("typealias", "type")],
        test_paths: &["/test/", "Test.kt"],
        test_marks: &["@Test"],
        ecosystem: "jvm",
        ..BRACE
    },
    Lang {
        id: "scala",
        exts: &["scala", "sc"],
        modifiers: JVM_MODIFIERS,
        defs: &[("def", "function"), ("class", "class"), ("object", "class"), ("trait", "trait")],
        ecosystem: "jvm",
        ..BRACE
    },
    Lang {
        id: "csharp",
        exts: &["cs"],
        csharp_strings: true,
        modifiers: JVM_MODIFIERS,
        defs: &[
            ("class", "class"),
            ("interface", "interface"),
            ("struct", "struct"),
            ("enum", "enum"),
            ("record", "class"),
            ("namespace", "namespace"),
        ],
        test_paths: &["Tests/", "Test.cs", "Tests.cs"],
        test_marks: &["[Fact]", "[Test]", "[TestMethod]"],
        ecosystem: "dotnet",
        ..BRACE
    },
    Lang {
        id: "swift",
        exts: &["swift"],
        modifiers: JVM_MODIFIERS,
        defs: &[
            ("func", "function"),
            ("class", "class"),
            ("struct", "struct"),
            ("enum", "enum"),
            ("protocol", "interface"),
            ("extension", "class"),
        ],
        ecosystem: "swift",
        ..BRACE
    },
    Lang {
        id: "dart",
        exts: &["dart"],
        modifiers: JVM_MODIFIERS,
        defs: &[("class", "class"), ("mixin", "class"), ("enum", "enum"), ("extension", "class")],
        ecosystem: "dart",
        ..BRACE
    },
    Lang {
        id: "php",
        exts: &["php"],
        line_comments: &["//", "#"],
        quotes: &['"', '\''],
        ident_extra: "$",
        ident_start: "$",
        modifiers: JVM_MODIFIERS,
        defs: &[("function", "function"), ("class", "class"), ("interface", "interface"), ("trait", "trait"), ("enum", "enum")],
        ecosystem: "php",
        ..BRACE
    },
    Lang {
        id: "ruby",
        exts: &["rb", "rake", "gemspec"],
        block: Block::Neutral,
        line_comments: &["#"],
        block_comments: &[("=begin", "=end")],
        quotes: &['"', '\''],
        defs: &[("def", "function"), ("class", "class"), ("module", "module")],
        test_paths: &["_spec.rb", "_test.rb", "spec/"],
        ecosystem: "ruby",
        ..BRACE
    },
    Lang {
        id: "lua",
        exts: &["lua"],
        block: Block::Neutral,
        line_comments: &["--"],
        block_comments: &[("--[[", "]]")],
        quotes: &['"', '\''],
        modifiers: &["local"],
        defs: &[("function", "function")],
        ecosystem: "lua",
        ..BRACE
    },
    Lang {
        id: "shell",
        exts: &["sh", "bash", "zsh", "fish"],
        block: Block::Neutral,
        line_comments: &["#"],
        block_comments: &[],
        quotes: &['"', '\''],
        ident_extra: "-",
        ident_start: "$",
        defs: &[("function", "function")],
        ecosystem: "shell",
        ..BRACE
    },
    Lang {
        id: "sql",
        exts: &["sql"],
        block: Block::Neutral,
        line_comments: &["--"],
        quotes: &['\''],
        doubled_quote: true,
        modifiers: &["create", "CREATE", "or", "OR", "replace", "REPLACE", "temporary", "TEMPORARY", "materialized", "MATERIALIZED"],
        defs: &[
            ("table", "table"),
            ("view", "table"),
            ("function", "function"),
            ("procedure", "function"),
            ("TABLE", "table"),
            ("VIEW", "table"),
            ("FUNCTION", "function"),
            ("PROCEDURE", "function"),
        ],
        camel: false,
        ecosystem: "sql",
        ..BRACE
    },
];

/// The base of the thin brace entries: C-style comments and strings, no special forms.
const BRACE: Lang = Lang {
    id: "",
    exts: &[],
    block: Block::Brace,
    line_comments: &["//"],
    block_comments: &[("/*", "*/")],
    quotes: &['"', '\'', '`'],
    escape: '\\',
    doubled_quote: false,
    python_strings: false,
    template: false,
    backtick_raw: false,
    csharp_strings: false,
    here_strings: false,
    cpp_raw: false,
    regex_literal: false,
    ident_extra: "",
    ident_start: "",
    modifiers: &[],
    defs: &[],
    test_paths: &["/test/", "/tests/", "test/", "tests/"],
    test_marks: &[],
    public_marks: &[],
    camel: true,
    ecosystem: "code",
};

const GO_DEFS: &[(&str, &str)] = &[("func", "function"), ("type", "type")];
const JVM_MODIFIERS: &[&str] = &[
    "public",
    "private",
    "protected",
    "internal",
    "static",
    "final",
    "abstract",
    "open",
    "override",
    "sealed",
    "data",
    "inline",
    "suspend",
    "async",
    "virtual",
    "partial",
    "readonly",
    "unsafe",
    "extern",
    "export",
    "fileprivate",
    "mutating",
];

/// The language-neutral entry for code in a language without its own entry: `//`, `#` and
/// `/* */` comments, three kinds of quotes, the common definition head words, neutral blocks.
pub const GENERIC: Lang = Lang {
    id: "code",
    block: Block::Neutral,
    line_comments: &["//", "#"],
    modifiers: &["public", "private", "protected", "static", "export", "async", "local", "pub", "final", "abstract", "override", "inline"],
    defs: GENERIC_DEFS,
    ..BRACE
};
const GENERIC_DEFS: &[(&str, &str)] = &[
    ("function", "function"),
    ("func", "function"),
    ("fun", "function"),
    ("fn", "function"),
    ("def", "function"),
    ("defn", "function"),
    ("sub", "function"),
    ("proc", "function"),
    ("procedure", "function"),
    ("method", "method"),
    ("class", "class"),
    ("struct", "struct"),
    ("interface", "interface"),
    ("trait", "trait"),
    ("module", "module"),
    ("object", "class"),
    ("type", "type"),
    ("enum", "enum"),
    ("record", "class"),
    ("protocol", "interface"),
    ("macro", "macro"),
    ("let", "variable"),
];

/// The language of a path by its extension, if the table has it.
#[must_use]
pub fn lang_of(path: &str) -> Option<&'static Lang> {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path).to_lowercase();
    let (_, ext) = name.rsplit_once('.')?;
    LANGS.iter().find(|l| l.exts.contains(&ext))
}

/// The language with this id (`code` is `GENERIC`).
#[must_use]
pub fn by_id(id: &str) -> Option<&'static Lang> {
    if id == GENERIC.id {
        return Some(&GENERIC);
    }
    LANGS.iter().find(|l| l.id == id)
}

/// True when `path` is test code by the conventions of its language.
#[must_use]
pub fn is_test_path(path: &str) -> bool {
    let p = path.replace('\\', "/");
    let name = p.rsplit('/').next().unwrap_or(&p);
    lang_of(&p).is_some_and(|l| {
        l.test_paths.iter().any(|t| {
            if t.ends_with('/') && !t.starts_with('/') {
                p.starts_with(t) || p.contains(&format!("/{t}"))
            } else if t.starts_with('/') {
                p.contains(t)
            } else if t.ends_with('_') {
                name.starts_with(t)
            } else {
                name.contains(t)
            }
        })
    })
}

/// True when `text` holds one of the language's test markers at the start of a word
/// (`it(` counts, `submit(` does not).
#[must_use]
pub fn has_test_mark(lang: &Lang, text: &str) -> bool {
    lang.test_marks.iter().any(|m| {
        text.match_indices(m)
            .any(|(i, _)| text.get(..i).and_then(|b| b.chars().last()).is_none_or(|c| !(c.is_alphanumeric() || c == '_' || c == '.')))
    })
}

/// The parts of a `camelCase`, `PascalCase` or `HTTPServer` word, lower case, when it has more
/// than one (`UserService` gives `user`, `service`; `parseHTTPHeader` gives `parse`, `http`,
/// `header`). Digits stay with the part before them.
#[must_use]
pub fn camel_parts(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        let prev = i.checked_sub(1).and_then(|j| chars.get(j)).copied();
        let next = chars.get(i + 1).copied();
        let boundary = c.is_uppercase()
            && prev.is_some_and(|p| p.is_lowercase() || p.is_ascii_digit() || (p.is_uppercase() && next.is_some_and(char::is_lowercase)));
        if !c.is_alphanumeric() {
            if !cur.is_empty() {
                parts.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if boundary && !cur.is_empty() {
            parts.push(std::mem::take(&mut cur));
        }
        cur.extend(c.to_lowercase());
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    if parts.len() > 1 { parts } else { Vec::new() }
}

pub(crate) fn check() -> Result<(), String> {
    let cases = [
        ("src/app.ts", Some("typescript")),
        ("a/B.TSX", Some("typescript")),
        ("x.mjs", Some("javascript")),
        ("tool.py", Some("python")),
        ("k/attention.cu", Some("cuda")),
        ("inc/emma.h", Some("c")),
        ("tools/verify.ps1", Some("powershell")),
        ("app/globals.css", Some("css")),
        ("lib.rs", Some("rust")),
        ("notes.md", None),
        ("Makefile", None),
    ];
    for (p, want) in cases {
        if lang_of(p).map(|l| l.id) != want {
            return Err(format!("nmlex.lang: {p} is {:?}, want {want:?}", lang_of(p).map(|l| l.id)));
        }
    }
    let tests =
        [("src/a.test.ts", true), ("src/__tests__/b.ts", true), ("tests/test_x.py", true), ("pkg/test_y.py", true), ("src/app.ts", false)];
    for (p, want) in tests {
        if is_test_path(p) != want {
            return Err(format!("nmlex.lang: is_test_path({p}) is {}", !want));
        }
    }
    if camel_parts("UserService") != ["user", "service"]
        || camel_parts("parseHTTPHeader") != ["parse", "http", "header"]
        || !camel_parts("lower").is_empty()
        || camel_parts("toUTF8String") != ["to", "utf8", "string"]
    {
        return Err(format!("nmlex.lang: camel_parts {:?}", camel_parts("toUTF8String")));
    }
    Ok(())
}
