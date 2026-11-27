//! The format table: which files are read, by which decoder, and why the others are skipped.
//! First by name and extension, then by the first bytes (magic numbers and binary content), so a
//! PDF named `.txt` is still a PDF. A file no specific reader fits is read generally
//! (`crate::general`); only what holds nothing to read or must not be read is skipped, with a
//! named reason, and counted: model weights, files over the size limit, secrets, and generated,
//! minified, lock and hidden files.

/// What to do with one file. `Doc("general")` (`general::GENERAL`) marks a file read generally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Source code: `rust` and `toml` on the code path they always had, and every language of the
    /// language table (`nmlex::lang`) by its id (`typescript`, `python`, `c`, ...).
    Code(&'static str),
    /// A document or data file, read by the named decoder.
    Doc(&'static str),
    /// Not read, with the reason.
    Skip(&'static str),
}

/// Largest file read as text, in bytes (the present limit).
pub const MAX_TEXT: u64 = 512 * 1024;
/// Largest file read generally (a container or raw bytes), in bytes.
pub const MAX_GENERAL: u64 = 64 << 20;

/// Why code in a language outside the language table is not read. Read as plain text, without its
/// definitions, it crowds out code that is tagged, so a language is read only once the table
/// holds it and its fixtures pass.
pub const LANG_LATER: &str = "code: language not in the language table yet";

/// Code languages recognised by extension; those outside the language table are counted under
/// `LANG_LATER`.
const CODE: &[&str] = &[
    "py", "pyi", "js", "mjs", "cjs", "jsx", "ts", "mts", "cts", "tsx", "go", "java", "kt", "kts", "cs", "fs", "c", "h", "cpp", "cc", "cxx",
    "hpp", "hh", "hxx", "swift", "rb", "php", "scala", "lua", "sh", "bash", "zsh", "fish", "ps1", "psm1", "psd1", "cmd", "bat", "sql", "css",
    "scss", "sass", "less", "cu", "cuh", "r", "jl", "dart", "ex", "exs", "erl", "hrl", "hs", "ml", "mli", "vue", "svelte", "gradle", "cmake",
    "mk", "pl", "pm", "vb", "zig", "nim", "proto", "graphql", "gql", "tf", "hcl", "nix", "ini", "cfg", "conf", "properties", "editorconfig",
    "gitignore", "gitattributes", "ptx",
];
/// File names (no extension) read as code text.
const CODE_NAMES: &[&str] = &["makefile", "dockerfile", "cmakelists.txt", "jenkinsfile", "vagrantfile", "gemfile", "rakefile", "procfile"];
/// File names (no extension) read as plain text documents.
const TEXT_NAMES: &[&str] =
    &["readme", "license", "licence", "copying", "notice", "authors", "contributors", "changelog", "changes", "history", "news", "todo", "install"];
/// Private keys and credential stores by name and extension: never read.
const SECRET_NAMES: &[&str] = &["id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials", "credentials.json", "client_secret.json"];
const SECRET_EXT: &[&str] = &["pem", "key", "p12", "pfx", "jks", "keystore", "kdbx", "ppk", "asc", "gpg"];
/// Lock files: generated, they tell the package manager; never read.
const LOCKS: &[&str] = &[
    "cargo.lock",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lock",
    "bun.lockb",
    "poetry.lock",
    "uv.lock",
    "pipfile.lock",
    "composer.lock",
    "gemfile.lock",
    "go.sum",
    "flake.lock",
    "packages.lock.json",
];

/// The class of a file by its name and size alone.
#[must_use]
pub fn by_name(name: &str, size: u64) -> Class {
    let low = name.to_lowercase();
    if low.starts_with('.') {
        return Class::Skip(if low.starts_with(".env") { "secret store (.env): never read" } else { "hidden file" });
    }
    if SECRET_NAMES.contains(&low.as_str()) || low.rsplit_once('.').is_some_and(|(_, e)| SECRET_EXT.contains(&e)) {
        return Class::Skip("secret store (keys or credentials): never read");
    }
    if LOCKS.contains(&low.as_str()) {
        return Class::Skip("lock file (generated): not read");
    }
    if low.contains(".min.") && (low.ends_with(".js") || low.ends_with(".css")) {
        return Class::Skip("minified file (generated): not read");
    }
    let ext = low.rsplit_once('.').map_or("", |(_, e)| e);
    let class = match ext {
        "rs" => Class::Code("rust"),
        "toml" => Class::Code("toml"),
        "md" | "markdown" | "mdx" | "mdown" | "mkd" => Class::Doc("markdown"),
        "txt" | "text" | "rst" | "adoc" | "asciidoc" | "org" | "me" | "nfo" => {
            if CODE_NAMES.contains(&low.as_str()) {
                Class::Code("code")
            } else {
                Class::Doc("text")
            }
        }
        "csv" => Class::Doc("csv"),
        "tsv" | "tab" => Class::Doc("tsv"),
        "html" | "htm" | "xhtml" => Class::Doc("html"),
        "xml" | "xsd" | "xsl" | "xslt" | "rss" | "atom" | "plist" | "csproj" | "vbproj" | "fsproj" | "props" | "targets" | "resx" | "xaml" | "nuspec"
        | "kml" | "gpx" | "dita" | "opf" | "ncx" => Class::Doc("xml"),
        "json" | "geojson" | "jsonc" | "json5" | "webmanifest" => Class::Doc("json"),
        "jsonl" | "ndjson" => Class::Doc("jsonl"),
        "ipynb" => Class::Doc("notebook"),
        "yaml" | "yml" => Class::Doc("yaml"),
        "log" => Class::Doc("log"),
        "pdf" => Class::Doc("pdf"),
        // Read generally: containers opened in memory, anything else as runs of text from its
        // bytes (`general`).
        "docx" | "docm" | "dotx" | "xlsx" | "xlsm" | "xltx" | "pptx" | "pptm" | "potx" | "epub" | "odt" | "ods" | "odp" | "odg" | "rtf" | "doc"
        | "xls" | "ppt" | "dot" | "xlt" | "pot" | "msg" | "eml" | "png" | "jpg" | "jpeg" | "gif" | "bmp" | "tif" | "tiff" | "webp" | "ico" | "heic" | "svgz"
        | "psd" | "zip" | "7z" | "rar" | "tar" | "gz" | "tgz" | "gzip" | "bz2" | "xz" | "zst" | "cab" | "iso" | "mp3" | "mp4" | "wav" | "avi" | "mov"
        | "mkv" | "flac" | "ogg" | "m4a" | "webm" | "exe" | "dll" | "so" | "dylib" | "o" | "obj" | "a" | "lib" | "pdb" | "bin" | "class" | "jar"
        | "war" | "apk" | "nupkg" | "vsix" | "whl" | "wasm" | "fatbin" | "cubin" | "pyc" | "node" | "db" | "sqlite" | "sqlite3" | "mdb" | "accdb"
        | "ttf" | "otf" | "woff" | "woff2" | "eot" => Class::Doc(crate::general::GENERAL),
        "svg" => Class::Doc("xml"),
        "safetensors" | "gguf" | "onnx" | "pt" | "pth" | "ckpt" | "npy" | "npz" | "h5" | "pkl" | "parquet" | "arrow" | "f32" | "f16" => {
            Class::Skip("model weights or binary data: not read")
        }
        "map" => Class::Skip("source map (generated): not read"),
        "lock" | "lockb" => Class::Skip("lock file (generated): not read"),
        "tsbuildinfo" => Class::Skip("build cache (generated): not read"),
        _ if nmlex::lang::lang_of(&low).is_some() => nmlex::lang::lang_of(&low).map_or(Class::Skip(LANG_LATER), |l| Class::Code(l.id)),
        "ptx" => Class::Skip("compiled kernel assembly (generated): not read"),
        // Code in a language without its own table entry: read through the language-neutral entry.
        _ if CODE.contains(&ext) || CODE_NAMES.contains(&low.as_str()) => Class::Code("code"),
        _ if TEXT_NAMES.contains(&low.as_str()) || TEXT_NAMES.contains(&ext) => Class::Doc("text"),
        // Any other name: read as plain text when its bytes are text (`by_bytes` decides).
        _ => Class::Doc("text?"),
    };
    match class {
        Class::Skip(_) => class,
        Class::Doc(crate::general::GENERAL | "pdf") if size >= MAX_GENERAL => Class::Skip("too large: over 64 MB"),
        Class::Doc(crate::general::GENERAL | "pdf") => class,
        _ if size >= MAX_TEXT => Class::Skip("too large: over 512 KB"),
        _ => class,
    }
}

/// The class of a file once its first bytes are known: magic numbers win over the name, binary
/// content is skipped, and a file of unknown name is read as text only when it is text.
#[must_use]
pub fn by_bytes(class: Class, bytes: &[u8]) -> Class {
    if matches!(class, Class::Skip(_) | Class::Doc(crate::general::GENERAL | "pdf")) {
        return class;
    }
    let head = bytes.get(..bytes.len().min(8)).unwrap_or(&[]);
    if head.starts_with(b"%PDF-") {
        return Class::Doc("pdf");
    }
    // Magic numbers of formats read generally (zip containers, RTF, old binary Office) win over
    // the name, and so does binary content.
    if head.starts_with(b"PK\x03\x04")
        || head.starts_with(b"{\\rtf")
        || head.starts_with(&[0xD0, 0xCF, 0x11, 0xE0])
        || head.starts_with(&[0x1F, 0x8B, 0x08])
        || crate::text::is_binary(bytes)
    {
        return Class::Doc(crate::general::GENERAL);
    }
    match class {
        Class::Doc("text?") => {
            // Unknown name: text only when it is not one long line of data.
            let sample = bytes.get(..bytes.len().min(8192)).unwrap_or(&[]);
            let lines = sample.iter().filter(|b| **b == b'\n').count().max(1);
            if sample.len() / lines > 400 { Class::Skip("unknown format, one long line of data: not read") } else { Class::Doc("text") }
        }
        // Linguist's rule is for JavaScript and CSS only: hand-written scripts in other languages
        // may have long lines.
        Class::Code(lang) if matches!(lang, "javascript" | "css") && minified(bytes) => Class::Skip("minified file (generated): not read"),
        Class::Code(lang) if !matches!(lang, "rust" | "toml") && generated(bytes) => Class::Skip("generated file (marked in its first lines): not read"),
        _ => class,
    }
}

/// A generated-file mark in the first five lines: `Code generated ... DO NOT EDIT` (Go and many
/// generators), `<auto-generated` (.NET), `@generated`, or `DO NOT EDIT` next to `generated`.
fn generated(bytes: &[u8]) -> bool {
    let head = String::from_utf8_lossy(bytes.get(..bytes.len().min(2048)).unwrap_or(&[]));
    head.lines().take(5).any(|l| {
        let low = l.to_lowercase();
        low.contains("<auto-generated") || low.contains("@generated") || (low.contains("generated") && low.contains("do not edit"))
    })
}

/// Average line over 110 characters in the first 8 KB (the Linguist rule for generated files).
fn minified(bytes: &[u8]) -> bool {
    let sample = bytes.get(..bytes.len().min(8192)).unwrap_or(&[]);
    let lines = sample.iter().filter(|b| **b == b'\n').count().max(1);
    sample.len() > 1000 && sample.len() / lines > 110
}

/// Folders never read in any studied tree: build output, installed packages, caches.
pub const SKIP_DIRS: &[&str] = &[
    "target",
    "node_modules",
    "dist",
    "build",
    "out",
    "coverage",
    "__pycache__",
    "venv",
    "Pods",
    "__generated__",
    "vendor",
    "site-packages",
    "bower_components",
    "jspm_packages",
    "htmlcov",
];

pub(crate) fn check() -> Result<(), String> {
    let cases: &[(&str, Class)] = &[
        ("lib.rs", Class::Code("rust")),
        ("Cargo.toml", Class::Code("toml")),
        ("Cargo.lock", Class::Skip("lock file (generated): not read")),
        ("notes.md", Class::Doc("markdown")),
        ("ch2.txt", Class::Doc("text")),
        ("parts.csv", Class::Doc("csv")),
        ("app.ts", Class::Code("typescript")),
        ("tool.py", Class::Code("python")),
        ("attention.cu", Class::Code("cuda")),
        ("main.go", Class::Code("go")),
        ("build.gradle", Class::Code("code")),
        ("README", Class::Doc("text")),
        ("Makefile", Class::Code("code")),
        (".env.local", Class::Skip("secret store (.env): never read")),
        ("manual.pdf", Class::Doc("pdf")),
        ("photo.JPG", Class::Doc("general")),
        ("report.docx", Class::Doc("general")),
        ("model.safetensors", Class::Skip("model weights or binary data: not read")),
    ];
    for (name, want) in cases {
        let got = by_name(name, 10);
        if got != *want {
            return Err(format!("nmdoc.format: {name} is {got:?}, want {want:?}"));
        }
    }
    if by_name("big.md", MAX_TEXT) != Class::Skip("too large: over 512 KB") {
        return Err("nmdoc.format: size limit".into());
    }
    if by_bytes(Class::Doc("text"), b"%PDF-1.7\n") != Class::Doc("pdf")
        || by_bytes(Class::Doc("text?"), b"plain words\n") != Class::Doc("text")
        || by_bytes(Class::Doc("text"), b"ab\0\x01\x02cd\0\0\x03") != Class::Doc("general")
        || !matches!(by_bytes(Class::Code("typescript"), b"// Code generated by a tool. DO NOT EDIT.\nexport const a = 1;\n"), Class::Skip(_))
        || by_bytes(Class::Code("typescript"), b"export const a = 1;\n") != Class::Code("typescript")
    {
        return Err("nmdoc.format: magic bytes or binary".into());
    }
    Ok(())
}
