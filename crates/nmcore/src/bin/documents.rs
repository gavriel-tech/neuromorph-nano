//! `neuromorph-nano check documents`: the document fixtures of `tests/documents/` against their
//! hand-written expected results. Reader (format table, encodings, skip reasons), document model
//! (outlines, every line in exactly one block) and the structured formats (`formats/<format>/`).
use nmsubs::repo::{self, FileInfo, Skipped};
use std::path::{Path, PathBuf};

fn read_dir_files(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            read_dir_files(root, &p, out);
        } else {
            out.push(p.strip_prefix(root).unwrap_or(&p).to_string_lossy().replace('\\', "/"));
        }
    }
}

fn walk(root: &Path) -> (Vec<FileInfo>, Vec<Skipped>) {
    let (mut files, mut skipped) = (Vec::new(), Vec::new());
    repo::walk_counted(root, root, &[], &mut files, &mut skipped);
    (files, skipped)
}

/// The reader on `formats/encodings/`: every file read with its decoder and encoding and its text
/// equal to the hand text, or skipped with its reason; nothing else read or skipped.
fn encodings(base: &Path) -> Result<usize, String> {
    let dir = base.join("formats/encodings");
    let v = nmjson::parse(&std::fs::read(dir.join("expected.json")).map_err(|e| format!("encodings: {e}"))?).map_err(|e| format!("encodings: {e}"))?;
    let want_text = std::fs::read_to_string(dir.join("text.txt")).map_err(|e| format!("encodings: {e}"))?;
    let (files, skipped) = walk(&dir);
    let list = v.get("files").and_then(nmjson::Value::array).map_err(|e| format!("encodings: {e}"))?;
    let mut n = 0;
    let mut expected_paths = Vec::new();
    for e in list {
        let s = |k: &str| e.get(k).ok().and_then(|x| x.string().ok()).unwrap_or("").to_owned();
        let path = s("path");
        expected_paths.push(path.clone());
        if let Ok(reason) = e.get("skip").and_then(nmjson::Value::string) {
            let got = skipped.iter().find(|x| x.path == path).map(|x| x.reason);
            if got != Some(reason) {
                return Err(format!("encodings: {path} skipped as {got:?}, want {reason:?}"));
            }
        } else {
            let f = files.iter().find(|f| f.path == path).ok_or_else(|| format!("encodings: {path} not read"))?;
            if f.decoder != s("decoder") {
                return Err(format!("encodings: {path} read by {}, want {}", f.decoder, s("decoder")));
            }
            if f.general.unwrap_or("") != s("general") {
                return Err(format!("encodings: {path} read generally as {:?}, want {:?}", f.general, s("general")));
            }
            let bytes = std::fs::read(dir.join(&path)).map_err(|e| format!("encodings: {e}"))?;
            let enc = nmdoc::text::decode(&bytes).map(|x| x.1).unwrap_or("binary");
            if enc != s("encoding") {
                return Err(format!("encodings: {path} decoded as {enc}, want {}", s("encoding")));
            }
            if !matches!(e.get("own_text"), Ok(nmjson::Value::Bool(true))) && f.text != want_text {
                return Err(format!("encodings: {path} text {:?} differs from text.txt", f.text));
            }
        }
        n += 1;
    }
    let mut got: Vec<String> = files.iter().map(|f| f.path.clone()).chain(skipped.iter().map(|s| s.path.clone())).collect();
    got.sort();
    expected_paths.sort();
    if got != expected_paths {
        return Err(format!("encodings: the reader saw {got:?}, expected {expected_paths:?}"));
    }
    println!("ok   documents.reader: {n} encoding and skip cases right, file list exact");
    Ok(n)
}

/// The reader on the corpus: every file on disk read, none skipped.
fn corpus(base: &Path) -> Result<(usize, Vec<FileInfo>), String> {
    let dir = base.join("corpus");
    let (files, skipped) = walk(&dir);
    if let Some(s) = skipped.first() {
        return Err(format!("corpus: {} skipped ({}) of {}", s.path, s.reason, skipped.len()));
    }
    let mut on_disk = Vec::new();
    read_dir_files(&dir, &dir, &mut on_disk);
    on_disk.sort();
    let got: Vec<String> = files.iter().map(|f| f.path.clone()).collect();
    if got != on_disk {
        return Err(format!("corpus: read {} files, {} on disk", got.len(), on_disk.len()));
    }
    let mut by: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for f in &files {
        *by.entry(f.decoder).or_default() += 1;
    }
    let by: Vec<String> = by.iter().map(|(d, c)| format!("{d} {c}")).collect();
    println!("ok   documents.corpus: {} files read, none skipped ({})", files.len(), by.join(", "));
    Ok((files.len(), files))
}

/// Outlines against `expected/outline/` and every line of every file in exactly one block. The
/// blocks of code files come from the substrate (symbols), so the corpus is indexed first.
fn model(base: &Path, files: &[FileInfo]) -> Result<usize, String> {
    let sub = nmsubs::Substrate::scan(&base.join("corpus"), &[], std::sync::Arc::new(|_: &str| Vec::new()));
    let mut n = 0;
    for f in &sub.files {
        nmtype::doc::check_cover(&f.blocks, f.text.lines().count()).map_err(|e| format!("model: {}: {e}", f.path))?;
        n += 1;
    }
    if sub.files.len() != files.len() {
        return Err(format!("model: the substrate read {} files, the reader {}", sub.files.len(), files.len()));
    }
    let dir = base.join("expected/outline");
    let mut names = Vec::new();
    read_dir_files(&dir, &dir, &mut names);
    names.retain(|x| x.ends_with(".tsv"));
    names.sort();
    let mut headings = 0;
    for name in &names {
        let path = name.trim_end_matches(".tsv").replace("__", "/");
        let f = sub.files.iter().find(|f| f.path == path).ok_or_else(|| format!("model: {path} not read"))?;
        let want: Vec<(usize, u8, String)> = std::fs::read_to_string(dir.join(name))
            .map_err(|e| format!("model: {e}"))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| {
                let mut c = l.splitn(3, '\t');
                Some((c.next()?.trim().parse().ok()?, c.next()?.trim().parse().ok()?, c.next()?.trim().to_owned()))
            })
            .collect();
        let got = nmdoc::md::outline(&f.blocks);
        if got != want {
            let first = got.iter().zip(&want).find(|(a, b)| a != b).map(|(a, b)| format!("got {a:?} want {b:?}")).unwrap_or_else(|| {
                format!("got {} headings, want {}", got.len(), want.len())
            });
            return Err(format!("model: outline of {path}: {first}"));
        }
        headings += want.len();
    }
    println!("ok   documents.model: every line of {n} files in exactly one block; {headings} headings of {} outlines equal", names.len());
    Ok(n + names.len())
}

/// Text with white space normalised: lines trimmed, runs of space collapsed, blank lines dropped.
fn normal(t: &str) -> Vec<String> {
    t.lines().map(nmdoc::markup::collapse).filter(|l| !l.is_empty()).collect()
}

fn tsv(path: &Path) -> Result<Vec<Vec<String>>, String> {
    Ok(std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').map(str::to_owned).collect())
        .collect())
}

/// Each `formats/<format>/` folder: its one input read as the reader reads it, against the hand
/// files there: `expected.txt` (derived text after white space normalising), `expected-outline.tsv`
/// (line, level, heading), `expected-blocks.tsv` (first line, kind, label of every block) and
/// `expected-events.tsv` (first line, last line, label, time in milliseconds of every log event).
fn formats(base: &Path) -> Result<usize, String> {
    let dir = base.join("formats");
    let mut names: Vec<String> = std::fs::read_dir(&dir).map_err(|e| format!("formats: {e}"))?.flatten().filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    names.retain(|n| n != "encodings");
    names.sort();
    let mut n = 0;
    for name in &names {
        let fdir = dir.join(name);
        let mut inputs: Vec<PathBuf> = std::fs::read_dir(&fdir)
            .map_err(|e| format!("formats: {e}"))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.file_name().is_some_and(|f| { let f = f.to_string_lossy(); !f.starts_with("expected") && f != "README.md" }))
            .collect();
        inputs.sort();
        let [input] = inputs.as_slice() else { return Err(format!("formats/{name}: want one input file, found {}", inputs.len())) };
        let file = input.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        let bytes = std::fs::read(input).map_err(|e| format!("formats/{name}: {e}"))?;
        let class = nmdoc::format::by_name(&file, bytes.len() as u64);
        let doc = nmdoc::decode(&file, class, &bytes).map_err(|e| format!("formats/{name}: {file} not read: {e}"))?;
        nmtype::doc::check_cover(&doc.blocks, doc.text.lines().count()).map_err(|e| format!("formats/{name}: {e}"))?;
        let mut what = vec![doc.decoder.to_owned()];
        if let Ok(want) = std::fs::read_to_string(fdir.join("expected.txt")) {
            let (g, w) = (normal(&doc.text), normal(&want));
            if g != w {
                let at = g.iter().zip(&w).position(|(a, b)| a != b).unwrap_or(g.len().min(w.len()));
                return Err(format!(
                    "formats/{name}: text differs at line {} (normalised): got {:?} want {:?}",
                    at + 1,
                    g.get(at).map_or("", String::as_str),
                    w.get(at).map_or("", String::as_str)
                ));
            }
            what.push(format!("text {} lines", w.len()));
        }
        let p = fdir.join("expected-outline.tsv");
        if p.is_file() {
            let want: Vec<(usize, u8, String)> =
                tsv(&p)?.iter().filter_map(|r| Some((r.first()?.parse().ok()?, r.get(1)?.parse().ok()?, r.get(2)?.clone()))).collect();
            let got = nmdoc::md::outline(&doc.blocks);
            if got != want {
                return Err(format!("formats/{name}: outline {got:?}, want {want:?}"));
            }
            what.push(format!("outline {}", want.len()));
        }
        let p = fdir.join("expected-blocks.tsv");
        if p.is_file() {
            let want: Vec<(usize, String, String)> =
                tsv(&p)?.iter().filter_map(|r| Some((r.first()?.parse().ok()?, r.get(1)?.clone(), r.get(2).cloned().unwrap_or_default()))).collect();
            let got: Vec<(usize, String, String)> = doc.blocks.iter().map(|b| (b.first, b.kind.name().to_owned(), b.label.clone())).collect();
            if got != want {
                return Err(format!("formats/{name}: blocks {got:?}, want {want:?}"));
            }
            what.push(format!("blocks {}", want.len()));
        }
        let p = fdir.join("expected-events.tsv");
        if p.is_file() {
            let want: Vec<(usize, usize, String, u64)> = tsv(&p)?
                .iter()
                .filter_map(|r| Some((r.first()?.parse().ok()?, r.get(1)?.parse().ok()?, r.get(2)?.clone(), r.get(3)?.parse().ok()?)))
                .collect();
            let got: Vec<(usize, usize, String, u64)> = doc
                .blocks
                .iter()
                .filter(|b| b.kind == nmtype::doc::BlockKind::LogEvent)
                .map(|b| (b.first, b.last, b.label.clone(), b.loc.time.map_or(0, |t| t.0)))
                .collect();
            if got != want {
                return Err(format!("formats/{name}: events {got:?}, want {want:?}"));
            }
            what.push(format!("events {}", want.len()));
        }
        if what.len() < 2 {
            return Err(format!("formats/{name}: no expected file"));
        }
        println!("ok   documents.formats.{name}: {file} ({})", what.join(", "));
        n += 1;
    }
    Ok(n)
}

/// Inflate on `inflate/`: every stream in `compressed/` (deflate, zlib, gzip, made by Windows'
/// own Deflate at each level) inflates to its original in `original/`, byte for byte; every file
/// in `corrupt/` gives the named fault `expected/inflate.tsv` lists, never a panic; the bombs in
/// `bomb/` trip the guard.
fn inflate(base: &Path) -> Result<usize, String> {
    use nmdoc::inflate::{self as inf, Limits};
    let dir = base.join("inflate");
    let lim = Limits::default();
    let mut names = Vec::new();
    read_dir_files(&dir.join("compressed"), &dir.join("compressed"), &mut names);
    names.sort();
    for name in &names {
        let bytes = std::fs::read(dir.join("compressed").join(name)).map_err(|e| format!("inflate: {name}: {e}"))?;
        let orig_name = name.split('.').take(2).collect::<Vec<_>>().join(".");
        let want = std::fs::read(dir.join("original").join(&orig_name)).map_err(|e| format!("inflate: {orig_name}: {e}"))?;
        let got = if name.ends_with(".zlib") {
            inf::zlib(&bytes, &lim)
        } else if name.ends_with(".gz") {
            inf::gzip(&bytes, &lim).map(|g| g.data)
        } else {
            inf::inflate(&bytes, &lim)
        };
        match got {
            Ok(g) if g == want => {}
            Ok(g) => return Err(format!("inflate: {name} gave {} bytes, not equal to {orig_name} ({} bytes)", g.len(), want.len())),
            Err(f) => return Err(format!("inflate: {name}: {}", f.name())),
        }
    }
    let table = tsv(&base.join("expected/inflate.tsv"))?;
    for row in &table {
        let (Some(file), Some(want)) = (row.first(), row.get(1)) else { continue };
        let bytes = std::fs::read(dir.join(file)).map_err(|e| format!("inflate: {file}: {e}"))?;
        let got: String = if file.ends_with(".zip") {
            let items = nmdoc::read("zeros.zip", nmdoc::Class::Doc("general"), &bytes);
            items.iter().find_map(|i| i.as_ref().err().map(|(_, why)| (*why).to_owned())).unwrap_or_else(|| "read".into())
        } else {
            let r = if file.ends_with(".zlib") {
                inf::zlib(&bytes, &lim)
            } else if file.ends_with(".gz") {
                inf::gzip(&bytes, &lim).map(|g| g.data)
            } else {
                inf::inflate(&bytes, &lim)
            };
            r.map_or_else(|f| f.name().to_owned(), |_| "inflated".to_owned())
        };
        if got != *want {
            return Err(format!("inflate: {file} gave {got:?}, want {want:?}"));
        }
    }
    println!("ok   documents.inflate: {} streams equal to their originals; {} corrupt or bomb files give their named faults", names.len(), table.len());
    Ok(names.len() + table.len())
}

/// General reading on `general/`: containers, binaries and their members, each read (with its
/// decoder and mark) or skipped (with its reason) as `expected/general.tsv` lists, nothing else.
fn general(base: &Path) -> Result<usize, String> {
    let (files, skipped) = walk(&base.join("general"));
    let mut got: Vec<String> = files
        .iter()
        .map(|f| format!("{}\t{}\t{}", f.path, f.decoder, f.general.unwrap_or("-")))
        .chain(skipped.iter().map(|s| format!("{}\tskip\t{}", s.path, s.reason)))
        .collect();
    got.sort();
    let mut want: Vec<String> = tsv(&base.join("expected/general.tsv"))?.iter().map(|r| r.join("\t")).collect();
    want.sort();
    if got != want {
        let extra: Vec<&String> = got.iter().filter(|g| !want.contains(g)).collect();
        let missing: Vec<&String> = want.iter().filter(|w| !got.contains(w)).collect();
        return Err(format!("general: read but not expected {extra:?}; expected but not read {missing:?}"));
    }
    for f in &files {
        nmtype::doc::check_cover(&f.blocks, f.text.lines().count()).map_err(|e| format!("general: {}: {e}", f.path))?;
    }
    println!("ok   documents.general: {} documents read generally and {} skipped as listed", files.len(), skipped.len());
    Ok(got.len())
}

/// All document fixture checks; the count of cases.
pub fn check(home: &Path) -> Result<usize, String> {
    let base: PathBuf = home.join("tests/documents");
    if !base.is_dir() {
        return Err(format!("documents: {} is missing", base.display()));
    }
    let mut n = encodings(&base)?;
    let (c, files) = corpus(&base)?;
    n += c;
    n += model(&base, &files)?;
    n += formats(&base)?;
    n += inflate(&base)?;
    n += general(&base)?;
    n += nmdoc::pdf::fixtures::check(&base)?;
    Ok(n)
}
