//! Small file helpers shared by the memory sub-components: append one JSON line, read JSON
//! lines, write and read one JSON file, inside the memory folder. Not a sub-component.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::io::Write as _;
use std::path::Path;

pub(crate) fn fault(e: impl std::fmt::Display) -> Fault {
    Fault::new("nmmem", FaultKind::Resource, e.to_string())
}
/// Append one canonical JSON line (append-only logs).
pub(crate) fn append(path: &Path, v: &Value) -> Result<(), Fault> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(fault)?;
    }
    let line = v.encode().map_err(fault)?;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(fault)?;
    writeln!(f, "{line}").map_err(fault)
}
/// An append-only log of JSON lines whose file stays open between appends: each append is one
/// write of the whole line on the open handle, with the same durability as opening, writing and
/// closing (the bytes reach the system's file cache at once; nothing waits for the disk), without
/// the cost of opening and closing the file each time. A failed write drops the handle and is
/// tried once more on a fresh one.
#[derive(Debug)]
pub(crate) struct Appender {
    path: std::path::PathBuf,
    file: Option<std::fs::File>,
}
impl Appender {
    pub(crate) fn new(path: std::path::PathBuf) -> Self {
        Self { path, file: None }
    }
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
    /// Append one canonical JSON line.
    pub(crate) fn append(&mut self, v: &Value) -> Result<(), Fault> {
        let mut line = v.encode().map_err(fault)?;
        line.push('\n');
        for attempt in 0..2 {
            if self.file.is_none() {
                if let Some(d) = self.path.parent() {
                    std::fs::create_dir_all(d).map_err(fault)?;
                }
                self.file = Some(std::fs::OpenOptions::new().create(true).append(true).open(&self.path).map_err(fault)?);
            }
            match self.file.as_mut().map(|f| f.write_all(line.as_bytes())) {
                Some(Ok(())) => return Ok(()),
                Some(Err(e)) if attempt == 1 => return Err(fault(e)),
                _ => self.file = None,
            }
        }
        Err(fault("append: no file"))
    }
}
/// Every parseable JSON line of a file (a missing file is empty).
pub(crate) fn lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path).map(|t| t.lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).collect()).unwrap_or_default()
}
/// True when `path` already holds exactly `bytes`. Reading a small file costs a fraction of a
/// millisecond; writing one costs several on a drive with a scanner on file close, and most
/// records a memory write touches are unchanged by it.
fn same(path: &Path, bytes: &[u8]) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.len() == bytes.len() as u64) && std::fs::read(path).is_ok_and(|b| b == bytes)
}
/// Write one readable JSON file, replacing the previous version of that same record; a file
/// that already holds these bytes is left as it is.
pub(crate) fn put(path: &Path, v: &Value) -> Result<(), Fault> {
    let text = nmjson::pretty::pretty(v);
    if same(path, text.as_bytes()) {
        return Ok(());
    }
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(fault)?;
    }
    std::fs::write(path, text).map_err(fault)
}
/// Write one file safely (2026-11-12, memory revision 2, section 6): the bytes go to a temporary
/// file in the same folder, are flushed to disk, and the temporary file is renamed over the target,
/// retried briefly when Windows holds the target open. A crash leaves either the old or the new
/// file, never a half-written one. A file that already holds these bytes is left as it is.
pub(crate) fn put_atomic(path: &Path, bytes: &[u8]) -> Result<(), Fault> {
    put_replacing(path, bytes, true)
}
/// A view derived from the journal (tier folders, index, undo list, snapshot), written as
/// `put_atomic` without waiting for the disk: the temporary file is renamed over the target, so a
/// reader never sees half a file, but after a crash of the machine the view may be stale or empty.
/// Nothing is lost by that: the journal is flushed to disk before any view is written, opening
/// the memory writes every view again from it, and the snapshot is used only when its sequence
/// number and hash match a journal event (else the journal is replayed). A file that already
/// holds these bytes is left as it is.
pub(crate) fn put_view(path: &Path, v: &Value) -> Result<(), Fault> {
    put_replacing(path, nmjson::pretty::pretty(v).as_bytes(), false)
}
fn put_replacing(path: &Path, bytes: &[u8], durable: bool) -> Result<(), Fault> {
    use std::io::Write as _;
    if same(path, bytes) {
        return Ok(());
    }
    let dir = path.parent().ok_or_else(|| fault("no parent folder"))?;
    std::fs::create_dir_all(dir).map_err(fault)?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = dir.join(format!(".{name}.tmp-{}", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp).map_err(fault)?;
        f.write_all(bytes).map_err(fault)?;
        if durable {
            f.sync_all().map_err(fault)?;
        }
    }
    let mut last = None;
    for attempt in 0..8_u32 {
        match std::fs::rename(&tmp, path) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last = Some(e);
                std::thread::park_timeout(std::time::Duration::from_millis(5 << attempt));
            }
        }
    }
    let _ = std::fs::remove_file(&tmp);
    Err(fault(format!("{}: {}", path.display(), last.map_or_else(String::new, |e| e.to_string()))))
}
/// `put` written safely.
pub(crate) fn put_safe(path: &Path, v: &Value) -> Result<(), Fault> {
    put_atomic(path, nmjson::pretty::pretty(v).as_bytes())
}
/// Read one JSON file, if present and valid.
pub(crate) fn get(path: &Path) -> Option<Value> {
    std::fs::read(path).ok().and_then(|b| nmjson::parse(&b).ok())
}
pub(crate) fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
pub(crate) fn u(n: u64) -> Value {
    Value::Number(n.to_string())
}
pub(crate) fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
pub(crate) fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).unwrap_or("").to_owned()
}
pub(crate) fn num(v: &Value, k: &str) -> u64 {
    v.get(k).and_then(Value::integer).unwrap_or(0)
}
pub(crate) fn strings(v: &Value, k: &str) -> Vec<String> {
    v.get(k).and_then(Value::array).map(|a| a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect()).unwrap_or_default()
}
