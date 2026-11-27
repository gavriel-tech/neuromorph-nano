//! Hard-drive storage of the Neuromorph Nano, by category, with its controls.
//!
//! Roadmap Phase 4.5 (2026-11-24), item 2. One place measures what the Nano keeps on disk, with
//! exact absolute paths, sizes and times, in four categories:
//!
//! - Memory & World Model: per memory folder (`<home>/memory` for the Nano itself, and
//!   `<home>/memory/projects/<key>` for each studied project, `nmtype::project::memory_dir`), the
//!   world-model notes (`world-model/`, the older `writer/`), the training examples (`training/`,
//!   `nmtype::examples::DIR`; `examples/` too if present), the episode, decision, evidence and
//!   other logs, the memory tiers and journal; and, for the Nano itself, the audit log, the
//!   machine measures (`machine.json`, `nmengn::capacity`) and the G3 muscle store (`<home>/muscle`
//!   by `designs/muscle.json`).
//! - Dense Vectors & Index Caches: the dense store (`dense/`, `nmsubs::dense::store`), the meaning
//!   floors and rerank-pair cache (`meaning/`) and the substrate index caches
//!   (`substrate-cache-*.bin`). The auxiliary note vectors live in RAM only and are seeded again from
//!   the notes at start (`writer_idle::seed_aux`), so nothing of theirs is on disk.
//! - Compression & Quantized Copies: the 8-bit learned-unit weights (`units/`, `.q8` files), the
//!   affine compensation cartridges of the compressed bases (`library/cartridges/learned/`), the
//!   per-card precision cartridges (`library/cartridges/hardware/`), and rebuildable scratch and
//!   cache files: build and step-check folders under `scratch/` and temporary files left in memory
//!   folders. The `bmq8`, `q8` and `q4` copies of the base MicroModels are made on the card at load
//!   and are not kept on disk.
//! - Parameter Cartridges & Weights: base cartridges (`library/cartridges/base/`, read-only),
//!   parameter maps and job filters (`library/parameter-maps/`), project lenses and judge read-outs
//!   (`library/cartridges/adapter/`), preference cartridges (`library/cartridges/preference/`) and
//!   the released open weights (`nmtype::models_dir`, `weights/` by default, read-only).
//!
//! Each item is rebuildable (made again by the Nano when missing), removable (a learned or derived
//! piece the user may delete) or read-only (kept, or cleared only through its own control).
//! Controls: prune a category's rebuildable items, delete one removable item, read the storage
//! paths and limits. Prune and delete need `confirm: true`, refuse anything outside the Nano's
//! folders, and are written to the audit log. The code has no recoverable trash convention (the
//! interface's job-filter delete removes the file), so a delete removes the item for good.
//! File contents are never read here: the view holds paths, sizes and times only (no sensory
//! material crosses into these behavioural reports, `nmtype::class`).
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

/// The categories: (id, name).
pub const CATEGORIES: [(&str, &str); 5] = [
    ("memory", "Memory & World Model"),
    ("vectors", "Dense Vectors & Index Caches"),
    ("compression", "Compression & Quantized Copies"),
    ("cartridges", "Parameter Cartridges & Weights"),
    // 2026-11-26 (Luis Jake Gabriel III: "in the app we should be able to manage and delete junk
    // files produced"): test leftovers, old experiment builds, stale copies; each kind is listed in the junk register.
    ("junk", "Junk: test leftovers and stale copies"),
];

/// The control holding the disk floor in GB. It lives in `settings.json` and is written only
/// through the program's controls (`set_by`, with its audit row), never here.
pub const FLOOR_KEY: &str = "storage.disk_floor_gb";
/// The disk floor when none is set, in GB.
pub const FLOOR_DEFAULT_GB: u64 = 4;
const FLOOR_MAX_GB: u64 = 1000;
/// A rebuildable item changed within this many milliseconds may be in use, and a prune skips it.
const IN_USE_MS: u64 = 10 * 60 * 1000;
const GB: u64 = 1 << 30;

/// What may be done with an item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Made again by the Nano when missing: a prune removes it.
    Rebuildable,
    /// A learned or derived piece the user may delete, one at a time, with confirmation.
    Removable,
    /// Kept: shown only, or cleared through its own control.
    ReadOnly,
    /// Left behind by tests, experiments or stopped runs, used by nothing: a prune of the junk
    /// group removes it, or one at a time (2026-11-26).
    Junk,
}
impl Class {
    /// The name the interface and the tools show.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rebuildable => "rebuildable",
            Self::Removable => "removable",
            Self::ReadOnly => "read-only",
            Self::Junk => "junk",
        }
    }
}

/// One file or folder on disk.
#[derive(Clone, Debug)]
pub struct Item {
    pub category: &'static str,
    pub path: PathBuf,
    pub label: String,
    /// `nano` for the Nano's own memory, a project key, or empty for shared folders.
    pub project: String,
    pub class: Class,
    pub folder: bool,
    pub bytes: u64,
    pub files: u64,
    pub modified_ms: u64,
}

fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
fn n(x: u64) -> Value {
    Value::Number(x.to_string())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn mtime_ms(m: &std::fs::Metadata) -> u64 {
    m.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}
fn is_tmp(p: &Path) -> bool {
    p.extension().is_some_and(|e| e.eq_ignore_ascii_case("tmp"))
}

/// Bytes, files and the newest change (ms) under `path`, links not followed; files `skip` names
/// are left out and returned apart. The newest change is the newest file's, or the folder's own
/// when it holds no file.
fn measure(path: &Path, skip: fn(&Path) -> bool) -> ((u64, u64, u64), Vec<PathBuf>) {
    let mut acc = (0_u64, 0_u64, 0_u64);
    let mut folder_ms = 0_u64;
    let mut skipped = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(p) = stack.pop() {
        let Ok(meta) = std::fs::symlink_metadata(&p) else { continue };
        if meta.is_dir() {
            folder_ms = folder_ms.max(mtime_ms(&meta));
            if let Ok(rd) = std::fs::read_dir(&p) {
                stack.extend(rd.flatten().map(|e| e.path()));
            }
        } else if meta.is_file() {
            if skip(&p) {
                skipped.push(p);
                continue;
            }
            acc.0 += meta.len();
            acc.1 += 1;
            acc.2 = acc.2.max(mtime_ms(&meta));
        }
    }
    if acc.1 == 0 {
        acc.2 = folder_ms;
    }
    (acc, skipped)
}

/// Sorted entries of a folder: (name, path, is a folder).
fn entries(dir: &Path) -> Vec<(String, PathBuf, bool)> {
    let mut v: Vec<(String, PathBuf, bool)> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| {
                    let p = e.path();
                    let folder = std::fs::symlink_metadata(&p).is_ok_and(|m| m.is_dir());
                    (e.file_name().to_string_lossy().into_owned(), p, folder)
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

struct Scan {
    items: Vec<Item>,
}
impl Scan {
    /// Adds `path` if it exists. Temporary files inside a folder that is not itself rebuildable
    /// become rebuildable compression items of their own.
    fn add(&mut self, category: &'static str, path: PathBuf, label: &str, project: &str, class: Class) {
        let Ok(meta) = std::fs::symlink_metadata(&path) else { return };
        let folder = meta.is_dir();
        let skip: fn(&Path) -> bool = if folder && class != Class::Rebuildable { is_tmp } else { |_| false };
        let ((bytes, files, modified_ms), tmps) = measure(&path, skip);
        self.items.push(Item { category, path, label: label.to_owned(), project: project.to_owned(), class, folder, bytes, files, modified_ms });
        for t in tmps {
            self.add("compression", t, "temporary file left by a stopped write", project, Class::Rebuildable);
        }
    }
}

/// The memory folders: the Nano's own, then each project's.
#[must_use]
pub fn memory_roots(home: &Path) -> Vec<(String, PathBuf)> {
    let mut v = vec![("nano".to_owned(), home.join("memory"))];
    for (name, path, folder) in entries(&nmtype::project::projects_dir(home)) {
        if folder {
            v.push((name, path));
        }
    }
    v
}

/// Where an entry at the top of a memory folder belongs: (category, label, class).
fn memory_entry(name: &str, folder: bool) -> (&'static str, &'static str, Class) {
    match name {
        "dense" => ("vectors", "dense vectors kept on disk", Class::Rebuildable),
        "meaning" => ("vectors", "meaning floors and rerank-pair cache", Class::Rebuildable),
        x if x.starts_with("substrate-cache-") => ("vectors", "substrate index cache", Class::Rebuildable),
        x if !folder && is_tmp(Path::new(x)) => ("compression", "temporary file left by a stopped write", Class::Rebuildable),
        "training" | "examples" => ("memory", "training examples (cleared by the examples control)", Class::ReadOnly),
        "episodes.jsonl" => ("memory", "episode log", Class::ReadOnly),
        "decisions.jsonl" => ("memory", "decision log", Class::ReadOnly),
        "audit.jsonl" => ("memory", "audit log", Class::ReadOnly),
        "machine.json" => ("memory", "this machine's measures", Class::ReadOnly),
        "evidence.jsonl" => ("memory", "evidence ledger (cleared by the ledger control)", Class::ReadOnly),
        "1-core" | "2-active" | "3-archive" => ("memory", "memory tier", Class::ReadOnly),
        "journal" => ("memory", "memory journal", Class::ReadOnly),
        "live" => ("memory", "live server records", Class::ReadOnly),
        x if x.ends_with(".jsonl") || x.ends_with(".pruned") => ("memory", "log", Class::ReadOnly),
        _ => ("memory", "memory record", Class::ReadOnly),
    }
}

fn scan_memory(sc: &mut Scan, project: &str, root: &Path) {
    for (name, path, folder) in entries(root) {
        // `projects` is scanned per project; `stacks` (this machine's stack state) is listed with the
        // cartridge library it belongs to.
        if project == "nano" && (name == "projects" || name == "stacks") {
            continue;
        }
        if folder && (name == "world-model" || name == "writer") {
            let label = if name == "world-model" { "world-model notes (rewritten at idle)" } else { "older writer notes (rewritten at idle)" };
            for (_, f, sub_folder) in entries(&path) {
                if sub_folder || !is_tmp(&f) {
                    sc.add("memory", f, label, project, Class::Removable);
                } else {
                    sc.add("compression", f, "temporary file left by a stopped write", project, Class::Rebuildable);
                }
            }
            continue;
        }
        let (cat, label, class) = memory_entry(&name, folder);
        sc.add(cat, path, label, project, class);
    }
}

fn scan_scratch(sc: &mut Scan, home: &Path) {
    // The Nano's own rebuildable scratch: the build folder of `nm_build_run`, the step-check
    // copies and their build folder (`build.rs`), and the job-filter build's temporary folder.
    for (name, path, folder) in entries(&home.join("scratch")) {
        let label = match name.as_str() {
            "build-target" => "build folder of nm_build_run",
            "stepcheck-target" => "build folder of nm_stepcheck",
            "tmp" => "temporary folder of job-filter builds",
            x if folder && x.strip_prefix("stepcheck-").is_some_and(|ms| ms.chars().all(|c| c.is_ascii_digit())) => "step-check copy",
            _ => continue,
        };
        sc.add("compression", path, label, "", Class::Rebuildable);
    }
}

/// A file or folder older than `days` days (by its newest change).
fn older_than(path: &Path, days: u64) -> bool {
    let (_, _, modified) = measure(path, |_| false).0;
    nmtype::now_ms().saturating_sub(modified) > days * 86_400_000
}

/// Junk (2026-11-26): what tests, experiments and stopped runs leave behind and nothing uses. Each
/// kind is in the junk register with who makes it.
fn scan_junk(sc: &mut Scan, home: &Path) {
    let scratch = home.join("scratch");
    for (_, path, _) in entries(&scratch.join("library-test")) {
        sc.add("junk", path, "library test leftover", "", Class::Junk);
    }
    for (name, path, folder) in entries(&scratch) {
        let label = match name.as_str() {
            x if folder && x.starts_with("soak-") => "soak copy left behind",
            "dist-dryrun" => "release packager dry run",
            _ => continue,
        };
        sc.add("junk", path, label, "", Class::Junk);
    }
    sc.add("junk", scratch.join("target-claude").join("debug"), "debug build", "", Class::Junk);
    // Old experiment builds: executables in the lane folders, a day or more old.
    for (name, lane, folder) in entries(&scratch) {
        if !folder || !name.starts_with("lane-") {
            continue;
        }
        for (_, f, sub) in entries(&lane) {
            if !sub && f.extension().is_some_and(|e| e.eq_ignore_ascii_case("exe")) && older_than(&f, 1) {
                sc.add("junk", f, "old experiment build", "", Class::Junk);
            }
        }
    }
    // Bench logs a week old.
    for (_, f, sub) in entries(&scratch.join("stage-bench-tmp")) {
        if !sub && f.extension().is_some_and(|e| e.eq_ignore_ascii_case("log")) && older_than(&f, 7) {
            sc.add("junk", f, "old bench log", "", Class::Junk);
        }
    }
    for (name, path, folder) in entries(&home.join("dist")) {
        if folder && name.starts_with(".stage-") {
            sc.add("junk", path, "release packager staging left behind", "", Class::Junk);
        }
    }
    for (name, path, folder) in entries(&home.join("tests").join("mcp-launcher")) {
        if folder && name.starts_with("soak-") {
            sc.add("junk", path, "soak test memory left behind", "", Class::Junk);
        }
    }
}

fn scan_library(sc: &mut Scan, home: &Path) {
    let lib = home.join("library");
    let carts = lib.join("cartridges");
    for (_, path, _) in entries(&carts.join("base")) {
        sc.add("cartridges", path, "base cartridge (read-only)", "", Class::ReadOnly);
    }
    // Lens versions and judge read-outs, one removable row per version file.
    for (name, path, folder) in entries(&carts.join("adapter")) {
        let label = if name.starts_with("judge-") { "judge read-out version" } else { "project lens version" };
        if !folder {
            sc.add("cartridges", path, label, "", Class::Removable);
            continue;
        }
        for (_, f, _) in entries(&path) {
            let removable = f.extension().is_some_and(|e| e == "cartridge");
            sc.add("cartridges", f, if removable { label } else { "adapter record" }, "", if removable { Class::Removable } else { Class::ReadOnly });
        }
    }
    for (_, path, folder) in entries(&carts.join("preference")) {
        let files = if folder { entries(&path) } else { vec![(String::new(), path, false)] };
        for (_, f, _) in files {
            sc.add("cartridges", f, "preference cartridge", "", Class::Removable);
        }
    }
    for (_, path, _) in entries(&carts.join("learned")) {
        sc.add("compression", path, "affine compensation of a compressed base", "", Class::ReadOnly);
    }
    for (_, path, _) in entries(&carts.join("hardware")) {
        sc.add("compression", path, "precision and compression settings for a card", "", Class::ReadOnly);
    }
    // Parameter maps: each job filter is removable; the other maps of a role stay together.
    for (_, role_dir, folder) in entries(&lib.join("parameter-maps")) {
        if !folder {
            sc.add("cartridges", role_dir, "parameter map", "", Class::ReadOnly);
            continue;
        }
        for (name, f, sub) in entries(&role_dir) {
            let filter = !sub && name.starts_with("job-filter-") && name.ends_with(".json");
            sc.add("cartridges", f, if filter { "job filter" } else { "parameter map" }, "", if filter { Class::Removable } else { Class::ReadOnly });
        }
    }
    for name in [nmlearn::library::CATALOGUE, nmlearn::library::RELEASE_CATALOGUE, nmlearn::library::RELEASE_FILES] {
        sc.add("cartridges", carts.join(name), "cartridge library catalogue", "", Class::ReadOnly);
    }
    for f in [nmlearn::library::in_use_file(home), nmlearn::library::machine_file(home)] {
        sc.add("cartridges", f, "this machine's stack state", "", Class::ReadOnly);
    }
    for (_, path, _) in entries(&home.join("units")) {
        sc.add("compression", path, "learned unit (8-bit weights, managed by nm_unit)", "", Class::ReadOnly);
    }
    for (_, path, _) in entries(&nmtype::models_dir(home)) {
        sc.add("cartridges", path, "released open weights (read-only)", "", Class::ReadOnly);
    }
}

fn absolute(home: &Path) -> PathBuf {
    std::path::absolute(home).unwrap_or_else(|_| home.to_path_buf())
}

/// Every item on disk, in category order of discovery.
#[must_use]
pub fn items(home: &Path) -> Vec<Item> {
    let home = absolute(home);
    let mut sc = Scan { items: Vec::new() };
    for (project, root) in memory_roots(&home) {
        scan_memory(&mut sc, &project, &root);
    }
    let muscle = home.join(nmmusc::Policy::load().store_dir);
    sc.add("memory", muscle, "G3 muscle store", "", Class::ReadOnly);
    scan_scratch(&mut sc, &home);
    scan_library(&mut sc, &home);
    scan_junk(&mut sc, &home);
    sc.items
}

fn item_value(i: &Item) -> Value {
    obj(vec![
        ("path", s(&i.path.display().to_string())),
        ("label", s(&i.label)),
        ("project", s(&i.project)),
        ("class", s(i.class.name())),
        ("kind", s(if i.folder { "folder" } else { "file" })),
        ("bytes", n(i.bytes)),
        ("files", n(i.files)),
        ("modified_ms", n(i.modified_ms)),
    ])
}

/// `GET /api/storage` and `nm_storage`: every category with its total bytes and files, the bytes
/// a prune would free and the bytes removable one by one, and its items (largest first), with the
/// disk's free space against the floor.
#[must_use]
pub fn view(home: &Path) -> Value {
    let all = items(home);
    let mut cats = Vec::new();
    let (mut total, mut files) = (0_u64, 0_u64);
    for (id, name) in CATEGORIES {
        let mut mine: Vec<&Item> = all.iter().filter(|i| i.category == id).collect();
        mine.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.path.cmp(&b.path)));
        let sum = |f: &dyn Fn(&Item) -> bool| mine.iter().filter(|i| f(i)).map(|i| i.bytes).sum::<u64>();
        let bytes = sum(&|_| true);
        let count: u64 = mine.iter().map(|i| i.files).sum();
        total += bytes;
        files += count;
        let mut o = vec![
            ("id", s(id)),
            ("name", s(name)),
            ("bytes", n(bytes)),
            ("files", n(count)),
            // What a prune of the group frees: rebuildable items, and junk in the junk group.
            ("rebuildable_bytes", n(sum(&|i| matches!(i.class, Class::Rebuildable | Class::Junk)))),
            ("removable_bytes", n(sum(&|i| i.class == Class::Removable))),
            ("read_only_bytes", n(sum(&|i| i.class == Class::ReadOnly))),
            ("items", Value::Array(mine.iter().map(|i| item_value(i)).collect())),
        ];
        if id == "vectors" {
            o.push(("note", s("Auxiliary note vectors are held in RAM and seeded again from the world-model notes at start; none are kept on disk.")));
        } else if id == "compression" {
            o.push(("note", s("The bmq8, q8 and q4 copies of the base MicroModels are made on the card at load and are not kept on disk.")));
        }
        cats.push(obj(o));
    }
    let home_abs = absolute(home);
    let mut out = vec![
        ("home", s(&home_abs.display().to_string())),
        ("time_ms", n(nmtype::now_ms())),
        ("total_bytes", n(total)),
        ("total_files", n(files)),
        ("categories", Value::Array(cats)),
    ];
    out.extend(disk_fields(&home_abs));
    obj(out)
}

// ---------------------------------------------------------------------------------------------
// Free space and the disk floor.

/// Free and total bytes of the drive holding `path`: from Windows itself (`GetDiskFreeSpaceExW`),
/// falling back to the system's own tool (`fsutil` on Windows, `df` elsewhere); none when it
/// cannot be read.
#[must_use]
pub fn disk_space(path: &Path) -> Option<(u64, u64)> {
    disk_space_os(path)
}

#[cfg(windows)]
fn disk_space_os(path: &Path) -> Option<(u64, u64)> {
    use std::os::windows::process::CommandExt as _;
    let drive = path.components().find_map(|c| match c {
        Component::Prefix(p) => match p.kind() {
            std::path::Prefix::Disk(b) | std::path::Prefix::VerbatimDisk(b) => Some(format!("{}:", char::from(b))),
            _ => None,
        },
        _ => None,
    })?;
    // 2026-11-26: from Windows itself (nmffi, GetDiskFreeSpaceExW); the fsutil program below is
    // kept only as the fallback when that call cannot read the volume.
    if let Some(v) = nmffi::dynlib::disk_space(path) {
        return Some(v);
    }
    // CREATE_NO_WINDOW: no console window flashes when the interface asks.
    let out = std::process::Command::new("fsutil").args(["volume", "diskfree", &drive]).creation_flags(0x0800_0000).output().ok().filter(|o| o.status.success())?;
    parse_fsutil(&String::from_utf8_lossy(&out.stdout))
}

#[cfg(not(windows))]
fn disk_space_os(path: &Path) -> Option<(u64, u64)> {
    let out = std::process::Command::new("df").arg("-Pk").arg(path).output().ok().filter(|o| o.status.success())?;
    let text = String::from_utf8_lossy(&out.stdout);
    let cols: Vec<&str> = text.lines().nth(1)?.split_whitespace().collect();
    let total = cols.get(1)?.parse::<u64>().ok()?.saturating_mul(1024);
    let free = cols.get(3)?.parse::<u64>().ok()?.saturating_mul(1024);
    Some((free, total))
}

/// `fsutil volume diskfree` output: the first line is the free bytes, the second the total; the
/// number sits between the colon and an optional bracketed size, with digit group marks.
fn parse_fsutil(text: &str) -> Option<(u64, u64)> {
    let mut nums = text.lines().filter_map(|l| {
        let (_, rest) = l.split_once(':')?;
        let digits: String = rest.split('(').next()?.chars().filter(char::is_ascii_digit).collect();
        digits.parse::<u64>().ok()
    });
    let free = nums.next()?;
    let total = nums.next()?;
    Some((free, total))
}

/// The disk floor in GB (`storage.disk_floor_gb` in `settings.json`, default 4).
#[must_use]
pub fn floor_gb(home: &Path) -> u64 {
    crate::whole::control_int(home, FLOOR_KEY, 1, FLOOR_MAX_GB).unwrap_or(FLOOR_DEFAULT_GB)
}

fn disk_fields(home: &Path) -> Vec<(&'static str, Value)> {
    let floor = floor_gb(home);
    let space = disk_space(home);
    vec![
        (
            "disk",
            space.map_or(Value::Null, |(free, total)| obj(vec![("path", s(&home.display().to_string())), ("free_bytes", n(free)), ("total_bytes", n(total))])),
        ),
        ("disk_floor_gb", n(floor)),
        ("disk_floor_bytes", n(floor.saturating_mul(GB))),
        ("below_floor", Value::Bool(space.is_some_and(|(free, _)| free < floor.saturating_mul(GB)))),
    ]
}

/// Refuses work that writes much to disk while the drive's free space is under the disk floor.
/// Unknown free space lets the work go on.
/// # Errors
/// The free space is under the floor.
pub fn disk_guard(home: &Path) -> Result<(), String> {
    let floor = floor_gb(home);
    match disk_space(home) {
        Some((free, _)) if free < floor.saturating_mul(GB) => Err(format!(
            "free space on the Nano's drive is {:.1} GB, under the disk floor of {floor} GB ({FLOOR_KEY}); prune rebuildable storage or lower the floor",
            free as f64 / GB as f64
        )),
        _ => Ok(()),
    }
}

/// `GET /api/storage/settings`: the storage paths and limits. The memory folder is fixed by the
/// Nano's layout (`<home>/memory`, a project's under `memory/projects/`); one server may be given
/// another with `serve --memory-dir`. The disk floor is the one storage limit kept in the
/// controls, set through `POST /api/storage/settings` (the program's `set_by`).
#[must_use]
pub fn settings(home: &Path) -> Value {
    let home = absolute(home);
    let roots: Vec<Value> = memory_roots(&home)
        .into_iter()
        .filter(|(_, p)| p.is_dir())
        .map(|(k, p)| obj(vec![("project", s(&k)), ("path", s(&p.display().to_string()))]))
        .collect();
    let mut out = vec![
        ("home", s(&home.display().to_string())),
        ("memory_dir", s(&home.join("memory").display().to_string())),
        ("projects_dir", s(&nmtype::project::projects_dir(&home).display().to_string())),
        ("models_dir", s(&nmtype::models_dir(&home).display().to_string())),
        ("library_dir", s(&home.join("library").display().to_string())),
        ("memory_dirs", Value::Array(roots)),
        (
            "memory_dir_note",
            s("The memory folder is fixed by the Nano's layout (<home>/memory, and memory/projects/<key> per project); one server may use another with serve --memory-dir."),
        ),
        ("settable", Value::Array(vec![obj(vec![("key", s(FLOOR_KEY)), ("unit", s("GB")), ("min", n(1)), ("max", n(FLOOR_MAX_GB)), ("default", n(FLOOR_DEFAULT_GB))])])),
    ];
    out.extend(disk_fields(&home));
    obj(out)
}

/// The control key a storage setting name stands for (`disk_floor_gb` or its full key).
/// # Errors
/// The memory folder (fixed) or an unknown name.
pub fn setting_key(name: &str) -> Result<&'static str, String> {
    match name {
        "disk_floor_gb" | "storage.disk_floor_gb" => Ok(FLOOR_KEY),
        "memory_dir" | "storage.memory_dir" => Err("memory_dir: the memory folder is fixed by the Nano's layout (<home>/memory); give one server another with serve --memory-dir".to_owned()),
        other => Err(format!("{other}: not a storage setting (settable: {FLOOR_KEY})")),
    }
}

// ---------------------------------------------------------------------------------------------
// Prune and delete.

/// A path compared component by component, without case on Windows.
fn folded(p: &Path) -> PathBuf {
    if cfg!(windows) { PathBuf::from(p.to_string_lossy().to_lowercase()) } else { p.to_path_buf() }
}
fn same_path(a: &Path, b: &Path) -> bool {
    folded(a) == folded(b)
}

fn remove(i: &Item) -> Result<(), String> {
    let r = if i.folder {
        if i.path.file_name().is_some_and(|f| f == nmsubs::dense::store::DIR) && i.category == "vectors" {
            // The dense store's own clear (its files and their temporary copies).
            return nmsubs::dense::store::clear(&i.path).map(|_| ());
        }
        std::fs::remove_dir_all(&i.path)
    } else {
        std::fs::remove_file(&i.path)
    };
    r.map_err(|e| format!("{}: {e}", i.path.display()))
}

/// `POST /api/storage/prune` and `nm_storage` `prune`: removes the rebuildable items of one
/// category. Items changed in the last ten minutes may be in use and are skipped. Logged as a
/// `storage-prune` audit row.
/// # Errors
/// No `confirm`, an unknown category, a removal or the audit log failing.
/// What a prune of `category` would remove now, without removing anything (2026-11-26, Luis Jake
/// Gabriel III: every delete is previewed and confirmed in more than one step): each rebuildable
/// item with its size, the totals, and the items changed in the last ten minutes that it would skip.
/// # Errors
/// An unknown category.
pub fn prune_plan(home: &Path, category: &str) -> Result<Value, String> {
    let Some((id, name)) = CATEGORIES.iter().find(|(c, _)| *c == category) else {
        return Err(format!("category: {category} is not one of memory, vectors, compression, cartridges"));
    };
    let now = nmtype::now_ms();
    let (mut bytes, mut files) = (0_u64, 0_u64);
    let (mut gone, mut skipped) = (Vec::new(), Vec::new());
    for i in items(home).into_iter().filter(|i| i.category == *id && matches!(i.class, Class::Rebuildable | Class::Junk)) {
        if now.saturating_sub(i.modified_ms) < IN_USE_MS {
            skipped.push(s(&i.path.display().to_string()));
            continue;
        }
        bytes += i.bytes;
        files += i.files;
        gone.push(obj(vec![("label", s(&i.label)), ("path", s(&i.path.display().to_string())), ("bytes", n(i.bytes)), ("files", n(i.files))]));
    }
    Ok(obj(vec![
        ("category", s(id)),
        ("name", s(name)),
        ("items", Value::Array(gone)),
        ("bytes", n(bytes)),
        ("files", n(files)),
        ("skipped_in_use", Value::Array(skipped)),
    ]))
}

pub fn prune(home: &Path, category: &str, confirm: bool, by: &str) -> Result<Value, String> {
    let Some((id, name)) = CATEGORIES.iter().find(|(c, _)| *c == category) else {
        return Err(format!("category: {category} is not one of memory, vectors, compression, cartridges"));
    };
    if !confirm {
        return Err("confirm: a prune removes files; send confirm: true".to_owned());
    }
    let now = nmtype::now_ms();
    let (mut bytes, mut files) = (0_u64, 0_u64);
    let (mut removed, mut skipped) = (Vec::new(), Vec::new());
    for i in items(home).into_iter().filter(|i| i.category == *id && matches!(i.class, Class::Rebuildable | Class::Junk)) {
        if now.saturating_sub(i.modified_ms) < IN_USE_MS {
            skipped.push(s(&i.path.display().to_string()));
            continue;
        }
        remove(&i)?;
        bytes += i.bytes;
        files += i.files;
        removed.push(s(&i.path.display().to_string()));
    }
    let audit = crate::audit::audit(
        home,
        "storage-prune",
        vec![("category", s(id)), ("bytes", n(bytes)), ("files", n(files)), ("by", s(by)), ("label", s(&format!("Pruned rebuildable {name}")))],
    )?;
    Ok(obj(vec![
        ("audit_id", s(&audit)),
        ("category", s(id)),
        ("bytes", n(bytes)),
        ("files", n(files)),
        ("removed", Value::Array(removed)),
        ("skipped_in_use", Value::Array(skipped)),
    ]))
}

/// The reason a path may not be deleted, before it is matched to an item.
fn refused(home: &Path, p: &Path) -> Option<String> {
    let under = |base: &Path| folded(p).starts_with(folded(base));
    let locked = [
        (home.join("library").join("cartridges").join("base"), "base cartridges are read-only"),
        (absolute(&nmtype::models_dir(home)), "released weights are read-only"),
        (home.join("bin"), "the program folder is read-only"),
    ];
    if let Some((_, why)) = locked.iter().find(|(b, _)| under(b)) {
        return Some(format!("{}: {why}", p.display()));
    }
    if !under(home) {
        return Some(format!("{}: outside the Nano's folders", p.display()));
    }
    None
}

/// A cartridge file `<id>-v<N>.cartridge` named in `memory/stacks/in-use.json` as `<id>@<N>`.
fn in_use(home: &Path, p: &Path) -> bool {
    let Some(stem) = p.file_name().and_then(|f| f.to_str()).and_then(|f| f.strip_suffix(".cartridge")) else { return false };
    let Some((id, ver)) = stem.rsplit_once("-v") else { return false };
    let text = std::fs::read_to_string(nmlearn::library::in_use_file(home)).unwrap_or_default();
    text.contains(&format!("\"{id}@{ver}\""))
}

/// `POST /api/storage/delete` and `nm_storage` `delete`: deletes one removable item (a lens or
/// judge read-out version, a job filter, a preference cartridge, a world-model note store) by its
/// exact path (absolute, or relative to the Nano's folder). Refuses base cartridges, weights,
/// `bin/`, anything outside the Nano's folders, anything not removable and a cartridge in use.
/// Logged as a `storage-delete` audit row. There is no trash folder: the item is gone.
/// # Errors
/// No `confirm`, a refused path, the removal or the audit log failing.
/// The storage item a delete of `path` names, with every refusal checked first.
fn deletable(home: &Path, path: &str) -> Result<Item, String> {
    let home_abs = absolute(home);
    let given = Path::new(path);
    if given.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err("path: '..' is not allowed".to_owned());
    }
    let p = absolute(&if given.is_absolute() { given.to_path_buf() } else { home_abs.join(given) });
    if let Some(why) = refused(&home_abs, &p) {
        return Err(why);
    }
    let Some(item) = items(&home_abs).into_iter().find(|i| same_path(&i.path, &p)) else {
        return Err(format!("{}: not an item of the storage view", p.display()));
    };
    if !matches!(item.class, Class::Removable | Class::Junk) {
        return Err(format!("{}: {} ({}); only removable items and junk are deleted", p.display(), item.class.name(), item.label));
    }
    if in_use(&home_abs, &item.path) {
        return Err(format!("{}: the cartridge is in use (memory/stacks/in-use.json); switch it off first", p.display()));
    }
    Ok(item)
}

/// What a delete of `path` would remove, without removing anything (2026-11-26: every delete is
/// previewed and confirmed in more than one step).
/// # Errors
/// The same refusals as `delete`.
pub fn delete_plan(home: &Path, path: &str) -> Result<Value, String> {
    let item = deletable(home, path)?;
    Ok(obj(vec![
        ("path", s(&item.path.display().to_string())),
        ("label", s(&item.label)),
        ("category", s(item.category)),
        ("bytes", n(item.bytes)),
        ("files", n(item.files)),
    ]))
}

pub fn delete(home: &Path, path: &str, confirm: bool, by: &str) -> Result<Value, String> {
    let item = deletable(home, path)?;
    let item = &item;
    if !confirm {
        return Err("confirm: a delete removes the item for good; send confirm: true".to_owned());
    }
    remove(item)?;
    let shown = item.path.display().to_string();
    let audit = crate::audit::audit(
        home,
        "storage-delete",
        vec![
            ("path", s(&shown)),
            ("category", s(item.category)),
            ("bytes", n(item.bytes)),
            ("files", n(item.files)),
            ("by", s(by)),
            ("label", s(&format!("Deleted {}", item.label))),
        ],
    )?;
    Ok(obj(vec![("audit_id", s(&audit)), ("path", s(&shown)), ("bytes", n(item.bytes)), ("files", n(item.files)), ("label", s(&item.label))]))
}

/// Totals by category id, for callers that want only the sums.
#[must_use]
pub fn totals(home: &Path) -> BTreeMap<&'static str, (u64, u64)> {
    let mut m: BTreeMap<&'static str, (u64, u64)> = CATEGORIES.iter().map(|(c, _)| (*c, (0, 0))).collect();
    for i in items(home) {
        if let Some(t) = m.get_mut(i.category) {
            t.0 += i.bytes;
            t.1 += i.files;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    const OLD: u64 = 3 * 60 * 60 * 1000;

    /// A temporary Nano folder under `scratch/unit-tests/` (never the real one).
    fn home(name: &str) -> PathBuf {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scratch/unit-tests");
        let h = absolute(&root.join(format!("storage-{name}-{}-{}", std::process::id(), nmtype::now_ms())));
        std::fs::create_dir_all(&h).unwrap();
        h
    }
    fn put(h: &Path, rel: &str, bytes: usize, age_ms: u64) {
        let p = h.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, vec![b'x'; bytes]).unwrap();
        let t = std::time::SystemTime::now() - std::time::Duration::from_millis(age_ms);
        std::fs::File::options().write(true).open(&p).unwrap().set_modified(t).unwrap();
    }
    fn layout(h: &Path) {
        put(h, "memory/world-model/notes.jsonl", 100, OLD);
        put(h, "memory/episodes.jsonl", 10, OLD);
        put(h, "memory/audit.jsonl", 5, OLD);
        put(h, "memory/machine.json", 7, OLD);
        put(h, "memory/dense/vectors.bin", 1000, OLD);
        put(h, "memory/substrate-cache-abc.bin", 2000, OLD);
        put(h, "memory/live/x.json.tmp", 3, OLD);
        put(h, "memory/projects/p-1/decisions.jsonl", 20, OLD);
        put(h, "memory/projects/p-1/meaning/floors.json", 40, OLD);
        put(h, "memory/projects/p-1/training/examples.jsonl", 30, OLD);
        put(h, "library/cartridges/base/reader/reader-base-v3.cartridge", 500, OLD);
        put(h, "library/cartridges/adapter/lens-a/lens-a-v1.cartridge", 60, OLD);
        put(h, "library/cartridges/adapter/lens-a/lens-a-v2.cartridge", 60, OLD);
        put(h, "library/parameter-maps/reader/job-filter-read.json", 11, OLD);
        put(h, "library/parameter-maps/reader/pmap-v1.json", 13, OLD);
        put(h, "memory/stacks/in-use.json", 2, OLD);
        put(h, "weights/qwen3-0.6b/files/model.safetensors", 800, OLD);
        put(h, "scratch/build-target/debug/x.rlib", 300, OLD);
        put(h, "scratch/stepcheck-123/src.rs", 50, 1000);
        put(h, "bin/neuromorph-nano.exe", 9, OLD);
    }
    fn class_of(h: &Path, rel: &str) -> Option<(&'static str, Class)> {
        let p = h.join(rel);
        items(h).into_iter().find(|i| same_path(&i.path, &p)).map(|i| (i.category, i.class))
    }

    #[test]
    fn storage_view_groups_measures_and_classes() {
        let h = home("view");
        layout(&h);
        assert_eq!(class_of(&h, "memory/world-model/notes.jsonl"), Some(("memory", Class::Removable)));
        assert_eq!(class_of(&h, "memory/episodes.jsonl"), Some(("memory", Class::ReadOnly)));
        assert_eq!(class_of(&h, "memory/dense"), Some(("vectors", Class::Rebuildable)));
        assert_eq!(class_of(&h, "memory/substrate-cache-abc.bin"), Some(("vectors", Class::Rebuildable)));
        assert_eq!(class_of(&h, "memory/projects/p-1/meaning"), Some(("vectors", Class::Rebuildable)));
        assert_eq!(class_of(&h, "memory/live/x.json.tmp"), Some(("compression", Class::Rebuildable)));
        assert_eq!(class_of(&h, "scratch/build-target"), Some(("compression", Class::Rebuildable)));
        assert_eq!(class_of(&h, "library/cartridges/base/reader"), Some(("cartridges", Class::ReadOnly)));
        assert_eq!(class_of(&h, "library/cartridges/adapter/lens-a/lens-a-v2.cartridge"), Some(("cartridges", Class::Removable)));
        assert_eq!(class_of(&h, "library/parameter-maps/reader/job-filter-read.json"), Some(("cartridges", Class::Removable)));
        assert_eq!(class_of(&h, "library/parameter-maps/reader/pmap-v1.json"), Some(("cartridges", Class::ReadOnly)));
        assert_eq!(class_of(&h, "weights/qwen3-0.6b"), Some(("cartridges", Class::ReadOnly)));
        let t = totals(&h);
        assert_eq!(t["memory"], (100 + 10 + 5 + 7 + 20 + 30, 6));
        assert_eq!(t["vectors"], (1000 + 2000 + 40, 3));
        assert_eq!(t["compression"], (3 + 300 + 50, 3));
        assert_eq!(t["cartridges"], (500 + 60 + 60 + 11 + 13 + 2 + 800, 7));
        let v = view(&h);
        let cats = v.get("categories").and_then(Value::array).unwrap();
        assert_eq!(cats.len(), 5, "memory, vectors, compression, cartridges, junk");
        assert_eq!(cats[1].get("rebuildable_bytes").and_then(Value::integer).unwrap(), 3040);
        assert_eq!(v.get("total_bytes").and_then(Value::integer).unwrap(), 5011);
        let first = cats[1].get("items").and_then(Value::array).unwrap()[0].clone();
        assert!(first.get("path").and_then(Value::string).unwrap().ends_with("substrate-cache-abc.bin"));
        let _ = std::fs::remove_dir_all(&h);
    }

    #[test]
    fn storage_prune_needs_confirm_skips_in_use_and_audits() {
        let h = home("prune");
        layout(&h);
        assert!(prune(&h, "vectors", false, "test").is_err());
        assert!(prune(&h, "nothing", true, "test").is_err());
        let r = prune(&h, "compression", true, "test").unwrap();
        // The step-check copy changed a second ago: in use, kept.
        assert_eq!(r.get("bytes").and_then(Value::integer).unwrap(), 303);
        assert!(h.join("scratch/stepcheck-123/src.rs").exists());
        assert!(!h.join("scratch/build-target").exists() && !h.join("memory/live/x.json.tmp").exists());
        let r = prune(&h, "vectors", true, "test").unwrap();
        assert_eq!(r.get("files").and_then(Value::integer).unwrap(), 3);
        assert!(!h.join("memory/dense/vectors.bin").exists() && !h.join("memory/substrate-cache-abc.bin").exists());
        assert!(h.join("memory/world-model/notes.jsonl").exists() && h.join("memory/episodes.jsonl").exists());
        let log = std::fs::read_to_string(h.join("memory/audit.jsonl")).unwrap();
        assert_eq!(log.matches("\"kind\":\"storage-prune\"").count(), 2);
        let _ = std::fs::remove_dir_all(&h);
    }

    #[test]
    fn storage_delete_refuses_locked_and_removes_removable() {
        let h = home("delete");
        layout(&h);
        for rel in ["library/cartridges/base/reader/reader-base-v3.cartridge", "weights/qwen3-0.6b/files/model.safetensors", "bin/neuromorph-nano.exe", "memory/episodes.jsonl", "memory/../x"] {
            assert!(delete(&h, rel, true, "test").is_err(), "{rel} was not refused");
        }
        let outside = h.parent().unwrap().join("elsewhere.txt");
        assert!(delete(&h, &outside.display().to_string(), true, "test").unwrap_err().contains("outside"));
        assert!(delete(&h, "library/parameter-maps/reader/job-filter-read.json", false, "test").is_err());
        assert!(h.join("library/parameter-maps/reader/job-filter-read.json").exists());
        std::fs::create_dir_all(h.join("memory/stacks")).unwrap();
        std::fs::write(h.join("memory/stacks/in-use.json"), r#"{"loaded":[{"key":"lens-a@2"}]}"#).unwrap();
        assert!(delete(&h, "library/cartridges/adapter/lens-a/lens-a-v2.cartridge", true, "test").unwrap_err().contains("in use"));
        let abs = h.join("library/cartridges/adapter/lens-a/lens-a-v1.cartridge").display().to_string();
        let r = delete(&h, &abs, true, "test").unwrap();
        assert_eq!(r.get("bytes").and_then(Value::integer).unwrap(), 60);
        assert!(!h.join("library/cartridges/adapter/lens-a/lens-a-v1.cartridge").exists());
        delete(&h, "memory/world-model/notes.jsonl", true, "test").unwrap();
        let log = std::fs::read_to_string(h.join("memory/audit.jsonl")).unwrap();
        assert_eq!(log.matches("\"kind\":\"storage-delete\"").count(), 2);
        let _ = std::fs::remove_dir_all(&h);
    }

    #[test]
    fn storage_settings_read_floor_and_parse_fsutil() {
        let h = home("settings");
        assert_eq!(floor_gb(&h), FLOOR_DEFAULT_GB);
        std::fs::write(h.join("settings.json"), r#"{"storage.disk_floor_gb":"12"}"#).unwrap();
        assert_eq!(floor_gb(&h), 12);
        let v = settings(&h);
        assert_eq!(v.get("disk_floor_gb").and_then(Value::integer).unwrap(), 12);
        assert!(v.get("memory_dir").and_then(Value::string).unwrap().ends_with("memory"));
        assert_eq!(setting_key("disk_floor_gb").unwrap(), FLOOR_KEY);
        assert!(setting_key("memory_dir").is_err() && setting_key("weights").is_err());
        let text = "Total free bytes        :     5,583,667,200 (  5.2 GB)\nTotal bytes             : 1,000,169,668,608 (931.5 GB)\n";
        assert_eq!(parse_fsutil(text), Some((5_583_667_200, 1_000_169_668_608)));
        let _ = std::fs::remove_dir_all(&h);
    }
}
