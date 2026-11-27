//! The dense index kept on disk: the lane's vectors in a project's memory folder, so a restart
//! loads them instead of embedding the whole material again.
//!
//! Vectors live in `<memory folder>/dense/` in time-ordered `NMDENSE1` segment files
//! (`vectors.bin` for segment 0, then `seg-000001.bin`, `seg-000002.bin`, ...), holding per chunk
//! its content hash (the lane's key, 64 hex digits) and its vector at 16 bits per value. No text
//! of the material and no file paths are kept on disk. Each segment starts with a header naming
//! what made the vectors (the embedder's weight hash, its precision, the arithmetic switches, the
//! recipe of the text embedded and its pooling); a header that does not equal the running
//! embedder's makes the file unused, with the differing field named, never mixed in. Every write
//! goes to `<file>.tmp`, syncs, and renames over the target.
//!
//! Dead-skin biomimetic shedding (Hypothesis H32, Luis Jake Gabriel III, 2026-11-17):
//! 1. **Continuous micro-shedding**: writes and syncs shed at most `SHED_BATCH` expired dead rows
//!    at a time (rewriting at most the touched segment, never the whole store).
//! 2. **Barrier / undo layer**: recently dead rows stay on disk (tracked in `skin.json` by opaque
//!    chunk hash and 16-hex file-path digest only) instead of disappearing immediately; when an
//!    undo, revert, or branch switch brings a chunk back, its vector is revived with zero GPU
//!    re-embedding.
//! 3. **Age-calibrated enzymatic timing**: shedding age (`ttl_edit_ms`) is calibrated from the
//!    project's own observed revival rate and revival-age distribution (`p90 * 2`), shrinking
//!    toward `MIN_TTL_MS` when revivals are rare and expanding up to `MAX_TTL_MS` when undos or
//!    branch switches pay for the barrier bytes.
//! 4. **Time-ordered segments**: new rows append to the newest segment (`SEGMENT_ROWS` per file);
//!    any old segment whose rows are all expired dead rows is dropped whole (`remove_file`) with
//!    zero bytes rewritten.
//! 5. **Deleted-file priority**: rows whose file digest disappeared from the workspace get a
//!    shorter TTL (`1/4` of edited-file TTL) and shed ahead of edited-file rows.
//!
//! Layout of each segment file: `NMDENSE1`, a little-endian u32 header length, the header as
//! sorted `key=value` lines (with `dim` and `rows`), the SHA-256 of the body as 64 hex digits,
//! then the body: `rows` records of 64 hash bytes and `dim` binary16 values, little-endian.
//!
//! Each row's recipe (Roadmap Phase 4 (2026-11-24)): the recipe of the text embedded (`recipe`,
//! plain or with the code window's lead) and whether long pieces were embedded in parts, and of
//! which length (`parts_within`), name how a row was made, not which embedder made it. They are
//! no longer compared with the running embedder's: a segment whose rows were all made under the
//! running recipe is written exactly as before (the header names `recipe` and `parts_within`); a
//! segment holding rows of more than one recipe names `row_recipe` (`ROW_RECIPE`) and a table
//! `recipe.0`, `recipe.1`, ... in place of those two fields, and each record then carries one
//! byte after its hash, the index of its recipe in that table. A file without `row_recipe` reads
//! as today: every row made under the recipe its header names (plain when it names the plain
//! recipe and no parts). A row of another recipe is kept and loaded only where its text equals
//! the running recipe's (a piece with no lead under the lead recipe); every other such row is
//! embedded again through the lane's usual fill and the new row appended, so switching
//! `NANO_DENSE_LEAD` or `NANO_DENSE_PARTS` embeds only the rows whose recipe differs. A piece
//! never has two vectors: when two rows of one hash are read, the one of the running recipe wins.
//! Layout of a segment with row recipes: as above, each record being 64 hash bytes, one recipe
//! byte and `dim` binary16 values.
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// The folder under a project's memory folder.
pub const DIR: &str = "dense";
/// The first segment file in that folder (segment 0; backwards-compatible with single-file stores).
pub const FILE: &str = "vectors.bin";
/// Sidecar metadata file for dead-skin lifecycle and age calibration (hashes & timestamps only).
pub const SKIN_FILE: &str = "skin.json";
/// Rows gathered before they are written.
pub const FLUSH_ROWS: usize = 128;
/// Target rows per time-ordered segment file before rolling to `seg-<id>.bin`.
pub const SEGMENT_ROWS: usize = 512;
/// Maximum expired dead rows actively rewritten/shed in one micro-shed step.
pub const SHED_BATCH: usize = 64;
/// Default dead-skin barrier retention for edited-file chunks before enough history accumulates.
pub const DEFAULT_EDIT_TTL_MS: u64 = 7 * 24 * 3600 * 1000;
/// Lower bound for calibrated edited-file barrier retention (10 minutes).
pub const MIN_TTL_MS: u64 = 10 * 60 * 1000;
/// Upper bound for calibrated edited-file barrier retention (30 days).
pub const MAX_TTL_MS: u64 = 30 * 24 * 3600 * 1000;
/// Lower bound for deleted-file barrier retention (2 minutes).
pub const MIN_DELETED_TTL_MS: u64 = 2 * 60 * 1000;
/// Deleted-file retention ratio relative to edited-file TTL (`1 / 4`).
pub const DELETED_TTL_DIV: u64 = 4;
/// Minimum observed chunk deaths before age calibration shifts away from the default TTL.
const CALIBRATION_MIN_DEATHS: u64 = 16;
/// Sliding window of observed revival ages kept for percentile calibration.
const MAX_REVIVAL_SAMPLES: usize = 256;
/// The settings key (`settings.json` at the Nano's root): `on` (default) or `off`.
pub const SETTING: &str = "dense_cache";
const MAGIC: &[u8; 8] = b"NMDENSE1";
const KEY: usize = 64;

/// Header fields that name the recipe of the text embedded rather than the embedder (Roadmap
/// Phase 4 (2026-11-24)): recorded per row, never compared with the running embedder's.
pub const RECIPE_FIELDS: [&str; 2] = ["recipe", "parts_within"];
/// The header field of a segment whose records each carry their recipe's index.
const ROW_RECIPE_KEY: &str = "row_recipe";
/// Its value: the record layout of a segment with row recipes.
pub const ROW_RECIPE: &str = "one byte per row after its hash, the index of its recipe.<n>; version 1";
/// Recipes one store can name (the recipe byte's range).
const MAX_RECIPES: usize = 256;

/// Rows of a segment: chunk hash, binary16 values, and the index of the row's recipe in the
/// store's table (0 the running recipe).
type Rows = Vec<(String, Vec<u16>, u8)>;

/// What made the vectors: field name to value. The store adds `dim` and `rows` itself.
pub type Header = BTreeMap<String, String>;

/// The name of a recipe as rows record it (Roadmap Phase 4 (2026-11-24)): the header's `recipe`
/// (empty when absent), with the parts' length when `parts_within` names one. A file written
/// before row recipes gives every row this name from its own header.
#[must_use]
pub fn recipe_name(text: Option<&str>, parts_within: Option<&str>) -> String {
    let text = text.unwrap_or("");
    match parts_within {
        Some(n) => format!("{text}; long pieces in parts within {n} tokens"),
        None => text.to_owned(),
    }
}

/// Two recipe names give the same text for a piece with no lead: they are equal once the lead
/// recipe is read as the plain one (a piece with no lead is embedded from the same text under both).
#[must_use]
pub fn same_without_lead(a: &str, b: &str) -> bool {
    let plain = |n: &str| n.strip_prefix(super::RECIPE_LEAD).map_or_else(|| n.to_owned(), |rest| format!("{}{rest}", super::RECIPE));
    a == b || plain(a) == plain(b)
}

/// The row `(key, recipe)` is the one a piece is read from: the running recipe's when the piece
/// has one, else the recipe `others` holds for it.
fn chosen(others: &BTreeMap<String, u8>, key: &str, recipe: u8) -> bool {
    others.get(key).map_or(recipe == 0, |&i| i == recipe)
}

/// Privacy-preserving 16-hex-character digest of a normalized file path (no paths are stored on disk).
#[must_use]
pub fn file_tag(path: &str) -> String {
    let norm = path.replace('\\', "/");
    let hash = nmhash::sha256_bytes(norm.as_bytes());
    hash.get(..16).unwrap_or(&hash).to_owned()
}

/// The store setting in `settings.json` under `home`: on unless it says `off`.
#[must_use]
pub fn cache_on(home: &Path) -> bool {
    let v = std::fs::read(home.join("settings.json")).ok().and_then(|b| nmjson::parse(&b).ok());
    v.as_ref().and_then(|v| v.get(SETTING).ok()).and_then(|v| v.string().ok()) != Some("off")
}

/// Removes all store files (`vectors.bin`, `seg-*.bin`, `skin.json`, and any `.tmp` files) from
/// `dir`; returns the bytes removed. A missing folder removes nothing.
/// # Errors
/// A file exists and cannot be removed.
pub fn clear(dir: &Path) -> Result<u64, String> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Ok(0) };
    let mut bytes = 0_u64;
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_store = name == FILE
            || name == format!("{FILE}.tmp")
            || name == SKIN_FILE
            || name == format!("{SKIN_FILE}.tmp")
            || parse_segment_id(&name).is_some()
            || name.strip_suffix(".tmp").and_then(parse_segment_id).is_some();
        if !is_store {
            continue;
        }
        let p = entry.path();
        if let Ok(m) = std::fs::metadata(&p) {
            std::fs::remove_file(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            bytes += m.len();
        }
    }
    Ok(bytes)
}

/// Total `(rows, bytes)` across all segment files in `dir`, or `None` when no segment exists.
#[must_use]
pub fn disk_usage(dir: &Path) -> Option<(u64, u64)> {
    use std::io::Read as _;
    let segs = list_segments(dir);
    if segs.is_empty() {
        return None;
    }
    let mut rows = 0_u64;
    let mut bytes = 0_u64;
    for (_, path) in segs {
        if let Ok(meta) = std::fs::metadata(&path) {
            bytes += meta.len();
        }
        let mut head = Vec::new();
        if let Ok(f) = std::fs::File::open(&path)
            && f.take(64 * 1024).read_to_end(&mut head).is_ok()
            && let Some(len_b) = head.get(8..12)
            && let Ok(arr) = <[u8; 4]>::try_from(len_b)
        {
            let len = u32::from_le_bytes(arr) as usize;
            if let Some(h) = head.get(12..12 + len) {
                let text = String::from_utf8_lossy(h);
                if let Some(r) = text.lines().find_map(|l| l.strip_prefix("rows=")).and_then(|v| v.trim().parse::<u64>().ok()) {
                    rows += r;
                }
            }
        }
    }
    if let Ok(meta) = std::fs::metadata(dir.join(SKIN_FILE)) {
        bytes += meta.len();
    }
    Some((rows, bytes))
}

/// Binary32 to binary16, rounded to nearest, ties to even.
#[must_use]
pub fn to_half(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = u16::try_from((b >> 16) & 0x8000).unwrap_or(0);
    let exp = i32::try_from((b >> 23) & 0xff).unwrap_or(0);
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man == 0 { 0 } else { 0x0200 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = u32::try_from(14 - e).unwrap_or(31);
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let up = rest > mid || (rest == mid && half & 1 == 1);
        return sign | u16::try_from(half + u32::from(up)).unwrap_or(0);
    }
    let half = (u32::try_from(e).unwrap_or(0) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    let up = rest > 0x1000 || (rest == 0x1000 && half & 1 == 1);
    // A carry out of the fraction moves into the exponent, as binary16 rounding does.
    sign | u16::try_from(half + u32::from(up)).unwrap_or(0x7c00)
}

/// Binary16 to binary32 (exact).
#[must_use]
pub fn from_half(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: shift the fraction up until its leading bit is the hidden one.
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// A vector as it is stored and read back (each value through binary16).
#[must_use]
pub fn rounded(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| from_half(to_half(*x))).collect()
}

/// What opening the store found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Found {
    /// No store file yet.
    Empty,
    /// The store was read: rows held across segments, rows of live chunks among them.
    Read { rows: usize, live: usize },
    /// The store was not used, with the reason.
    Ignored(String),
}

impl Found {
    /// One line for the lane's note.
    #[must_use]
    pub fn line(&self) -> String {
        match self {
            Self::Empty => "no stored vectors yet".to_owned(),
            Self::Read { rows, live } => format!("{live} stored vectors of current chunks loaded ({rows} rows in the file)"),
            Self::Ignored(why) => format!("stored vectors not used: {why}"),
        }
    }
}

/// Metadata for one dead row held in the barrier/undo layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeadRow {
    /// Wall-clock timestamp (ms) when this chunk left the live set.
    pub dead_since_ms: u64,
    /// True when the chunk's parent file digest also disappeared from the workspace.
    pub file_deleted: bool,
}

/// Persisted dead-skin lifecycle state and revival-age calibration history.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SkinState {
    pub dead: BTreeMap<String, DeadRow>,
    pub file_tags: BTreeMap<String, String>,
    pub deaths_total: u64,
    pub revivals_total: u64,
    pub shed_total: u64,
    pub segments_dropped: u64,
    pub revival_ages_ms: Vec<u64>,
}

impl SkinState {
    /// Observed revival probability over all recorded chunk deaths.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "ratio of event counts for TTL calibration")]
    pub fn revival_rate(&self) -> f64 {
        if self.deaths_total == 0 { 0.0 } else { self.revivals_total as f64 / self.deaths_total as f64 }
    }

    /// Calibrated barrier retention window (ms) for chunks from edited files.
    #[must_use]
    pub fn ttl_edit_ms(&self) -> u64 {
        if self.deaths_total < CALIBRATION_MIN_DEATHS {
            return DEFAULT_EDIT_TTL_MS;
        }
        let rate = self.revival_rate();
        if self.revivals_total >= 4 && !self.revival_ages_ms.is_empty() {
            let mut ages = self.revival_ages_ms.clone();
            ages.sort_unstable();
            let idx = (ages.len().saturating_mul(9) / 10).min(ages.len().saturating_sub(1));
            let p90 = ages.get(idx).copied().unwrap_or(DEFAULT_EDIT_TTL_MS);
            let target = if rate < 0.02 { p90 } else { p90.saturating_mul(2) };
            return target.clamp(MIN_TTL_MS, MAX_TTL_MS);
        }
        if rate < 0.02 { MIN_TTL_MS } else { DEFAULT_EDIT_TTL_MS }
    }

    /// Calibrated barrier retention window (ms) for chunks from deleted files (shorter, sheds first).
    #[must_use]
    pub fn ttl_deleted_ms(&self) -> u64 {
        (self.ttl_edit_ms() / DELETED_TTL_DIV).max(MIN_DELETED_TTL_MS)
    }

    fn record_revival(&mut self, age_ms: u64) {
        self.revivals_total = self.revivals_total.saturating_add(1);
        if self.revival_ages_ms.len() >= MAX_REVIVAL_SAMPLES {
            self.revival_ages_ms.remove(0);
        }
        self.revival_ages_ms.push(age_ms);
    }

    fn is_expired(&self, row: DeadRow, now_ms: u64) -> bool {
        let ttl = if row.file_deleted { self.ttl_deleted_ms() } else { self.ttl_edit_ms() };
        now_ms.saturating_sub(row.dead_since_ms) >= ttl
    }

    fn encode(&self) -> String {
        let mut root = BTreeMap::new();
        root.insert("version".to_owned(), nmjson::Value::Number("1".to_owned()));
        root.insert("deaths_total".to_owned(), nmjson::Value::Number(self.deaths_total.to_string()));
        root.insert("revivals_total".to_owned(), nmjson::Value::Number(self.revivals_total.to_string()));
        root.insert("shed_total".to_owned(), nmjson::Value::Number(self.shed_total.to_string()));
        root.insert("segments_dropped".to_owned(), nmjson::Value::Number(self.segments_dropped.to_string()));
        root.insert(
            "revival_ages_ms".to_owned(),
            nmjson::Value::Array(self.revival_ages_ms.iter().map(|a| nmjson::Value::Number(a.to_string())).collect()),
        );
        let mut dead_obj = BTreeMap::new();
        for (k, d) in &self.dead {
            let mut row = BTreeMap::new();
            row.insert("since".to_owned(), nmjson::Value::Number(d.dead_since_ms.to_string()));
            row.insert("deleted".to_owned(), nmjson::Value::Bool(d.file_deleted));
            dead_obj.insert(k.clone(), nmjson::Value::Object(row));
        }
        root.insert("dead".to_owned(), nmjson::Value::Object(dead_obj));
        let mut tags_obj = BTreeMap::new();
        for (k, t) in &self.file_tags {
            tags_obj.insert(k.clone(), nmjson::Value::String(t.clone()));
        }
        root.insert("file_tags".to_owned(), nmjson::Value::Object(tags_obj));
        nmjson::Value::Object(root).encode().unwrap_or_default()
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let v = nmjson::parse(bytes).ok()?;
        let u64_of = |key: &str| v.get(key).ok().and_then(|x| x.integer().ok()).unwrap_or(0);
        let mut st = Self {
            deaths_total: u64_of("deaths_total"),
            revivals_total: u64_of("revivals_total"),
            shed_total: u64_of("shed_total"),
            segments_dropped: u64_of("segments_dropped"),
            ..Self::default()
        };
        if let Ok(arr) = v.get("revival_ages_ms").and_then(nmjson::Value::array) {
            for item in arr.iter().take(MAX_REVIVAL_SAMPLES) {
                if let Ok(ms) = item.integer() {
                    st.revival_ages_ms.push(ms);
                }
            }
        }
        if let Ok(obj) = v.get("dead").and_then(nmjson::Value::object) {
            for (k, entry) in obj {
                let since = entry.get("since").ok().and_then(|x| x.integer().ok()).unwrap_or(0);
                let deleted = matches!(entry.get("deleted"), Ok(nmjson::Value::Bool(true)));
                st.dead.insert(k.clone(), DeadRow { dead_since_ms: since, file_deleted: deleted });
            }
        }
        if let Ok(obj) = v.get("file_tags").and_then(nmjson::Value::object) {
            for (k, val) in obj {
                if let Ok(s) = val.string() {
                    st.file_tags.insert(k.clone(), s.to_owned());
                }
            }
        }
        Some(st)
    }
}

/// Outcome of one dead-skin sync + micro-shed pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShedReport {
    /// Vectors revived from the dead-skin barrier layer (zero GPU re-embedding needed).
    pub revived: Vec<(String, Vec<f32>)>,
    /// Newly dead rows moved into the barrier layer during this pass.
    pub marked_dead: usize,
    /// Expired dead rows permanently shed during this pass (whole-segment drops + micro-batch).
    pub shed_rows: usize,
    /// Whole 100%-dead segments dropped with zero rewrite.
    pub segments_dropped: usize,
    /// Segments rewritten during bounded micro-shedding.
    pub segments_rewritten: usize,
}

/// Snapshot of the dead-skin store's health and calibration metrics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SkinStats {
    pub live_rows: usize,
    pub dead_rows: usize,
    pub dead_deleted_file: usize,
    pub dead_edited_file: usize,
    pub segments: usize,
    pub segments_dropped: u64,
    pub shed_total: u64,
    pub revivals_total: u64,
    pub deaths_total: u64,
    pub revival_rate: f64,
    pub ttl_edit_ms: u64,
    pub ttl_deleted_ms: u64,
}

#[derive(Debug)]
struct Segment {
    id: u64,
    rows: Rows,
    written_len: usize,
    dirty: bool,
}

/// The store of one project: time-ordered segments on disk, dead-skin barrier layer, and rows
/// waiting to be written.
#[derive(Debug)]
pub struct Store {
    dir: PathBuf,
    header: Header,
    dim: usize,
    segments: Vec<Segment>,
    keys: BTreeSet<String>,
    next_seg_id: u64,
    skin: SkinState,
    skin_dirty: bool,
    seg_cap: usize,
    shed_batch: usize,
    /// Roadmap Phase 4 (2026-11-24): the recipes rows name, 0 the running embedder's (from its
    /// header), then those read from files.
    recipes: Vec<String>,
    /// Pieces whose row is of another recipe than the running one, with that recipe's index.
    others: BTreeMap<String, u8>,
    /// Segment file writes made, and the last write's fault, for the lane's note.
    pub writes: usize,
    pub fault: Option<String>,
    /// The primary file was not used (another embedder's header, or damaged): the first row
    /// written goes to it, so the folder reads as this embedder's at the next open rather than
    /// staying refused by the old primary while new rows land in later segments.
    replace_primary: bool,
}

impl Store {
    /// Open the store in `dir` for an embedder described by `header`, reading all existing
    /// segment files (`vectors.bin` and `seg-*.bin`). Returns the store, the vectors of chunks in
    /// `live` (by hash), and what was found. Strictly read-only on disk; a file that cannot be
    /// used is left in place until the first write replaces it.
    #[must_use]
    pub fn open(dir: &Path, header: Header, live: &BTreeSet<&str>) -> (Self, Vec<(String, Vec<f32>)>, Found) {
        Self::open_at(dir, header, live, nmtype::now_ms())
    }

    /// As `open`, with an explicit wall-clock timestamp `now_ms` (for deterministic testing).
    #[must_use]
    pub fn open_at(dir: &Path, header: Header, live: &BTreeSet<&str>, now_ms: u64) -> (Self, Vec<(String, Vec<f32>)>, Found) {
        let running = recipe_name(header.get("recipe").map(String::as_str), header.get("parts_within").map(String::as_str));
        let mut st = Self {
            dir: dir.to_path_buf(),
            header,
            dim: 0,
            segments: Vec::new(),
            keys: BTreeSet::new(),
            next_seg_id: 1,
            skin: SkinState::default(),
            skin_dirty: false,
            seg_cap: SEGMENT_ROWS,
            shed_batch: SHED_BATCH,
            recipes: vec![running],
            others: BTreeMap::new(),
            writes: 0,
            fault: None,
            replace_primary: false,
        };
        let files = list_segments(dir);
        if files.is_empty() {
            return (st, Vec::new(), Found::Empty);
        }
        let mut loaded_segments = Vec::new();
        let mut first_err: Option<String> = None;
        for (id, path) in files {
            let Ok(bytes) = std::fs::read(&path) else { continue };
            match parse(&bytes, &st.header, &mut st.recipes) {
                Err(why) => {
                    if id == 0 {
                        // Backwards-compatible contract: a bad primary vectors.bin invalidates open.
                        st.replace_primary = true;
                        return (st, Vec::new(), Found::Ignored(why));
                    }
                    if first_err.is_none() {
                        first_err = Some(why);
                    }
                }
                Ok((dim, rows)) => {
                    if st.dim == 0 {
                        st.dim = dim;
                    } else if dim != 0 && dim != st.dim {
                        continue;
                    }
                    st.next_seg_id = st.next_seg_id.max(id.saturating_add(1));
                    loaded_segments.push((id, rows));
                }
            }
        }
        if loaded_segments.is_empty() {
            return match first_err {
                Some(why) => (st, Vec::new(), Found::Ignored(why)),
                None => (st, Vec::new(), Found::Empty),
            };
        }
        if let Ok(skin_bytes) = std::fs::read(dir.join(SKIN_FILE))
            && let Some(skin) = SkinState::decode(&skin_bytes)
        {
            st.skin = skin;
        }
        // Roadmap Phase 4 (2026-11-24): one row per piece, the running recipe's when a file holds
        // one (a refresh appends it after the older row), else the first read.
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for (_, rows) in &loaded_segments {
            for (k, _, r) in rows {
                if seen.insert(k.as_str()) {
                    if *r != 0 {
                        st.others.insert(k.clone(), *r);
                    }
                } else if *r == 0 {
                    st.others.remove(k);
                }
            }
        }
        drop(seen);
        let running = st.recipes.first().cloned().unwrap_or_default();
        let mut out = Vec::new();
        for (id, rows) in loaded_segments {
            let mut kept = Vec::with_capacity(rows.len());
            for (k, v, r) in rows {
                if chosen(&st.others, &k, r) && st.keys.insert(k.clone()) {
                    if live.contains(k.as_str()) {
                        if let Some(dead) = st.skin.dead.remove(&k) {
                            st.skin.record_revival(now_ms.saturating_sub(dead.dead_since_ms));
                            st.skin_dirty = true;
                        }
                        // A row of another recipe is handed out only where its text may equal
                        // the running recipe's (the caller checks the piece's lead).
                        if st.recipes.get(usize::from(r)).is_some_and(|n| same_without_lead(n, &running)) {
                            out.push((k.clone(), v.iter().map(|h| from_half(*h)).collect()));
                        }
                    } else if !st.skin.dead.contains_key(&k) {
                        st.skin.dead.insert(k.clone(), DeadRow { dead_since_ms: now_ms, file_deleted: false });
                        st.skin.deaths_total = st.skin.deaths_total.saturating_add(1);
                        st.skin_dirty = true;
                    }
                    kept.push((k, v, r));
                }
            }
            let written_len = kept.len();
            st.segments.push(Segment { id, rows: kept, written_len, dirty: false });
        }
        // Prune sidecar entries for keys that no longer exist in any loaded segment.
        st.skin.dead.retain(|k, _| st.keys.contains(k));
        st.skin.file_tags.retain(|k, _| st.keys.contains(k));
        let found = Found::Read { rows: st.rows(), live: out.len() };
        (st, out, found)
    }

    /// Override segment capacity and micro-shed batch size (for deterministic tests and tuning).
    #[must_use]
    pub fn with_limits(mut self, seg_cap: usize, shed_batch: usize) -> Self {
        self.seg_cap = seg_cap.max(1);
        self.shed_batch = shed_batch.max(1);
        self
    }

    /// The primary store file's path (`vectors.bin`).
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.dir.join(FILE)
    }

    /// Total rows held across all segments (live + dead-skin barrier + pending).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.segments.iter().map(|s| s.rows.len()).sum()
    }

    /// Number of active segment files held by this store.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Rows waiting to be written to disk.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.segments.iter().map(|s| s.rows.len().saturating_sub(s.written_len)).sum()
    }

    /// Current dead-skin barrier and calibration statistics.
    #[must_use]
    pub fn skin_stats(&self) -> SkinStats {
        let total = self.rows();
        let dead_rows = self.skin.dead.len().min(total);
        let dead_deleted_file = self.skin.dead.values().filter(|d| d.file_deleted).count();
        let dead_edited_file = dead_rows.saturating_sub(dead_deleted_file);
        SkinStats {
            live_rows: total.saturating_sub(dead_rows),
            dead_rows,
            dead_deleted_file,
            dead_edited_file,
            segments: self.segments.len(),
            segments_dropped: self.skin.segments_dropped,
            shed_total: self.skin.shed_total,
            revivals_total: self.skin.revivals_total,
            deaths_total: self.skin.deaths_total,
            revival_rate: self.skin.revival_rate(),
            ttl_edit_ms: self.skin.ttl_edit_ms(),
            ttl_deleted_ms: self.skin.ttl_deleted_ms(),
        }
    }

    /// Register opaque file digests for already-stored live chunks (e.g. loaded from a legacy
    /// `vectors.bin` written before `skin.json` existed).
    pub fn note_file_tags<'a, I>(&mut self, pairs: I)
    where
        I: IntoIterator<Item = (&'a str, String)>,
    {
        for (k, tag) in pairs {
            if self.keys.contains(k) && !self.skin.file_tags.contains_key(k) {
                self.skin.file_tags.insert(k.to_owned(), tag);
                self.skin_dirty = true;
            }
        }
    }

    /// The recipe rows are made under now (Roadmap Phase 4 (2026-11-24)): named by the running
    /// embedder's header (`recipe_name`).
    #[must_use]
    pub fn recipe(&self) -> &str {
        self.recipes.first().map_or("", String::as_str)
    }

    /// The recipe of the row a piece is read from, when the store holds one.
    #[must_use]
    pub fn row_recipe(&self, key: &str) -> Option<&str> {
        if !self.keys.contains(key) {
            return None;
        }
        let i = self.others.get(key).copied().unwrap_or(0);
        self.recipes.get(usize::from(i)).map(String::as_str)
    }

    /// The piece's row may be used under the running recipe: it was made under it, or (for a
    /// piece with no lead, `has_lead` false) under a recipe whose text is the same for it. Any
    /// other held row is to be embedded again (`add` then appends the new row).
    #[must_use]
    pub fn current(&self, key: &str, has_lead: bool) -> bool {
        match self.others.get(key) {
            None => self.keys.contains(key),
            Some(&i) => !has_lead && self.recipes.get(usize::from(i)).is_some_and(|n| same_without_lead(n, self.recipe())),
        }
    }

    /// Pieces whose row is of another recipe than the running one.
    #[must_use]
    pub fn other_recipe_rows(&self) -> usize {
        self.others.len()
    }

    /// Add one vector (rounded to binary16) under its chunk hash, made under the running recipe;
    /// a hash already held under that recipe or a vector of another width is skipped. A hash held
    /// under another recipe gets the new row appended (Roadmap Phase 4 (2026-11-24)); the older
    /// row is left out when its segment is next written. Writes dirty segments when `FLUSH_ROWS`
    /// rows wait.
    pub fn add(&mut self, key: &str, v: &[f32]) {
        self.add_tagged(key, v, None);
    }

    /// As `add`, attaching an opaque file digest (`file_tag`) so deleted-file rows shed first.
    pub fn add_tagged(&mut self, key: &str, v: &[f32], tag: Option<&str>) {
        let other = self.others.contains_key(key);
        if key.len() != KEY || (self.keys.contains(key) && !other) || v.is_empty() || (self.dim != 0 && v.len() != self.dim) {
            return;
        }
        if self.dim == 0 {
            self.dim = v.len();
        }
        if other {
            self.others.remove(key);
        }
        self.keys.insert(key.to_owned());
        self.skin.dead.remove(key);
        if let Some(t) = tag {
            self.skin.file_tags.insert(key.to_owned(), t.to_owned());
            self.skin_dirty = true;
        }
        let encoded: Vec<u16> = v.iter().map(|x| to_half(*x)).collect();
        let need_new = self.segments.last().is_none_or(|s| s.rows.len() >= self.seg_cap);
        if need_new {
            let id = if self.segments.is_empty() && (self.replace_primary || !self.dir.join(FILE).exists()) {
                self.replace_primary = false;
                0
            } else {
                let id = self.next_seg_id;
                self.next_seg_id = self.next_seg_id.saturating_add(1);
                id
            };
            self.next_seg_id = self.next_seg_id.max(id.saturating_add(1));
            self.segments.push(Segment { id, rows: vec![(key.to_owned(), encoded, 0)], written_len: 0, dirty: true });
        } else if let Some(seg) = self.segments.last_mut() {
            seg.rows.push((key.to_owned(), encoded, 0));
            seg.dirty = true;
        }
        if self.pending() >= FLUSH_ROWS {
            self.flush();
        }
    }

    /// Flush dirty segments and run one bounded micro-shed pass for expired dead rows.
    pub fn flush(&mut self) {
        self.flush_at(nmtype::now_ms());
    }

    /// As `flush`, with an explicit wall-clock timestamp `now_ms`.
    pub fn flush_at(&mut self, now_ms: u64) {
        let mut rep = ShedReport::default();
        if !self.skin.dead.is_empty() {
            self.micro_shed(now_ms, &mut rep);
        }
        self.write_dirty();
    }

    /// Dead-skin biomimetic sync & micro-shedding (Hypothesis H32):
    /// - Revives any dead-barrier vectors whose keys reappeared in `live` (free undo / revert).
    /// - Moves chunks absent from `live` into the dead-skin barrier layer, tagging deleted files
    ///   when `live_file_tags` is given.
    /// - Drops any 100%-dead expired segment whole with zero rewrite.
    /// - Micro-sheds at most `shed_batch` expired dead rows (deleted-file rows first, oldest
    ///   first), rewriting only touched segments.
    pub fn shed(&mut self, live: &BTreeSet<&str>, live_file_tags: Option<&BTreeSet<String>>, now_ms: u64) -> ShedReport {
        let mut report = ShedReport::default();
        for seg in &self.segments {
            for (k, v, r) in &seg.rows {
                if live.contains(k.as_str()) {
                    // Roadmap Phase 4 (2026-11-24): a piece is revived from the row it is read
                    // from, never from an older row of another recipe.
                    if chosen(&self.others, k, *r)
                        && let Some(dead) = self.skin.dead.remove(k)
                    {
                        self.skin.record_revival(now_ms.saturating_sub(dead.dead_since_ms));
                        self.skin_dirty = true;
                        report.revived.push((k.clone(), v.iter().map(|h| from_half(*h)).collect()));
                    }
                } else {
                    let deleted = live_file_tags
                        .and_then(|tags| self.skin.file_tags.get(k).map(|t| !tags.contains(t)))
                        .unwrap_or(false);
                    match self.skin.dead.get_mut(k) {
                        Some(d) => {
                            if deleted && !d.file_deleted {
                                d.file_deleted = true;
                                self.skin_dirty = true;
                            }
                        }
                        None => {
                            self.skin.dead.insert(k.clone(), DeadRow { dead_since_ms: now_ms, file_deleted: deleted });
                            self.skin.deaths_total = self.skin.deaths_total.saturating_add(1);
                            report.marked_dead += 1;
                            self.skin_dirty = true;
                        }
                    }
                }
            }
        }
        self.micro_shed(now_ms, &mut report);
        self.write_dirty();
        report
    }

    /// Legacy bulk compaction: drops all rows not in `live` when strictly more than half of all
    /// rows are dead. Returns how many rows were dropped (0 when left as-is).
    pub fn compact(&mut self, live: &BTreeSet<&str>) -> usize {
        let total = self.rows();
        let dead: usize = self.segments.iter().flat_map(|s| s.rows.iter()).filter(|(k, _, _)| !live.contains(k.as_str())).count();
        if dead * 2 <= total || dead == 0 {
            return 0;
        }
        let mut dropped_ids = Vec::new();
        for seg in &mut self.segments {
            let before = seg.rows.len();
            seg.rows.retain(|(k, _, _)| live.contains(k.as_str()));
            if seg.rows.len() != before {
                seg.dirty = true;
            }
            if seg.rows.is_empty() && seg.id != 0 {
                dropped_ids.push(seg.id);
            }
        }
        for id in &dropped_ids {
            let _ = std::fs::remove_file(self.segment_path(*id));
            self.skin.segments_dropped = self.skin.segments_dropped.saturating_add(1);
        }
        self.segments.retain(|s| !dropped_ids.contains(&s.id));
        self.keys = self.segments.iter().flat_map(|s| s.rows.iter().map(|(k, _, _)| k.clone())).collect();
        self.others.retain(|k, _| self.keys.contains(k));
        self.skin.dead.retain(|k, _| self.keys.contains(k));
        self.skin.file_tags.retain(|k, _| self.keys.contains(k));
        self.skin.shed_total = self.skin.shed_total.saturating_add(u64::try_from(dead).unwrap_or(0));
        self.skin_dirty = true;
        self.write_dirty();
        dead
    }

    /// Pass 1: drop any segment whose rows are all expired dead rows with zero rewrite.
    /// Pass 2: micro-shed up to `shed_batch` remaining expired dead rows (deleted-file first,
    /// oldest first). If a large store exceeds 50% dead rows (`dead > seg_cap`), also allow
    /// micro-shedding the oldest dead rows (deleted-file first) so disk usage stays bounded.
    fn micro_shed(&mut self, now_ms: u64, report: &mut ShedReport) {
        // Pass 1: O(1) whole-segment drop when every row in a segment is expired dead.
        let mut drop_ids = Vec::new();
        for seg in &self.segments {
            if seg.rows.is_empty() {
                if self.segments.len() > 1 {
                    drop_ids.push(seg.id);
                }
                continue;
            }
            let all_expired = seg.rows.iter().all(|(k, _, _)| self.skin.dead.get(k).is_some_and(|d| self.skin.is_expired(*d, now_ms)));
            if all_expired {
                drop_ids.push(seg.id);
            }
        }
        if !drop_ids.is_empty() {
            let drop_set: BTreeSet<u64> = drop_ids.into_iter().collect();
            let mut dropped_rows = 0_usize;
            let mut gone: BTreeSet<String> = BTreeSet::new();
            for seg in &self.segments {
                if drop_set.contains(&seg.id) {
                    let _ = std::fs::remove_file(self.segment_path(seg.id));
                    gone.extend(seg.rows.iter().map(|(k, _, _)| k.clone()));
                    dropped_rows += seg.rows.len();
                    report.segments_dropped += 1;
                    self.skin.segments_dropped = self.skin.segments_dropped.saturating_add(1);
                }
            }
            self.segments.retain(|s| !drop_set.contains(&s.id));
            // Roadmap Phase 4 (2026-11-24): a piece with a second row (of another recipe) in a
            // segment kept stays held; that row is shed with its key in pass 2.
            let still: BTreeSet<&str> =
                self.segments.iter().flat_map(|s| s.rows.iter().map(|(k, _, _)| k.as_str())).filter(|k| gone.contains(*k)).collect();
            for k in gone.iter().filter(|k| !still.contains(k.as_str())) {
                self.keys.remove(k);
                self.skin.dead.remove(k);
                self.skin.file_tags.remove(k);
                self.others.remove(k);
            }
            report.shed_rows += dropped_rows;
            self.skin.shed_total = self.skin.shed_total.saturating_add(u64::try_from(dropped_rows).unwrap_or(0));
            self.skin_dirty = true;
        }

        // Pass 2: bounded micro-batch shedding (deleted-file rows first, oldest first).
        let total = self.rows();
        let over_ceiling = self.skin.dead.len() >= FLUSH_ROWS && self.skin.dead.len().saturating_mul(2) > total;
        let mut candidates: Vec<(bool, bool, u64, String)> = Vec::new();
        for (k, d) in &self.skin.dead {
            let expired = self.skin.is_expired(*d, now_ms);
            if expired || over_ceiling {
                candidates.push((expired, d.file_deleted, d.dead_since_ms, k.clone()));
            }
        }
        if candidates.is_empty() {
            return;
        }
        // Order: expired first, then deleted-file first, then oldest dead_since_ms first.
        candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)).then_with(|| a.3.cmp(&b.3)));
        let victim_keys: BTreeSet<String> = candidates.into_iter().take(self.shed_batch).map(|c| c.3).collect();
        let mut empty_ids = Vec::new();
        let mut shed_here = 0_usize;
        for seg in &mut self.segments {
            let before = seg.rows.len();
            seg.rows.retain(|(k, _, _)| !victim_keys.contains(k));
            let removed = before - seg.rows.len();
            if removed > 0 {
                shed_here += removed;
                seg.dirty = true;
                if seg.rows.is_empty() {
                    empty_ids.push(seg.id);
                } else {
                    report.segments_rewritten += 1;
                }
            }
        }
        for k in &victim_keys {
            self.keys.remove(k);
            self.skin.dead.remove(k);
            self.skin.file_tags.remove(k);
            self.others.remove(k);
        }
        for id in &empty_ids {
            let _ = std::fs::remove_file(self.segment_path(*id));
            report.segments_dropped += 1;
            self.skin.segments_dropped = self.skin.segments_dropped.saturating_add(1);
        }
        if !empty_ids.is_empty() {
            self.segments.retain(|s| !empty_ids.contains(&s.id));
        }
        if shed_here > 0 {
            report.shed_rows += shed_here;
            self.skin.shed_total = self.skin.shed_total.saturating_add(u64::try_from(shed_here).unwrap_or(0));
            self.skin_dirty = true;
        }
    }

    fn segment_path(&self, id: u64) -> PathBuf {
        self.dir.join(segment_name(id))
    }

    /// Atomically write every dirty segment and `skin.json` if modified.
    fn write_dirty(&mut self) {
        let any_seg_dirty = self.segments.iter().any(|s| s.dirty);
        if !any_seg_dirty && !self.skin_dirty {
            return;
        }
        let r = (|| -> Result<(), String> {
            std::fs::create_dir_all(&self.dir).map_err(|e| format!("{}: {e}", self.dir.display()))?;
            for seg in &mut self.segments {
                if !seg.dirty {
                    continue;
                }
                // Roadmap Phase 4 (2026-11-24): rows a newer row of the piece replaced are
                // left out of the segment written.
                seg.rows.retain(|(k, _, r)| chosen(&self.others, k, *r));
                let bytes = encode(&self.header, self.dim, &seg.rows, &self.recipes);
                let name = segment_name(seg.id);
                let path = self.dir.join(&name);
                let tmp = self.dir.join(format!("{name}.tmp"));
                let mut f = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
                f.write_all(&bytes).and_then(|()| f.sync_all()).map_err(|e| format!("{}: {e}", tmp.display()))?;
                drop(f);
                std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
                seg.written_len = seg.rows.len();
                seg.dirty = false;
                self.writes += 1;
            }
            if self.skin_dirty {
                let path = self.dir.join(SKIN_FILE);
                let tmp = self.dir.join(format!("{SKIN_FILE}.tmp"));
                let mut f = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
                f.write_all(self.skin.encode().as_bytes()).and_then(|()| f.sync_all()).map_err(|e| format!("{}: {e}", tmp.display()))?;
                drop(f);
                std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
                self.skin_dirty = false;
            }
            Ok(())
        })();
        match r {
            Ok(()) => self.fault = None,
            Err(e) => self.fault = Some(e),
        }
    }
}

fn segment_name(id: u64) -> String {
    if id == 0 { FILE.to_owned() } else { format!("seg-{id:06}.bin") }
}

fn parse_segment_id(name: &str) -> Option<u64> {
    if name == FILE {
        Some(0)
    } else {
        name.strip_prefix("seg-")?.strip_suffix(".bin")?.parse::<u64>().ok().filter(|&id| id > 0)
    }
}

fn list_segments(dir: &Path) -> Vec<(u64, PathBuf)> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(id) = parse_segment_id(&name) {
            out.push((id, entry.path()));
        }
    }
    out.sort_by_key(|(id, _)| *id);
    out
}

/// The header lines with `dim` and `rows` added, sorted.
fn header_text(header: &Header, dim: usize, rows: usize) -> String {
    let mut h = header.clone();
    h.insert("dim".into(), dim.to_string());
    h.insert("rows".into(), rows.to_string());
    h.iter().map(|(k, v)| format!("{k}={}\n", v.replace('\n', " "))).collect()
}

/// The header of a segment whose rows are not all of the running recipe (Roadmap Phase 4
/// (2026-11-24)): the running embedder's fields without the recipe fields, `row_recipe`, and the
/// table of recipes (`recipe.<n>`, the index each record's byte names).
fn row_recipe_header(header: &Header, recipes: &[String]) -> Header {
    let mut h: Header = header.iter().filter(|(k, _)| !RECIPE_FIELDS.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect();
    h.insert(ROW_RECIPE_KEY.to_owned(), ROW_RECIPE.to_owned());
    for (i, name) in recipes.iter().enumerate() {
        h.insert(format!("recipe.{i}"), name.clone());
    }
    h
}

/// The file's bytes. A segment whose rows are all of the running recipe is written exactly as
/// before row recipes; any other carries one recipe byte per row (Roadmap Phase 4 (2026-11-24)).
fn encode(header: &Header, dim: usize, rows: &[(String, Vec<u16>, u8)], recipes: &[String]) -> Vec<u8> {
    let per_row = rows.iter().any(|(_, _, r)| *r != 0);
    let mut body = Vec::with_capacity(rows.len() * (KEY + usize::from(per_row) + dim * 2));
    for (k, v, r) in rows {
        body.extend_from_slice(k.as_bytes());
        if per_row {
            body.push(*r);
        }
        for h in v {
            body.extend_from_slice(&h.to_le_bytes());
        }
    }
    let head = if per_row { header_text(&row_recipe_header(header, recipes), dim, rows.len()) } else { header_text(header, dim, rows.len()) };
    let mut out = Vec::with_capacity(8 + 4 + head.len() + KEY + body.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&u32::try_from(head.len()).unwrap_or(0).to_le_bytes());
    out.extend_from_slice(head.as_bytes());
    out.extend_from_slice(nmhash::sha256_bytes(&body).as_bytes());
    out.extend_from_slice(&body);
    out
}

/// Every row of a store file whose header equals `want` in every field but the recipe fields,
/// or the reason it cannot be used. Each row comes with the index of its recipe in `table` (the
/// store's; names it does not hold yet are added, at most `MAX_RECIPES`, a row of a recipe past
/// that being left out). A file without `row_recipe` gives every row the recipe its header
/// names (`recipe_name`), as before row recipes (Roadmap Phase 4 (2026-11-24)).
fn parse(bytes: &[u8], want: &Header, table: &mut Vec<String>) -> Result<(usize, Rows), String> {
    if bytes.get(..8) != Some(MAGIC.as_slice()) {
        return Err("not a dense store file (no NMDENSE1 mark)".into());
    }
    let n = bytes.get(8..12).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize).ok_or("cut off inside the header")?;
    let head = bytes.get(12..12 + n).ok_or("cut off inside the header")?;
    let head = std::str::from_utf8(head).map_err(|_| "header is not text")?;
    let mut got: Header = head.lines().filter_map(|l| l.split_once('=')).map(|(k, v)| (k.to_owned(), v.to_owned())).collect();
    let num = |got: &mut Header, k: &str| got.remove(k).and_then(|v| v.parse::<usize>().ok()).ok_or(format!("header has no {k}"));
    let dim = num(&mut got, "dim")?;
    let rows = num(&mut got, "rows")?;
    let per_row = match got.remove(ROW_RECIPE_KEY) {
        None => false,
        Some(v) if v == ROW_RECIPE => true,
        Some(v) => return Err(format!("header mismatch: the file's rows name their recipe as {v}, which this build does not read")),
    };
    let names: Vec<String> = if per_row {
        let mut names = Vec::new();
        while let Some(name) = got.remove(&format!("recipe.{}", names.len())) {
            names.push(name);
        }
        if names.is_empty() {
            return Err("header names row recipes but no recipe.0".into());
        }
        names
    } else {
        let text = got.remove("recipe");
        let parts = got.remove("parts_within");
        vec![recipe_name(text.as_deref(), parts.as_deref())]
    };
    let is_recipe = |k: &str| RECIPE_FIELDS.contains(&k);
    if let Some((k, v)) = want.iter().filter(|(k, _)| !is_recipe(k)).find(|(k, v)| got.get(*k) != Some(*v)) {
        return Err(format!("header mismatch: {k} is {} in the file, {v} for the running embedder", got.get(k).map_or("absent", String::as_str)));
    }
    if let Some(k) = got.keys().find(|k| !want.contains_key(*k) || is_recipe(k)) {
        return Err(format!("header mismatch: the file names {k}, which the running embedder does not"));
    }
    let sum = bytes.get(12 + n..12 + n + KEY).ok_or("cut off before the body")?;
    let body = bytes.get(12 + n + KEY..).unwrap_or_default();
    let size = KEY + usize::from(per_row) + dim * 2;
    if body.len() != rows * size {
        return Err(format!("cut off: the header names {rows} rows ({} bytes), the body holds {} bytes", rows * size, body.len()));
    }
    if nmhash::sha256_bytes(body).as_bytes() != sum {
        return Err("checksum mismatch: the body is not the one the header was written with".into());
    }
    let index: Vec<Option<u8>> = names
        .iter()
        .map(|name| match table.iter().position(|t| t == name) {
            Some(i) => u8::try_from(i).ok(),
            None if table.len() < MAX_RECIPES => {
                table.push(name.clone());
                u8::try_from(table.len() - 1).ok()
            }
            None => None,
        })
        .collect();
    let mut out = Vec::with_capacity(rows);
    for rec in body.chunks_exact(size.max(1)) {
        let (k, rest) = rec.split_at(KEY);
        let k = std::str::from_utf8(k).ok().filter(|k| k.bytes().all(|c| c.is_ascii_hexdigit())).ok_or("a row hash is not hex")?;
        let (at, v) = if per_row { rest.split_at(1) } else { (&[0_u8][..], rest) };
        let at = usize::from(at.first().copied().unwrap_or(0));
        let r = match index.get(at) {
            Some(Some(r)) => *r,
            Some(None) => continue,
            None => return Err(format!("a row names recipe.{at}, which the header does not")),
        };
        out.push((k.to_owned(), v.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect(), r));
    }
    Ok((dim, out))
}

/// Self-check: binary16 round trip, a store written and read back, a header mismatch and a cut
/// file each ignored with a named reason, compaction past half dead rows, clearing, the setting,
/// and all five H32 dead-skin biomimetic shedding behaviors.
pub(crate) fn check() -> Result<(), String> {
    for (x, h) in [(1.0_f32, 0x3c00_u16), (-2.0, 0xc000), (0.5, 0x3800), (65504.0, 0x7bff), (1e-7, 0x0002), (0.0, 0)] {
        if to_half(x) != h || (from_half(h) - x).abs() > x.abs() * 1e-3 + 1e-7 {
            return Err(format!("nmsubs.dense_store: binary16 of {x} is {:#06x}, want {h:#06x}", to_half(x)));
        }
    }
    if to_half(1.0 + 1.0 / 2048.0) != 0x3c00 || to_half(1.0 + 3.0 / 2048.0) != 0x3c02 {
        return Err("nmsubs.dense_store: binary16 ties do not round to even".into());
    }
    let dir = std::env::temp_dir().join(format!("nano-dense-store-check-{}-{}", std::process::id(), nmtype::now_ms()));
    let r = check_in(&dir).and_then(|()| check_dead_skin(&dir.join("h32")));
    let _ = std::fs::remove_dir_all(&dir);
    r
}

fn check_in(dir: &Path) -> Result<(), String> {
    let header: Header = [("weights".to_owned(), "abc".to_owned())].into();
    let (a, b) = ("a".repeat(64), "b".repeat(64));
    let live: BTreeSet<&str> = [a.as_str(), b.as_str()].into();
    let (mut st, got, found) = Store::open(dir, header.clone(), &live);
    if found != Found::Empty || !got.is_empty() {
        return Err(format!("nmsubs.dense_store: empty folder found {found:?}"));
    }
    st.add(&a, &[0.5, -1.0]);
    st.add(&b, &[0.25, 2.0]);
    st.flush();
    let (_, got, found) = Store::open(dir, header.clone(), &[a.as_str()].into());
    if found != (Found::Read { rows: 2, live: 1 }) || got != vec![(a.clone(), vec![0.5, -1.0])] {
        return Err(format!("nmsubs.dense_store: read back {found:?} {got:?}"));
    }
    let other: Header = [("weights".to_owned(), "xyz".to_owned())].into();
    if !matches!(Store::open(dir, other, &live).2, Found::Ignored(w) if w.contains("weights")) {
        return Err("nmsubs.dense_store: a header mismatch was not named".into());
    }
    let path = dir.join(FILE);
    let full = std::fs::read(&path).map_err(|e| e.to_string())?;
    std::fs::write(&path, &full[..full.len() - 3]).map_err(|e| e.to_string())?;
    if !matches!(Store::open(dir, header.clone(), &live).2, Found::Ignored(w) if w.starts_with("cut off")) {
        return Err("nmsubs.dense_store: a cut file was not named".into());
    }
    std::fs::write(&path, &full).map_err(|e| e.to_string())?;
    let (mut st, _, _) = Store::open(dir, header, &live);
    if st.compact(&[a.as_str(), b.as_str()].into()) != 0 || st.compact(&BTreeSet::new()) != 2 || st.rows() != 0 {
        return Err("nmsubs.dense_store: compaction did not wait for half dead rows".into());
    }
    if clear(dir)? == 0 || dir.join(FILE).exists() {
        return Err("nmsubs.dense_store: clear left the file".into());
    }
    let on = cache_on(dir);
    std::fs::write(dir.join("settings.json"), format!("{{\"{SETTING}\": \"off\"}}")).map_err(|e| e.to_string())?;
    if !on || cache_on(dir) {
        return Err("nmsubs.dense_store: the setting is not on by default or not off when set off".into());
    }
    Ok(())
}

/// Verifies all five H32 dead-skin behaviors:
/// (1) free barrier revival on undo / branch switch,
/// (2) deleted-file rows shed before edited-file rows,
/// (3) bounded micro-shedding drops at most `shed_batch` rows per step,
/// (4) 100%-dead expired old segments drop whole with zero rewrite of live segments,
/// (5) age calibration adapts TTL from observed revival ages and rate.
fn check_dead_skin(dir: &Path) -> Result<(), String> {
    let header: Header = [("weights".to_owned(), "h32".to_owned())].into();
    let keys: Vec<String> = (0..6).map(|i| format!("{i:064x}")).collect();
    let f_edit = file_tag("src/edited.rs");
    let f_del = file_tag("src/deleted.rs");
    let f_live = file_tag("src/live.rs");

    // 2 rows per segment -> 3 segments: [0,1] in vectors.bin, [2,3] in seg-000001.bin, [4,5] in seg-000002.bin.
    let (st, _, _) = Store::open_at(dir, header.clone(), &BTreeSet::new(), 1_000);
    let mut st = st.with_limits(2, 1);
    st.add_tagged(&keys[0], &[1.0, 0.0], Some(&f_del));
    st.add_tagged(&keys[1], &[0.0, 1.0], Some(&f_del));
    st.add_tagged(&keys[2], &[0.5, 0.5], Some(&f_edit));
    st.add_tagged(&keys[3], &[-0.5, 0.5], Some(&f_del));
    st.add_tagged(&keys[4], &[0.25, -0.25], Some(&f_live));
    st.add_tagged(&keys[5], &[-0.25, -0.25], Some(&f_live));
    st.flush_at(1_000);
    if st.segment_count() != 3 || !dir.join(FILE).exists() || !dir.join("seg-000001.bin").exists() || !dir.join("seg-000002.bin").exists() {
        return Err(format!("nmsubs.dense_store(H32): expected 3 segments, got {}", st.segment_count()));
    }

    // (1) Move keys[2] (edited file) into the dead-skin barrier layer, then undo -> free revival!
    let t1 = 2_000_u64;
    let live_after_edit: BTreeSet<&str> = [keys[0].as_str(), keys[1].as_str(), keys[3].as_str(), keys[4].as_str(), keys[5].as_str()].into();
    let tags_all: BTreeSet<String> = [f_edit.clone(), f_del.clone(), f_live.clone()].into();
    let rep1 = st.shed(&live_after_edit, Some(&tags_all), t1);
    if rep1.marked_dead != 1 || rep1.shed_rows != 0 || st.rows() != 6 {
        return Err(format!("nmsubs.dense_store(H32): newly dead row did not stay in barrier: {rep1:?}"));
    }
    let t_undo = t1 + 60_000;
    let all_live: BTreeSet<&str> = keys.iter().map(String::as_str).collect();
    let rep_undo = st.shed(&all_live, Some(&tags_all), t_undo);
    if rep_undo.revived != vec![(keys[2].clone(), vec![0.5, 0.5])] || st.skin_stats().revivals_total != 1 {
        return Err(format!("nmsubs.dense_store(H32): barrier revival failed: {rep_undo:?}"));
    }

    // (2) + (3) + (4): Delete src/deleted.rs (keys 0, 1, 3) and edit src/edited.rs (key 2).
    // Segment 0 holds [keys 0, 1] (both deleted-file) -> drops whole in O(1) once deleted TTL passes!
    // Segment 1 holds [key 2 (edited), key 3 (deleted)] -> key 3 sheds BEFORE key 2, bounded by shed_batch=1!
    let t_kill = t_undo + 1_000;
    let live_only: BTreeSet<&str> = [keys[4].as_str(), keys[5].as_str()].into();
    let tags_after_del: BTreeSet<String> = [f_edit.clone(), f_live.clone()].into();
    let rep_kill = st.shed(&live_only, Some(&tags_after_del), t_kill);
    if rep_kill.marked_dead != 4 || st.skin_stats().dead_deleted_file != 3 || st.skin_stats().dead_edited_file != 1 {
        return Err(format!("nmsubs.dense_store(H32): deleted/edited classification wrong: {:?}", st.skin_stats()));
    }

    // Advance past deleted-file TTL (1/4 of 7 days), still before edited-file TTL (7 days).
    let t_del_exp = t_kill + st.skin_stats().ttl_deleted_ms + 1;
    let writes_before = st.writes;
    let rep_del = st.shed(&live_only, Some(&tags_after_del), t_del_exp);
    // Segment 0 ([0,1]) must drop whole (zero rewrite!), and key 3 (deleted) in segment 1 must shed while key 2 (edited) stays!
    if rep_del.segments_dropped != 1 || dir.join(FILE).exists() || rep_del.shed_rows != 3 || rep_del.segments_rewritten != 1 {
        return Err(format!("nmsubs.dense_store(H32): whole-segment drop / deleted-first failed: {rep_del:?}"));
    }
    if st.writes != writes_before + 1 || !st.keys.contains(&keys[2]) || st.keys.contains(&keys[3]) {
        return Err("nmsubs.dense_store(H32): live segment was needlessly rewritten or edited-file row shed too early".into());
    }
    // Re-open from disk with segment 0 already unlinked -> must still load live keys [4, 5] seamlessly!
    let (_, reloaded, found) = Store::open_at(dir, header, &live_only, t_del_exp + 10);
    if found != (Found::Read { rows: 3, live: 2 }) || reloaded.len() != 2 {
        return Err(format!("nmsubs.dense_store(H32): reopen after segment drop failed: {found:?} len={}", reloaded.len()));
    }

    // (5) Age calibration: with >= 16 deaths and 4 revivals at 30 minutes, TTL adapts from 7 days to ~60 minutes.
    let mut cal = SkinState { deaths_total: 20, ..SkinState::default() };
    for _ in 0..4 {
        cal.record_revival(30 * 60 * 1000);
    }
    let adapted = cal.ttl_edit_ms();
    if adapted != 60 * 60 * 1000 {
        return Err(format!("nmsubs.dense_store(H32): expected calibrated TTL 3600000 ms, got {adapted}"));
    }
    // Rare revivals (< 2% with >= 16 deaths and < 4 revivals) decay TTL to MIN_TTL_MS.
    let cold = SkinState { deaths_total: 100, revivals_total: 1, ..SkinState::default() };
    if cold.ttl_edit_ms() != MIN_TTL_MS {
        return Err(format!("nmsubs.dense_store(H32): expected cold TTL {MIN_TTL_MS}, got {}", cold.ttl_edit_ms()));
    }
    // Clear removes all segments + skin.json.
    if clear(dir)? == 0 || dir.join(SKIN_FILE).exists() || dir.join("seg-000002.bin").exists() {
        return Err("nmsubs.dense_store(H32): clear left segment or skin.json behind".into());
    }
    Ok(())
}

/// Row recipes (Roadmap Phase 4 (2026-11-24)): a file written before them reads as its header
/// says, a recipe that differs marks only its rows for embedding, the new header reads back, and
/// a piece with two rows is read from one.
#[cfg(test)]
mod tests {
    use super::super::{RECIPE, RECIPE_LEAD};
    use super::*;
    use std::sync::Arc;

    fn folder(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("nano-dense-recipe-{name}-{}-{}", std::process::id(), nmtype::now_ms()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test folder");
        dir
    }

    fn header(recipe: &str, parts: Option<&str>) -> Header {
        let mut h: Header = [("weights".to_owned(), "recipe-test".to_owned()), ("recipe".to_owned(), recipe.to_owned())].into();
        if let Some(n) = parts {
            h.insert("parts_within".to_owned(), n.to_owned());
        }
        h
    }

    fn key(i: u32) -> String {
        format!("{i:064x}")
    }

    fn row(i: u32, a: f32, b: f32) -> (String, Vec<u16>, u8) {
        (key(i), vec![to_half(a), to_half(b)], 0)
    }

    fn head_of(bytes: &[u8]) -> String {
        let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        String::from_utf8_lossy(&bytes[12..12 + n]).into_owned()
    }

    #[test]
    fn old_format_header_reads_as_plain() {
        let dir = folder("old");
        let plain = header(RECIPE, None);
        let bytes = encode(&plain, 2, &[row(1, 0.5, -1.0), row(2, 0.25, 2.0)], &[]);
        // The file a build before row recipes wrote: no `row_recipe`, the recipe in the header.
        let head = head_of(&bytes);
        assert!(head.contains(&format!("recipe={RECIPE}\n")) && !head.contains(ROW_RECIPE_KEY), "{head}");
        std::fs::write(dir.join(FILE), &bytes).expect("write");
        let (k1, k2) = (key(1), key(2));
        let live: BTreeSet<&str> = [k1.as_str(), k2.as_str()].into();

        // Under the plain recipe: exactly today's behaviour, every row loaded as plain.
        let (st, got, found) = Store::open(&dir, plain.clone(), &live);
        assert_eq!(found, Found::Read { rows: 2, live: 2 });
        assert_eq!(got.len(), 2);
        assert_eq!(st.row_recipe(&k1), Some(RECIPE));
        assert!(st.current(&k1, true) && st.current(&k2, false));
        assert_eq!(st.other_recipe_rows(), 0);

        // Under the lead recipe the file is still read (no longer ignored), its rows plain: a
        // piece with no lead keeps its row, a piece with a lead is to be embedded again.
        let (st, got, found) = Store::open(&dir, header(RECIPE_LEAD, None), &live);
        assert_eq!(found, Found::Read { rows: 2, live: 2 });
        assert_eq!(got.len(), 2);
        assert_eq!(st.row_recipe(&k1), Some(RECIPE));
        assert!(!st.current(&k1, true) && st.current(&k1, false));

        // Under parts no plain row gives the running text: none is handed out.
        let (st, got, _) = Store::open(&dir, header(RECIPE, Some("512")), &live);
        assert!(got.is_empty());
        assert!(!st.current(&k1, false) && !st.current(&k2, false));

        // Another embedder is still not mixed in.
        let mut other = plain;
        other.insert("weights".to_owned(), "another".to_owned());
        assert!(matches!(Store::open(&dir, other, &live).2, Found::Ignored(w) if w.contains("weights")));

        // Opening never changed the file.
        assert_eq!(std::fs::read(dir.join(FILE)).expect("read"), bytes);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recipe_mismatch_marks_only_mismatched_rows() {
        let dir = folder("mismatch");
        let (k1, k2, k3) = (key(1), key(2), key(3));
        let live: BTreeSet<&str> = [k1.as_str(), k2.as_str(), k3.as_str()].into();
        let (mut st, _, _) = Store::open(&dir, header(RECIPE, None), &live);
        st.add(&k1, &[1.0, 0.0]);
        st.add(&k2, &[0.0, 1.0]);
        st.add(&k3, &[0.5, 0.5]);
        st.flush();

        // Served with the lead: only k1 has a lead, so only its row is to be embedded again.
        let lead = header(RECIPE_LEAD, None);
        let has_lead = |k: &str| k == k1.as_str();
        let (mut st, _, _) = Store::open(&dir, lead.clone(), &live);
        let again: Vec<&str> = live.iter().copied().filter(|k| !st.current(k, has_lead(k))).collect();
        assert_eq!(again, vec![k1.as_str()]);
        st.add(&k1, &[-1.0, 0.0]);
        // k2's plain row stands for it, but a new row of the running recipe replaces it; a
        // second row of the running recipe is then skipped.
        st.add(&k2, &[2.0, 2.0]);
        st.add(&k2, &[3.0, 3.0]);
        st.flush();

        let (st, got, found) = Store::open(&dir, lead, &live);
        assert_eq!(found, Found::Read { rows: 3, live: 3 });
        assert!(live.iter().all(|k| st.current(k, has_lead(k))));
        assert_eq!(st.row_recipe(&k1), Some(RECIPE_LEAD));
        assert_eq!(st.row_recipe(&k2), Some(RECIPE_LEAD));
        assert_eq!(st.row_recipe(&k3), Some(RECIPE));
        let got: BTreeMap<String, Vec<f32>> = got.into_iter().collect();
        assert_eq!(got.get(&k1), Some(&vec![-1.0, 0.0]));
        assert_eq!(got.get(&k2), Some(&vec![2.0, 2.0]));
        assert_eq!(got.get(&k3), Some(&vec![0.5, 0.5]));

        // Back to plain: only the piece with a lead is embedded again.
        let (st, _, _) = Store::open(&dir, header(RECIPE, None), &live);
        let again: Vec<&str> = live.iter().copied().filter(|k| !st.current(k, has_lead(k))).collect();
        assert_eq!(again, vec![k1.as_str()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lane_embeds_only_rows_of_another_recipe() {
        let dir = folder("lane");
        let embed: crate::Embed = Arc::new(|_: &str| vec![1.0]);
        let full = crate::from_files(
            &[("a.rs", "fn parse_header() { read the header bytes }"), ("b.rs", "fn other_thing() { write the footer }")],
            Arc::clone(&embed),
        );
        let small = crate::from_files(&[("a.rs", "fn parse_header() { read the header bytes }")], embed);
        let lane = |x: f32| crate::dense::DenseLane::in_ram(Arc::new(move |_: &str| Some(vec![x, 1.0])), Arc::new(|_: &str| Some(vec![1.0, 0.0])));
        // Arbitrary recipe names: no lead makes them give one text.
        let (ha, hb) = (header("recipe A", None), header("recipe B", None));
        let first = lane(0.25);
        first.keep_on_disk(&dir, ha, &full);
        assert_eq!(first.fill(&full, &|| true), full.chunks.len());
        // Recipe B over a.rs only: its rows embedded again; b.rs keeps its recipe A rows.
        let second = lane(0.5);
        second.keep_on_disk(&dir, hb.clone(), &small);
        assert_eq!(second.fill(&small, &|| true), small.chunks.len());
        // Recipe B over both: a mixed store; only b.rs is embedded.
        let third = lane(0.75);
        third.keep_on_disk(&dir, hb, &full);
        let b = full.files.iter().position(|f| f.path == "b.rs");
        let b_chunks = full.chunks.iter().filter(|c| Some(c.file) == b).count();
        assert!(b_chunks > 0);
        assert_eq!(third.fill(&full, &|| true), b_chunks);
        assert!(third.ready(&full));
        // Each piece read from one row of the running recipe: a.rs from the second lane, b.rs
        // from the third.
        for c in &full.chunks {
            let want = if Some(c.file) == b { 0.75 } else { 0.5 };
            assert_eq!(third.vector(&c.key).map(|v| v[0]), Some(want));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The lane's and the store's section checks (stores written, read back, kept across a
    /// restart, the five dead-skin behaviours) still pass with row recipes.
    #[test]
    fn section_checks_pass_with_row_recipes() {
        assert_eq!(check(), Ok(()));
        assert_eq!(super::super::check(), Ok(()));
    }

    #[test]
    fn row_recipe_header_round_trip() {
        let dir = folder("round");
        let parts = header(RECIPE, Some("512"));
        let running = recipe_name(Some(RECIPE), Some("512"));
        // A segment of the running recipe only is written exactly as before row recipes.
        let (mut st, _, _) = Store::open(&dir, parts.clone(), &BTreeSet::new());
        st.add(&key(3), &[0.5, 0.5]);
        st.flush();
        let bytes = std::fs::read(dir.join(FILE)).expect("read");
        assert_eq!(bytes, encode(&parts, 2, &[row(3, 0.5, 0.5)], &[]));
        assert!(head_of(&bytes).contains("parts_within=512\n"));
        std::fs::remove_file(dir.join(FILE)).expect("remove");

        // A plain file, then a row of the running recipe (parts) appended: rows of two recipes.
        std::fs::write(dir.join(FILE), encode(&header(RECIPE, None), 2, &[row(1, 1.0, 0.0), row(2, 0.0, 1.0)], &[])).expect("write");
        let (k1, k2, k3) = (key(1), key(2), key(3));
        let live: BTreeSet<&str> = [k1.as_str(), k2.as_str(), k3.as_str()].into();
        let (mut st, got, _) = Store::open(&dir, parts.clone(), &live);
        assert!(got.is_empty());
        assert_eq!(st.other_recipe_rows(), 2);
        st.add(&k3, &[0.5, -0.5]);
        st.flush();
        let bytes = std::fs::read(dir.join(FILE)).expect("read");
        let head = head_of(&bytes);
        assert!(head.contains(&format!("{ROW_RECIPE_KEY}={ROW_RECIPE}\n")), "{head}");
        assert!(head.contains(&format!("recipe.0={running}\n")) && head.contains(&format!("recipe.1={RECIPE}\n")), "{head}");
        assert!(!head.lines().any(|l| l.starts_with("recipe=") || l.starts_with("parts_within=")), "{head}");

        // Read back: each row's recipe and values as written, into a fresh table.
        let mut table = vec![RECIPE.to_owned()];
        let (dim, rows) = parse(&bytes, &header(RECIPE, None), &mut table).expect("parse");
        assert_eq!(dim, 2);
        let named: Vec<(String, &str, Vec<f32>)> =
            rows.iter().map(|(k, v, r)| (k.clone(), table[usize::from(*r)].as_str(), v.iter().map(|h| from_half(*h)).collect())).collect();
        assert_eq!(
            named,
            vec![(k1.clone(), RECIPE, vec![1.0, 0.0]), (k2.clone(), RECIPE, vec![0.0, 1.0]), (k3.clone(), running.as_str(), vec![0.5, -0.5])]
        );
        let (st, got, found) = Store::open(&dir, parts, &live);
        assert_eq!(found, Found::Read { rows: 3, live: 1 });
        assert_eq!(got, vec![(k3.clone(), vec![0.5, -0.5])]);
        assert_eq!(st.row_recipe(&k1), Some(RECIPE));
        assert_eq!(st.row_recipe(&k3), Some(running.as_str()));
        let (st, got, _) = Store::open(&dir, header(RECIPE, None), &live);
        assert_eq!(got.len(), 2);
        assert!(st.current(&k1, true) && !st.current(&k3, false));

        // A record naming a recipe the header lacks is refused, not misread.
        let mut bad = bytes.clone();
        let at = 12 + head.len();
        bad[at + KEY + KEY] = 7;
        let body = bad[at + KEY..].to_vec();
        bad[at..at + KEY].copy_from_slice(nmhash::sha256_bytes(&body).as_bytes());
        assert!(parse(&bad, &header(RECIPE, None), &mut vec![RECIPE.to_owned()]).is_err_and(|e| e.contains("recipe.7")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_rows_of_one_piece_read_as_one() {
        let dir = folder("two");
        let k1 = key(1);
        let live: BTreeSet<&str> = [k1.as_str()].into();
        // A plain row, then a lead row of the same piece in a later segment (a refresh under way).
        std::fs::write(dir.join(FILE), encode(&header(RECIPE, None), 2, &[row(1, 1.0, 0.0)], &[])).expect("write");
        std::fs::write(dir.join(segment_name(1)), encode(&header(RECIPE_LEAD, None), 2, &[row(1, 0.0, 1.0)], &[])).expect("write");
        let (st, got, _) = Store::open(&dir, header(RECIPE_LEAD, None), &live);
        assert_eq!(got, vec![(k1.clone(), vec![0.0, 1.0])]);
        assert_eq!(st.rows(), 1);
        assert_eq!(st.row_recipe(&k1), Some(RECIPE_LEAD));
        let (st, got, _) = Store::open(&dir, header(RECIPE, None), &live);
        assert_eq!(got, vec![(k1.clone(), vec![1.0, 0.0])]);
        assert_eq!(st.rows(), 1);
        assert_eq!(st.row_recipe(&k1), Some(RECIPE));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A folder written for one embedder, opened for another (its served values moved): the old
    /// rows are refused, the first row written replaces the primary file, and the folder then
    /// reads as the new embedder's with none of the old rows mixed in.
    #[test]
    fn another_embedder_replaces_the_primary() {
        let dir = folder("rebind");
        let (k1, k2) = (key(1), key(2));
        let live: BTreeSet<&str> = [k1.as_str(), k2.as_str()].into();
        let mut old = header(RECIPE, None);
        old.insert("parameters".to_owned(), "open weights a".to_owned());
        let mut new = header(RECIPE, None);
        new.insert("parameters".to_owned(), "open weights a with overlay b".to_owned());
        let mut st = Store::open(&dir, old.clone(), &live).0.with_limits(1, SHED_BATCH);
        st.add(&k1, &[1.0, 0.0]);
        st.add(&k2, &[0.0, 1.0]);
        st.flush();
        assert!(dir.join(segment_name(1)).exists(), "two segments of one row each");
        let (mut st, got, found) = Store::open(&dir, new.clone(), &live);
        assert!(got.is_empty());
        assert!(matches!(&found, Found::Ignored(w) if w.contains("parameters")), "{found:?}");
        st.add(&k1, &[0.5, 0.5]);
        st.flush();
        let (_, got, found) = Store::open(&dir, new, &live);
        assert_eq!(got, vec![(k1.clone(), vec![0.5, 0.5])], "{found:?}");
        let (_, got, found) = Store::open(&dir, old, &live);
        assert!(got.is_empty() && matches!(found, Found::Ignored(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
