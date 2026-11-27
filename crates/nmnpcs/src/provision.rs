//! Provision (guide 9.3 responsibilities 7, 9, 10, 12, 14, 15: provisioning, sharing and
//! reference tracking, residency, lifecycle, load and unload, reclamation; PLAN 4c). A
//! tensor's file range is read once into an `Arc<[u8]>` (the mini's zero copy: one read, many
//! views, no memory-mapping crate); a cluster is a `View` over it; acquiring the same cluster
//! again returns the same `Arc<View>` and counts a reference; a view retires only at zero
//! references, and the tensor's bytes are dropped when no view holds them. Device residency
//! is recorded here by the engine (the bytes a cluster holds on the GPU), so one table answers
//! where every cluster lives.
use crate::decompose::{ParameterCluster, Region};
use crate::lifecycle::{State, Tracker};
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmnpcs.provision", FaultKind::Resource, m)
}

/// Where a cluster's bytes are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Storage {
    /// Not read yet, or its host copy dropped and not on the GPU: the file range.
    Mapped { file: PathBuf, offset: u64, len: u64 },
    /// In RAM (the tensor's shared bytes).
    Resident,
    /// On the GPU (device bytes); its host copy is in RAM or dropped (`Provisioner::host_held`).
    Device { bytes: u64 },
}

/// A view's host copy (2026-11-27, weight placement): `kept` while it is resident, `None` once
/// dropped. No weak pointer is left to a dropped copy: an `Arc<[u8]>` keeps its whole buffer
/// allocated while any `Weak` to it lives, so a weak entry would hold the RAM the drop meant to free.
#[derive(Debug)]
struct Slot {
    kept: Option<Arc<[u8]>>,
}

/// One provisioned cluster: identity, the tensor's shared bytes (droppable once the device holds
/// a copy, `Provisioner::drop_host`), references, device bytes.
#[derive(Debug)]
pub struct View {
    pub cluster: ParameterCluster,
    /// The file that holds the tensor (read again by `tensor_bytes` when the copy is dropped).
    src: PathBuf,
    slot: RwLock<Slot>,
    refs: AtomicUsize,
    device: AtomicUsize,
    hash: OnceLock<String>,
}
impl View {
    /// The whole tensor's bytes this view is a region of, while its host copy is kept; `None` once
    /// dropped. `Provisioner::bytes` reads them again from their file.
    #[must_use]
    pub fn bytes(&self) -> Option<Arc<[u8]>> {
        self.slot.read().ok()?.kept.clone()
    }
    /// `bytes`, or, when the host copy is dropped, the bytes read from its file range for this
    /// caller alone (a copy held by content hash first; a read whose hash differs from the
    /// recorded one, or that fails, gives an empty slice and says so). The engine reads through
    /// `Provisioner::bytes`, which also checks the file against its first read.
    #[must_use]
    pub fn tensor_bytes(&self) -> Arc<[u8]> {
        if let Some(b) = self.bytes() {
            return b;
        }
        if !self.cluster.sha256.is_empty()
            && let Some(b) = held(&self.cluster.sha256)
        {
            return b;
        }
        let read = || -> std::io::Result<Vec<u8>> {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = std::fs::File::open(&self.src)?;
            f.seek(SeekFrom::Start(self.cluster.offset))?;
            let mut buf = vec![0_u8; usize::try_from(self.cluster.len).map_err(std::io::Error::other)?];
            f.read_exact(&mut buf)?;
            Ok(buf)
        };
        match read() {
            Ok(b) if self.cluster.sha256.is_empty() || nmhash::sha256_bytes(&b) == self.cluster.sha256 => Arc::from(b),
            Ok(_) => {
                eprintln!("nmnpcs: {}: read again from {}, the bytes do not match its content hash", self.cluster.tensor, self.src.display());
                Arc::from(Vec::new())
            }
            Err(e) => {
                eprintln!("nmnpcs: {}: not read again from {}: {e}", self.cluster.tensor, self.src.display());
                Arc::from(Vec::new())
            }
        }
    }
    /// Whether the host copy is kept resident (not dropped).
    #[must_use]
    pub fn resident(&self) -> bool {
        self.slot.read().is_ok_and(|s| s.kept.is_some())
    }
    #[must_use]
    pub fn refs(&self) -> usize {
        self.refs.load(Ordering::SeqCst)
    }
    /// Bytes this cluster holds on the GPU (0 when not resident there).
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.device.load(Ordering::SeqCst)
    }
    /// SHA-256 of the cluster's own bytes in region order (taken once, on first request while
    /// the bytes are in RAM; empty while the host copy is dropped and not yet taken).
    pub fn hash(&self) -> &str {
        if let Some(h) = self.hash.get() {
            return h;
        }
        let Some(bytes) = self.bytes() else { return "" };
        self.hash.get_or_init(|| {
            let (r, c) = self.cluster.dims();
            let w = bytes.len() / self.cluster.shape.iter().product::<usize>().max(1);
            let mut own = Vec::with_capacity(r * c * w);
            for row in 0..r {
                own.extend_from_slice(self.row_in(&bytes, row, w));
            }
            nmhash::sha256_bytes(&own)
        })
    }
    /// Row `row` of the cluster's own block in `bytes` (this view's tensor bytes), `w` bytes per
    /// element.
    #[must_use]
    pub fn row_in<'a>(&self, bytes: &'a [u8], row: usize, w: usize) -> &'a [u8] {
        let full_cols = self.cluster.shape.get(1).copied().unwrap_or_else(|| self.cluster.shape.first().copied().unwrap_or(0));
        let (start, n) = match &self.cluster.region {
            Region::Whole => (row * full_cols, full_cols),
            Region::Rows(r) => ((r.start + row) * full_cols, full_cols),
            Region::Cols(c) => (row * full_cols + c.start, c.len()),
        };
        bytes.get(start * w..(start + n) * w).unwrap_or(&[])
    }
}

/// The provisioner of one population: the tensor index, the live views and the lifecycle.
pub struct Provisioner {
    pub model: String,
    dir: PathBuf,
    views: Mutex<BTreeMap<String, Arc<View>>>,
    tensors: Mutex<BTreeMap<String, Weak<[u8]>>>,
    /// Length and modification time of each file at its first read: a read again on demand
    /// checks them, so a changed file is refused, never mixed with the bytes read before.
    stamps: Mutex<BTreeMap<PathBuf, (u64, Option<std::time::SystemTime>)>>,
    reads: AtomicUsize,
    rereads: AtomicUsize,
    shared_hits: AtomicUsize,
    pub lifecycle: Tracker,
}

/// Tensor bytes in RAM by content hash, across every population of the process: a tensor that
/// two cartridges carry with the same hash is read once and held once while either uses it.
static BY_HASH: Mutex<BTreeMap<String, Weak<[u8]>>> = Mutex::new(BTreeMap::new());

/// The bytes held under a content hash, if any population still holds them.
#[must_use]
pub fn held(sha256: &str) -> Option<Arc<[u8]>> {
    BY_HASH.lock().ok()?.get(sha256).and_then(Weak::upgrade)
}

/// Offer bytes under their content hash, so a later tensor with the same hash uses them. The
/// caller keeps them alive; nothing is held here once every holder drops them.
pub fn hold(sha256: &str, bytes: &Arc<[u8]>) {
    if let Ok(mut m) = BY_HASH.lock() {
        m.retain(|_, w| w.strong_count() > 0);
        m.insert(sha256.to_owned(), Arc::downgrade(bytes));
    }
}
impl std::fmt::Debug for Provisioner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Provisioner").field("model", &self.model).finish_non_exhaustive()
    }
}
impl Provisioner {
    /// A provisioner reading the released files in `dir` (the population folder); each
    /// cluster names the file that holds its tensor.
    #[must_use]
    pub fn new(model: &str, dir: &Path, clusters: &[ParameterCluster]) -> Self {
        let lifecycle = Tracker::default();
        for c in clusters {
            lifecycle.register(&c.id);
        }
        Self {
            model: model.to_owned(),
            dir: dir.to_owned(),
            views: Mutex::new(BTreeMap::new()),
            tensors: Mutex::new(BTreeMap::new()),
            stamps: Mutex::new(BTreeMap::new()),
            reads: AtomicUsize::new(0),
            rereads: AtomicUsize::new(0),
            shared_hits: AtomicUsize::new(0),
            lifecycle,
        }
    }
    /// Tensor reads made again on demand after a host copy was dropped (`bytes`).
    #[must_use]
    pub fn rereads(&self) -> usize {
        self.rereads.load(Ordering::SeqCst)
    }
    /// Tensor reads from the file so far (one per tensor while any view holds it).
    #[must_use]
    pub fn reads(&self) -> usize {
        self.reads.load(Ordering::SeqCst)
    }
    /// Tensors this provisioner took from another source's copy by content hash (no file read).
    #[must_use]
    pub fn shared_hits(&self) -> usize {
        self.shared_hits.load(Ordering::SeqCst)
    }
    /// The tensor's bytes: shared while any view or reader holds them, else by content hash, else
    /// read from its file range. `again` marks a read after a host copy was dropped: the file must
    /// have the length and modification time it had at its first read, and a tensor with a content
    /// hash must read back to that hash (an overlay copy no longer held is refused, never served
    /// as the released values).
    fn tensor(&self, c: &ParameterCluster, again: bool) -> Result<Arc<[u8]>, Fault> {
        let mut t = self.tensors.lock().map_err(|_| bad("tensor table lock"))?;
        if let Some(b) = t.get(&c.tensor).and_then(Weak::upgrade) {
            return Ok(b);
        }
        // A tensor whose content hash is already held in RAM by any population is used by
        // reference: one copy for every cartridge that carries it.
        if !c.sha256.is_empty()
            && let Some(b) = held(&c.sha256)
        {
            if !again {
                t.insert(c.tensor.clone(), Arc::downgrade(&b));
            }
            self.shared_hits.fetch_add(1, Ordering::SeqCst);
            return Ok(b);
        }
        let path = self.dir.join(&c.file);
        let mut f = std::fs::File::open(&path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let meta = f.metadata().map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let stamp = (meta.len(), meta.modified().ok());
        {
            let mut s = self.stamps.lock().map_err(|_| bad("file stamp lock"))?;
            match s.get(&path) {
                Some(first) if again && *first != stamp => {
                    return Err(bad(format!(
                        "{}: the file changed since it was loaded ({} bytes then, {} now); the host copy of {} is not read again from it",
                        path.display(),
                        first.0,
                        stamp.0,
                        c.tensor
                    )));
                }
                Some(_) => {}
                None => {
                    s.insert(path.clone(), stamp);
                }
            }
        }
        use std::io::{Read, Seek, SeekFrom};
        f.seek(SeekFrom::Start(c.offset)).map_err(|e| bad(e.to_string()))?;
        let mut buf = vec![0_u8; usize::try_from(c.len).map_err(|e| bad(e.to_string()))?];
        f.read_exact(&mut buf).map_err(|e| bad(format!("{}: {e}", c.tensor)))?;
        if again && !c.sha256.is_empty() && nmhash::sha256_bytes(&buf) != c.sha256 {
            return Err(bad(format!(
                "{}: read again from {}, the bytes do not match its content hash (a patched copy no longer held, or a changed file)",
                c.tensor,
                path.display()
            )));
        }
        let arc: Arc<[u8]> = Arc::from(buf);
        // A read again for one reader registers no weak entry (it would keep the buffer allocated
        // after the reader drops it); `keep` registers the copy it keeps.
        if again {
            self.rereads.fetch_add(1, Ordering::SeqCst);
        } else {
            t.insert(c.tensor.clone(), Arc::downgrade(&arc));
            self.reads.fetch_add(1, Ordering::SeqCst);
            if !c.sha256.is_empty() {
                hold(&c.sha256, &arc);
            }
        }
        Ok(arc)
    }
    /// The host copy of `v`'s tensor (2026-11-27, weight placement): the kept bytes, or a copy a
    /// reader already holds, or a read again from the file range (with the checks of `tensor`).
    /// The caller holds the bytes while it uses them; they are freed when the last holder drops
    /// them, unless the view keeps them (`keep`).
    /// # Errors
    /// The file cannot be read, or it changed since it was loaded.
    pub fn bytes(&self, v: &View) -> Result<Arc<[u8]>, Fault> {
        if let Some(b) = v.bytes() {
            return Ok(b);
        }
        self.tensor(&v.cluster, true)
    }
    /// Keep `v`'s host copy resident again (read again when dropped, shared with the other views
    /// of the tensor that keep it).
    /// # Errors
    /// As `bytes`.
    pub fn keep(&self, v: &View) -> Result<(), Fault> {
        let b = self.bytes(v)?;
        if let Ok(mut t) = self.tensors.lock()
            && !t.get(&v.cluster.tensor).is_some_and(|w| w.strong_count() > 0)
        {
            t.insert(v.cluster.tensor.clone(), Arc::downgrade(&b));
        }
        if let Ok(mut s) = v.slot.write() {
            s.kept = Some(b);
        }
        Ok(())
    }
    /// Drop `v`'s kept host copy. When no other view or reader holds the tensor's bytes, its weak
    /// entries (the tensor table, the content-hash table) are removed first, so the buffer is
    /// freed now. Returns the tensor bytes no longer kept by this view (0 when already dropped).
    pub fn drop_host(&self, v: &View) -> u64 {
        let Some(b) = v.slot.write().ok().and_then(|mut s| s.kept.take()) else { return 0 };
        let len = b.len() as u64;
        if Arc::strong_count(&b) == 1 {
            if let Ok(mut t) = self.tensors.lock()
                && t.get(&v.cluster.tensor).is_some_and(|w| std::ptr::addr_eq(w.as_ptr(), Arc::as_ptr(&b)))
            {
                t.remove(&v.cluster.tensor);
            }
            if !v.cluster.sha256.is_empty()
                && let Ok(mut m) = BY_HASH.lock()
                && m.get(&v.cluster.sha256).is_some_and(|w| std::ptr::addr_eq(w.as_ptr(), Arc::as_ptr(&b)))
            {
                m.remove(&v.cluster.sha256);
            }
        }
        drop(b);
        len
    }
    /// Whether the cluster's host copy is in RAM now (kept, or held by a reader).
    #[must_use]
    pub fn host_held(&self, c: &ParameterCluster) -> bool {
        self.views.lock().ok().and_then(|v| v.get(&c.id).map(|x| x.bytes().is_some())).unwrap_or(false)
    }
    /// Acquire a cluster: the same `Arc<View>` for the same cluster while it is live, one more
    /// reference counted.
    /// # Errors
    /// The file cannot be read, or the cluster is retired.
    pub fn acquire(&self, c: &ParameterCluster) -> Result<Arc<View>, Fault> {
        let mut v = self.views.lock().map_err(|_| bad("view table lock"))?;
        if let Some(view) = v.get(&c.id) {
            view.refs.fetch_add(1, Ordering::SeqCst);
            self.lifecycle.to(&c.id, State::Bound)?;
            return Ok(Arc::clone(view));
        }
        let bytes = self.tensor(c, false)?;
        let view = Arc::new(View {
            cluster: c.clone(),
            src: self.dir.join(&c.file),
            slot: RwLock::new(Slot { kept: Some(bytes) }),
            refs: AtomicUsize::new(1),
            device: AtomicUsize::new(0),
            hash: OnceLock::new(),
        });
        self.lifecycle.to(&c.id, State::Provisioned)?;
        self.lifecycle.to(&c.id, State::Bound)?;
        v.insert(c.id.clone(), Arc::clone(&view));
        Ok(view)
    }
    /// Release one reference; at zero the cluster is Released (still in the table).
    /// # Errors
    /// The cluster is not provisioned.
    pub fn release(&self, id: &str) -> Result<usize, Fault> {
        let v = self.views.lock().map_err(|_| bad("view table lock"))?;
        let view = v.get(id).ok_or_else(|| bad(format!("{id} is not provisioned")))?;
        let left = view.refs.load(Ordering::SeqCst).saturating_sub(1);
        view.refs.store(left, Ordering::SeqCst);
        if left == 0 {
            self.lifecycle.to(id, State::Released)?;
        }
        Ok(left)
    }
    /// Retire a cluster at zero references: its view leaves the table, and the tensor's bytes
    /// are freed once no other view holds them (reclamation).
    /// # Errors
    /// References remain.
    pub fn retire(&self, id: &str) -> Result<(), Fault> {
        let mut v = self.views.lock().map_err(|_| bad("view table lock"))?;
        if v.get(id).is_some_and(|x| x.refs() > 0) {
            return Err(Fault::new("nmnpcs.provision", FaultKind::Policy, format!("{id} still has references; retire refused")));
        }
        v.remove(id);
        self.lifecycle.to(id, State::Retired)?;
        Ok(())
    }
    /// Record that a cluster now holds `bytes` on the GPU (0 when released there).
    pub fn set_device(&self, id: &str, bytes: usize) {
        if let Ok(v) = self.views.lock()
            && let Some(view) = v.get(id)
        {
            view.device.store(bytes, Ordering::SeqCst);
        }
    }
    /// Every live cluster: id, storage, RAM bytes it accounts for (its own bytes while its host
    /// copy is in RAM, 0 once dropped).
    #[must_use]
    pub fn residency(&self) -> Vec<(String, Storage, u64)> {
        let v = self.views.lock().map(|v| v.values().cloned().collect::<Vec<_>>()).unwrap_or_default();
        v.iter()
            .map(|x| {
                let ram = if x.bytes().is_some() { x.cluster.bytes } else { 0 };
                let s = if x.device_bytes() > 0 {
                    Storage::Device { bytes: x.device_bytes() as u64 }
                } else if ram > 0 {
                    Storage::Resident
                } else {
                    Storage::Mapped { file: self.dir.join(&x.cluster.file), offset: x.cluster.offset, len: x.cluster.len }
                };
                (x.cluster.id.clone(), s, ram)
            })
            .collect()
    }
    /// RAM bytes held (distinct tensors in RAM now, kept or held by a reader) and GPU bytes
    /// recorded.
    #[must_use]
    pub fn totals(&self) -> (u64, u64) {
        let v = self.views.lock().map(|v| v.values().cloned().collect::<Vec<_>>()).unwrap_or_default();
        let mut seen: Vec<*const u8> = Vec::new();
        let (mut ram, mut gpu) = (0_u64, 0_u64);
        for x in &v {
            if let Some(b) = x.bytes() {
                let p = b.as_ptr();
                if !seen.contains(&p) {
                    seen.push(p);
                    ram += b.len() as u64;
                }
            }
            gpu += x.device_bytes() as u64;
        }
        (ram, gpu)
    }
    /// Every live view (for placement reports).
    #[must_use]
    pub fn views(&self) -> Vec<Arc<View>> {
        self.views.lock().map(|v| v.values().cloned().collect()).unwrap_or_default()
    }
    /// Views whose host copy is dropped now (not kept; it may still be held by a reader).
    #[must_use]
    pub fn dropped(&self) -> usize {
        self.views.lock().map_or(0, |v| v.values().filter(|x| !x.resident()).count())
    }
    /// Live views (for the binding table and reports).
    #[must_use]
    pub fn live(&self) -> usize {
        self.views.lock().map_or(0, |v| v.len())
    }
    #[must_use]
    pub fn storage_of(&self, c: &ParameterCluster) -> Storage {
        match self.views.lock().ok().and_then(|v| v.get(&c.id).map(|x| (x.device_bytes(), x.bytes().is_some()))) {
            Some((0, true)) => Storage::Resident,
            Some((b, _)) if b > 0 => Storage::Device { bytes: b as u64 },
            _ => Storage::Mapped { file: self.dir.join(&c.file), offset: c.offset, len: c.len },
        }
    }
    /// Residency summary as JSON.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let (ram, gpu) = self.totals();
        let mut m = BTreeMap::new();
        m.insert("model".into(), Value::String(self.model.clone()));
        m.insert("live_clusters".into(), Value::Number(self.live().to_string()));
        m.insert("tensor_reads".into(), Value::Number(self.reads().to_string()));
        m.insert("ram_bytes".into(), Value::Number(ram.to_string()));
        m.insert("gpu_bytes".into(), Value::Number(gpu.to_string()));
        Value::Object(m)
    }
}

/// Self-check of `provision`: on a small file in the scratch folder, two acquires of one
/// cluster share one `Arc` and one tensor read; two clusters of one tensor share its bytes;
/// retire is refused with references and allowed at zero; after retiring both, the tensor is
/// read again on the next acquire.
pub(crate) fn check() -> Result<(), String> {
    let dir = nmtype::tests_dir("section-checks/nmnpcs");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let file = dir.join("toy.bin");
    std::fs::write(&file, (0_u8..32).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
    let mk = |id: &str, region: Region| ParameterCluster {
        id: id.into(),
        model: "toy".into(),
        tensor: "t".into(),
        region,
        group: "g".into(),
        dtype: "BF16".into(),
        shape: vec![4, 4],
        file: "toy.bin".into(),
        offset: 0,
        len: 32,
        bytes: 16,
        sha256: String::new(),
    };
    let (a, b) = (mk("toy/t/rows0-2", Region::Rows(0..2)), mk("toy/t/rows2-4", Region::Rows(2..4)));
    let p = Provisioner::new("toy", &dir, &[a.clone(), b.clone()]);
    let (x1, x2, y) =
        (p.acquire(&a).map_err(|e| e.to_string())?, p.acquire(&a).map_err(|e| e.to_string())?, p.acquire(&b).map_err(|e| e.to_string())?);
    let yb = y.tensor_bytes();
    if !Arc::ptr_eq(&x1, &x2)
        || x1.refs() != 2
        || !Arc::ptr_eq(&x1.tensor_bytes(), &yb)
        || p.reads() != 1
        || y.row_in(&yb, 0, 2) != [16, 17, 18, 19, 20, 21, 22, 23]
    {
        return Err("nmnpcs.provision: sharing".into());
    }
    drop(yb);
    // 2026-11-27, weight placement: dropping both views' host copies frees the tensor (no weak
    // entry keeps its buffer: the table entry goes with the last copy); a read on demand gives the
    // same bytes from the file to its reader alone; views kept again share one copy.
    let probe = Arc::downgrade(&x1.tensor_bytes());
    if p.drop_host(&x1) != 32 || p.drop_host(&y) != 32 || x1.bytes().is_some() || p.totals().0 != 0 || p.dropped() != 2 {
        return Err("nmnpcs.provision: host copy not dropped".into());
    }
    if probe.strong_count() != 0 || p.tensors.lock().map_or(true, |t| t.contains_key("t")) {
        return Err("nmnpcs.provision: a weak entry outlives the dropped copy".into());
    }
    let again = p.bytes(&y).map_err(|e| e.to_string())?;
    if y.row_in(&again, 0, 2) != [16, 17, 18, 19, 20, 21, 22, 23] || x1.bytes().is_some() || p.rereads() != 1 || p.totals().0 != 0 {
        return Err("nmnpcs.provision: read on demand".into());
    }
    if Arc::weak_count(&again) != 0 {
        return Err("nmnpcs.provision: a read on demand left a weak entry".into());
    }
    drop(again);
    p.keep(&x1).map_err(|e| e.to_string())?;
    p.keep(&y).map_err(|e| e.to_string())?;
    let shared = x1.bytes().zip(y.bytes()).is_some_and(|(a, b)| Arc::ptr_eq(&a, &b));
    if !x1.resident() || !shared || p.rereads() != 2 || p.dropped() != 0 || p.totals().0 != 32 {
        return Err("nmnpcs.provision: keep".into());
    }
    if p.retire(&a.id).is_ok() {
        return Err("nmnpcs.provision: retired with references".into());
    }
    drop((x1, x2, y));
    for id in [&a.id, &a.id, &b.id] {
        p.release(id).map_err(|e| e.to_string())?;
    }
    p.retire(&a.id).map_err(|e| e.to_string())?;
    p.retire(&b.id).map_err(|e| e.to_string())?;
    let _again = p.acquire(&a).map_err(|e| e.to_string())?;
    if p.reads() != 2 {
        return Err("nmnpcs.provision: bytes not reclaimed after retire".into());
    }
    Ok(())
}
