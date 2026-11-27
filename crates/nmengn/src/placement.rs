//! Placement (PLAN 4c `backend`, 4d item 10; guide 9.3 responsibility 10, 13.1): every job
//! runs where it is best at that moment, decided per job, never pinned by a setting. The CPU
//! is always a candidate (RAM holds every weight copy). The GPU is a candidate when a device
//! is open, no timing run holds it exclusively, and its free memory covers the job's device
//! bytes. Between candidates, the measured milliseconds for the job's shape class decide
//! (learned once per shape, kept in `designs/placement.json`); an unmeasured shape goes to the
//! GPU and is measured there.
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Where a job runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}
impl Device {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

/// The shape class of a job: operation and a token bucket.
#[must_use]
pub fn shape(op: &str, tokens: usize) -> String {
    let b = match tokens {
        0..=1 => "1",
        2..=16 => "16",
        17..=64 => "64",
        65..=256 => "256",
        _ => "long",
    };
    format!("{op}/t{b}")
}

/// What the placement saw and chose, for reports.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub device: Device,
    pub why: String,
}

/// Measured milliseconds for one shape class: (CPU, GPU).
type Measured = (Option<f64>, Option<f64>);

/// Roadmap Phase 4.6 (2026-11-25): the machine the timings in a table without machine keys were
/// measured on (`nano-lab run s6 place` on the development machine's GTX 1650), in the form
/// `capacity::machine_key` gives (card name and memory in MB).
pub const LEGACY_MACHINE: &str = "NVIDIA GeForce GTX 1650|4095";

/// The placement table and the GPU hold flag.
#[derive(Debug, Default)]
pub struct Placement {
    table: Mutex<BTreeMap<String, Measured>>,
    held: AtomicBool,
    /// Roadmap Phase 4.6 (2026-11-25): the machine this table is for (`None` without a card).
    machine: Option<String>,
    /// Roadmap Phase 4.6 (2026-11-25): the other machines' tables as read, written back as they
    /// were.
    others: BTreeMap<String, Value>,
}

/// The shapes of one machine's table (`{"op/tN": {"cpu_ms": x, "gpu_ms": y}}`).
fn shapes_of(m: &BTreeMap<String, Value>) -> BTreeMap<String, Measured> {
    m.iter()
        .map(|(k, e)| {
            let f = |n: &str| match e.get(n) {
                Ok(Value::Number(x)) => x.parse().ok(),
                _ => None,
            };
            (k.clone(), (f("cpu_ms"), f("gpu_ms")))
        })
        .collect()
}

impl Placement {
    /// Read `designs/placement.json` for the machine the timings without machine keys were
    /// measured on (`LEGACY_MACHINE`).
    #[must_use]
    pub fn load(path: &Path) -> Self {
        Self::load_for(path, Some(LEGACY_MACHINE))
    }
    /// Roadmap Phase 4.6 (2026-11-25): read `designs/placement.json` for one machine
    /// (`capacity::machine_key`; `None` without a card). The table is keyed by machine
    /// (`{"machines": {"<card>|<MB>": {"shapes": {...}}}}`), so a different card starts from
    /// neutral (every shape unmeasured) and measures its own timings; a table in the older form
    /// (top-level `shapes`) belongs to `LEGACY_MACHINE`. Other machines' tables are kept for
    /// `to_value`.
    #[must_use]
    pub fn load_for(path: &Path, machine: Option<&str>) -> Self {
        let mut p = Self { machine: machine.map(str::to_owned), ..Self::default() };
        let Ok(text) = std::fs::read_to_string(path) else { return p };
        let Ok(v) = nmjson::parse(text.as_bytes()) else { return p };
        let mut machines: BTreeMap<String, Value> = match v.get("machines") {
            Ok(Value::Object(m)) => m.clone(),
            _ => BTreeMap::new(),
        };
        if let Ok(Value::Object(legacy)) = v.get("shapes") {
            machines
                .entry(LEGACY_MACHINE.to_owned())
                .or_insert_with(|| Value::Object([("shapes".to_owned(), Value::Object(legacy.clone()))].into_iter().collect()));
        }
        if let Some(key) = machine
            && let Some(mine) = machines.remove(key)
            && let (Ok(Value::Object(m)), Ok(mut t)) = (mine.get("shapes"), p.table.lock())
        {
            *t = shapes_of(m);
        }
        p.others = machines;
        p
    }
    /// The machine this table is for.
    #[must_use]
    pub fn machine(&self) -> Option<&str> {
        self.machine.as_deref()
    }
    /// A timing run holds the GPU exclusively (true) or releases it (false).
    pub fn hold_gpu(&self, held: bool) {
        self.held.store(held, Ordering::SeqCst);
    }
    #[must_use]
    pub fn gpu_held(&self) -> bool {
        self.held.load(Ordering::SeqCst)
    }
    /// Record a measured run.
    pub fn record(&self, shape: &str, device: Device, ms: f64) {
        if let Ok(mut t) = self.table.lock() {
            let e = t.entry(shape.to_owned()).or_insert((None, None));
            match device {
                Device::Cpu => e.0 = Some(e.0.map_or(ms, |o| o.min(ms))),
                Device::Gpu => e.1 = Some(e.1.map_or(ms, |o| o.min(ms))),
            }
        }
    }
    /// Choose the device for one job. `gpu_free` is `None` when no device is open.
    #[must_use]
    pub fn choose(&self, shape: &str, need_bytes: u64, gpu_free: Option<u64>) -> Choice {
        let Some(free) = gpu_free else { return Choice { device: Device::Cpu, why: "no GPU open".into() } };
        if self.gpu_held() {
            return Choice { device: Device::Cpu, why: "a timing run holds the GPU".into() };
        }
        if need_bytes > free {
            return Choice { device: Device::Cpu, why: format!("GPU free {free} bytes < {need_bytes} needed") };
        }
        match self.table.lock().ok().and_then(|t| t.get(shape).copied()) {
            Some((Some(c), Some(g))) if c < g => {
                Choice { device: Device::Cpu, why: format!("{shape}: CPU {c:.1} ms < GPU {g:.1} ms measured") }
            }
            Some((c, Some(g))) => Choice {
                device: Device::Gpu,
                why: format!("{shape}: GPU {g:.1} ms measured{}", c.map(|c| format!(" < CPU {c:.1} ms")).unwrap_or_default()),
            },
            _ => Choice { device: Device::Gpu, why: format!("{shape}: unmeasured, GPU free and fits") },
        }
    }
    /// The table as written to `designs/placement.json`: this machine's shapes under its key,
    /// every other machine's table as it was read (Roadmap Phase 4.6 (2026-11-25)).
    #[must_use]
    pub fn to_value(&self, note: &str) -> Value {
        let t = self.table.lock().map(|t| t.clone()).unwrap_or_default();
        let shapes: BTreeMap<String, Value> = t
            .into_iter()
            .map(|(k, (c, g))| {
                let mut m = BTreeMap::new();
                if let Some(c) = c {
                    m.insert("cpu_ms".to_owned(), Value::Number(format!("{c:.2}")));
                }
                if let Some(g) = g {
                    m.insert("gpu_ms".to_owned(), Value::Number(format!("{g:.2}")));
                }
                (k, Value::Object(m))
            })
            .collect();
        let mut machines = self.others.clone();
        if let Some(key) = &self.machine
            && !shapes.is_empty()
        {
            machines.insert(key.clone(), Value::Object([("shapes".to_owned(), Value::Object(shapes))].into_iter().collect()));
        }
        Value::Object([("note".to_owned(), Value::String(note.into())), ("machines".to_owned(), Value::Object(machines))].into_iter().collect())
    }
}

/// The device gate (step P2, 2026-11-10): the placement rule's exclusive hold extended to the
/// whole device and to waiting. Two kinds of GPU job must not share the card: the tentacle
/// pass (encoding and reading; a foreground job) and the dense lane's background indexing. A
/// job enters before it starts and leaves when its guard drops; a background job also waits
/// while a foreground job is waiting, so a long background loop cannot starve the pass. A job
/// that has to wait says so (`status`, and the waits are counted for the pass's note).
#[derive(Debug, Default)]
pub struct GpuGate {
    st: Mutex<GateState>,
    cv: std::sync::Condvar,
}
#[derive(Debug, Default)]
struct GateState {
    holder: Option<String>,
    fg_waiting: usize,
    /// Questions being answered now (`GpuGate::question`): background jobs wait meanwhile.
    questions: usize,
    waiting: Vec<String>,
    bg_waits: u64,
    bg_wait_secs: f64,
    /// Set at shutdown (`GpuGate::close`): no job enters any more.
    closing: bool,
}
/// Held while a job has the device; dropping it lets the next job in.
#[derive(Debug)]
pub struct GateGuard {
    gate: &'static GpuGate,
    /// Seconds this job waited, and for which job (empty when it did not wait).
    pub waited: f64,
    pub waited_for: String,
}
impl Drop for GateGuard {
    fn drop(&mut self) {
        let mut st = self.gate.st.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        st.holder = None;
        drop(st);
        self.gate.cv.notify_all();
    }
}
/// Held while a question is answered (`GpuGate::question`).
#[derive(Debug)]
pub struct QuestionGuard {
    gate: &'static GpuGate,
}
impl Drop for QuestionGuard {
    fn drop(&mut self) {
        let mut st = self.gate.st.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        st.questions = st.questions.saturating_sub(1);
        drop(st);
        self.gate.cv.notify_all();
    }
}
static GATE: std::sync::OnceLock<GpuGate> = std::sync::OnceLock::new();
/// The device gate of this process.
pub fn gpu_gate() -> &'static GpuGate {
    GATE.get_or_init(GpuGate::default)
}
impl GpuGate {
    /// Enter the gate as `job`; `foreground` jobs go before waiting background jobs.
    pub fn enter(&'static self, job: &str, foreground: bool) -> GateGuard {
        let started = std::time::Instant::now();
        let mut st = self.st.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut waited_for = String::new();
        let blocked = |s: &GateState| s.closing || s.holder.is_some() || (!foreground && (s.fg_waiting > 0 || s.questions > 0));
        if blocked(&st) {
            waited_for = st.holder.clone().unwrap_or_else(|| {
                if st.questions > 0 { "a question being answered".to_owned() } else { "a waiting foreground job".to_owned() }
            });
            let line = format!("{job} waiting for {waited_for}");
            st.waiting.push(line.clone());
            if foreground {
                st.fg_waiting += 1;
            }
            while blocked(&st) {
                st = self.cv.wait(st).unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            if foreground {
                st.fg_waiting -= 1;
            }
            if let Some(i) = st.waiting.iter().position(|w| *w == line) {
                st.waiting.remove(i);
            }
        }
        let waited = if waited_for.is_empty() { 0.0 } else { started.elapsed().as_secs_f64() };
        if !foreground && waited > 0.0 {
            st.bg_waits += 1;
            st.bg_wait_secs += waited;
        }
        st.holder = Some(job.to_owned());
        GateGuard { gate: self, waited, waited_for }
    }
    /// Mark a question as being answered until the guard drops: background jobs (the dense
    /// lane's index) do not enter the gate meanwhile, so the question has the card to itself
    /// once the background job in flight ends. Questions do not exclude each other.
    pub fn question(&'static self) -> QuestionGuard {
        let mut st = self.st.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        st.questions += 1;
        QuestionGuard { gate: self }
    }
    /// Shutdown (lane T1, 2026-11-19, T-086): no job enters from now on (a job that asks waits
    /// until the process ends, outside any card call), and the job holding the device is waited
    /// for, at most `wait`. Returns the job still holding it when the wait ran out.
    pub fn close(&'static self, wait: std::time::Duration) -> Option<String> {
        let started = std::time::Instant::now();
        let mut st = self.st.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        st.closing = true;
        while st.holder.is_some() {
            let left = wait.saturating_sub(started.elapsed());
            if left.is_zero() {
                return st.holder.clone();
            }
            st = self.cv.wait_timeout(st, left).unwrap_or_else(std::sync::PoisonError::into_inner).0;
        }
        None
    }
    /// Who holds the device and who waits, in words.
    #[must_use]
    pub fn status(&self) -> String {
        let st = self.st.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        format!(
            "device gate: held by {}; waiting: {}",
            st.holder.as_deref().unwrap_or("nobody"),
            if st.waiting.is_empty() { "none".to_owned() } else { st.waiting.join("; ") }
        )
    }
    /// Background waits since the last call: (count, seconds); the counters restart.
    pub fn take_background_waits(&self) -> (u64, f64) {
        let mut st = self.st.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        (std::mem::take(&mut st.bg_waits), std::mem::take(&mut st.bg_wait_secs))
    }
}

/// Self-check of the device gate (step P2): a background job waits while a foreground job
/// holds the device and says so while it waits; a background job does not enter while a
/// foreground job waits; the foreground job learns what it waited for.
pub(crate) fn check_gate() -> Result<(), String> {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    let gate: &'static GpuGate = Box::leak(Box::new(GpuGate::default()));
    let order = Arc::new(Mutex::new(Vec::<&str>::new()));
    let fg = gate.enter("encode", true);
    let o2 = Arc::clone(&order);
    let bg = std::thread::spawn(move || {
        let _g = gate.enter("indexing", false);
        if let Ok(mut o) = o2.lock() {
            o.push("indexing");
        }
    });
    let seen = Arc::new(AtomicUsize::new(0));
    for _ in 0..200 {
        if gate.status().contains("indexing waiting for encode") {
            seen.store(1, Ordering::SeqCst);
            break;
        }
        std::thread::yield_now();
        std::thread::park_timeout(std::time::Duration::from_millis(5));
    }
    if let Ok(mut o) = order.lock() {
        o.push("encode done");
    }
    drop(fg);
    bg.join().map_err(|_| "nmengn.placement.gate: background thread failed")?;
    let got = order.lock().map(|o| o.clone()).unwrap_or_default();
    let (waits, _) = gate.take_background_waits();
    if got != ["encode done", "indexing"] || seen.load(Ordering::SeqCst) != 1 || waits != 1 {
        return Err(format!(
            "nmengn.placement.gate: order {got:?}, waiting stated {}, background waits {waits}",
            seen.load(Ordering::SeqCst) == 1
        ));
    }
    // A background job holds; a foreground job waits; a second background job must not slip in
    // ahead of the waiting foreground job.
    let bg1 = gate.enter("indexing", false);
    let order = Arc::new(Mutex::new(Vec::<&str>::new()));
    let (o3, o4) = (Arc::clone(&order), Arc::clone(&order));
    let f = std::thread::spawn(move || {
        let g = gate.enter("encode", true);
        if let Ok(mut o) = o3.lock() {
            o.push("encode");
        }
        (g.waited > 0.0, g.waited_for.clone())
    });
    for _ in 0..200 {
        if gate.status().contains("encode waiting for indexing") {
            break;
        }
        std::thread::park_timeout(std::time::Duration::from_millis(5));
    }
    let b2 = std::thread::spawn(move || {
        let _g = gate.enter("indexing again", false);
        if let Ok(mut o) = o4.lock() {
            o.push("indexing again");
        }
    });
    for _ in 0..200 {
        if gate.status().contains("indexing again waiting") {
            break;
        }
        std::thread::park_timeout(std::time::Duration::from_millis(5));
    }
    drop(bg1);
    let (waited, for_whom) = f.join().map_err(|_| "nmengn.placement.gate: foreground thread failed")?;
    b2.join().map_err(|_| "nmengn.placement.gate: background thread failed")?;
    let got = order.lock().map(|o| o.clone()).unwrap_or_default();
    if got != ["encode", "indexing again"] || !waited || for_whom != "indexing" {
        return Err(format!("nmengn.placement.gate: foreground priority {got:?}, waited {waited} for {for_whom}"));
    }
    // While a question is answered a background job waits, and enters once it ends.
    let q = gate.question();
    let entered = Arc::new(AtomicUsize::new(0));
    let e2 = Arc::clone(&entered);
    let b3 = std::thread::spawn(move || {
        let _g = gate.enter("indexing during a question", false);
        e2.store(1, Ordering::SeqCst);
    });
    for _ in 0..200 {
        if gate.status().contains("indexing during a question waiting for a question being answered") {
            break;
        }
        std::thread::park_timeout(std::time::Duration::from_millis(5));
    }
    let early = entered.load(Ordering::SeqCst);
    drop(q);
    b3.join().map_err(|_| "nmengn.placement.gate: background thread failed")?;
    if early != 0 || entered.load(Ordering::SeqCst) != 1 {
        return Err("nmengn.placement.gate: background job entered during a question".into());
    }
    Ok(())
}

/// Self-check of `placement`: no device means CPU; a held GPU means CPU; a job too big for the
/// free memory means CPU; a measured faster CPU wins; otherwise the GPU.
pub(crate) fn check() -> Result<(), String> {
    let p = Placement::default();
    let s = shape("embed", 12);
    let ok = p.choose(&s, 10, None).device == Device::Cpu
        && p.choose(&s, 10, Some(100)).device == Device::Gpu
        && p.choose(&s, 1000, Some(100)).device == Device::Cpu;
    p.hold_gpu(true);
    let held = p.choose(&s, 10, Some(100)).device == Device::Cpu;
    p.hold_gpu(false);
    p.record(&s, Device::Cpu, 5.0);
    p.record(&s, Device::Gpu, 9.0);
    if !ok || !held || p.choose(&s, 10, Some(100)).device != Device::Cpu {
        return Err("nmengn.placement: choice rules".into());
    }
    keyed_check()
}

/// Roadmap Phase 4.6 (2026-11-25): the table is keyed by machine. The older form belongs to the
/// GTX 1650 and reads the same there; another card starts neutral (unmeasured shapes go to the
/// card and are measured); writing keeps every machine's table.
fn keyed_check() -> Result<(), String> {
    let dir = nmtype::check_dir("nmengn-placement-keyed").map_err(|e| e.to_string())?;
    let file = dir.join("placement.json");
    let legacy = r#"{"note": "n", "shapes": {"embed/t16": {"cpu_ms": 457.36, "gpu_ms": 99.83}, "rerank/t64": {"cpu_ms": 50.0, "gpu_ms": 116.60}}}"#;
    std::fs::write(&file, legacy).map_err(|e| e.to_string())?;
    let other_card = "NVIDIA GeForce RTX 4090|24564";
    let r = (|| -> Result<(), String> {
        let own = Placement::load_for(&file, Some(LEGACY_MACHINE));
        let other = Placement::load_for(&file, Some(other_card));
        let rr = shape("rerank", 40);
        let mine = own.choose(&rr, 10, Some(100));
        if mine.device != Device::Cpu || !mine.why.contains("measured") {
            return Err("nmengn.placement: the GTX 1650's measured timings were not read under its key".into());
        }
        if Placement::load(&file).choose(&rr, 10, Some(100)).device != Device::Cpu {
            return Err("nmengn.placement: load without a machine is not the GTX 1650's table".into());
        }
        let c = other.choose(&rr, 10, Some(100));
        if c.device != Device::Gpu || !c.why.contains("unmeasured") {
            return Err(format!("nmengn.placement: another card reused the GTX 1650's timings ({})", c.why));
        }
        other.record(&rr, Device::Gpu, 3.0);
        std::fs::write(&file, nmjson::pretty::pretty(&other.to_value("n"))).map_err(|e| e.to_string())?;
        let again_own = Placement::load_for(&file, Some(LEGACY_MACHINE));
        let again_other = Placement::load_for(&file, Some(other_card));
        if again_own.choose(&rr, 10, Some(100)).device != Device::Cpu || !again_other.choose(&rr, 10, Some(100)).why.contains("3.0 ms") {
            return Err("nmengn.placement: writing one machine's table lost another's".into());
        }
        let none = Placement::load_for(&file, None);
        if none.choose(&rr, 10, None).device != Device::Cpu || none.machine().is_some() {
            return Err("nmengn.placement: no card".into());
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    r
}

#[cfg(test)]
mod tests {
    #[test]
    fn placement_keyed_by_machine() {
        super::check().unwrap();
    }
}
