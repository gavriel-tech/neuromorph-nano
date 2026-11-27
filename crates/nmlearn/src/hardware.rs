//! Hardware identity for the cartridge library: the pieces of a machine (graphics card,
//! processor, RAM, the disk holding the Nano folder), each with an exact identity and a family,
//! read with the operating system's own commands (`nvidia-smi`, the registry or `/proc`, and one
//! PowerShell query on Windows for RAM, disk and a graphics card that is not NVIDIA). Nothing
//! opens the graphics card. A machine description is cached in `library/machine.json` for a day;
//! the fast probe (graphics card and processor only) runs at every server start, so a changed card
//! is seen at once.
//!
//! Identity rules, used by stacks to say when they apply:
//! - graphics card: vendor, model, memory in MiB and driver range (the driver's major version);
//!   family is the vendor and architecture (from the compute capability on NVIDIA).
//! - processor: vendor and model; family is the vendor and generation.
//! - RAM: the size in whole GiB; every RAM is one family.
//! - disk: media, bus and model; family is the media (ssd, hdd).
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// How long a cached machine description is used before it is read again.
pub const MACHINE_MS: u64 = 24 * 60 * 60 * 1000;
/// The cache file inside the library folder.
pub const MACHINE_FILE: &str = "machine.json";

/// The kinds of piece, from the most general to the most specific for serving: a later piece's
/// stack is placed over an earlier one's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Ram,
    Disk,
    Cpu,
    Gpu,
}
impl Kind {
    pub const ALL: [Self; 4] = [Self::Ram, Self::Disk, Self::Cpu, Self::Gpu];
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ram => "ram",
            Self::Disk => "disk",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
    /// # Errors
    /// A name that is not a piece kind.
    pub fn parse(t: &str) -> Result<Self, String> {
        Self::ALL.into_iter().find(|k| k.name() == t).ok_or_else(|| format!("{t} is not a piece kind (ram, disk, cpu, gpu)"))
    }
}

/// One piece of hardware.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub kind: Kind,
    /// `nvidia`, `amd`, `intel`, or empty when unknown.
    pub vendor: String,
    /// Model as a lowercase slug (`geforce-gtx-1650`, `core-i5-12450h`, `samsung-pssd-t7`).
    pub model: String,
    /// Family slug (`turing`, `core-gen12`, `ram`, `ssd`).
    pub family: String,
    /// Graphics memory or RAM in MiB; 0 when not a measure of this piece.
    pub memory_mib: u64,
    /// Graphics driver range (major version), or empty.
    pub driver: String,
    /// Bus of a disk (`nvme`, `usb`, `sata`), or empty.
    pub bus: String,
    /// The name the system gives the piece.
    pub name: String,
}

/// Lowercase slug: letters and digits kept, runs of anything else become one `-`.
#[must_use]
pub fn slug(t: &str) -> String {
    let mut out = String::new();
    for c in t.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_owned()
}

/// RAM size in whole GiB, rounded to the nearest (a 32 GiB machine reports a little less).
#[must_use]
pub fn gib(mib: u64) -> u64 {
    (mib + 512) / 1024
}

impl Piece {
    /// The exact identity a stack is keyed by.
    #[must_use]
    pub fn identity(&self) -> String {
        match self.kind {
            Kind::Gpu => format!("gpu:{}:{}:{}mib:driver-{}", self.vendor, self.model, self.memory_mib, if self.driver.is_empty() { "unknown" } else { &self.driver }),
            Kind::Cpu => format!("cpu:{}:{}", self.vendor, self.model),
            Kind::Ram => format!("ram:{}gib", gib(self.memory_mib)),
            Kind::Disk => format!("disk:{}:{}:{}", self.family, self.bus, self.model),
        }
    }
    /// The family a stack with the family rule applies to.
    #[must_use]
    pub fn family_key(&self) -> String {
        match self.kind {
            Kind::Ram => "ram".to_owned(),
            Kind::Disk => format!("disk:{}", self.family),
            k => format!("{}:{}:{}", k.name(), self.vendor, self.family),
        }
    }
    /// How far a relative of the same family is: memory ratio in powers of two, plus a small step
    /// for another model and for another driver range, so the nearest memory wins first.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "sizes in MiB, well inside f64")]
    pub fn distance(&self, other: &Self) -> f64 {
        let mem = if self.memory_mib > 0 && other.memory_mib > 0 { (self.memory_mib as f64 / other.memory_mib as f64).log2().abs() } else { 0.0 };
        let model = if self.model == other.model { 0.0 } else { 0.01 };
        let driver = if self.driver == other.driver { 0.0 } else { 0.001 };
        mem + model + driver
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let s = |v: &str| Value::String(v.to_owned());
        Value::Object(BTreeMap::from([
            ("kind".to_owned(), s(self.kind.name())),
            ("vendor".to_owned(), s(&self.vendor)),
            ("model".to_owned(), s(&self.model)),
            ("family".to_owned(), s(&self.family)),
            ("memory_mib".to_owned(), Value::Number(self.memory_mib.to_string())),
            ("driver".to_owned(), s(&self.driver)),
            ("bus".to_owned(), s(&self.bus)),
            ("name".to_owned(), s(&self.name)),
            ("identity".to_owned(), s(&self.identity())),
            ("family_key".to_owned(), s(&self.family_key())),
        ]))
    }
    /// # Errors
    /// A missing kind or an unknown one.
    pub fn from_value(v: &Value) -> Result<Self, String> {
        let t = |k: &str| v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default();
        Ok(Self {
            kind: Kind::parse(v.get("kind").and_then(Value::string).map_err(|e| format!("piece kind: {e}"))?)?,
            vendor: t("vendor"),
            model: t("model"),
            family: t("family"),
            memory_mib: v.get("memory_mib").and_then(Value::integer).unwrap_or(0),
            driver: t("driver"),
            bus: t("bus"),
            name: t("name"),
        })
    }
}

/// A machine: its pieces in kind order, when and how they were read.
#[derive(Clone, Debug, PartialEq)]
pub struct Machine {
    pub pieces: Vec<Piece>,
    pub probed_ms: u64,
    /// `full`, `fast` (graphics card and processor read now, the rest from the cache) or
    /// `given` (a description handed in, as tests do).
    pub how: String,
}
impl Machine {
    #[must_use]
    pub fn piece(&self, kind: Kind) -> Option<&Piece> {
        self.pieces.iter().find(|p| p.kind == kind)
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        Value::Object(BTreeMap::from([
            ("pieces".to_owned(), Value::Array(self.pieces.iter().map(Piece::to_value).collect())),
            ("probed_ms".to_owned(), Value::Number(self.probed_ms.to_string())),
            ("how".to_owned(), Value::String(self.how.clone())),
            ("note".to_owned(), Value::String("This machine's pieces as the cartridge library keys its stacks; read again after a day. Written by the Neuromorph Nano.".into())),
        ]))
    }
    /// # Errors
    /// A piece that does not parse.
    pub fn from_value(v: &Value) -> Result<Self, String> {
        let mut pieces = v.get("pieces").and_then(Value::array).map_err(|e| format!("pieces: {e}"))?.iter().map(Piece::from_value).collect::<Result<Vec<_>, _>>()?;
        pieces.sort_by_key(|p| p.kind);
        Ok(Self {
            pieces,
            probed_ms: v.get("probed_ms").and_then(Value::integer).unwrap_or(0),
            how: v.get("how").and_then(Value::string).unwrap_or("given").to_owned(),
        })
    }
}

/// Architecture of an NVIDIA card from its compute capability.
#[must_use]
pub fn nvidia_family(cc: &str) -> String {
    let fam = match cc.trim() {
        "5.0" | "5.2" | "5.3" => "maxwell",
        "6.0" | "6.1" | "6.2" => "pascal",
        "7.0" | "7.2" => "volta",
        "7.5" => "turing",
        "8.0" | "8.6" | "8.7" => "ampere",
        "8.9" => "ada",
        "9.0" => "hopper",
        c if c.starts_with("10.") || c.starts_with("12.") => "blackwell",
        "" => "unknown",
        c => return format!("cc-{}", slug(c)),
    };
    fam.to_owned()
}

/// A graphics card from the name the system gives it, its memory, driver version and (NVIDIA)
/// compute capability.
#[must_use]
pub fn gpu_piece(name: &str, memory_mib: u64, driver_version: &str, compute_cap: &str) -> Piece {
    let low = name.to_lowercase();
    let vendor = if ["nvidia", "geforce", "quadro", "tesla", "rtx"].iter().any(|w| low.contains(w)) {
        "nvidia"
    } else if low.contains("amd") || low.contains("radeon") {
        "amd"
    } else if low.contains("intel") {
        "intel"
    } else {
        ""
    };
    let model = slug(&low.replace("nvidia", "").replace("(r)", "").replace("(tm)", "").replace("amd", "").replace("intel", ""));
    let family = match vendor {
        "nvidia" => nvidia_family(compute_cap),
        "amd" => "radeon".to_owned(),
        "intel" => "intel-graphics".to_owned(),
        _ => "unknown".to_owned(),
    };
    let driver = driver_version.trim().split('.').next().unwrap_or("").to_owned();
    Piece { kind: Kind::Gpu, vendor: vendor.into(), model, family, memory_mib, driver, bus: String::new(), name: name.trim().to_owned() }
}

/// A processor from the name the system gives it.
#[must_use]
pub fn cpu_piece(name: &str) -> Piece {
    let low = name.to_lowercase();
    let vendor = if low.contains("intel") {
        "intel"
    } else if low.contains("amd") || low.contains("ryzen") || low.contains("epyc") {
        "amd"
    } else if low.contains("apple") {
        "apple"
    } else {
        ""
    };
    let mut clean = low.split('@').next().unwrap_or("").to_owned();
    for w in ["(r)", "(tm)", "intel", "amd", "processor", "cpu", "with radeon graphics"] {
        clean = clean.replace(w, " ");
    }
    // `12th gen` names the generation, not the model.
    let words: Vec<&str> = clean.split_whitespace().collect();
    let gen_word = words.windows(2).find(|w| w.get(1) == Some(&"gen")).and_then(|w| w.first().copied());
    let model_words: Vec<&str> = words.iter().copied().filter(|w| Some(*w) != gen_word && *w != "gen" && !w.ends_with("-core")).collect();
    let model = slug(&model_words.join(" "));
    let digits = |w: &str| w.chars().filter(char::is_ascii_digit).collect::<String>();
    let family = match vendor {
        "intel" => {
            let g = gen_word.map(digits).filter(|d| !d.is_empty()).or_else(|| {
                let num = model_words.iter().map(|w| digits(w)).find(|d| d.len() >= 4)?;
                Some(if num.len() >= 5 { num.get(..2)?.to_owned() } else { num.get(..1)?.to_owned() })
            });
            g.map_or_else(|| "core".to_owned(), |g| format!("core-gen{g}"))
        }
        "amd" => {
            let num = model_words.iter().map(|w| digits(w)).find(|d| d.len() == 4);
            num.and_then(|d| d.get(..1).map(|x| format!("ryzen-{x}000"))).unwrap_or_else(|| "amd".to_owned())
        }
        v => if v.is_empty() { "unknown".to_owned() } else { v.to_owned() },
    };
    Piece { kind: Kind::Cpu, vendor: vendor.into(), model, family, memory_mib: 0, driver: String::new(), bus: String::new(), name: name.trim().to_owned() }
}

/// RAM of a size in MiB.
#[must_use]
pub fn ram_piece(mib: u64) -> Piece {
    Piece {
        kind: Kind::Ram,
        vendor: String::new(),
        model: format!("{}gib", gib(mib)),
        family: "ram".into(),
        memory_mib: mib,
        driver: String::new(),
        bus: String::new(),
        name: format!("{} GiB RAM", gib(mib)),
    }
}

/// A disk from its media type, bus and model name.
#[must_use]
pub fn disk_piece(media: &str, bus: &str, name: &str) -> Piece {
    let m = media.to_lowercase();
    let family = if m.contains("ssd") || m.contains("solid") {
        "ssd"
    } else if m.contains("hdd") || m.contains("rotat") {
        "hdd"
    } else {
        "unknown"
    };
    let bus_l = slug(bus);
    let family = if family == "unknown" && bus_l == "nvme" { "ssd" } else { family };
    Piece { kind: Kind::Disk, vendor: String::new(), model: slug(name), family: family.into(), memory_mib: 0, driver: String::new(), bus: bus_l, name: name.trim().to_owned() }
}

fn run_quiet(cmd: &str, args: &[&str]) -> Option<String> {
    let mut c = std::process::Command::new(cmd);
    c.args(args).stdin(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        // No console window for a program started without one.
        c.creation_flags(0x0800_0000);
    }
    let o = c.output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_owned())
}

/// The NVIDIA card named by its driver's `nvidia-smi`, if there is one.
fn probe_nvidia() -> Option<Piece> {
    let line = run_quiet("nvidia-smi", &["--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"])?;
    let first = line.lines().next()?;
    let f: Vec<&str> = first.split(',').map(str::trim).collect();
    let name = f.first()?;
    Some(gpu_piece(name, f.get(1).and_then(|m| m.parse().ok()).unwrap_or(0), f.get(2).unwrap_or(&""), f.get(3).unwrap_or(&"")))
}

/// The processor's name from the registry (Windows) or `/proc/cpuinfo`.
fn probe_cpu_name() -> Option<String> {
    if cfg!(windows) {
        let out = run_quiet("reg", &["query", r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0", "/v", "ProcessorNameString"])?;
        let line = out.lines().find(|l| l.contains("ProcessorNameString"))?;
        return line.split("REG_SZ").nth(1).map(|x| x.trim().to_owned()).filter(|x| !x.is_empty());
    }
    std::fs::read_to_string("/proc/cpuinfo").ok()?.lines().find(|l| l.starts_with("model name"))?.split_once(':').map(|x| x.1.trim().to_owned())
}

/// RAM in MiB from `/proc/meminfo` (Linux); none elsewhere (the full probe reads it).
fn probe_ram_linux() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo").ok()?.lines().find(|l| l.starts_with("MemTotal:"))?.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb / 1024)
}

/// The slow part of the probe on Windows, one PowerShell query: RAM, the disk holding `home`,
/// and the first graphics adapter (used only when `nvidia-smi` names none).
fn probe_windows_slow(home: &Path) -> (Option<u64>, Option<Piece>, Option<(String, u64)>) {
    let drive = home.display().to_string().chars().next().filter(char::is_ascii_alphabetic).unwrap_or('C');
    let script = format!(
        "$c=Get-CimInstance Win32_ComputerSystem; \"ram|\"+$c.TotalPhysicalMemory; \
         try {{ $n=(Get-Partition -DriveLetter {drive} -ErrorAction Stop | Get-Disk).Number; $x=Get-PhysicalDisk | Where-Object {{$_.DeviceId -eq \"$n\"}}; \"disk|\"+$x.MediaType+\"|\"+$x.BusType+\"|\"+$x.FriendlyName }} catch {{}}; \
         $v=Get-CimInstance Win32_VideoController | Select-Object -First 1; \"video|\"+$v.Name+\"|\"+$v.AdapterRAM"
    );
    let Some(out) = run_quiet("powershell", &["-NoProfile", "-NonInteractive", "-Command", &script]) else { return (None, None, None) };
    let mut ram = None;
    let mut disk = None;
    let mut video = None;
    for l in out.lines() {
        let f: Vec<&str> = l.split('|').map(str::trim).collect();
        match f.first().copied() {
            Some("ram") => ram = f.get(1).and_then(|b| b.parse::<u64>().ok()).map(|b| b / (1024 * 1024)),
            Some("disk") if f.len() >= 4 => disk = Some(disk_piece(f.get(1).unwrap_or(&""), f.get(2).unwrap_or(&""), f.get(3).unwrap_or(&""))),
            Some("video") if f.len() >= 3 => video = f.get(1).map(|n| ((*n).to_owned(), f.get(2).and_then(|b| b.parse::<u64>().ok()).map_or(0, |b| b / (1024 * 1024)))),
            _ => {}
        }
    }
    (ram, disk, video)
}

/// The disk holding `home` on Linux, from `/sys/block` (rotational flag and bus by name).
fn probe_disk_linux() -> Option<Piece> {
    let rd = std::fs::read_dir("/sys/block").ok()?;
    let dev = rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).find(|n| n.starts_with("nvme") || n.starts_with("sd"))?;
    let rot = std::fs::read_to_string(format!("/sys/block/{dev}/queue/rotational")).ok()?;
    let model = std::fs::read_to_string(format!("/sys/block/{dev}/device/model")).unwrap_or_default();
    let bus = if dev.starts_with("nvme") { "nvme" } else { "sata" };
    Some(disk_piece(if rot.trim() == "1" { "hdd" } else { "ssd" }, bus, &model))
}

/// Reads this machine's pieces now. `full` also reads RAM, the disk and a graphics card that is
/// not NVIDIA (a slow query on Windows, a few seconds); otherwise those come from `cached`.
#[must_use]
pub fn probe(home: &Path, full: bool, cached: Option<&Machine>) -> Machine {
    let mut pieces = Vec::new();
    let mut gpu = probe_nvidia();
    if let Some(name) = probe_cpu_name() {
        pieces.push(cpu_piece(&name));
    }
    let from_cache = |k: Kind| cached.and_then(|m| m.piece(k).cloned());
    if full {
        if cfg!(windows) {
            let (ram, disk, video) = probe_windows_slow(home);
            if let Some(r) = ram {
                pieces.push(ram_piece(r));
            }
            if let Some(d) = disk {
                pieces.push(d);
            }
            if gpu.is_none()
                && let Some((name, mib)) = video
            {
                gpu = Some(gpu_piece(&name, mib, "", ""));
            }
        } else {
            if let Some(r) = probe_ram_linux() {
                pieces.push(ram_piece(r));
            }
            if let Some(d) = probe_disk_linux() {
                pieces.push(d);
            }
        }
    } else {
        match probe_ram_linux() {
            Some(r) => pieces.push(ram_piece(r)),
            None => pieces.extend(from_cache(Kind::Ram)),
        }
        pieces.extend(from_cache(Kind::Disk));
        // A card that is not NVIDIA is known only from the full probe.
        if gpu.is_none()
            && let Some(g) = from_cache(Kind::Gpu).filter(|g| g.vendor != "nvidia")
        {
            gpu = Some(g);
        }
    }
    pieces.extend(gpu);
    pieces.sort_by_key(|p| p.kind);
    Machine { pieces, probed_ms: nmtype::now_ms(), how: if full { "full" } else { "fast" }.into() }
}

/// This machine, from the cache in `memory/stacks/machine.json` when it is fresh and the
/// graphics card and processor read now still match it; otherwise read again (fully when `full`
/// or when the cache lacks RAM or the disk) and the cache rewritten.
#[must_use]
pub fn machine(home: &Path, full: bool) -> Machine {
    crate::library::tidy(home);
    let file = crate::library::machine_file(home);
    let cached = std::fs::read(&file).ok().and_then(|b| nmjson::parse(&b).ok()).and_then(|v| Machine::from_value(&v).ok());
    let fast = probe(home, false, cached.as_ref());
    if let Some(c) = &cached {
        let fresh = nmtype::now_ms().saturating_sub(c.probed_ms) < MACHINE_MS;
        let same = [Kind::Gpu, Kind::Cpu].iter().all(|k| fast.piece(*k) == c.piece(*k));
        let complete = c.piece(Kind::Ram).is_some() && c.piece(Kind::Disk).is_some();
        if fresh && same && (complete || !full) {
            return c.clone();
        }
    }
    // Off Windows the full probe is as quick as the fast one.
    let m = if full || !cfg!(windows) { probe(home, true, cached.as_ref()) } else { fast };
    if file.parent().is_some_and(|d| std::fs::create_dir_all(d).is_ok()) {
        let _ = std::fs::write(&file, nmjson::pretty::pretty(&m.to_value()));
    }
    m
}

/// Self-check of `hardware`: names become the expected identities and families, and relatives
/// measure nearer by memory.
pub(crate) fn check() -> Result<(), String> {
    let g = gpu_piece("NVIDIA GeForce GTX 1650", 4096, "581.83", "7.5");
    if g.identity() != "gpu:nvidia:geforce-gtx-1650:4096mib:driver-581" || g.family_key() != "gpu:nvidia:turing" {
        return Err(format!("nmlearn.hardware: graphics card identity {} family {}", g.identity(), g.family_key()));
    }
    let c = cpu_piece("12th Gen Intel(R) Core(TM) i5-12450H");
    if c.identity() != "cpu:intel:core-i5-12450h" || c.family_key() != "cpu:intel:core-gen12" {
        return Err(format!("nmlearn.hardware: processor identity {} family {}", c.identity(), c.family_key()));
    }
    let a = cpu_piece("AMD Ryzen 7 5800X 8-Core Processor");
    if a.identity() != "cpu:amd:ryzen-7-5800x" || a.family_key() != "cpu:amd:ryzen-5000" {
        return Err(format!("nmlearn.hardware: processor identity {} family {}", a.identity(), a.family_key()));
    }
    if ram_piece(32_436).identity() != "ram:32gib" {
        return Err("nmlearn.hardware: RAM identity".into());
    }
    if disk_piece("SSD", "USB", "Samsung PSSD T7").identity() != "disk:ssd:usb:samsung-pssd-t7" {
        return Err("nmlearn.hardware: disk identity".into());
    }
    let near = gpu_piece("NVIDIA GeForce GTX 1660", 6144, "581.83", "7.5");
    let far = gpu_piece("NVIDIA GeForce RTX 2080 Ti", 11_264, "581.83", "7.5");
    if g.distance(&near) >= g.distance(&far) {
        return Err("nmlearn.hardware: nearer memory does not measure nearer".into());
    }
    let m = Machine { pieces: vec![c, g], probed_ms: 1, how: "given".into() };
    let back = Machine::from_value(&m.to_value())?;
    if back.pieces != m.pieces {
        return Err("nmlearn.hardware: a machine does not read back as written".into());
    }
    Ok(())
}
