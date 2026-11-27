//! The machine's capacity profile (Roadmap Phase 4.6, brought forward on 2026-11-24 at the
//! direction of Luis Jake Gabriel III: the whole system adapts and scales to the hardware it runs
//! on, from a small card to a large one, or none). One place reads what this machine has: the
//! card (name, total and free memory, compute capability), host memory (total and available) and
//! the thread budget, and the limits that were fixed for one GTX 1650 derive from it:
//!
//! - the card ceiling homeostasis acts at: the card's total less the live calls' room (the
//!   measured working peak of this process's runs, an eighth of the card until one is measured),
//!   in place of a fixed 3,500 MiB;
//! - the host memory ceiling: three quarters of the machine's memory, in place of a fixed 24,000 MB.
//!
//! Measured numbers are kept per machine (`<home>/memory/machine.json`, keyed by the card), so a
//! restart starts from this machine's measures rather than the fallbacks: the working peak is
//! saved when it grows and seeded when the card is first registered (`residency`), and other parts
//! keep theirs through `remember` and `recall`.
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// What this machine has now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Profile {
    /// The card in use, if any.
    pub card: Option<Card>,
    /// Host memory in MiB: (available now, total).
    pub host_mb: (u64, u64),
    /// Worker threads the process budget grants (under the user's thread limit).
    pub threads: usize,
    /// The processor's own thread count (Roadmap Phase 4.5, 2026-11-25).
    pub hw_threads: usize,
}

/// The card in use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Card {
    pub name: String,
    pub total_mb: usize,
    pub free_mb: usize,
    pub compute: (i32, i32),
    /// The live calls' room in MiB, and whether it is measured (else an eighth of the card).
    pub reserve_mb: (usize, bool),
}

/// The profile now.
#[must_use]
pub fn profile() -> Profile {
    let r = crate::residency::get();
    let card = r.card().map(|i| Card {
        name: i.name,
        total_mb: i.total_bytes >> 20,
        // Roadmap Phase 4.5 (2026-11-25): the driver's figure, not the one under the user's
        // card limit, so the recommendation sees what the card really has free.
        free_mb: r.memory_driver().map_or(0, |(free, _)| free >> 20),
        compute: (i.major, i.minor),
        reserve_mb: { let (b, m) = r.reserve(); (b >> 20, m) },
    });
    // 2026-11-25: a process that has not opened a card (the interface server, a one-shot command)
    // reads the card the machine would use through the driver, without a context, so the profile
    // and the recommendation see this machine's card rather than none (seen in the interface: no
    // card, so it recommended processor only for a GTX 1650 machine). Free memory cannot be read
    // without a context there; it is given as the card's total and the reserve as unmeasured.
    let card = card.or_else(unopened_card);
    let host_mb = nmffi::dynlib::system_memory().map_or((0, 0), |(avail, total)| (avail >> 20, total >> 20));
    Profile { card, host_mb, threads: nmtype::budget::total(), hw_threads: nmtype::budget::hardware() }
}

/// The card this machine would use, read once through the driver without opening it; `None` when
/// the processor path would be taken.
fn unopened_card() -> Option<Card> {
    static SEEN: std::sync::OnceLock<Option<Card>> = std::sync::OnceLock::new();
    SEEN.get_or_init(|| {
        let devices = nmffi::cuda::devices().ok()?;
        let choice = crate::device::choose(&devices.cards, std::env::var(crate::device::SWITCH).ok().as_deref(), std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref());
        let crate::device::Choice::Card { ordinal, .. } = choice else { return None };
        let (_, i) = devices.cards.iter().find(|(o, _)| *o == ordinal)?;
        let total_mb = i.total_bytes >> 20;
        Some(Card { name: i.name.clone(), total_mb, free_mb: total_mb, compute: (i.major, i.minor), reserve_mb: ((total_mb / 8), false) })
    })
    .clone()
}

/// The card memory (MiB) above which homeostasis narrows and suspends: the card's total less the
/// live calls' room; `None` without a card.
#[must_use]
pub fn gpu_ceiling_mb() -> Option<usize> {
    let c = profile().card?;
    Some(c.total_mb.saturating_sub(c.reserve_mb.0).max(c.total_mb / 2))
}

/// The host memory (MiB) above which homeostasis suspends: three quarters of the machine's
/// memory (2,048 MiB when it cannot be read).
#[must_use]
pub fn ram_ceiling_mb() -> u64 {
    match nmffi::dynlib::system_memory() {
        Ok((_, total)) if total > 0 => (total >> 20) / 4 * 3,
        _ => 2_048,
    }
}

/// The profile as a value (for the interface, `nm_live` and the audit).
#[must_use]
pub fn to_value(p: &Profile) -> Value {
    let n = |x: u64| Value::Number(x.to_string());
    let mut o = BTreeMap::new();
    o.insert(
        "card".to_owned(),
        p.card.as_ref().map_or(Value::Null, |c| {
            let mut m = BTreeMap::new();
            m.insert("name".to_owned(), Value::String(c.name.clone()));
            m.insert("total_mb".to_owned(), n(c.total_mb as u64));
            m.insert("free_mb".to_owned(), n(c.free_mb as u64));
            m.insert("compute".to_owned(), Value::String(format!("{}.{}", c.compute.0, c.compute.1)));
            m.insert("reserve_mb".to_owned(), n(c.reserve_mb.0 as u64));
            m.insert("reserve_measured".to_owned(), Value::Bool(c.reserve_mb.1));
            Value::Object(m)
        }),
    );
    o.insert("host_available_mb".to_owned(), n(p.host_mb.0));
    o.insert("host_total_mb".to_owned(), n(p.host_mb.1));
    o.insert("threads".to_owned(), n(p.threads as u64));
    o.insert("hardware_threads".to_owned(), n(p.hw_threads as u64));
    o.insert("gpu_ceiling_mb".to_owned(), gpu_ceiling_mb().map_or(Value::Null, |x| n(x as u64)));
    o.insert("ram_ceiling_mb".to_owned(), n(ram_ceiling_mb()));
    Value::Object(o)
}

/// Roadmap Phase 4.5 (2026-11-25): the hardware tier the recommendation picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// No card the engine can use.
    CpuOnly,
    /// A card under 6 GB (for example the 4 GB GTX 1650).
    Small,
    /// 6 GB up to 16 GB (the 8 to 12 GB cards).
    Mid,
    /// 16 GB and up.
    Large,
}

impl Tier {
    /// The tier of a card of `total_mb` (`None`: no card).
    #[must_use]
    pub fn of(card_total_mb: Option<usize>) -> Self {
        match card_total_mb {
            None | Some(0) => Self::CpuOnly,
            Some(mb) if mb < 6 * 1024 => Self::Small,
            Some(mb) if mb < 16 * 1024 => Self::Mid,
            Some(_) => Self::Large,
        }
    }
    /// The tier's key.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::CpuOnly => "cpu-only",
            Self::Small => "small",
            Self::Mid => "mid",
            Self::Large => "large",
        }
    }
    /// The tier in words.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::CpuOnly => "Processor only (no usable graphics card)",
            Self::Small => "Small card (under 6 GB, for example a 4 GB GTX 1650)",
            Self::Mid => "Mid card (8 to 12 GB)",
            Self::Large => "Large card (16 GB and up)",
        }
    }
}

/// One recommended limit: the value as `settings.json` holds it (`auto` or a number) and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Advice {
    pub key: &'static str,
    pub value: String,
    pub reason: String,
}

/// The recommended limits for this machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recommendation {
    pub tier: Tier,
    pub vram: Advice,
    pub ram: Advice,
    pub cpu: Advice,
    /// Other things the person should know (low free disk, a busy card).
    pub notes: Vec<String>,
}

impl Recommendation {
    /// The three advised values, in `KEYS` order: (key, value).
    #[must_use]
    pub fn values(&self) -> [(&'static str, &str); 3] {
        [(self.vram.key, self.vram.value.as_str()), (self.ram.key, self.ram.value.as_str()), (self.cpu.key, self.cpu.value.as_str())]
    }
}

/// Card memory the four MicroModels take together when seated at their default precisions
/// (embedder, reranker, reader and writer, measured on the 4 GB GTX 1650: 2.17 GiB).
pub const FOUR_SEATED_MB: usize = 2_222;
/// Host memory the idle writer's load needs free (`nmapi::writer_idle::NEED_MB`).
pub const WRITER_HOST_MB: u64 = 7_000;
/// Free disk below which a note warns (the default disk floor and the room for one dense index
/// and one job filter build).
pub const DISK_LOW_GB: u64 = 8;

fn round_down(x: u64, step: u64) -> u64 {
    x / step * step
}

/// Roadmap Phase 4.5 (2026-11-25): the recommended limits for a machine profile and its free disk
/// space (bytes, when known). Every value keeps what was measured on the 4 GB GTX 1650: with the
/// card limit at `auto` all four MicroModels stay seated there.
///
/// - Card: `auto` on small and mid cards (the four seated take about 2.2 GiB, and `auto` follows
///   what the driver reports free, so the residency manager evicts only when another program
///   needs the card); on a large card a limit of half the card (at least 8 GiB), which holds the
///   four with ample room for window copies and leaves the rest to other programs; 0 (processor
///   only) without a card.
/// - Host memory: `auto` from 16 GB up (the chunk cache takes half the available memory and
///   homeostasis acts at three quarters); below that three quarters of the machine, so the system
///   and the editor keep a quarter.
/// - Threads: `auto` with a card (the processor only helps; background work already yields to a
///   question at each partition); processor only, every thread but one, so the machine stays
///   responsive while every MicroModel runs on the processor.
#[must_use]
pub fn recommend(p: &Profile, disk_free_bytes: Option<u64>) -> Recommendation {
    let tier = Tier::of(p.card.as_ref().map(|c| c.total_mb));
    let mut notes = Vec::new();
    let vram = match (tier, p.card.as_ref()) {
        (Tier::Large, Some(c)) => {
            let cap = round_down((c.total_mb as u64 / 2).max(8_192), 1_024);
            Advice {
                key: crate::limits::VRAM_KEY,
                value: cap.to_string(),
                reason: format!(
                    "{} has {} MiB. The four MicroModels seated take about {FOUR_SEATED_MB} MiB, so a limit of {cap} MiB keeps all four seated with ample room for window copies and leaves the rest of the card to other programs.",
                    c.name, c.total_mb
                ),
            }
        }
        (Tier::Small | Tier::Mid, Some(c)) => {
            // Free memory is measured only by a process that holds the card (the measured reserve
            // says so); a process that read the card through the driver alone says it was not read.
            let measured = c.reserve_mb.1;
            if measured && c.free_mb < FOUR_SEATED_MB {
                notes.push(format!(
                    "The card has {} MiB free now, under the {FOUR_SEATED_MB} MiB the four MicroModels take seated: other programs are using it, so the writer will load per idle session (evict and restore) until they let go.",
                    c.free_mb
                ));
            }
            let free = if measured { format!("{} MiB free now", c.free_mb) } else { "free memory is read once an engine holds the card".to_owned() };
            Advice {
                key: crate::limits::VRAM_KEY,
                value: "auto".to_owned(),
                reason: format!(
                    "{} has {} MiB ({free}). The four MicroModels seated take about {FOUR_SEATED_MB} MiB; auto lets them all stay seated and follows what the driver reports free, evicting and restoring only when another program needs the card. A fixed limit here would only force evictions.",
                    c.name, c.total_mb
                ),
            }
        }
        _ => Advice {
            key: crate::limits::VRAM_KEY,
            value: "0".to_owned(),
            reason: "No graphics card the engine can use was found, so the MicroModels run on the processor (0 means processor only).".to_owned(),
        },
    };
    let (avail, total) = p.host_mb;
    let ram = if total == 0 || total >= 16 * 1024 {
        Advice {
            key: crate::limits::RAM_KEY,
            value: "auto".to_owned(),
            reason: if total == 0 {
                "The machine's memory could not be read; auto keeps the derived limits (the chunk cache takes half of what is available).".to_owned()
            } else {
                format!(
                    "The machine has {total} MiB ({avail} MiB available now). Auto lets the encoded-chunk cache take half of what is available and homeostasis act at three quarters, which leaves the system ample room."
                )
            },
        }
    } else {
        let cap = round_down(total / 4 * 3, 512).max(1_024);
        if cap < WRITER_HOST_MB {
            notes.push(format!("With {total} MiB of memory the idle writer's load ({WRITER_HOST_MB} MiB free asked) waits until enough is free; notes come more slowly."));
        }
        Advice {
            key: crate::limits::RAM_KEY,
            value: cap.to_string(),
            reason: format!("The machine has {total} MiB. A limit of {cap} MiB (three quarters) keeps a quarter for the system and the editor; the chunk cache then takes at most half the limit."),
        }
    };
    let hw = p.hw_threads.max(1);
    let cpu = if tier == Tier::CpuOnly && hw > 2 {
        let n = hw - 1;
        Advice {
            key: crate::limits::CPU_KEY,
            value: n.to_string(),
            reason: format!("Every MicroModel runs on the processor here. {n} of the {hw} threads leaves one for the system and the editor, so the machine stays responsive."),
        }
    } else {
        Advice {
            key: crate::limits::CPU_KEY,
            value: "auto".to_owned(),
            reason: format!(
                "{hw} threads. {}Background work already yields to a question at every partition, so auto keeps questions fast without slowing the machine.",
                if tier == Tier::CpuOnly { "" } else { "The card does the heavy work. " }
            ),
        }
    };
    if let Some(free) = disk_free_bytes {
        let gb = free >> 30;
        if gb < DISK_LOW_GB {
            notes.push(format!("Only {gb} GB is free on the Nano's drive: the meaning index and job filter builds need room. Prune rebuildable storage (nm_storage prune, or the Storage page)."));
        }
    }
    Recommendation { tier, vram, ram, cpu, notes }
}

/// The recommendation as a value.
#[must_use]
pub fn recommendation_value(r: &Recommendation) -> Value {
    let advice = |a: &Advice| {
        let mut m = BTreeMap::new();
        m.insert("key".to_owned(), Value::String(a.key.to_owned()));
        m.insert("value".to_owned(), Value::String(a.value.clone()));
        m.insert("reason".to_owned(), Value::String(a.reason.clone()));
        Value::Object(m)
    };
    let mut o = BTreeMap::new();
    o.insert("tier".to_owned(), Value::String(r.tier.key().to_owned()));
    o.insert("tier_label".to_owned(), Value::String(r.tier.label().to_owned()));
    o.insert("vram_mb".to_owned(), advice(&r.vram));
    o.insert("ram_mb".to_owned(), advice(&r.ram));
    o.insert("cpu_threads".to_owned(), advice(&r.cpu));
    o.insert("notes".to_owned(), Value::Array(r.notes.iter().map(|n| Value::String(n.clone())).collect()));
    Value::Object(o)
}

/// The file holding this machine's measures.
fn measures_file(home: &Path) -> PathBuf {
    home.join("memory").join("machine.json")
}

/// The key a measure set is kept under: the card's name and size, or "processor".
#[must_use]
pub fn machine_key(card: Option<(&str, usize)>) -> String {
    card.map_or_else(|| "processor".to_owned(), |(name, mb)| format!("{name}|{mb}"))
}

fn read_all(home: &Path) -> BTreeMap<String, Value> {
    std::fs::read(measures_file(home))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| if let Value::Object(m) = v { Some(m) } else { None })
        .unwrap_or_default()
}

/// A measure kept for this machine (`key` names the machine, `name` the measure).
#[must_use]
pub fn recall(home: &Path, key: &str, name: &str) -> Option<u64> {
    read_all(home).get(key).and_then(|m| m.get(name).ok()).and_then(|v| v.integer().ok())
}

/// Keeps a measure for this machine (best effort: a failed write is ignored).
pub fn remember(home: &Path, key: &str, name: &str, value: u64) {
    let mut all = read_all(home);
    let entry = all.entry(key.to_owned()).or_insert_with(|| Value::Object(BTreeMap::new()));
    if let Value::Object(m) = entry {
        m.insert(name.to_owned(), Value::Number(value.to_string()));
    }
    let file = measures_file(home);
    if let Some(d) = file.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    if let Ok(text) = Value::Object(all).encode() {
        let tmp = file.with_extension("json.tmp");
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, &file);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_round_trip_per_machine() {
        let d = nmtype::nano_root().join("scratch").join("unit-tests").join(format!("nmengn-capacity-{}-{}", std::process::id(), nmtype::now_ms()));
        let _ = std::fs::remove_dir_all(&d);
        let a = machine_key(Some(("Card A", 4096)));
        let b = machine_key(None);
        assert_eq!(recall(&d, &a, "work_peak_bytes"), None);
        remember(&d, &a, "work_peak_bytes", 512);
        remember(&d, &b, "work_peak_bytes", 64);
        remember(&d, &a, "writer_mb", 1105);
        assert_eq!(recall(&d, &a, "work_peak_bytes"), Some(512));
        assert_eq!(recall(&d, &a, "writer_mb"), Some(1105));
        assert_eq!(recall(&d, &b, "work_peak_bytes"), Some(64));
        let _ = std::fs::remove_dir_all(&d);
    }

    fn card(name: &str, total_mb: usize, free_mb: usize) -> Option<Card> {
        Some(Card { name: name.to_owned(), total_mb, free_mb, compute: (7, 5), reserve_mb: (300, true) })
    }
    fn machine(card: Option<Card>, host_total: u64, threads: usize) -> Profile {
        Profile { card, host_mb: (host_total / 2, host_total), threads, hw_threads: threads }
    }

    // Roadmap Phase 4.5 (2026-11-25): one synthetic machine per tier.
    #[test]
    fn recommend_small_card_keeps_four_seated_with_auto() {
        // This machine: a 4 GB GTX 1650 with the four MicroModels seated in 2.17 GiB.
        let r = recommend(&machine(card("NVIDIA GeForce GTX 1650", 4_096, 3_500), 32_768, 12), Some(50 << 30));
        assert_eq!(r.tier, Tier::Small);
        assert_eq!(r.values(), [(crate::limits::VRAM_KEY, "auto"), (crate::limits::RAM_KEY, "auto"), (crate::limits::CPU_KEY, "auto")]);
        assert!(r.vram.reason.contains("all stay seated"), "{}", r.vram.reason);
        assert!(r.notes.is_empty(), "{:?}", r.notes);
        // A busy small card: still auto, with a note.
        let busy = recommend(&machine(card("GTX 1650", 4_096, 1_200), 32_768, 12), None);
        assert_eq!(busy.vram.value, "auto");
        assert!(busy.notes.iter().any(|n| n.contains("evict and restore")));
    }

    #[test]
    fn recommend_mid_and_large_cards() {
        for total in [8_192, 12_288] {
            let r = recommend(&machine(card("RTX", total, total - 500), 32_768, 16), Some(100 << 30));
            assert_eq!(r.tier, Tier::Mid);
            assert_eq!(r.vram.value, "auto");
        }
        let r = recommend(&machine(card("RTX 4090", 24_576, 23_000), 65_536, 32), Some(100 << 30));
        assert_eq!(r.tier, Tier::Large);
        assert_eq!(r.vram.value, "12288");
        let r = recommend(&machine(card("RTX 4080", 16_384, 15_000), 65_536, 32), None);
        assert_eq!((r.tier, r.vram.value.as_str()), (Tier::Large, "8192"));
        assert_eq!(r.cpu.value, "auto");
    }

    #[test]
    fn recommend_cpu_only_and_small_host() {
        let r = recommend(&machine(None, 8_192, 8), Some(3 << 30));
        assert_eq!(r.tier, Tier::CpuOnly);
        assert_eq!(r.vram.value, "0");
        assert_eq!(r.ram.value, "6144");
        assert_eq!(r.cpu.value, "7");
        assert!(r.notes.iter().any(|n| n.contains("GB is free")));
        assert!(r.notes.iter().any(|n| n.contains("idle writer")));
        // Every advised value passes the controls' own validation.
        for (k, v) in r.values() {
            assert_eq!(crate::limits::normalize(k, v), Some(Ok(v.to_owned())));
        }
        let v = recommendation_value(&r);
        assert_eq!(v.get("tier").and_then(Value::string).ok(), Some("cpu-only"));
        assert!(v.get("vram_mb").and_then(|a| a.get("reason")).and_then(Value::string).is_ok_and(|s| !s.is_empty()));
    }

    #[test]
    fn ceilings_have_sane_floors() {
        assert!(ram_ceiling_mb() >= 1_024);
        if let Some(g) = gpu_ceiling_mb() {
            assert!(g > 0);
        }
    }
}
