//! Which CUDA card the served code uses, and the plain-English line that says so (Roadmap Phase
//! 4.6 (2026-11-25)). One card per process: the first decision is kept for the life of the
//! process, so every MicroModel, the residency manager and the placement table see the same card.
//!
//! The choice, in order:
//! - `NANO_GPU=cpu` (also `off`, `none`, `processor`) forces the processor path (for testing a
//!   machine without a card on this one); `NANO_GPU=<ordinal>` names a card.
//! - `CUDA_VISIBLE_DEVICES`, when set, is honoured: the driver itself hides and renumbers cards by
//!   it, so the first card it lists is used; an empty value or `-1` hides every card.
//! - Otherwise the card with the most total memory among those at compute capability 7.5 or newer
//!   (the oldest target the kernels are built for), the lowest ordinal on a tie.
//!
//! Every way of not getting a card ends on the processor with one line that says why: no driver
//! (`nvcuda.dll` missing), no NVIDIA card, a card older than the kernels (Pascal and earlier), a
//! driver too old for the kernels or for the entry points this program calls, or the switch.
use nmffi::GpuInfo;
use nmffi::cuda::MIN_CAPABILITY;
use std::sync::{Mutex, OnceLock};

/// The switch that names a card or forces the processor path.
pub const SWITCH: &str = "NANO_GPU";

/// Why the processor path is taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// `NANO_GPU=cpu`.
    Forced,
    /// `nvcuda.dll` is missing: no NVIDIA display driver.
    NoDriver,
    /// The driver is there but reports no card.
    NoCard,
    /// `CUDA_VISIBLE_DEVICES` hides every card.
    Hidden,
    /// The card is older than the kernels (compute capability under 7.5).
    OldCard,
    /// The driver is too old for the kernels or lacks an entry point this program calls.
    OldDriver,
    /// Any other failure to open the card.
    Other,
}

/// The device decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Choice {
    /// Use the card with this ordinal; `line` says which and why.
    Card { ordinal: i32, line: String },
    /// Run on the processor; `line` says why in plain English.
    Processor { reason: Reason, line: String },
}

impl Choice {
    /// The plain-English line.
    #[must_use]
    pub fn line(&self) -> &str {
        match self {
            Self::Card { line, .. } | Self::Processor { line, .. } => line,
        }
    }
}

fn eligible(i: &GpuInfo) -> bool {
    (i.major, i.minor) >= MIN_CAPABILITY
}

fn describe(o: i32, i: &GpuInfo) -> String {
    format!("card {o}, {} (compute {}.{}, {} MB)", i.name, i.major, i.minor, i.total_bytes >> 20)
}

fn old_card(o: i32, i: &GpuInfo) -> String {
    format!(
        "Running on the processor: this card ({}) is older than the kernels support (compute {}.{} or newer, Turing and later).",
        describe(o, i),
        MIN_CAPABILITY.0,
        MIN_CAPABILITY.1
    )
}

/// The card with the most memory among those the kernels run on (lowest ordinal on a tie).
fn largest(cards: &[(i32, GpuInfo)]) -> Option<&(i32, GpuInfo)> {
    cards.iter().filter(|(_, i)| eligible(i)).fold(None, |best: Option<&(i32, GpuInfo)>, c| match best {
        Some(b) if b.1.total_bytes >= c.1.total_bytes => Some(b),
        _ => Some(c),
    })
}

fn forced(v: &str) -> bool {
    matches!(v.trim().to_ascii_lowercase().as_str(), "cpu" | "off" | "none" | "processor")
}

fn hides_all(v: &str) -> bool {
    let v = v.trim();
    v.is_empty() || v == "-1" || v.eq_ignore_ascii_case("NoDevFiles")
}

/// The automatic pick: the largest card the kernels run on, or the processor with the reason.
fn automatic(cards: &[(i32, GpuInfo)], note: &str) -> Choice {
    if let Some((o, i)) = largest(cards) {
        let why = if cards.iter().filter(|(_, c)| eligible(c)).count() > 1 {
            "the most memory among the cards the kernels run on"
        } else {
            "the only card the kernels run on"
        };
        return Choice::Card { ordinal: *o, line: format!("Graphics card in use: {}, chosen as {why}{note}.", describe(*o, i)) };
    }
    match cards.first() {
        Some((o, i)) => Choice::Processor { reason: Reason::OldCard, line: old_card(*o, i) },
        None => Choice::Processor { reason: Reason::NoCard, line: "Running on the processor: no NVIDIA graphics card was found.".to_owned() },
    }
}

/// The device decision from the card list and the two settings (`NANO_GPU`, then
/// `CUDA_VISIBLE_DEVICES`); pure, so it is tested on synthetic card lists.
#[must_use]
pub fn choose(cards: &[(i32, GpuInfo)], nano_gpu: Option<&str>, visible: Option<&str>) -> Choice {
    let mut note = String::new();
    if let Some(v) = nano_gpu.map(str::trim).filter(|v| !v.is_empty()) {
        if forced(v) {
            return Choice::Processor { reason: Reason::Forced, line: format!("Running on the processor: {SWITCH}={v} turns the graphics card off for this process.") };
        }
        match v.parse::<i32>() {
            Ok(n) => match cards.iter().find(|(o, _)| *o == n) {
                Some((o, i)) if eligible(i) => {
                    return Choice::Card { ordinal: *o, line: format!("Graphics card in use: {}, chosen by {SWITCH}={v}.", describe(*o, i)) };
                }
                Some((o, i)) => {
                    let other = largest(cards).map(|(b, _)| format!(" Set {SWITCH}={b} to use a card the kernels run on.")).unwrap_or_default();
                    return Choice::Processor { reason: Reason::OldCard, line: format!("{}{other}", old_card(*o, i)) };
                }
                None => note = format!(" ({SWITCH}={v} names no card, so the automatic choice was made)"),
            },
            Err(_) => note = format!(" ({SWITCH}={v} is not a card number or cpu, so the automatic choice was made)"),
        }
    }
    if note.is_empty()
        && let Some(v) = visible
    {
        if hides_all(v) {
            return Choice::Processor { reason: Reason::Hidden, line: "Running on the processor: CUDA_VISIBLE_DEVICES hides every card.".to_owned() };
        }
        // The driver has already hidden and renumbered the cards by the variable: its first card
        // is the first one listed.
        if let Some((o, i)) = cards.first()
            && eligible(i)
        {
            return Choice::Card { ordinal: *o, line: format!("Graphics card in use: {}, the first card CUDA_VISIBLE_DEVICES={} lists.", describe(*o, i), v.trim()) };
        }
        note = format!(" (the first card CUDA_VISIBLE_DEVICES={} lists is older than the kernels support)", v.trim());
    }
    automatic(cards, &note)
}

/// True when `err` holds `CUDA error <code>` as a whole number.
fn has_code(err: &str, code: i32) -> bool {
    let pat = format!("CUDA error {code}");
    err.match_indices(&pat).any(|(at, m)| err.get(at + m.len()..).is_some_and(|rest| !rest.starts_with(|c: char| c.is_ascii_digit())))
}

/// Why opening the card failed, from the driver layer's error text.
#[must_use]
pub fn classify(err: &str) -> Reason {
    if err.contains("forces the processor path") {
        Reason::Forced
    } else if err.contains("nvcuda.dll not found") {
        Reason::NoDriver
    } else if err.contains("older than the kernels support") {
        Reason::OldCard
    } else if err.contains("reports no CUDA card") || has_code(err, 100) {
        Reason::NoCard
    } else if err.contains("has no symbol")
        // 35 insufficient driver, 209 no kernel image for the card, 218 invalid PTX, 222 PTX newer
        // than the driver, 803 and 804 driver and system mismatches.
        || [35, 209, 218, 222, 803, 804].iter().any(|c| has_code(err, *c))
    {
        Reason::OldDriver
    } else {
        Reason::Other
    }
}

/// The driver version as people say it (13000 is 13.0).
#[must_use]
pub fn driver_text(v: i32) -> String {
    format!("{}.{}", v / 1000, (v % 1000) / 10)
}

/// The plain-English line for a processor fallback found by an error (`err`, the driver layer's
/// text; `driver`, the driver version when known; `visible`, `CUDA_VISIBLE_DEVICES`).
#[must_use]
pub fn fallback_line(err: &str, driver: Option<i32>, visible: Option<&str>) -> (Reason, String) {
    let reason = match classify(err) {
        Reason::NoCard if visible.is_some() => Reason::Hidden,
        r => r,
    };
    let line = match reason {
        Reason::Forced => format!("Running on the processor: {SWITCH}=cpu turns the graphics card off for this process."),
        Reason::NoDriver => "Running on the processor: no NVIDIA driver was found (nvcuda.dll is missing). Install the NVIDIA display driver to use a graphics card.".to_owned(),
        Reason::NoCard => "Running on the processor: the NVIDIA driver is installed but reports no graphics card.".to_owned(),
        Reason::Hidden => format!("Running on the processor: CUDA_VISIBLE_DEVICES={} leaves no card visible.", visible.unwrap_or_default().trim()),
        Reason::OldCard => {
            // The driver layer names the card and its capability; say it in the same plain form.
            let card = err.split("nmffi.cuda: ").nth(1).and_then(|r| r.split(" is compute capability").next()).unwrap_or("the card");
            format!(
                "Running on the processor: this card ({card}) is older than the kernels support (compute {}.{} or newer, Turing and later).",
                MIN_CAPABILITY.0, MIN_CAPABILITY.1
            )
        }
        Reason::OldDriver => format!(
            "Running on the processor: the NVIDIA driver{} is too old for the kernels. Update the NVIDIA display driver (CUDA 13.0 or newer) to use the graphics card.",
            driver.map(|v| format!(" (CUDA {})", driver_text(v))).unwrap_or_default()
        ),
        Reason::Other => "Running on the processor: the graphics card could not be opened.".to_owned(),
    };
    (reason, line)
}

/// The kept decision: the ordinal, or the processor line with the driver layer's text after it.
static DECISION: OnceLock<Result<i32, String>> = OnceLock::new();
/// The line of the last device open (what the card actually did), for status.
static OPENED: Mutex<Option<String>> = Mutex::new(None);

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// The process's card ordinal, decided once (and logged once, in plain English, on standard
/// error).
/// # Errors
/// The processor path: the plain line, then the driver layer's text in brackets.
pub fn ordinal() -> Result<i32, String> {
    DECISION
        .get_or_init(|| {
            let nano = env(SWITCH);
            let visible = env("CUDA_VISIBLE_DEVICES");
            let r = if nano.as_deref().is_some_and(forced) {
                match choose(&[], nano.as_deref(), None) {
                    Choice::Processor { line, .. } => Err(format!("{line} [{}]", nmffi::cuda::forced_processor().unwrap_or_default())),
                    Choice::Card { ordinal, .. } => Ok(ordinal),
                }
            } else {
                match nmffi::cuda::devices() {
                    Ok(d) => match choose(&d.cards, nano.as_deref(), visible.as_deref()) {
                        Choice::Card { ordinal, line } => {
                            eprintln!("device: {line}");
                            Ok(ordinal)
                        }
                        Choice::Processor { line, .. } => Err(format!("{line} [{}]", nmffi::cuda::NO_USABLE)),
                    },
                    Err(e) => Err(format!("{} [{e}]", fallback_line(&e, None, visible.as_deref()).1)),
                }
            };
            if let Err(e) = &r {
                eprintln!("device: {}", plain(e));
            }
            r
        })
        .clone()
}

/// The plain part of a fallback error (before the bracketed driver text).
#[must_use]
pub fn plain(err: &str) -> &str {
    err.split(" [").next().unwrap_or(err).trim()
}

/// Opens the process's card (`ordinal`).
/// # Errors
/// The processor path, as `ordinal` says it; or the card failed to open, said the same way.
pub fn open() -> Result<nmffi::Gpu, String> {
    // The decision's own fallback was logged when it was made; a failed open of the chosen card is
    // logged here, once per distinct line.
    let (r, opened) = match ordinal() {
        Err(e) => (Err(e), false),
        Ok(o) => (
            nmffi::Gpu::open(o).map_err(|e| {
                let driver = nmffi::cuda::devices().ok().and_then(|d| d.driver);
                let line = fallback_line(&e, driver, env("CUDA_VISIBLE_DEVICES").as_deref()).1;
                format!("{line} [{e}]")
            }),
            true,
        ),
    };
    let line = match &r {
        Ok(g) => format!("Graphics card in use: {} (compute {}.{}, {} MB).", g.info().name, g.info().major, g.info().minor, g.info().total_bytes >> 20),
        Err(e) => plain(e).to_owned(),
    };
    if let Ok(mut o) = OPENED.lock() {
        if opened && r.is_err() && o.as_deref() != Some(line.as_str()) {
            eprintln!("device: {line}");
        }
        *o = Some(line);
    }
    r
}

/// The status line for the command line, `nm_status` and `/api/live`: what the last open did,
/// or, before any open, the decision (which reads the card list without opening a card).
#[must_use]
pub fn status_line() -> String {
    if let Some(l) = OPENED.lock().ok().and_then(|o| o.clone()) {
        return l;
    }
    match ordinal() {
        Ok(o) => format!("Graphics card chosen (not opened yet): card {o}."),
        Err(e) => plain(&e).to_owned(),
    }
}

/// The status line as far as this process already knows it, without reading the card list:
/// what the last open did, else the kept decision, else `None` (nothing decided yet).
#[must_use]
pub fn status_cached() -> Option<String> {
    if let Some(l) = OPENED.lock().ok().and_then(|o| o.clone()) {
        return Some(l);
    }
    DECISION.get().map(|d| match d {
        Ok(o) => format!("Graphics card chosen (not opened yet): card {o}."),
        Err(e) => plain(e).to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(o: i32, name: &str, mb: usize, major: i32, minor: i32) -> (i32, GpuInfo) {
        (o, GpuInfo { name: name.to_owned(), total_bytes: mb << 20, major, minor })
    }

    #[test]
    fn device_picks_largest_eligible_card() {
        let cards = [card(0, "GTX 1080 Ti", 11264, 6, 1), card(1, "GTX 1650", 4095, 7, 5), card(2, "RTX 3060", 12288, 8, 6)];
        assert!(matches!(choose(&cards, None, None), Choice::Card { ordinal: 2, .. }));
        // The 11 GB Pascal card is larger but too old: the 1650 wins over it alone.
        let two = [card(0, "GTX 1080 Ti", 11264, 6, 1), card(1, "GTX 1650", 4095, 7, 5)];
        let c = choose(&two, None, None);
        assert!(matches!(c, Choice::Card { ordinal: 1, .. }), "{c:?}");
        assert!(c.line().contains("the only card the kernels run on"));
        // A tie keeps the lowest ordinal.
        let tie = [card(0, "A", 8192, 8, 9), card(1, "B", 8192, 8, 9)];
        assert!(matches!(choose(&tie, None, None), Choice::Card { ordinal: 0, .. }));
    }

    #[test]
    fn device_switch_and_visible_devices() {
        let cards = [card(0, "GTX 1650", 4095, 7, 5), card(1, "RTX 4090", 24564, 8, 9)];
        assert!(matches!(choose(&cards, Some("0"), None), Choice::Card { ordinal: 0, .. }));
        assert!(matches!(choose(&cards, Some("cpu"), None), Choice::Processor { reason: Reason::Forced, .. }));
        assert!(matches!(choose(&cards, Some("OFF"), Some("1")), Choice::Processor { reason: Reason::Forced, .. }));
        // A number naming no card, or a word that is neither, falls to the automatic pick and says so.
        let c = choose(&cards, Some("7"), None);
        assert!(matches!(c, Choice::Card { ordinal: 1, .. }) && c.line().contains("names no card"), "{c:?}");
        assert!(choose(&cards, Some("fast"), None).line().contains("not a card number"));
        // CUDA_VISIBLE_DEVICES: the driver renumbers, so its first card is used.
        assert!(matches!(choose(&cards, None, Some("1,0")), Choice::Card { ordinal: 0, .. }));
        assert!(matches!(choose(&cards, None, Some("")), Choice::Processor { reason: Reason::Hidden, .. }));
        assert!(matches!(choose(&cards, None, Some("-1")), Choice::Processor { reason: Reason::Hidden, .. }));
        // NANO_GPU wins over CUDA_VISIBLE_DEVICES.
        assert!(matches!(choose(&cards, Some("1"), Some("0")), Choice::Card { ordinal: 1, .. }));
    }

    #[test]
    fn device_old_cards_fall_back_with_a_plain_line() {
        let pascal = [card(0, "NVIDIA GeForce GTX 1060 6GB", 6144, 6, 1)];
        let c = choose(&pascal, None, None);
        assert!(matches!(c, Choice::Processor { reason: Reason::OldCard, .. }));
        assert!(c.line().contains("older than the kernels support") && c.line().contains("GTX 1060") && c.line().starts_with("Running on the processor"));
        // Naming an old card with the switch says which card would work.
        let mixed = [card(0, "GTX 1060", 6144, 6, 1), card(1, "RTX 2060", 6144, 7, 5)];
        let c = choose(&mixed, Some("0"), None);
        assert!(matches!(c, Choice::Processor { reason: Reason::OldCard, .. }) && c.line().contains("NANO_GPU=1"), "{c:?}");
        assert!(matches!(choose(&[], None, None), Choice::Processor { reason: Reason::NoCard, .. }));
        // The visible card is old: the automatic pick takes the next.
        assert!(matches!(choose(&mixed, None, Some("0,1")), Choice::Card { ordinal: 1, .. }));
    }

    #[test]
    fn device_classifies_driver_errors() {
        let forced = nmffi::cuda::forced_processor().unwrap_or_else(|| {
            "nmffi.dynlib: nvcuda.dll not found (treated as missing: NANO_GPU=cpu forces the processor path); no usable CUDA card".to_owned()
        });
        assert_eq!(classify(&forced), Reason::Forced);
        assert_eq!(classify("nmffi.dynlib: nvcuda.dll not found"), Reason::NoDriver);
        assert_eq!(classify("nmffi.cuda: the driver is installed but reports no CUDA card (CUDA error 100); no usable CUDA card"), Reason::NoCard);
        assert_eq!(classify("nmffi.cuda: cuInit failed with CUDA error 100"), Reason::NoCard);
        assert_eq!(classify("nmffi.cuda: NVIDIA GeForce GTX 1060 is compute capability 6.1, older than the kernels support (7.5 or newer); no usable CUDA card"), Reason::OldCard);
        assert_eq!(classify("nmffi.cuda: cuModuleLoadData bmq8 failed with CUDA error 222"), Reason::OldDriver);
        assert_eq!(classify("nmffi.cuda: cuModuleLoadData x failed with CUDA error 209"), Reason::OldDriver);
        assert_eq!(classify("nmffi.cuda: cuInit failed with CUDA error 35"), Reason::OldDriver);
        assert_eq!(classify("nmffi.dynlib: nvcuda.dll has no symbol cuMemcpyHtoDAsync_v2"), Reason::OldDriver);
        assert_eq!(classify("nmffi.cuda: cuInit failed with CUDA error 350"), Reason::Other);
        assert_eq!(classify("nmffi.cuda: cuDeviceGet failed with CUDA error 2"), Reason::Other);
    }

    #[test]
    fn device_fallback_lines_are_plain() {
        let (r, l) = fallback_line("nmffi.dynlib: nvcuda.dll not found", None, None);
        assert_eq!(r, Reason::NoDriver);
        assert!(l.contains("nvcuda.dll is missing"));
        let (r, l) = fallback_line("nmffi.cuda: cuModuleLoadData x failed with CUDA error 222", Some(12040), None);
        assert_eq!(r, Reason::OldDriver);
        assert!(l.contains("CUDA 12.4") && l.contains("too old"), "{l}");
        let (r, l) = fallback_line("nmffi.cuda: NVIDIA GeForce GTX 1060 is compute capability 6.1, older than the kernels support (7.5 or newer); no usable CUDA card", None, None);
        assert_eq!(r, Reason::OldCard);
        assert!(l.contains("(NVIDIA GeForce GTX 1060)") && l.starts_with("Running on the processor"), "{l}");
        let (r, _) = fallback_line("nmffi.cuda: cuInit failed with CUDA error 100", None, Some("3"));
        assert_eq!(r, Reason::Hidden);
        assert_eq!(plain("Running on the processor: x. [nmffi.dynlib: nvcuda.dll not found]"), "Running on the processor: x.");
        assert_eq!(driver_text(13000), "13.0");
    }
}
