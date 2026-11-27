//! Memory tiers and their room (PLAN 4c, from `memory/memprt`): T1 working (RAM), T2 episodic
//! (file), T3 structural (files). Usage is measured. No tier has a fixed byte budget: working
//! memory's room is the physical memory available now, measured at run time by the caller; the
//! episodes and the structural summaries are kept and dropped by their own lifetimes and never
//! removed for size, so they have no budget.
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TierUse {
    pub tier: &'static str,
    pub bytes: u64,
    /// The room the hardware gives this tier now, when it has one and it could be measured.
    pub budget: Option<u64>,
}
impl TierUse {
    #[must_use]
    pub fn over(&self) -> bool {
        self.budget.is_some_and(|b| self.bytes > b)
    }
}

fn size(path: &Path) -> u64 {
    if path.is_dir() {
        std::fs::read_dir(path).map(|rd| rd.flatten().map(|e| size(&e.path())).sum()).unwrap_or(0)
    } else {
        std::fs::metadata(path).map_or(0, |m| m.len())
    }
}

/// Current use of the three tiers; `ram_available` is the physical memory available now (bytes),
/// `None` when it could not be measured.
#[must_use]
pub fn usage(home: &Path, working_bytes: usize, ram_available: Option<u64>) -> Vec<TierUse> {
    vec![
        TierUse { tier: "T1", bytes: working_bytes as u64, budget: ram_available },
        TierUse { tier: "T2", bytes: size(&home.join("episodes.jsonl")), budget: None },
        TierUse { tier: "T3", bytes: size(&home.join("structural")), budget: None },
    ]
}

/// Self-check of the `tiers` sub-component: working memory's room is the measured RAM, a tier past
/// it reports over, and the file tiers have no budget.
pub(crate) fn check() -> Result<(), String> {
    let home = crate::scratch_dir("tiers")?;
    let u = usage(&home, 11, Some(10));
    if u.first().is_none_or(|t| !t.over() || t.budget != Some(10)) || u.iter().skip(1).any(|t| t.over() || t.budget.is_some()) {
        return Err("nmmem.tiers: room or over flag".into());
    }
    if usage(&home, 11, None).iter().any(TierUse::over) {
        return Err("nmmem.tiers: over without a measured room".into());
    }
    Ok(())
}
