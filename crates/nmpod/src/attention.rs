//! Demand-based attention (guide 4.1, 13; law 4): a unit's share of the task budget follows
//! the demand of its territory, `uncertainty * recency / cost`, with a floor so no
//! territory is starved.
use nmtype::{Budget, Region};

/// Demand of one claim or territory.
#[must_use]
pub fn demand(uncertainty: f32, recency: f32, cost: f32) -> f32 {
    (uncertainty.max(0.0) * recency.max(0.0)) / cost.max(1e-3)
}

/// Split a task budget across territories by demand (priority as cost proxy: larger
/// territories cost more). Every territory gets at least one unit and an equal floor share
/// of tokens; the rest follows demand.
#[must_use]
pub fn allocate(territories: &[Vec<Region>], budget: &Budget) -> Vec<Budget> {
    let n = territories.len().max(1);
    let demands: Vec<f32> = territories
        .iter()
        .map(|t| {
            let cost: f32 = t.iter().map(|r| r.priority).sum::<f32>().max(1.0);
            demand(1.0, 1.0, cost.sqrt())
        })
        .collect();
    let total: f32 = demands.iter().sum::<f32>().max(1e-6);
    let n32 = u32::try_from(n).unwrap_or(1);
    let floor = budget.tokens / (2 * n32);
    let rest = budget.tokens.saturating_sub(floor * n32);
    demands
        .iter()
        .map(|d| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss,
                reason = "shares of a u32 budget"
            )]
            let share = (rest as f32 * d / total) as u32;
            Budget { tokens: floor + share, ms: budget.ms, depth: budget.depth, units: 1 }
        })
        .collect()
}

/// Self-check of the `attention` sub-component: shares never exceed the budget and every
/// territory gets a floor.
pub(crate) fn check() -> Result<(), String> {
    let r = |p: f32| vec![Region { territory: 0, path: "x/".into(), start: 0, end: 0, priority: p }];
    let b = Budget { tokens: 4000, ms: 50_000, depth: 1, units: 16 };
    let shares = allocate(&[r(1.0), r(100.0), r(10.0)], &b);
    let sum: u32 = shares.iter().map(|s| s.tokens).sum();
    if sum > b.tokens || shares.iter().any(|s| s.tokens < 4000 / 6) || shares[0].tokens <= shares[1].tokens {
        return Err(format!("nmpod.attention: shares {:?}", shares.iter().map(|s| s.tokens).collect::<Vec<_>>()));
    }
    Ok(())
}
