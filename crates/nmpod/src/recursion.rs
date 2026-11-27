//! Recursive drone deployment (guide 4.2, 12; PLAN 4c): a unit may ask for one subordinate
//! drone as part of its outcome. The request is a `question` packet to the Core Brain, which
//! decides; there is no direct spawn. Termination: depth at the maximum, no budget left, or
//! two rounds with no new claim keys.
use nmtype::{Budget, Packet, Provenance, Status, Task, TaskKind};

/// Should this unit ask for a subordinate drone, and with what task?
#[must_use]
pub fn should_spawn(task: &Task, depth: u8, max_depth: u8, anomaly: bool, unknown_keys: &[String], rounds_without_new: u8) -> Option<Task> {
    if depth >= max_depth || task.budget.depth == 0 || rounds_without_new >= 2 || (!anomaly && unknown_keys.is_empty()) {
        return None;
    }
    let scope = unknown_keys.first().cloned().unwrap_or_else(|| task.scope.clone());
    Some(Task {
        id: task.id * 100 + u64::from(depth) + 1,
        objective: task.objective.clone(),
        budget: Budget { depth: task.budget.depth - 1, units: 1, ..task.budget },
        scope,
        parent: Some(task.id),
        kind: TaskKind::Verify,
    })
}

/// The request packet that carries a subordinate task to the Core Brain.
#[must_use]
pub fn request(from: &str, sub: &Task, version: u64) -> Packet {
    Packet::new(
        "question",
        &format!("subordinate:{from}"),
        &format!("{from} asks for a subordinate drone on {} at depth budget {}", sub.scope, sub.budget.depth),
        Provenance::new(&format!("unit:{from}"), from, version, nmtype::now_ms()),
    )
    .status(Status::Unknown)
    .scope(&sub.scope)
    .relate("parent_task", sub.parent.unwrap_or(0))
}

/// Self-check of the `recursion` sub-component: spawn on anomaly below max depth, stop at
/// max depth and after two empty rounds.
pub(crate) fn check() -> Result<(), String> {
    let t = Task {
        id: 7,
        objective: "q".into(),
        budget: Budget { tokens: 100, ms: 1000, depth: 1, units: 8 },
        scope: "a/".into(),
        parent: None,
        kind: TaskKind::Context,
    };
    let sub = should_spawn(&t, 0, 1, true, &[], 0).ok_or("nmpod.recursion: no spawn on anomaly")?;
    if sub.parent != Some(7) || sub.budget.depth != 0 {
        return Err("nmpod.recursion: subordinate task fields".into());
    }
    if should_spawn(&t, 1, 1, true, &[], 0).is_some() || should_spawn(&t, 0, 1, true, &[], 2).is_some() {
        return Err("nmpod.recursion: termination rule ignored".into());
    }
    if request("nmpod.unit#t0", &sub, 1).kind() != "question" {
        return Err("nmpod.recursion: request is not a question packet".into());
    }
    Ok(())
}
