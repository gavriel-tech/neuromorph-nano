//! Development (guide 11, 11.2; PLAN 4c): repeated findings become anatomy proposals. A
//! territory that comes back empty-handed twice asks for one more tentacle; a tentacle idle
//! for the configured number of tasks may be removed. Proposals are descriptions only; RNAM
//! runs them through the lifecycle and the Integrity Validator, or refuses them.
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Proposal {
    AddTentacle { reason: String },
    RemoveTentacle { id: String, reason: String },
}

#[derive(Debug, Default)]
pub struct Development {
    starved: u32,
    idle: BTreeMap<String, u32>,
    pub idle_limit: u32,
}
impl Development {
    #[must_use]
    pub fn new(idle_limit: u32) -> Self {
        Self { starved: 0, idle: BTreeMap::new(), idle_limit: idle_limit.max(1) }
    }
    /// One task: tentacles that produced packets, tentacles that produced none, and whether
    /// the task ended withheld for lack of evidence.
    pub fn observe(&mut self, busy: &[String], idle: &[String], starved: bool) -> Vec<Proposal> {
        let mut out = Vec::new();
        self.starved = if starved { self.starved + 1 } else { 0 };
        if self.starved == 2 {
            out.push(Proposal::AddTentacle { reason: "two tasks in a row withheld for lack of evidence".into() });
            self.starved = 0;
        }
        for id in busy {
            self.idle.remove(id);
        }
        for id in idle {
            let n = self.idle.entry(id.clone()).or_default();
            *n += 1;
            if *n == self.idle_limit {
                out.push(Proposal::RemoveTentacle { id: id.clone(), reason: format!("no packets in {} tasks", self.idle_limit) });
            }
        }
        out
    }
}

/// Self-check of the `development` sub-component: two starved tasks propose a tentacle; an
/// idle tentacle is proposed for removal at the limit.
pub(crate) fn check() -> Result<(), String> {
    let mut d = Development::new(2);
    let t = vec!["t#7".to_owned()];
    let first = d.observe(&[], &t, true);
    let second = d.observe(&[], &t, true);
    if !first.is_empty() || second.len() != 2 || !matches!(second.first(), Some(Proposal::AddTentacle { .. })) {
        return Err(format!("nmbrain.development: proposals {second:?}"));
    }
    Ok(())
}
