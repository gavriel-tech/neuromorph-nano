//! Cluster lifecycle (guide 9.3 responsibility 12; PLAN 4c): Registered, Provisioned, Bound,
//! Released, Retired, with every transition checked. A retired cluster may be provisioned
//! again (its identity stays registered).
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Registered,
    Provisioned,
    Bound,
    Released,
    Retired,
}

/// Allowed transitions.
#[must_use]
pub fn allowed(from: State, to: State) -> bool {
    use State::{Bound, Provisioned, Registered, Released, Retired};
    matches!(
        (from, to),
        (Registered | Retired, Provisioned)
            | (Provisioned | Released | Bound, Bound)
            | (Bound, Released)
            | (Released | Registered | Provisioned, Retired)
    )
}

/// The state of every registered cluster.
#[derive(Debug, Default)]
pub struct Tracker {
    states: Mutex<BTreeMap<String, State>>,
}
impl Tracker {
    pub fn register(&self, id: &str) {
        if let Ok(mut s) = self.states.lock() {
            s.entry(id.to_owned()).or_insert(State::Registered);
        }
    }
    /// Move a cluster to `to`.
    /// # Errors
    /// An unregistered cluster or a transition the table does not allow.
    pub fn to(&self, id: &str, to: State) -> Result<(), Fault> {
        let mut s = self.states.lock().map_err(|_| Fault::new("nmnpcs.lifecycle", FaultKind::Execution, "lock"))?;
        let from = *s.get(id).ok_or_else(|| Fault::new("nmnpcs.lifecycle", FaultKind::Config, format!("{id} is not registered")))?;
        if !allowed(from, to) {
            return Err(Fault::new("nmnpcs.lifecycle", FaultKind::Policy, format!("{id}: {from:?} to {to:?} is not allowed")));
        }
        s.insert(id.to_owned(), to);
        Ok(())
    }
    #[must_use]
    pub fn state(&self, id: &str) -> Option<State> {
        self.states.lock().ok().and_then(|s| s.get(id).copied())
    }
    /// How many clusters are in each state.
    #[must_use]
    pub fn counts(&self) -> BTreeMap<String, usize> {
        let mut m = BTreeMap::new();
        if let Ok(s) = self.states.lock() {
            for v in s.values() {
                *m.entry(format!("{v:?}")).or_default() += 1;
            }
        }
        m
    }
}

/// Self-check of `lifecycle`: the normal path is allowed, binding a retired cluster without
/// provisioning it again is refused.
pub(crate) fn check() -> Result<(), String> {
    let t = Tracker::default();
    t.register("c");
    for s in [State::Provisioned, State::Bound, State::Released, State::Retired] {
        t.to("c", s).map_err(|e| e.to_string())?;
    }
    if t.to("c", State::Bound).is_ok() {
        return Err("nmnpcs.lifecycle: a retired cluster was bound without provisioning".into());
    }
    Ok(())
}
