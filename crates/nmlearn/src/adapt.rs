//! Adaptations (PLAN 4c `adapt`; guide 10.1 to 10.3; Comprehensive Vision 2.3 and 9.5): the
//! three forms of change are kept apart. A parameter adaptation is a learning-weight unit
//! and is applied here through the registry by its Step Check verdict; a functional
//! adaptation (a reflex row) and a structural one (an RNAM change) are routed to the reflex
//! table and to RNAM, which live above this crate, so `apply` hands them back as typed
//! hand-offs. Nothing persists on Reject.
use crate::registry::Units;
use crate::stepspec::Verdict;
use crate::unit::{obj, s};
use nmjson::Value;
use nmtype::{Fault, FaultKind};

/// One proposed change.
#[derive(Clone, Debug, PartialEq)]
pub enum Adaptation {
    /// A learning-weight unit (by id) after its Step Check.
    Parameter { unit: String },
    /// A reflex table row (PLAN 4d item 2 shape).
    Functional { row: Value },
    /// An RNAM change (`nmnte::plasticity::Change` as JSON).
    Structural { change: Value },
}
impl Adaptation {
    /// Which mechanism applies it.
    #[must_use]
    pub const fn route(&self) -> &'static str {
        match self {
            Self::Parameter { .. } => "registry",
            Self::Functional { .. } => "reflexes",
            Self::Structural { .. } => "rnam",
        }
    }
}

/// What `apply` did.
#[derive(Clone, Debug, PartialEq)]
pub enum Applied {
    /// The unit's new status and whether it stays loaded.
    Unit { id: String, status: String, loaded: bool },
    /// Handed to the named mechanism with its payload (the caller applies it there).
    Handoff { route: &'static str, payload: Value },
}
impl Applied {
    #[must_use]
    pub fn to_value(&self) -> Value {
        match self {
            Self::Unit { id, status, loaded } => obj(vec![("unit", s(id)), ("status", s(status)), ("loaded", Value::Bool(*loaded))]),
            Self::Handoff { route, payload } => obj(vec![("route", s(route)), ("payload", payload.clone())]),
        }
    }
}

/// Apply `a` under `verdict`. Parameter: Accept marks the unit fitted (it stays loaded),
/// Candidate marks it candidate (unloaded until a later check), Reject marks it rejected and
/// unloaded. Functional and structural changes are handed off only on Accept or Candidate.
/// # Errors
/// An unknown unit, or a registry write failure.
pub fn apply(a: &Adaptation, verdict: Verdict, units: &mut Units) -> Result<Applied, Fault> {
    match a {
        Adaptation::Parameter { unit } => {
            let (status, loaded) = match verdict {
                Verdict::Accept => ("fitted", true),
                Verdict::Candidate => ("candidate", false),
                Verdict::Reject => ("rejected", false),
            };
            units.set_status(unit, status)?;
            if loaded {
                units.mark_loaded(unit)?;
            } else {
                units.mark_unloaded(unit)?;
            }
            Ok(Applied::Unit { id: unit.clone(), status: status.into(), loaded })
        }
        Adaptation::Functional { row } | Adaptation::Structural { change: row } => {
            if verdict == Verdict::Reject {
                return Err(Fault::new(
                    "nmlearn.adapt",
                    FaultKind::Policy,
                    format!("{} change rejected by its Step Check; nothing persists", a.route()),
                ));
            }
            Ok(Applied::Handoff { route: a.route(), payload: row.clone() })
        }
    }
}

/// Self-check of `adapt`: a rejected unit is unloaded and marked; an accepted one is fitted
/// and loaded; a rejected structural change is refused; an accepted one is handed to RNAM.
pub(crate) fn check() -> Result<(), String> {
    let home = nmtype::check_dir("nmlearn-adapt").map_err(|e| e.to_string())?;
    let mut u = Units::open(&home);
    u.register(crate::unit::UnitCard::new("lwu-x", "pop", "aa", "reranker", "n#1", 8, 1)).map_err(|e| e.to_string())?;
    let p = Adaptation::Parameter { unit: "lwu-x".into() };
    match apply(&p, Verdict::Reject, &mut u).map_err(|e| e.to_string())? {
        Applied::Unit { status, loaded, .. } if status == "rejected" && !loaded => {}
        other => return Err(format!("nmlearn.adapt: reject gave {other:?}")),
    }
    u.register(crate::unit::UnitCard::new("lwu-y", "pop", "aa", "reranker", "n#1", 8, 2)).map_err(|e| e.to_string())?;
    match apply(&Adaptation::Parameter { unit: "lwu-y".into() }, Verdict::Accept, &mut u).map_err(|e| e.to_string())? {
        Applied::Unit { status, loaded, .. } if status == "fitted" && loaded => {}
        other => return Err(format!("nmlearn.adapt: accept gave {other:?}")),
    }
    let st = Adaptation::Structural { change: Value::Null };
    if apply(&st, Verdict::Reject, &mut u).is_ok() || st.route() != "rnam" {
        return Err("nmlearn.adapt: a rejected structural change was applied".into());
    }
    if !matches!(apply(&st, Verdict::Accept, &mut u), Ok(Applied::Handoff { route: "rnam", .. })) {
        return Err("nmlearn.adapt: structural hand-off".into());
    }
    Ok(())
}
