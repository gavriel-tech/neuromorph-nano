//! Memory areas as the served engine reports them (2026-11-27; Luis Jake Gabriel III: "The user
//! should control how much to allocate memory wise to where"). The process budget is divided
//! among three areas (`nmengn::limits::area_budget`), each `auto` or a size the user sets (on
//! `auto` 25, 15 and 45 percent; 15 percent stays with the engine's own working memory):
//!
//! - `content`: the substrate's content (`substrate`), the drones' intake (`drone_intake`), the
//!   relevance keys (`relevance_keys`) and the dense lane's vectors (`dense_vectors`). The
//!   intake, the relevance-key intake and the dense lane run in bounded steps and stop when the
//!   area is full; the pieces not reached wait until the area is raised;
//! - `caches`: the kept encoded-chunk cache (`encoded_chunks`), bounded by the area;
//! - `weights`: the host copies of the MicroModels' weights (`micromodels`).
//!
//! Each holder reports its bytes as it changes; `refresh` reads the holders that do not report
//! by themselves and returns the block `nm_status`, `nm_live` and `GET /api/live` carry.
use crate::forage::Organism;
use nmjson::Value;
use std::collections::BTreeMap;

fn n(x: u64) -> Value {
    Value::Number(x.to_string())
}

/// Reads the holders that do not report by themselves (the MicroModels' weights, the
/// encoded-chunk cache, the relevance keys) and returns `areas` (each area's setting, budget,
/// use, holders and state) and `intake` (where the drones' intake stands).
#[must_use]
pub fn refresh(o: &Organism) -> Value {
    let mut weights = 0_u64;
    for (_, m) in crate::neural::bound_models() {
        weights = weights.saturating_add(m.pop.prov.totals().0);
    }
    if let Some(ram) = crate::writer_idle::resident_writer_view(|w| w.pop.prov.totals().0) {
        weights = weights.saturating_add(ram);
    }
    nmengn::limits::set_area_used("weights", "micromodels", weights);
    if let Some(p) = o.tentacle_pass.read().ok().and_then(|p| p.clone()) {
        nmengn::limits::set_area_used("caches", "encoded_chunks", p.held_bytes() as u64);
    }
    let whole = o.whole.read().ok().and_then(|w| w.clone());
    if let Some(k) = whole.as_ref().and_then(|w| w.relevance_keys()) {
        nmengn::limits::set_area_used("content", "relevance_keys", k.held().0 as u64);
    }
    let areas = nmengn::limits::areas_value();
    let intake = match whole.as_ref().and_then(|w| w.keep_state().map(|s| (s, w.warming()))) {
        Some((s, running)) => {
            let note = if s.full {
                "stopped: the content memory area is full; the pieces waiting are read as their paths until limits.memory.content_mb is raised"
            } else if s.complete() {
                "every piece taken in"
            } else if running {
                "taking pieces in at idle, in bounded steps; calls read what is taken in"
            } else {
                "pieces waiting; the next call starts the background pass"
            };
            Value::Object(
                [
                    ("pieces_taken", n(s.taken as u64)),
                    ("pieces_waiting", n(s.pending as u64)),
                    ("items_taken", n(s.items_taken as u64)),
                    ("items_waiting", n(s.items_pending as u64)),
                    ("full", Value::Bool(s.full)),
                    ("background_pass", Value::Bool(running)),
                    ("note", Value::String(note.to_owned())),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect::<BTreeMap<_, _>>(),
            )
        }
        None => Value::String("the drones' intake has not opened".to_owned()),
    };
    Value::Object([("areas".to_owned(), areas), ("intake".to_owned(), intake)].into_iter().collect())
}

/// A mark of the drones' intake for the answer cache: empty once every piece is in, else the
/// pieces and items taken in so far, so a reply made while the intake goes on is not served again
/// after more of the material is in.
#[must_use]
pub fn intake_mark(o: &Organism) -> String {
    match o.whole.read().ok().and_then(|w| w.as_ref().and_then(|w| w.keep_state())) {
        Some(s) if !s.complete() => format!("intake {} {}", s.taken, s.items_taken),
        _ => String::new(),
    }
}

/// `refresh` on the installed organism; the areas alone before one is installed.
#[must_use]
pub fn current() -> Value {
    match crate::tools::installed() {
        Some(o) => refresh(o),
        None => Value::Object(
            [("areas".to_owned(), nmengn::limits::areas_value()), ("intake".to_owned(), Value::String("no project is open in this process".to_owned()))]
                .into_iter()
                .collect(),
        ),
    }
}
