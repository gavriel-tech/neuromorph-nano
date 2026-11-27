//! Component Registry (guide 11.1 item 1): every card with its lifecycle state. If a
//! component is not in the registry, it does not run. Shape from emma-labs `fwcore/regist`,
//! written fresh over `nmtype::Card`.
use nmtype::{Card, Fault, FaultKind, PartRef, Resolve};
use std::collections::BTreeMap;
use std::path::Path;

/// Lifecycle state of a registered card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardState {
    Candidate,
    Promoted,
    Rejected,
    Archived,
}
impl CardState {
    /// Candidates and promoted cards may run; rejected and archived ones may not.
    #[must_use]
    pub const fn runnable(self) -> bool {
        matches!(self, Self::Candidate | Self::Promoted)
    }
}

fn fault(kind: FaultKind, detail: impl Into<String>) -> Fault {
    Fault::new("nmnte.registry", kind, detail)
}

/// Cards by id, each with its state.
#[derive(Clone, Debug, Default)]
pub struct Registry {
    cards: BTreeMap<String, (Card, CardState)>,
}
impl Registry {
    /// Load every `<dir>/*/part.json` (hash-verified) as promoted.
    /// # Errors
    /// An unreadable folder, or any card that fails to load or verify.
    pub fn load(dir: &Path) -> Result<Self, Fault> {
        let entries = std::fs::read_dir(dir).map_err(|e| fault(FaultKind::Resource, format!("{}: {e}", dir.display())))?;
        let mut paths: Vec<_> = entries.flatten().map(|e| e.path().join("part.json")).filter(|p| p.is_file()).collect();
        paths.sort();
        let mut reg = Self::default();
        for p in paths {
            reg.register(Card::load(&p)?, CardState::Promoted);
        }
        Ok(reg)
    }
    /// Add or replace a card with a state.
    pub fn register(&mut self, card: Card, state: CardState) {
        self.cards.insert(card.descriptor.id.clone(), (card, state));
    }
    /// Change the state of a registered card.
    /// # Errors
    /// Unknown id.
    pub fn set_state(&mut self, id: &str, state: CardState) -> Result<(), Fault> {
        self.cards.get_mut(id).map(|slot| slot.1 = state).ok_or_else(|| fault(FaultKind::Config, format!("no card {id}")))
    }
    #[must_use]
    pub fn state(&self, id: &str) -> Option<CardState> {
        self.cards.get(id).map(|c| c.1)
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.cards.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }
    /// Does a node id resolve to a runnable card? `crate#inst` names the crate card itself;
    /// `crate.module#inst` must be listed among that card's sub-components.
    /// # Errors
    /// Unknown card, a state that may not run, or a sub-component the card does not list.
    pub fn resolve_node(&self, node_id: &str) -> Result<&Card, Fault> {
        let part = node_id.split('#').next().unwrap_or(node_id);
        let krate = part.split('.').next().unwrap_or(part);
        let (card, state) =
            self.cards.get(krate).ok_or_else(|| fault(FaultKind::Integrity, format!("{node_id}: no card {krate} in the registry")))?;
        if !state.runnable() {
            return Err(fault(FaultKind::Policy, format!("{node_id}: card {krate} is {state:?}")));
        }
        if part != krate && !card.subcomponents.iter().any(|s| s.id == part) {
            return Err(fault(FaultKind::Integrity, format!("{node_id}: card {krate} lists no sub-component {part}")));
        }
        Ok(card)
    }
}
impl Resolve for Registry {
    fn resolve(&self, r: &PartRef) -> Result<Card, Fault> {
        let (card, state) = self.cards.get(&r.id).ok_or_else(|| fault(FaultKind::Integrity, format!("no card {}", r.id)))?;
        if card.descriptor.card_hash != r.hash || card.descriptor.version != r.version {
            return Err(fault(FaultKind::Integrity, format!("{r}: registry holds {}", card.part_ref())));
        }
        if !state.runnable() {
            return Err(fault(FaultKind::Policy, format!("{r}: card is {state:?}")));
        }
        Ok(card.clone())
    }
}

/// Self-check of the `registry` sub-component: register, resolve by ref and by node id,
/// refuse a rejected card and a sub-component the card does not list.
pub(crate) fn check() -> Result<(), String> {
    let raw = br#"{"id":"nmx","kind":"tool","version":1,"parent":null,"ports_in":[],"ports_out":[],"deps":[],"config_schema":{},"config_default":{},"subcomponents":[{"id":"nmx.a","version":1,"hash":""}],"lineage":[],"source":"new","hash":""}"#;
    let v = nmjson::parse(raw).map_err(|e| e.to_string())?;
    let mut card = Card::from_value(&v).map_err(|e| e.to_string())?;
    card.descriptor.card_hash = nmtype::card_hash(&v);
    let mut reg = Registry::default();
    reg.register(card.clone(), CardState::Candidate);
    reg.resolve(&card.part_ref()).map_err(|e| e.to_string())?;
    reg.resolve_node("nmx.a#3").map_err(|e| e.to_string())?;
    reg.resolve_node("nmx#0").map_err(|e| e.to_string())?;
    if reg.resolve_node("nmx.b#0").is_ok() {
        return Err("nmnte.registry: unlisted sub-component resolved".into());
    }
    reg.set_state("nmx", CardState::Rejected).map_err(|e| e.to_string())?;
    if reg.resolve(&card.part_ref()).is_ok() {
        return Err("nmnte.registry: rejected card resolved".into());
    }
    Ok(())
}
