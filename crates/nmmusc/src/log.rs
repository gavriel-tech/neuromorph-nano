//! Behaviour log (RAM): one append-only symbol sequence per question or task, with the
//! question's outcome measure attached when it is known. Nothing here touches disk.
use crate::symbol::{BehaviouralSymbol, Codebook};
use crate::{Fault, Result};
use std::collections::HashMap;

/// One question's behaviour: the symbols in the order the messages were sent.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    pub id: String,
    pub symbols: Vec<u32>,
    /// Outcome measure of the question (higher is better), when known.
    pub outcome: Option<f64>,
}

/// The log: codebook plus traces. Append-only: no call removes or rewrites a symbol.
#[derive(Clone, Debug, Default)]
pub struct BehaviourLog {
    pub codebook: Codebook,
    traces: Vec<Trace>,
    by_id: HashMap<String, usize>,
}
impl BehaviourLog {
    fn slot(&mut self, id: &str) -> usize {
        if let Some(&i) = self.by_id.get(id) {
            return i;
        }
        self.traces.push(Trace { id: id.to_owned(), symbols: Vec::new(), outcome: None });
        let i = self.traces.len() - 1;
        self.by_id.insert(id.to_owned(), i);
        i
    }
    /// Append one behavioural message to question `id`.
    /// # Errors
    /// `nmmusc-sensory-refused` for a sensory item (nothing is appended).
    pub fn record(&mut self, id: &str, m: &impl BehaviouralSymbol) -> Result<u32> {
        let c = self.codebook.code(m)?;
        let i = self.slot(id);
        if let Some(t) = self.traces.get_mut(i) {
            t.symbols.push(c);
        }
        Ok(c)
    }
    /// Attach the outcome of question `id` (the last value given wins).
    /// # Errors
    /// `nmmusc-unknown-trace` or a non-finite value.
    pub fn set_outcome(&mut self, id: &str, outcome: f64) -> Result<()> {
        if !outcome.is_finite() {
            return Err(Fault::new("nmmusc-bad-outcome", "outcome must be finite"));
        }
        let i = *self.by_id.get(id).ok_or_else(|| Fault::new("nmmusc-unknown-trace", format!("no trace {id:?}")))?;
        if let Some(t) = self.traces.get_mut(i) {
            t.outcome = Some(outcome);
        }
        Ok(())
    }
    #[must_use]
    pub fn traces(&self) -> &[Trace] {
        &self.traces
    }
}
