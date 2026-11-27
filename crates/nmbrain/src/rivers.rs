//! Context Rivers mailbox (guide 6.1; PLAN 4c): the only path between units. Packets are
//! routed by kind into queues; components subscribe by kind. Nothing is dropped for capacity:
//! a queue keeps every packet, in arrival order, until its subscriber takes it, and the
//! subscriber's own room decides what it uses now (the reduce step takes every packet and the
//! reducer holds what its token room cannot take in the question's queue, marked low). A new
//! substrate version invalidates older packets except verified ones: they describe material
//! that changed and are read again.
use nmtype::{ComponentId, Packet, Status};
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Default)]
pub struct Mailbox {
    queues: BTreeMap<String, VecDeque<Packet>>,
    subs: Vec<(String, ComponentId)>,
    pub version: u64,
    /// Packets released because the material they were read from changed.
    pub stale: u64,
}
impl Mailbox {
    /// Route a packet into its kind's queue; every packet is kept.
    pub fn publish(&mut self, p: Packet) {
        self.queues.entry(p.kind().to_owned()).or_default().push_back(p);
    }
    pub fn subscribe(&mut self, kind: &str, who: &str) {
        self.subs.push((kind.to_owned(), who.to_owned()));
    }
    /// Take every packet of a kind (the subscriber drains its queue).
    pub fn take(&mut self, kind: &str) -> Vec<Packet> {
        self.queues.get_mut(kind).map(|q| q.drain(..).collect()).unwrap_or_default()
    }
    /// Take everything, all kinds.
    pub fn take_all(&mut self) -> Vec<Packet> {
        let kinds: Vec<String> = self.queues.keys().cloned().collect();
        kinds.iter().flat_map(|k| self.take(k)).collect()
    }
    /// A new substrate version: drop packets read at older versions, except verified ones.
    pub fn invalidate(&mut self, version: u64) {
        self.version = version;
        for q in self.queues.values_mut() {
            let before = q.len();
            q.retain(|p| p.provenance().version >= version || p.get_status() == Status::Verified);
            self.stale += (before - q.len()) as u64;
        }
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.queues.values().map(VecDeque::len).sum()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Longest single queue.
    #[must_use]
    pub fn deepest(&self) -> usize {
        self.queues.values().map(VecDeque::len).max().unwrap_or(0)
    }
}

/// Self-check of the `rivers` sub-component: however many packets arrive, every one is kept in
/// arrival order and the question with them; invalidation keeps verified packets.
pub(crate) fn check() -> Result<(), String> {
    use nmtype::Provenance;
    let mut m = Mailbox::default();
    m.publish(Packet::new("question", "q", "keep me", Provenance::new("unit:x", "x", 1, 0)).status(Status::Unknown));
    for i in 0..600_u64 {
        m.publish(Packet::new("grep", &format!("k{i}"), "t", Provenance::new("a.rs:1", "x", 1, 0)).status(Status::Hypothesized));
    }
    let grep = m.take("grep");
    let ends = (grep.first().map(Packet::key), grep.last().map(Packet::key));
    if grep.len() != 600 || ends != (Some("k0"), Some("k599")) || m.take("question").len() != 1 {
        return Err("nmbrain.rivers: a packet was dropped or reordered".into());
    }
    m.publish(Packet::new("grep", "v", "t", Provenance::new("a.rs:1", "x", 1, 0)).status(Status::Verified));
    m.invalidate(2);
    let left = m.take_all();
    if left.len() != 1 || left.first().map(Packet::get_status) != Some(Status::Verified) {
        return Err("nmbrain.rivers: invalidation".into());
    }
    Ok(())
}
