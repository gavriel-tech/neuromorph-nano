//! The relevance gate and checked writer (guide 7; PLAN 4c, from `relevance-gate` and
//! `checked-writer`): a write needs enough confidence and evidence, and a packet whose provenance
//! is memory cannot be stored again as observed (the provenance cap: memory never launders itself
//! into fact). Refused writes are logged. No count of writes is refused.
use crate::store::{self, obj, s, u};
use nmjson::Value;
use nmtype::{Packet, Status};
use std::path::PathBuf;

/// Writer policy from `designs/memory.json` (`writer_policy`).
#[derive(Clone, Debug, PartialEq)]
pub struct WriterPolicy {
    pub min_confidence: f32,
    pub require_evidence: bool,
}
impl Default for WriterPolicy {
    fn default() -> Self {
        Self { min_confidence: 0.5, require_evidence: true }
    }
}
impl WriterPolicy {
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let p = v.get("writer_policy").ok();
        let d = Self::default();
        let f = |k: &str| p.and_then(|p| p.get(k).ok()).and_then(|x| if let Value::Number(n) = x { n.parse().ok() } else { None });
        Self {
            min_confidence: f("min_confidence").unwrap_or(d.min_confidence),
            require_evidence: p.and_then(|p| p.get("require_evidence").ok()).map_or(d.require_evidence, |x| matches!(x, Value::Bool(true))),
        }
    }
}

/// The gate with its refusal log.
#[derive(Debug)]
pub struct Gate {
    pub policy: WriterPolicy,
    log: PathBuf,
    pub refused: usize,
    /// Locked test questions' wording, never saved (2026-11-12, memory revision 2 section 8).
    locked: Vec<String>,
}
impl Gate {
    #[must_use]
    pub fn new(dir: &std::path::Path, policy: WriterPolicy) -> Self {
        Self { policy, log: dir.join("refused.jsonl"), refused: 0, locked: crate::classes::locked_questions(&nmtype::nano_root()) }
    }
    /// Refuse `text` when it must never be saved (a credential, key, token, e-mail address or
    /// locked test wording), logging the refusal under the hash of `key` like any other.
    /// # Errors
    /// The never-save reason, as `never save: <why>`.
    pub fn screen(&mut self, key: &str, text: &str) -> Result<(), String> {
        let Some(why) = crate::classes::never_save(text, &self.locked) else { return Ok(()) };
        let r = format!("never save: {why}");
        self.refused += 1;
        let _ = store::append(&self.log, &refusal(key, &r));
        Err(r)
    }
    /// Admit or refuse one packet. A refusal is appended to `refused.jsonl` with its reason.
    pub fn admit(&mut self, p: &Packet) -> Result<(), String> {
        // Never save (2026-11-12): credentials, keys, tokens, e-mail addresses, locked test wording.
        let secret = std::iter::once(p.text())
            .chain(p.get_evidence().iter().map(String::as_str))
            .find_map(|t| crate::classes::never_save(t, &self.locked));
        let never = secret.map(|why| format!("never save: {why}"));
        let reason = if let Some(n) = never.as_deref() {
            Some(n)
        } else if p.provenance().source.starts_with("memory:") && matches!(p.get_status(), Status::Observed | Status::Verified) {
            Some("provenance cap: a memory packet cannot be stored as observed or verified")
        } else if self.policy.require_evidence && p.get_evidence().is_empty() {
            Some("no evidence")
        } else if p.get_confidence() < self.policy.min_confidence {
            Some("below minimum confidence")
        } else {
            None
        };
        match reason {
            None => Ok(()),
            Some(r) => {
                self.refused += 1;
                // The key of an offered note is its text, and its source is caller-supplied, so
                // neither is written: a refused item leaves only a hash of its key, the reason
                // and the time on disk.
                let _ = store::append(&self.log, &refusal(p.key(), r));
                Err(r.to_owned())
            }
        }
    }
}

/// The refusal log row for an item with key `key`: the time, the SHA-256 of the key and the
/// reason. Nothing of the item's text is kept.
fn refusal(key: &str, reason: &str) -> Value {
    obj(vec![("at", u(nmtype::now_ms())), ("key_sha256", s(&nmhash::sha256_bytes(key.as_bytes()))), ("reason", s(reason))])
}

/// Self-check of the `gate` sub-component: an evidenced packet passes; an uncited one, a
/// low-confidence one and a memory packet claiming observed status are refused and logged.
pub(crate) fn check() -> Result<(), String> {
    use nmtype::Provenance;
    let dir = crate::scratch_dir("gate")?;
    let mut g = Gate::new(&dir, WriterPolicy::default());
    let ok = Packet::new("pack", "k", "t", Provenance::new("a.rs:1", "x", 1, 0))
        .status(Status::Observed)
        .confidence(0.9)
        .evidence(vec!["a.rs:1".into()]);
    g.admit(&ok).map_err(|e| format!("nmmem.gate: good packet refused: {e}"))?;
    let laundered = Packet::new("pack", "k", "t", Provenance::new("memory:abc", "x", 1, 0))
        .status(Status::Observed)
        .confidence(0.9)
        .evidence(vec!["memory:abc".into()]);
    let secret = Packet::new("pack", "k", "the key is sk-ABCdef1234567890abcdefGHIJ", Provenance::new("a.rs:1", "x", 1, 0))
        .status(Status::Observed)
        .confidence(0.9)
        .evidence(vec!["a.rs:1".into()]);
    let bad = [
        ok.clone().confidence(0.1),
        Packet::new("pack", "k", "t", Provenance::new("a.rs:1", "x", 1, 0)).confidence(0.9),
        laundered,
        secret,
    ];
    for p in &bad {
        if g.admit(p).is_ok() {
            return Err("nmmem.gate: a packet that should be refused passed".into());
        }
    }
    // Many writes in one task are each judged on their own: none is refused for the count.
    for _ in 0..40 {
        g.admit(&ok).map_err(|e| format!("nmmem.gate: a write refused for its count: {e}"))?;
    }
    if g.refused != 4 || store::lines(&dir.join("refused.jsonl")).len() != 4 {
        return Err("nmmem.gate: refusal log".into());
    }
    // A refused note's text never reaches the log: only the key's hash, the reason and the time.
    let note = Packet::new(
        "memory",
        "note:the key is sk-ABCdef1234567890abcdefGHIJ",
        "the key is sk-ABCdef1234567890abcdefGHIJ",
        Provenance::new("unit:claude-code", "x", 0, 0),
    )
    .status(Status::Observed)
    .confidence(1.0)
    .evidence(vec!["unit:claude-code".into()]);
    if g.admit(&note).is_ok() {
        return Err("nmmem.gate: a note holding a secret passed".into());
    }
    let log = std::fs::read_to_string(dir.join("refused.jsonl")).map_err(|e| format!("nmmem.gate: refusal log: {e}"))?;
    let rows = store::lines(&dir.join("refused.jsonl"));
    let fields_ok = rows.iter().all(|v| matches!(v, Value::Object(m) if m.len() == 3 && m.contains_key("at") && m.contains_key("key_sha256") && m.contains_key("reason")));
    if log.contains("ABCdef") || log.contains("the key is") || log.contains("unit:claude-code") || !fields_ok {
        return Err("nmmem.gate: the refusal log holds item text or extra fields".into());
    }
    Ok(())
}
