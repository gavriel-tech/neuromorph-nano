//! Lossless read caches of the served flow (Stage 5 WP-C3 of the 2026-11-22 action plan).
//!
//! J3, the cross-call judge score cache: the close-call judge's final log-odds per piece
//! (`neural::judge_close`, after the judge-code read-out), kept in a bounded least-recently-used map
//! keyed by the question, the piece's text, the reranker's parameters key (its base cartridge and
//! compensation), the judge read-out in force, the plugged filter generation, and whether the
//! read-out applied to the batch. A batch is served from the cache only when every piece of it hits,
//! so a cached batch returns exactly the values its first read returned; a partly new batch is read
//! whole and stored. `NANO_JUDGE_CACHE=off` disables it (the cold arm of the paired check).
//!
//! The constant prompt starts (J1) are cached on the card by the engine (`nmengn::roles`,
//! `NANO_PREFIX_CACHE`, Antigravity's); content-free pair labels are held per process by
//! `drone_path::judge_pairs` (R2 for the pair reads). Nothing here touches the engine.
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use nmengn::model::Model;
use nmtype::Fault;

/// The most entries a cache holds (each a few dozen bytes); the least recently used leaves first.
pub const CAP: usize = 8192;

/// A bounded least-recently-used map from a 64-bit key to a value.
#[derive(Debug)]
pub struct Lru<V> {
    cap: usize,
    tick: u64,
    map: HashMap<u64, (V, u64)>,
}

impl<V: Clone> Lru<V> {
    #[must_use]
    pub fn new(cap: usize) -> Self {
        Self { cap: cap.max(1), tick: 0, map: HashMap::new() }
    }
    /// The value for `k`, marked as just used.
    pub fn get(&mut self, k: u64) -> Option<V> {
        self.tick += 1;
        let t = self.tick;
        self.map.get_mut(&k).map(|e| {
            e.1 = t;
            e.0.clone()
        })
    }
    /// Store `v` at `k`, dropping the least recently used entry when full.
    pub fn put(&mut self, k: u64, v: V) {
        self.tick += 1;
        if !self.map.contains_key(&k)
            && self.map.len() >= self.cap
            && let Some(old) = self.map.iter().min_by_key(|(_, e)| e.1).map(|(k, _)| *k)
        {
            self.map.remove(&old);
        }
        self.map.insert(k, (v, self.tick));
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// FNV-1a over the parts, each followed by a separator byte.
#[must_use]
pub fn key(parts: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        for b in p.bytes().chain(std::iter::once(0xff)) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

static JUDGE: Mutex<Option<Lru<f32>>> = Mutex::new(None);
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Whether the judge score cache serves (`NANO_JUDGE_CACHE=off` disables it).
#[must_use]
pub fn judge_cache_on() -> bool {
    !std::env::var("NANO_JUDGE_CACHE").is_ok_and(|v| v == "off")
}

/// The judge cache's hits and misses (pieces) since the process started.
#[must_use]
pub fn judge_stats() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What the judge's values depend on besides the question and the piece: the reranker's parameters
/// key, the judge read-out in force, the plugged filter generation, and whether the read-out applies
/// to this batch.
fn version(m: &Model, docs: &[String]) -> String {
    let readout = crate::adapter::readout().map_or_else(|| "none".to_owned(), |r| format!("{}:{}", r.key, r.values));
    format!("{}|{readout}|{}|{}", crate::neural::parameters_key(m), nmengn::filter::generation(), crate::adapter::judge_applies(m, docs))
}

/// `neural::judge_close` through the J3 cache: every piece a hit returns the cached values; otherwise
/// the whole batch is read and each piece's value stored.
/// # Errors
/// As `neural::judge_close`.
pub fn judge_close(m: &Model, question: &str, docs: &[String]) -> Result<Vec<f32>, Fault> {
    if !judge_cache_on() || docs.is_empty() {
        return crate::neural::judge_close(m, question, docs);
    }
    let ver = version(m, docs);
    let keys: Vec<u64> = docs.iter().map(|d| key(&["judge-close", &ver, question, d])).collect();
    if let Ok(mut g) = JUDGE.lock() {
        let lru = g.get_or_insert_with(|| Lru::new(CAP));
        let got: Vec<Option<f32>> = keys.iter().map(|k| lru.get(*k)).collect();
        if got.iter().all(Option::is_some) {
            HITS.fetch_add(docs.len() as u64, Ordering::Relaxed);
            return Ok(got.into_iter().flatten().collect());
        }
    }
    MISSES.fetch_add(docs.len() as u64, Ordering::Relaxed);
    let z = crate::neural::judge_close(m, question, docs)?;
    if z.len() == keys.len()
        && let Ok(mut g) = JUDGE.lock()
    {
        let lru = g.get_or_insert_with(|| Lru::new(CAP));
        for (k, v) in keys.iter().zip(&z) {
            lru.put(*k, *v);
        }
    }
    Ok(z)
}

/// Section check: the cache stays bounded, keeps the most recently used entries, and keys differ
/// when any part differs.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let mut l: Lru<u32> = Lru::new(2);
    l.put(1, 10);
    l.put(2, 20);
    let _ = l.get(1);
    l.put(3, 30);
    if l.len() != 2 || l.get(2).is_some() || l.get(1) != Some(10) || l.get(3) != Some(30) {
        return Err("nmapi.cache_reads: least recently used order".into());
    }
    if key(&["a", "b"]) == key(&["ab", ""]) || key(&["q", "d1"]) == key(&["q", "d2"]) {
        return Err("nmapi.cache_reads: keys".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn lru_and_keys() {
        assert_eq!(super::check(), Ok(()));
        let mut l: super::Lru<f32> = super::Lru::new(super::CAP);
        for i in 0..(super::CAP as u64 + 100) {
            l.put(i, 1.0);
        }
        assert_eq!(l.len(), super::CAP, "bounded");
        assert!(l.get(0).is_none() && l.get(super::CAP as u64 + 99).is_some(), "oldest dropped, newest kept");
    }
}
