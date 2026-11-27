//! Base cartridges and their loading (2026-11-12; research file, section "Cartridges: the
//! distinctions, the mapping strategy and the loading strategy in one place", build steps I to M).
//!
//! A base cartridge is a frozen, quantized extract of the released parameters one job needs,
//! named by a manifest of packets. A packet is one parameter cluster (a tensor or a row or column
//! range of it) at one precision, identified by the hash of its content, so two cartridges that
//! need the same content name the same packet.
//!
//! - I, manifest and label: population and its base hash, atlas version, job, layer span, mask
//!   hash, closeness result, date, and the packets. Text form round-trips exactly.
//! - J, shared packets: the resident ledger holds each packet once, by content hash; its bytes are
//!   the sum over distinct hashes, whatever the number of cartridges using them.
//! - K, switch cost: moving from one job to another loads only the packets the next one needs
//!   that are not already resident.
//! - M, fallback: a cartridge whose label does not match the weights it would run on (population
//!   hash or atlas version) is refused, and the full model's manifest is loaded instead.
//! - L, prefetch, needs a second device stream and lives with the engine; here the ledger gives
//!   it the missing packets to move.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::{BTreeMap, BTreeSet};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmnpcs.cartridge", FaultKind::Config, m)
}

/// Schema of a manifest file.
pub const SCHEMA: &str = "nm-cartridge/1";

/// One packet: a cluster at a precision, by content.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PacketRef {
    /// Content hash (SHA-256 of the cluster's bytes, its region and precision).
    pub hash: String,
    pub cluster: String,
    pub precision: String,
    pub bytes: u64,
}

/// A base cartridge's manifest and label.
#[derive(Clone, Debug, PartialEq)]
pub struct Manifest {
    pub id: String,
    pub population: String,
    pub base_sha256: String,
    pub atlas_version: u32,
    pub job: String,
    /// First and last layer covered.
    pub layers: (u16, u16),
    pub mask_sha256: String,
    pub closeness: String,
    pub date: String,
    pub packets: Vec<PacketRef>,
}

impl Manifest {
    /// Bytes of its distinct packets.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        let mut seen = BTreeSet::new();
        self.packets.iter().filter(|p| seen.insert(p.hash.as_str())).map(|p| p.bytes).sum()
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let s = |t: &str| Value::String(t.to_owned());
        let n = |x: u64| Value::Number(x.to_string());
        let o = |pairs: Vec<(&str, Value)>| Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect());
        o(vec![
            ("schema", s(SCHEMA)),
            ("id", s(&self.id)),
            ("population", s(&self.population)),
            ("base_sha256", s(&self.base_sha256)),
            ("atlas_version", n(u64::from(self.atlas_version))),
            ("job", s(&self.job)),
            ("layers", Value::Array(vec![n(u64::from(self.layers.0)), n(u64::from(self.layers.1))])),
            ("mask_sha256", s(&self.mask_sha256)),
            ("closeness", s(&self.closeness)),
            ("date", s(&self.date)),
            (
                "packets",
                Value::Array(
                    self.packets
                        .iter()
                        .map(|p| {
                            o(vec![("hash", s(&p.hash)), ("cluster", s(&p.cluster)), ("precision", s(&p.precision)), ("bytes", n(p.bytes))])
                        })
                        .collect(),
                ),
            ),
        ])
    }
    /// # Errors
    /// A wrong schema or a missing field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let s = |x: &Value, k: &str| x.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| bad(format!("{k}: {e}")));
        let u = |x: &Value, k: &str| x.get(k).and_then(Value::integer).map_err(|e| bad(format!("{k}: {e}")));
        if s(v, "schema")? != SCHEMA {
            return Err(bad("not an nm-cartridge/1 manifest"));
        }
        let layers = v.get("layers").and_then(Value::array).map_err(|e| bad(e.to_string()))?;
        let l = |i: usize| layers.get(i).and_then(|x| x.integer().ok()).and_then(|x| u16::try_from(x).ok()).ok_or_else(|| bad("layers"));
        let packets = v
            .get("packets")
            .and_then(Value::array)
            .map_err(|e| bad(e.to_string()))?
            .iter()
            .map(|p| Ok(PacketRef { hash: s(p, "hash")?, cluster: s(p, "cluster")?, precision: s(p, "precision")?, bytes: u(p, "bytes")? }))
            .collect::<Result<Vec<_>, Fault>>()?;
        Ok(Self {
            id: s(v, "id")?,
            population: s(v, "population")?,
            base_sha256: s(v, "base_sha256")?,
            atlas_version: u32::try_from(u(v, "atlas_version")?).map_err(|e| bad(e.to_string()))?,
            job: s(v, "job")?,
            layers: (l(0)?, l(1)?),
            mask_sha256: s(v, "mask_sha256")?,
            closeness: s(v, "closeness")?,
            date: s(v, "date")?,
            packets,
        })
    }
    /// Text form (pretty JSON).
    #[must_use]
    pub fn to_text(&self) -> String {
        nmjson::pretty::pretty(&self.to_value())
    }
    /// # Errors
    /// Malformed text or manifest.
    pub fn from_text(t: &str) -> Result<Self, Fault> {
        Self::from_value(&nmjson::parse(t.as_bytes()).map_err(|e| bad(e.to_string()))?)
    }
    /// M, the label check: the manifest must name the weights it would run on.
    /// # Errors
    /// A population, base hash or atlas version that does not match.
    pub fn check_label(&self, population: &str, base_sha256: &str, atlas_version: u32) -> Result<(), Fault> {
        if self.population != population || self.base_sha256 != base_sha256 {
            return Err(bad(format!(
                "{}: made for {}@{}, not {}@{}",
                self.id,
                self.population,
                short(&self.base_sha256),
                population,
                short(base_sha256)
            )));
        }
        if self.atlas_version != atlas_version {
            return Err(bad(format!("{}: atlas version {} against {}", self.id, self.atlas_version, atlas_version)));
        }
        Ok(())
    }
}

fn short(h: &str) -> &str {
    h.get(..12).unwrap_or(h)
}

/// J and K: the packets resident on the card (or in RAM), each held once by content hash, with
/// the cartridges using it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Resident {
    pub packets: BTreeMap<String, (PacketRef, BTreeSet<String>)>,
}

impl Resident {
    /// Bytes held (each distinct packet once).
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.packets.values().map(|(p, _)| p.bytes).sum()
    }
    /// K: what loading `m` would move: its packets not yet resident (each once).
    #[must_use]
    pub fn missing(&self, m: &Manifest) -> Vec<PacketRef> {
        let mut seen = BTreeSet::new();
        m.packets.iter().filter(|p| !self.packets.contains_key(&p.hash) && seen.insert(p.hash.clone())).cloned().collect()
    }
    /// K: bytes of a switch to `m`.
    #[must_use]
    pub fn switch_cost(&self, m: &Manifest) -> u64 {
        self.missing(m).iter().map(|p| p.bytes).sum()
    }
    /// J: load `m` (only missing packets are added; shared ones gain a user). Returns the bytes moved.
    pub fn load(&mut self, m: &Manifest) -> u64 {
        let moved = self.switch_cost(m);
        for p in &m.packets {
            self.packets.entry(p.hash.clone()).or_insert_with(|| (p.clone(), BTreeSet::new())).1.insert(m.id.clone());
        }
        moved
    }
    /// Release `m`: its packets lose that user; a packet with no user leaves. Returns bytes freed.
    pub fn release(&mut self, id: &str) -> u64 {
        let mut freed = 0;
        self.packets.retain(|_, (p, users)| {
            users.remove(id);
            if users.is_empty() {
                freed += p.bytes;
                false
            } else {
                true
            }
        });
        freed
    }
    /// M: load `want` if its label matches, else the full model `full` (the fallback). Returns
    /// (the manifest loaded, bytes moved, the refusal if any).
    pub fn load_or_fallback<'a>(
        &mut self,
        want: &'a Manifest,
        full: &'a Manifest,
        population: &str,
        base_sha256: &str,
        atlas_version: u32,
    ) -> (&'a Manifest, u64, Option<String>) {
        match want.check_label(population, base_sha256, atlas_version) {
            Ok(()) => (want, self.load(want), None),
            Err(e) => (full, self.load(full), Some(e.message)),
        }
    }
}

/// Section check: round trip; label refusals and the fallback; shared packets held once; switch
/// cost equals the bytes of the difference; release frees only unshared packets.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let p = |h: &str, b: u64| PacketRef { hash: h.into(), cluster: format!("m/t/{h}"), precision: "q8".into(), bytes: b };
    let m = |id: &str, ps: Vec<PacketRef>| Manifest {
        id: id.into(),
        population: "pop".into(),
        base_sha256: "aa".repeat(32),
        atlas_version: 3,
        job: id.into(),
        layers: (0, 27),
        mask_sha256: "bb".repeat(32),
        closeness: "within".into(),
        date: "2026-11-12".into(),
        packets: ps,
    };
    let a = m("judge", vec![p("table", 100), p("l0", 10), p("l1", 10)]);
    let b = m("embed", vec![p("table", 100), p("e0", 20)]);
    if Manifest::from_text(&a.to_text()).map_err(|e| e.to_string())? != a {
        return Err("nmnpcs.cartridge: round trip".into());
    }
    let mut r = Resident::default();
    if r.load(&a) != 120 || r.switch_cost(&b) != 20 || r.load(&b) != 20 || r.bytes() != 140 {
        return Err(format!("nmnpcs.cartridge: sharing {}", r.bytes()));
    }
    if r.release("judge") != 20 || r.bytes() != 120 {
        return Err("nmnpcs.cartridge: release".into());
    }
    let full = m("full", vec![p("table", 100), p("l0", 10), p("l1", 10), p("e0", 20)]);
    let mut r2 = Resident::default();
    let (got, _, why) = r2.load_or_fallback(&a, &full, "pop", &"aa".repeat(32), 4);
    if got.id != "full" || why.is_none() {
        return Err("nmnpcs.cartridge: a wrong atlas version was not refused".into());
    }
    let (got, _, why) = r2.load_or_fallback(&a, &full, "pop", &"aa".repeat(32), 3);
    if got.id != "judge" || why.is_some() {
        return Err("nmnpcs.cartridge: a matching label was refused".into());
    }
    Ok(())
}
