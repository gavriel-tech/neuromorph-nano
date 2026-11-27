//! The Context Packet, the only traffic between units (guide 6, 6.1, 6.3, 6.4).
//! Six epistemic statuses, stance and certainty kept separate, twelve context categories,
//! provenance, evidence, relationships, scope, constraints, anomaly flag and hop log.
//! Packets are immutable: there is no `&mut` method. The builder methods consume the
//! packet and return a new one; `hopped` returns a new packet with one more hop and the
//! same id.
//! Since 2026-11-10 (step P5) a packet carries a data class (specification 5.1). Default
//! sensory: `Packet::new` and `from_value` without a `class` field give sensory, so every
//! existing packet, its id and its JSON form are unchanged. A behavioural packet is made
//! only by `Packet::new_behavioural` (its id seed adds the class, so it never shares an id
//! with a sensory reading); no method turns an existing packet into the other class.
use crate::class::{Classed, DataClass};
use crate::component::{ComponentId, Fault};
use crate::hop::Hop;
use crate::wire::{self, named_enum};
use nmjson::Value;

named_enum! {
    /// What is known about a claim.
    pub enum Status in "packet" {
        Observed => "observed", Inferred => "inferred", Hypothesized => "hypothesized",
        Verified => "verified", Contradicted => "contradicted", Unknown => "unknown",
    }
}
named_enum! {
    /// Stance of the source.
    pub enum Stance in "packet" {
        Objective => "objective", Subjective => "subjective", Mixed => "mixed", Unknown => "unknown",
    }
}
named_enum! {
    /// Certainty, kept separate from status and stance.
    pub enum Certainty in "packet" { Absolute => "absolute", Relative => "relative" }
}
named_enum! {
    /// Context category (guide 6.3). A classification, never a substitute for evidence.
    pub enum Category in "packet" {
        Instruction => "instruction", Constraint => "constraint", Correction => "correction",
        Support => "support", Example => "example", Preference => "preference", Fact => "fact",
        Speculation => "speculation", Inference => "inference", Perspective => "perspective",
        Temporal => "temporal", Domain => "domain",
    }
}

/// Packet kinds the mini uses. Other kinds are accepted; these are the documented set.
pub const KINDS: &[&str] = &[
    "symbol",
    "deps",
    "grep",
    "chunk",
    "history",
    "build-error",
    "test-result",
    "finding",
    "memory",
    "question",
    "invalidation",
    "telemetry",
    "plan",
    "pack",
];

/// Where a claim came from. `source` is `path:line`, `crate:name`, `git:sha`, `memory:id`,
/// `build:run` or `unit:id`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub source: String,
    pub origin: ComponentId,
    pub version: u64,
    pub read_at: u64,
}
impl Provenance {
    #[must_use]
    pub fn new(source: &str, origin: &str, version: u64, read_at: u64) -> Self {
        Self { source: source.to_owned(), origin: origin.to_owned(), version, read_at }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        wire::obj(vec![
            ("source", wire::s(&self.source)),
            ("origin", wire::s(&self.origin)),
            ("version", wire::u(self.version)),
            ("read_at", wire::u(self.read_at)),
        ])
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        Ok(Self {
            source: wire::get_s(v, "packet", "source")?,
            origin: wire::get_s(v, "packet", "origin")?,
            version: wire::get_u(v, "packet", "version")?,
            read_at: wire::get_u(v, "packet", "read_at")?,
        })
    }
}

/// One typed, cited claim.
#[derive(Clone, Debug, PartialEq)]
pub struct Packet {
    id: u64,
    kind: String,
    key: String,
    text: String,
    category: Category,
    status: Status,
    stance: Stance,
    certainty: Certainty,
    confidence: f32,
    provenance: Provenance,
    evidence: Vec<String>,
    relations: Vec<(String, u64)>,
    scope: String,
    constraints: Vec<String>,
    anomaly: bool,
    hops: Vec<Hop>,
    class: DataClass,
}

impl Packet {
    /// Start a packet. The id is derived from kind, key, text, source and version, so the
    /// same reading gives the same id on every run. Defaults: status, stance `Unknown`;
    /// certainty `Relative`; category `Domain`; confidence 0.
    #[must_use]
    pub fn new(kind: &str, key: &str, text: &str, provenance: Provenance) -> Self {
        let seed = format!("{kind}\u{1f}{key}\u{1f}{text}\u{1f}{}\u{1f}{}", provenance.source, provenance.version);
        let digest = nmhash::sha256_bytes(seed.as_bytes());
        let id = digest.get(..16).and_then(|h| u64::from_str_radix(h, 16).ok()).unwrap_or_default();
        Self {
            id,
            kind: kind.to_owned(),
            key: key.to_owned(),
            text: text.to_owned(),
            category: Category::Domain,
            status: Status::Unknown,
            stance: Stance::Unknown,
            certainty: Certainty::Relative,
            confidence: 0.0,
            provenance,
            evidence: Vec::new(),
            relations: Vec::new(),
            scope: String::new(),
            constraints: Vec::new(),
            anomaly: false,
            hops: Vec::new(),
            class: DataClass::Sensory,
        }
    }
    /// Start a behavioural packet (an in-house signal carried as a packet). Same defaults
    /// as `new`; the id seed adds the class name, so it differs from a sensory packet with
    /// the same fields.
    #[must_use]
    pub fn new_behavioural(kind: &str, key: &str, text: &str, provenance: Provenance) -> Self {
        let mut p = Self::new(kind, key, text, provenance);
        let seed = format!("{}\u{1f}{}", p.id, DataClass::Behavioural.name());
        p.id = nmhash::sha256_bytes(seed.as_bytes()).get(..16).and_then(|h| u64::from_str_radix(h, 16).ok()).unwrap_or_default();
        p.class = DataClass::Behavioural;
        p
    }
    /// Like `hopped`, but refuses a hop whose data class differs from the packet's.
    /// # Errors
    /// `class::WRONG_CLASS` fault.
    pub fn hopped_checked(self, hop: Hop) -> Result<Self, Fault> {
        crate::class::guard("packet-hop", self.class, hop.class)?;
        Ok(self.hopped(hop))
    }
    #[must_use]
    pub const fn get_class(&self) -> DataClass {
        self.class
    }
    #[must_use]
    pub fn status(self, status: Status) -> Self {
        Self { status, ..self }
    }
    #[must_use]
    pub fn stance(self, stance: Stance) -> Self {
        Self { stance, ..self }
    }
    #[must_use]
    pub fn certainty(self, certainty: Certainty) -> Self {
        Self { certainty, ..self }
    }
    #[must_use]
    pub fn category(self, category: Category) -> Self {
        Self { category, ..self }
    }
    /// Confidence clamped to 0..=1 (non-finite becomes 0).
    #[must_use]
    pub fn confidence(self, confidence: f32) -> Self {
        let confidence = if confidence.is_finite() { confidence.clamp(0.0, 1.0) } else { 0.0 };
        Self { confidence, ..self }
    }
    #[must_use]
    pub fn evidence(mut self, lines: Vec<String>) -> Self {
        self.evidence.extend(lines);
        self
    }
    #[must_use]
    pub fn relate(mut self, rel: &str, id: u64) -> Self {
        self.relations.push((rel.to_owned(), id));
        self
    }
    #[must_use]
    pub fn scope(self, scope: &str) -> Self {
        Self { scope: scope.to_owned(), ..self }
    }
    #[must_use]
    pub fn constraints(mut self, rules: Vec<String>) -> Self {
        self.constraints.extend(rules);
        self
    }
    #[must_use]
    pub fn anomaly(self, anomaly: bool) -> Self {
        Self { anomaly, ..self }
    }
    /// A new packet with `hop` appended; the id and every other field stay.
    #[must_use]
    pub fn hopped(mut self, hop: Hop) -> Self {
        self.hops.push(hop);
        self
    }

    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }
    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
    #[must_use]
    pub const fn get_category(&self) -> Category {
        self.category
    }
    #[must_use]
    pub const fn get_status(&self) -> Status {
        self.status
    }
    #[must_use]
    pub const fn get_stance(&self) -> Stance {
        self.stance
    }
    #[must_use]
    pub const fn get_certainty(&self) -> Certainty {
        self.certainty
    }
    #[must_use]
    pub const fn get_confidence(&self) -> f32 {
        self.confidence
    }
    #[must_use]
    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    #[must_use]
    pub fn get_evidence(&self) -> &[String] {
        &self.evidence
    }
    #[must_use]
    pub fn relations(&self) -> &[(String, u64)] {
        &self.relations
    }
    #[must_use]
    pub fn get_scope(&self) -> &str {
        &self.scope
    }
    #[must_use]
    pub fn get_constraints(&self) -> &[String] {
        &self.constraints
    }
    #[must_use]
    pub const fn is_anomaly(&self) -> bool {
        self.anomaly
    }
    #[must_use]
    pub fn hops(&self) -> &[Hop] {
        &self.hops
    }

    #[must_use]
    pub fn to_value(&self) -> Value {
        let relations = self.relations.iter().map(|(r, id)| Value::Array(vec![wire::s(r), wire::u(*id)])).collect();
        let mut fields = vec![
            ("id", wire::u(self.id)),
            ("kind", wire::s(&self.kind)),
            ("key", wire::s(&self.key)),
            ("text", wire::s(&self.text)),
            ("category", self.category.to_value()),
            ("status", self.status.to_value()),
            ("stance", self.stance.to_value()),
            ("certainty", self.certainty.to_value()),
            ("confidence", wire::f(self.confidence)),
            ("provenance", self.provenance.to_value()),
            ("evidence", wire::strings(&self.evidence)),
            ("relations", Value::Array(relations)),
            ("scope", wire::s(&self.scope)),
            ("constraints", wire::strings(&self.constraints)),
            ("anomaly", Value::Bool(self.anomaly)),
            ("hops", Value::Array(self.hops.iter().map(Hop::to_value).collect())),
        ];
        if self.class != DataClass::Sensory {
            fields.push(("class", self.class.to_value()));
        }
        wire::obj(fields)
    }
    /// # Errors
    /// Rejects a value missing any field or with an unknown enum name.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "packet";
        let relations = wire::get_a(v, m, "relations")?
            .iter()
            .map(|pair| match pair.array().map_err(|e| wire::bad(m, e.to_string()))? {
                [Value::String(r), id] => Ok((r.clone(), id.integer().map_err(|e| wire::bad(m, e.to_string()))?)),
                _ => Err(wire::bad(m, "relation must be [name, id]")),
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            id: wire::get_u(v, m, "id")?,
            kind: wire::get_s(v, m, "kind")?,
            key: wire::get_s(v, m, "key")?,
            text: wire::get_s(v, m, "text")?,
            category: wire::get_e(v, m, "category", Category::from_value)?,
            status: wire::get_e(v, m, "status", Status::from_value)?,
            stance: wire::get_e(v, m, "stance", Stance::from_value)?,
            certainty: wire::get_e(v, m, "certainty", Certainty::from_value)?,
            confidence: wire::get_f(v, m, "confidence")?,
            provenance: Provenance::from_value(wire::get_v(v, m, "provenance")?)?,
            evidence: wire::get_strings(v, m, "evidence")?,
            relations,
            scope: wire::get_s(v, m, "scope")?,
            constraints: wire::get_strings(v, m, "constraints")?,
            anomaly: wire::get_b(v, m, "anomaly")?,
            hops: wire::get_a(v, m, "hops")?.iter().map(Hop::from_value).collect::<Result<_, _>>()?,
            class: match v.get("class") {
                Ok(c) => DataClass::from_value(c)?,
                Err(_) => DataClass::Sensory,
            },
        })
    }
}
impl Classed for Packet {
    fn data_class(&self) -> DataClass {
        self.class
    }
}

/// Self-check of the `packet` sub-component: a fully populated packet survives a JSON
/// round trip unchanged, `hopped` keeps the id, and the id is reproducible.
pub(crate) fn check() -> Result<(), String> {
    let prov = Provenance::new("crates/nmtype/src/packet.rs:1", "nmpod.unit#d1", 7, 42);
    let p = Packet::new("symbol", "nmtype::Packet", "struct Packet", prov.clone())
        .status(Status::Observed)
        .stance(Stance::Objective)
        .certainty(Certainty::Relative)
        .category(Category::Fact)
        .confidence(0.75)
        .evidence(vec!["packet.rs:1 pub struct Packet".into()])
        .relate("defines", 9)
        .scope("t3")
        .constraints(vec!["no em dashes".into()])
        .anomaly(true);
    let back = Packet::from_value(&p.to_value()).map_err(|e| e.to_string())?;
    if back != p {
        return Err("nmtype.packet: round trip changed a field".into());
    }
    let hopped = p.clone().hopped(Hop::new("nmbrain.synthesis#b0", 43, "reduce", 1));
    if hopped.id() != p.id() || hopped.hops().len() != 1 {
        return Err("nmtype.packet: hopped changed the id or lost the hop".into());
    }
    if Packet::new("symbol", "nmtype::Packet", "struct Packet", prov).id() != p.id() {
        return Err("nmtype.packet: id not reproducible".into());
    }
    if p.get_class() != DataClass::Sensory || p.to_value().get("class").is_ok() {
        return Err("nmtype.packet: default class or default JSON form changed".into());
    }
    let b = Packet::new_behavioural("report", "k", "t", p.provenance().clone());
    if b.get_class() != DataClass::Behavioural
        || b.id() == Packet::new("report", "k", "t", p.provenance().clone()).id()
        || Packet::from_value(&b.to_value()).map_err(|e| e.to_string())? != b
    {
        return Err("nmtype.packet: behavioural packet id or round trip".into());
    }
    if p.hopped_checked(Hop::new("x", 1, "y", 0).with_class(DataClass::Behavioural)).is_ok() {
        return Err("nmtype.packet: wrong-class hop accepted".into());
    }
    Ok(())
}
