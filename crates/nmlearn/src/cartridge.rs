//! Cartridges, first form (step P9; guide 6 "Cartridges", "Written parameters"; guide 9 item 4;
//! research note on the open gaps, gap 4 and the cross-analysis). A cartridge is anything
//! trained that plugs in beside the released weights and stacks: a learning-weight unit, a
//! trained encoded cache for one body of text, or another kind. Muscle memory packs are not
//! cartridges and are refused by name. Released weights are never changed: a cartridge is
//! loaded and unloaded beside them (for learning units through the existing `registry`).
//!
//! Separable components:
//! - `Label`: the self-describing header (kind, data class, base hash, components covered,
//!   scopes, quantization, description, version, check result, date) with a loader and a
//!   validator that return named faults (`CartFault`).
//! - `Policy`: the schema values and stacking policy read from `designs/cartridges.json`, with
//!   a stated fallback when the file cannot be read.
//! - Unit adapter (`from_unit`, `units_as_cartridges`): existing learning units shown as
//!   cartridges without touching their files.
//! - Slot check (`Loaded`, `slot_check`, `plug`, `unplug`): a cartridge loads only when its base
//!   hash and covered components match what is loaded.
//! - Selection (`Request`, `select`): the active set per request, scope match first, a fixed
//!   deterministic order, sensory and behavioural cartridges never in one set.
//! - Disagreement (`Track`, `threshold`, `resolve`): a reliability rule over each cartridge's
//!   recorded track record with three outcomes: combine, one controls, re-read. The bar a
//!   cartridge must clear is the base path's recorded success rate, not a fixed number.
//!
//! Composition is by selection and output-level arbitration, never by merging parameters
//! (selection per input after Arrow and LoraRetriever; concatenation without retraining for
//! encoded caches after Eyuboglu et al. 2025, arXiv 2506.06266, which gives no ordering,
//! selection or conflict rule, so those are ours).
use crate::outcomes::Outcome;
use crate::registry::Units;
use crate::unit::{UnitCard, n, obj, s};
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::path::Path;

/// Label schema id.
pub const SCHEMA: &str = "nm-cartridge/1";
/// The design file holding the schema values and the stacking policy.
pub const DESIGN: &str = "designs/cartridges.json";

/// What a cartridge is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    LearningUnit,
    EncodedCache,
    Other,
}
impl Kind {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::LearningUnit => "learning-unit",
            Self::EncodedCache => "encoded-cache",
            Self::Other => "other",
        }
    }
    /// # Errors
    /// `NotACartridge` for a muscle memory pack, `BadValue` for an unknown kind.
    pub fn parse(t: &str) -> Result<Self, CartFault> {
        match t {
            "learning-unit" => Ok(Self::LearningUnit),
            "encoded-cache" => Ok(Self::EncodedCache),
            "other" => Ok(Self::Other),
            "muscle-memory-pack" | "muscle-memory" => Err(CartFault::NotACartridge(t.into())),
            _ => Err(CartFault::BadValue { field: "kind".into(), value: t.into() }),
        }
    }
}

/// The data class a cartridge belongs to (guide 5.1): the two are never mixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataClass {
    Sensory,
    Behavioural,
}
impl DataClass {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sensory => "sensory",
            Self::Behavioural => "behavioural",
        }
    }
    /// # Errors
    /// An unknown class.
    pub fn parse(t: &str) -> Result<Self, CartFault> {
        match t {
            "sensory" => Ok(Self::Sensory),
            "behavioural" => Ok(Self::Behavioural),
            _ => Err(CartFault::BadValue { field: "data_class".into(), value: t.into() }),
        }
    }
}

/// Named faults of this module.
#[derive(Clone, Debug, PartialEq)]
pub enum CartFault {
    MissingField(String),
    BadValue { field: String, value: String },
    NotACartridge(String),
    BaseMismatch { id: String, want: String, got: String },
    ComponentNotLoaded { id: String, component: String },
    ClassMismatch { id: String, want: DataClass, got: DataClass },
    Unreadable(String),
}
impl CartFault {
    /// The fault's stable name.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::MissingField(_) => "cart-missing-field",
            Self::BadValue { .. } => "cart-bad-value",
            Self::NotACartridge(_) => "cart-not-a-cartridge",
            Self::BaseMismatch { .. } => "cart-base-mismatch",
            Self::ComponentNotLoaded { .. } => "cart-component-not-loaded",
            Self::ClassMismatch { .. } => "cart-class-mismatch",
            Self::Unreadable(_) => "cart-unreadable",
        }
    }
    /// As a component fault of `nmlearn.cartridge`.
    #[must_use]
    pub fn to_fault(&self) -> Fault {
        let kind = match self {
            Self::MissingField(_) | Self::BadValue { .. } | Self::Unreadable(_) => FaultKind::Config,
            Self::NotACartridge(_) | Self::ClassMismatch { .. } => FaultKind::Policy,
            Self::BaseMismatch { .. } => FaultKind::Integrity,
            Self::ComponentNotLoaded { .. } => FaultKind::Interface,
        };
        Fault::new("nmlearn.cartridge", kind, self.to_string())
    }
}
impl std::fmt::Display for CartFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let detail = match self {
            Self::MissingField(k) => format!("label field {k} missing"),
            Self::BadValue { field, value } => format!("label field {field} has value {value:?}"),
            Self::NotACartridge(k) => format!("{k} is not a cartridge (muscle memory is its own store)"),
            Self::BaseMismatch { id, want, got } => format!("{id} fits base {} but {} is loaded; refused", short(want), short(got)),
            Self::ComponentNotLoaded { id, component } => format!("{id} covers {component}, which is not loaded; refused"),
            Self::ClassMismatch { id, want, got } => format!("{id} is {} but the request is {}", got.name(), want.name()),
            Self::Unreadable(m) => m.clone(),
        };
        write!(f, "{}: {detail}", self.code())
    }
}
fn short(h: &str) -> &str {
    h.get(..12).unwrap_or(h)
}

fn num_f(x: f64) -> Value {
    Value::Number(format!("{x:.6}"))
}
fn float(v: &Value) -> Option<f64> {
    match v {
        Value::Number(t) => t.parse().ok(),
        _ => None,
    }
}
fn strings(v: &Value, k: &str) -> Option<Vec<String>> {
    v.get(k).ok()?.array().ok().map(|a| a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect())
}

/// Schema values and stacking policy (`designs/cartridges.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct Policy {
    /// Where the values came from: the design file, or "fallback: <reason>".
    pub source: String,
    pub quantizations: Vec<String>,
    pub checks: Vec<String>,
    /// Kinds whose data class is fixed (an encoded cache of a body of text is sensory).
    pub class_by_kind: BTreeMap<String, String>,
    /// Data class given to learning units by the adapter.
    pub unit_class: DataClass,
    /// Registry status to check result.
    pub status_check: BTreeMap<String, String>,
    /// Most active cartridges in one set (rules.md: a cap of 16 active units).
    pub max_active: usize,
    /// One-sided normal quantile of the reliability interval.
    pub z: f64,
    /// Beta prior (successes, failures) of a track record.
    pub prior: (f64, f64),
    /// Bar used when no base outcome is recorded.
    pub fallback_threshold: f64,
}
impl Policy {
    /// The stated fallback values (the same values the design file holds).
    #[must_use]
    pub fn fallback(reason: &str) -> Self {
        let m = |p: &[(&str, &str)]| p.iter().map(|(a, b)| ((*a).to_owned(), (*b).to_owned())).collect();
        Self {
            source: format!("fallback: {reason}"),
            quantizations: ["nmq8-row-f16", "f16", "f32", "none"].map(str::to_owned).to_vec(),
            checks: ["accept", "candidate", "reject", "rolled-back", "retired", "none"].map(str::to_owned).to_vec(),
            class_by_kind: m(&[("encoded-cache", "sensory")]),
            unit_class: DataClass::Behavioural,
            status_check: m(&[
                ("fitted", "accept"),
                ("active", "accept"),
                ("accepted-inactive", "accept"),
                ("candidate", "candidate"),
                ("rejected", "reject"),
                ("rolled-back", "rolled-back"),
                ("retired", "retired"),
                ("unfitted", "none"),
            ]),
            max_active: 16,
            z: 1.645,
            prior: (1.0, 1.0),
            fallback_threshold: 0.5,
        }
    }
    /// Read the policy from `<home>/designs/cartridges.json`, or the fallback with its reason.
    #[must_use]
    pub fn load(home: &Path) -> Self {
        let path = home.join(DESIGN);
        let parsed = std::fs::read(&path).map_err(|e| e.to_string()).and_then(|b| nmjson::parse(&b).map_err(|e| e.to_string()));
        match parsed.and_then(|v| Self::from_value(&v)) {
            Ok(mut p) => {
                p.source = DESIGN.into();
                p
            }
            Err(e) => Self::fallback(&e),
        }
    }
    /// # Errors
    /// A missing or malformed policy field.
    pub fn from_value(v: &Value) -> Result<Self, String> {
        let label = v.get("label").map_err(|e| e.to_string())?;
        let st = v.get("stacking").map_err(|e| e.to_string())?;
        let map = |x: &Value, k: &str| -> Result<BTreeMap<String, String>, String> {
            Ok(x.get(k)
                .and_then(Value::object)
                .map_err(|e| format!("{k}: {e}"))?
                .iter()
                .filter_map(|(a, b)| Some((a.clone(), b.string().ok()?.to_owned())))
                .collect())
        };
        let f = |x: &Value, k: &str| x.get(k).ok().and_then(float).ok_or_else(|| format!("{k}: not a number"));
        let prior = st.get("beta_prior").and_then(Value::array).map_err(|e| e.to_string())?;
        Ok(Self {
            source: String::new(),
            quantizations: strings(label, "quantizations").ok_or("quantizations")?,
            checks: strings(label, "check_results").ok_or("check_results")?,
            class_by_kind: map(label, "data_class_by_kind")?,
            unit_class: DataClass::parse(label.get("unit_data_class").and_then(Value::string).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?,
            status_check: map(label, "unit_status_to_check")?,
            max_active: usize::try_from(st.get("max_active").and_then(Value::integer).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?,
            z: f(st, "z")?,
            prior: (prior.first().and_then(float).ok_or("beta_prior")?, prior.get(1).and_then(float).ok_or("beta_prior")?),
            fallback_threshold: f(st, "fallback_threshold")?,
        })
    }
}

/// The self-describing header of a cartridge.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub id: String,
    pub kind: Kind,
    pub data_class: DataClass,
    /// Base population name (`weights/<base>`).
    pub base: String,
    /// SHA-256 of the released base weights the cartridge fits.
    pub base_sha256: String,
    /// Component nodes or layers covered.
    pub components: Vec<String>,
    /// Scope keys it serves, `role:kind:repo`, missing or `*` segments match any.
    pub scopes: Vec<String>,
    pub quantization: String,
    pub description: String,
    pub version: u64,
    /// Step Check result.
    pub check: String,
    /// UTC date written, `YYYY-MM-DD`.
    pub date: String,
    pub loaded: bool,
}
impl Label {
    #[must_use]
    pub fn to_value(&self) -> Value {
        let list = |l: &[String]| Value::Array(l.iter().map(|x| s(x)).collect());
        obj(vec![
            ("schema", s(SCHEMA)),
            ("id", s(&self.id)),
            ("kind", s(self.kind.name())),
            ("data_class", s(self.data_class.name())),
            ("base", s(&self.base)),
            ("base_sha256", s(&self.base_sha256)),
            ("components", list(&self.components)),
            ("scopes", list(&self.scopes)),
            ("quantization", s(&self.quantization)),
            ("description", s(&self.description)),
            ("version", n(self.version)),
            ("check", s(&self.check)),
            ("date", s(&self.date)),
            ("loaded", Value::Bool(self.loaded)),
        ])
    }
    /// Parse and validate a label.
    /// # Errors
    /// A named fault for the first field that is missing or wrong.
    pub fn from_value(v: &Value, policy: &Policy) -> Result<Self, CartFault> {
        let g = |k: &str| v.get(k).and_then(Value::string).map(str::to_owned).map_err(|_| CartFault::MissingField(k.into()));
        let l = |k: &str| strings(v, k).ok_or_else(|| CartFault::MissingField(k.into()));
        let schema = g("schema")?;
        if schema != SCHEMA {
            return Err(CartFault::BadValue { field: "schema".into(), value: schema });
        }
        let label = Self {
            id: g("id")?,
            kind: Kind::parse(&g("kind")?)?,
            data_class: DataClass::parse(&g("data_class")?)?,
            base: g("base")?,
            base_sha256: g("base_sha256")?,
            components: l("components")?,
            scopes: l("scopes")?,
            quantization: g("quantization")?,
            description: g("description")?,
            version: v.get("version").and_then(Value::integer).map_err(|_| CartFault::MissingField("version".into()))?,
            check: g("check")?,
            date: g("date")?,
            loaded: matches!(v.get("loaded"), Ok(Value::Bool(true))),
        };
        label.validate(policy)?;
        Ok(label)
    }
    /// Read and validate `cartridge.json` (or any label file).
    /// # Errors
    /// `Unreadable`, or a validation fault.
    pub fn load(path: &Path, policy: &Policy) -> Result<Self, CartFault> {
        let bytes = std::fs::read(path).map_err(|e| CartFault::Unreadable(format!("{}: {e}", path.display())))?;
        let v = nmjson::parse(&bytes).map_err(|e| CartFault::Unreadable(format!("{}: {e}", path.display())))?;
        Self::from_value(&v, policy)
    }
    /// Field rules of the schema.
    /// # Errors
    /// The first rule broken, by name.
    pub fn validate(&self, policy: &Policy) -> Result<(), CartFault> {
        let bad = |field: &str, value: &str| Err(CartFault::BadValue { field: field.into(), value: value.into() });
        if self.id.is_empty() {
            return bad("id", "");
        }
        if self.base_sha256.len() != 64 || !self.base_sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return bad("base_sha256", &self.base_sha256);
        }
        if self.components.is_empty() || self.components.iter().any(String::is_empty) {
            return bad("components", "empty");
        }
        if self.scopes.is_empty() {
            return bad("scopes", "empty");
        }
        if !policy.quantizations.contains(&self.quantization) {
            return bad("quantization", &self.quantization);
        }
        if !policy.checks.contains(&self.check) {
            return bad("check", &self.check);
        }
        if self.version == 0 {
            return bad("version", "0");
        }
        let d = self.date.as_bytes();
        if d.len() != 10
            || d.get(4) != Some(&b'-')
            || d.get(7) != Some(&b'-')
            || !d.iter().enumerate().all(|(i, b)| i == 4 || i == 7 || b.is_ascii_digit())
        {
            return bad("date", &self.date);
        }
        if policy.class_by_kind.get(self.kind.name()).is_some_and(|c| c != self.data_class.name()) {
            return bad("data_class", self.data_class.name());
        }
        Ok(())
    }
}

/// UTC calendar date of a time in milliseconds since 1970 (civil-from-days).
#[must_use]
pub fn date_of(ms: u64) -> String {
    let z = ms / 86_400_000 + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + u64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// A learning unit seen as a cartridge. Read only: the unit's folder and card are unchanged.
/// Scopes come from the unit's `verdict.json` accepted scopes when it has one, else its slot.
#[must_use]
pub fn from_unit(card: &UnitCard, dir: &Path, policy: &Policy) -> Label {
    let scopes = std::fs::read(dir.join("verdict.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| strings(&v, "accepted_scopes"))
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![card.slot.clone()]);
    let quantization =
        if card.files.iter().any(|(p, _)| p.ends_with(".q8")) || dir.join("readout.q8").is_file() { "nmq8-row-f16" } else { "none" };
    Label {
        id: card.id.clone(),
        kind: Kind::LearningUnit,
        data_class: policy.unit_class,
        base: card.base.clone(),
        base_sha256: card.base_sha256.clone(),
        components: vec![card.target.clone()],
        scopes,
        quantization: quantization.into(),
        description: format!(
            "learning-weight unit, slot {}, target {}, rank {}, status {}",
            card.slot, card.target, card.rank, card.status
        ),
        version: 1 + card.lineage.len() as u64,
        check: policy.status_check.get(&card.status).cloned().unwrap_or_else(|| "none".into()),
        date: date_of(card.registered_at),
        loaded: card.loaded,
    }
}

/// Every registered unit as a cartridge, in registration order.
#[must_use]
pub fn units_as_cartridges(units: &Units, policy: &Policy) -> Vec<Label> {
    units.cards.iter().map(|c| from_unit(c, &units.unit_dir(&c.id), policy)).collect()
}

/// What is loaded now: the base population and the component nodes present.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loaded {
    pub base: String,
    pub base_sha256: String,
    /// Node ids; an entry ending in `*` covers every node with that prefix.
    pub components: Vec<String>,
}
impl Loaded {
    /// The population `base` as recorded in `weights/<base>/part.json` (field `weights.sha256`).
    /// # Errors
    /// `Unreadable` when the card or its hash cannot be read.
    pub fn population(home: &Path, base: &str, components: Vec<String>) -> Result<Self, CartFault> {
        let p = nmtype::models_dir(home).join(base).join("part.json");
        let v =
            std::fs::read(&p).ok().and_then(|b| nmjson::parse(&b).ok()).ok_or_else(|| CartFault::Unreadable(format!("{}", p.display())))?;
        let h = v
            .get("weights")
            .and_then(|w| w.get("sha256"))
            .and_then(Value::string)
            .map_err(|e| CartFault::Unreadable(format!("{}: {e}", p.display())))?;
        Ok(Self { base: base.into(), base_sha256: h.into(), components })
    }
    #[must_use]
    pub fn covers(&self, node: &str) -> bool {
        self.components.iter().any(|c| c == node || c.strip_suffix('*').is_some_and(|p| node.starts_with(p)))
    }
}

/// The slot check: base hash equal and every covered component loaded.
/// # Errors
/// `BaseMismatch` or `ComponentNotLoaded`.
pub fn slot_check(label: &Label, loaded: &Loaded) -> Result<(), CartFault> {
    if label.base_sha256 != loaded.base_sha256 {
        return Err(CartFault::BaseMismatch { id: label.id.clone(), want: label.base_sha256.clone(), got: loaded.base_sha256.clone() });
    }
    if let Some(c) = label.components.iter().find(|c| !loaded.covers(c)) {
        return Err(CartFault::ComponentNotLoaded { id: label.id.clone(), component: c.clone() });
    }
    Ok(())
}

/// Plug a learning-unit cartridge in: slot check, then the registry's hash-checked load.
/// # Errors
/// A slot-check fault, a kind this first form cannot load, or a registry fault.
pub fn plug(units: &mut Units, label: &Label, loaded: &Loaded) -> Result<(), Fault> {
    slot_check(label, loaded).map_err(|e| e.to_fault())?;
    if label.kind != Kind::LearningUnit {
        return Err(Fault::new(
            "nmlearn.cartridge",
            FaultKind::Policy,
            format!("{}: kind {} has no loader yet", label.id, label.kind.name()),
        ));
    }
    units.mark_loaded(&label.id)
}

/// Unplug (the exact rollback: the base never changed).
/// # Errors
/// A registry fault.
pub fn unplug(units: &mut Units, id: &str) -> Result<(), Fault> {
    units.mark_unloaded(id)
}

/// A cartridge's recorded track record in one scope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Track {
    pub trials: u64,
    pub successes: u64,
}
/// A Beta posterior summary: mean, spread and a one-sided interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reliability {
    pub mean: f64,
    pub std: f64,
    pub lower: f64,
    pub upper: f64,
}
impl Track {
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "counts are far below 2^52")]
    pub fn reliability(&self, policy: &Policy) -> Reliability {
        let a = policy.prior.0 + self.successes as f64;
        let b = policy.prior.1 + self.trials.saturating_sub(self.successes) as f64;
        let mean = a / (a + b);
        let std = (a * b / ((a + b) * (a + b) * (a + b + 1.0))).sqrt();
        Reliability { mean, std, lower: mean - policy.z * std, upper: mean + policy.z * std }
    }
}
/// Track records from use events `(cartridge id, outcome right)`.
#[must_use]
pub fn tracks_from(events: &[(String, bool)]) -> BTreeMap<String, Track> {
    let mut m: BTreeMap<String, Track> = BTreeMap::new();
    for (id, ok) in events {
        let t = m.entry(id.clone()).or_default();
        t.trials += 1;
        t.successes += u64::from(*ok);
    }
    m
}

/// The bar a cartridge must clear to count as reliable, and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Threshold {
    pub value: f64,
    pub source: String,
}
/// The base path's recorded success rate in `slot` (from the checked-outcome log): a
/// cartridge is reliable only when it is confidently better than running without it.
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "counts are far below 2^52")]
pub fn threshold(outcomes: &[Outcome], slot: &str, policy: &Policy) -> Threshold {
    let mine: Vec<&Outcome> = outcomes.iter().filter(|o| o.slot == slot).collect();
    if mine.is_empty() {
        return Threshold {
            value: policy.fallback_threshold,
            source: format!("fallback {} (no outcome recorded in slot {slot})", policy.fallback_threshold),
        };
    }
    let right = mine.iter().filter(|o| o.correct).count();
    Threshold { value: right as f64 / mine.len() as f64, source: format!("base success rate in slot {slot}: {right} of {}", mine.len()) }
}

/// The request a set is selected for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// `role:kind:repo`.
    pub scope: String,
    pub data_class: DataClass,
}
/// How many scope segments a cartridge scope matches (`None` when one conflicts).
#[must_use]
pub fn scope_tier(want: &str, have: &str) -> Option<usize> {
    let w: Vec<&str> = want.split(':').collect();
    let mut tier = 0;
    for (i, seg) in have.split(':').enumerate() {
        if seg == "*" {
            continue;
        }
        if w.get(i) != Some(&seg) {
            return None;
        }
        tier += 1;
    }
    Some(tier)
}
/// One selected cartridge.
#[derive(Clone, Debug, PartialEq)]
pub struct Selected {
    pub id: String,
    pub tier: usize,
    pub lower: f64,
}
/// The selection for one request.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Selection {
    pub active: Vec<Selected>,
    pub refused: Vec<(String, CartFault)>,
    pub out_of_scope: Vec<String>,
}
impl Selection {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            (
                "active",
                Value::Array(
                    self.active.iter().map(|a| obj(vec![("id", s(&a.id)), ("tier", n(a.tier)), ("lower", num_f(a.lower))])).collect(),
                ),
            ),
            (
                "refused",
                Value::Array(
                    self.refused
                        .iter()
                        .map(|(id, f)| obj(vec![("id", s(id)), ("fault", s(f.code())), ("detail", s(&f.to_string()))]))
                        .collect(),
                ),
            ),
            ("out_of_scope", Value::Array(self.out_of_scope.iter().map(|x| s(x)).collect())),
        ])
    }
}
/// Select the active set for `req`: data class equal, slot check passed, best scope tier at
/// least one segment unless the cartridge scope is all wildcards; ordered by tier (most
/// specific first), then reliability lower bound, then id; at most `max_active`.
#[must_use]
pub fn select(req: &Request, labels: &[Label], loaded: &Loaded, tracks: &BTreeMap<String, Track>, policy: &Policy) -> Selection {
    let mut sel = Selection::default();
    for l in labels {
        if l.data_class != req.data_class {
            sel.refused.push((l.id.clone(), CartFault::ClassMismatch { id: l.id.clone(), want: req.data_class, got: l.data_class }));
            continue;
        }
        if let Err(f) = slot_check(l, loaded) {
            sel.refused.push((l.id.clone(), f));
            continue;
        }
        match l.scopes.iter().filter_map(|sc| scope_tier(&req.scope, sc)).max() {
            Some(tier) => sel.active.push(Selected {
                id: l.id.clone(),
                tier,
                lower: tracks.get(&l.id).copied().unwrap_or_default().reliability(policy).lower,
            }),
            None => sel.out_of_scope.push(l.id.clone()),
        }
    }
    sel.active.sort_by(|a, b| b.tier.cmp(&a.tier).then(b.lower.total_cmp(&a.lower)).then(a.id.cmp(&b.id)));
    sel.active.truncate(policy.max_active);
    sel
}

/// One cartridge's answer: scores over the same candidates, and its own confidence in [0, 1].
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    pub cartridge: String,
    pub scores: Vec<f64>,
    pub confidence: f64,
}
/// The three outcomes of the disagreement rule.
#[derive(Clone, Debug, PartialEq)]
pub enum Resolved {
    /// Reliability-weighted scores, with each cartridge's weight.
    Combine { scores: Vec<f64>, weights: Vec<(String, f64)> },
    /// One cartridge's answer is used.
    Controls { cartridge: String },
    /// Neither is reliable: read the source again.
    ReRead,
}
/// The resolution, with both answers kept side by side (conflicts are kept, never merged away).
#[derive(Clone, Debug, PartialEq)]
pub struct Resolution {
    pub resolved: Resolved,
    pub kept: Vec<Answer>,
    pub reason: String,
}
impl Resolution {
    #[must_use]
    pub const fn outcome(&self) -> &'static str {
        match self.resolved {
            Resolved::Combine { .. } => "combine",
            Resolved::Controls { .. } => "controls",
            Resolved::ReRead => "re-read",
        }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut pairs = vec![("outcome", s(self.outcome())), ("reason", s(&self.reason))];
        match &self.resolved {
            Resolved::Combine { scores, weights } => {
                pairs.push(("scores", Value::Array(scores.iter().map(|x| num_f(*x)).collect())));
                pairs.push((
                    "weights",
                    Value::Array(weights.iter().map(|(id, w)| obj(vec![("id", s(id)), ("weight", num_f(*w))])).collect()),
                ));
            }
            Resolved::Controls { cartridge } => pairs.push(("cartridge", s(cartridge))),
            Resolved::ReRead => {}
        }
        pairs.push((
            "kept",
            Value::Array(
                self.kept
                    .iter()
                    .map(|a| {
                        obj(vec![
                            ("cartridge", s(&a.cartridge)),
                            ("confidence", num_f(a.confidence)),
                            ("scores", Value::Array(a.scores.iter().map(|x| num_f(*x)).collect())),
                        ])
                    })
                    .collect(),
            ),
        ));
        obj(pairs)
    }
}
fn argmax(x: &[f64]) -> Option<usize> {
    x.iter()
        .enumerate()
        .fold(None, |best: Option<(usize, f64)>, (i, v)| match best {
            Some((_, b)) if b >= *v => best,
            _ => Some((i, *v)),
        })
        .map(|(i, _)| i)
}

/// The disagreement rule over two answers. Each cartridge's interval comes from its track
/// record, scaled by its own confidence; it is reliable when its scaled lower bound is above
/// `bar` (the base path's recorded success rate).
/// - Neither reliable, or the candidate sets differ: re-read.
/// - Same top candidate: combine the reliable answers, each weighted by confidence over
///   variance.
/// - Different top candidate, one reliable: it controls.
/// - Different top candidate, both reliable: the one whose scaled lower bound is above the
///   other's upper bound controls (the less uncertain wins); overlapping intervals combine.
#[must_use]
pub fn resolve(a: &Answer, b: &Answer, tracks: &BTreeMap<String, Track>, bar: &Threshold, policy: &Policy) -> Resolution {
    let kept = vec![a.clone(), b.clone()];
    let rel = |x: &Answer| {
        let r = tracks.get(&x.cartridge).copied().unwrap_or_default().reliability(policy);
        let c = x.confidence.clamp(0.0, 1.0);
        (r, r.lower * c, c / (r.std * r.std).max(f64::MIN_POSITIVE))
    };
    let ((ra, la, wa), (rb, lb, wb)) = (rel(a), rel(b));
    let (oka, okb) = (la > bar.value, lb > bar.value);
    let done = |resolved, reason: String| Resolution { resolved, kept: kept.clone(), reason };
    if a.scores.len() != b.scores.len() {
        return done(Resolved::ReRead, "the two answers score different candidate sets".into());
    }
    if !oka && !okb {
        return done(
            Resolved::ReRead,
            format!("neither reliable: lower bounds {la:.3} and {lb:.3}, bar {:.3} ({})", bar.value, bar.source),
        );
    }
    let combine = |ws: Vec<(&Answer, f64)>| {
        let total: f64 = ws.iter().map(|(_, w)| w).sum();
        let scores =
            (0..a.scores.len()).map(|i| ws.iter().map(|(x, w)| x.scores.get(i).copied().unwrap_or(0.0) * w / total).sum()).collect();
        Resolved::Combine { scores, weights: ws.iter().map(|(x, w)| (x.cartridge.clone(), w / total)).collect() }
    };
    let both = || combine([(a, wa, oka), (b, wb, okb)].into_iter().filter(|(_, _, ok)| *ok).map(|(x, w, _)| (x, w)).collect());
    if argmax(&a.scores) == argmax(&b.scores) {
        return done(both(), "same top candidate".into());
    }
    match (oka, okb) {
        (true, false) => done(
            Resolved::Controls { cartridge: a.cartridge.clone() },
            format!("only {} reliable ({la:.3} over bar {:.3})", a.cartridge, bar.value),
        ),
        (false, true) => done(
            Resolved::Controls { cartridge: b.cartridge.clone() },
            format!("only {} reliable ({lb:.3} over bar {:.3})", b.cartridge, bar.value),
        ),
        _ if la > rb.upper => done(
            Resolved::Controls { cartridge: a.cartridge.clone() },
            format!("{} less uncertain: {la:.3} above {:.3}", a.cartridge, rb.upper),
        ),
        _ if lb > ra.upper => done(
            Resolved::Controls { cartridge: b.cartridge.clone() },
            format!("{} less uncertain: {lb:.3} above {:.3}", b.cartridge, ra.upper),
        ),
        _ => done(both(), "both reliable, intervals overlap: weighted by reliability".into()),
    }
}

const H0: &str = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b";
const H1: &str = "0000000000000000000000000000000000000000000000000000000000000001";

/// Self-check of `cartridge`: the design file gives the policy; a label round-trips and a
/// muscle memory pack is refused; a mismatched base and a missing component are refused by
/// name; units listed as cartridges carry the right labels (a constructed registry, then every
/// card of the live registry, read only); selection keeps the classes apart, orders by tier;
/// the disagreement rule returns each of its three outcomes on constructed cases.
pub(crate) fn check() -> Result<(), String> {
    let f = |m: String| Err(format!("nmlearn.cartridge: {m}"));
    let root = nmtype::nano_root();
    let policy = Policy::load(&root);
    if policy.source != DESIGN {
        return f(format!("policy not read from {DESIGN}: {}", policy.source));
    }
    let mut fb = Policy::fallback("check");
    fb.source.clone_from(&policy.source);
    if fb != policy {
        return f("design file and stated fallback disagree".into());
    }
    // Units through the adapter, on a constructed registry.
    let home = nmtype::check_dir("nmlearn-cartridge").map_err(|e| e.to_string())?;
    let mut units = Units::open(&home);
    let mut a = UnitCard::new("lwu-ra", "qwen3-0.6b", H0, "reranker", "nmengn.components#L27.down", 8, 1);
    a.registered_at = 1_794_125_375_751;
    units.register(a).map_err(|e| e.to_string())?;
    units
        .register(UnitCard::new("lwu-rb", "qwen3-0.6b", H0, "reranker", "nmengn.components#final_norm", 0, 2))
        .map_err(|e| e.to_string())?;
    let h = units.commit_file("lwu-ra", "up.q8", b"NMQ8").map_err(|e| e.to_string())?;
    units.set_files("lwu-ra", vec![("up.q8".into(), h)], "fitted").map_err(|e| e.to_string())?;
    units.commit_file("lwu-rb", "verdict.json", br#"{"accepted_scopes":["reranker:where-fn"]}"#).map_err(|e| e.to_string())?;
    let labels = units_as_cartridges(&units, &policy);
    let (la, lb) = (labels.first().ok_or("no label")?, labels.get(1).ok_or("no label")?);
    let want_a = la.kind == Kind::LearningUnit
        && la.data_class == DataClass::Behavioural
        && la.quantization == "nmq8-row-f16"
        && la.check == "accept"
        && la.date == "2026-11-08"
        && la.components == ["nmengn.components#L27.down"]
        && la.scopes == ["reranker"];
    let want_b = lb.quantization == "none" && lb.check == "none" && lb.scopes == ["reranker:where-fn"];
    if !want_a || !want_b || labels.iter().any(|l| l.validate(&policy).is_err()) {
        return f(format!("unit labels: {la:?} / {lb:?}"));
    }
    let back = Label::from_value(&la.to_value(), &policy).map_err(|e| e.to_string())?;
    if back != *la {
        return f("label round trip".into());
    }
    let mut pack = la.to_value();
    if let Value::Object(m) = &mut pack {
        m.insert("kind".into(), s("muscle-memory-pack"));
    }
    if !matches!(Label::from_value(&pack, &policy), Err(CartFault::NotACartridge(_))) {
        return f("a muscle memory pack was accepted as a cartridge".into());
    }
    // Every live unit lists with a valid label (read only).
    let live = Units::open(&root);
    let live_labels = units_as_cartridges(&live, &policy);
    if live_labels.len() != live.cards.len()
        || live_labels.iter().zip(&live.cards).any(|(l, c)| {
            l.id != c.id || l.base_sha256 != c.base_sha256 || l.components != [c.target.clone()] || l.validate(&policy).is_err()
        })
    {
        return f("live units do not all list as valid cartridges".into());
    }
    // Slot check.
    let loaded = Loaded::population(&root, "qwen3-0.6b", vec!["nmengn.components#*".into()]).map_err(|e| e.to_string())?;
    if loaded.base_sha256 != H0 || slot_check(la, &loaded).is_err() {
        return f("slot check refused a matching cartridge".into());
    }
    let mut other = la.clone();
    other.base_sha256 = H1.into();
    let narrow = Loaded { components: vec!["nmengn.components#L0.q".into()], ..loaded.clone() };
    match (slot_check(&other, &loaded), slot_check(la, &narrow)) {
        (Err(CartFault::BaseMismatch { .. }), Err(CartFault::ComponentNotLoaded { .. })) => {}
        x => return f(format!("mismatched cartridge not refused by name: {x:?}")),
    }
    if plug(&mut units, &other, &loaded).is_ok() || units.get("lwu-ra").is_some_and(|c| c.loaded) {
        return f("plug loaded a mismatched cartridge".into());
    }
    plug(&mut units, la, &loaded).map_err(|e| e.to_string())?;
    unplug(&mut units, "lwu-ra").map_err(|e| e.to_string())?;
    // Selection.
    let mut cache = la.clone();
    cache.id = "cache-x".into();
    cache.kind = Kind::EncodedCache;
    cache.data_class = DataClass::Sensory;
    let mut wrong = la.clone();
    wrong.id = "lwu-wrong".into();
    wrong.base_sha256 = H1.into();
    let all = vec![la.clone(), lb.clone(), cache, wrong];
    let none = BTreeMap::new();
    let sel =
        select(&Request { scope: "reranker:where-fn:nano".into(), data_class: DataClass::Behavioural }, &all, &loaded, &none, &policy);
    let ids: Vec<&str> = sel.active.iter().map(|x| x.id.as_str()).collect();
    if ids != ["lwu-rb", "lwu-ra"]
        || sel.refused.len() != 2
        || !sel.refused.iter().any(|(_, e)| e.code() == "cart-class-mismatch")
        || !sel.refused.iter().any(|(_, e)| e.code() == "cart-base-mismatch")
    {
        return f(format!("selection {:?}", sel.to_value().encode()));
    }
    if scope_tier("reranker:where-fn:nano", "reranker:value").is_some() || scope_tier("reranker:where-fn:nano", "*:where-fn") != Some(1) {
        return f("scope tiers".into());
    }
    // The three outcomes.
    let ev = |id: &'static str, right: u64, total: u64| (0..total).map(move |i| (id.to_owned(), i < right));
    let tr = tracks_from(
        &ev("strong", 90, 100)
            .chain(ev("good", 85, 100))
            .chain(ev("weak", 10, 20))
            .chain(ev("thin", 5, 10))
            .chain(ev("thin2", 6, 12))
            .collect::<Vec<_>>(),
    );
    let base = vec![
        Outcome {
            at: 1,
            slot: "reranker".into(),
            task: "t".into(),
            query: "q".into(),
            answer: "a".into(),
            correct: true,
            tokens: 1,
            verified: true,
        },
        Outcome {
            at: 2,
            slot: "reranker".into(),
            task: "t".into(),
            query: "q".into(),
            answer: "b".into(),
            correct: false,
            tokens: 1,
            verified: true,
        },
    ];
    let bar = threshold(&base, "reranker", &policy);
    let ans = |id: &str, top: usize| Answer {
        cartridge: id.into(),
        scores: (0..3).map(|i| if i == top { 1.0 } else { 0.0 }).collect(),
        confidence: 1.0,
    };
    let outs = [
        resolve(&ans("strong", 0), &ans("good", 0), &tr, &bar, &policy),
        resolve(&ans("strong", 0), &ans("weak", 1), &tr, &bar, &policy),
        resolve(&ans("thin", 0), &ans("thin2", 1), &tr, &bar, &policy),
        resolve(&ans("strong", 0), &ans("good", 2), &tr, &bar, &policy),
    ];
    let got: Vec<&str> = outs.iter().map(Resolution::outcome).collect();
    if (bar.value - 0.5).abs() > 1e-12 || got != ["combine", "controls", "re-read", "combine"] || outs.iter().any(|r| r.kept.len() != 2) {
        return f(format!("disagreement rule gave {got:?}, bar {}", bar.value));
    }
    if !matches!(&outs[1].resolved, Resolved::Controls { cartridge } if cartridge == "strong") {
        return f("the reliable cartridge did not control".into());
    }
    let mut high = tr.clone();
    high.insert("good".into(), Track { trials: 1000, successes: 600 });
    let sep = resolve(&ans("strong", 0), &ans("good", 2), &high, &bar, &policy);
    if !matches!(&sep.resolved, Resolved::Controls { cartridge } if cartridge == "strong") {
        return f(format!("separated intervals gave {}", sep.outcome()));
    }
    Ok(())
}
