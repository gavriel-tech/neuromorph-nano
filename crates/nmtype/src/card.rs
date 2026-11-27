//! Component cards (`part.json`) and part references (guide 7.1 identity, 8.1, 11.1
//! Component Registry). A card is flat JSON: the descriptor fields, config schema and
//! default, sub-components, lineage, source and `hash`. The hash is SHA-256 of the
//! canonical JSON of the card without its `hash` field (shape from `fwcore/prtdef`).
use crate::component::{ComponentDescriptor, Fault, FaultKind};
use crate::wire;
use nmjson::Value;
use std::path::Path;

/// `id@version@hash`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartRef {
    pub id: String,
    pub version: u64,
    pub hash: String,
}
impl PartRef {
    /// Parse `id@version@hash`.
    /// # Errors
    /// Rejects text without three `@`-separated parts or a non-numeric version.
    pub fn parse(text: &str) -> Result<Self, Fault> {
        let mut it = text.splitn(3, '@');
        match (it.next(), it.next(), it.next()) {
            (Some(id), Some(v), Some(hash)) if !id.is_empty() => Ok(Self {
                id: id.to_owned(),
                version: v.parse().map_err(|_| wire::bad("card", format!("version in {text:?}")))?,
                hash: hash.to_owned(),
            }),
            _ => Err(wire::bad("card", format!("part ref {text:?} is not id@version@hash"))),
        }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        wire::obj(vec![("id", wire::s(&self.id)), ("version", wire::u(self.version)), ("hash", wire::s(&self.hash))])
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        Ok(Self { id: wire::get_s(v, "card", "id")?, version: wire::get_u(v, "card", "version")?, hash: wire::get_s(v, "card", "hash")? })
    }
}
impl std::fmt::Display for PartRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}@{}", self.id, self.version, self.hash)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Card {
    pub descriptor: ComponentDescriptor,
    pub config_schema: Value,
    pub config_default: Value,
    pub subcomponents: Vec<PartRef>,
    pub lineage: Vec<PartRef>,
    pub source: String,
}

/// SHA-256 of the canonical JSON of `card` without its `hash` field.
#[must_use]
pub fn card_hash(card: &Value) -> String {
    let mut fields = card.object().cloned().unwrap_or_default();
    fields.remove("hash");
    let text = Value::Object(fields).encode().unwrap_or_default();
    nmhash::sha256_bytes(text.as_bytes())
}

impl Card {
    /// Flat card JSON including `hash` (the descriptor's `card_hash`).
    #[must_use]
    pub fn to_value(&self) -> Value {
        let refs = |r: &[PartRef]| Value::Array(r.iter().map(PartRef::to_value).collect());
        let mut fields = self.descriptor.fields();
        fields.extend([
            ("config_schema", self.config_schema.clone()),
            ("config_default", self.config_default.clone()),
            ("subcomponents", refs(&self.subcomponents)),
            ("lineage", refs(&self.lineage)),
            ("source", wire::s(&self.source)),
            ("hash", wire::s(&self.descriptor.card_hash)),
        ]);
        wire::obj(fields)
    }
    /// Read a card without checking its hash.
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "card";
        let refs = |key| -> Result<Vec<PartRef>, Fault> { wire::get_a(v, m, key)?.iter().map(PartRef::from_value).collect() };
        Ok(Self {
            descriptor: ComponentDescriptor::read(v, "hash")?,
            config_schema: wire::get_v(v, m, "config_schema")?.clone(),
            config_default: wire::get_v(v, m, "config_default")?.clone(),
            subcomponents: refs("subcomponents")?,
            lineage: refs("lineage")?,
            source: wire::get_s(v, m, "source")?,
        })
    }
    /// Read a card and require its stored `hash` to equal the recomputed one.
    /// # Errors
    /// Malformed card, or an integrity fault whose message carries the recomputed hash.
    pub fn verified(v: &Value) -> Result<Self, Fault> {
        let card = Self::from_value(v)?;
        let want = card_hash(v);
        if card.descriptor.card_hash == want {
            Ok(card)
        } else {
            Err(Fault::new(
                "nmtype.card",
                FaultKind::Integrity,
                format!("{}: stored hash {:?}, computed {want}", card.descriptor.id, card.descriptor.card_hash),
            ))
        }
    }
    /// Load and verify a `part.json`.
    /// # Errors
    /// Unreadable file, malformed JSON or card, or hash mismatch.
    pub fn load(path: &Path) -> Result<Self, Fault> {
        let bytes = std::fs::read(path).map_err(|e| Fault::new("nmtype.card", FaultKind::Resource, format!("{}: {e}", path.display())))?;
        let v = nmjson::parse(&bytes).map_err(|e| wire::bad("card", format!("{}: {e}", path.display())))?;
        Self::verified(&v)
    }
    /// This card as a reference.
    #[must_use]
    pub fn part_ref(&self) -> PartRef {
        PartRef { id: self.descriptor.id.clone(), version: self.descriptor.version, hash: self.descriptor.card_hash.clone() }
    }
}

/// Self-check of the `card` sub-component: the hash ignores key order and the `hash`
/// field, a sealed card verifies, a tampered card does not, a part ref round-trips.
pub(crate) fn check() -> Result<(), String> {
    let a = br#"{"id":"nmx.y","kind":"tool","version":1,"parent":null,"ports_in":[],"ports_out":[],"deps":[],"config_schema":{},"config_default":{},"subcomponents":[],"lineage":[],"source":"new","hash":""}"#;
    let b = br#"{"source":"new","hash":"x","lineage":[],"subcomponents":[],"config_default":{},"config_schema":{},"deps":[],"ports_out":[],"ports_in":[],"parent":null,"version":1,"kind":"tool","id":"nmx.y"}"#;
    let va = nmjson::parse(a).map_err(|e| e.to_string())?;
    let vb = nmjson::parse(b).map_err(|e| e.to_string())?;
    let h = card_hash(&va);
    if h != card_hash(&vb) || h.len() != 64 {
        return Err("nmtype.card: hash depends on key order or the hash field".into());
    }
    let mut card = Card::from_value(&va).map_err(|e| e.to_string())?;
    card.descriptor.card_hash = h;
    let sealed = card.to_value();
    Card::verified(&sealed).map_err(|e| e.to_string())?;
    card.source = "tampered".into();
    if Card::verified(&card.to_value()).is_ok() {
        return Err("nmtype.card: tampered card verified".into());
    }
    let r = PartRef::parse(&sealed_ref(&sealed)?).map_err(|e| e.to_string())?;
    if r.to_string() != sealed_ref(&sealed)? {
        return Err("nmtype.card: part ref round trip".into());
    }
    Ok(())
}
fn sealed_ref(v: &Value) -> Result<String, String> {
    Card::verified(v).map(|c| c.part_ref().to_string()).map_err(|e| e.to_string())
}
