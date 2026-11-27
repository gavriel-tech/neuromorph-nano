//! The configuration record of a call (WP6; master design 2.3 steps 3, 5, 7, 14, 15 and 3.7 rule
//! 12; lane M, 2026-11-19): kind, aspects, lenses, orders, chain, cartridges plugged, budgets and
//! outcome, written per call. Behaviour only: every field is a short label or a number, never the
//! request's words or the material (a path, a sentence or code is refused). It renders as behaviour
//! steps for G3's record (`g3::Record::observed`), which G3 mines, and as a value for the call's
//! diagnostics and audit row; the closed record (with its verified outcome) is the case G2's chain
//! choice and G3 learn from.
use crate::{Fault, Result};
use nmjson::Value;

/// The longest label a field may hold.
pub const LABEL_MAX: usize = 48;

fn label(what: &str, x: &str) -> Result<()> {
    let ok = !x.is_empty() && x.len() <= LABEL_MAX && x.chars().all(|c| c.is_ascii_alphanumeric() || "-_.:+#".contains(c));
    if ok { Ok(()) } else { Err(Fault::new("nmmusc-sensory-refused", format!("configuration {what} {x:?} is not a short label"))) }
}

/// One call's configuration record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CallConfig {
    pub call: String,
    /// The request's kind.
    pub kind: String,
    /// The request's aspects, as kind labels (`definition`, `callers`, ...), never their words.
    pub aspects: Vec<String>,
    /// The lenses (perspectives) the tentacles read with.
    pub lenses: Vec<String>,
    /// Orders issued: verb and whether it landed.
    pub orders: Vec<(String, bool)>,
    /// The chain: hop labels in order (`code-close:embedder`, ...).
    pub chain: Vec<String>,
    /// Cartridges plugged, by name.
    pub cartridges: Vec<String>,
    /// Budgets: the pack's tokens and the call's milliseconds.
    pub budget_tokens: usize,
    pub budget_ms: u64,
    /// Verified outcome in [0, 1] once known.
    pub outcome: Option<f64>,
}

/// A size bucket for a budget (so records recur across calls): `<=1k`, `<=4k`, `<=16k`, `>16k`.
#[must_use]
pub fn bucket(n: usize) -> &'static str {
    match n {
        0..=1_000 => "le1k",
        1_001..=4_000 => "le4k",
        4_001..=16_000 => "le16k",
        _ => "gt16k",
    }
}

impl CallConfig {
    /// Check every field is a label.
    /// # Errors
    /// `nmmusc-sensory-refused` for the first field that is not.
    pub fn validate(&self) -> Result<()> {
        label("call", &self.call)?;
        label("kind", &self.kind)?;
        for (w, xs) in [("aspect", &self.aspects), ("lens", &self.lenses), ("chain", &self.chain), ("cartridge", &self.cartridges)] {
            for x in xs {
                label(w, x)?;
            }
        }
        for (v, _) in &self.orders {
            label("order", v)?;
        }
        if self.outcome.is_some_and(|o| !(0.0..=1.0).contains(&o)) {
            return Err(Fault::new("nmmusc-bad-config", "outcome outside 0 to 1"));
        }
        Ok(())
    }
    /// The behaviour steps for G3's record.
    /// # Errors
    /// As `validate`.
    pub fn steps(&self) -> Result<Vec<String>> {
        self.validate()?;
        let mut out = vec![format!("config:kind:{}", self.kind)];
        out.extend(self.aspects.iter().map(|a| format!("config:aspect:{a}")));
        out.extend(self.lenses.iter().map(|l| format!("config:lens:{l}")));
        out.extend(self.orders.iter().map(|(v, l)| format!("config:order:{v}:{}", if *l { "landed" } else { "missed" })));
        out.extend(self.chain.iter().map(|c| format!("config:chain:{c}")));
        out.extend(self.cartridges.iter().map(|c| format!("config:cartridge:{c}")));
        out.push(format!("config:budget:{}", bucket(self.budget_tokens)));
        Ok(out)
    }
    /// The record as a value (diagnostics, audit row).
    /// # Errors
    /// As `validate`.
    pub fn to_value(&self) -> Result<Value> {
        self.validate()?;
        let s = |t: &str| Value::String(t.to_owned());
        let list = |xs: &[String]| Value::Array(xs.iter().map(|x| s(x)).collect());
        let mut f: Vec<(&str, Value)> = vec![
            ("call", s(&self.call)),
            ("kind", s(&self.kind)),
            ("aspects", list(&self.aspects)),
            ("lenses", list(&self.lenses)),
            ("orders", Value::Array(self.orders.iter().map(|(v, l)| s(&format!("{v}:{}", if *l { "landed" } else { "missed" }))).collect())),
            ("chain", list(&self.chain)),
            ("cartridges", list(&self.cartridges)),
            ("budget_tokens", Value::Number(self.budget_tokens.to_string())),
            ("budget_ms", Value::Number(self.budget_ms.to_string())),
        ];
        if let Some(o) = self.outcome {
            f.push(("outcome", Value::Number(format!("{o}"))));
        }
        Ok(Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()))
    }
}

/// Section check: a full record renders as steps and a value; a field holding material is refused;
/// budgets bucket.
/// # Errors
/// One fault line.
pub fn check() -> std::result::Result<(), String> {
    let c = CallConfig {
        call: "c1".into(),
        kind: "where".into(),
        aspects: vec!["definition".into(), "callers".into()],
        lenses: vec!["meaning".into()],
        orders: vec![("focus".into(), true), ("reshape-window".into(), false)],
        chain: vec!["code-close:embedder".into()],
        cartridges: vec!["code-terms".into()],
        budget_tokens: 4_000,
        budget_ms: 2_500,
        outcome: Some(1.0),
    };
    let st = c.steps().map_err(|f| f.detail)?;
    if st.len() != 9 || !st.contains(&"config:order:reshape-window:missed".to_owned()) || !st.contains(&"config:budget:le4k".to_owned()) || c.to_value().is_err() {
        return Err(format!("nmmusc.config: steps {st:?}"));
    }
    let bad = CallConfig { aspects: vec!["where is derive_limit".into()], ..c.clone() };
    let path = CallConfig { cartridges: vec!["src/a.rs".into()], ..c };
    if bad.steps().is_ok() || path.validate().is_ok() {
        return Err("nmmusc.config: material entered a configuration record".into());
    }
    Ok(())
}
