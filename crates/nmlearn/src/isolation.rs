//! Bit-for-bit isolation (PLAN 4c `isolation`; guide 10.1): a unit loaded and unloaded
//! around the same probes must leave every output byte equal, and the base cards unchanged.
//! The engine runs the probes; this module holds the comparison and the byte forms.

/// The byte form of an f32 output (little-endian, exact).
#[must_use]
pub fn bytes_of(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// One probe's outputs: a label and the bytes it produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Probe {
    pub label: String,
    pub bytes: Vec<u8>,
}

/// Compare three probe sets (no unit, unit loaded, unit unloaded again) and two card lists.
/// Returns whether all agree and one line saying what differed.
#[must_use]
pub fn isolation(before: &[Probe], loaded: &[Probe], after: &[Probe], cards_before: &[String], cards_after: &[String]) -> (bool, String) {
    let diff =
        |a: &[Probe], b: &[Probe]| a.iter().zip(b).filter(|(x, y)| x.bytes != y.bytes).map(|(x, _)| x.label.clone()).collect::<Vec<_>>();
    let (dl, da) = (diff(before, loaded), diff(before, after));
    let same_cards = cards_before == cards_after;
    let ok = dl.is_empty() && da.is_empty() && same_cards && before.len() == loaded.len() && before.len() == after.len();
    (
        ok,
        format!(
            "{} probes; loaded differs on {:?}; after unload differs on {:?}; base cards unchanged: {same_cards}",
            before.len(),
            dl,
            da
        ),
    )
}

/// Self-check of `isolation`: equal probes pass; one changed byte names its probe; a changed
/// card list fails.
pub(crate) fn check() -> Result<(), String> {
    let p = |l: &str, b: &[u8]| Probe { label: l.into(), bytes: b.to_vec() };
    let a = vec![p("s0", &[1, 2]), p("s1", &[3, 4])];
    let cards = vec!["nmengn@1@x".to_owned()];
    if !isolation(&a, &a, &a, &cards, &cards).0 {
        return Err("nmlearn.isolation: equal probes failed".into());
    }
    let b = vec![p("s0", &[1, 2]), p("s1", &[3, 5])];
    let (ok, why) = isolation(&a, &b, &a, &cards, &cards);
    if ok || !why.contains("[\"s1\"]") {
        return Err(format!("nmlearn.isolation: {why}"));
    }
    if isolation(&a, &a, &a, &cards, &["other".to_owned()]).0 || bytes_of(&[1.0]) != 1.0_f32.to_le_bytes() {
        return Err("nmlearn.isolation: cards or bytes".into());
    }
    Ok(())
}
