//! Reload path: load a pack by fingerprint, verify header and base hash, refuse a withdrawn
//! pack, and hand its sequence to G3's reasoning layer, which returns the control messages to
//! send downward. The first-form layer replays the chunk as proposed instructions; a later
//! layer may adapt, shorten or decline it, behind the same trait.
use crate::pack::Pack;
use crate::store::MuscleStore;
use crate::symbol::MessageKind;
use crate::{Fault, Result};

/// One control message for the levels below G3. Behavioural by construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlMessage {
    /// Kind of the original message being reproduced.
    pub kind: MessageKind,
    /// Descriptor of the message (target part and action).
    pub descriptor: String,
    /// Position in the replayed chunk.
    pub step: usize,
    /// Fingerprint of the pack it came from.
    pub pack: String,
    /// True while the message is a proposal that G2 or the receiving part may decline.
    pub proposed: bool,
}

/// G3's reasoning layer: turns a reloaded pack into downward control.
pub trait G3Reasoning {
    fn control(&mut self, pack: &Pack) -> Vec<ControlMessage>;
}

/// First form: replay the chunk in order, each step as a proposed message.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReplayLayer;
impl G3Reasoning for ReplayLayer {
    fn control(&mut self, pack: &Pack) -> Vec<ControlMessage> {
        pack.steps()
            .into_iter()
            .enumerate()
            .map(|(step, (kind, descriptor))| ControlMessage {
                kind,
                descriptor,
                step,
                pack: pack.header.fingerprint.clone(),
                proposed: true,
            })
            .collect()
    }
}

/// Load, verify and apply one pack through the reasoning layer.
/// # Errors
/// Faults of `MuscleStore::load`; `nmmusc-withdrawn` for a withdrawn pack.
pub fn reload(store: &MuscleStore, fp: &str, base_hash: &str, layer: &mut impl G3Reasoning) -> Result<Vec<ControlMessage>> {
    let p = store.load(fp, base_hash)?;
    if p.header.withdrawn {
        return Err(Fault::new("nmmusc-withdrawn", format!("pack {fp} is withdrawn and not applied")));
    }
    Ok(layer.control(&p))
}

/// Section check: a stored pack replays as its exact message sequence; after recorded gains
/// collapse it is withdrawn, kept on disk and no longer applied.
/// # Errors
/// One fault line.
pub fn check() -> std::result::Result<(), String> {
    let dir = crate::scratch_dir("reload").map_err(|e| e.to_string())?;
    let st = MuscleStore::open(&dir).map_err(|e| e.to_string())?;
    let p = crate::pack::sample("base-a").map_err(|e| e.to_string())?;
    let fp = p.header.fingerprint.clone();
    st.write(&p).map_err(|e| e.to_string())?;
    let ctl = reload(&st, &fp, "base-a", &mut ReplayLayer).map_err(|e| e.to_string())?;
    let want: Vec<String> = p.steps().into_iter().map(|(_, d)| d).collect();
    let got: Vec<String> = ctl.iter().map(|c| c.descriptor.clone()).collect();
    if got != want || ctl.len() != 4 {
        return Err(format!("replay differs: {got:?}"));
    }
    let z = crate::Policy::default().withdraw_z;
    let mut withdrawn = false;
    // Collapsed gains (about 0 against a promotion gain of 0.15, per-use spread about 0.14 from
    // the promotion record): the change rule withdraws the pack once the evidence is enough.
    for x in [0.0, -0.01, 0.01, 0.0].into_iter().cycle().take(20) {
        withdrawn = st.record_gain(&fp, "base-a", x, z).map_err(|e| e.to_string())?.withdrawn;
        if withdrawn {
            break;
        }
    }
    if !withdrawn {
        return Err("collapsed gain did not withdraw the pack".to_owned());
    }
    match reload(&st, &fp, "base-a", &mut ReplayLayer) {
        Err(e) if e.code == "nmmusc-withdrawn" => {}
        other => return Err(format!("withdrawn pack applied: {other:?}")),
    }
    if !dir.join(format!("{fp}.nmmusc")).is_file() {
        return Err("withdrawn pack not kept on disk".to_owned());
    }
    let idx = st.index().map_err(|e| e.to_string())?;
    if !idx.iter().any(|r| r.fingerprint == fp && r.withdrawn) {
        return Err("index does not mark the pack withdrawn".to_owned());
    }
    Ok(())
}
