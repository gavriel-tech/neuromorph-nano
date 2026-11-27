//! Data classes (specification section 5.1, 2026-11-10): every encoded item is either
//! sensory (acquired for processing: the code base and its content) or behavioural
//! (internal communication: instructions, controlled reflexes, reports, requests, action
//! copies). The two are strictly segregated.
//!
//! Three layers of enforcement, from strongest to weakest:
//! 1. Types: `Sensory<T>` and `Behavioural<T>` are distinct wrappers. A function that takes
//!    `Behavioural<T>` cannot be given a `Sensory<T>`; there is no conversion from sensory to
//!    behavioural at all. The one conversion that exists, `Behavioural::into_sensory`, is
//!    explicit, needs a stated reason and is meant for the system's own output re-read as
//!    data (for example an applied edit read back by a tentacle).
//! 2. Tags: items that cross untyped paths (`Packet`, `Hop`, JSON) carry a `DataClass` tag
//!    through the `Classed` trait.
//! 3. Boundaries: every store and channel declares the one class it accepts as a `Boundary`
//!    and calls `Boundary::admit` (or `guard`), which refuses a wrong-class hand-over with
//!    the named fault `WRONG_CLASS`.
//!
//! ```
//! use nmtype::class::{Behavioural, Sensory};
//! fn muscle_memory_store(_item: Behavioural<String>) {}
//! muscle_memory_store(Behavioural::new("route t2 to the coder chain".to_owned()));
//! let _read = Sensory::new("fn main() {}".to_owned());
//! ```
//!
//! ```compile_fail
//! // The same store refuses sensory data at compile time:
//! use nmtype::class::{Behavioural, Sensory};
//! fn muscle_memory_store(_item: Behavioural<String>) {}
//! let read = Sensory::new("fn main() {}".to_owned());
//! muscle_memory_store(read); // sensory data can never reach a behavioural store
//! ```
use crate::component::{Fault, FaultKind};
use crate::wire::named_enum;

named_enum! {
    /// The class of one encoded item.
    pub enum DataClass in "class" { Sensory => "sensory", Behavioural => "behavioural" }
}
impl Default for DataClass {
    /// Sensory: everything the mini carried before the class tag existed was data read
    /// from the code base, so untagged items are sensory.
    fn default() -> Self {
        Self::Sensory
    }
}

/// The named fault code of a wrong-class hand-over (the fault message starts with it).
pub const WRONG_CLASS: &str = "nm-class-mismatch";

/// Anything that knows its own data class.
pub trait Classed {
    fn data_class(&self) -> DataClass;
}

/// An item of the sensory class. The field is private: the only way in is `Sensory::new`
/// (data acquired for processing) or `Behavioural::into_sensory` (checked).
#[derive(Clone, Debug, PartialEq)]
pub struct Sensory<T>(T);
impl<T> Sensory<T> {
    /// Mark data acquired for processing as sensory.
    pub const fn new(item: T) -> Self {
        Self(item)
    }
    pub const fn get(&self) -> &T {
        &self.0
    }
    pub fn into_inner(self) -> T {
        self.0
    }
}
impl<T> Classed for Sensory<T> {
    fn data_class(&self) -> DataClass {
        DataClass::Sensory
    }
}

/// An item of the behavioural class. No constructor takes a `Sensory` value.
#[derive(Clone, Debug, PartialEq)]
pub struct Behavioural<T>(T);
impl<T> Behavioural<T> {
    /// Mark a signal made by one of the system's own parts as behavioural.
    pub const fn new(item: T) -> Self {
        Self(item)
    }
    pub const fn get(&self) -> &T {
        &self.0
    }
    pub fn into_inner(self) -> T {
        self.0
    }
    /// The one checked conversion: the system's own output re-read as data (for example an
    /// applied edit, now part of the code base). Needs a non-empty one-line reason, which
    /// the caller records with the item.
    /// # Errors
    /// `WRONG_CLASS` fault when the reason is empty or spans lines.
    pub fn into_sensory(self, reason: &str) -> Result<Sensory<T>, Fault> {
        if reason.trim().is_empty() || reason.contains('\n') {
            return Err(Fault::new(
                "nmtype.class",
                FaultKind::Policy,
                format!("{WRONG_CLASS}: behavioural to sensory needs a one-line reason"),
            ));
        }
        Ok(Sensory(self.0))
    }
}
impl<T> Classed for Behavioural<T> {
    fn data_class(&self) -> DataClass {
        DataClass::Behavioural
    }
}

/// One store or channel and the single class it accepts. A store never holds both classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Boundary {
    pub name: &'static str,
    pub accepts: DataClass,
}
impl Boundary {
    #[must_use]
    pub const fn new(name: &'static str, accepts: DataClass) -> Self {
        Self { name, accepts }
    }
    /// Accept or refuse one item by class.
    /// # Errors
    /// `WRONG_CLASS` fault naming the boundary, the class it accepts and the class given.
    pub fn admit<T: Classed + ?Sized>(&self, item: &T) -> Result<(), Fault> {
        guard(self.name, self.accepts, item.data_class())
    }
    /// Accept a whole batch only if every item is of the right class (a store never takes a
    /// mixed batch in part).
    /// # Errors
    /// The first wrong-class item's fault, with its index.
    pub fn admit_all<T: Classed>(&self, items: &[T]) -> Result<(), Fault> {
        for (i, item) in items.iter().enumerate() {
            self.admit(item).map_err(|mut f| {
                f.message = format!("{} (item {i})", f.message);
                f
            })?;
        }
        Ok(())
    }
}

/// The boundaries named by the specification and the Globalized Context Stack design (2026-11-10). Other
/// stores declare their own `Boundary` the same way.
pub mod boundaries {
    use super::{Boundary, DataClass};
    /// Per-tentacle encoded read and drone readings (`g0_tx`).
    pub const G0_STORE: Boundary = Boundary::new("g0-store", DataClass::Sensory);
    /// G1's whole-code-base layer.
    pub const G1_CODEBASE: Boundary = Boundary::new("g1-codebase", DataClass::Sensory);
    /// Packets handed to a MicroModel chain for processing.
    pub const CHAIN_INPUT: Boundary = Boundary::new("chain-input", DataClass::Sensory);
    /// The Core Brain's evidence pool for the answer.
    pub const ANSWER_EVIDENCE: Boundary = Boundary::new("answer-evidence", DataClass::Sensory);
    /// Addressed in-house messages between parts and GCX levels.
    pub const MESSAGE_CHANNEL: Boundary = Boundary::new("message-channel", DataClass::Behavioural);
    /// The broadcast settings channel.
    pub const BROADCAST: Boundary = Boundary::new("broadcast", DataClass::Behavioural);
    /// Muscle memory packs: behavioural only (section 5.3).
    pub const MUSCLE_MEMORY: Boundary = Boundary::new("muscle-memory", DataClass::Behavioural);
    /// Upward copies of self-issued actions (G1 and G2).
    pub const ACTION_COPIES: Boundary = Boundary::new("action-copies", DataClass::Behavioural);
    /// Every named boundary, for tables and checks.
    pub const ALL: &[Boundary] =
        &[G0_STORE, G1_CODEBASE, CHAIN_INPUT, ANSWER_EVIDENCE, MESSAGE_CHANNEL, BROADCAST, MUSCLE_MEMORY, ACTION_COPIES];
}

/// The guard every store and channel calls: accept `got` at `boundary` only when it equals
/// `required`.
/// # Errors
/// Policy fault whose message starts with `WRONG_CLASS`.
pub fn guard(boundary: &str, required: DataClass, got: DataClass) -> Result<(), Fault> {
    if required == got {
        Ok(())
    } else {
        Err(Fault::new(
            "nmtype.class",
            FaultKind::Policy,
            format!("{WRONG_CLASS}: {boundary} accepts {} only, got {}", required.name(), got.name()),
        ))
    }
}

/// Test B0 (plan P5): a table of hand-overs, every item kind against every named boundary.
/// Every wrong-class hand-over must be refused with `WRONG_CLASS`, every right-class one
/// accepted. Returns (accepted, refused) counts.
/// # Errors
/// The first case that went the wrong way.
pub fn b0_table() -> Result<(usize, usize), String> {
    use crate::hop::Hop;
    use crate::message::{Envelope, MessageKind};
    use crate::packet::{Packet, Provenance};
    use crate::settings::Settings;
    let prov = Provenance::new("crates/nmtype/src/class.rs:1", "nmpod.unit#t1", 1, 1);
    let env = Envelope::new(MessageKind::Instruction, "nmbrain.g2#0", "nmpod.unit#t1", 0.5, "focus src", Vec::new(), Vec::new())
        .map_err(|e| e.to_string())?;
    // Each case: (label, class of the item as the boundary sees it).
    let items: Vec<(&str, Box<dyn Classed>)> = vec![
        ("packet (default)", Box::new(Packet::new("chunk", "k", "t", prov.clone()))),
        ("packet (behavioural)", Box::new(Packet::new_behavioural("report", "k", "t", prov))),
        ("hop (default)", Box::new(Hop::new("nmpod.unit#t1", 1, "read", 0))),
        ("hop (behavioural)", Box::new(Hop::new("nmbrain.g2#0", 1, "send", 0).with_class(DataClass::Behavioural))),
        ("sensory wrapper", Box::new(Sensory::new(vec![1_u32, 2, 3]))),
        ("behavioural wrapper", Box::new(Behavioural::new(vec![1_u32, 2, 3]))),
        ("envelope", Box::new(env.clone())),
        ("action copy", Box::new(env.action_copy("nmbrain.g1#0"))),
        ("settings", Box::new(Settings::default())),
    ];
    let (mut ok, mut refused) = (0, 0);
    for b in boundaries::ALL {
        for (label, item) in &items {
            let right = item.data_class() == b.accepts;
            match (right, b.admit(item.as_ref())) {
                (true, Ok(())) => ok += 1,
                (false, Err(f)) if f.message.starts_with(WRONG_CLASS) && f.kind == FaultKind::Policy => {
                    refused += 1;
                }
                (true, Err(f)) => return Err(format!("B0: {label} refused at {}: {f}", b.name)),
                (false, Ok(())) => return Err(format!("B0: {label} accepted at {}", b.name)),
                (false, Err(f)) => return Err(format!("B0: {label} at {}: wrong fault {f}", b.name)),
            }
        }
    }
    // A mixed batch is refused whole.
    let mixed: Vec<DataClass> = vec![DataClass::Behavioural, DataClass::Sensory];
    if boundaries::MUSCLE_MEMORY.admit_all(&mixed).is_ok() {
        return Err("B0: mixed batch accepted by muscle memory".into());
    }
    // The checked conversion refuses an empty reason and keeps the item on a stated one.
    if Behavioural::new(1_u8).into_sensory(" ").is_ok() {
        return Err("B0: behavioural to sensory accepted without a reason".into());
    }
    if Behavioural::new(7_u8).into_sensory("applied edit read back").map(Sensory::into_inner) != Ok(7) {
        return Err("B0: behavioural to sensory lost the item".into());
    }
    Ok((ok, refused))
}
impl Classed for DataClass {
    fn data_class(&self) -> DataClass {
        *self
    }
}

named_enum! {
    /// Where a sensory item came from: the request (a prompt or a question) or the material (what
    /// is crawled: code, documents, any data). The two are never confused or crossed.
    pub enum Origin in "origin" { Request => "request", Material => "material" }
}

/// The named fault code of a hand-over from the wrong origin (the fault message starts with it).
pub const WRONG_ORIGIN: &str = "nm-origin-mismatch";

/// Anything that knows whether it came from the request or from the material.
pub trait Sourced {
    fn origin(&self) -> Origin;
}
impl Sourced for Origin {
    fn origin(&self) -> Origin {
        *self
    }
}

/// An item taken from the request. The field is private and there is no conversion to or from
/// `FromMaterial`: a request item can never be handed on as material, nor the reverse.
#[derive(Clone, Debug, PartialEq)]
pub struct FromRequest<T>(T);
impl<T> FromRequest<T> {
    /// Mark an item taken from the request.
    pub const fn new(item: T) -> Self {
        Self(item)
    }
    pub const fn get(&self) -> &T {
        &self.0
    }
    pub fn into_inner(self) -> T {
        self.0
    }
}
impl<T> Sourced for FromRequest<T> {
    fn origin(&self) -> Origin {
        Origin::Request
    }
}

/// An item taken from the material (see `FromRequest`).
#[derive(Clone, Debug, PartialEq)]
pub struct FromMaterial<T>(T);
impl<T> FromMaterial<T> {
    /// Mark an item taken from the material.
    pub const fn new(item: T) -> Self {
        Self(item)
    }
    pub const fn get(&self) -> &T {
        &self.0
    }
    pub fn into_inner(self) -> T {
        self.0
    }
}
impl<T> Sourced for FromMaterial<T> {
    fn origin(&self) -> Origin {
        Origin::Material
    }
}

/// What one judged input compares the new request with: one item of one origin. An earlier request
/// is compared with the new request (request against request); lines of the material are compared
/// with the new request as material. No judged input holds both, because this holds one item.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Against<'a, T> {
    Request(&'a FromRequest<T>),
    Material(&'a FromMaterial<T>),
}
impl<T> Against<'_, T> {
    /// The item compared, whatever its origin.
    pub const fn item(&self) -> &T {
        match self {
            Self::Request(r) => r.get(),
            Self::Material(m) => m.get(),
        }
    }
}
impl<T> Sourced for Against<'_, T> {
    fn origin(&self) -> Origin {
        match self {
            Self::Request(_) => Origin::Request,
            Self::Material(_) => Origin::Material,
        }
    }
}

/// One hand-over point and the single origin it accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inlet {
    pub name: &'static str,
    pub accepts: Origin,
}
impl Inlet {
    #[must_use]
    pub const fn new(name: &'static str, accepts: Origin) -> Self {
        Self { name, accepts }
    }
    /// Accept or refuse one item by origin.
    /// # Errors
    /// `WRONG_ORIGIN` fault naming the inlet, the origin it accepts and the origin given.
    pub fn admit<T: Sourced + ?Sized>(&self, item: &T) -> Result<(), Fault> {
        let got = item.origin();
        if got == self.accepts {
            Ok(())
        } else {
            Err(Fault::new(
                "nmtype.class",
                FaultKind::Policy,
                format!("{WRONG_ORIGIN}: {} accepts {} items only, got a {} item", self.name, self.accepts.name(), got.name()),
            ))
        }
    }
    /// Accept a whole batch only if every item is of the right origin.
    /// # Errors
    /// The first wrong-origin item's fault, with its index.
    pub fn admit_all<T: Sourced>(&self, items: &[T]) -> Result<(), Fault> {
        for (i, item) in items.iter().enumerate() {
            self.admit(item).map_err(|mut f| {
                f.message = format!("{} (item {i})", f.message);
                f
            })?;
        }
        Ok(())
    }
}

/// The hand-over points of the sensory side, each taking one origin.
pub mod inlets {
    use super::{Inlet, Origin};
    /// The request's own drone: the request, never a piece of the material.
    pub const REQUEST_DRONE: Inlet = Inlet::new("request-drone", Origin::Request);
    /// The material's drones and the pieces they keep: material, never the request.
    pub const MATERIAL_DRONES: Inlet = Inlet::new("material-drones", Origin::Material);
    /// The substrate's pieces, symbols and indexes, and the reference graph built on them.
    pub const SUBSTRATE: Inlet = Inlet::new("substrate", Origin::Material);
    /// Every named inlet, for tables and checks.
    pub const ALL: &[Inlet] = &[REQUEST_DRONE, MATERIAL_DRONES, SUBSTRATE];
}

/// Every item origin against every named inlet: each wrong-origin hand-over refused with
/// `WRONG_ORIGIN`, each right one accepted, a mixed batch refused whole. Returns (accepted,
/// refused) counts.
/// # Errors
/// The first case that went the wrong way.
pub fn origin_table() -> Result<(usize, usize), String> {
    let items: Vec<(&str, Box<dyn Sourced>)> = vec![
        ("request item", Box::new(FromRequest::new("where is the limit set".to_owned()))),
        ("material item", Box::new(FromMaterial::new("pub const LIMIT: usize = 4;".to_owned()))),
    ];
    let (mut ok, mut refused) = (0, 0);
    for inlet in inlets::ALL {
        for (label, item) in &items {
            let right = item.origin() == inlet.accepts;
            match (right, inlet.admit(item.as_ref())) {
                (true, Ok(())) => ok += 1,
                (false, Err(f)) if f.message.starts_with(WRONG_ORIGIN) && f.kind == FaultKind::Policy => refused += 1,
                (true, Err(f)) => return Err(format!("origin: {label} refused at {}: {f}", inlet.name)),
                (false, Ok(())) => return Err(format!("origin: {label} accepted at {}", inlet.name)),
                (false, Err(f)) => return Err(format!("origin: {label} at {}: wrong fault {f}", inlet.name)),
            }
        }
    }
    if inlets::MATERIAL_DRONES.admit_all(&[Origin::Material, Origin::Request]).is_ok() {
        return Err("origin: a batch holding a request item accepted by the material drones".into());
    }
    Ok((ok, refused))
}

/// Self-check of the `class` sub-component: test B0 and the origin table.
pub(crate) fn check() -> Result<(), String> {
    let (ok, refused) = b0_table()?;
    if ok == 0 || refused == 0 {
        return Err(format!("nmtype.class: B0 table degenerate ({ok} accepted, {refused} refused)"));
    }
    let (ok, refused) = origin_table()?;
    if ok == 0 || refused == 0 {
        return Err(format!("nmtype.class: origin table degenerate ({ok} accepted, {refused} refused)"));
    }
    Ok(())
}
