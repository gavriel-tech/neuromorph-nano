//! A behavioural message as a symbol. The data-class tag and message envelope that another
//! step is adding to `nmtype` plug in here later: implement `BehaviouralSymbol` for the
//! `nmtype` envelope (class from its data-class tag, kind and descriptor from its fields) and
//! the rest of this crate takes it unchanged. Until then `Message` is the local stand-in.
use crate::{Fault, Result};
use std::collections::HashMap;

/// The two segregated classes of encoded data (specification 5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataClass {
    /// Code base content and its encodings. Never enters muscle memory.
    Sensory,
    /// Internal communication between the system's own parts.
    Behavioural,
}

/// Kind of behavioural message (specification 5.1 and 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MessageKind {
    Instruction,
    Route,
    Reflex,
    Report,
    Request,
}
impl MessageKind {
    /// The organism's one message vocabulary (2026-11-12, motor MO1: muscle memory reads the
    /// message kinds as a view): a route is an instruction to send something on.
    #[must_use]
    pub const fn as_message(self) -> nmtype::MessageKind {
        match self {
            Self::Instruction | Self::Route => nmtype::MessageKind::Instruction,
            Self::Reflex => nmtype::MessageKind::Reflex,
            Self::Report => nmtype::MessageKind::Report,
            Self::Request => nmtype::MessageKind::Request,
        }
    }
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Instruction => "instruction",
            Self::Route => "route",
            Self::Reflex => "reflex",
            Self::Report => "report",
            Self::Request => "request",
        }
    }
    /// # Errors
    /// An unknown kind name.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "instruction" => Self::Instruction,
            "route" => Self::Route,
            "reflex" => Self::Reflex,
            "report" => Self::Report,
            "request" => Self::Request,
            _ => return Err(Fault::new("nmmusc-bad-kind", format!("unknown message kind {s:?}"))),
        })
    }
}

/// What G3 needs from any message to log it as a symbol.
pub trait BehaviouralSymbol {
    /// Data class set where the message was made.
    fn data_class(&self) -> DataClass;
    /// Message kind.
    fn kind(&self) -> MessageKind;
    /// Stable descriptor naming the message's effect (target part and action, no sensory
    /// content); readable as language, so a pack converts back to words.
    fn descriptor(&self) -> String;
}

/// Local stand-in for the `nmtype` message envelope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub class: DataClass,
    pub kind: MessageKind,
    /// Receiving part, for example `g0_t3` or `tentacle:4`.
    pub target: String,
    /// Action, for example `focus:tests` or `forward:packet`.
    pub action: String,
}
impl Message {
    #[must_use]
    pub fn behavioural(kind: MessageKind, target: &str, action: &str) -> Self {
        Self { class: DataClass::Behavioural, kind, target: target.to_owned(), action: action.to_owned() }
    }
}
impl BehaviouralSymbol for Message {
    fn data_class(&self) -> DataClass {
        self.class
    }
    fn kind(&self) -> MessageKind {
        self.kind
    }
    fn descriptor(&self) -> String {
        format!("{} {} {}", self.kind.as_str(), self.target, self.action)
    }
}

/// One codebook entry: the kind and descriptor behind a code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub kind: MessageKind,
    pub descriptor: String,
}

/// Maps behavioural messages to compact codes, first seen first numbered. Holds behavioural
/// entries only.
#[derive(Clone, Debug, Default)]
pub struct Codebook {
    by_descriptor: HashMap<String, u32>,
    entries: Vec<Entry>,
}
impl Codebook {
    /// Code for a message, adding it if new.
    /// # Errors
    /// `nmmusc-sensory-refused` for a sensory item; `nmmusc-alphabet-full` past `u32`.
    pub fn code(&mut self, m: &impl BehaviouralSymbol) -> Result<u32> {
        if m.data_class() != DataClass::Behavioural {
            return Err(Fault::new(
                "nmmusc-sensory-refused",
                format!("sensory item refused by the behaviour codebook: {}", m.kind().as_str()),
            ));
        }
        let d = m.descriptor();
        if let Some(&c) = self.by_descriptor.get(&d) {
            return Ok(c);
        }
        let c = u32::try_from(self.entries.len()).map_err(|_| Fault::new("nmmusc-alphabet-full", "more than u32 symbols"))?;
        self.by_descriptor.insert(d.clone(), c);
        self.entries.push(Entry { kind: m.kind(), descriptor: d });
        Ok(c)
    }
    #[must_use]
    pub fn entry(&self, code: u32) -> Option<&Entry> {
        self.entries.get(usize::try_from(code).ok()?)
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Section check: codes are stable and a sensory item is refused with its named fault.
/// # Errors
/// One fault line.
pub fn check() -> std::result::Result<(), String> {
    for k in [MessageKind::Instruction, MessageKind::Route, MessageKind::Reflex, MessageKind::Report, MessageKind::Request] {
        if nmtype::MessageKind::from_name(k.as_message().name()).is_none() {
            return Err(format!("nmmusc.symbol: {} has no verb in the one vocabulary", k.as_str()));
        }
    }
    let mut cb = Codebook::default();
    let a = Message::behavioural(MessageKind::Instruction, "g0_t1", "focus:tests");
    let b = Message::behavioural(MessageKind::Route, "g1", "forward:g2");
    let (ca, cb_, ca2) = (cb.code(&a), cb.code(&b), cb.code(&a));
    if ca != Ok(0) || cb_ != Ok(1) || ca2 != Ok(0) {
        return Err(format!("codes not stable: {ca:?} {cb_:?} {ca2:?}"));
    }
    let s = Message { class: DataClass::Sensory, ..a };
    match cb.code(&s) {
        Err(f) if f.code == "nmmusc-sensory-refused" => {}
        other => return Err(format!("sensory item not refused: {other:?}")),
    }
    if cb.len() != 2 {
        return Err("sensory item entered the codebook".to_owned());
    }
    Ok(())
}
