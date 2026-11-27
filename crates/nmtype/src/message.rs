//! The in-house message envelope, first form (specification section 5.2; research note gap
//! 3, 2026-11-10). Behavioural class. A message is a typed envelope (kind, sender, target,
//! confidence, one-line description) plus references to what it is about: source spans
//! `path:start-end`, token id sequences, and encoding handles (content hash plus the hash of
//! the base weights that made the encoding, with the token ids always attached). Payloads
//! pass by reference and are never re-encoded; tokens are the exact, decodable truth and an
//! encoding is a cached view of them, so every message reads back exactly as English, code
//! or mathematics.
//!
//! Text form (`render`, `parse`), line based, version `nm-msg/1`:
//! ```text
//! nm-msg/1
//! kind <instruction|controlled-reflex|report|request|action-copy>
//! from <component id>
//! to <component id>
//! confidence <f32, shortest exact spelling>
//! description <one line of English>
//! label-tokens <ids separated by spaces>
//! [copy-of <sha256 of the original's bare text>]
//! ref span <path>:<start>-<end>
//! ref tokens <ids>
//! ref encoding <content sha256> <base-weights sha256> <ids>
//! end
//! ```
//! After `label-tokens` and after any `ref` line the renderer may insert a resolved block
//! `text <byte count>` followed by exactly that many bytes and a newline: the label's
//! tokens or the referenced source text, as the caller's `Resolver` gives it. `parse`
//! skips these blocks by length, so they may contain any text, and the envelope parsed back
//! from the full form equals the one rendered.
use crate::class::{Classed, DataClass};
use crate::component::{ComponentId, Fault, FaultKind};
use crate::wire::{self, named_enum};
use nmjson::Value;

named_enum! {
    /// What a message is for.
    pub enum MessageKind in "message" {
        Instruction => "instruction", Reflex => "controlled-reflex", Report => "report",
        Request => "request", ActionCopy => "action-copy",
    }
}

/// The text-form version line.
pub const FORMAT: &str = "nm-msg/1";

/// What a message points at. Nothing is copied or re-encoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reference {
    /// Source lines `start..=end` (1-based) of `path`.
    Span { path: String, start: u32, end: u32 },
    /// A token id sequence (generated text, a formula, a code fragment).
    Tokens(Vec<u32>),
    /// An encoding held in the content-addressed store: valid only for the base weights
    /// whose hash it names. The token ids it was made from are always attached.
    Encoding { content: String, base: String, tokens: Vec<u32> },
}
impl Reference {
    fn line(&self) -> String {
        match self {
            Self::Span { path, start, end } => format!("ref span {path}:{start}-{end}"),
            Self::Tokens(ids) => format!("ref tokens{}", ids_text(ids)),
            Self::Encoding { content, base, tokens } => format!("ref encoding {content} {base}{}", ids_text(tokens)),
        }
    }
    fn parse(rest: &str) -> Result<Self, Fault> {
        if let Some(r) = rest.strip_prefix("span ") {
            let (path, range) = r.rsplit_once(':').ok_or_else(|| bad("span without :"))?;
            let (a, b) = range.split_once('-').ok_or_else(|| bad("span without -"))?;
            return Ok(Self::Span {
                path: path.to_owned(),
                start: a.parse().map_err(|_| bad("span start"))?,
                end: b.parse().map_err(|_| bad("span end"))?,
            });
        }
        if let Some(r) = rest.strip_prefix("tokens") {
            return Ok(Self::Tokens(parse_ids(r)?));
        }
        if let Some(r) = rest.strip_prefix("encoding ") {
            let mut it = r.splitn(3, ' ');
            let content = it.next().unwrap_or_default().to_owned();
            let base = it.next().ok_or_else(|| bad("encoding without base hash"))?.to_owned();
            let tokens = parse_ids(&format!(" {}", it.next().unwrap_or_default()))?;
            return Ok(Self::Encoding { content, base, tokens });
        }
        Err(bad(format!("unknown reference {rest:?}")))
    }
    fn validate(&self) -> Result<(), Fault> {
        match self {
            Self::Span { path, start, end } => {
                if path.is_empty() || path.contains(['\n', '\r']) || *start == 0 || start > end {
                    return Err(bad(format!("bad span {path:?} {start}-{end}")));
                }
            }
            Self::Tokens(_) => {}
            Self::Encoding { content, base, tokens } => {
                if !is_sha256(content) || !is_sha256(base) {
                    return Err(bad("encoding handle needs content and base-weights sha256"));
                }
                if tokens.is_empty() {
                    return Err(bad("encoding handle without its token ids"));
                }
            }
        }
        Ok(())
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let ids = |t: &[u32]| Value::Array(t.iter().map(|i| wire::u(u64::from(*i))).collect());
        match self {
            Self::Span { path, start, end } => wire::obj(vec![
                ("ref", wire::s("span")),
                ("path", wire::s(path)),
                ("start", wire::u(u64::from(*start))),
                ("end", wire::u(u64::from(*end))),
            ]),
            Self::Tokens(t) => wire::obj(vec![("ref", wire::s("tokens")), ("tokens", ids(t))]),
            Self::Encoding { content, base, tokens } => wire::obj(vec![
                ("ref", wire::s("encoding")),
                ("content", wire::s(content)),
                ("base", wire::s(base)),
                ("tokens", ids(tokens)),
            ]),
        }
    }
    /// # Errors
    /// Rejects an unknown reference type or a missing field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "message";
        let ids = |v: &Value| -> Result<Vec<u32>, Fault> {
            wire::get_a(v, m, "tokens")?
                .iter()
                .map(|i| i.integer().ok().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| bad("token id")))
                .collect()
        };
        Ok(match wire::get_s(v, m, "ref")?.as_str() {
            "span" => Self::Span { path: wire::get_s(v, m, "path")?, start: wire::get_n(v, m, "start")?, end: wire::get_n(v, m, "end")? },
            "tokens" => Self::Tokens(ids(v)?),
            "encoding" => Self::Encoding { content: wire::get_s(v, m, "content")?, base: wire::get_s(v, m, "base")?, tokens: ids(v)? },
            other => return Err(bad(format!("unknown reference {other:?}"))),
        })
    }
}

/// Turns references back into exact text. Supplied by the caller (the tokenizer and the
/// source tree live in other crates). `Ok(None)` means "not resolvable here": the block is
/// left out of the rendering and the envelope itself is unchanged.
pub trait Resolver {
    /// Lines `start..=end` of `path`, exactly as stored.
    /// # Errors
    /// The span exists but could not be read.
    fn span(&self, path: &str, start: u32, end: u32) -> Result<Option<String>, Fault>;
    /// The text the token ids decode to.
    /// # Errors
    /// An id is outside the vocabulary.
    fn tokens(&self, ids: &[u32]) -> Result<Option<String>, Fault>;
}

/// The resolver that resolves nothing: rendering with it gives the bare envelope.
pub struct Bare;
impl Resolver for Bare {
    fn span(&self, _: &str, _: u32, _: u32) -> Result<Option<String>, Fault> {
        Ok(None)
    }
    fn tokens(&self, _: &[u32]) -> Result<Option<String>, Fault> {
        Ok(None)
    }
}

/// One in-house message. Immutable once made; always behavioural.
#[derive(Clone, Debug, PartialEq)]
pub struct Envelope {
    kind: MessageKind,
    sender: ComponentId,
    target: ComponentId,
    confidence: f32,
    description: String,
    refs: Vec<Reference>,
    label_tokens: Vec<u32>,
    copy_of: Option<String>,
}

impl Envelope {
    /// Make a message. `label_tokens` are the token ids of `description` under the base
    /// tokenizer (checked by `verify_label`). Confidence is clamped to 0..=1, non-finite 0.
    /// # Errors
    /// Empty or whitespace-bearing sender or target, a description that is empty or spans
    /// lines, or an invalid reference.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: MessageKind,
        sender: &str,
        target: &str,
        confidence: f32,
        description: &str,
        refs: Vec<Reference>,
        label_tokens: Vec<u32>,
    ) -> Result<Self, Fault> {
        for (what, id) in [("sender", sender), ("target", target)] {
            if id.is_empty() || id.contains(char::is_whitespace) {
                return Err(bad(format!("{what} must be one unit id, got {id:?}")));
            }
        }
        if description.trim().is_empty() || description.contains(['\n', '\r']) {
            return Err(bad("description must be one non-empty line"));
        }
        for r in &refs {
            r.validate()?;
        }
        let confidence = if confidence.is_finite() { confidence.clamp(0.0, 1.0) } else { 0.0 };
        Ok(Self {
            kind,
            sender: sender.to_owned(),
            target: target.to_owned(),
            confidence,
            description: description.to_owned(),
            refs,
            label_tokens,
            copy_of: None,
        })
    }

    /// The copy of a self-issued action sent upward to `observer` (G1 or G2), so the system
    /// does not read its own edits as new input. Same sender, description, references and
    /// label tokens; kind `action-copy`; `copy_of` names the original by its digest.
    #[must_use]
    pub fn action_copy(&self, observer: &str) -> Self {
        Self { kind: MessageKind::ActionCopy, target: observer.to_owned(), copy_of: Some(self.digest()), ..self.clone() }
    }

    #[must_use]
    pub const fn kind(&self) -> MessageKind {
        self.kind
    }
    #[must_use]
    pub fn sender(&self) -> &str {
        &self.sender
    }
    #[must_use]
    pub fn target(&self) -> &str {
        &self.target
    }
    #[must_use]
    pub const fn confidence(&self) -> f32 {
        self.confidence
    }
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }
    #[must_use]
    pub fn refs(&self) -> &[Reference] {
        &self.refs
    }
    #[must_use]
    pub fn label_tokens(&self) -> &[u32] {
        &self.label_tokens
    }
    #[must_use]
    pub fn copy_of(&self) -> Option<&str> {
        self.copy_of.as_deref()
    }
    /// SHA-256 of the bare text form: the message's content identity.
    #[must_use]
    pub fn digest(&self) -> String {
        nmhash::sha256_bytes(self.render_bare().as_bytes())
    }

    /// Whether the label tokens decode exactly to the description under `resolver`.
    /// `Ok(None)` when the resolver cannot decode tokens.
    /// # Errors
    /// The resolver failed.
    pub fn verify_label(&self, resolver: &dyn Resolver) -> Result<Option<bool>, Fault> {
        Ok(resolver.tokens(&self.label_tokens)?.map(|t| t == self.description))
    }

    /// The bare text form, no resolved blocks.
    #[must_use]
    pub fn render_bare(&self) -> String {
        self.render(&Bare).unwrap_or_default()
    }

    /// The full text form: every field, plus the label's decoded tokens and each
    /// reference's exact source text or decoded tokens where `resolver` gives them.
    /// # Errors
    /// The resolver failed on a reference.
    pub fn render(&self, resolver: &dyn Resolver) -> Result<String, Fault> {
        let mut out = String::with_capacity(256);
        let block = |out: &mut String, text: Option<String>| {
            if let Some(t) = text {
                out.push_str(&format!("text {}\n{t}\n", t.len()));
            }
        };
        out.push_str(FORMAT);
        out.push('\n');
        out.push_str(&format!("kind {}\n", self.kind.name()));
        out.push_str(&format!("from {}\n", self.sender));
        out.push_str(&format!("to {}\n", self.target));
        out.push_str(&format!("confidence {}\n", self.confidence));
        out.push_str(&format!("description {}\n", self.description));
        out.push_str(&format!("label-tokens{}\n", ids_text(&self.label_tokens)));
        block(&mut out, resolver.tokens(&self.label_tokens)?);
        if let Some(d) = &self.copy_of {
            out.push_str(&format!("copy-of {d}\n"));
        }
        for r in &self.refs {
            out.push_str(&r.line());
            out.push('\n');
            let text = match r {
                Reference::Span { path, start, end } => resolver.span(path, *start, *end)?,
                Reference::Tokens(ids) | Reference::Encoding { tokens: ids, .. } => resolver.tokens(ids)?,
            };
            block(&mut out, text);
        }
        out.push_str("end\n");
        Ok(out)
    }

    /// Read the text form back (bare or full). Resolved blocks are skipped by length.
    /// # Errors
    /// Wrong version line, a missing or unknown line, a bad number, a block longer than
    /// the text, or an envelope that fails `Envelope::new`'s validation.
    pub fn parse(text: &str) -> Result<Self, Fault> {
        let mut rest = text;
        let mut next = || -> Result<&str, Fault> {
            let (line, tail) = rest.split_once('\n').ok_or_else(|| bad("unterminated line"))?;
            rest = tail;
            Ok(line)
        };
        if next()? != FORMAT {
            return Err(bad("not an nm-msg/1 envelope"));
        }
        let field = |line: &str, key: &str| -> Result<String, Fault> {
            line.strip_prefix(key).and_then(|r| r.strip_prefix(' ')).map(str::to_owned).ok_or_else(|| bad(format!("expected {key}")))
        };
        let kind_name = field(next()?, "kind")?;
        let kind = MessageKind::from_name(&kind_name).ok_or_else(|| bad(format!("kind {kind_name:?}")))?;
        let sender = field(next()?, "from")?;
        let target = field(next()?, "to")?;
        let confidence: f32 = field(next()?, "confidence")?.parse().map_err(|_| bad("confidence"))?;
        let description = field(next()?, "description")?;
        let label_line = next()?;
        let label_tokens = parse_ids(label_line.strip_prefix("label-tokens").ok_or_else(|| bad("expected label-tokens"))?)?;
        let mut copy_of = None;
        let mut refs = Vec::new();
        loop {
            let (line, tail) = rest.split_once('\n').ok_or_else(|| bad("missing end"))?;
            rest = tail;
            if line == "end" {
                break;
            }
            if let Some(n) = line.strip_prefix("text ") {
                let n: usize = n.parse().map_err(|_| bad("text length"))?;
                let after = rest.get(n..).ok_or_else(|| bad("text block past the end"))?;
                rest = after.strip_prefix('\n').ok_or_else(|| bad("text block not closed"))?;
            } else if let Some(d) = line.strip_prefix("copy-of ") {
                copy_of = Some(d.to_owned());
            } else if let Some(r) = line.strip_prefix("ref ") {
                refs.push(Reference::parse(r)?);
            } else {
                return Err(bad(format!("unknown line {line:?}")));
            }
        }
        let mut e = Self::new(kind, &sender, &target, confidence, &description, refs, label_tokens)?;
        e.copy_of = copy_of;
        Ok(e)
    }

    #[must_use]
    pub fn to_value(&self) -> Value {
        wire::obj(vec![
            ("class", DataClass::Behavioural.to_value()),
            ("kind", self.kind.to_value()),
            ("sender", wire::s(&self.sender)),
            ("target", wire::s(&self.target)),
            ("confidence", wire::f(self.confidence)),
            ("description", wire::s(&self.description)),
            ("refs", Value::Array(self.refs.iter().map(Reference::to_value).collect())),
            ("label_tokens", Value::Array(self.label_tokens.iter().map(|i| wire::u(u64::from(*i))).collect())),
            ("copy_of", wire::opt_s(self.copy_of.as_deref())),
        ])
    }
    /// # Errors
    /// Rejects a malformed value or one tagged with any class but behavioural.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "message";
        crate::class::guard("message-channel", DataClass::Behavioural, wire::get_e(v, m, "class", DataClass::from_value)?)?;
        let refs = wire::get_a(v, m, "refs")?.iter().map(Reference::from_value).collect::<Result<_, _>>()?;
        let label_tokens = wire::get_a(v, m, "label_tokens")?
            .iter()
            .map(|i| i.integer().ok().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| bad("label token id")))
            .collect::<Result<_, _>>()?;
        let mut e = Self::new(
            wire::get_e(v, m, "kind", MessageKind::from_value)?,
            &wire::get_s(v, m, "sender")?,
            &wire::get_s(v, m, "target")?,
            wire::get_f(v, m, "confidence")?,
            &wire::get_s(v, m, "description")?,
            refs,
            label_tokens,
        )?;
        e.copy_of = wire::get_opt_s(v, m, "copy_of")?;
        Ok(e)
    }
}
impl Classed for Envelope {
    fn data_class(&self) -> DataClass {
        DataClass::Behavioural
    }
}

/// The version line of the motor nervous system's envelope.
pub const FORMAT_2: &str = "nm-msg/2";

named_enum! {
    /// The verbs of `nm-msg/2`, one closed vocabulary: G2's instructions I1 to I11 and the
    /// tentacles' replies upward. A new behaviour with no verb is added by a dated version bump.
    pub enum Verb in "verb" {
        Focus => "focus", Reread => "reread", Answer => "answer", Forward => "forward",
        ForwardAll => "forward-all", Joint => "joint", Expand => "expand", Reshape => "reshape-window",
        Gain => "gain", Inhibit => "inhibit", Next => "next", Reply => "reply",
    }
}

/// A typed argument of an `nm-msg/2` envelope: low-dimensional (it selects and scales), never a
/// sensed item. Pieces are named by content key, by reference.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Count(u64),
    Scale(f32),
    Word(String),
    Pieces(Vec<String>),
}
impl Arg {
    fn text(&self) -> String {
        match self {
            Self::Count(n) => format!("count {n}"),
            Self::Scale(x) => format!("scale {x}"),
            Self::Word(w) => format!("word {w}"),
            Self::Pieces(p) => format!("pieces {}", p.join(" ")),
        }
    }
    fn parse(t: &str) -> Result<Self, Fault> {
        let (k, v) = t.split_once(' ').unwrap_or((t, ""));
        Ok(match k {
            "count" => Self::Count(v.parse().map_err(|_| bad("arg count"))?),
            "scale" => Self::Scale(v.parse().map_err(|_| bad("arg scale"))?),
            "word" => Self::Word(v.to_owned()),
            "pieces" => Self::Pieces(v.split(' ').filter(|x| !x.is_empty()).map(str::to_owned).collect()),
            _ => return Err(bad(format!("unknown argument {k:?}"))),
        })
    }
}

/// One `nm-msg/2` envelope (protocol pass Solution 3, Motor): version, id, parent id, kind and
/// verb, sender, receiver, the copy to G1 (always on: the constructor sets it and nothing can
/// clear it), typed arguments, references, confidence, a one-line description, the pack's
/// fingerprint when a muscle-memory pack drove it, expiry, and a goal id for a long order.
/// Behavioural class; it never carries sensed content, only references to it.
#[derive(Clone, Debug, PartialEq)]
pub struct Motor {
    id: u64,
    parent: Option<u64>,
    kind: MessageKind,
    verb: Verb,
    sender: ComponentId,
    receiver: ComponentId,
    copy_to_g1: bool,
    args: Vec<(String, Arg)>,
    refs: Vec<Reference>,
    confidence: f32,
    description: String,
    pack: Option<String>,
    expires_ms: u64,
    goal: Option<u64>,
}
impl Motor {
    /// A new envelope with the copy to G1 on.
    /// # Errors
    /// An empty or whitespace-bearing sender or receiver, a description that is empty or spans
    /// lines, an argument name with a space, an invalid reference.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u64,
        parent: Option<u64>,
        kind: MessageKind,
        verb: Verb,
        sender: &str,
        receiver: &str,
        confidence: f32,
        description: &str,
        expires_ms: u64,
    ) -> Result<Self, Fault> {
        for (what, x) in [("sender", sender), ("receiver", receiver)] {
            if x.is_empty() || x.contains(char::is_whitespace) {
                return Err(bad(format!("{what} must be one unit id, got {x:?}")));
            }
        }
        if description.trim().is_empty() || description.contains(['\n', '\r']) {
            return Err(bad("description must be one non-empty line"));
        }
        Ok(Self {
            id,
            parent,
            kind,
            verb,
            sender: sender.to_owned(),
            receiver: receiver.to_owned(),
            copy_to_g1: true,
            args: Vec::new(),
            refs: Vec::new(),
            confidence: if confidence.is_finite() { confidence.clamp(0.0, 1.0) } else { 0.0 },
            description: description.to_owned(),
            pack: None,
            expires_ms,
            goal: None,
        })
    }
    /// With a typed argument.
    /// # Errors
    /// A name with a space or line break, or an argument value spanning lines.
    pub fn arg(mut self, name: &str, a: Arg) -> Result<Self, Fault> {
        if name.is_empty() || name.contains(char::is_whitespace) || a.text().contains(['\n', '\r']) {
            return Err(bad(format!("argument {name:?}")));
        }
        self.args.push((name.to_owned(), a));
        Ok(self)
    }
    /// With a reference.
    /// # Errors
    /// An invalid reference.
    pub fn reference(mut self, r: Reference) -> Result<Self, Fault> {
        r.validate()?;
        self.refs.push(r);
        Ok(self)
    }
    /// A long order's goal id.
    #[must_use]
    pub const fn goal(mut self, goal: u64) -> Self {
        self.goal = Some(goal);
        self
    }
    /// The fingerprint of the muscle-memory pack that drove it.
    #[must_use]
    pub fn pack(mut self, fingerprint: &str) -> Self {
        self.pack = Some(fingerprint.to_owned());
        self
    }
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }
    #[must_use]
    pub const fn parent(&self) -> Option<u64> {
        self.parent
    }
    #[must_use]
    pub const fn verb(&self) -> Verb {
        self.verb
    }
    #[must_use]
    pub const fn kind(&self) -> MessageKind {
        self.kind
    }
    #[must_use]
    pub fn sender(&self) -> &str {
        &self.sender
    }
    #[must_use]
    pub fn receiver(&self) -> &str {
        &self.receiver
    }
    #[must_use]
    pub const fn copy_to_g1(&self) -> bool {
        self.copy_to_g1
    }
    #[must_use]
    pub fn args(&self) -> &[(String, Arg)] {
        &self.args
    }
    /// The pieces named by argument `name` (empty when absent).
    #[must_use]
    pub fn pieces(&self, name: &str) -> Vec<String> {
        self.args.iter().find(|(n, _)| n == name).map_or_else(Vec::new, |(_, a)| if let Arg::Pieces(p) = a { p.clone() } else { Vec::new() })
    }
    #[must_use]
    pub fn refs(&self) -> &[Reference] {
        &self.refs
    }
    #[must_use]
    pub const fn confidence(&self) -> f32 {
        self.confidence
    }
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }
    #[must_use]
    pub const fn expires_ms(&self) -> u64 {
        self.expires_ms
    }
    #[must_use]
    pub const fn goal_id(&self) -> Option<u64> {
        self.goal
    }
    /// The text form, line based: it reads back as English and references.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = format!(
            "{FORMAT_2}\nid {}\nparent {}\nkind {}\nverb {}\nfrom {}\nto {}\ncopy-to-g1 {}\nconfidence {}\ndescription {}\nexpires {}\n",
            self.id,
            self.parent.map_or_else(|| "-".to_owned(), |p| p.to_string()),
            self.kind.name(),
            self.verb.name(),
            self.sender,
            self.receiver,
            self.copy_to_g1,
            self.confidence,
            self.description,
            self.expires_ms
        );
        if let Some(g) = self.goal {
            out.push_str(&format!("goal {g}\n"));
        }
        if let Some(p) = &self.pack {
            out.push_str(&format!("pack {p}\n"));
        }
        for (n, a) in &self.args {
            out.push_str(&format!("arg {n} {}\n", a.text()));
        }
        for r in &self.refs {
            out.push_str(&r.line());
            out.push('\n');
        }
        out.push_str("end\n");
        out
    }
    /// SHA-256 of the text form: the envelope's identity (what G1's ledger acknowledges).
    #[must_use]
    pub fn digest(&self) -> String {
        nmhash::sha256_bytes(self.render().as_bytes())
    }
    /// Read the text form back.
    /// # Errors
    /// Not an `nm-msg/2` envelope, a missing or unknown line, a copy to G1 that is off.
    pub fn parse(text: &str) -> Result<Self, Fault> {
        let mut lines = text.lines();
        if lines.next() != Some(FORMAT_2) {
            return Err(bad("not an nm-msg/2 envelope"));
        }
        let mut get = |key: &str| -> Result<String, Fault> {
            lines.next().and_then(|l| l.strip_prefix(key)).and_then(|r| r.strip_prefix(' ')).map(str::to_owned).ok_or_else(|| bad(format!("expected {key}")))
        };
        let id: u64 = get("id")?.parse().map_err(|_| bad("id"))?;
        let parent = match get("parent")?.as_str() {
            "-" => None,
            p => Some(p.parse().map_err(|_| bad("parent"))?),
        };
        let kind_name = get("kind")?;
        let kind = MessageKind::from_name(&kind_name).ok_or_else(|| bad(format!("kind {kind_name:?}")))?;
        let verb_name = get("verb")?;
        let verb = Verb::from_name(&verb_name).ok_or_else(|| bad(format!("verb {verb_name:?}")))?;
        let sender = get("from")?;
        let receiver = get("to")?;
        if get("copy-to-g1")? != "true" {
            return Err(bad("an order without its copy to G1"));
        }
        let confidence: f32 = get("confidence")?.parse().map_err(|_| bad("confidence"))?;
        let description = get("description")?;
        let expires: u64 = get("expires")?.parse().map_err(|_| bad("expires"))?;
        let mut m = Self::new(id, parent, kind, verb, &sender, &receiver, confidence, &description, expires)?;
        for l in lines {
            if l == "end" {
                return Ok(m);
            }
            if let Some(g) = l.strip_prefix("goal ") {
                m.goal = Some(g.parse().map_err(|_| bad("goal"))?);
            } else if let Some(p) = l.strip_prefix("pack ") {
                m.pack = Some(p.to_owned());
            } else if let Some(a) = l.strip_prefix("arg ") {
                let (n, v) = a.split_once(' ').ok_or_else(|| bad("arg"))?;
                m.args.push((n.to_owned(), Arg::parse(v)?));
            } else if let Some(r) = l.strip_prefix("ref ") {
                m.refs.push(Reference::parse(r)?);
            } else {
                return Err(bad(format!("unknown line {l:?}")));
            }
        }
        Err(bad("missing end"))
    }
}
impl Classed for Motor {
    fn data_class(&self) -> DataClass {
        DataClass::Behavioural
    }
}

/// The record of self-issued actions, fed by action copies: lets G1 and G2 tell an edit the
/// system made from new input arriving from outside.
#[derive(Clone, Debug, Default)]
pub struct ActionLedger {
    originals: std::collections::BTreeSet<String>,
    spans: Vec<(String, u32, u32)>,
    /// Predicted post-edit content (2026-11-12, motor MO3 by content): (path, SHA-256 of the file
    /// as it will be once the system's own change is applied), used up once matched.
    predicted: Vec<(String, String)>,
}
impl ActionLedger {
    /// Record that `path` is expected to arrive with content hash `sha` (an own change).
    pub fn predict(&mut self, path: &str, sha: &str) {
        self.predicted.retain(|(p, _)| p != path);
        self.predicted.push((path.to_owned(), sha.to_owned()));
    }
    /// A change seen on `path` with content hash `sha`: true when it is the system's own (a
    /// prediction matched, which is then used up); false for a change from outside, which is new
    /// sensory input.
    pub fn observe(&mut self, path: &str, sha: &str) -> bool {
        match self.predicted.iter().position(|(p, h)| p == path && h == sha) {
            Some(i) => {
                self.predicted.remove(i);
                true
            }
            None => false,
        }
    }
    /// Predictions not yet matched.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.predicted.len()
    }
    /// Record one action copy.
    /// # Errors
    /// The envelope is not an action copy.
    pub fn record(&mut self, copy: &Envelope) -> Result<(), Fault> {
        let original = copy
            .copy_of()
            .filter(|_| copy.kind() == MessageKind::ActionCopy)
            .ok_or_else(|| bad("only action copies enter the action ledger"))?;
        self.originals.insert(original.to_owned());
        for r in copy.refs() {
            if let Reference::Span { path, start, end } = r {
                self.spans.push((path.clone(), *start, *end));
            }
        }
        Ok(())
    }
    /// Whether `msg` is one of the system's own recorded actions.
    #[must_use]
    pub fn is_own(&self, msg: &Envelope) -> bool {
        self.originals.contains(&msg.digest())
    }
    /// Whether line `line` of `path` lies in a span the system itself acted on.
    #[must_use]
    pub fn is_self_caused(&self, path: &str, line: u32) -> bool {
        self.spans.iter().any(|(p, a, b)| p == path && (*a..=*b).contains(&line))
    }
}

fn bad(detail: impl Into<String>) -> Fault {
    Fault::new("nmtype.message", FaultKind::Interface, detail)
}
fn ids_text(ids: &[u32]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(ids.len() * 7);
    for i in ids {
        let _ = write!(s, " {i}");
    }
    s
}
fn parse_ids(rest: &str) -> Result<Vec<u32>, Fault> {
    if rest.is_empty() {
        return Ok(Vec::new());
    }
    rest.strip_prefix(' ')
        .ok_or_else(|| bad("token ids must follow one space"))?
        .split(' ')
        .map(|t| t.parse().map_err(|_| bad(format!("token id {t:?}"))))
        .collect()
}
fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A resolver for checks and measurements: an in-memory file table and a byte-level
/// tokenizer (one token per byte, id = byte value), so decoding is exact by construction.
/// Real callers supply the base tokenizer and the source tree.
pub struct TableResolver {
    pub files: Vec<(String, String)>,
}
impl TableResolver {
    /// Byte-level token ids of `text`.
    #[must_use]
    pub fn encode(text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }
}
impl Resolver for TableResolver {
    fn span(&self, path: &str, start: u32, end: u32) -> Result<Option<String>, Fault> {
        let Some((_, body)) = self.files.iter().find(|(p, _)| p == path) else {
            return Ok(None);
        };
        let lines: Vec<&str> = body.split_inclusive('\n').collect();
        let (a, b) = (start as usize, end as usize);
        lines
            .get(a.saturating_sub(1)..b.min(lines.len()))
            .filter(|_| b <= lines.len())
            .map(|ls| Some(ls.concat()))
            .ok_or_else(|| bad(format!("span {path}:{start}-{end} past the end")))
    }
    fn tokens(&self, ids: &[u32]) -> Result<Option<String>, Fault> {
        let bytes = ids
            .iter()
            .map(|i| u8::try_from(*i).map_err(|_| bad(format!("token id {i} outside the byte vocabulary"))))
            .collect::<Result<Vec<u8>, _>>()?;
        String::from_utf8(bytes).map(Some).map_err(|_| bad("tokens are not valid UTF-8"))
    }
}

/// The example set of test C1 (first half): one envelope per message kind, with span,
/// token and encoding references, a code example and a mathematics example, plus the
/// resolver holding their sources.
/// # Errors
/// An example failed validation (a defect in this function).
pub fn c1_examples() -> Result<(Vec<Envelope>, TableResolver), Fault> {
    let code = "pub fn area(r: f64) -> f64 {\n    std::f64::consts::PI * r * r\n}\n";
    let tricky = "line one\nend\nref span x:1-2\ntext 9\n\u{e9}t\u{e9} \u{2211}\n";
    let resolver = TableResolver {
        files: vec![("crates/demo/src/geom.rs".into(), code.into()), ("docs/notes with spaces:v2.md".into(), tricky.into())],
    };
    let h = |s: &str| nmhash::sha256_bytes(s.as_bytes());
    let base = h("qwen3-0.6b base weights (example handle)");
    let maths = "d/dx (x^2 + 3x) = 2x + 3; \u{222b}_0^1 x dx = 1/2";
    let code_gen = "fn twice(x: i32) -> i32 { 2 * x }";
    let mk =
        |kind, from: &str, to: &str, conf, desc: &str, refs| Envelope::new(kind, from, to, conf, desc, refs, TableResolver::encode(desc));
    let instruction = mk(
        MessageKind::Instruction,
        "nmbrain.g2#0",
        "nmpod.unit#t3",
        0.875,
        "Focus tentacle 3 on the area function and report its callers.",
        vec![
            Reference::Span { path: "crates/demo/src/geom.rs".into(), start: 1, end: 3 },
            Reference::Encoding { content: h(code), base: base.clone(), tokens: TableResolver::encode(code) },
        ],
    )?;
    let reflex = mk(MessageKind::Reflex, "nmbrain.g1#0", "nmpod.unit#t1", 1.0, "Stop reading: the window budget is spent.", Vec::new())?;
    let report = mk(
        MessageKind::Report,
        "nmpod.unit#t3",
        "nmbrain.g1#0",
        0.6,
        "Derivative checked; the formula below holds.",
        vec![Reference::Tokens(TableResolver::encode(maths))],
    )?;
    let request = mk(
        MessageKind::Request,
        "nmbrain.g2#0",
        "nmbrain.chain#coder",
        0.333_333_34,
        "Write a function that doubles an integer (draft attached).",
        vec![
            Reference::Tokens(TableResolver::encode(code_gen)),
            Reference::Span { path: "docs/notes with spaces:v2.md".into(), start: 2, end: 5 },
        ],
    )?;
    let action = instruction.action_copy("nmbrain.g1#0");
    Ok((vec![instruction, reflex, report, request, action], resolver))
}

/// Self-check of the `message` sub-component, test C1 first half: for each message kind,
/// full render then parse and bare render then parse give the same envelope, the JSON form
/// round-trips, the label tokens decode to the description, the resolved code and
/// mathematics appear verbatim, invalid envelopes are refused, and the action ledger knows
/// the system's own action.
pub(crate) fn check() -> Result<(), String> {
    let mut led = ActionLedger::default();
    led.predict("a.rs", "h1");
    if led.observe("a.rs", "h2") || !led.observe("a.rs", "h1") || led.observe("a.rs", "h1") || led.pending() != 0 {
        return Err("nmtype.message: own-edit prediction (matched once, outside otherwise)".into());
    }
    let s = |e: Fault| e.to_string();
    let (envs, resolver) = c1_examples().map_err(s)?;
    for k in MessageKind::ALL {
        if !envs.iter().any(|e| e.kind() == *k) {
            return Err(format!("nmtype.message: C1 set lacks kind {}", k.name()));
        }
    }
    for e in &envs {
        let full = e.render(&resolver).map_err(s)?;
        if Envelope::parse(&full).map_err(s)? != *e {
            return Err(format!("nmtype.message: full round trip changed a {} envelope", e.kind().name()));
        }
        if Envelope::parse(&e.render_bare()).map_err(s)? != *e {
            return Err(format!("nmtype.message: bare round trip changed a {} envelope", e.kind().name()));
        }
        if Envelope::from_value(&e.to_value()).map_err(s)? != *e {
            return Err(format!("nmtype.message: JSON round trip changed a {} envelope", e.kind().name()));
        }
        if e.verify_label(&resolver).map_err(s)? != Some(true) {
            return Err("nmtype.message: label tokens do not decode to the description".into());
        }
        for r in e.refs() {
            let want = match r {
                Reference::Span { path, start, end } => resolver.span(path, *start, *end),
                Reference::Tokens(t) | Reference::Encoding { tokens: t, .. } => resolver.tokens(t),
            }
            .map_err(s)?
            .unwrap_or_default();
            if !full.contains(&format!("text {}\n{want}\n", want.len())) {
                return Err("nmtype.message: a referenced text is not rendered verbatim".into());
            }
        }
    }
    // Refusals.
    let bad_cases = [
        Envelope::new(MessageKind::Report, "a b", "c", 0.5, "x", Vec::new(), Vec::new()),
        Envelope::new(MessageKind::Report, "a", "c", 0.5, "two\nlines", Vec::new(), Vec::new()),
        Envelope::new(
            MessageKind::Report,
            "a",
            "c",
            0.5,
            "x",
            vec![Reference::Encoding { content: "0".repeat(64), base: "0".repeat(64), tokens: Vec::new() }],
            Vec::new(),
        ),
    ];
    if bad_cases.iter().any(Result::is_ok) {
        return Err("nmtype.message: an invalid envelope was accepted".into());
    }
    // Action copies.
    let mut ledger = ActionLedger::default();
    let (original, copy) = (&envs[0], &envs[4]);
    ledger.record(copy).map_err(s)?;
    if !ledger.is_own(original) || ledger.is_own(&envs[1]) || ledger.record(original).is_ok() {
        return Err("nmtype.message: action ledger".into());
    }
    if !ledger.is_self_caused("crates/demo/src/geom.rs", 2) || ledger.is_self_caused("crates/demo/src/geom.rs", 4) {
        return Err("nmtype.message: self-caused span".into());
    }
    Ok(())
}
