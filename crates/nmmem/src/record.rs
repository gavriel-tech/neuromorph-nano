//! The universal memory record and its source anchors (2026-11-12, memory revision 2, section 5;
//! direction of Luis Jake Gabriel III: the memory is universal, not only about code). One shape
//! for every kind of memory (fact, procedure, decision, failure, preference) and every kind of
//! source (file, document, page, conversation, table, log, tool output). An anchor points to the
//! exact part of a source the memory rests on, with a hash of that part and a stable name to find
//! it again; the excerpt is kept only when the source can vanish (a conversation, a page, a tool's
//! output), since a file is already on disk and is re-read fresh.
//!
//! The file adapter (`relocate_lines`) finds an anchored part again after the source changed: the
//! same content where it was (valid), the same content elsewhere (moved, by hash), the stable name
//! still present with other content (changed), or neither (gone).
use crate::store::{fault, obj, s, u};
use nmjson::Value;
use nmtype::Fault;

macro_rules! labels {
    ($name:ident { $($v:ident => $t:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
        pub enum $name { $($v),+ }
        impl $name {
            pub const ALL: &'static [Self] = &[$(Self::$v),+];
            #[must_use]
            pub const fn label(self) -> &'static str {
                match self { $(Self::$v => $t),+ }
            }
            /// # Errors
            /// An unknown label.
            pub fn parse(t: &str) -> Result<Self, Fault> {
                Self::ALL.iter().copied().find(|x| x.label() == t).ok_or_else(|| fault(format!("unknown {}: {t}", stringify!($name))))
            }
        }
    };
}

labels!(Kind { Fact => "fact", Procedure => "procedure", Decision => "decision", Failure => "failure", Preference => "preference" });
labels!(Class { Pinned => "pinned", Protected => "protected", Normal => "normal", Ephemeral => "ephemeral" });
labels!(State { Active => "active", Archived => "archived", Superseded => "superseded", Invalid => "invalid", Compacted => "compacted" });
labels!(Tier { Core => "1-core", Active => "2-active", Archive => "3-archive", Temp => "temp" });
labels!(Source { File => "file", Document => "document", Page => "page", Conversation => "conversation", Table => "table", Log => "log", Tool => "tool" });

impl Source {
    /// Whether the source can be read again later (else the anchor keeps the excerpt).
    #[must_use]
    pub const fn rereadable(self) -> bool {
        matches!(self, Self::File | Self::Document)
    }
}

/// The part of a source an anchor rests on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    /// Lines `start..=end` (1-based) of a file or document.
    Lines { start: usize, end: usize },
    /// A section by its heading.
    Heading(String),
    /// A conversation turn by its id.
    Turn(String),
    /// A table row by its key.
    Row(String),
    /// A time range of a log, in milliseconds.
    Time { from: u64, to: u64 },
    /// The whole source.
    Whole,
}

impl Part {
    fn to_value(&self) -> Value {
        match self {
            Self::Lines { start, end } => obj(vec![("lines", Value::Array(vec![u(*start as u64), u(*end as u64)]))]),
            Self::Heading(h) => obj(vec![("heading", s(h))]),
            Self::Turn(t) => obj(vec![("turn", s(t))]),
            Self::Row(r) => obj(vec![("row", s(r))]),
            Self::Time { from, to } => obj(vec![("time", Value::Array(vec![u(*from), u(*to)]))]),
            Self::Whole => obj(vec![("whole", Value::Bool(true))]),
        }
    }
    fn from_value(v: &Value) -> Result<Self, Fault> {
        let pair = |k: &str| -> Option<(u64, u64)> {
            let a = v.get(k).ok()?.array().ok()?;
            Some((a.first()?.integer().ok()?, a.get(1)?.integer().ok()?))
        };
        let text = |k: &str| v.get(k).and_then(Value::string).ok().map(str::to_owned);
        if let Some((a, b)) = pair("lines") {
            return Ok(Self::Lines { start: usize::try_from(a).map_err(fault)?, end: usize::try_from(b).map_err(fault)? });
        }
        if let Some(h) = text("heading") {
            return Ok(Self::Heading(h));
        }
        if let Some(t) = text("turn") {
            return Ok(Self::Turn(t));
        }
        if let Some(r) = text("row") {
            return Ok(Self::Row(r));
        }
        if let Some((a, b)) = pair("time") {
            return Ok(Self::Time { from: a, to: b });
        }
        Ok(Self::Whole)
    }
}

/// Where a memory points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Anchor {
    pub source: Source,
    /// Path, address or id of the source.
    pub location: String,
    pub part: Part,
    /// SHA-256 of the part's content when the memory was made.
    pub part_hash: String,
    /// Stable name to find the part again (a symbol, a heading, an id); may be empty.
    pub name: String,
    /// The part's text, kept only when the source cannot be read again.
    pub excerpt: Option<String>,
}

impl Anchor {
    fn to_value(&self) -> Value {
        obj(vec![
            ("source", s(self.source.label())),
            ("location", s(&self.location)),
            ("part", self.part.to_value()),
            ("part_hash", s(&self.part_hash)),
            ("name", s(&self.name)),
            ("excerpt", self.excerpt.as_deref().map_or(Value::Null, s)),
        ])
    }
    fn from_value(v: &Value) -> Result<Self, Fault> {
        let t = |k: &str| v.get(k).and_then(Value::string).map(str::to_owned).map_err(fault);
        Ok(Self {
            source: Source::parse(&t("source")?)?,
            location: t("location")?,
            part: Part::from_value(v.get("part").map_err(fault)?)?,
            part_hash: t("part_hash")?,
            name: t("name").unwrap_or_default(),
            excerpt: v.get("excerpt").and_then(Value::string).ok().map(str::to_owned),
        })
    }
}

/// When a memory is true and when the system knew it (two timelines).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Validity {
    pub from_day: u64,
    pub from_version: u64,
    pub until_day: Option<u64>,
    pub until_version: Option<u64>,
    pub recorded_day: u64,
    pub retired_day: Option<u64>,
}

/// How strongly a memory is held (section 7).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Retention {
    /// Stability in run-days and difficulty (1 to 10).
    pub stability: f64,
    pub difficulty: f64,
    pub last_use_day: u64,
    /// Distinct run-days on which it was used.
    pub days_used: u64,
    pub uses: u64,
    pub useful: u64,
    pub failures: u64,
    pub salience_pending: f64,
}

/// One memory.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub id: String,
    /// SHA-256 of kind, text and anchors' part hashes (deduplication).
    pub content: String,
    pub kind: Kind,
    pub class: Class,
    pub tier: Tier,
    pub state: State,
    pub topic: String,
    pub text: String,
    pub anchors: Vec<Anchor>,
    pub keys: Vec<String>,
    /// (relation, record id): cited-together, supersedes, superseded-by, derived-from.
    pub links: Vec<(String, String)>,
    pub validity: Validity,
    pub retention: Retention,
    /// (right, who gave it).
    pub verdicts: Vec<(bool, String)>,
}

fn opt(v: Option<u64>) -> Value {
    v.map_or(Value::Null, u)
}
fn f(x: f64) -> Value {
    Value::Number(format!("{x}"))
}

impl Record {
    /// The content hash of a record's kind, text and anchors.
    #[must_use]
    pub fn content_hash(kind: Kind, text: &str, anchors: &[Anchor]) -> String {
        let mut b = format!("{}\n{text}", kind.label());
        for a in anchors {
            b.push('\n');
            b.push_str(&a.part_hash);
        }
        nmhash::sha256_bytes(b.as_bytes())
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let v = &self.validity;
        let r = &self.retention;
        obj(vec![
            ("id", s(&self.id)),
            ("content", s(&self.content)),
            ("kind", s(self.kind.label())),
            ("class", s(self.class.label())),
            ("tier", s(self.tier.label())),
            ("state", s(self.state.label())),
            ("topic", s(&self.topic)),
            ("text", s(&self.text)),
            ("anchors", Value::Array(self.anchors.iter().map(Anchor::to_value).collect())),
            ("keys", Value::Array(self.keys.iter().map(|k| s(k)).collect())),
            ("links", Value::Array(self.links.iter().map(|(rel, id)| obj(vec![("rel", s(rel)), ("id", s(id))])).collect())),
            (
                "valid",
                obj(vec![
                    ("from_day", u(v.from_day)),
                    ("from_version", u(v.from_version)),
                    ("until_day", opt(v.until_day)),
                    ("until_version", opt(v.until_version)),
                ]),
            ),
            ("recorded", obj(vec![("day", u(v.recorded_day)), ("retired_day", opt(v.retired_day))])),
            (
                "retention",
                obj(vec![
                    ("stability", f(r.stability)),
                    ("difficulty", f(r.difficulty)),
                    ("last_use_day", u(r.last_use_day)),
                    ("days_used", u(r.days_used)),
                    ("uses", u(r.uses)),
                    ("useful", u(r.useful)),
                    ("failures", u(r.failures)),
                    ("salience_pending", f(r.salience_pending)),
                ]),
            ),
            (
                "verdicts",
                Value::Array(self.verdicts.iter().map(|(ok, by)| obj(vec![("right", Value::Bool(*ok)), ("by", s(by))])).collect()),
            ),
        ])
    }
    /// # Errors
    /// A missing field or an unknown label.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let t = |x: &Value, k: &str| x.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| fault(format!("{k}: {e}")));
        let n = |x: &Value, k: &str| x.get(k).and_then(Value::integer).ok().unwrap_or(0);
        let on = |x: &Value, k: &str| x.get(k).and_then(Value::integer).ok();
        let fl = |x: &Value, k: &str| match x.get(k) {
            Ok(Value::Number(n)) => n.parse().unwrap_or(0.0),
            _ => 0.0,
        };
        let valid = v.get("valid").map_err(fault)?;
        let recorded = v.get("recorded").map_err(fault)?;
        let ret = v.get("retention").map_err(fault)?;
        let list = |k: &str| v.get(k).and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default();
        Ok(Self {
            id: t(v, "id")?,
            content: t(v, "content")?,
            kind: Kind::parse(&t(v, "kind")?)?,
            class: Class::parse(&t(v, "class")?)?,
            tier: Tier::parse(&t(v, "tier")?)?,
            state: State::parse(&t(v, "state")?)?,
            topic: t(v, "topic").unwrap_or_default(),
            text: t(v, "text")?,
            anchors: list("anchors").iter().map(Anchor::from_value).collect::<Result<_, _>>()?,
            keys: list("keys").iter().filter_map(|k| k.string().ok().map(str::to_owned)).collect(),
            links: list("links")
                .iter()
                .filter_map(|l| Some((l.get("rel").ok()?.string().ok()?.to_owned(), l.get("id").ok()?.string().ok()?.to_owned())))
                .collect(),
            validity: Validity {
                from_day: n(valid, "from_day"),
                from_version: n(valid, "from_version"),
                until_day: on(valid, "until_day"),
                until_version: on(valid, "until_version"),
                recorded_day: n(recorded, "day"),
                retired_day: on(recorded, "retired_day"),
            },
            retention: Retention {
                stability: fl(ret, "stability"),
                difficulty: fl(ret, "difficulty"),
                last_use_day: n(ret, "last_use_day"),
                days_used: n(ret, "days_used"),
                uses: n(ret, "uses"),
                useful: n(ret, "useful"),
                failures: n(ret, "failures"),
                salience_pending: fl(ret, "salience_pending"),
            },
            verdicts: list("verdicts")
                .iter()
                .filter_map(|x| Some((matches!(x.get("right").ok()?, Value::Bool(true)), x.get("by").ok()?.string().ok()?.to_owned())))
                .collect(),
        })
    }
}

/// Hash of a set of lines (the part hash of a `Lines` anchor).
#[must_use]
pub fn lines_hash(lines: &[&str]) -> String {
    nmhash::sha256_bytes(lines.join("\n").as_bytes())
}

/// Where an anchored part is now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Relocation {
    /// Same content at the same place.
    Valid,
    /// Same content at other lines.
    Moved { start: usize, end: usize },
    /// The stable name is still there, with other content (these lines now).
    Changed { start: usize, end: usize },
    /// Neither the content nor the name is there.
    Gone,
}

/// The file adapter: find a `Lines` anchor again in the source's current text. First the same
/// place, then the same content anywhere (by hash), then the stable name (a whole-word match), else
/// gone. Other parts are not lines; they report `Gone` here and use their own adapters.
#[must_use]
pub fn relocate_lines(a: &Anchor, text: &str) -> Relocation {
    let Part::Lines { start, end } = a.part else { return Relocation::Gone };
    let lines: Vec<&str> = text.lines().collect();
    let len = end.saturating_sub(start) + 1;
    if start >= 1 && end <= lines.len() && lines_hash(lines.get(start - 1..end).unwrap_or(&[])) == a.part_hash {
        return Relocation::Valid;
    }
    for i in 0..lines.len().saturating_sub(len - 1) {
        if lines_hash(lines.get(i..i + len).unwrap_or(&[])) == a.part_hash {
            return Relocation::Moved { start: i + 1, end: i + len };
        }
    }
    if !a.name.is_empty() {
        let word = |l: &str| l.split(|c: char| !(c.is_alphanumeric() || c == '_')).any(|w| w == a.name);
        if let Some(i) = lines.iter().position(|l| word(l)) {
            return Relocation::Changed { start: i + 1, end: (i + len).min(lines.len()) };
        }
    }
    Relocation::Gone
}

/// Section check: a record round-trips through its value; labels parse back; an anchor relocates as
/// valid, moved (lines inserted above), changed (body edited, name kept) and gone (removed).
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    let code = "fn a() {}\nfn decide_switch(x: u8) -> bool {\n    x > 3\n}\nfn z() {}\n";
    let part: Vec<&str> = code.lines().collect();
    let anchor = Anchor {
        source: Source::File,
        location: "src/x.rs".into(),
        part: Part::Lines { start: 2, end: 4 },
        part_hash: lines_hash(part.get(1..4).ok_or("lines")?),
        name: "decide_switch".into(),
        excerpt: None,
    };
    let rec = Record {
        id: "rd5-1".into(),
        content: Record::content_hash(Kind::Fact, "the switch rule", std::slice::from_ref(&anchor)),
        kind: Kind::Fact,
        class: Class::Normal,
        tier: Tier::Active,
        state: State::Active,
        topic: "context".into(),
        text: "the switch rule".into(),
        anchors: vec![
            anchor.clone(),
            Anchor {
                source: Source::Conversation,
                location: "session-3".into(),
                part: Part::Turn("t7".into()),
                part_hash: "h".into(),
                name: String::new(),
                excerpt: Some("asked about the switch".into()),
            },
        ],
        keys: vec!["decide_switch".into()],
        links: vec![("cited-together".into(), "rd5-2".into())],
        validity: Validity { from_day: 5, from_version: 40, until_day: None, until_version: None, recorded_day: 5, retired_day: None },
        retention: Retention {
            stability: 8.3,
            difficulty: 5.0,
            last_use_day: 5,
            days_used: 1,
            uses: 1,
            useful: 0,
            failures: 0,
            salience_pending: 0.0,
        },
        verdicts: vec![(true, "nm_feedback".into())],
    };
    if Record::from_value(&rec.to_value()).map_err(|e| e.to_string())? != rec {
        return Err("nmmem.record: round trip".into());
    }
    let cases = [
        (code.to_owned(), Relocation::Valid),
        (format!("// new\n// lines\n{code}"), Relocation::Moved { start: 4, end: 6 }),
        (code.replace("x > 3", "x > 9"), Relocation::Changed { start: 2, end: 4 }),
        ("fn a() {}\nfn z() {}\n".to_owned(), Relocation::Gone),
    ];
    for (text, want) in cases {
        let got = relocate_lines(&anchor, &text);
        if got != want {
            return Err(format!("nmmem.record: relocation {got:?}, want {want:?}"));
        }
    }
    if !Source::File.rereadable() || Source::Conversation.rereadable() {
        return Err("nmmem.record: rereadable sources".into());
    }
    Ok(())
}

/// What reconsolidation found for one recalled memory.
#[derive(Clone, Debug, PartialEq)]
pub struct Reconsolidated {
    /// Journal events to append (relocations, a failed re-read, an invalid state).
    pub events: Vec<(String, Value)>,
    /// The content at an anchor changed while its name stayed: the caller's gate decides whether
    /// the memory still holds (then a use) or is superseded by a new memory.
    pub needs_check: bool,
}

/// Reconsolidation on recall (2026-11-12, memory revision 2, section 12 step 6): each re-readable
/// `Lines` anchor is looked up in its source's current text (`read` returns it, `None` when the
/// source is gone): moved, its part follows; changed, its part and hash follow and the memory is
/// flagged for the gate; gone, a failed re-read is counted, and at the second the memory becomes
/// invalid.
#[must_use]
pub fn reconsolidate(r: &Record, read: &dyn Fn(&Anchor) -> Option<String>) -> Reconsolidated {
    let mut events = Vec::new();
    let mut needs_check = false;
    // Two causes of a failed re-read, decided apart (main session, 2026-11-16): the source itself
    // is gone (rule 15: invalid), or the source is there but the part is found neither by its hash
    // nor by its stable name (rule 7: one tier down at the second failure).
    let (mut gone, mut unconfirmed) = (false, false);
    for (i, a) in r.anchors.iter().enumerate().filter(|(_, a)| a.source.rereadable()) {
        let relocate = |start: usize, end: usize, hash: &str| {
            (
                "relocate".to_owned(),
                obj(vec![
                    ("id", s(&r.id)),
                    ("anchor", u(i as u64)),
                    ("part", Part::Lines { start, end }.to_value()),
                    ("part_hash", s(hash)),
                    ("from", a.part.to_value()),
                ]),
            )
        };
        match read(a).map(|t| (relocate_lines(a, &t), t)) {
            Some((Relocation::Valid, _)) => {}
            Some((Relocation::Moved { start, end }, _)) => events.push(relocate(start, end, &a.part_hash)),
            Some((Relocation::Changed { start, end }, t)) => {
                let lines: Vec<&str> = t.lines().collect();
                events.push(relocate(start, end, &lines_hash(lines.get(start - 1..end).unwrap_or(&[]))));
                needs_check = true;
            }
            Some((Relocation::Gone, _)) => unconfirmed = true,
            None => gone = true,
        }
    }
    if gone {
        events.push(("failed".into(), obj(vec![("id", s(&r.id)), ("why", s("cited source no longer exists"))])));
        events.push((
            "state".into(),
            obj(vec![("id", s(&r.id)), ("to", s(State::Invalid.label())), ("from", s(r.state.label())), ("why", s("cited source no longer exists"))]),
        ));
    } else if unconfirmed {
        events.push(("failed".into(), obj(vec![("id", s(&r.id)), ("why", s("cited part not found by hash or name on re-read"))])));
        let lower = match r.tier {
            Tier::Core => Some(Tier::Active),
            Tier::Active => Some(Tier::Archive),
            Tier::Archive | Tier::Temp => None,
        };
        if (r.retention.failures + 1).is_multiple_of(2)
            && r.class != Class::Pinned
            && let Some(to) = lower
        {
            events.push((
                "tier".into(),
                obj(vec![("id", s(&r.id)), ("to", s(to.label())), ("from", s(r.tier.label())), ("why", s("cited part not confirmed on two re-reads"))]),
            ));
        }
    }
    Reconsolidated { events, needs_check }
}

/// Section check for reconsolidation: moved gives one relocation, changed a relocation and a check
/// flag, gone a failed re-read and at the second an invalid state; a valid anchor gives nothing.
/// # Errors
/// One fault line.
pub(crate) fn check_reconsolidate() -> Result<(), String> {
    let code = "fn a() {}\nfn decide_switch(x: u8) -> bool {\n    x > 3\n}\n";
    let lines: Vec<&str> = code.lines().collect();
    let anchor = Anchor {
        source: Source::File,
        location: "src/x.rs".into(),
        part: Part::Lines { start: 2, end: 4 },
        part_hash: lines_hash(lines.get(1..4).ok_or("lines")?),
        name: "decide_switch".into(),
        excerpt: None,
    };
    let mut rec = Record {
        id: "m".into(),
        content: "c".into(),
        kind: Kind::Fact,
        class: Class::Normal,
        tier: Tier::Active,
        state: State::Active,
        topic: String::new(),
        text: "t".into(),
        anchors: vec![anchor],
        keys: Vec::new(),
        links: Vec::new(),
        validity: Validity::default(),
        retention: Retention::default(),
        verdicts: Vec::new(),
    };
    let run = |r: &Record, text: Option<&str>| reconsolidate(r, &|_| text.map(str::to_owned));
    let kinds = |x: &Reconsolidated| x.events.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(",");
    let valid = run(&rec, Some(code));
    let moved = run(&rec, Some(&format!("// a\n{code}")));
    let changed = run(&rec, Some(&code.replace("x > 3", "x > 7")));
    let gone = run(&rec, None);
    let lost = run(&rec, Some("fn a() {}\nfn other() {}\n"));
    if !valid.events.is_empty()
        || kinds(&moved) != "relocate"
        || moved.needs_check
        || kinds(&changed) != "relocate"
        || !changed.needs_check
        || kinds(&gone) != "failed,state"
        || kinds(&lost) != "failed"
    {
        return Err(format!(
            "nmmem.record: reconsolidation valid [{}] moved [{}] changed [{}] gone [{}]",
            kinds(&valid),
            kinds(&moved),
            kinds(&changed),
            kinds(&gone)
        ));
    }
    // The source there, the part found by neither hash nor name: the second failure drops one
    // tier (2-active to 3-archive), never to invalid; a pinned memory does not move.
    rec.retention.failures = 1;
    let second = run(&rec, Some("fn a() {}\nfn other() {}\n"));
    let to = second.events.iter().find(|(k, _)| k == "tier").and_then(|(_, v)| v.get("to").ok().and_then(|t| t.string().ok().map(str::to_owned)));
    if kinds(&second) != "failed,tier" || to.as_deref() != Some(Tier::Archive.label()) {
        return Err(format!("nmmem.record: a second unconfirmed re-read gave [{}] to {to:?}", kinds(&second)));
    }
    rec.class = Class::Pinned;
    if kinds(&run(&rec, Some("fn a() {}\nfn other() {}\n"))) != "failed" {
        return Err("nmmem.record: a pinned memory moved on a failed re-read".into());
    }
    Ok(())
}
