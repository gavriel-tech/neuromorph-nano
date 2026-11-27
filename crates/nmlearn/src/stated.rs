//! Stated preferences: rules the user says in plain words, kept as a `preference` cartridge of the
//! cartridge library (layer 4 of the stacking order, `library::family_rank`), beside the learned
//! preference values (`pref.global`, `pref.key.*`, `crate::pref_fit`) and the values distillation
//! fills in. A stated preference is one more learning source next to usage signals and teacher
//! distillation; it uses the library's write path, versions, stacks, trial flag and plug state.
//!
//! Inputs: a message of the user's words and a default scope. Outputs: structured rules, each with
//! the words it came from (`said`), what it does, and how it is held (`Enforcement`): a transform
//! the Nano applies to text it writes itself, a check it reports on a draft, or guidance passed to
//! the user's agent. Parsing is deterministic: the same message always gives the same rules, and a
//! statement that matches no known form is kept verbatim as guidance, never dropped and never
//! guessed into an enforced rule.
//!
//! Storage: a cartridge holds its stated rules as settings values under their own prefix,
//! `pref.stated.<cartridge id>.r<NNN>`, one compact JSON rule each, so they live beside learned
//! `pref.*` values in the same cartridge without changing the `pref_fit` format. A new rule set is
//! always a new version (`Library::add_cartridge`, files are never written over); every stack that
//! held the earlier version gets a new version holding the new one, and a use that ran such a stack
//! is switched to it, so a plugged cartridge stays plugged.
//!
//! Invariants: transforms touch only text the Nano writes. Fenced code blocks, inline code spans,
//! quoted lines (`>`) and evidence lines (`path:line`, `@@`, `~~` headers) are never altered, so
//! citations stay byte-faithful. A replacement never contains the text it replaces, so applying the
//! rules twice gives the same text as applying them once.
use crate::library::{CartridgeSpec, Filter, Library, PREFERENCE, Stack, preference_use};
use crate::unit::{n, obj, s};
use nmjson::Value;
use std::collections::BTreeMap;

/// The knowledge domain of a cartridge written from stated rules.
pub const DOMAIN: &str = "stated-preferences";
/// The settings key prefix of stated rules.
pub const KEY_PREFIX: &str = "pref.stated.";
/// The stack use of a stated cartridge that has no stack of its own yet.
pub const USE_PREFIX: &str = "preference:stated:";
/// Rules one cartridge holds at most.
pub const MAX_RULES: usize = 64;
/// Characters one message may hold.
pub const MAX_TEXT: usize = 4000;
/// Characters a replaced or banned term may hold.
const MAX_TERM: usize = 64;
/// Violations reported per rule at most (the count beyond is reported as a number).
pub const MAX_VIOLATIONS: usize = 20;
/// Rules the `nm_context_pack` block lists at most.
pub const BLOCK_RULES: usize = 12;
/// A want's strength in hundredths: the least, the default and the most (0.1, 0.5 and 2.0).
pub const STRENGTH_MIN: u64 = 10;
pub const STRENGTH_DEFAULT: u64 = 50;
pub const STRENGTH_MAX: u64 = 200;
/// The strength "strongly" (and "always", "really", ...) gives, and "slightly" (and "a bit", ...).
const STRENGTH_STRONG: u64 = 100;
const STRENGTH_SLIGHT: u64 = 25;

/// A strength in hundredths as its decimal text, kept in range: 50 is `0.5`, 100 is `1`, 25 is
/// `0.25`.
#[must_use]
pub fn strength_text(c: u64) -> String {
    let c = c.clamp(STRENGTH_MIN, STRENGTH_MAX);
    match (c / 100, c % 100) {
        (w, 0) => format!("{w}"),
        (w, f) if f % 10 == 0 => format!("{w}.{}", f / 10),
        (w, f) => format!("{w}.{f:02}"),
    }
}

/// A strength from its decimal text (`0.1` to `2.0`), in hundredths.
/// # Errors
/// Not a number, or out of range.
pub fn parse_strength(t: &str) -> Result<u64, String> {
    let x: f64 = t.trim().parse().map_err(|_| format!("strength {t}: a number from 0.1 to 2.0"))?;
    if !x.is_finite() || !(0.1 - 1e-9..=2.0 + 1e-9).contains(&x) {
        return Err(format!("strength {t}: a number from 0.1 to 2.0"));
    }
    let c = (x * 100.0).round() as u64;
    Ok(c.clamp(STRENGTH_MIN, STRENGTH_MAX))
}

/// A strength from a JSON number or its text, in hundredths (`None`: neither, or out of range).
#[must_use]
pub fn strength_of(v: &Value) -> Option<u64> {
    match v {
        Value::Number(t) | Value::String(t) => parse_strength(t).ok(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// The schema.

/// A naming convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Case {
    Snake,
    Camel,
    Pascal,
    Kebab,
    Screaming,
}

impl Case {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Snake => "snake_case",
            Self::Camel => "camelCase",
            Self::Pascal => "PascalCase",
            Self::Kebab => "kebab-case",
            Self::Screaming => "SCREAMING_SNAKE_CASE",
        }
    }
    /// From a case word as people write it (`snake_case`, `camel case`, `PascalCase`, ...).
    #[must_use]
    pub fn parse(t: &str) -> Option<Self> {
        let w: String = t.to_ascii_lowercase().chars().filter(char::is_ascii_alphabetic).collect();
        match w.as_str() {
            "snakecase" | "lowersnakecase" => Some(Self::Snake),
            "camelcase" | "lowercamelcase" => Some(Self::Camel),
            "pascalcase" | "uppercamelcase" => Some(Self::Pascal),
            "kebabcase" | "dashcase" => Some(Self::Kebab),
            "screamingsnakecase" | "uppersnakecase" | "constantcase" | "screamingcase" => Some(Self::Screaming),
            _ => None,
        }
    }
    /// Whether `name` is written in this case (a leading underscore is allowed; one-letter names fit
    /// every case but kebab and screaming, which need their own letters).
    #[must_use]
    pub fn fits(self, name: &str) -> bool {
        let n = name.trim_start_matches('_');
        if n.is_empty() {
            return true;
        }
        let first_lower = n.chars().next().is_some_and(|c| c.is_ascii_lowercase());
        let first_upper = n.chars().next().is_some_and(|c| c.is_ascii_uppercase());
        match self {
            Self::Snake => n.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            Self::Screaming => n.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'),
            Self::Kebab => n.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            Self::Camel => first_lower && n.chars().all(|c| c.is_ascii_alphanumeric()),
            Self::Pascal => {
                let shouting = n.chars().count() > 1 && n.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit());
                first_upper && n.chars().all(|c| c.is_ascii_alphanumeric()) && !shouting
            }
        }
    }
}

/// What a rule does. Open-ended: a kind this build does not know reads as guidance
/// (`Rule::from_value`), so a cartridge written by a newer build still loads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A text transform: every `from` (a character or literal text) becomes `to`. `what` names it
    /// for people (for example "em dash (U+2014)").
    Replace { from: String, to: String, what: String },
    /// Vocabulary: the whole word or phrase `avoid` (any letter case) becomes `prefer` (empty: the
    /// word is taken out).
    Vocabulary { avoid: String, prefer: String },
    /// A word or phrase to avoid (kind `ban` in the stored form): reported where it appears, never
    /// rewritten.
    Ban { phrase: String },
    /// A want: a `term` (a word or phrase) or a `style` the parser knows (`STYLES`), exactly one of
    /// the two non-empty, with its `strength` in hundredths (`STRENGTH_MIN` to `STRENGTH_MAX`). A
    /// term is a bounded push toward it in the Writer's decoding and is noted where a draft lacks
    /// it; a style is guidance for the agent.
    Want { term: String, style: String, strength: u64 },
    /// Indentation: tabs, or spaces in steps of `width` (0: spaces of any width).
    Indent { tabs: bool, width: u64 },
    /// Lines of at most `chars` characters.
    MaxLine { chars: u64 },
    /// Names of `target` (functions, variables, types, constants, files, modules or names) in `case`.
    Naming { case: Case, target: String },
    /// Free text for the agent to follow, kept verbatim in the rule's `said`; `domain` tags it
    /// (for example `pcb`), empty for none.
    Guidance { domain: String },
}

/// How a rule is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Enforcement {
    /// Applied by the Nano to text it writes, and by `apply` to a draft.
    Transform,
    /// Detected and reported by `apply` with the line; the text is left as it is.
    Check,
    /// A bounded logit bias toward a wanted term in the Writer's decoding; `apply` notes where a
    /// draft lacks the term, as information, never as a violation.
    Bias,
    /// Passed to the user's agent (the `nm_context_pack` block, `apply`'s guidance list).
    Guidance,
}

impl Enforcement {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Transform => "enforced",
            Self::Check => "checked",
            Self::Bias => "biased",
            Self::Guidance => "guidance",
        }
    }
}

/// Where a rule applies: a project (none: every project), file kinds by extension (none: every
/// kind) and folders by path prefix (none: every folder). All empty is global.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scope {
    pub project: Option<String>,
    pub kinds: Vec<String>,
    pub folders: Vec<String>,
}

/// A plain label: letters, digits, `- _ .` and `/` for folders; at most 128 characters.
fn plain(t: &str, slash: bool) -> bool {
    !t.is_empty() && t.len() <= 128 && !t.contains("..") && t.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') || slash && c == '/')
}

impl Scope {
    /// The global scope.
    #[must_use]
    pub fn global() -> Self {
        Self::default()
    }
    /// Whether nothing narrows it.
    #[must_use]
    pub fn is_global(&self) -> bool {
        self.project.is_none() && self.kinds.is_empty() && self.folders.is_empty()
    }
    /// From its text form: `global`, or parts joined by `;`: `project:<key>`,
    /// `kinds:<ext>,<ext>` (with or without the dot) and `folders:<path>,<path>`.
    /// # Errors
    /// A part that does not read, named.
    pub fn parse(t: &str) -> Result<Self, String> {
        let mut out = Self::default();
        for part in t.split(';').map(str::trim).filter(|p| !p.is_empty() && *p != "global") {
            let (k, v) = part.split_once(':').ok_or_else(|| format!("scope {part}: use global, project:<key>, kinds:<ext>,<ext> or folders:<path>,<path>"))?;
            let items: Vec<&str> = v.split(',').map(str::trim).filter(|x| !x.is_empty()).collect();
            match k.trim() {
                "project" => {
                    let p = v.trim();
                    if !plain(p, false) {
                        return Err(format!("scope project:{p}: a project key as the projects list names it"));
                    }
                    out.project = Some(p.to_owned());
                }
                "kinds" | "kind" | "files" => {
                    for x in items {
                        let e = x.trim_start_matches("*.").trim_start_matches('.').to_ascii_lowercase();
                        if !plain(&e, false) {
                            return Err(format!("scope kinds:{x}: a file extension such as kicad_pcb"));
                        }
                        if !out.kinds.contains(&e) {
                            out.kinds.push(e);
                        }
                    }
                }
                "folders" | "folder" => {
                    for x in items {
                        let f = x.replace('\\', "/").trim_matches('/').to_owned();
                        if !plain(&f, true) {
                            return Err(format!("scope folders:{x}: a folder path inside the project such as docs or src/ui"));
                        }
                        if !out.folders.contains(&f) {
                            out.folders.push(f);
                        }
                    }
                }
                other => return Err(format!("scope {other}: use project, kinds or folders")),
            }
        }
        Ok(out)
    }
    /// The text form `parse` reads.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut parts = Vec::new();
        if let Some(p) = &self.project {
            parts.push(format!("project:{p}"));
        }
        if !self.kinds.is_empty() {
            parts.push(format!("kinds:{}", self.kinds.join(",")));
        }
        if !self.folders.is_empty() {
            parts.push(format!("folders:{}", self.folders.join(",")));
        }
        if parts.is_empty() { "global".to_owned() } else { parts.join("; ") }
    }
    /// A person's reading of it.
    #[must_use]
    pub fn words(&self) -> String {
        let mut parts = Vec::new();
        if let Some(p) = &self.project {
            parts.push(format!("project {p}"));
        }
        if !self.kinds.is_empty() {
            parts.push(format!("{} files", self.kinds.iter().map(|k| format!(".{k}")).collect::<Vec<_>>().join(", ")));
        }
        if !self.folders.is_empty() {
            parts.push(format!("under {}", self.folders.iter().map(|f| format!("{f}/")).collect::<Vec<_>>().join(", ")));
        }
        if parts.is_empty() { "everywhere".to_owned() } else { parts.join(", ") }
    }
    /// A statement's own scope over the cartridge's: its kinds and folders when it names any, else
    /// the cartridge's; its project when it names one, else the cartridge's.
    #[must_use]
    pub fn over(&self, base: &Self) -> Self {
        Self {
            project: self.project.clone().or_else(|| base.project.clone()),
            kinds: if self.kinds.is_empty() { base.kinds.clone() } else { self.kinds.clone() },
            folders: if self.folders.is_empty() { base.folders.clone() } else { self.folders.clone() },
        }
    }
    /// Whether it applies in `project` to `path`. A rule narrowed to kinds or folders applies only
    /// where a path is known and fits.
    #[must_use]
    pub fn matches(&self, project: Option<&str>, path: Option<&str>) -> bool {
        if let (Some(want), Some(p)) = (&self.project, project)
            && want != p
        {
            return false;
        }
        if self.kinds.is_empty() && self.folders.is_empty() {
            return true;
        }
        let Some(path) = path else { return false };
        let path = path.replace('\\', "/");
        let kind_ok = self.kinds.is_empty() || ext_of(&path).is_some_and(|e| self.kinds.contains(&e));
        let folder_ok = self.folders.is_empty() || self.folders.iter().any(|f| path == *f || path.starts_with(&format!("{f}/")) || path.contains(&format!("/{f}/")));
        kind_ok && folder_ok
    }
    fn to_value(&self) -> Value {
        obj(vec![
            ("project", self.project.as_deref().map_or(Value::Null, s)),
            ("kinds", Value::Array(self.kinds.iter().map(|k| s(k)).collect())),
            ("folders", Value::Array(self.folders.iter().map(|f| s(f)).collect())),
        ])
    }
    fn from_value(v: &Value) -> Self {
        let list = |k: &str| v.get(k).and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect();
        Self { project: v.get("project").and_then(Value::string).ok().map(str::to_owned), kinds: list("kinds"), folders: list("folders") }
    }
}

/// The lower-case extension of a path's file name.
fn ext_of(path: &str) -> Option<String> {
    path.rsplit('/').next().and_then(|f| f.rsplit_once('.')).map(|(_, e)| e.to_ascii_lowercase()).filter(|e| !e.is_empty())
}

/// One stated rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    /// `r1`, `r2`, ... within its cartridge.
    pub id: String,
    pub kind: Kind,
    /// The user's words it came from, verbatim.
    pub said: String,
    pub scope: Scope,
}

/// Text shown with its invisible or look-alike characters named.
fn shown(t: &str) -> String {
    if t.is_empty() {
        return "nothing".to_owned();
    }
    let named = t.chars().all(|c| !c.is_ascii() || c.is_ascii_punctuation() || c == ' ');
    if named { format!("\"{t}\" ({})", t.chars().map(|c| format!("U+{:04X}", u32::from(c))).collect::<Vec<_>>().join(" ")) } else { format!("\"{t}\"") }
}

impl Rule {
    #[must_use]
    pub const fn enforcement(&self) -> Enforcement {
        match self.kind {
            Kind::Replace { .. } | Kind::Vocabulary { .. } => Enforcement::Transform,
            Kind::Ban { .. } | Kind::Indent { .. } | Kind::MaxLine { .. } | Kind::Naming { .. } => Enforcement::Check,
            Kind::Want { ref term, .. } if !term.is_empty() => Enforcement::Bias,
            Kind::Want { .. } | Kind::Guidance { .. } => Enforcement::Guidance,
        }
    }
    /// The kind's name.
    #[must_use]
    pub const fn kind_name(&self) -> &'static str {
        match self.kind {
            Kind::Replace { .. } => "replace",
            Kind::Vocabulary { .. } => "vocabulary",
            Kind::Ban { .. } => "ban",
            Kind::Want { .. } => "want",
            Kind::Indent { .. } => "indent",
            Kind::MaxLine { .. } => "max_line",
            Kind::Naming { .. } => "naming",
            Kind::Guidance { .. } => "guidance",
        }
    }
    /// The kind as people read it: `avoid` for a ban, else the kind's name. The stored kind
    /// (`kind_name`) is unchanged.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self.kind {
            Kind::Ban { .. } => "avoid",
            _ => self.kind_name(),
        }
    }
    /// A want's strength in hundredths (`None` for every other kind).
    #[must_use]
    pub const fn strength(&self) -> Option<u64> {
        match self.kind {
            Kind::Want { strength, .. } => Some(strength),
            _ => None,
        }
    }
    /// Sets a want's strength (in hundredths, kept in `STRENGTH_MIN..=STRENGTH_MAX`).
    /// # Errors
    /// The rule is not a want.
    pub fn set_strength(&mut self, c: u64) -> Result<(), String> {
        match &mut self.kind {
            Kind::Want { strength, .. } => {
                *strength = c.clamp(STRENGTH_MIN, STRENGTH_MAX);
                Ok(())
            }
            _ => Err(format!("rule {}: a {} rule has no strength; only a want has one", self.id, self.label())),
        }
    }
    /// What it does, in one line for a person.
    #[must_use]
    pub fn does(&self) -> String {
        match &self.kind {
            Kind::Replace { from, to, what } => {
                let w = if what.is_empty() { shown(from) } else { what.clone() };
                if to.is_empty() { format!("removes every {w}") } else { format!("replaces every {w} with {}", shown(to)) }
            }
            Kind::Vocabulary { avoid, prefer } if prefer.is_empty() => format!("takes the word \"{avoid}\" out"),
            Kind::Vocabulary { avoid, prefer } => format!("writes \"{prefer}\" in place of \"{avoid}\""),
            Kind::Ban { phrase } => format!("reports every \"{phrase}\" (to avoid)"),
            Kind::Want { term, strength, .. } if !term.is_empty() => {
                format!("pushes the Writer toward \"{term}\" (strength {}); preference_apply notes a draft without it", strength_text(*strength))
            }
            Kind::Want { style, strength, .. } => format!("passed to your agent as a style to prefer: {style} (strength {})", strength_text(*strength)),
            Kind::Indent { tabs: true, .. } => "reports code lines indented with spaces (indent with tabs)".to_owned(),
            Kind::Indent { width: 0, .. } => "reports code lines indented with tabs (indent with spaces)".to_owned(),
            Kind::Indent { width, .. } => format!("reports code lines not indented in steps of {width} spaces"),
            Kind::MaxLine { chars } => format!("reports lines longer than {chars} characters"),
            Kind::Naming { case, target } => format!("reports {target} not named in {}", case.name()),
            Kind::Guidance { domain } if domain.is_empty() => "passed to your agent as written".to_owned(),
            Kind::Guidance { domain } => format!("passed to your agent as written ({domain} convention)"),
        }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut f = vec![("id", s(&self.id)), ("kind", s(self.kind_name())), ("said", s(&self.said)), ("scope", self.scope.to_value())];
        match &self.kind {
            Kind::Replace { from, to, what } => f.extend([("from", s(from)), ("to", s(to)), ("what", s(what))]),
            Kind::Vocabulary { avoid, prefer } => f.extend([("avoid", s(avoid)), ("prefer", s(prefer))]),
            Kind::Ban { phrase } => f.push(("phrase", s(phrase))),
            Kind::Want { term, style, strength } => f.extend([("term", s(term)), ("style", s(style)), ("strength", Value::Number(strength_text(*strength)))]),
            Kind::Indent { tabs, width } => f.extend([("tabs", Value::Bool(*tabs)), ("width", n(*width))]),
            Kind::MaxLine { chars } => f.push(("chars", n(*chars))),
            Kind::Naming { case, target } => f.extend([("case", s(case.name())), ("target", s(target))]),
            Kind::Guidance { domain } => f.push(("domain", s(domain))),
        }
        obj(f)
    }
    /// The rule as a reply row: its fields, what it does, how it is held and its scope in words.
    #[must_use]
    pub fn view(&self) -> Value {
        let mut v = self.to_value();
        if let Value::Object(m) = &mut v {
            m.insert("does".into(), s(&self.does()));
            m.insert("label".into(), s(self.label()));
            m.insert("enforcement".into(), s(self.enforcement().name()));
            m.insert("scope_text".into(), s(&self.scope.to_text()));
            m.insert("scope_words".into(), s(&self.scope.words()));
        }
        v
    }
    /// A rule from its stored form. A kind this build does not know, or a field that does not read,
    /// is kept as guidance with its words, so nothing stated is ever lost or enforced by a guess.
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let t = |k: &str| v.get(k).and_then(Value::string).unwrap_or("").to_owned();
        let said = t("said");
        let kind = match t("kind").as_str() {
            "replace" if !t("from").is_empty() => Kind::Replace { from: t("from"), to: t("to"), what: t("what") },
            "vocabulary" if !t("avoid").is_empty() => Kind::Vocabulary { avoid: t("avoid"), prefer: t("prefer") },
            "ban" if !t("phrase").is_empty() => Kind::Ban { phrase: t("phrase") },
            "want" if t("term").is_empty() != t("style").is_empty() => {
                let strength = v.get("strength").ok().and_then(strength_of).unwrap_or(STRENGTH_DEFAULT);
                Kind::Want { term: t("term"), style: t("style"), strength }
            }
            "indent" => Kind::Indent { tabs: matches!(v.get("tabs"), Ok(Value::Bool(true))), width: v.get("width").and_then(Value::integer).unwrap_or(0) },
            "max_line" if v.get("chars").and_then(Value::integer).is_ok_and(|c| c > 0) => Kind::MaxLine { chars: v.get("chars").and_then(Value::integer).unwrap_or(0) },
            "naming" if Case::parse(&t("case")).is_some() => Kind::Naming { case: Case::parse(&t("case")).unwrap_or(Case::Snake), target: t("target") },
            _ => Kind::Guidance { domain: t("domain") },
        };
        Self { id: t("id"), kind, said, scope: v.get("scope").map(Scope::from_value).unwrap_or_default() }
    }
}

// ---------------------------------------------------------------------------------------------
// The parser.

/// Characters people name in words: the names, the character, what replaces it when the user says
/// only to remove it, and its name for the preview. A removed dash becomes punctuation that keeps
/// the sentence readable; the user can name any other replacement ("replace em dashes with -").
const NAMED: &[(&[&str], &str, &str, &str)] = &[
    (&["em dash", "em dashes", "em-dash", "em-dashes", "emdash", "emdashes", "\u{2014}"], "\u{2014}", ", ", "em dash (U+2014)"),
    (&["en dash", "en dashes", "en-dash", "en-dashes", "endash", "endashes", "\u{2013}"], "\u{2013}", "-", "en dash (U+2013)"),
    (&["ellipsis character", "ellipsis characters", "unicode ellipsis", "\u{2026}"], "\u{2026}", "...", "ellipsis character (U+2026)"),
    (&["non-breaking space", "non-breaking spaces", "nbsp", "nbsps"], "\u{a0}", " ", "non-breaking space (U+00A0)"),
    (&["left double quote", "left double quotes", "\u{201c}"], "\u{201c}", "\"", "left curly double quote (U+201C)"),
    (&["right double quote", "right double quotes", "\u{201d}"], "\u{201d}", "\"", "right curly double quote (U+201D)"),
];
/// Names that stand for several characters.
const NAMED_GROUPS: &[(&[&str], &[&str])] = &[
    (&["smart quotes", "curly quotes", "smart quote", "curly quote", "typographic quotes"], &["\u{201c}", "\u{201d}", "\u{2018}", "\u{2019}"]),
    (&["fancy dashes", "long dashes", "unicode dashes"], &["\u{2014}", "\u{2013}"]),
];
const SINGLE_QUOTES: [(&str, &str, &str); 2] = [("\u{2018}", "'", "left curly single quote (U+2018)"), ("\u{2019}", "'", "right curly single quote (U+2019)")];

/// The replace rules for a named character or group, with `to` in place of the default when given.
fn named(t: &str, to: Option<&str>) -> Option<Vec<Kind>> {
    let t = t.trim().trim_end_matches('.').trim();
    let one = |ch: &str| -> Option<Kind> {
        NAMED
            .iter()
            .find(|(_, c, _, _)| *c == ch)
            .map(|(_, c, d, w)| Kind::Replace { from: (*c).to_owned(), to: to.unwrap_or(d).to_owned(), what: (*w).to_owned() })
            .or_else(|| SINGLE_QUOTES.iter().find(|(c, _, _)| *c == ch).map(|(c, d, w)| Kind::Replace { from: (*c).to_owned(), to: to.unwrap_or(d).to_owned(), what: (*w).to_owned() }))
    };
    let low = t.to_lowercase();
    let low = low.trim_start_matches("all ").trim_start_matches("the ").trim_start_matches("any ");
    if let Some((_, chars)) = NAMED_GROUPS.iter().find(|(names, _)| names.contains(&low)) {
        return chars.iter().map(|c| one(c)).collect();
    }
    NAMED.iter().find(|(names, _, _, _)| names.contains(&low)).and_then(|(_, c, _, _)| one(c)).map(|k| vec![k])
}

/// Text in quotes ("...", '...', `...`, curly quotes) at the start of `t`, unquoted; what follows
/// the closing quote is left out.
fn quoted(t: &str) -> Option<String> {
    let t = t.trim();
    let pairs = [('"', '"'), ('\'', '\''), ('`', '`'), ('\u{201c}', '\u{201d}'), ('\u{2018}', '\u{2019}')];
    for (a, b) in pairs {
        if let Some(body) = t.strip_prefix(a)
            && let Some(end) = body.find(b)
            && end > 0
        {
            return body.get(..end).map(str::to_owned);
        }
    }
    None
}

/// A term as written: unquoted when quoted, else trimmed of trailing punctuation and of a leading
/// "the word" or "the phrase".
fn term(t: &str) -> String {
    quoted(t).unwrap_or_else(|| {
        let x = t.trim().trim_end_matches(['.', '!', ',']).trim();
        let low = x.to_lowercase();
        let cut = ["the word ", "the phrase ", "the term ", "words ", "word "].iter().find_map(|p| low.starts_with(p).then(|| x.get(p.len()..).unwrap_or(x)));
        let x = cut.unwrap_or(x).trim();
        quoted(x).unwrap_or_else(|| x.to_owned())
    })
}

fn has_alnum(t: &str) -> bool {
    t.chars().any(char::is_alphanumeric)
}

/// Splits a message into statements: at line breaks, semicolons and sentence ends, never inside
/// double quotes or backticks.
fn statements(message: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let (mut dq, mut bt) = (false, false);
    let chars: Vec<char> = message.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        match c {
            '"' | '\u{201c}' | '\u{201d}' if !bt => dq = !dq,
            '`' if !dq => bt = !bt,
            _ => {}
        }
        let next = chars.get(i + 1).copied();
        let end_sentence = matches!(c, '.' | '!' | '?') && next.is_none_or(char::is_whitespace);
        if !dq && !bt && (matches!(c, '\n' | ';') || end_sentence) {
            if !matches!(c, '\n' | ';') {
                cur.push(*c);
            }
            out.push(std::mem::take(&mut cur));
            continue;
        }
        cur.push(*c);
    }
    out.push(cur);
    out.into_iter().map(|x| x.trim().trim_start_matches(['-', '*']).trim().to_owned()).filter(|x| x.chars().any(char::is_alphanumeric) || !x.is_ascii()).collect()
}

/// Domain words and the file kinds they stand for.
const DOMAINS: &[(&[&str], &str, &[&str])] = &[
    (&["pcb", "pcbs", "kicad", "schematic", "schematics", "electrical", "circuit", "circuits", "board layout", "board layouts"], "pcb", &["kicad_pcb", "kicad_sch", "kicad_pro", "kicad_sym", "kicad_mod"]),
    (&["rust"], "rust", &["rs"]),
    (&["python"], "python", &["py"]),
    (&["typescript"], "typescript", &["ts", "tsx"]),
    (&["javascript"], "javascript", &["js", "jsx", "mjs"]),
    (&["markdown", "documentation"], "docs", &["md"]),
    (&["powershell"], "powershell", &["ps1"]),
    (&["shell scripts", "bash"], "shell", &["sh"]),
];

/// What a scope clause names: (scope, domain tag).
fn clause_scope(clause: &str, project: Option<&str>) -> Option<(Scope, String)> {
    let low = clause.to_lowercase();
    let words: Vec<&str> = low.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()).collect();
    let mut sc = Scope::default();
    let mut domain = String::new();
    for w in &words {
        let w = w.trim_end_matches(['.', ':', ')']).trim_start_matches(['(', '*']);
        // German joins the file noun to the extension (".kicad_pcb-Dateien").
        let w = ["-dateien", "-datei"].iter().find_map(|x| w.strip_suffix(x)).unwrap_or(w);
        if let Some(e) = w.strip_prefix('.').filter(|e| plain(e, false) && e.chars().any(|c| c.is_ascii_alphabetic())) {
            sc.kinds.push(e.to_owned());
        }
    }
    for (names, tag, kinds) in DOMAINS {
        if names.iter().any(|nm| words.windows(nm.split(' ').count()).any(|win| win.join(" ").trim_end_matches(['.', ':']) == *nm)) {
            domain = (*tag).to_owned();
            if sc.kinds.is_empty() {
                sc.kinds = kinds.iter().map(|k| (*k).to_owned()).collect();
            }
        }
    }
    for (i, w) in words.iter().enumerate() {
        let prev = i.checked_sub(1).and_then(|j| words.get(j)).copied().unwrap_or("");
        let f = w.trim_end_matches(['.', ':']);
        if (f.ends_with('/') && f.len() > 1 || matches!(prev, "folder" | "directory" | "dir")) && plain(f.trim_matches('/'), true) && !f.starts_with('.') {
            sc.folders.push(f.trim_matches('/').to_owned());
        }
        if prev == "project" && plain(f, false) && f != "this" {
            sc.project = Some(f.to_owned());
        }
        if (prev == "this" && f == "project") && project.is_some() {
            sc.project = project.map(str::to_owned);
        }
    }
    (!sc.is_global() || !domain.is_empty()).then_some((sc, domain))
}

/// The statement with its scope clause taken off: a leading clause ending in a comma ("For PCB
/// designs, ...") or a trailing one led by for, in, when, on, under or within.
fn split_scope(stmt: &str, project: Option<&str>) -> (String, Scope, String) {
    let t = stmt.trim();
    if let Some((head, rest)) = t.split_once(',') {
        let hl = head.to_lowercase();
        if ["for ", "in ", "when ", "on ", "under ", "within ", "with "].iter().any(|p| hl.starts_with(p))
            && let Some((sc, d)) = clause_scope(head, project)
        {
            return (rest.trim().to_owned(), sc, d);
        }
    }
    let low = t.to_lowercase();
    let leads = [" for ", " in ", " when ", " on ", " under ", " within ", " inside "];
    let mut best: Option<usize> = None;
    for l in leads {
        let mut from = 0;
        while let Some(i) = low.get(from..).and_then(|x| x.find(l)).map(|i| i + from) {
            if clause_scope(low.get(i..).unwrap_or(""), project).is_some() && best.is_none_or(|b| i < b) {
                best = Some(i);
                break;
            }
            from = i + 1;
        }
    }
    if let Some(i) = best
        && let (Some(head), Some(tail)) = (t.get(..i), t.get(i..))
        && let Some((sc, d)) = clause_scope(tail, project)
    {
        return (head.trim().trim_end_matches(',').to_owned(), sc, d);
    }
    other_split_scope(t, project).unwrap_or_else(|| (t.to_owned(), Scope::default(), String::new()))
}

// Scope clauses in Spanish, French, German and Portuguese. Tried only when no English clause
// scopes a statement (`split_scope`), so an English statement keeps its scope. A clause is read
// by `clause_scope` once its words are put in the English words it reads: folder words become
// "folder", project words "project", "this" words "this".

/// Leads of a scope clause in the other languages (folded).
const OTHER_SCOPE_LEADS: &[&str] = &[
    "para ", "en ", "dentro de ", "bajo ", "pour ", "dans ", "sous ", "fur ", "fuer ", "im ", "in ", "bei ", "unter ", "innerhalb von ", "em ", "nos ", "nas ", "no ", "na ", "sob ",
];
/// Words for a folder in the other languages (folded).
const OTHER_FOLDER_WORDS: &[&str] = &["carpeta", "directorio", "dossier", "repertoire", "ordner", "verzeichnis", "pasta", "diretorio"];
/// Words for a project in the other languages (folded).
const OTHER_PROJECT_WORDS: &[&str] = &["proyecto", "projet", "projekt", "projeto"];
/// Words for "this" before a project in the other languages (folded).
const OTHER_THIS_WORDS: &[&str] = &["este", "esta", "ce", "cet", "dieses", "diesem", "dieser", "neste", "nesse"];
/// Nouns for files that may follow an extension and end a clause ("para .rs archivos").
const OTHER_FILE_NOUNS: &[&str] = &["archivos", "archivo", "ficheros", "fichero", "fichiers", "fichier", "dateien", "datei", "arquivos", "arquivo"];

/// A folded clause of another language as `clause_scope` reads it.
fn other_clause_scope(low: &str, project: Option<&str>) -> Option<(Scope, String)> {
    let words: Vec<&str> = low
        .split_whitespace()
        .map(|w| {
            let core = w.trim_end_matches([',', ':', ')']);
            if OTHER_FOLDER_WORDS.contains(&core) {
                "folder"
            } else if OTHER_PROJECT_WORDS.contains(&core) {
                "project"
            } else if OTHER_THIS_WORDS.contains(&core) {
                "this"
            } else {
                w
            }
        })
        .collect();
    clause_scope(&words.join(" "), project)
}

/// Whether a folded word ends a scope clause: an extension (".rs", ".kicad_pcb-dateien") or a
/// folder ("docs/"), or follows a folder word.
fn scope_token(w: &str, prev: &str) -> bool {
    let w = w.trim_end_matches([',', ':', ')']);
    let ext = w.strip_prefix('.').is_some_and(|e| e.chars().any(|c| c.is_ascii_alphabetic()));
    ext || w.len() > 1 && w.ends_with('/') || OTHER_FOLDER_WORDS.contains(&prev) || OTHER_PROJECT_WORDS.contains(&prev)
}

/// The statement with a scope clause of another language taken off: a leading clause ended by a
/// comma, a leading clause ended by its extension, folder or project name ("para archivos .rs usa
/// 4 espacios"), or a trailing clause ("sin rayas en docs/"); `None` when it holds none.
fn other_split_scope(t: &str, project: Option<&str>) -> Option<(String, Scope, String)> {
    let f = Folded::new(t.trim());
    let n = f.orig.len();
    let leads = |x: &str| OTHER_SCOPE_LEADS.iter().any(|p| x.starts_with(p));
    if let Some(ci) = f.low.find(',')
        && let Some(head) = f.low.get(..ci)
        && leads(head)
        && let Some((sc, d)) = other_clause_scope(head, project)
    {
        let rest = f.text(f.chars_before(ci) + 1, n).trim().to_owned();
        if !rest.is_empty() {
            return Some((rest, sc, d));
        }
    }
    if leads(&f.low) {
        let words: Vec<&str> = f.low.split(' ').collect();
        for i in 1..words.len().min(7) {
            let (Some(w), Some(prev)) = (words.get(i), words.get(i - 1)) else { break };
            if !scope_token(w, prev) {
                continue;
            }
            let end = if words.get(i + 1).is_some_and(|x| OTHER_FILE_NOUNS.contains(x)) { i + 1 } else { i };
            let clause = words.get(..=end).map(|x| x.join(" ")).unwrap_or_default();
            let rest = f.text(clause.chars().count(), n).trim().to_owned();
            if let Some((sc, d)) = other_clause_scope(&clause, project)
                && !rest.is_empty()
            {
                return Some((rest, sc, d));
            }
            break;
        }
    }
    let mut best: Option<usize> = None;
    for l in OTHER_SCOPE_LEADS {
        let pat = format!(" {l}");
        let mut from = 0;
        while let Some(i) = f.low.get(from..).and_then(|x| x.find(&pat)).map(|i| i + from) {
            if other_clause_scope(f.low.get(i..).unwrap_or(""), project).is_some() && best.is_none_or(|b| i < b) {
                best = Some(i);
                break;
            }
            from = i + 1;
        }
    }
    let i = best?;
    let (sc, d) = other_clause_scope(f.low.get(i..).unwrap_or(""), project)?;
    let head = f.text(0, f.chars_before(i)).trim().trim_end_matches(',').trim().to_owned();
    (!head.is_empty()).then_some((head, sc, d))
}

/// Lead-in words that change nothing in what a statement asks.
fn strip_lead(t: &str) -> &str {
    let mut x = t.trim();
    loop {
        let low = x.to_lowercase();
        let lead = ["please ", "always ", "i want you to ", "i want ", "i'd like you to ", "i'd like ", "i would like ", "make sure to ", "make sure you ", "you should ", "you must ", "we ", "i "]
            .iter()
            .find(|p| low.starts_with(**p));
        match lead {
            Some(p) => x = x.get(p.len()..).unwrap_or("").trim_start(),
            None => return x.trim_end_matches(['.', '!']).trim(),
        }
    }
}

/// `low` split at the first of `seps`: (before, after) in the original letters of `t`.
fn split_at<'a>(t: &'a str, low: &str, seps: &[&str]) -> Option<(&'a str, &'a str)> {
    seps.iter().filter_map(|sep| low.find(sep).map(|i| (i, sep.len()))).min_by_key(|x| x.0).and_then(|(i, l)| Some((t.get(..i)?, t.get(i + l..)?)))
}

/// A preferred and an avoided term: "use A instead of B", "prefer A over B", "replace B with A",
/// "A not B", "B -> A".
fn swap_form(t: &str) -> Option<(String, String)> {
    let low = t.to_lowercase();
    for (lead, seps, avoid_first) in [
        (&["replace ", "change ", "swap "][..], &[" with ", " to ", " for ", " by "][..], true),
        (&["use ", "say ", "write ", "prefer ", ""][..], &[" instead of ", " rather than ", " over ", " in place of ", " not "][..], false),
    ] {
        let Some(l) = lead.iter().find(|p| low.starts_with(**p)) else { continue };
        let (body, lbody) = (t.get(l.len()..)?, low.get(l.len()..)?);
        if let Some((a, b)) = split_at(body, lbody, seps) {
            let (prefer, avoid) = if avoid_first { (b, a) } else { (a, b) };
            let (p, v) = (term(prefer), term(avoid));
            if !v.is_empty() && v.chars().count() <= MAX_TERM && p.chars().count() <= MAX_TERM {
                return Some((p, v));
            }
        }
    }
    for arrow in ["->", "=>", "\u{2192}"] {
        if let Some((a, b)) = t.split_once(arrow) {
            let (v, p) = (term(a), term(b));
            if !v.is_empty() && v.chars().count() <= MAX_TERM && p.chars().count() <= MAX_TERM {
                return Some((p, v));
            }
        }
    }
    None
}

fn number_in(low: &str) -> Option<u64> {
    low.split(|c: char| !c.is_ascii_digit()).filter(|x| !x.is_empty()).find_map(|x| x.parse().ok())
}

fn indent_form(low: &str) -> Option<Kind> {
    let about_indent = low.contains("indent");
    let tabs_word = low.contains("tab");
    let spaces_word = low.contains("space");
    if !about_indent && !(tabs_word && spaces_word) && !matches!(low, "use tabs" | "tabs") {
        return None;
    }
    // The first named of tabs and spaces is the one wanted ("tabs not spaces", "4 spaces instead of tabs").
    let tab_at = low.find("tab");
    let space_at = low.find("space");
    let tabs = match (tab_at, space_at) {
        (Some(t), Some(sp)) => t < sp,
        (Some(_), None) => true,
        _ => false,
    };
    if tabs {
        return Some(Kind::Indent { tabs: true, width: 0 });
    }
    if spaces_word {
        let w = number_in(low).filter(|w| (1..=16).contains(w)).unwrap_or(0);
        return Some(Kind::Indent { tabs: false, width: w });
    }
    let w = number_in(low).filter(|w| (1..=16).contains(w))?;
    Some(Kind::Indent { tabs: false, width: w })
}

fn max_line_form(low: &str) -> Option<Kind> {
    let line = low.contains("line") || low.contains("wrap") || low.contains("column");
    let length = ["length", "long", "width", "characters", "chars", "columns", "wrap", "under", "max", "at most"].iter().any(|w| low.contains(w));
    if !(line && length) {
        return None;
    }
    let c = number_in(low).filter(|c| (20..=1000).contains(c))?;
    Some(Kind::MaxLine { chars: c })
}

fn naming_form(t: &str, low: &str) -> Option<Kind> {
    // The case named before "instead of", "not", "rather than" or "over" is the one wanted.
    let wanted = split_at(t, low, &[" instead of ", " rather than ", " not ", " over "]).map_or(t, |(a, _)| a);
    let words: Vec<&str> = wanted.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()).collect();
    let mut case = None;
    for i in 0..words.len() {
        let one = words.get(i).copied().unwrap_or("");
        let two = format!("{one} {}", words.get(i + 1).copied().unwrap_or(""));
        let three = format!("{two} {}", words.get(i + 2).copied().unwrap_or(""));
        if let Some(c) = Case::parse(one).filter(|_| one.contains(['_', '-']) || one.chars().skip(1).any(|c| c.is_ascii_uppercase()) || one.to_lowercase().ends_with("case")).or_else(|| Case::parse(&two).filter(|_| two.to_lowercase().contains("case"))).or_else(|| Case::parse(&three).filter(|_| three.to_lowercase().contains("case"))) {
            case = Some(c);
            break;
        }
    }
    let case = case?;
    let target = [
        (&["function", "method", "fn "][..], "functions"),
        (&["variable", "local", "field"][..], "variables"),
        (&["type", "class", "struct", "enum", "interface", "trait"][..], "types"),
        (&["constant", "const "][..], "constants"),
        (&["file", "filename"][..], "files"),
        (&["module", "crate", "package"][..], "modules"),
    ]
    .iter()
    .find(|(ws, _)| ws.iter().any(|w| low.contains(w)))
    .map_or("names", |(_, t)| *t);
    Some(Kind::Naming { case, target: target.to_owned() })
}

/// A ban: "never say X", "don't say X", "do not write X", "ban the word X", "avoid the word X".
fn ban_form(t: &str, low: &str) -> Option<String> {
    let leads = [
        "never say ", "don't say ", "dont say ", "do not say ", "stop saying ", "never write ", "don't write ", "do not write ", "never use the word ", "never use the phrase ",
        "don't use the word ", "do not use the word ", "don't use the phrase ", "do not use the phrase ", "avoid the word ", "avoid the phrase ", "avoid saying ", "ban the word ", "ban the phrase ", "ban ",
    ];
    let l = leads.iter().find(|p| low.starts_with(**p))?;
    let p = term(t.get(l.len()..)?);
    (!p.is_empty() && p.chars().count() <= MAX_TERM && has_alnum(&p)).then_some(p)
}

/// A removal: "no X", "remove X", "strip X", "don't use X", "avoid X", "without X" with X a named
/// character or quoted text (or, for "never use" and "avoid", a word: a ban).
fn remove_form(t: &str, low: &str) -> Option<Vec<Kind>> {
    let leads = ["remove all ", "remove any ", "remove ", "strip all ", "strip ", "delete all ", "delete ", "no more ", "no ", "never use ", "don't use ", "dont use ", "do not use ", "avoid ", "without ", "get rid of "];
    let l = leads.iter().find(|p| low.starts_with(**p))?;
    let rest = t.get(l.len()..)?.trim();
    if let Some(k) = named(rest, None) {
        return Some(k);
    }
    let removing = matches!(*l, "remove all " | "remove any " | "remove " | "strip all " | "strip " | "delete all " | "delete " | "get rid of ");
    let lit = quoted(rest)?;
    if lit.chars().count() > MAX_TERM {
        return None;
    }
    Some(vec![match (removing, has_alnum(&lit)) {
        (_, false) => Kind::Replace { from: lit.clone(), to: String::new(), what: String::new() },
        (true, true) => Kind::Vocabulary { avoid: lit, prefer: String::new() },
        (false, true) => Kind::Ban { phrase: lit },
    }])
}

/// A preferred and avoided pair as rules: named characters and text without letters are literal
/// replacements, words and phrases are vocabulary. A replacement that holds what it replaces is
/// refused (it would never settle).
fn swap_kinds(prefer: &str, avoid: &str) -> Option<Vec<Kind>> {
    let to = named(prefer, None).and_then(|k| match k.first() {
        Some(Kind::Replace { from, .. }) if k.len() == 1 => Some(from.clone()),
        _ => None,
    });
    let to_text = to.unwrap_or_else(|| prefer.to_owned());
    if let Some(k) = named(avoid, Some(&to_text)) {
        return k.iter().all(|x| matches!(x, Kind::Replace { from, to, .. } if !to.contains(from.as_str()))).then_some(k);
    }
    if to_text.to_lowercase().contains(&avoid.to_lowercase()) {
        return None;
    }
    if has_alnum(avoid) {
        Some(vec![Kind::Vocabulary { avoid: avoid.to_owned(), prefer: to_text }])
    } else {
        Some(vec![Kind::Replace { from: avoid.to_owned(), to: to_text, what: String::new() }])
    }
}

/// The English forms of one statement; `None` when none matches.
fn english_forms(body: &str) -> Option<Vec<Kind>> {
    let t = strip_lead(body);
    let low = t.to_lowercase();
    if t.is_empty() {
        return None;
    }
    if let Some(k) = indent_form(&low) {
        return Some(vec![k]);
    }
    if let Some(k) = naming_form(t, &low) {
        return Some(vec![other_target(k, t)]);
    }
    if let Some(k) = max_line_form(&low) {
        return Some(vec![k]);
    }
    if let Some(p) = ban_form(t, &low) {
        return Some(vec![Kind::Ban { phrase: p }]);
    }
    if let Some(k) = remove_form(t, &low) {
        return Some(k);
    }
    if let Some((p, v)) = swap_form(t) {
        return swap_kinds(&p, &v);
    }
    // "never use X" / "avoid X" with a plain word or short phrase: a ban.
    for l in ["never use ", "don't use ", "do not use ", "avoid "] {
        if let Some(rest) = low.starts_with(l).then(|| t.get(l.len()..)).flatten() {
            let p = term(rest);
            if has_alnum(&p) && p.split_whitespace().count() <= 3 && p.chars().count() <= MAX_TERM {
                return Some(vec![Kind::Ban { phrase: p }]);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------
// Forms in Spanish, French, German and Portuguese.
//
// Rules for these forms:
// - They are tried only when no English form reads a statement (`forms`), and their conjunctions
//   only when no English form reads it whole (`parse`), so an English statement gives the rules it
//   gave before they existed.
// - Matching is on the statement lower-cased with the accents of these languages taken off
//   (`fold_char`), one character for one character, so a term is cut from the user's own letters
//   at the same character positions.
// - Each form gives the rule kinds of its English counterpart: no X or remove X (`other_remove`),
//   use A instead of B or replace B with A (`other_swap`), never say X (`other_ban`), indent with
//   N spaces or tabs (`other_indent`), lines of at most N characters (`other_max_line`), and a
//   naming case with its target named in these languages (`other_target`).
// - Characters named in these languages are mapped to the English names `named` knows
//   (`NAMED_OTHER`); nothing else is translated, and a statement no form reads stays verbatim
//   guidance.

/// One character lower-cased (when its lower case is one character) with its accent taken off:
/// the vowels with acute, grave, circumflex, tilde or diaeresis, c cedilla and n tilde. The right
/// single quote becomes the apostrophe ("n\u{2019}utilise").
fn fold_char(c: char) -> char {
    let mut lower = c.to_lowercase();
    let l = match (lower.next(), lower.next()) {
        (Some(x), None) => x,
        _ => c,
    };
    match l {
        '\u{e1}' | '\u{e0}' | '\u{e2}' | '\u{e3}' | '\u{e4}' => 'a',
        '\u{e9}' | '\u{e8}' | '\u{ea}' | '\u{eb}' => 'e',
        '\u{ed}' | '\u{ec}' | '\u{ee}' | '\u{ef}' => 'i',
        '\u{f3}' | '\u{f2}' | '\u{f4}' | '\u{f5}' | '\u{f6}' => 'o',
        '\u{fa}' | '\u{f9}' | '\u{fb}' | '\u{fc}' => 'u',
        '\u{e7}' => 'c',
        '\u{f1}' => 'n',
        '\u{2019}' => '\'',
        x => x,
    }
}

/// A statement and its folded form, character for character.
struct Folded {
    orig: Vec<char>,
    low: String,
}

impl Folded {
    fn new(t: &str) -> Self {
        let orig: Vec<char> = t.chars().collect();
        let low = orig.iter().copied().map(fold_char).collect();
        Self { orig, low }
    }
    /// The characters of `low` before byte `at`.
    fn chars_before(&self, at: usize) -> usize {
        self.low.get(..at).map_or(0, |x| x.chars().count())
    }
    /// The user's letters from character `from` to character `to`.
    fn text(&self, from: usize, to: usize) -> String {
        let n = self.orig.len();
        self.orig.get(from.min(n)..to.min(n)).unwrap_or_default().iter().collect()
    }
    /// The longest of `leads` the statement starts with, and the user's text after it.
    fn after_lead<'a>(&self, leads: &[&'a str]) -> Option<(&'a str, String)> {
        let l = leads.iter().copied().filter(|p| self.low.starts_with(p)).max_by_key(|p| p.len())?;
        Some((l, self.text(l.chars().count(), self.orig.len()).trim().to_owned()))
    }
    /// The user's text before and after the first of `seps` found from character `from`.
    fn split(&self, from: usize, seps: &[&str]) -> Option<(String, String)> {
        let start = self.low.char_indices().nth(from).map_or(self.low.len(), |(i, _)| i);
        let rest = self.low.get(start..)?;
        let (i, sep) = seps.iter().filter_map(|sep| rest.find(sep).map(|i| (i, *sep))).min_by_key(|x| x.0)?;
        let a = self.chars_before(start + i);
        let b = a + sep.chars().count();
        Some((self.text(from, a), self.text(b, self.orig.len())))
    }
}

/// Courtesy words at the start or end of a statement that change nothing in what it asks.
const POLITE_LEAD: &[&str] = &[
    "por favor, ", "por favor ", "s'il te plait, ", "s'il te plait ", "s'il vous plait, ", "s'il vous plait ", "bitte ", "siempre ", "toujours ", "immer ", "sempre ",
];
const POLITE_END: &[&str] = &[", por favor", " por favor", ", s'il te plait", " s'il te plait", ", s'il vous plait", " s'il vous plait", ", bitte", " bitte"];

/// The statement without inverted marks, courtesy words and its closing mark.
fn other_strip(t: &str) -> String {
    let mut x = t.trim().to_owned();
    loop {
        let y = x.trim_start_matches(['\u{a1}', '\u{bf}']).trim_end_matches(['.', '!', '?']).trim().to_owned();
        let f = Folded::new(&y);
        let z = if let Some(p) = POLITE_LEAD.iter().find(|p| f.low.starts_with(**p)) {
            f.text(p.chars().count(), f.orig.len()).trim().to_owned()
        } else if let Some(p) = POLITE_END.iter().find(|p| f.low.ends_with(**p)) {
            f.text(0, f.orig.len().saturating_sub(p.chars().count())).trim().to_owned()
        } else {
            y
        };
        if z == x {
            return x;
        }
        x = z;
    }
}

/// Characters named in Spanish, French, German and Portuguese (folded), and the English name
/// `named` reads for them. German "Gedankenstrich" names the dash of a sentence without saying
/// which, so it stands for both long dashes.
const NAMED_OTHER: &[(&[&str], &str)] = &[
    (
        &[
            "raya", "rayas", "guion largo", "guiones largos", "guion eme", "guiones eme", "tiret cadratin", "tirets cadratins", "tiret long", "tirets longs", "geviertstrich",
            "geviertstriche", "langer gedankenstrich", "lange gedankenstriche", "langen gedankenstriche", "travessao", "travessoes",
        ],
        "em dash",
    ),
    (
        &[
            "semirraya", "semirrayas", "guion medio", "guiones medios", "tiret demi-cadratin", "tirets demi-cadratins", "halbgeviertstrich", "halbgeviertstriche", "meia-risca",
            "meias-riscas", "meia risca", "meias riscas",
        ],
        "en dash",
    ),
    (&["gedankenstrich", "gedankenstriche"], "fancy dashes"),
    (
        &[
            "comillas tipograficas", "comillas curvas", "guillemets typographiques", "guillemets courbes", "typografische anfuhrungszeichen", "typografische anfuehrungszeichen",
            "geschwungene anfuhrungszeichen", "geschwungene anfuehrungszeichen", "aspas curvas", "aspas tipograficas", "aspas inteligentes",
        ],
        "smart quotes",
    ),
    (
        &[
            "espacio duro", "espacios duros", "espacio de no separacion", "espacios de no separacion", "espace insecable", "espaces insecables", "geschutztes leerzeichen",
            "geschutzte leerzeichen", "geschuetztes leerzeichen", "geschuetzte leerzeichen", "espaco nao separavel", "espacos nao separaveis", "espaco rigido", "espacos rigidos",
        ],
        "non-breaking spaces",
    ),
];
/// Articles and quantity words before a named character or a punctuation word.
const ARTICLES: &[&str] = &[
    "todos los ", "todas las ", "tous les ", "toutes les ", "todos os ", "todas as ", "ningun ", "ninguna ", "aucun ", "aucune ", "nenhum ", "nenhuma ", "alle ", "los ", "las ",
    "les ", "des ", "die ", "der ", "den ", "das ", "el ", "la ", "le ", "os ", "as ", "de ", "l'", "d'", "o ", "a ",
];
/// Words after a named character that change nothing ("keine Gedankenstriche mehr").
const TRAILING: &[&str] = &[" mehr", " nunca", " jamais", " nie", " niemals", " mais"];
/// Punctuation named as the replacement ("por comas", "durch Kommas") and the text it stands for.
const PUNCT_OTHER: &[(&[&str], &str)] = &[
    (&["coma", "comas", "virgule", "virgules", "komma", "kommas", "kommata", "virgula", "virgulas"], ", "),
    (&["guion", "guiones", "tiret", "tirets", "bindestrich", "bindestriche", "hifen", "hifens"], "-"),
];

/// A name or punctuation word folded, without articles, trailing words and closing marks.
fn bare(t: &str) -> String {
    let mut x: String = t.chars().map(fold_char).collect::<String>().trim().trim_end_matches(['.', '!', ',', ';']).trim().to_owned();
    loop {
        let before = x.len();
        if let Some(a) = ARTICLES.iter().find(|a| x.starts_with(**a)) {
            x = x.get(a.len()..).unwrap_or("").trim_start().to_owned();
        }
        if let Some(w) = TRAILING.iter().find(|w| x.ends_with(**w)) {
            x = x.get(..x.len() - w.len()).unwrap_or("").trim_end().to_owned();
        }
        if x.len() == before {
            return x;
        }
    }
}

/// The English name of a character named in another language, if `t` is one.
fn canon(t: &str) -> Option<&'static str> {
    let b = bare(t);
    NAMED_OTHER.iter().find(|(names, _)| names.contains(&b.as_str())).map(|(_, en)| *en)
}

/// `named` for a character named in English or in another language.
fn named_any(t: &str, to: Option<&str>) -> Option<Vec<Kind>> {
    named(t, to).or_else(|| canon(t).and_then(|en| named(en, to)))
}

/// A term in another language: "the word" and "the phrase" taken off, then as `term`.
fn other_term(t: &str) -> String {
    let leads = [
        "la palabra ", "la expresion ", "el termino ", "le mot ", "l'expression ", "le terme ", "das wort ", "den ausdruck ", "den begriff ", "a palavra ", "a expressao ", "o termo ",
    ];
    let f = Folded::new(t.trim());
    let x = f.after_lead(&leads).map_or_else(|| t.trim().to_owned(), |(_, rest)| rest);
    term(&x)
}

/// "Never say X" in another language: a ban.
fn other_ban(f: &Folded) -> Option<String> {
    let leads = [
        "nunca digas ", "no digas ", "nunca escribas ", "no escribas ", "nunca uses la palabra ", "no uses la palabra ", "nunca uses la expresion ", "no uses la expresion ",
        "evita la palabra ", "evita la expresion ", "ne dis jamais ", "ne dites jamais ", "ne dis pas ", "ne dites pas ", "n'ecris jamais ", "n'ecrivez jamais ", "n'ecris pas ",
        "n'ecrivez pas ", "n'utilise jamais le mot ", "n'utilisez jamais le mot ", "n'utilise pas le mot ", "n'utilisez pas le mot ", "evite le mot ", "evitez le mot ",
        "sag niemals ", "sage niemals ", "sag nie ", "sage nie ", "sag nicht ", "sage nicht ", "schreib niemals ", "schreibe niemals ", "schreib nie ", "schreibe nie ",
        "schreib nicht ", "schreibe nicht ", "verwende nie das wort ", "verwende niemals das wort ", "benutze nie das wort ", "benutze niemals das wort ", "vermeide das wort ",
        "nunca diga ", "nao diga ", "nunca escreva ", "nao escreva ", "nunca use a palavra ", "nao use a palavra ", "evite a palavra ", "evite a expressao ",
    ];
    let (_, rest) = f.after_lead(&leads)?;
    let p = other_term(&rest);
    (!p.is_empty() && p.chars().count() <= MAX_TERM && has_alnum(&p)).then_some(p)
}

/// Leads of another language that take something out of the text.
const OTHER_REMOVING: &[&str] = &[
    "elimina ", "eliminar ", "quita ", "quitar ", "borra ", "borrar ", "supprime ", "supprimez ", "supprimer ", "enleve ", "enlevez ", "enlever ", "retirez ", "retirer ",
    "entferne ", "entfernen ", "entfernt ", "losche ", "loesche ", "loschen ", "loeschen ", "remova ", "remover ", "elimine ", "apague ", "apagar ",
];
/// Leads of another language that forbid something.
const OTHER_FORBIDDING: &[&str] = &[
    "sin ", "nada de ", "no mas ", "no uses ", "no usar ", "nunca uses ", "evita ", "evitar ", "pas de ", "pas d'", "plus de ", "sans ", "aucun ", "aucune ", "n'utilise pas ",
    "n'utilisez pas ", "n'utilise jamais ", "n'utilisez jamais ", "ne pas utiliser ", "evite ", "evitez ", "eviter ", "keine ", "kein ", "ohne ", "verwende keine ", "verwende kein ",
    "benutze keine ", "benutze kein ", "nutze keine ", "verwende nie ", "verwende niemals ", "vermeide ", "vermeiden ", "sem ", "nao use ", "nao usar ", "nunca use ", "nenhum ",
    "nenhuma ",
];

/// Leads of `OTHER_FORBIDDING` that are also English words.
const OTHER_ENGLISH_WORDS: &[&str] = &["sin ", "plus de "];

/// "No X" or "remove X" in another language, as `remove_form`: a named character, or quoted text
/// (taken out of the text when the lead removes, reported as a ban when it forbids).
fn other_remove(f: &Folded) -> Option<Vec<Kind>> {
    let all: Vec<&str> = OTHER_REMOVING.iter().chain(OTHER_FORBIDDING).copied().collect();
    let (l, rest) = f.after_lead(&all)?;
    // A lead that is also an English word ("sin", "plus de") takes only a character named in
    // another language, so no English statement reads as one of these forms.
    if OTHER_ENGLISH_WORDS.contains(&l) {
        return canon(&rest).and_then(|en| named(en, None));
    }
    if let Some(k) = named_any(&rest, None) {
        return Some(k);
    }
    let lit = quoted(&rest)?;
    if lit.chars().count() > MAX_TERM {
        return None;
    }
    Some(vec![match (OTHER_REMOVING.contains(&l), has_alnum(&lit)) {
        (_, false) => Kind::Replace { from: lit.clone(), to: String::new(), what: String::new() },
        (true, true) => Kind::Vocabulary { avoid: lit, prefer: String::new() },
        (false, true) => Kind::Ban { phrase: lit },
    }])
}

/// Leads and separators of "replace B with A" in another language.
const SWAP_AVOID_FIRST: (&[&str], &[&str]) = (
    &[
        "reemplaza ", "reemplazar ", "sustituye ", "sustituir ", "cambia ", "cambiar ", "remplace ", "remplacez ", "remplacer ", "ersetze ", "ersetzen ", "ersetzt ", "substitua ",
        "substituir ", "substitui ", "troque ", "trocar ", "troca ",
    ],
    &[" por ", " con ", " par ", " avec ", " durch ", " mit ", " com "],
);
/// Leads and separators of "use A instead of B" in another language (the lead may be absent).
const SWAP_PREFER_FIRST: (&[&str], &[&str]) = (
    &[
        "usa ", "usar ", "utiliza ", "utilizar ", "emplea ", "escribe ", "di ", "prefiere ", "utilise ", "utilisez ", "utiliser ", "emploie ", "employez ", "ecris ", "ecrivez ",
        "dis ", "dites ", "prefere ", "preferez ", "verwende ", "verwenden ", "benutze ", "benutzen ", "nutze ", "nutzen ", "schreibe ", "schreib ", "sag ", "sage ", "bevorzuge ",
        "use ", "utilize ", "escreva ", "diga ", "prefira ", "",
    ],
    &[
        " en lugar de ", " en vez de ", " y no ", " au lieu de ", " au lieu d'", " plutot que d'", " plutot que ", " a la place de ", " a la place d'", " et non ", " et pas ",
        " anstelle von ", " anstelle ", " anstatt ", " statt ", ", nicht ", " und nicht ", " em vez de ", " ao inves de ", " no lugar de ", " em lugar de ", " e nao ",
    ],
);

/// "Use A instead of B" or "replace B with A" in another language: (preferred, avoided), with a
/// named character given by its English name and a named punctuation mark by its text.
fn other_swap(f: &Folded) -> Option<(String, String)> {
    for ((leads, seps), avoid_first) in [(SWAP_AVOID_FIRST, true), (SWAP_PREFER_FIRST, false)] {
        let Some(l) = leads.iter().copied().filter(|p| f.low.starts_with(p)).max_by_key(|p| p.len()) else { continue };
        let Some((a, b)) = f.split(l.chars().count(), seps) else { continue };
        let (prefer, avoid) = if avoid_first { (b, a) } else { (a, b) };
        let (p, v) = (other_term(&prefer), other_term(&avoid));
        if v.is_empty() || v.chars().count() > MAX_TERM || p.chars().count() > MAX_TERM {
            continue;
        }
        let punct = |x: &str| {
            let b = bare(x);
            PUNCT_OTHER.iter().find(|(names, _)| names.contains(&b.as_str())).map(|(_, to)| (*to).to_owned())
        };
        let p = punct(&p).or_else(|| canon(&p).map(str::to_owned)).unwrap_or(p);
        let v = canon(&v).map_or(v, str::to_owned);
        return Some((p, v));
    }
    None
}

/// Where a word for tabs starts in a folded statement: "tab", "tabs", or a word from "tabul"
/// (tabulaciones, tabuladores, tabulations, Tabulatoren, tabulacoes); "tabla" and "Tabelle" are
/// not.
fn tab_word_at(low: &str) -> Option<usize> {
    let mut at = 0;
    for w in low.split(|c: char| !c.is_alphanumeric()) {
        if w == "tab" || w == "tabs" || w.starts_with("tabul") {
            return Some(at);
        }
        at += w.len() + 1;
    }
    None
}

/// Indentation in another language, as `indent_form`: the first named of tabs and spaces is the
/// one wanted; spaces take the number given (1 to 16), and a number alone needs an indentation
/// word.
fn other_indent(low: &str) -> Option<Kind> {
    let about = ["indent", "sangr", "retrait", "einruck", "einrueck", "einzug", "recuo"].iter().any(|w| low.contains(w));
    let tab_at = tab_word_at(low);
    let space_at = ["espacio", "espace", "leerzeichen", "espaco"].iter().filter_map(|w| low.find(w)).min();
    let use_lead = ["usa ", "usar ", "utiliza ", "utilizar ", "utilise ", "utilisez ", "utiliser ", "verwende ", "verwenden ", "benutze ", "nutze ", "use ", "utilize "]
        .iter()
        .any(|p| low.starts_with(p));
    if !about && !(tab_at.is_some() && space_at.is_some()) && !(use_lead && tab_at.is_some()) {
        return None;
    }
    let tabs = match (tab_at, space_at) {
        (Some(t), Some(sp)) => t < sp,
        (Some(_), None) => true,
        _ => false,
    };
    if tabs {
        return Some(Kind::Indent { tabs: true, width: 0 });
    }
    let w = number_in(low).filter(|w| (1..=16).contains(w));
    if space_at.is_some() {
        return Some(Kind::Indent { tabs: false, width: w.unwrap_or(0) });
    }
    Some(Kind::Indent { tabs: false, width: w? })
}

/// Lines of at most N characters in another language, as `max_line_form`: a word for line or
/// column, a word for length or a limit, and a number from 20 to 1000.
fn other_max_line(low: &str) -> Option<Kind> {
    let line = ["linea", "ligne", "zeile", "linha", "columna", "colonne", "spalte", "coluna"].iter().any(|w| low.contains(w));
    let length = [
        "longitud", "longueur", "lange", "laenge", "comprimento", "tamano", "tamanho", "largo", "maxim", "hochstens", "hoechstens", "au plus", "a lo sumo", "como mucho",
        "no maximo", "moins de", "menos de", "caracter", "zeichen", "breite", "limite",
    ]
    .iter()
    .any(|w| low.contains(w));
    if !(line && length) {
        return None;
    }
    let c = number_in(low).filter(|c| (20..=1000).contains(c))?;
    Some(Kind::MaxLine { chars: c })
}

/// "Never use X" or "avoid X" in another language with a plain word or short phrase: a ban.
fn other_plain_ban(f: &Folded) -> Option<Vec<Kind>> {
    let leads = [
        "nunca uses ", "no uses ", "evita ", "evitar ", "n'utilise jamais ", "n'utilisez jamais ", "n'utilise pas ", "n'utilisez pas ", "evite ", "evitez ", "eviter ",
        "verwende nie ", "verwende niemals ", "verwende kein ", "verwende keine ", "benutze nie ", "benutze keine ", "vermeide ", "vermeiden ", "nunca use ", "nao use ",
        "nao usar ",
    ];
    let (_, rest) = f.after_lead(&leads)?;
    let p = other_term(&rest);
    (has_alnum(&p) && p.split_whitespace().count() <= 3 && p.chars().count() <= MAX_TERM).then(|| vec![Kind::Ban { phrase: p }])
}

/// The forms of Spanish, French, German and Portuguese for one statement; `None` when none
/// matches. Tried after the English forms (`forms`).
fn other_forms(body: &str) -> Option<Vec<Kind>> {
    let t = other_strip(body);
    if t.is_empty() {
        return None;
    }
    let f = Folded::new(&t);
    if let Some(k) = other_indent(&f.low) {
        return Some(vec![k]);
    }
    if let Some(k) = other_max_line(&f.low) {
        return Some(vec![k]);
    }
    if let Some(p) = other_ban(&f) {
        return Some(vec![Kind::Ban { phrase: p }]);
    }
    if let Some(k) = other_remove(&f) {
        return Some(k);
    }
    if let Some((p, v)) = other_swap(&f) {
        return swap_kinds(&p, &v);
    }
    other_plain_ban(&f)
}

/// Naming targets in the other languages (folded whole words), read only when no English target
/// word named one (`naming_form` gave `names`).
const TARGETS_OTHER: &[(&[&str], &str)] = &[
    (&["funcion", "funciones", "funcao", "funcoes", "fonction", "fonctions", "funktion", "funktionen", "funktionsnamen", "metodo", "metodos", "methoden"], "functions"),
    (&["variavel", "variaveis"], "variables"),
    (&["tipo", "tipos", "typ", "klasse", "klassen", "clase", "clases"], "types"),
    (&["konstante", "konstanten"], "constants"),
    (&["archivo", "archivos", "fichier", "fichiers", "datei", "dateien", "dateinamen", "arquivo", "arquivos"], "files"),
    (&["modulos", "paquete", "paquetes", "paket", "pakete", "pacote", "pacotes"], "modules"),
];

/// A naming rule with its target read from the other languages' words when the English words
/// named none.
fn other_target(k: Kind, t: &str) -> Kind {
    match k {
        Kind::Naming { case, target } if target == "names" => {
            let low: String = t.chars().map(fold_char).collect();
            let words: Vec<&str> = low.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
            let found = TARGETS_OTHER.iter().find(|(ws, _)| words.iter().any(|w| ws.contains(w))).map(|(_, t)| (*t).to_owned());
            Kind::Naming { case, target: found.unwrap_or(target) }
        }
        other => other,
    }
}

/// A statement split at the conjunctions of the other languages (y, et, und, e, ni) and at
/// commas; empty when it holds none of those conjunctions.
fn other_parts(body: &str) -> Vec<String> {
    if ![" y ", " et ", " und ", " e ", " ni "].iter().any(|c| Folded::new(body).low.contains(c)) {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut rest = body.to_owned();
    loop {
        match Folded::new(&rest).split(0, &[", y ", ", et ", ", und ", ", e ", " y ", " et ", " und ", " e ", " ni ", ", "]) {
            Some((a, b)) => {
                out.push(a.trim().to_owned());
                rest = b;
            }
            None => {
                out.push(rest.trim().to_owned());
                return out;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Want forms, in all five languages.
//
// Rules for these forms:
// - They are tried only for a statement no other form reads, and a message is read with them only
//   when it gave no rule without them (`parse`), so every statement that gave rules before they
//   existed gives the same rules.
// - Matching is on the folded statement (`Folded`), as for the other languages; a term is cut from
//   the user's own letters.
// - The object is a style the parser knows (`STYLES`), a quoted term that is the whole object, or
//   one plain word. Anything else (several plain words, a negation, a clause) stays verbatim
//   guidance, so nothing is guessed into a push.

/// Styles the parser knows: the canonical name, then its folded spellings in the five languages.
/// A want naming one is guidance for the agent ("prefer: concise").
const STYLES: &[(&str, &[&str])] = &[
    (
        "concise",
        &[
            "concise", "conciseness", "concision", "brevity", "brief", "conciso", "concisa", "concisos", "concisas", "concis", "concises", "knapp", "pragnant", "praegnant", "sucinto",
            "sucinta", "sucintos", "sucintas",
        ],
    ),
    ("terse", &["terse", "terseness", "escueto", "escueta", "escuetos", "escuetas", "laconique", "laconiques", "lakonisch"]),
    ("formal", &["formal", "formality", "formales", "formel", "formelle", "formels", "formelles", "formell", "formais"]),
    (
        "plain language",
        &[
            "plain", "plain language", "plain english", "plain words", "simple language", "simple words", "simple", "lenguaje sencillo", "lenguaje claro", "lenguaje simple",
            "langage simple", "langage clair", "einfache sprache", "klare sprache", "linguagem simples", "linguagem clara",
        ],
    ),
    ("numbered lists", &["numbered lists", "numbered list", "numbered steps", "listas numeradas", "lista numerada", "listes numerotees", "liste numerotee", "nummerierte listen", "nummerierte liste"]),
    (
        "bulleted lists",
        &[
            "bullet points", "bullet lists", "bullet list", "bulleted lists", "bullets", "vinetas", "listas con vinetas", "puces", "listes a puces", "aufzahlungszeichen", "aufzaehlungszeichen",
            "stichpunkte", "marcadores", "listas com marcadores",
        ],
    ),
    ("active voice", &["active voice", "voz activa", "voix active", "aktiv", "aktivform", "aktive form", "voz ativa"]),
    ("short sentences", &["short sentences", "frases cortas", "oraciones cortas", "phrases courtes", "kurze satze", "kurze saetze", "frases curtas"]),
];
/// Nouns a style may stand beside ("concise replies", "respuestas concisas").
const STYLE_NOUNS: &[&str] = &[
    "replies", "reply", "answers", "answer", "responses", "response", "writing", "text", "texts", "prose", "style", "tone", "language", "wording", "output", "comments", "docs",
    "documentation", "messages", "notes", "summaries", "explanations", "respuestas", "respuesta", "textos", "texto", "estilo", "tono", "redaccion", "comentarios", "mensajes", "reponses",
    "reponse", "textes", "texte", "ton", "redaction", "commentaires", "antworten", "antwort", "stil", "formulierungen", "kommentare", "respostas", "resposta", "tom", "redacao",
    "mensagens",
];
/// Courtesy words before a want's verb that change nothing.
const WANT_SKIP: &[&str] = &["please ", "por favor, ", "por favor ", "s'il te plait, ", "s'il te plait ", "s'il vous plait, ", "s'il vous plait ", "bitte "];
/// Courtesy words after a want's object that change nothing.
const WANT_END: &[&str] = &[", please", " please"];
/// The speaker before a verb that needs one ("I like", "j'aime").
const WANT_SPEAKER: &[&str] = &["i ", "we ", "i'd ", "i would ", "we'd ", "we would ", "je ", "j'", "nous ", "ich ", "wir ", "yo ", "eu "];
/// Words that make a want strong (`STRENGTH_STRONG`), before or after its verb.
const WANT_STRONG: &[&str] = &[
    "strongly ", "really ", "definitely ", "much ", "greatly ", "always ", "realmente ", "de verdad ", "siempre ", "mucho ", "vraiment ", "fortement ", "toujours ", "absolument ",
    "unbedingt ", "wirklich ", "stark ", "immer ", "sempre ", "muito ",
];
/// Words that make a want slight (`STRENGTH_SLIGHT`), before or after its verb.
const WANT_SLIGHT: &[&str] = &[
    "slightly ", "somewhat ", "mildly ", "a bit ", "a little ", "ligeramente ", "un poco ", "legerement ", "un peu ", "etwas ", "leicht ", "ein wenig ", "ein bisschen ",
    "ligeiramente ", "um pouco ",
];
/// Words after a want's verb that change nothing ("verwende lieber").
const WANT_NEUTRAL: &[&str] = &["rather ", "lieber ", "plutot ", "mejor ", "de preferencia ", "de preference "];
/// The verbs that lead a want: (folded lead, needs a speaker before it, takes only a style).
const WANT_VERBS: &[(&str, bool, bool)] = &[
    ("prefer ", false, false),
    ("favour ", false, false),
    ("favor ", false, false),
    ("use ", false, false),
    ("more ", false, false),
    ("be ", false, true),
    ("like ", true, false),
    ("want ", true, false),
    ("prefiero ", false, false),
    ("preferimos ", false, false),
    ("prefiere ", false, false),
    ("prefieres ", false, false),
    ("prefiera ", false, false),
    ("favorece ", false, false),
    ("usa ", false, false),
    ("utiliza ", false, false),
    ("emplea ", false, false),
    ("mas ", false, false),
    ("me gusta ", false, false),
    ("me gustan ", false, false),
    ("nos gusta ", false, false),
    ("nos gustan ", false, false),
    ("quiero ", false, false),
    ("queremos ", false, false),
    ("se ", false, true),
    ("sed ", false, true),
    ("prefere ", false, false),
    ("preferez ", false, false),
    ("preferons ", false, false),
    ("privilegie ", false, false),
    ("privilegiez ", false, false),
    ("favorise ", false, false),
    ("favorisez ", false, false),
    ("utilise ", false, false),
    ("utilisez ", false, false),
    ("emploie ", false, false),
    ("employez ", false, false),
    ("davantage de ", false, false),
    ("davantage d'", false, false),
    ("aime ", true, false),
    ("veux ", true, false),
    ("voudrais ", true, false),
    ("sois ", false, true),
    ("soyez ", false, true),
    ("bevorzuge ", false, false),
    ("bevorzugen ", false, false),
    ("bevorzugt ", false, false),
    ("verwende ", false, false),
    ("verwenden ", false, false),
    ("benutze ", false, false),
    ("nutze ", false, false),
    ("mehr ", false, false),
    ("mag ", true, false),
    ("mochte ", true, false),
    ("sei ", false, true),
    ("seid ", false, true),
    ("prefiro ", false, false),
    ("prefira ", false, false),
    ("utilize ", false, false),
    ("mais ", false, false),
    ("gosto de ", false, false),
    ("gostamos de ", false, false),
    ("quero ", false, false),
    ("seja ", false, true),
    ("sejam ", false, true),
];
/// Words a want's object may not start with: a negation reads as the opposite of a want.
const WANT_NEGATION: &[&str] = &["no ", "not ", "never ", "pas ", "ne ", "nicht ", "kein", "nao ", "nunca ", "jamais ", "nie ", "ningun", "aucun", "nenhum"];
/// One-word objects that are not terms: the words of the indentation form, and words that only
/// point.
const WANT_NOT_TERMS: &[&str] = &[
    "tabs", "tab", "spaces", "space", "it", "this", "that", "them", "these", "those", "you", "lo", "la", "le", "les", "es", "das", "isso", "eso", "esto", "ceci", "cela", "ca", "dies",
];
/// Leads of a term's own name ("the word", "le mot") before the term.
const WANT_TERM_LEADS: &[&str] = &[
    "the word ", "the term ", "the phrase ", "la palabra ", "la expresion ", "el termino ", "le mot ", "l'expression ", "le terme ", "das wort ", "den ausdruck ", "den begriff ",
    "a palavra ", "a expressao ", "o termo ",
];

/// The strength word `f` starts with: the strength it gives and its length in characters.
fn strength_lead(f: &Folded) -> Option<(u64, usize)> {
    f.after_lead(WANT_STRONG).map(|(l, _)| (STRENGTH_STRONG, l.chars().count())).or_else(|| f.after_lead(WANT_SLIGHT).map(|(l, _)| (STRENGTH_SLIGHT, l.chars().count())))
}

/// The style a want's object names (`STYLES`), with its leading articles and "more" taken off, on
/// its own or beside one or two nouns of `STYLE_NOUNS`.
fn style_of(x: &str) -> Option<&'static str> {
    let mut b = bare(x);
    loop {
        let before = b.len();
        for lead in ["the ", "an ", "more ", "mas ", "mais ", "mehr ", "un ", "une ", "um ", "uma ", "ein ", "eine ", "einen "] {
            if let Some(r) = b.strip_prefix(lead) {
                b = bare(r);
            }
        }
        if b.len() == before {
            break;
        }
    }
    let words: Vec<&str> = b.split_whitespace().collect();
    let nouns = |ws: &[&str]| ws.len() <= 2 && ws.iter().all(|w| STYLE_NOUNS.contains(w));
    for (name, spellings) in STYLES {
        for sp in *spellings {
            let n = sp.split(' ').count();
            let (Some(head), Some(tail)) = (words.get(..n), words.get(n..)) else { continue };
            if head.join(" ") == *sp && nouns(tail) {
                return Some(name);
            }
            let cut = words.len().saturating_sub(n);
            if cut > 0
                && let (Some(lead), Some(end)) = (words.get(..cut), words.get(cut..))
                && end.join(" ") == *sp
                && nouns(lead)
            {
                return Some(name);
            }
        }
    }
    None
}

/// The term a want's object names: a quoted term that is the whole object, or one plain word (any
/// letter, digit or `- _ . ' + # /`, not one of `WANT_NOT_TERMS`).
fn want_term(x: &str) -> Option<String> {
    let f = Folded::new(x.trim());
    let x = f.after_lead(WANT_TERM_LEADS).map_or_else(|| x.trim().to_owned(), |(_, rest)| rest);
    let x = x.trim_end_matches(['.', '!', ',', ';', '?']).trim();
    if let Some(q) = quoted(x) {
        let whole = x.chars().count() == q.chars().count() + 2;
        let q = q.trim().to_owned();
        return (whole && !q.is_empty() && q.chars().count() <= MAX_TERM && has_alnum(&q)).then_some(q);
    }
    let plain = !x.is_empty() && x.chars().count() <= MAX_TERM && x.chars().all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | '\'' | '+' | '#' | '/'));
    let folded: String = x.chars().map(fold_char).collect();
    (plain && x.chars().any(char::is_alphabetic) && !WANT_NOT_TERMS.contains(&folded.as_str())).then(|| x.to_owned())
}

/// A want in any of the five languages: "prefer X", "always use X", "use X", "I like X",
/// "favour X", "more X", "be <style>" and their equivalents, with "strongly" or "slightly" (and
/// their kin) setting the strength; `None` when the statement is not one.
fn want_form(body: &str) -> Option<Kind> {
    let mut t = body.trim().trim_start_matches(['\u{a1}', '\u{bf}']).trim_end_matches(['.', '!', '?']).trim().to_owned();
    let mut strength = None;
    let mut speaker = false;
    loop {
        let f = Folded::new(&t);
        if let Some(p) = POLITE_END.iter().chain(WANT_END).find(|p| f.low.ends_with(**p)) {
            t = f.text(0, f.orig.len().saturating_sub(p.chars().count())).trim().to_owned();
        } else if let Some((_, rest)) = f.after_lead(WANT_SKIP) {
            t = rest;
        } else if let Some((st, len)) = strength_lead(&f) {
            strength = Some(st);
            t = f.text(len, f.orig.len()).trim().to_owned();
        } else if let Some((_, rest)) = f.after_lead(WANT_SPEAKER).filter(|_| !speaker) {
            speaker = true;
            t = rest;
        } else {
            break;
        }
    }
    let f = Folded::new(&t);
    let &(verb, needs_speaker, style_only) = WANT_VERBS.iter().filter(|(v, _, _)| f.low.starts_with(v)).max_by_key(|(v, _, _)| v.len())?;
    if needs_speaker && !speaker {
        return None;
    }
    let mut x = f.text(verb.chars().count(), f.orig.len()).trim().to_owned();
    loop {
        let g = Folded::new(&x);
        if let Some((st, len)) = strength_lead(&g) {
            strength = Some(st);
            x = g.text(len, g.orig.len()).trim().to_owned();
        } else if let Some((_, rest)) = g.after_lead(WANT_NEUTRAL) {
            x = rest;
        } else {
            break;
        }
    }
    let low = Folded::new(&x).low;
    if x.is_empty() || WANT_NEGATION.iter().any(|n| low.starts_with(n)) {
        return None;
    }
    let strength = strength.unwrap_or(STRENGTH_DEFAULT);
    if let Some(style) = style_of(&x) {
        return Some(Kind::Want { term: String::new(), style: style.to_owned(), strength });
    }
    if style_only {
        return None;
    }
    want_term(&x).map(|term| Kind::Want { term, style: String::new(), strength })
}

/// The rules one statement gives: its English forms, else the forms of the other languages, else
/// (with `want`) the want forms; `None` when no form matches.
fn forms_with(body: &str, want: bool) -> Option<Vec<Kind>> {
    let found = english_forms(body).or_else(|| other_forms(body));
    if found.is_some() || !want {
        return found;
    }
    want_form(body).map(|k| vec![k])
}

/// How a statement was read: part by part (each part with its words) or whole.
enum Read {
    Parts(Vec<(Vec<Kind>, String)>),
    Whole(Vec<Kind>),
}

/// One statement read with the forms (`forms_with`): parts joined by "and" or commas when every
/// part matches, then parts joined by the other languages' conjunctions, then the whole
/// statement; `None` when nothing reads it.
fn read(body: &str, want: bool) -> Option<Read> {
    let low = body.to_lowercase();
    let parts: Vec<&str> = if low.contains(" and ") || body.contains(',') {
        let mut v = Vec::new();
        let mut rest = body;
        loop {
            let lr = rest.to_lowercase();
            match split_at(rest, &lr, &[", and ", " and ", ", "]) {
                Some((a, b)) => {
                    v.push(a);
                    rest = b;
                }
                None => {
                    v.push(rest);
                    break;
                }
            }
        }
        v
    } else {
        Vec::new()
    };
    let each: Option<Vec<(Vec<Kind>, String)>> = (parts.len() > 1).then(|| parts.iter().map(|p| forms_with(p, want).map(|k| (k, (*p).to_owned()))).collect()).flatten();
    // Parts joined by the conjunctions of the other languages, taken only when every part
    // matches a form and at least one part matches no English form, so an English statement
    // keeps its rules.
    let other: Vec<String> = if each.is_none() { other_parts(body) } else { Vec::new() };
    let foreign = other.iter().any(|p| english_forms(p).is_none());
    let each = each.or_else(|| (other.len() > 1 && foreign).then(|| other.iter().map(|p| forms_with(p, want).map(|k| (k, p.clone()))).collect()).flatten());
    // Parts first: a whole statement can match one form by a word of another part
    // ("no em dashes and max line length 100" holds a line length).
    match each {
        Some(all) => Some(Read::Parts(all)),
        None => forms_with(body, want).map(Read::Whole),
    }
}

/// Parses a message of the user's words into rules (ids left empty; `with_ids` numbers them).
/// Each statement is parsed on its own; one that matches no form is tried split at "and" and
/// commas (each part must match), else kept verbatim as guidance. A statement the forms before
/// wants do not read is read again with the want forms (`want_form`), so a statement that gave
/// rules without them gives the same rules. `base` is the cartridge's scope; a statement's own
/// scope clause narrows it. `project` stands for "this project".
#[must_use]
pub fn parse(message: &str, base: &Scope, project: Option<&str>) -> Vec<Rule> {
    let mut out = Vec::new();
    for stmt in statements(message) {
        let (body, own, domain) = split_scope(&stmt, project);
        let scope = own.over(base);
        let rule = |kind: Kind, said: &str| Rule { id: String::new(), kind, said: said.to_owned(), scope: scope.clone() };
        match read(&body, false).or_else(|| read(&body, true)) {
            Some(Read::Parts(all)) => {
                for (kinds, part) in all {
                    out.extend(kinds.into_iter().map(|k| rule(k, part.trim())));
                }
            }
            Some(Read::Whole(kinds)) => out.extend(kinds.into_iter().map(|k| rule(k, &stmt))),
            None => out.push(rule(Kind::Guidance { domain }, &stmt)),
        }
    }
    out
}

/// Numbers new rules after the existing ones (`r<next>`).
#[must_use]
pub fn with_ids(existing: &[Rule], new: Vec<Rule>) -> Vec<Rule> {
    let mut next = existing.iter().filter_map(|r| r.id.strip_prefix('r').and_then(|x| x.parse::<u64>().ok())).max().unwrap_or(0);
    new.into_iter()
        .map(|mut r| {
            next += 1;
            r.id = format!("r{next}");
            r
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The rules in force, and applying them.

/// A rule with the cartridge that holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Held {
    /// The cartridge's key (`id@version`), or its id where only the values are known.
    pub cartridge: String,
    pub name: String,
    pub rule: Rule,
}

/// The stated rules of every plugged preference cartridge.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InForce {
    pub held: Vec<Held>,
}

impl InForce {
    /// The rules among served settings values (`pref.stated.<id>.r<NNN>`); each rule is named by its
    /// cartridge's id.
    #[must_use]
    pub fn from_settings(m: &BTreeMap<String, String>) -> Self {
        let held = rules_of(m).into_iter().flat_map(|(id, rules)| rules.into_iter().map(move |rule| Held { cartridge: id.clone(), name: id.clone(), rule })).collect();
        Self { held }
    }
    /// The rules of the preference cartridges the library serves now: every preference use switched
    /// to a stack, cartridges and stacks on trial left out (a trial is recorded, never applied).
    /// A cartridge whose file fails its hash check is left out, as at load.
    #[must_use]
    pub fn from_library(lib: &Library) -> Self {
        let mut held = Vec::new();
        for (use_for, key) in &lib.active {
            if !preference_use(use_for) || key == "off" {
                continue;
            }
            let Some(st) = lib.stack(key).filter(|st| !st.trial) else { continue };
            for k in &st.cartridges {
                let Some(c) = lib.cartridge(k).filter(|c| !c.trial) else { continue };
                let Ok(Filter::Settings(m)) = lib.filter(c) else { continue };
                for rule in rules_of(&m).remove(&c.id).unwrap_or_default() {
                    held.push(Held { cartridge: c.key(), name: c.name.clone(), rule });
                }
            }
        }
        Self { held }
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }
    /// The rules that apply in `project` to `path`.
    #[must_use]
    pub fn in_scope(&self, project: Option<&str>, path: Option<&str>) -> Vec<&Held> {
        self.held.iter().filter(|h| h.rule.scope.matches(project, path)).collect()
    }
    /// The strings the Writer's decoding may mask at the token level: (characters and text without
    /// letters a global transform replaces, whole words a global rule bans or replaces).
    #[must_use]
    pub fn banned(&self) -> (Vec<String>, Vec<String>) {
        let mut marks = Vec::new();
        let mut words = Vec::new();
        for h in self.held.iter().filter(|h| h.rule.scope.kinds.is_empty() && h.rule.scope.folders.is_empty()) {
            match &h.rule.kind {
                Kind::Replace { from, .. } if !has_alnum(from) => marks.push(from.clone()),
                Kind::Vocabulary { avoid, .. } | Kind::Ban { phrase: avoid } if !avoid.contains(' ') => words.push(avoid.to_lowercase()),
                _ => {}
            }
        }
        (marks, words)
    }
    /// The terms the Writer's decoding pushes toward (`Kind::Want` with a term), each with its
    /// strength in hundredths, from rules whose scope names no file kind or folder (the Writer's
    /// own text has no path), in order.
    #[must_use]
    pub fn wanted(&self) -> Vec<(String, u64)> {
        self.held
            .iter()
            .filter(|h| h.rule.scope.kinds.is_empty() && h.rule.scope.folders.is_empty())
            .filter_map(|h| match &h.rule.kind {
                Kind::Want { term, strength, .. } if !term.is_empty() => Some((term.clone(), *strength)),
                _ => None,
            })
            .collect()
    }
}

/// One rule found broken in a draft.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub cartridge: String,
    pub rule: String,
    /// 1-based line of the draft (0: the file name).
    pub line: usize,
    pub what: String,
}

/// A wanted term in scope that a draft does not hold: information, never a violation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Missed {
    pub cartridge: String,
    pub rule: String,
    pub term: String,
    /// In hundredths.
    pub strength: u64,
}

/// What applying the rules to a draft did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// The draft with the transforms applied (prose only; code is never rewritten).
    pub text: String,
    /// `prose` or `code`.
    pub mode: &'static str,
    /// (cartridge, rule, places changed).
    pub changes: Vec<(String, String, usize)>,
    pub violations: Vec<Violation>,
    /// Violations beyond `MAX_VIOLATIONS` per rule, by `cartridge rule`.
    pub more: BTreeMap<String, usize>,
    /// The wanted terms in scope the draft does not hold (`Kind::Want`), in rule order.
    pub missed: Vec<Missed>,
    /// The rules that applied (in scope), in order.
    pub rules: Vec<Held>,
}

impl Outcome {
    #[must_use]
    pub fn changed(&self) -> bool {
        self.changes.iter().any(|c| c.2 > 0)
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let row = |h: &Held| {
            let mut v = h.rule.view();
            if let Value::Object(m) = &mut v {
                m.insert("cartridge".into(), s(&h.cartridge));
            }
            v
        };
        obj(vec![
            ("text", s(&self.text)),
            ("changed", Value::Bool(self.changed())),
            ("mode", s(self.mode)),
            ("changes", Value::Array(self.changes.iter().filter(|c| c.2 > 0).map(|(c, r, k)| obj(vec![("cartridge", s(c)), ("rule", s(r)), ("count", n(*k))])).collect())),
            ("violations", Value::Array(self.violations.iter().map(|v| obj(vec![("cartridge", s(&v.cartridge)), ("rule", s(&v.rule)), ("line", n(v.line)), ("what", s(&v.what))])).collect())),
            ("violations_more", Value::Object(self.more.iter().map(|(k, v)| (k.clone(), n(*v))).collect())),
            (
                "wants_missed",
                Value::Array(
                    self.missed
                        .iter()
                        .map(|m| {
                            obj(vec![
                                ("cartridge", s(&m.cartridge)),
                                ("rule", s(&m.rule)),
                                ("term", s(&m.term)),
                                ("strength", Value::Number(strength_text(m.strength))),
                                ("what", s(&format!("no \"{}\" in the draft (wanted, strength {}); information, not a violation", m.term, strength_text(m.strength)))),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("guidance", Value::Array(self.rules.iter().filter(|h| h.rule.enforcement() == Enforcement::Guidance).map(row).collect())),
            ("rules_in_force", Value::Array(self.rules.iter().map(row).collect())),
        ])
    }
}

/// Whether text at `path` is prose (Markdown, plain text, or no path): transforms apply outside
/// code. Any other kind is code: nothing in it is rewritten, every rule is checked.
#[must_use]
pub fn prose_path(path: Option<&str>) -> bool {
    path.is_none_or(|p| ext_of(&p.replace('\\', "/")).is_none_or(|e| matches!(e.as_str(), "md" | "markdown" | "txt" | "text" | "rst" | "adoc")))
}

/// A line quoted from the user's material: an evidence line (`path:line ...`), a span or packet
/// header of a context pack (`@@ `, `~~ `) or a quoted line (`>`). Never altered.
#[must_use]
pub fn evidence_line(line: &str) -> bool {
    let t = line.trim_start();
    if t.starts_with("@@ ") || t.starts_with("~~ ") || t.starts_with('>') {
        return true;
    }
    let first = t.split_whitespace().next().unwrap_or("");
    first.split_once(':').is_some_and(|(p, rest)| !p.is_empty() && (p.contains('.') || p.contains('/')) && rest.starts_with(|c: char| c.is_ascii_digit()))
}

fn word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The places `needle` stands in `hay` as a whole word or phrase (letter case ignored in ASCII),
/// as byte ranges.
fn word_matches(hay: &str, needle: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if needle.is_empty() {
        return out;
    }
    let len = needle.len();
    let need_left = needle.chars().next().is_some_and(word_char);
    let need_right = needle.chars().last().is_some_and(word_char);
    let mut i = 0;
    while i + len <= hay.len() {
        if hay.is_char_boundary(i)
            && hay.is_char_boundary(i + len)
            && hay.get(i..i + len).is_some_and(|w| w.eq_ignore_ascii_case(needle))
            && (!need_left || !hay.get(..i).and_then(|b| b.chars().last()).is_some_and(word_char))
            && (!need_right || !hay.get(i + len..).and_then(|a| a.chars().next()).is_some_and(word_char))
        {
            out.push((i, i + len));
            i += len;
        } else {
            i += hay.get(i..).and_then(|x| x.chars().next()).map_or(1, char::len_utf8);
        }
    }
    out
}

/// `prefer` in the letter case of the word it replaces.
fn cased(found: &str, avoid: &str, prefer: &str) -> String {
    let letters: Vec<char> = found.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.len() > 1 && letters.iter().all(|c| c.is_uppercase()) && !avoid.chars().filter(|c| c.is_alphabetic()).all(char::is_uppercase) {
        return prefer.to_uppercase();
    }
    let first_up = found.chars().next().is_some_and(char::is_uppercase);
    let avoid_up = avoid.chars().next().is_some_and(char::is_uppercase);
    if first_up && !avoid_up {
        let mut c = prefer.chars();
        return c.next().map_or_else(String::new, |f| f.to_uppercase().collect::<String>() + c.as_str());
    }
    prefer.to_owned()
}

const LEAD_PUNCT: [char; 6] = [',', '.', ';', ':', '!', '?'];

/// Puts `piece` in place of a match in `out`, keeping the spacing readable: punctuation sits on the
/// word before it, nothing is left dangling at a line's start or end, and a removal leaves one
/// space. Returns how many bytes of `after` (whitespace) to skip.
fn splice(out: &mut String, piece: &str, after: &str, line_start: bool, line_end: bool) -> usize {
    let at_start = line_start && out.trim().is_empty();
    let at_end = line_end && after.trim().is_empty();
    let mut piece = piece.to_owned();
    if piece.starts_with(LEAD_PUNCT) {
        let kept = out.trim_end().len();
        out.truncate(kept);
        if at_start {
            piece = piece.trim_start_matches(LEAD_PUNCT).trim_start().to_owned();
        }
    }
    if at_end {
        piece = piece.trim_end().to_owned();
        // A comma or semicolon left at the end of a line reads as a fault; nothing is put there.
        if !piece.is_empty() && piece.chars().all(|c| matches!(c, ',' | ';')) {
            piece.clear();
        }
    }
    let skip_ws = piece.ends_with(char::is_whitespace) || piece.is_empty() && (out.ends_with(char::is_whitespace) || out.is_empty() && at_start);
    let skip = if skip_ws { after.len() - after.trim_start().len() } else { 0 };
    if piece.is_empty() && after.trim_start().starts_with(LEAD_PUNCT) {
        let kept = out.trim_end().len();
        out.truncate(kept);
    }
    out.push_str(&piece);
    skip
}

/// One rule over a piece of prose: the text with its transform applied, and the places changed.
fn transform(text: &str, kind: &Kind, line_start: bool, line_end: bool) -> (String, usize) {
    let found: Vec<(usize, usize)> = match kind {
        Kind::Replace { from, .. } => text.match_indices(from.as_str()).map(|(i, m)| (i, i + m.len())).collect(),
        Kind::Vocabulary { avoid, .. } => word_matches(text, avoid),
        _ => return (text.to_owned(), 0),
    };
    if found.is_empty() {
        return (text.to_owned(), 0);
    }
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    let mut count = 0;
    for (a, b) in &found {
        if *a < at {
            continue;
        }
        out.push_str(text.get(at..*a).unwrap_or(""));
        let piece = match kind {
            Kind::Replace { to, .. } => to.clone(),
            Kind::Vocabulary { avoid, prefer } => cased(text.get(*a..*b).unwrap_or(""), avoid, prefer),
            _ => String::new(),
        };
        let skip = splice(&mut out, &piece, text.get(*b..).unwrap_or(""), line_start, line_end);
        at = *b + skip;
        count += 1;
    }
    out.push_str(text.get(at..).unwrap_or(""));
    (out, count)
}

/// The prose and code spans of a line, in order: inline code between backticks is code (`true`);
/// an unclosed backtick leaves the rest prose.
fn spans(line: &str) -> Vec<(bool, &str)> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(open) = rest.find('`') {
        let Some(close) = rest.get(open + 1..).and_then(|x| x.find('`')) else { break };
        let end = open + 1 + close + 1;
        if open > 0 {
            out.push((false, rest.get(..open).unwrap_or("")));
        }
        out.push((true, rest.get(open..end).unwrap_or("")));
        rest = rest.get(end..).unwrap_or("");
    }
    if !rest.is_empty() {
        out.push((false, rest));
    }
    out
}

/// The names a code line defines, by target (functions, types, constants, variables, modules).
fn definitions(line: &str, kind: Option<&str>) -> Vec<(&'static str, String)> {
    let js = kind.is_some_and(|k| matches!(k, "js" | "jsx" | "ts" | "tsx" | "mjs"));
    let leads: [(&str, &'static str); 15] = [
        ("fn ", "functions"),
        ("def ", "functions"),
        ("function ", "functions"),
        ("func ", "functions"),
        ("struct ", "types"),
        ("enum ", "types"),
        ("trait ", "types"),
        ("class ", "types"),
        ("interface ", "types"),
        ("type ", "types"),
        ("const ", if js { "variables" } else { "constants" }),
        ("static ", "constants"),
        ("let ", "variables"),
        ("var ", "variables"),
        ("mod ", "modules"),
    ];
    let mut out = Vec::new();
    for (lead, target) in leads {
        for (i, _) in line.match_indices(lead) {
            if line.get(..i).and_then(|b| b.chars().last()).is_some_and(word_char) {
                continue;
            }
            let rest = line.get(i + lead.len()..).unwrap_or("").trim_start();
            let rest = rest.strip_prefix("mut ").unwrap_or(rest);
            let name: String = rest.chars().take_while(|c| word_char(*c)).collect();
            if !name.is_empty() && !name.starts_with(|c: char| c.is_ascii_digit()) && name != "self" {
                out.push((target, name));
            }
        }
    }
    out
}

/// What a code line breaks of a check rule (`Indent`, `MaxLine`, `Naming`).
fn check_code(line: &str, kind: &Kind, ext: Option<&str>) -> Option<String> {
    match kind {
        Kind::Indent { tabs, width } => {
            let lead: String = line.chars().take_while(|c| *c == ' ' || *c == '\t').collect();
            if line.trim().is_empty() || lead.is_empty() {
                return None;
            }
            if *tabs {
                lead.contains(' ').then(|| "indented with spaces".to_owned())
            } else if lead.contains('\t') {
                Some("indented with a tab".to_owned())
            } else {
                let w = lead.chars().count() as u64;
                (*width > 0 && !w.is_multiple_of(*width)).then(|| format!("indented {w} spaces, not a step of {width}"))
            }
        }
        Kind::MaxLine { chars } => {
            let c = line.chars().count() as u64;
            (c > *chars).then(|| format!("{c} characters, over {chars}"))
        }
        Kind::Naming { case, target } => {
            let bad: Vec<String> = definitions(line, ext).into_iter().filter(|(t, _)| target == "names" || t == target).filter(|(_, nm)| !case.fits(nm)).map(|(t, nm)| format!("{nm} ({t})")).collect();
            (!bad.is_empty()).then(|| format!("not {}: {}", case.name(), bad.join(", ")))
        }
        _ => None,
    }
}

/// Text-level rules as checks in code: where a replace, vocabulary or ban rule's text appears.
fn check_text(line: &str, kind: &Kind) -> Option<String> {
    match kind {
        Kind::Replace { from, what, .. } => line.contains(from.as_str()).then(|| format!("holds {}", if what.is_empty() { shown(from) } else { what.clone() })),
        Kind::Vocabulary { avoid, .. } | Kind::Ban { phrase: avoid } => (!word_matches(line, avoid).is_empty()).then(|| format!("holds \"{avoid}\"")),
        _ => None,
    }
}

/// Counts violations per rule and keeps the first `MAX_VIOLATIONS` of each.
struct Reporter {
    found: Vec<usize>,
}

impl Reporter {
    fn add(&mut self, out: &mut Outcome, i: usize, h: &Held, line: usize, what: String) {
        let Some(f) = self.found.get_mut(i) else { return };
        *f += 1;
        if *f > MAX_VIOLATIONS {
            *out.more.entry(format!("{} {}", h.cartridge, h.rule.id)).or_default() += 1;
            return;
        }
        out.violations.push(Violation { cartridge: h.cartridge.clone(), rule: h.rule.id.clone(), line, what });
    }
}

/// One prose line with the transforms applied outside its inline code, and its bans reported.
fn prose_line(line: &str, no: usize, held: &[Held], counts: &mut [usize], rep: &mut Reporter, out: &mut Outcome) -> String {
    let parts = spans(line);
    let last = parts.len().saturating_sub(1);
    let mut new_line = String::with_capacity(line.len());
    for (k, (code, part)) in parts.iter().enumerate() {
        if *code {
            new_line.push_str(part);
            continue;
        }
        let mut cur = (*part).to_owned();
        for (i, h) in held.iter().enumerate() {
            if h.rule.enforcement() == Enforcement::Transform {
                let (next, c) = transform(&cur, &h.rule.kind, k == 0, k == last);
                cur = next;
                if let Some(x) = counts.get_mut(i) {
                    *x += c;
                }
            }
        }
        for (i, h) in held.iter().enumerate() {
            if let Kind::Ban { phrase } = &h.rule.kind {
                for _ in word_matches(&cur, phrase) {
                    rep.add(out, i, h, no, format!("holds \"{phrase}\""));
                }
            }
        }
        new_line.push_str(&cur);
    }
    new_line
}

/// Applies the rules in force to a draft for `path` in `project`.
///
/// Prose (`prose_path`): enforced transforms are applied outside fenced code blocks, inline code
/// spans and evidence or quoted lines (`evidence_line`), which stay byte for byte; bans are
/// reported where they appear in prose; indentation, line length and naming are checked inside the
/// fenced code blocks. Code: nothing is rewritten; every rule is checked on every line, and a
/// naming rule for files is checked on the file name. Guidance rules in scope are listed for the
/// agent to follow. A wanted term in scope that the draft does not hold anywhere (as a whole word
/// or phrase, any letter case) is noted in `missed`, never as a violation.
#[must_use]
pub fn apply(text: &str, rules: &InForce, project: Option<&str>, path: Option<&str>) -> Outcome {
    let held: Vec<Held> = rules.in_scope(project, path).into_iter().cloned().collect();
    let prose = prose_path(path);
    let ext = path.and_then(|p| ext_of(&p.replace('\\', "/")));
    let mut counts: Vec<usize> = vec![0; held.len()];
    let mut out = Outcome { mode: if prose { "prose" } else { "code" }, ..Outcome::default() };
    let mut rep = Reporter { found: vec![0; held.len()] };
    let mut fence = false;
    let mut lines_out: Vec<String> = Vec::new();
    for (no, line) in text.split('\n').enumerate() {
        let no = no + 1;
        let trimmed = line.trim_start();
        let fence_mark = trimmed.starts_with("```") || trimmed.starts_with("~~~");
        if !prose || fence && !fence_mark {
            for (i, h) in held.iter().enumerate() {
                let what = check_code(line, &h.rule.kind, ext.as_deref()).or_else(|| if prose { None } else { check_text(line, &h.rule.kind) });
                if let Some(w) = what {
                    rep.add(&mut out, i, h, no, w);
                }
            }
            lines_out.push(line.to_owned());
        } else if fence_mark || evidence_line(line) {
            lines_out.push(line.to_owned());
        } else {
            lines_out.push(prose_line(line, no, &held, &mut counts, &mut rep, &mut out));
        }
        if fence_mark && prose {
            fence = !fence;
        }
    }
    if !prose && let Some(p) = path {
        let stem = p.replace('\\', "/").rsplit('/').next().unwrap_or("").split('.').next().unwrap_or("").to_owned();
        for (i, h) in held.iter().enumerate() {
            if let Kind::Naming { case, target } = &h.rule.kind
                && target == "files"
                && !case.fits(&stem)
            {
                rep.add(&mut out, i, h, 0, format!("file name {stem} is not {}", case.name()));
            }
        }
    }
    for h in &held {
        if let Kind::Want { term, strength, .. } = &h.rule.kind
            && !term.is_empty()
            && word_matches(text, term).is_empty()
        {
            out.missed.push(Missed { cartridge: h.cartridge.clone(), rule: h.rule.id.clone(), term: term.clone(), strength: *strength });
        }
    }
    out.text = lines_out.join("\n");
    out.changes = held.iter().zip(&counts).map(|(h, c)| (h.cartridge.clone(), h.rule.id.clone(), *c)).collect();
    out.rules = held;
    out
}

/// Applies the enforced transforms to text the Nano writes itself (Writer notes, summaries,
/// outlines), for the piece at `path` when known; protected spans stay as they are. The Nano's own
/// words are prose even when they describe a code file. Text with no rule in force comes back
/// unchanged.
#[must_use]
pub fn own_text(text: &str, rules: &InForce, path: Option<&str>) -> String {
    if rules.is_empty() || text.is_empty() {
        return text.to_owned();
    }
    let keep: Vec<Held> = rules.held.iter().filter(|h| h.rule.enforcement() == Enforcement::Transform && h.rule.scope.matches(None, path)).cloned().map(|mut h| {
        // Scope was matched here against the piece's path; the prose pass below has no path.
        h.rule.scope = Scope::global();
        h
    }).collect();
    if keep.is_empty() {
        return text.to_owned();
    }
    apply(text, &InForce { held: keep }, None, None).text
}

/// The `nm_context_pack` block: the rules in force whose scope fits the project and at least one
/// cited path (or no path at all), one line each, at most `BLOCK_RULES`; a want reads
/// `prefer: <term or style>` with its strength. Empty when none applies.
#[must_use]
pub fn block(rules: &InForce, project: Option<&str>, paths: &[String]) -> Vec<String> {
    let fit: Vec<&Held> = rules
        .held
        .iter()
        .filter(|h| {
            let sc = &h.rule.scope;
            if sc.kinds.is_empty() && sc.folders.is_empty() { sc.matches(project, None) } else { paths.iter().any(|p| sc.matches(project, Some(p))) }
        })
        .collect();
    if fit.is_empty() {
        return Vec::new();
    }
    let mut out = vec!["preferences in force (the user's stated rules; follow them in what you write, and check a draft with nm_learning op preference_apply):".to_owned()];
    for h in fit.iter().take(BLOCK_RULES) {
        let words = h.rule.said.replace('\n', " ");
        let scope = if h.rule.scope.is_global() { String::new() } else { format!("; {}", h.rule.scope.words()) };
        match &h.rule.kind {
            Kind::Want { term, style, strength } => {
                let what = if term.is_empty() { style.clone() } else { format!("\"{term}\"") };
                out.push(format!("- prefer: {what} [want, strength {}{scope}; said: {words}] ({} {})", strength_text(*strength), h.cartridge, h.rule.id));
            }
            _ => out.push(format!("- {words} [{}: {}{scope}] ({} {})", h.rule.enforcement().name(), h.rule.does(), h.cartridge, h.rule.id)),
        }
    }
    if fit.len() > BLOCK_RULES {
        out.push(format!("- and {} more: nm_learning op preference_list", fit.len() - BLOCK_RULES));
    }
    out
}

// ---------------------------------------------------------------------------------------------
// The cartridge form.

/// The settings values of a cartridge's stated rules.
#[must_use]
pub fn settings_of(id: &str, rules: &[Rule]) -> BTreeMap<String, String> {
    rules.iter().enumerate().map(|(i, r)| (format!("{KEY_PREFIX}{id}.r{:03}", i + 1), r.to_value().encode().unwrap_or_default())).collect()
}

/// The stated rules among settings values, by cartridge id, in key order. A value that does not
/// read as a rule is kept as guidance holding the value's text.
#[must_use]
pub fn rules_of(m: &BTreeMap<String, String>) -> BTreeMap<String, Vec<Rule>> {
    let mut out: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
    for (k, v) in m {
        let Some((id, _)) = k.strip_prefix(KEY_PREFIX).and_then(|r| r.rsplit_once('.')) else { continue };
        let rule = nmjson::parse(v.as_bytes()).map_or_else(|_| Rule { id: String::new(), kind: Kind::Guidance { domain: String::new() }, said: v.clone(), scope: Scope::default() }, |x| Rule::from_value(&x));
        out.entry(id.to_owned()).or_default().push(rule);
    }
    out
}

/// Whether settings values hold learned preference values (`pref.global`, `pref.key.*`, the
/// `pref_fit` form) beside any stated rules.
#[must_use]
pub fn learned_values(m: &BTreeMap<String, String>) -> bool {
    m.keys().any(|k| k == "pref.global" || k.starts_with("pref.key."))
}

/// A valid cartridge id made from a name: lower case letters and digits, other runs as one `-`,
/// prefixed `stated-`.
#[must_use]
pub fn id_from_name(name: &str) -> String {
    let mut slug = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
        } else if !slug.ends_with('-') && !slug.is_empty() {
            slug.push('-');
        }
    }
    let slug: String = slug.trim_matches('-').chars().take(48).collect();
    let slug = slug.trim_end_matches('-');
    format!("stated-{}", if slug.is_empty() { "preferences" } else { slug })
}

// ---------------------------------------------------------------------------------------------
// The library side: writing versions, plugging, removing, and the undo of each.

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && !id.starts_with('.') && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// The next version of cartridge `id`: past every version listed and every file on the drive (a
/// cartridge taken out of the library keeps its files, and files are never written over).
#[must_use]
pub fn next_version(lib: &Library, id: &str) -> u64 {
    let mut v = lib.cartridges.iter().filter(|c| c.id == id).map(|c| c.version).max().unwrap_or(0) + 1;
    while lib.dir.join("cartridges").join(crate::library::shelf_path(PREFERENCE, "settings", DOMAIN, "settings", id, v)).exists() {
        v += 1;
    }
    v
}

/// The stated rules and the other values (learned ones) of a cartridge.
/// # Errors
/// The cartridge's file is missing or fails its hash check, or it is not a settings cartridge.
pub fn contents(lib: &Library, key: &str) -> Result<(Vec<Rule>, BTreeMap<String, String>), String> {
    let c = lib.cartridge(key).ok_or_else(|| format!("no cartridge {key}"))?;
    let Filter::Settings(m) = lib.filter(c).map_err(|e| e.to_string())? else { return Err(format!("{}: not a settings cartridge", c.key())) };
    let rules = rules_of(&m).remove(&c.id).unwrap_or_default();
    let other = m.into_iter().filter(|(k, _)| !k.starts_with(KEY_PREFIX)).collect();
    Ok((rules, other))
}

/// What to write: a cartridge's id, name and scope, its whole rule set, and who asked.
#[derive(Clone, Debug)]
pub struct Write {
    pub id: String,
    pub name: String,
    pub scope: Scope,
    pub rules: Vec<Rule>,
    pub made_from: String,
    pub by: String,
    /// Written on trial: loaded and recorded, never applied (the library's trial rule).
    pub trial: bool,
    /// Switch it on (plug it in) once written.
    pub plug: bool,
}

/// What a write did, for the reply and for its undo.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Written {
    pub key: String,
    pub previous: Option<String>,
    /// Stack versions made: (from, to); `from` empty for a new stack.
    pub stacks: Vec<(String, String)>,
    /// Uses switched: (use, from, to); `from` empty when the use was not switched before.
    pub switched: Vec<(String, String, String)>,
}

impl Written {
    /// The write as audit row fields (read back by `unwrite`).
    #[must_use]
    pub fn fields(&self) -> Vec<(&'static str, Value)> {
        vec![
            ("cartridge", s(&self.key)),
            ("previous", self.previous.as_deref().map_or(Value::Null, s)),
            ("stacks", Value::Array(self.stacks.iter().map(|(a, b)| obj(vec![("from", s(a)), ("to", s(b))])).collect())),
            ("switched", Value::Array(self.switched.iter().map(|(u, a, b)| obj(vec![("use", s(u)), ("from", s(a)), ("to", s(b))])).collect())),
        ]
    }
}

/// Moves every stack holding `prev` to a new version holding `key`, and switches the uses that ran
/// them; returns (stacks made, uses switched).
fn restack(lib: &mut Library, prev: &str, key: &str, out: &mut Written) -> Result<(), String> {
    let holding: Vec<Stack> = lib.stacks.iter().filter(|st| st.cartridges.iter().any(|k| k == prev) && lib.stack(&st.id).is_some_and(|newest| newest.key() == st.key())).cloned().collect();
    for st in holding {
        let next = Stack {
            version: lib.stacks.iter().filter(|x| x.id == st.id).map(|x| x.version).max().unwrap_or(0) + 1,
            cartridges: st.cartridges.iter().map(|k| if k == prev { key.to_owned() } else { k.clone() }).collect(),
            hash: String::new(),
            ..st.clone()
        };
        let nk = lib.add_stack(next).map_err(|e| e.to_string())?;
        out.stacks.push((st.key(), nk.clone()));
        let uses: Vec<String> = lib.active.iter().filter(|(_, k)| **k == st.key()).map(|(u, _)| u.clone()).collect();
        for u in uses {
            lib.switch_use(&u, Some(&nk)).map_err(|e| e.to_string())?;
            out.switched.push((u, st.key(), nk.clone()));
        }
    }
    Ok(())
}

/// Writes a new version of a stated cartridge through the library (`Library::add_cartridge`): the
/// learned values of the version before are kept, the rules are the given set. Every stack that
/// held the version before gets a new version holding the new one, and every use that ran such a
/// stack is switched to it; a cartridge with no stack gets `<id>-stack` for use
/// `preference:stated:<id>`. With `plug`, the use is switched on. The catalogue is not saved here.
/// # Errors
/// A bad id, no rules or too many, an id that names a cartridge of another family or one the
/// preference fitter writes, or a library refusal.
pub fn write(lib: &mut Library, w: &Write) -> Result<Written, String> {
    if !valid_id(&w.id) {
        return Err(format!("{}: a cartridge id is letters, digits and - _ . (at most 64)", w.id));
    }
    if w.rules.is_empty() {
        return Err("a preference cartridge needs at least one rule; to take a whole cartridge out, remove it".into());
    }
    if w.rules.len() > MAX_RULES {
        return Err(format!("{} rules: a cartridge holds at most {MAX_RULES}", w.rules.len()));
    }
    let prev = lib.cartridge(&w.id).cloned();
    let mut values = BTreeMap::new();
    if let Some(p) = &prev {
        if p.family != PREFERENCE {
            return Err(format!("{} is a {} cartridge; stated rules go in preference cartridges", p.key(), if p.family.is_empty() { "unlabelled" } else { &p.family }));
        }
        if p.notes.get("held_brier").is_ok() {
            return Err(format!("{} is written by the idle preference fitter, which writes its next version from learned values only; give your rules their own cartridge", p.key()));
        }
        values = contents(lib, &p.key())?.1;
    }
    values.extend(settings_of(&w.id, &w.rules));
    let counts = |e: Enforcement| w.rules.iter().filter(|r| r.enforcement() == e).count();
    let wants = w.rules.iter().filter(|r| matches!(r.kind, Kind::Want { .. })).count();
    let wanted = if wants == 0 { String::new() } else { format!("; {wants} of them wants, {} pushed toward in the Writer's decoding", counts(Enforcement::Bias)) };
    let spec = CartridgeSpec {
        id: w.id.clone(),
        version: next_version(lib, &w.id),
        family: PREFERENCE.to_owned(),
        domain: prev.as_ref().map(|p| p.domain.clone()).filter(|d| !d.is_empty()).unwrap_or_else(|| DOMAIN.to_owned()),
        scope: w.scope.to_text(),
        model: "settings".to_owned(),
        weights: String::new(),
        layers: None,
        name: w.name.clone(),
        description: format!(
            "Your stated preferences: {} rules ({} enforced, {} checked, {} guidance{wanted}), applied to text the Nano writes and passed to your agent; holds no weights.",
            w.rules.len(),
            counts(Enforcement::Transform),
            counts(Enforcement::Check),
            counts(Enforcement::Guidance)
        ),
        made_from: w.made_from.clone(),
        effect: "stated rules, not measured: each is applied, checked or passed on as it says".to_owned(),
    };
    let notes = obj(vec![
        ("stated_rules", n(w.rules.len() as u64)),
        ("by", s(&w.by)),
        ("edited_from", prev.as_ref().map_or(Value::Null, |p| s(&p.key()))),
        ("trial_rule", s("a stated rule is the user's word, not a measured change: it serves when plugged unless written on trial")),
    ]);
    let key = lib.add_cartridge(&spec, &Filter::Settings(values), notes, Vec::new(), w.trial).map_err(|e| e.to_string())?;
    let mut out = Written { key: key.clone(), previous: prev.as_ref().map(crate::library::Cartridge::key), ..Written::default() };
    if let Some(p) = out.previous.clone() {
        restack(lib, &p, &key, &mut out)?;
    }
    if out.stacks.is_empty() {
        let id = format!("{}-stack", w.id);
        let st = Stack {
            id: id.clone(),
            version: lib.stacks.iter().filter(|x| x.id == id).map(|x| x.version).max().unwrap_or(0) + 1,
            use_for: format!("{USE_PREFIX}{}", w.id),
            piece: None,
            cartridges: vec![key.clone()],
            description: format!("the stated preference cartridge {}, applied after every other stack", w.id),
            hash: String::new(),
            trial: w.trial,
        };
        let nk = lib.add_stack(st).map_err(|e| e.to_string())?;
        out.stacks.push((String::new(), nk));
    }
    if w.plug && out.switched.is_empty() {
        let (use_for, stack) = plug_target(lib, &w.id).ok_or_else(|| format!("{key}: no stack to plug it in with"))?;
        let before = lib.active.get(&use_for).cloned().unwrap_or_default();
        if lib.switch_use(&use_for, Some(&stack)).map_err(|e| e.to_string())?.is_some() {
            out.switched.push((use_for, before, stack));
        }
    }
    Ok(out)
}

/// The use and stack that plug the newest version of cartridge `id` in: the newest version of a
/// stack holding it, its own stack (`<id>-stack`) first.
#[must_use]
pub fn plug_target(lib: &Library, id: &str) -> Option<(String, String)> {
    let key = lib.cartridge(id)?.key();
    let own = format!("{id}-stack");
    let mut holding: Vec<&Stack> = lib.stacks.iter().filter(|st| st.cartridges.contains(&key) && lib.stack(&st.id).is_some_and(|x| x.key() == st.key())).collect();
    holding.sort_by_key(|st| (st.id != own, st.key()));
    holding.first().map(|st| (st.use_for.clone(), st.key()))
}

/// Whether the newest version of cartridge `id` is plugged in: a use is switched to a stack holding it.
#[must_use]
pub fn plugged(lib: &Library, id: &str) -> bool {
    let Some(key) = lib.cartridge(id).map(crate::library::Cartridge::key) else { return false };
    lib.active.values().any(|k| lib.stack(k).is_some_and(|st| st.cartridges.contains(&key)))
}

/// Undoes a write recorded by `Written::fields`: each use switched goes back, the stack versions
/// made and the cartridge version written leave the catalogue (the file stays where it is). A use
/// switched since to something else is left as it is. Returns what was put back, in words.
#[must_use]
pub fn unwrite(lib: &mut Library, row: &Value) -> Vec<String> {
    let t = |v: &Value, k: &str| v.get(k).and_then(Value::string).unwrap_or("").to_owned();
    let mut done = Vec::new();
    for sw in row.get("switched").and_then(Value::array).unwrap_or_default() {
        let (u, from, to) = (t(sw, "use"), t(sw, "from"), t(sw, "to"));
        if lib.active.get(&u).is_some_and(|k| *k == to) {
            if from.is_empty() {
                lib.active.remove(&u);
            } else {
                lib.active.insert(u.clone(), from.clone());
            }
            done.push(format!("{u} back to {}", if from.is_empty() { "not switched" } else { &from }));
        }
    }
    for st in row.get("stacks").and_then(Value::array).unwrap_or_default() {
        let to = t(st, "to");
        if !lib.active.values().any(|k| *k == to) && lib.stacks.iter().any(|x| x.key() == to) {
            lib.stacks.retain(|x| x.key() != to);
            done.push(format!("stack {to} taken out"));
        }
    }
    let key = t(row, "cartridge");
    if !key.is_empty() && !lib.stacks.iter().any(|st| st.cartridges.contains(&key)) && lib.unlist(&key) {
        done.push(format!("cartridge {key} taken out (its file stays)"));
    }
    done
}

/// What removing a cartridge takes out, for its preview and its undo.
#[derive(Clone, Debug, PartialEq)]
pub struct Removal {
    pub cartridges: Vec<crate::library::Cartridge>,
    pub stacks: Vec<Stack>,
    /// (use, stack key) switched on now.
    pub active: Vec<(String, String)>,
    /// The files the versions keep on the drive.
    pub files: Vec<String>,
}

impl Removal {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("cartridges", Value::Array(self.cartridges.iter().map(crate::library::Cartridge::to_value).collect())),
            ("stacks", Value::Array(self.stacks.iter().map(Stack::to_value).collect())),
            ("active", Value::Array(self.active.iter().map(|(u, k)| obj(vec![("use", s(u)), ("stack", s(k))])).collect())),
            ("files", Value::Array(self.files.iter().map(|f| s(f)).collect())),
        ])
    }
}

/// What removing every version of cartridge `id` would take out: the versions, the stacks that
/// hold only its versions, and the uses switched to those stacks.
/// # Errors
/// No such cartridge, not a preference cartridge, or a stack that also holds other cartridges.
pub fn removal(lib: &Library, id: &str) -> Result<Removal, String> {
    let cartridges: Vec<crate::library::Cartridge> = lib.cartridges.iter().filter(|c| c.id == id).cloned().collect();
    let Some(first) = cartridges.first() else { return Err(format!("no cartridge {id}")) };
    if first.family != PREFERENCE {
        return Err(format!("{id} is not a preference cartridge; remove other cartridges from the Cartridges page"));
    }
    let keys: Vec<String> = cartridges.iter().map(crate::library::Cartridge::key).collect();
    let mut stacks = Vec::new();
    for st in lib.stacks.iter().filter(|st| st.cartridges.iter().any(|k| keys.contains(k))) {
        if let Some(other) = st.cartridges.iter().find(|k| !keys.contains(k)) {
            return Err(format!("stack {} also holds {other}; take {id} out of that stack first", st.key()));
        }
        stacks.push(st.clone());
    }
    let skeys: Vec<String> = stacks.iter().map(Stack::key).collect();
    let active = lib.active.iter().filter(|(_, k)| skeys.contains(k)).map(|(u, k)| (u.clone(), k.clone())).collect();
    let files = cartridges.iter().flat_map(|c| c.files.iter().map(|f| f.name.clone())).collect();
    Ok(Removal { cartridges, stacks, active, files })
}

/// Takes a removal out of the catalogue (the files stay on the drive, so it can be undone).
pub fn remove(lib: &mut Library, r: &Removal) {
    for (u, _) in &r.active {
        lib.active.remove(u);
    }
    let skeys: Vec<String> = r.stacks.iter().map(Stack::key).collect();
    lib.stacks.retain(|st| !skeys.contains(&st.key()));
    for c in &r.cartridges {
        lib.unlist(&c.key());
    }
}

/// Puts a removal recorded by `Removal::to_value` back: its versions, stacks and switched uses.
/// # Errors
/// A record that does not read.
pub fn restore(lib: &mut Library, v: &Value) -> Result<Vec<String>, String> {
    let mut done = Vec::new();
    for c in v.get("cartridges").and_then(Value::array).unwrap_or_default() {
        let c = crate::library::Cartridge::from_value(c).map_err(|e| e.to_string())?;
        if lib.cartridge(&c.key()).is_none() {
            done.push(format!("cartridge {}", c.key()));
            lib.cartridges.push(c);
        }
    }
    for st in v.get("stacks").and_then(Value::array).unwrap_or_default() {
        let st = Stack::from_value(st).map_err(|e| e.to_string())?;
        if lib.stack(&st.key()).is_none() {
            done.push(format!("stack {}", st.key()));
            lib.stacks.push(st);
        }
    }
    for a in v.get("active").and_then(Value::array).unwrap_or_default() {
        let (u, k) = (a.get("use").and_then(Value::string).unwrap_or(""), a.get("stack").and_then(Value::string).unwrap_or(""));
        if !u.is_empty() && !k.is_empty() && !lib.active.contains_key(u) {
            lib.active.insert(u.to_owned(), k.to_owned());
            done.push(format!("{u} switched to {k}"));
        }
    }
    Ok(done)
}

/// One stated cartridge as the list shows it: its newest version, name, scope, rules, whether it
/// also holds learned values, its versions, plug state and the use and stack that plug it.
#[must_use]
pub fn view(lib: &Library, id: &str) -> Option<Value> {
    let c = lib.cartridge(id)?;
    let (rules, other) = match contents(lib, &c.key()) {
        Ok(x) => x,
        Err(e) => return Some(obj(vec![("id", s(id)), ("key", s(&c.key())), ("error", s(&e))])),
    };
    let target = plug_target(lib, id);
    let versions: Vec<Value> = lib.cartridges.iter().filter(|x| x.id == id).map(|x| obj(vec![("key", s(&x.key())), ("version", n(x.version)), ("made", s(&x.made))])).collect();
    Some(obj(vec![
        ("id", s(&c.id)),
        ("key", s(&c.key())),
        ("version", n(c.version)),
        ("name", s(&c.name)),
        ("scope", s(&c.scope)),
        ("scope_words", s(&Scope::parse(&c.scope).map_or_else(|_| c.scope.clone(), |x| x.words()))),
        ("family", s(&c.family)),
        ("domain", s(&c.domain)),
        ("made", s(&c.made)),
        ("made_from", s(&c.made_from)),
        ("trial", Value::Bool(c.trial)),
        ("plugged", Value::Bool(plugged(lib, id))),
        ("use", target.as_ref().map_or(Value::Null, |t| s(&t.0))),
        ("stack", target.as_ref().map_or(Value::Null, |t| s(&t.1))),
        ("learned_values", Value::Bool(learned_values(&other))),
        ("learned_keys", n(other.keys().filter(|k| k.starts_with("pref.")).count() as u64)),
        ("weights", Value::Bool(false)),
        ("rules", Value::Array(rules.iter().map(Rule::view).collect())),
        ("versions", Value::Array(versions)),
    ]))
}

/// The ids of the cartridges that hold stated rules (their newest version's domain says so, or it
/// holds `pref.stated.*` values), in order.
#[must_use]
pub fn stated_ids(lib: &Library) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    for c in lib.cartridges.iter().filter(|c| c.family == PREFERENCE) {
        if ids.contains(&c.id) {
            continue;
        }
        let Some(newest) = lib.cartridge(&c.id) else { continue };
        if newest.domain == DOMAIN || contents(lib, &newest.key()).is_ok_and(|(r, _)| !r.is_empty()) {
            ids.push(c.id.clone());
        }
    }
    ids.sort();
    ids
}

/// Section check: the common forms parse to their rules, an unknown statement stays guidance, the
/// em dash transform leaves code and evidence lines untouched, and rules round-trip through values.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let rules = with_ids(&[], parse("No em dashes. Use 'use' instead of 'utilize'; keep power traces short on .kicad_pcb files", &Scope::global(), None));
    let kinds: Vec<&str> = rules.iter().map(Rule::kind_name).collect();
    if kinds != ["replace", "vocabulary", "guidance"] {
        return Err(format!("parse gave {kinds:?}"));
    }
    if rules.get(2).is_none_or(|r| r.scope.kinds.first().map(String::as_str) != Some("kicad_pcb")) {
        return Err("a scope clause naming .kicad_pcb did not scope the rule".into());
    }
    let held = InForce { held: rules.iter().map(|r| Held { cartridge: "c@1".into(), name: "c".into(), rule: r.clone() }).collect() };
    let draft = "It works \u{2014} mostly.\nsrc/a.rs:3 let x = 1; // \u{2014}\n```\nlet y = \u{2014};\n```";
    let out = apply(draft, &held, None, None);
    let want = "It works, mostly.\nsrc/a.rs:3 let x = 1; // \u{2014}\n```\nlet y = \u{2014};\n```";
    if out.text != want {
        return Err(format!("apply gave {:?}", out.text));
    }
    if rules_of(&settings_of("c", &rules)).get("c") != Some(&rules) {
        return Err("rules did not round-trip through settings values".into());
    }
    let want = parse("strongly prefer 'colour'", &Scope::global(), None);
    if !matches!(want.as_slice(), [Rule { kind: Kind::Want { term, strength: STRENGTH_STRONG, .. }, .. }] if term == "colour") {
        return Err(format!("a want parsed as {want:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn kinds(msg: &str) -> Vec<Kind> {
        parse(msg, &Scope::global(), Some("proj")).into_iter().map(|r| r.kind).collect()
    }
    fn held(msg: &str) -> InForce {
        let rules = with_ids(&[], parse(msg, &Scope::global(), Some("proj")));
        InForce { held: rules.into_iter().map(|rule| Held { cartridge: "stated-t@1".into(), name: "t".into(), rule }).collect() }
    }
    const EM: &str = "\u{2014}";

    #[test]
    fn the_common_forms_parse_to_their_rules() {
        assert_eq!(kinds("no em dashes"), vec![Kind::Replace { from: EM.into(), to: ", ".into(), what: "em dash (U+2014)".into() }]);
        assert_eq!(kinds("Please remove all em-dashes."), kinds("no em dashes"));
        assert_eq!(kinds("replace em dashes with ' - '"), vec![Kind::Replace { from: EM.into(), to: " - ".into(), what: "em dash (U+2014)".into() }]);
        assert_eq!(kinds("use \"use\" instead of \"utilize\""), vec![Kind::Vocabulary { avoid: "utilize".into(), prefer: "use".into() }]);
        assert_eq!(kinds("prefer colour over color"), vec![Kind::Vocabulary { avoid: "color".into(), prefer: "colour".into() }]);
        assert_eq!(kinds("never say 'leverage' in replies"), vec![Kind::Ban { phrase: "leverage".into() }]);
        assert_eq!(kinds("don't say basically"), vec![Kind::Ban { phrase: "basically".into() }]);
        assert_eq!(kinds("indent with 4 spaces"), vec![Kind::Indent { tabs: false, width: 4 }]);
        assert_eq!(kinds("use tabs for indentation"), vec![Kind::Indent { tabs: true, width: 0 }]);
        assert_eq!(kinds("max line length 100"), vec![Kind::MaxLine { chars: 100 }]);
        assert_eq!(kinds("keep lines under 120 characters"), vec![Kind::MaxLine { chars: 120 }]);
        assert_eq!(kinds("snake_case for functions"), vec![Kind::Naming { case: Case::Snake, target: "functions".into() }]);
        assert_eq!(kinds("use PascalCase for types"), vec![Kind::Naming { case: Case::Pascal, target: "types".into() }]);
        assert_eq!(kinds("camelCase instead of snake_case for variables"), vec![Kind::Naming { case: Case::Camel, target: "variables".into() }]);
        assert_eq!(kinds("remove smart quotes").len(), 4, "a named group gives one rule per character");
    }

    #[test]
    fn several_statements_give_several_rules_and_the_unknown_stays_verbatim() {
        let msg = "No em dashes; indent with 4 spaces.\nAlways ground the shield at one end only";
        let rules = with_ids(&[], parse(msg, &Scope::global(), None));
        assert_eq!(rules.iter().map(Rule::kind_name).collect::<Vec<_>>(), ["replace", "indent", "guidance"]);
        assert_eq!(rules.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["r1", "r2", "r3"]);
        assert_eq!(rules.get(2).map(|r| r.said.as_str()), Some("Always ground the shield at one end only"), "kept verbatim, never guessed into an enforced rule");
        let both = kinds("no em dashes and max line length 100");
        assert_eq!(both.len(), 2, "{both:?}");
        // A statement only partly known is kept whole as guidance.
        assert!(matches!(kinds("no em dashes and keep the copper pour solid").as_slice(), [Kind::Guidance { .. }]));
        assert!(kinds("   ").is_empty());
    }

    #[test]
    fn scope_clauses_narrow_rules_to_file_kinds_folders_and_projects() {
        let r = parse("For PCB designs, keep decoupling capacitors within 2 mm of the pin", &Scope::global(), None);
        let first = r.first().expect("rule");
        assert!(matches!(&first.kind, Kind::Guidance { domain } if domain == "pcb"));
        assert!(first.scope.kinds.contains(&"kicad_pcb".to_owned()) && first.scope.kinds.contains(&"kicad_sch".to_owned()));
        assert!(first.scope.matches(None, Some("hw/board.kicad_pcb")) && !first.scope.matches(None, Some("src/main.rs")) && !first.scope.matches(None, None));
        let r = parse("max line length 100 in .rs files", &Scope::global(), None);
        assert_eq!(r.first().map(|x| x.scope.kinds.clone()), Some(vec!["rs".to_owned()]));
        let r = parse("no em dashes under docs/", &Scope::global(), None);
        assert_eq!(r.first().map(|x| x.scope.folders.clone()), Some(vec!["docs".to_owned()]));
        let base = Scope::parse("project:alpha").expect("scope");
        let r = parse("no em dashes", &base, None);
        assert!(r.first().is_some_and(|x| x.scope.matches(Some("alpha"), None) && !x.scope.matches(Some("beta"), None)));
        assert_eq!(Scope::parse("project:a; kinds:.kicad_pcb,rs; folders:docs/").map(|s| s.to_text()), Ok("project:a; kinds:kicad_pcb,rs; folders:docs".to_owned()));
        assert!(Scope::parse("kinds:../x").is_err() && Scope::parse("planet:mars").is_err());
    }

    #[test]
    fn a_replacement_that_holds_what_it_replaces_is_not_enforced() {
        assert!(matches!(kinds("use 'used' instead of 'use'").as_slice(), [Kind::Guidance { .. }]));
    }

    #[test]
    fn transforms_apply_to_prose_and_never_to_code_or_evidence() {
        let rules = held("no em dashes. use 'use' instead of 'utilize'");
        let draft = [
            "We utilize the cache \u{2014} it is fast.",
            "Utilize it \u{2014}",
            "\u{2014} a list item",
            "crates/a/src/lib.rs:12 // utilize \u{2014} here",
            "@@ crates/a/src/lib.rs:10-14",
            "> quoted \u{2014} line",
            "Call `utilize()` \u{2014} then return.",
            "```rust",
            "let utilize = 1; // \u{2014}",
            "```",
        ]
        .join("\n");
        let out = apply(&draft, &rules, None, None);
        let want = [
            "We use the cache, it is fast.",
            "Use it",
            "a list item",
            "crates/a/src/lib.rs:12 // utilize \u{2014} here",
            "@@ crates/a/src/lib.rs:10-14",
            "> quoted \u{2014} line",
            "Call `utilize()`, then return.",
            "```rust",
            "let utilize = 1; // \u{2014}",
            "```",
        ]
        .join("\n");
        assert_eq!(out.text, want);
        assert!(out.changed() && out.mode == "prose");
        // Applying again changes nothing.
        let again = apply(&out.text, &rules, None, None);
        assert_eq!(again.text, out.text);
        assert!(!again.changed());
        // The Nano's own text gets the same transforms.
        assert_eq!(own_text("Fast \u{2014} cached.", &rules, Some("src/a.rs")), "Fast, cached.");
        assert_eq!(own_text("unchanged", &InForce::default(), None), "unchanged");
    }

    #[test]
    fn code_drafts_are_checked_never_rewritten() {
        let rules = held("no em dashes. indent with 4 spaces. max line length 20. snake_case for functions. never say 'hack'");
        let code = "fn doThing() {\n   let x = 1; // a hack \u{2014} here\n    let y = 2;\n}";
        let out = apply(code, &rules, None, Some("src/lib.rs"));
        assert_eq!(out.mode, "code");
        assert_eq!(out.text, code, "code is never rewritten");
        let lines: Vec<(usize, &str)> = out.violations.iter().map(|v| (v.line, v.rule.as_str())).collect();
        assert!(lines.contains(&(1, "r4")), "{lines:?}");
        assert!(lines.contains(&(2, "r1")) && lines.contains(&(2, "r2")) && lines.contains(&(2, "r3")) && lines.contains(&(2, "r5")), "{lines:?}");
        assert!(!lines.contains(&(3, "r2")));
        // In prose, a fenced block is checked for formatting only, and bans in prose are reported.
        let prose = "This is a hack.\n```\n\tlet a = 1;\n```";
        let out = apply(prose, &held("indent with 4 spaces. never say hack"), None, None);
        assert_eq!(out.text, prose);
        let found: Vec<(usize, &str)> = out.violations.iter().map(|v| (v.line, v.rule.as_str())).collect();
        assert_eq!(found, vec![(1, "r2"), (3, "r1")]);
    }

    #[test]
    fn scoped_rules_apply_only_where_they_fit_and_reach_the_block() {
        let mut rules = held("for .kicad_pcb files, keep power traces at least 0.5 mm wide");
        rules.held.extend(held("no em dashes").held);
        assert_eq!(apply("x", &rules, None, Some("a.md")).rules.len(), 1);
        assert_eq!(apply("x", &rules, None, Some("hw/main.kicad_pcb")).rules.len(), 2);
        let b = block(&rules, Some("proj"), &["src/main.rs".to_owned()]);
        assert_eq!(b.len(), 2, "{b:?}");
        assert!(b.first().is_some_and(|l| l.starts_with("preferences in force")));
        assert!(b.get(1).is_some_and(|l| l.contains("no em dashes") && l.contains("[enforced")));
        let b = block(&rules, Some("proj"), &["hw/main.kicad_pcb".to_owned()]);
        assert_eq!(b.len(), 3);
        assert!(b.iter().any(|l| l.contains("[guidance") && l.contains(".kicad_pcb")));
        assert!(block(&InForce::default(), None, &[]).is_empty());
    }

    #[test]
    fn banned_strings_for_the_writer_mask() {
        let rules = held("no em dashes. never say leverage. use 'use' instead of 'utilize'");
        let (marks, words) = rules.banned();
        assert_eq!(marks, vec![EM.to_owned()]);
        assert_eq!(words, vec!["leverage".to_owned(), "utilize".to_owned()]);
    }

    #[test]
    fn rules_round_trip_and_an_unknown_kind_reads_as_guidance() {
        let rules = with_ids(&[], parse("no em dashes. indent with tabs. snake_case for functions. hello there", &Scope::global(), None));
        let vals = settings_of("stated-x", &rules);
        assert!(vals.keys().all(|k| k.starts_with("pref.stated.stated-x.r")));
        assert_eq!(rules_of(&vals).remove("stated-x"), Some(rules));
        let odd = nmjson::parse(br#"{"id":"r9","kind":"telepathy","said":"read my mind"}"#).expect("json");
        let r = Rule::from_value(&odd);
        assert!(matches!(r.kind, Kind::Guidance { .. }) && r.said == "read my mind");
        // Learned values beside stated rules are told apart, and pref_fit reads only its own.
        let mut both = settings_of("stated-x", &[r]);
        both.insert("pref.global".into(), "3/5".into());
        assert!(learned_values(&both) && !learned_values(&settings_of("stated-x", &[])));
        assert_eq!(crate::pref_fit::PrefModel::from_settings(&both).global, (3, 5));
    }

    fn home(name: &str) -> PathBuf {
        let d = nmtype::nano_root().join("scratch").join("unit-tests").join(format!("nmlearn-stated-{name}-{}-{}", std::process::id(), nmtype::now_ms()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("dir");
        d
    }

    fn w(id: &str, msg: &str, existing: &[Rule], plug: bool) -> Write {
        let mut rules = existing.to_vec();
        rules.extend(with_ids(existing, parse(msg, &Scope::global(), None)));
        Write { id: id.into(), name: "my writing".into(), scope: Scope::global(), rules, made_from: "test".into(), by: "test".into(), trial: false, plug }
    }

    fn reopen(h: &Path, lib: &Library) -> Library {
        lib.save().expect("save");
        Library::open(h).expect("open")
    }

    #[test]
    fn every_edit_is_a_new_version_and_a_plugged_cartridge_stays_plugged() {
        let h = home("versions");
        let mut lib = Library::open(&h).expect("library");
        let one = write(&mut lib, &w("stated-mine", "no em dashes", &[], true)).expect("write");
        assert_eq!(one.key, "stated-mine@1");
        assert_eq!(one.stacks, vec![(String::new(), "stated-mine-stack@1".to_owned())]);
        assert_eq!(one.switched, vec![("preference:stated:stated-mine".to_owned(), String::new(), "stated-mine-stack@1".to_owned())]);
        let mut lib = reopen(&h, &lib);
        assert!(plugged(&lib, "stated-mine"));
        assert_eq!(InForce::from_library(&lib).held.len(), 1);
        let (rules, _) = contents(&lib, "stated-mine").expect("contents");
        let two = write(&mut lib, &w("stated-mine", "max line length 100", &rules, false)).expect("edit");
        assert_eq!(two.key, "stated-mine@2");
        assert_eq!(two.previous.as_deref(), Some("stated-mine@1"));
        assert_eq!(two.switched, vec![("preference:stated:stated-mine".to_owned(), "stated-mine-stack@1".to_owned(), "stated-mine-stack@2".to_owned())]);
        let mut lib = reopen(&h, &lib);
        assert!(plugged(&lib, "stated-mine"));
        assert_eq!(InForce::from_library(&lib).held.len(), 2);
        assert_eq!(lib.cartridges.iter().filter(|c| c.id == "stated-mine").count(), 2, "the earlier version stays in the library");
        // Undo of the edit: the use goes back to the first stack, version 2 leaves the catalogue.
        let row = obj(two.fields());
        let done = unwrite(&mut lib, &row);
        assert!(!done.is_empty());
        let mut lib = reopen(&h, &lib);
        assert_eq!(lib.cartridge("stated-mine").map(crate::library::Cartridge::key).as_deref(), Some("stated-mine@1"));
        assert_eq!(InForce::from_library(&lib).held.len(), 1);
        // The next write skips the version whose file stays on the drive.
        assert_eq!(next_version(&lib, "stated-mine"), 3);
        // Unplugged: nothing in force; trial cartridges are recorded, never applied.
        lib.switch_use("preference:stated:stated-mine", None).expect("off");
        assert!(!plugged(&lib, "stated-mine") && InForce::from_library(&lib).is_empty());
        let t = Write { trial: true, ..w("stated-trial", "no em dashes", &[], true) };
        write(&mut lib, &t).expect("trial write");
        assert!(InForce::from_library(&lib).is_empty(), "a trial cartridge never changes what is served");
        let _ = std::fs::remove_dir_all(&h);
    }

    #[test]
    fn removal_takes_every_version_out_and_restore_puts_it_back() {
        let h = home("remove");
        let mut lib = Library::open(&h).expect("library");
        write(&mut lib, &w("stated-gone", "no em dashes", &[], true)).expect("write");
        let (rules, _) = contents(&lib, "stated-gone").expect("contents");
        write(&mut lib, &w("stated-gone", "indent with tabs", &rules, false)).expect("edit");
        let mut lib = reopen(&h, &lib);
        assert_eq!(stated_ids(&lib), vec!["stated-gone".to_owned()]);
        let r = removal(&lib, "stated-gone").expect("removal");
        assert_eq!((r.cartridges.len(), r.stacks.len(), r.active.len()), (2, 2, 1));
        let rec = r.to_value();
        remove(&mut lib, &r);
        let mut lib = reopen(&h, &lib);
        assert!(lib.cartridge("stated-gone").is_none() && stated_ids(&lib).is_empty() && InForce::from_library(&lib).is_empty());
        let back = restore(&mut lib, &rec).expect("restore");
        assert_eq!(back.len(), 5, "{back:?}");
        let lib = reopen(&h, &lib);
        assert!(plugged(&lib, "stated-gone"));
        assert_eq!(InForce::from_library(&lib).held.len(), 2);
        assert!(removal(&lib, "stated-none").is_err());
        let _ = std::fs::remove_dir_all(&h);
    }

    #[test]
    fn learned_values_are_kept_and_the_fitters_cartridge_is_refused() {
        let h = home("learned");
        let mut lib = Library::open(&h).expect("library");
        let spec = CartridgeSpec {
            id: "pref-shared".into(),
            version: 1,
            family: PREFERENCE.into(),
            domain: "preferences".into(),
            scope: "user".into(),
            model: "settings".into(),
            weights: String::new(),
            layers: None,
            name: "distilled".into(),
            description: "learned values".into(),
            made_from: "distillation".into(),
            effect: String::new(),
        };
        let vals: BTreeMap<String, String> = [("pref.global".to_owned(), "4/6".to_owned())].into();
        lib.add_cartridge(&spec, &Filter::Settings(vals), Value::Null, Vec::new(), false).expect("add");
        let out = write(&mut lib, &w("pref-shared", "no em dashes", &[], false)).expect("stated rules beside learned values");
        let (rules, other) = contents(&lib, &out.key).expect("contents");
        assert_eq!(rules.len(), 1);
        assert_eq!(other.get("pref.global").map(String::as_str), Some("4/6"));
        let v = view(&lib, "pref-shared").expect("view");
        assert!(matches!(v.get("learned_values"), Ok(Value::Bool(true))));
        let fitted = CartridgeSpec { id: "pref-fitted".into(), ..spec };
        lib.add_cartridge(&fitted, &Filter::Settings(BTreeMap::new()), obj(vec![("held_brier", s("0.1"))]), Vec::new(), false).expect("add");
        assert!(write(&mut lib, &w("pref-fitted", "no em dashes", &[], false)).is_err_and(|e| e.contains("idle preference fitter")));
        let _ = std::fs::remove_dir_all(&h);
    }

    /// English statements whose parsed rules must not change when forms of other languages are
    /// added: the forms each English path takes, statements it keeps as guidance, and statements
    /// that hold words of other languages' forms by chance.
    const ENGLISH: &[&str] = &[
        "no em dashes",
        "Please remove all em-dashes.",
        "replace em dashes with ' - '",
        "replace em dashes with commas",
        "use \"use\" instead of \"utilize\"",
        "prefer colour over color",
        "never say 'leverage' in replies",
        "don't say basically",
        "avoid synergy",
        "never use the word 'robust'",
        "indent with 4 spaces",
        "use tabs for indentation",
        "tabs not spaces",
        "4 spaces instead of tabs",
        "max line length 100",
        "keep lines under 120 characters",
        "wrap at 80 columns",
        "snake_case for functions",
        "use PascalCase for types",
        "camelCase instead of snake_case for variables",
        "kebab-case for files",
        "SCREAMING_SNAKE_CASE for constants",
        "camelCase for func names",
        "remove smart quotes",
        "no 'TODO'",
        "strip '!!'",
        "colour -> color",
        "no em dashes and max line length 100",
        "no em dashes and keep the copper pour solid",
        "retire em dashes",
        "retire 'legacy'",
        "sin 'x'",
        "use e instead of f",
        "Always ground the shield at one end only",
        "keep functions short",
        "write tests first",
        "max line length 100 in .rs files",
        "no em dashes under docs/",
        "For PCB designs, keep decoupling capacitors within 2 mm of the pin",
        "use 'used' instead of 'use'",
    ];

    /// Each English statement's rules as `{:?}`, one line each.
    fn english_snapshot() -> String {
        ENGLISH.iter().map(|m| format!("{m} => {:?}", parse(m, &Scope::global(), Some("proj")))).collect::<Vec<_>>().join("\n")
    }

    /// The SHA-256 of `english_snapshot()` as the parser gave it before the forms of other
    /// languages were added. A change here is a change of English behaviour.
    const ENGLISH_SNAPSHOT_SHA: &str = "d76efeff198550b1fa57ec517926ad566f252d4733ab45c6b5ed3343c732c02c";

    fn em_rule() -> Kind {
        Kind::Replace { from: EM.into(), to: ", ".into(), what: "em dash (U+2014)".into() }
    }
    fn em_to(to: &str) -> Kind {
        Kind::Replace { from: EM.into(), to: to.into(), what: "em dash (U+2014)".into() }
    }
    fn vocab(avoid: &str, prefer: &str) -> Vec<Kind> {
        vec![Kind::Vocabulary { avoid: avoid.into(), prefer: prefer.into() }]
    }
    fn ban(p: &str) -> Vec<Kind> {
        vec![Kind::Ban { phrase: p.into() }]
    }
    fn naming(case: Case, target: &str) -> Vec<Kind> {
        vec![Kind::Naming { case, target: target.into() }]
    }
    /// A statement no form reads stays one guidance rule holding the user's words.
    fn stays_guidance(msg: &str) {
        let r = parse(msg, &Scope::global(), None);
        assert!(matches!(r.as_slice(), [Rule { kind: Kind::Guidance { .. }, said, .. }] if said == msg), "{msg}: {r:?}");
    }

    #[test]
    fn spanish_forms_parse_to_their_rules() {
        assert_eq!(kinds("sin rayas"), vec![em_rule()]);
        assert_eq!(kinds("Elimina todas las rayas."), vec![em_rule()]);
        assert_eq!(kinds("No uses guiones largos"), vec![em_rule()]);
        assert_eq!(kinds("quita las comillas curvas").len(), 4);
        assert_eq!(kinds("usa 'usar' en lugar de 'utilizar'"), vocab("utilizar", "usar"));
        assert_eq!(kinds("reemplaza las rayas por comas"), vec![em_rule()]);
        assert_eq!(kinds("reemplaza las rayas por ' - '"), vec![em_to(" - ")]);
        assert_eq!(kinds("nunca digas 'b\u{e1}sicamente'"), ban("b\u{e1}sicamente"));
        assert_eq!(kinds("no uses la palabra 'sinergia'"), ban("sinergia"));
        assert_eq!(kinds("evita sinergia"), ban("sinergia"));
        assert_eq!(kinds("elimina 'TODO'"), vocab("TODO", ""));
        assert_eq!(kinds("sangr\u{ed}a de 2 espacios"), vec![Kind::Indent { tabs: false, width: 2 }]);
        assert_eq!(kinds("indenta con 4 espacios"), vec![Kind::Indent { tabs: false, width: 4 }]);
        assert_eq!(kinds("usa tabuladores"), vec![Kind::Indent { tabs: true, width: 0 }]);
        assert_eq!(kinds("Longitud m\u{e1}xima de l\u{ed}nea 100"), vec![Kind::MaxLine { chars: 100 }]);
        assert_eq!(kinds("l\u{ed}neas de como mucho 120 caracteres"), vec![Kind::MaxLine { chars: 120 }]);
        assert_eq!(kinds("usa snake_case para las funciones"), naming(Case::Snake, "functions"));
        assert_eq!(kinds("PascalCase para las clases"), naming(Case::Pascal, "types"));
        assert_eq!(kinds("sin rayas y m\u{e1}ximo 100 caracteres por l\u{ed}nea"), vec![em_rule(), Kind::MaxLine { chars: 100 }]);
        assert_eq!(kinds("\u{a1}Sin rayas, por favor!"), vec![em_rule()]);
        assert_eq!(kinds("usa tablas para los datos"), vec![Kind::Guidance { domain: String::new() }], "a table is not a tab");
        stays_guidance("mant\u{e9}n las funciones cortas");
    }

    #[test]
    fn french_forms_parse_to_their_rules() {
        assert_eq!(kinds("pas de tirets cadratins"), vec![em_rule()]);
        assert_eq!(kinds("Supprime les tirets cadratins."), vec![em_rule()]);
        assert_eq!(kinds("supprime les guillemets typographiques").len(), 4);
        assert_eq!(kinds("n\u{2019}utilise pas d'espaces ins\u{e9}cables"), vec![Kind::Replace { from: "\u{a0}".into(), to: " ".into(), what: "non-breaking space (U+00A0)".into() }]);
        assert_eq!(kinds("utilise \"utiliser\" au lieu de \"employer\""), vocab("employer", "utiliser"));
        assert_eq!(kinds("\u{e9}cris 'courriel' plut\u{f4}t que 'email'"), vocab("email", "courriel"));
        assert_eq!(kinds("remplace les tirets cadratins par des virgules"), vec![em_rule()]);
        assert_eq!(kinds("ne dis jamais \"en fait\""), ban("en fait"));
        assert_eq!(kinds("n'utilise jamais le mot 'synergie'"), ban("synergie"));
        assert_eq!(kinds("indentation de 4 espaces"), vec![Kind::Indent { tabs: false, width: 4 }]);
        assert_eq!(kinds("retrait avec des tabulations"), vec![Kind::Indent { tabs: true, width: 0 }]);
        assert_eq!(kinds("longueur de ligne maximale de 100 caract\u{e8}res"), vec![Kind::MaxLine { chars: 100 }]);
        assert_eq!(kinds("utilise le camelCase pour les variables"), naming(Case::Camel, "variables"));
        assert_eq!(kinds("snake_case pour les fonctions"), naming(Case::Snake, "functions"));
        assert_eq!(kinds("pas de tirets cadratins et indentation de 2 espaces"), vec![em_rule(), Kind::Indent { tabs: false, width: 2 }]);
        stays_guidance("garde les fonctions courtes");
    }

    #[test]
    fn german_forms_parse_to_their_rules() {
        let both = kinds("keine Gedankenstriche");
        assert_eq!(both.len(), 2, "a Gedankenstrich names both long dashes: {both:?}");
        assert!(both.iter().all(|k| matches!(k, Kind::Replace { from, .. } if from == EM || from == "\u{2013}")));
        assert_eq!(kinds("Entferne alle Geviertstriche."), vec![em_rule()]);
        assert_eq!(kinds("keine Geviertstriche mehr, bitte"), vec![em_rule()]);
        assert_eq!(kinds("verwende 'nutzen' statt 'verwenden'"), vocab("verwenden", "nutzen"));
        assert_eq!(kinds("ersetze Geviertstriche durch Kommas"), vec![em_rule()]);
        assert_eq!(kinds("sag niemals 'eigentlich'"), ban("eigentlich"));
        assert_eq!(kinds("schreibe nie quasi"), ban("quasi"));
        assert_eq!(kinds("Einr\u{fc}ckung mit 4 Leerzeichen"), vec![Kind::Indent { tabs: false, width: 4 }]);
        assert_eq!(kinds("Tabs statt Leerzeichen"), vec![Kind::Indent { tabs: true, width: 0 }]);
        assert_eq!(kinds("maximale Zeilenl\u{e4}nge 100 Zeichen"), vec![Kind::MaxLine { chars: 100 }]);
        assert_eq!(kinds("Zeilen h\u{f6}chstens 80 Zeichen"), vec![Kind::MaxLine { chars: 80 }]);
        assert_eq!(kinds("snake_case f\u{fc}r Funktionen"), naming(Case::Snake, "functions"));
        assert_eq!(kinds("PascalCase f\u{fc}r Klassen"), naming(Case::Pascal, "types"));
        assert_eq!(kinds("kebab-case f\u{fc}r Dateinamen"), naming(Case::Kebab, "files"));
        assert_eq!(kinds("keine Geviertstriche und Einr\u{fc}ckung mit Tabs"), vec![em_rule(), Kind::Indent { tabs: true, width: 0 }]);
        stays_guidance("Halte Funktionen kurz");
    }

    #[test]
    fn portuguese_forms_parse_to_their_rules() {
        assert_eq!(kinds("sem travess\u{f5}es"), vec![em_rule()]);
        assert_eq!(kinds("Remova os travess\u{f5}es."), vec![em_rule()]);
        assert_eq!(kinds("n\u{e3}o use aspas curvas").len(), 4);
        assert_eq!(kinds("use 'usar' em vez de 'utilizar'"), vocab("utilizar", "usar"));
        assert_eq!(kinds("escreva 'arquivo' ao inv\u{e9}s de 'ficheiro'"), vocab("ficheiro", "arquivo"));
        assert_eq!(kinds("substitua travess\u{f5}es por v\u{ed}rgulas"), vec![em_rule()]);
        assert_eq!(kinds("nunca diga 'basicamente'"), ban("basicamente"));
        assert_eq!(kinds("n\u{e3}o use a palavra 'sinergia'"), ban("sinergia"));
        assert_eq!(kinds("recuo de 2 espa\u{e7}os"), vec![Kind::Indent { tabs: false, width: 2 }]);
        assert_eq!(kinds("use tabula\u{e7}\u{f5}es"), vec![Kind::Indent { tabs: true, width: 0 }]);
        assert_eq!(kinds("comprimento m\u{e1}ximo de linha 100"), vec![Kind::MaxLine { chars: 100 }]);
        assert_eq!(kinds("snake_case para fun\u{e7}\u{f5}es"), naming(Case::Snake, "functions"));
        assert_eq!(kinds("sem travess\u{f5}es e no m\u{e1}ximo 100 caracteres por linha"), vec![em_rule(), Kind::MaxLine { chars: 100 }]);
        stays_guidance("mantenha as fun\u{e7}\u{f5}es curtas");
    }

    #[test]
    fn other_languages_keep_the_users_words_and_parse_deterministically() {
        let msg = "Sin rayas. Ne dis jamais \"en fait\". Einr\u{fc}ckung mit 4 Leerzeichen. Comprimento m\u{e1}ximo de linha 100. Halte Funktionen kurz";
        let a = with_ids(&[], parse(msg, &Scope::global(), None));
        let b = with_ids(&[], parse(msg, &Scope::global(), None));
        assert_eq!(a, b, "the same message gives the same rules");
        assert_eq!(a.iter().map(Rule::kind_name).collect::<Vec<_>>(), ["replace", "ban", "indent", "max_line", "guidance"]);
        assert_eq!(a.iter().map(|r| r.said.as_str()).collect::<Vec<_>>(), ["Sin rayas.", "Ne dis jamais \"en fait\".", "Einr\u{fc}ckung mit 4 Leerzeichen.", "Comprimento m\u{e1}ximo de linha 100.", "Halte Funktionen kurz"]);
        // The rules hold the same way whatever language said them.
        let out = apply("Va r\u{e1}pido \u{2014} en cach\u{e9}.", &held("sin rayas"), None, None);
        assert_eq!(out.text, "Va r\u{e1}pido, en cach\u{e9}.");
    }

    /// The one rule `msg` gives, with its scope's kinds and folders.
    fn scoped(msg: &str) -> (Vec<Kind>, Vec<String>, Vec<String>) {
        let r = parse(msg, &Scope::global(), Some("proj"));
        let first = r.first().map(|x| (x.scope.kinds.clone(), x.scope.folders.clone())).unwrap_or_default();
        (r.into_iter().map(|x| x.kind).collect(), first.0, first.1)
    }

    #[test]
    fn spanish_scope_clauses_narrow_rules() {
        assert_eq!(scoped("para archivos .rs, sangr\u{ed}a de 4 espacios"), (vec![Kind::Indent { tabs: false, width: 4 }], vec!["rs".into()], vec![]));
        assert_eq!(scoped("para archivos .rs indenta con 4 espacios"), (vec![Kind::Indent { tabs: false, width: 4 }], vec!["rs".into()], vec![]));
        assert_eq!(scoped("sin rayas en docs/"), (vec![em_rule()], vec![], vec!["docs".into()]));
        assert_eq!(scoped("sin rayas en la carpeta docs"), (vec![em_rule()], vec![], vec!["docs".into()]));
        let r = parse("sin rayas en este proyecto", &Scope::global(), Some("proj"));
        assert!(r.first().is_some_and(|x| x.scope.project.as_deref() == Some("proj")), "{r:?}");
    }

    #[test]
    fn french_scope_clauses_narrow_rules() {
        assert_eq!(scoped("pour les fichiers .md, pas de tirets cadratins"), (vec![em_rule()], vec!["md".into()], vec![]));
        assert_eq!(scoped("pas de tirets cadratins dans docs/"), (vec![em_rule()], vec![], vec!["docs".into()]));
        assert_eq!(scoped("dans le dossier src/ui, indentation de 2 espaces"), (vec![Kind::Indent { tabs: false, width: 2 }], vec![], vec!["src/ui".into()]));
    }

    #[test]
    fn german_scope_clauses_narrow_rules() {
        assert_eq!(scoped("f\u{fc}r .kicad_pcb-Dateien keine Geviertstriche"), (vec![em_rule()], vec!["kicad_pcb".into()], vec![]));
        assert_eq!(scoped("keine Geviertstriche f\u{fc}r .kicad_pcb-Dateien"), (vec![em_rule()], vec!["kicad_pcb".into()], vec![]));
        assert_eq!(scoped("keine Geviertstriche im Ordner docs/"), (vec![em_rule()], vec![], vec!["docs".into()]));
        assert_eq!(scoped("im Verzeichnis docs, Einr\u{fc}ckung mit Tabs"), (vec![Kind::Indent { tabs: true, width: 0 }], vec![], vec!["docs".into()]));
    }

    #[test]
    fn portuguese_scope_clauses_narrow_rules() {
        assert_eq!(scoped("em arquivos .py, recuo de 4 espa\u{e7}os"), (vec![Kind::Indent { tabs: false, width: 4 }], vec!["py".into()], vec![]));
        assert_eq!(scoped("sem travess\u{f5}es na pasta docs/"), (vec![em_rule()], vec![], vec!["docs".into()]));
        assert_eq!(scoped("nos arquivos .md prefira 'arquivo'"), (vec![Kind::Want { term: "arquivo".into(), style: String::new(), strength: 50 }], vec!["md".into()], vec![]));
    }

    #[test]
    fn english_statements_parse_byte_identically() {
        let snap = english_snapshot();
        assert_eq!(nmhash::sha256_bytes(snap.as_bytes()), ENGLISH_SNAPSHOT_SHA, "English parsing changed:\n{snap}");
    }

    fn want_term(t: &str, strength: u64) -> Vec<Kind> {
        vec![Kind::Want { term: t.into(), style: String::new(), strength }]
    }
    fn want_style(st: &str, strength: u64) -> Vec<Kind> {
        vec![Kind::Want { term: String::new(), style: st.into(), strength }]
    }

    #[test]
    fn english_want_forms_parse_to_wants() {
        assert_eq!(kinds("prefer 'colour'"), want_term("colour", 50));
        assert_eq!(kinds("Always use 'Result'."), want_term("Result", 100));
        assert_eq!(kinds("use Result"), want_term("Result", 50));
        assert_eq!(kinds("favour \"early return\""), want_term("early return", 50));
        assert_eq!(kinds("favor numbered lists"), want_style("numbered lists", 50));
        assert_eq!(kinds("I like concise replies"), want_style("concise", 50));
        assert_eq!(kinds("I'd like plain English, please"), want_style("plain language", 50));
        assert_eq!(kinds("more concise"), want_style("concise", 50));
        assert_eq!(kinds("be terse"), want_style("terse", 50));
        assert_eq!(kinds("strongly prefer the active voice"), want_style("active voice", 100));
        assert_eq!(kinds("I slightly prefer 'utilise'"), want_term("utilise", 25));
        assert_eq!(kinds("a bit more formal"), want_style("formal", 25));
        assert_eq!(kinds("no em dashes and prefer concise"), vec![em_rule(), want_style("concise", 50).remove(0)]);
        let r = parse("prefer 'colour' in .md files", &Scope::global(), None);
        assert!(matches!(r.as_slice(), [Rule { kind: Kind::Want { term, .. }, scope, .. }] if term == "colour" && scope.kinds == ["md"]), "{r:?}");
        // Objects that are not a known style, one plain word or a whole quoted term stay guidance.
        stays_guidance("prefer elegant code");
        stays_guidance("use descriptive names");
        stays_guidance("I want you to write tests first");
        stays_guidance("prefer not 'x'");
        stays_guidance("use spaces");
        stays_guidance("be careful");
        stays_guidance("like 'x'");
    }

    #[test]
    fn spanish_want_forms_parse_to_wants() {
        assert_eq!(kinds("prefiero respuestas concisas"), want_style("concise", 50));
        assert_eq!(kinds("usa siempre 'Resultado'"), want_term("Resultado", 100));
        assert_eq!(kinds("me gustan las listas numeradas"), want_style("numbered lists", 50));
        assert_eq!(kinds("prefiero ligeramente 'vale'"), want_term("vale", 25));
        assert_eq!(kinds("\u{a1}S\u{e9} formal, por favor!"), want_style("formal", 50));
        assert_eq!(kinds("m\u{e1}s vi\u{f1}etas"), want_style("bulleted lists", 50));
        stays_guidance("prefiero funciones cortas y claras");
    }

    #[test]
    fn french_want_forms_parse_to_wants() {
        assert_eq!(kinds("je pr\u{e9}f\u{e8}re des r\u{e9}ponses concises"), want_style("concise", 50));
        assert_eq!(kinds("utilise toujours 'courriel'"), want_term("courriel", 100));
        assert_eq!(kinds("privil\u{e9}gie la voix active"), want_style("active voice", 50));
        assert_eq!(kinds("j\u{2019}aime le style formel"), want_style("formal", 50));
        assert_eq!(kinds("pr\u{e9}f\u{e8}re un peu 'donc'"), want_term("donc", 25));
        stays_guidance("pr\u{e9}f\u{e8}re le code lisible");
    }

    #[test]
    fn german_want_forms_parse_to_wants() {
        assert_eq!(kinds("ich bevorzuge kurze S\u{e4}tze"), want_style("short sentences", 50));
        assert_eq!(kinds("verwende immer 'Ergebnis'"), want_term("Ergebnis", 100));
        assert_eq!(kinds("sei knapp"), want_style("concise", 50));
        assert_eq!(kinds("mehr Stichpunkte"), want_style("bulleted lists", 50));
        assert_eq!(kinds("bevorzuge etwas 'also'"), want_term("also", 25));
        stays_guidance("bevorzuge kurze Funktionen mit klaren Namen");
    }

    #[test]
    fn portuguese_want_forms_parse_to_wants() {
        assert_eq!(kinds("prefiro respostas concisas"), want_style("concise", 50));
        assert_eq!(kinds("use sempre 'resultado'"), want_term("resultado", 100));
        assert_eq!(kinds("gosto de voz ativa"), want_style("active voice", 50));
        assert_eq!(kinds("seja formal"), want_style("formal", 50));
        assert_eq!(kinds("prefira um pouco 'arquivo'"), want_term("arquivo", 25));
        stays_guidance("prefiro c\u{f3}digo limpo e simples");
    }

    #[test]
    fn a_want_keeps_its_strength_through_values_and_reads_as_its_label() {
        assert_eq!((strength_text(50), strength_text(100), strength_text(25), strength_text(200), strength_text(10)), ("0.5".into(), "1".into(), "0.25".into(), "2".into(), "0.1".into()));
        assert_eq!((parse_strength("0.5"), parse_strength(" 1.25 "), parse_strength("2")), (Ok(50), Ok(125), Ok(200)));
        assert!(parse_strength("2.5").is_err() && parse_strength("0.05").is_err() && parse_strength("much").is_err());
        let rules = with_ids(&[], parse("strongly prefer 'colour'. be concise. never say 'leverage'", &Scope::global(), None));
        assert_eq!(rules_of(&settings_of("stated-w", &rules)).remove("stated-w"), Some(rules.clone()));
        assert_eq!(rules.iter().map(Rule::label).collect::<Vec<_>>(), ["want", "want", "avoid"]);
        assert_eq!(rules.iter().map(Rule::kind_name).collect::<Vec<_>>(), ["want", "want", "ban"], "the stored kind of an avoided word stays ban");
        assert_eq!(rules.iter().map(|r| r.enforcement().name()).collect::<Vec<_>>(), ["biased", "guidance", "checked"]);
        assert_eq!(rules.first().and_then(Rule::strength), Some(100));
        let v = rules.first().map(Rule::view).expect("view");
        assert!(matches!(v.get("strength"), Ok(Value::Number(x)) if x == "1"));
        assert!(matches!(v.get("label"), Ok(Value::String(x)) if x == "want"));
        // A stored strength out of range is brought into range; a missing one is the default.
        let odd = nmjson::parse(br#"{"id":"r1","kind":"want","said":"x","term":"x"}"#).expect("json");
        assert_eq!(Rule::from_value(&odd).strength(), Some(STRENGTH_DEFAULT));
        let both = nmjson::parse(br#"{"id":"r1","kind":"want","said":"x","term":"x","style":"concise"}"#).expect("json");
        assert!(matches!(Rule::from_value(&both).kind, Kind::Guidance { .. }), "a want naming both a term and a style is not read as one");
    }

    #[test]
    fn wants_reach_the_block_the_writer_and_apply_as_information() {
        let mut rules = held("prefer 'colour'. be concise");
        rules.held.extend(held("for .md files, prefer 'utilise'").held);
        assert_eq!(rules.wanted(), vec![("colour".to_owned(), 50)], "the Writer takes global term wants only");
        let b = block(&rules, Some("proj"), &["src/main.rs".to_owned()]);
        assert!(b.iter().any(|l| l.starts_with("- prefer: \"colour\" [want, strength 0.5")), "{b:?}");
        assert!(b.iter().any(|l| l.starts_with("- prefer: concise [want, strength 0.5")), "{b:?}");
        assert!(!b.iter().any(|l| l.contains("utilise")), "a scoped want reaches the block only where it applies: {b:?}");
        assert!(block(&rules, Some("proj"), &["docs/a.md".to_owned()]).iter().any(|l| l.contains("prefer: \"utilise\"") && l.contains(".md files")));
        let out = apply("The color is fine.", &rules, None, Some("docs/a.md"));
        assert_eq!(out.missed.iter().map(|m| (m.rule.as_str(), m.term.as_str())).collect::<Vec<_>>(), [("r1", "colour"), ("r1", "utilise")]);
        assert!(out.violations.is_empty() && !out.changed(), "a missed want is never a violation and never rewrites");
        let v = out.to_value();
        assert_eq!(v.get("wants_missed").and_then(Value::array).map_or(0, <[Value]>::len), 2);
        let met = apply("Colour, utilise.", &rules, None, Some("docs/a.md"));
        assert!(met.missed.is_empty(), "{:?}", met.missed);
        assert!(apply("x", &rules, None, Some("src/a.rs")).missed.iter().all(|m| m.term == "colour"), "a want out of scope is not noted");
    }

    #[test]
    fn the_section_check_passes() {
        check().expect("check");
        assert_eq!(id_from_name("My Writing Rules!"), "stated-my-writing-rules");
        assert_eq!(id_from_name("***"), "stated-preferences");
    }
}
