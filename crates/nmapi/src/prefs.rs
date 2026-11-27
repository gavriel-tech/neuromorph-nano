//! The serve-time preference reader (roadmap 2026-11-23, Phase 2 item 2, 2026-11-24). The
//! preference cartridge of the project (`nmlearn::pref_fit`, layer 4 of the stacking order, applied
//! after every other stack) holds how likely edits in each circumstance (file kind, top folder) are
//! to end well. The library hands its `pref.*` values here when serving starts and again whenever a
//! stack is switched (a promotion or a rollback); each `nm_context_pack` reply then carries one
//! caution line per cited file whose seen circumstances have mostly ended badly
//! (`nmlearn::pref_fit::guidance`). With no preference cartridge in use, nothing is added.
//!
//! The same values carry the user's stated rules (`pref.stated.*`, `nmlearn::stated`): each reply
//! then carries a "preferences in force" block of the rules whose scope fits the project and the
//! cited files (`stated_block`), and the text the Nano writes itself (the Writer's notes,
//! summaries and outlines) gets the enforced transforms (`own_text`); the Writer's decoding masks
//! the tokens that would spell a banned mark or word (`banned`) and pushes, by a bounded logit bias,
//! toward the terms the user wants (`wanted`, `WriterBias`); the mask wins over the bias. Cited
//! evidence is never altered.
use nmlearn::pref_fit::PrefModel;
use nmlearn::stated::InForce;
use nmtype::Packet;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// The preference model in use (`None`: no preference cartridge served).
static SERVED: RwLock<Option<PrefModel>> = RwLock::new(None);
/// Bumped on every change of the model in use, so cached replies are not reused across it.
static GENERATION: AtomicU64 = AtomicU64::new(0);
/// The stated rules of the plugged preference cartridges (empty: none).
static STATED: RwLock<InForce> = RwLock::new(InForce { held: Vec::new() });

/// Seats the preference cartridges' values (`pref.global`, `pref.key.<label>`, and stated rules
/// under `pref.stated.*`); values with no signals clear the learned model, values with no stated
/// rule clear the stated rules.
pub fn set_served(settings: &BTreeMap<String, String>) {
    let m = PrefModel::from_settings(settings);
    let next = (m.global.1 > 0).then_some(m);
    if let Ok(mut g) = SERVED.write()
        && *g != next
    {
        *g = next;
        GENERATION.fetch_add(1, Ordering::Relaxed);
    }
    let stated = InForce::from_settings(settings);
    if let Ok(mut g) = STATED.write()
        && *g != stated
    {
        *g = stated;
        GENERATION.fetch_add(1, Ordering::Relaxed);
    }
}

/// The generation of the model in use.
#[must_use]
pub fn generation() -> u64 {
    GENERATION.load(Ordering::Relaxed)
}

/// Whether a preference cartridge is in use.
#[must_use]
pub fn served() -> bool {
    SERVED.read().is_ok_and(|g| g.is_some())
}

/// The cited file of an evidence line (`path:first-last ...`).
fn path_of(line: &str) -> Option<&str> {
    let p = line.split_once(':').map_or(line, |(p, _)| p).trim();
    (!p.is_empty() && !p.contains(' ')).then_some(p)
}

/// The caution lines for the files `packets` cite, in order.
#[must_use]
pub fn guidance(packets: &[Packet]) -> Vec<String> {
    let Ok(g) = SERVED.read() else { return Vec::new() };
    let Some(m) = g.as_ref() else { return Vec::new() };
    let paths: Vec<String> = packets.iter().flat_map(|p| p.get_evidence().iter().filter_map(|e| path_of(e)).map(str::to_owned)).collect();
    nmlearn::pref_fit::guidance(m, &paths)
}

/// The stated rules in use (a copy).
#[must_use]
pub fn stated() -> InForce {
    STATED.read().map(|g| g.clone()).unwrap_or_default()
}

/// The "preferences in force" lines for a reply citing `packets` in `project` (empty when no
/// stated rule fits).
#[must_use]
pub fn stated_block(packets: &[Packet], project: Option<&str>) -> Vec<String> {
    let Ok(g) = STATED.read() else { return Vec::new() };
    if g.is_empty() {
        return Vec::new();
    }
    let paths: Vec<String> = packets.iter().flat_map(|p| p.get_evidence().iter().filter_map(|e| path_of(e)).map(str::to_owned)).collect();
    nmlearn::stated::block(&g, project, &paths)
}

/// Text the Nano writes itself, with the enforced stated transforms applied (for the piece at
/// `path` when known). Unchanged when no stated rule is in use.
#[must_use]
pub fn own_text(text: &str, path: Option<&str>) -> String {
    match STATED.read() {
        Ok(g) if !g.is_empty() => nmlearn::stated::own_text(text, &g, path),
        _ => text.to_owned(),
    }
}

/// The marks and whole words the Writer's decoding masks (`InForce::banned`).
#[must_use]
pub fn banned() -> (Vec<String>, Vec<String>) {
    STATED.read().map(|g| g.banned()).unwrap_or_default()
}

/// The token ids of the Writer's end of text and end of turn; never masked, so a masked decode
/// still ends.
pub const WRITER_END_IDS: [usize; 2] = [151_643, 151_645];

/// The token ids a Writer decode never takes under the marks and words `banned` gives: every id
/// below `vocab` whose text (`decode`) holds a banned mark (an em dash, say), or whose text is a
/// banned or replaced word on its own (trimmed, any letter case). A word the tokenizer spells over
/// several tokens is not caught here; the enforced transforms (`own_text`) take it out of the
/// result. The ids in `ends` are never masked. Empty when nothing is banned.
#[must_use]
pub fn mask_of(marks: &[String], words: &[String], vocab: usize, decode: &dyn Fn(i32) -> String, ends: &[usize]) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    if marks.is_empty() && words.is_empty() {
        return out;
    }
    for id in 0..vocab {
        if ends.contains(&id) {
            continue;
        }
        let Ok(t) = i32::try_from(id) else { break };
        let text = decode(t);
        let word = text.trim().to_lowercase();
        if marks.iter().any(|m| text.contains(m.as_str())) || !word.is_empty() && words.contains(&word) {
            out.insert(id);
        }
    }
    out
}

/// The Writer's token mask for the stated preferences in use (`mask_of` over `banned`), with the
/// generation it was built at. Every decode of the Writer that writes the Nano's own text (notes,
/// summaries, file and module notes, the outline, likely names) takes its tokens through one of
/// these and calls `refresh` before each result, so a preference stated while the Writer is
/// working holds from its next result on.
#[derive(Clone, Debug, Default)]
pub struct WriterMask {
    generation: u64,
    ids: BTreeSet<usize>,
}

impl WriterMask {
    /// The mask for the rules in use now.
    #[must_use]
    pub fn build(vocab: usize, decode: &dyn Fn(i32) -> String) -> Self {
        let generation = generation();
        let (marks, words) = banned();
        Self { generation, ids: mask_of(&marks, &words, vocab, decode, &WRITER_END_IDS) }
    }
    /// Builds the mask again when the rules in use changed since it was built; whether it did.
    pub fn refresh(&mut self, vocab: usize, decode: &dyn Fn(i32) -> String) -> bool {
        if self.generation == generation() {
            return false;
        }
        *self = Self::build(vocab, decode);
        true
    }
    /// The masked token ids.
    #[must_use]
    pub const fn ids(&self) -> &BTreeSet<usize> {
        &self.ids
    }
}

/// The terms the Writer's decoding pushes toward, with their strengths in hundredths
/// (`InForce::wanted`).
#[must_use]
pub fn wanted() -> Vec<(String, u64)> {
    STATED.read().map(|g| g.wanted()).unwrap_or_default()
}

/// Logits one unit of strength adds to a wanted term's next token (strength 0.5 adds 0.75).
pub const BIAS_UNIT: f32 = 1.5;
/// The most bias one token takes, summed over every want (strength 2.0 reaches it).
pub const BIAS_CAP: f32 = 3.0;
/// Tokens back within which a term already written gets no new start bias, so a push never
/// repeats a term over and over.
pub const BIAS_REPEAT_WINDOW: usize = 48;

/// One way the tokenizer spells a wanted term: its token ids, and whether it starts with a space
/// (the spelling after a word) or not (the spelling at the start or after whitespace).
#[derive(Clone, Debug, PartialEq)]
struct Spelling {
    ids: Vec<i32>,
    spaced: bool,
}

/// One wanted term: its spellings and the bias its strength gives.
#[derive(Clone, Debug, PartialEq)]
struct Want {
    spellings: Vec<Spelling>,
    bias: f32,
}

/// The Writer's push toward the wanted terms of the stated preferences in use, with the
/// generation it was built at. It sits beside `WriterMask` on the same refresh: every decode of
/// the Writer takes both, and the mask wins (a masked id is never biased and never picked).
///
/// A term's first token gets `strength * BIAS_UNIT` (at most `BIAS_CAP` per token, summed over
/// wants) where the term is a plausible continuation: its spelling with a leading space after a
/// token that does not end in whitespace, its bare spelling at the start or after whitespace or an
/// opening mark, and not while the term was written within the last `BIAS_REPEAT_WINDOW` tokens.
/// Once the decode has written a term's first tokens, its next token gets the same bias, so a
/// started term is followed through. The bias is additive and bounded: a token the model holds
/// far below its best is never taken by it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WriterBias {
    generation: u64,
    wants: Vec<Want>,
}

/// Whether `hay` ends with `tail`.
fn ends_with(hay: &[i32], tail: &[i32]) -> bool {
    !tail.is_empty() && hay.len() >= tail.len() && hay.get(hay.len() - tail.len()..) == Some(tail)
}

/// Whether `hay` holds `needle` as a run.
fn holds(hay: &[i32], needle: &[i32]) -> bool {
    !needle.is_empty() && hay.windows(needle.len()).any(|w| w == needle)
}

impl WriterBias {
    /// The bias for `wants` (term, strength in hundredths) as `encode` spells them, at
    /// `generation`. Each term is spelled as written and with its first letter in upper case, each
    /// with and without a leading space; a spelling whose first token is the space alone is left
    /// out (biasing a bare space would push every word).
    #[must_use]
    pub fn of(wants: &[(String, u64)], encode: &dyn Fn(&str) -> Vec<i32>, generation: u64) -> Self {
        let space = encode(" ");
        let lone_space = if space.len() == 1 { space.first().copied() } else { None };
        let mut out = Vec::new();
        for (term, strength) in wants {
            let mut forms = vec![term.clone()];
            let mut c = term.chars();
            if let Some(f) = c.next().filter(|f| f.is_lowercase()) {
                forms.push(f.to_uppercase().collect::<String>() + c.as_str());
            }
            let mut spellings: Vec<Spelling> = Vec::new();
            for f in &forms {
                for (text, spaced) in [(format!(" {f}"), true), (f.clone(), false)] {
                    let ids = encode(&text);
                    if ids.is_empty() || spaced && ids.first().copied() == lone_space {
                        continue;
                    }
                    if !spellings.iter().any(|s| s.ids == ids) {
                        spellings.push(Spelling { ids, spaced });
                    }
                }
            }
            let bias = (*strength as f32 / 100.0 * BIAS_UNIT).min(BIAS_CAP);
            if !spellings.is_empty() && bias > 0.0 {
                out.push(Want { spellings, bias });
            }
        }
        Self { generation, wants: out }
    }
    /// The bias for the wants in use now.
    #[must_use]
    pub fn build(encode: &dyn Fn(&str) -> Vec<i32>) -> Self {
        let generation = generation();
        Self::of(&wanted(), encode, generation)
    }
    /// Builds the bias again when the rules in use changed since it was built; whether it did.
    pub fn refresh(&mut self, encode: &dyn Fn(&str) -> Vec<i32>) -> bool {
        if self.generation == generation() {
            return false;
        }
        *self = Self::build(encode);
        true
    }
    /// Whether no term is wanted.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.wants.is_empty()
    }
    /// The additive logit bias by token id for the next token after `generated` (the tokens this
    /// decode wrote; `last_text` is the text of the last of them, empty at the start). No id in
    /// `mask` is ever biased: the mask wins.
    #[must_use]
    pub fn at(&self, generated: &[i32], last_text: &str, mask: &BTreeSet<usize>) -> BTreeMap<usize, f32> {
        let mut out: BTreeMap<usize, f32> = BTreeMap::new();
        let recent = generated.get(generated.len().saturating_sub(BIAS_REPEAT_WINDOW)..).unwrap_or_default();
        let after_word = last_text.chars().last().is_some_and(|c| !c.is_whitespace());
        let at_boundary = last_text.is_empty() || last_text.ends_with(|c: char| c.is_whitespace() || matches!(c, '(' | '[' | '{' | '"' | '\'' | '`'));
        for w in &self.wants {
            let mut add = |id: i32| {
                if let Ok(u) = usize::try_from(id)
                    && !mask.contains(&u)
                {
                    let e = out.entry(u).or_insert(0.0);
                    *e = (*e + w.bias).min(BIAS_CAP);
                }
            };
            let mut followed = false;
            for sp in &w.spellings {
                if let Some(k) = (1..sp.ids.len()).rev().find(|k| sp.ids.get(..*k).is_some_and(|head| ends_with(generated, head))) {
                    if let Some(next) = sp.ids.get(k) {
                        add(*next);
                    }
                    followed = true;
                }
            }
            if followed || w.spellings.iter().any(|sp| holds(recent, &sp.ids)) {
                continue;
            }
            for sp in &w.spellings {
                if (if sp.spaced { after_word } else { at_boundary })
                    && let Some(first) = sp.ids.first()
                {
                    add(*first);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_lines_give_their_file() {
        assert_eq!(path_of("crates/a/src/lib.rs:10-20"), Some("crates/a/src/lib.rs"));
        assert_eq!(path_of("docs/x.md"), Some("docs/x.md"));
        assert_eq!(path_of(""), None);
    }

    #[test]
    fn stated_rules_reach_the_block_and_the_nanos_own_text() {
        let rules = nmlearn::stated::with_ids(&[], nmlearn::stated::parse("no em dashes", &nmlearn::stated::Scope::global(), None));
        let mut vals = nmlearn::stated::settings_of("stated-t", &rules);
        vals.insert("pref.global".into(), "0/0".into());
        let before = generation();
        set_served(&vals);
        assert!(generation() > before, "a change of the stated rules is a new generation");
        assert!(!served(), "no learned signals: no learned model served");
        let p = Packet::new("chunk", "k", "x", nmtype::Provenance::new("a", "t", 1, 1)).evidence(vec!["src/a.rs:1-3 fn a()".into()]);
        let b = stated_block(std::slice::from_ref(&p), Some("proj"));
        assert!(b.len() == 2 && b.get(1).is_some_and(|l| l.contains("no em dashes")), "{b:?}");
        assert_eq!(own_text("Fast \u{2014} cached.", None), "Fast, cached.");
        assert_eq!(banned().0, vec!["\u{2014}".to_owned()]);
        // The Writer's mask follows the rules in use: built now it masks the em dash tokens, and
        // after the rules change it is built again and masks nothing.
        let vocab = ["a", " \u{2014}", "b", "\u{2014}x", "<end>"];
        let decode = |t: i32| usize::try_from(t).ok().and_then(|i| vocab.get(i)).map_or_else(String::new, |x| (*x).to_owned());
        let mut mask = WriterMask::build(vocab.len(), &decode);
        assert_eq!(mask.ids(), &BTreeSet::from([1, 3]));
        assert!(!mask.refresh(vocab.len(), &decode), "no change, no rebuild");
        // The Writer's bias follows the rules in use too, and a strength edit changes it.
        let wants = |strength: &str| {
            let mut rules = nmlearn::stated::with_ids(&[], nmlearn::stated::parse("prefer 'cache'", &nmlearn::stated::Scope::global(), None));
            if let Some(r) = rules.first_mut() {
                r.set_strength(nmlearn::stated::parse_strength(strength).expect("strength")).expect("a want");
            }
            nmlearn::stated::settings_of("stated-w", &rules)
        };
        let encode = |t: &str| match t {
            " cache" => vec![7],
            "cache" => vec![8],
            " Cache" => vec![9],
            "Cache" => vec![10],
            _ => Vec::new(),
        };
        set_served(&wants("0.5"));
        let mut bias = WriterBias::build(&encode);
        assert_eq!(wanted(), vec![("cache".to_owned(), 50)]);
        assert_eq!(bias.at(&[], "", &BTreeSet::new()).get(&8).copied(), Some(0.5 * BIAS_UNIT));
        assert!(!bias.refresh(&encode), "no change, no rebuild");
        set_served(&wants("1.0"));
        assert!(bias.refresh(&encode));
        assert_eq!(bias.at(&[], "", &BTreeSet::new()).get(&8).copied(), Some(BIAS_UNIT), "a strength edit changes the bias");
        set_served(&BTreeMap::new());
        assert!(mask.refresh(vocab.len(), &decode) && mask.ids().is_empty());
        assert!(bias.refresh(&encode) && bias.is_empty());
        assert!(stated_block(&[p], None).is_empty());
        assert_eq!(own_text("a \u{2014} b", None), "a \u{2014} b");
    }
    #[test]
    fn the_mask_takes_banned_marks_and_whole_words_and_never_the_end_ids() {
        let vocab = ["Hello", " leverage", "Leverage ", "leveraged", "\u{2014}", " ok", "<|endoftext|>"];
        let decode = |t: i32| usize::try_from(t).ok().and_then(|i| vocab.get(i)).map_or_else(String::new, |x| (*x).to_owned());
        let marks = vec!["\u{2014}".to_owned()];
        let words = vec!["leverage".to_owned()];
        assert_eq!(mask_of(&marks, &words, vocab.len(), &decode, &[]), BTreeSet::from([1, 2, 4]), "a whole word in any case, never a longer word");
        assert_eq!(mask_of(&marks, &words, vocab.len(), &decode, &[4]), BTreeSet::from([1, 2]), "an end id is never masked");
        assert!(mask_of(&[], &[], vocab.len(), &decode, &[]).is_empty(), "nothing banned, nothing masked");
    }

    /// A fake tokenizer: " colour" is one token, "colour" two ("col", "our"), " Colour" and
    /// "Colour" one each, " " one.
    fn fake_encode(t: &str) -> Vec<i32> {
        match t {
            " colour" => vec![1],
            "colour" => vec![2, 3],
            " Colour" => vec![4],
            "Colour" => vec![5],
            " " => vec![6],
            " grey" => vec![6, 11],
            "grey" => vec![11],
            " Grey" => vec![6, 12],
            "Grey" => vec![12],
            _ => Vec::new(),
        }
    }

    #[test]
    fn the_writer_bias_pushes_toward_wanted_terms_within_its_cap() {
        let none = BTreeSet::new();
        let b = WriterBias::of(&[("colour".to_owned(), 50)], &fake_encode, 0);
        // At the start: the bare spellings' first tokens.
        assert_eq!(b.at(&[], "", &none), BTreeMap::from([(2, 0.75), (5, 0.75)]));
        // After a word: the spaced spellings.
        assert_eq!(b.at(&[40], "the", &none), BTreeMap::from([(1, 0.75), (4, 0.75)]));
        // After whitespace: the bare spellings again.
        assert_eq!(b.at(&[40], "the ", &none), BTreeMap::from([(2, 0.75), (5, 0.75)]));
        // Started: the term is followed through to its next token, and nothing else is pushed.
        assert_eq!(b.at(&[40, 2], "col", &none), BTreeMap::from([(3, 0.75)]));
        // Written within the window: no new start.
        assert!(b.at(&[1, 40], "x", &none).is_empty(), "a term just written is not pushed again");
        assert!(!b.at(&[&[1][..], &[40; BIAS_REPEAT_WINDOW]].concat(), "x", &none).is_empty(), "past the window it is pushed again");
        // The mask wins: a masked id is never biased.
        assert_eq!(b.at(&[40], "the", &[1].into()), BTreeMap::from([(4, 0.75)]));
        // The cap: strength 2.0 gives BIAS_CAP, and two wants on one token sum to at most BIAS_CAP.
        let strong = WriterBias::of(&[("colour".to_owned(), 200)], &fake_encode, 0);
        assert_eq!(strong.at(&[], "", &none).get(&2).copied(), Some(BIAS_CAP));
        let twice = WriterBias::of(&[("colour".to_owned(), 150), ("colour".to_owned(), 150)], &fake_encode, 0);
        assert_eq!(twice.at(&[], "", &none).get(&2).copied(), Some(BIAS_CAP));
        assert!((0.0..=BIAS_CAP).contains(&(2.0 * BIAS_UNIT)), "the most strength reaches the cap and no further");
        // A spaced spelling that starts with the space token alone is left out.
        let grey = WriterBias::of(&[("grey".to_owned(), 50)], &fake_encode, 0);
        assert!(!grey.at(&[40], "the", &none).contains_key(&6), "a bare space is never pushed");
        // A strength edit changes the bias.
        let edited = WriterBias::of(&[("colour".to_owned(), 100)], &fake_encode, 0);
        assert_eq!(edited.at(&[], "", &none).get(&2).copied(), Some(BIAS_UNIT));
        assert!(WriterBias::of(&[], &fake_encode, 0).is_empty());
    }

    /// Every decode of the Writer in this crate and in the product binary goes through the masked
    /// decoder: the only token loop that takes Writer tokens is `writer_idle::decode_preemptible`,
    /// called with the session's `WriterMask` and `WriterBias`. A new decoder (a greedy or sampled loop, or the
    /// engine's write helpers) fails here until it takes the mask.
    #[test]
    fn every_writer_decode_takes_the_mask() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        // (file, call): calls that decode no Writer text. `learn.rs` checks that a one-layer
        // assembly with no head refuses greedy decoding; it never produces a token.
        let allowed: &[(&str, &str)] = &[("learn.rs", "nmengn::decode::greedy(")];
        let calls = ["decode::greedy(", "decode::greedy_until(", "decode::sample_until(", "speculate::greedy_until(", "roles::write(", "roles::write_chat(", "roles::write_sampled("];
        let mut files = Vec::new();
        let mut stack = vec![root.join("src"), root.join("../nmcore/src")];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).expect("source folder").flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    files.push(p);
                }
            }
        }
        assert!(files.len() > 20, "the sources were found");
        let mut found = Vec::new();
        for f in &files {
            let text = std::fs::read_to_string(f).expect("source");
            let name = f.file_name().and_then(|x| x.to_str()).unwrap_or_default();
            for line in text.lines().filter(|l| !l.trim_start().starts_with("//")) {
                for c in &calls {
                    // A call in code: not inside a string (an odd count of quotes before it).
                    let in_code = line.match_indices(c).any(|(i, _)| line.get(..i).is_some_and(|before| before.matches('"').count() % 2 == 0));
                    if in_code && !allowed.iter().any(|(af, ac)| *af == name && line.contains(ac)) {
                        found.push(format!("{name}: {c}"));
                    }
                }
            }
        }
        assert!(found.is_empty(), "Writer decodes without the stated-preference mask: {found:?}");
        let idle = std::fs::read_to_string(root.join("src/writer_idle.rs")).expect("writer_idle.rs");
        let decodes: Vec<&str> = idle.lines().filter(|l| l.contains("decode_preemptible(") && !l.trim_start().starts_with("//") && !l.contains("fn decode_preemptible")).collect();
        assert!(!decodes.is_empty() && decodes.iter().all(|l| l.contains("mask.ids()") && l.contains("&bias")), "every decode takes both the mask and the bias: {decodes:?}");
        assert!(idle.contains("mask.refresh("), "the mask is brought up to date before each result");
        assert!(idle.contains("bias.refresh("), "the bias is brought up to date before each result");
        assert!(idle.contains("bias.at(&tokens, &last, mask)"), "each step's bias is built under the mask");
    }
}
