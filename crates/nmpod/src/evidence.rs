//! The drones' evidence, calibrated: what every piece holds for each aspect of the request, in
//! independent families, each signal measured against a content-free pose.
//!
//! Families (scored per piece and per aspect of the request's drone, `intake::Request`):
//! - wording: BM25F (definition lines and the path weigh more inside the term count), BM25, and
//!   the aspect written exactly as asked (an identifier, a quantity, or two words or more);
//! - naming: the aspect's stems among the identifier parts the piece defines (on its definition
//!   lines), then among those it defines or uses in code (comments and notes left out: a word in
//!   a comment mentions a thing, it does not name it);
//! - meaning: the token-embedding affinity of the aspect's tokens (the late-interaction rule of
//!   `intake`), and, when the meaning lane runs, the cosine of the piece's embedding with the
//!   question's (one value per piece, read for every aspect).
//!
//! Structure and memory arrive with their own packages.
//!
//! Calibration, with no fitted weight: every aspect is posed again three times with each word
//! replaced by a word the material holds in as many pieces (chosen by a hash of the request, so a
//! run repeats). A signal's calibrated value at a piece is its posed score less the mean of its
//! three null scores, in units of the typical content-free difference (the median size of a null
//! pose's difference from the other two, where it fired), then the residual after a robust fit of
//! the content-free level on the piece's length, kind and place in its file (Tukey's resistant line
//! on the logarithm of its tokens, then the median of each kind and place cell, both taken over the
//! null poses). A piece the posed signal did not fire on holds no evidence for it. The band is the
//! largest residual a null pose reaches at the pieces the posed signal fired on. A piece is accepted
//! for an aspect (the drone's reflexive accept) when some signal stands above its band. Each accepted
//! finding is tagged by its epistemic class: tests, documents, defines, uses or mentions.
//!
//! The meaning lane's cosine is calibrated by the same rule against the cosines of content-free
//! questions with each piece (encoded once per material version), so a piece close to every
//! question holds no evidence for this one.
//!
//! Every number is derived from the request's own score distributions; the stated fallbacks are
//! the mean null difference where its median is zero, and one where no null pose fired.
use crate::intake::{Asks, BM25_B, BM25_K1, Drone, Form, Rarity, Record, Request, bm25_words};
use nmtype::Fault;
use std::collections::HashMap;

/// The signals, by name, in the order every table here holds them.
pub const SIGNALS: [&str; 7] = ["bm25f", "bm25", "exact", "structural", "ident", "affinity", "meaning"];
/// The signals read from the piece's own text (the first of `SIGNALS`); the last is the meaning
/// lane's cosine.
const RAW: usize = 6;
/// An evidence family: signals of one family read the same thing in different ways.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Wording,
    Naming,
    Meaning,
}
/// The family of each signal of `SIGNALS`.
pub const FAMILY: [Family; 7] = [Family::Wording, Family::Wording, Family::Wording, Family::Naming, Family::Naming, Family::Meaning, Family::Meaning];
/// Null poses per aspect.
pub const NULLS: usize = 3;

/// What an accepted finding is, epistemically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    Defines,
    Uses,
    Mentions,
    Tests,
    Documents,
}
impl Tag {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Defines => "defines",
            Self::Uses => "uses",
            Self::Mentions => "mentions",
            Self::Tests => "tests",
            Self::Documents => "documents",
        }
    }
}

/// The nuisance a piece's score may carry whatever the request: its length, the class of its
/// kind and whether it opens its file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Place {
    pub tokens: usize,
    pub class: Class,
    pub first: bool,
}
/// Kind classes for the robust fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// A definition (function, method, type, constant, module, macro).
    Code,
    /// A file's notes or imports.
    Head,
    /// A fixed-size window of code.
    Window,
    /// A piece of a document.
    Document,
}
impl Class {
    /// The class of a substrate chunk kind (`window`, a piece kind, a document block kind).
    #[must_use]
    pub fn of(kind: &str, document: bool) -> Self {
        if document {
            Self::Document
        } else {
            match kind {
                "window" => Self::Window,
                "notes" | "imports" => Self::Head,
                _ => Self::Code,
            }
        }
    }
}

/// One pose of one aspect: the words it scores, the stems it names, its exact form and its
/// tokens' unit rows (offsets into the request's query table) with their rarity.
#[derive(Clone, Debug, Default)]
struct Pose {
    bm25: Vec<(String, f32)>,
    stems: Vec<(String, f32)>,
    phrase: Vec<String>,
    queries: Vec<usize>,
    weights: Vec<f32>,
}

/// The calibrated evidence of every piece for one request.
#[derive(Clone, Debug, Default)]
pub struct Evidence {
    /// The aspects, as the request's drone wrote them.
    pub aspects: Vec<String>,
    /// `value[a][s][i]`: signal `s` for aspect `a` at piece `i`, calibrated and residual.
    pub value: Vec<Vec<Vec<f32>>>,
    /// `band[a][s]`: the content-free band of signal `s` for aspect `a`.
    pub band: Vec<Vec<f32>>,
    /// `surprise[a][s][i]`: how unlikely the value is under the content-free pose, in nats: minus
    /// the logarithm of the share of the null poses' values over every piece that reach it (0 where
    /// the posed signal did not fire). Signals and aspects are on one scale this way: a word's hit
    /// counts about its rarity, a meaning stands out as far as no content-free request reaches.
    pub surprise: Vec<Vec<Vec<f32>>>,
    /// The null words of each aspect's three poses (for the audit).
    pub nulls: Vec<Vec<String>>,
    /// Each aspect's stems, as the request's drone read them.
    pub stems: Vec<Vec<String>>,
    /// `share[a][i]`: the rarity-weighted share of aspect `a`'s stems piece `i` holds.
    pub share: Vec<Vec<f32>>,
    /// The parts of the aspects (the separate things each names and asks), in order: the aspect
    /// each belongs to, its text, and the rarity-weighted share of its stems each piece holds in
    /// code (defined or used).
    pub parts: Vec<(usize, String, Vec<f32>)>,
    /// `settle[k][i]`: the chance piece `i` settles part `k` (`settle`): a probability over the
    /// material, calibrated against the content-free poses.
    pub settle: Vec<Vec<f32>>,
    /// Per part, in nats, the evidence mass the material gives the part as posed and the mean mass
    /// it gives the part's content-free poses (the share no piece settles).
    pub settle_mass: Vec<(f32, f32)>,
    /// Per part and piece, for the audit: what the piece holds of the part by role (definition,
    /// code, anywhere) and its prior in nats.
    pub settle_why: Vec<Vec<([f32; 3], f32)>>,
    /// Per part, the content-free base rate of holding it: the mean share of the part a piece holds
    /// anywhere under the content-free poses, over every piece and pose.
    pub settle_base: Vec<f32>,
    /// With size classes (`sense_sized`): per part and class, the same rate over the class's units.
    pub settle_base_class: Vec<Vec<f32>>,
    /// Per part, how far the pieces holding it concentrate in files beyond what their number
    /// predicts if they were spread over the material at random (`concentration`): 1 when every
    /// pair of holders shares a file, 0 at random spread, below 0 when spread wider than random.
    /// `None` when fewer than two pieces hold the part (no pair to read). Then, per content-free
    /// pose, the same for the pose's words; and the part's holders and the files they lie in.
    pub concentration: Vec<Option<f32>>,
    pub concentration_null: Vec<Vec<Option<f32>>>,
    pub holders: Vec<(usize, usize)>,
    /// Per part, how much more of the request's other parts its holders hold than the material's
    /// pieces do on average (`associations`), in nats of joint value; then the same for each of
    /// its content-free poses. `None` for a part with no other part beside it or no holder.
    pub association: Vec<Option<f32>>,
    pub association_null: Vec<Vec<Option<f32>>>,
    /// Each unit's size class (`size_classes`); empty for pieces, which are read in one class.
    pub sizes: Vec<usize>,
    /// Relations of the request settled as facts of the code's structure: the relation (its
    /// place in the request's relations) and every piece whose own lines hold it, in source
    /// order. Its callers: the pieces whose code lines call a name the request writes as an
    /// identifier.
    pub facts: Vec<(usize, Vec<usize>)>,
}

impl Evidence {
    /// Pieces scored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.value.first().and_then(|a| a.first()).map_or(0, Vec::len)
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// How far signal `s` stands above its band for aspect `a` at piece `i` (negative: below).
    #[must_use]
    pub fn excess(&self, a: usize, s: usize, i: usize) -> f32 {
        let v = self.value.get(a).and_then(|x| x.get(s)).and_then(|x| x.get(i)).copied().unwrap_or(f32::NEG_INFINITY);
        let b = self.band.get(a).and_then(|x| x.get(s)).copied().unwrap_or(f32::INFINITY);
        v - b
    }
    /// The best signal's excess for aspect `a` at piece `i`, and that signal.
    #[must_use]
    pub fn best(&self, a: usize, i: usize) -> (f32, usize) {
        (0..SIGNALS.len()).map(|s| (self.excess(a, s, i), s)).fold((f32::NEG_INFINITY, 0), |m, x| if x.0 > m.0 { x } else { m })
    }
    /// The drone's reflexive accept: piece `i` holds aspect `a` (some signal above its band).
    #[must_use]
    pub fn accepted(&self, a: usize, i: usize) -> bool {
        self.best(a, i).0 > 0.0
    }
    /// The surprise of signal `s` for aspect `a` at piece `i`.
    #[must_use]
    pub fn surprise(&self, a: usize, s: usize, i: usize) -> f32 {
        self.surprise.get(a).and_then(|x| x.get(s)).and_then(|x| x.get(i)).copied().unwrap_or(0.0)
    }
    /// What piece `i` holds of aspect `a`: over the independent families, each family's best
    /// signal's surprise, added (signals of one family read the same evidence; families do not).
    #[must_use]
    pub fn aspect(&self, a: usize, i: usize) -> f32 {
        [Family::Wording, Family::Naming, Family::Meaning]
            .iter()
            .map(|f| FAMILY.iter().enumerate().filter(|(_, g)| *g == f).map(|(s, _)| self.surprise(a, s, i)).fold(0.0_f32, f32::max))
            .sum()
    }
    /// A piece's evidence: what it holds of every aspect, added.
    #[must_use]
    pub fn piece(&self, i: usize) -> f32 {
        (0..self.aspects.len()).map(|a| self.aspect(a, i)).sum()
    }
    /// Every piece's evidence.
    #[must_use]
    pub fn pieces(&self) -> Vec<f32> {
        (0..self.len()).map(|i| self.piece(i)).collect()
    }
    /// One signal's evidence at a piece: its surprise, added over the aspects.
    #[must_use]
    pub fn signal(&self, s: usize, i: usize) -> f32 {
        (0..self.aspects.len()).map(|a| self.surprise(a, s, i)).sum()
    }
    /// The independent families that accept piece `i` for some aspect (its quorum).
    #[must_use]
    pub fn families(&self, i: usize) -> usize {
        let mut f: Vec<Family> = Vec::new();
        for a in 0..self.aspects.len() {
            for (s, fam) in FAMILY.iter().enumerate() {
                if self.excess(a, s, i) > 0.0 && !f.contains(fam) {
                    f.push(*fam);
                }
            }
        }
        f.len()
    }
    /// Piece `i` holds aspect `a`: accepted for it, and holding at least half of its stems by
    /// rarity (the request drone's coverage rule), so a piece that holds an aspect's common words
    /// and not its rarest does not count as holding it.
    #[must_use]
    pub fn holds(&self, a: usize, i: usize) -> bool {
        self.accepted(a, i) && 2.0 * self.share.get(a).and_then(|x| x.get(i)).copied().unwrap_or(0.0) >= 1.0
    }
    /// The chance piece `i` settles part `k`: 0 unless the drones accept it for the part's aspect;
    /// else the share of the part's stems it holds by rarity, times the chance its evidence for
    /// the aspect is not content-free (one less the content-free tail its surprise stands at, e to
    /// the minus the surprise in nats). Nothing fitted.
    #[must_use]
    pub fn part_chance(&self, k: usize, i: usize) -> f32 {
        let Some((a, _, share)) = self.parts.get(k) else { return 0.0 };
        if !self.accepted(*a, i) {
            return 0.0;
        }
        share.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0) * (1.0 - (-self.aspect(*a, i)).exp())
    }
    /// Piece `i` holds part `k`: accepted for its aspect and holding at least half of the part's
    /// stems by rarity.
    #[must_use]
    pub fn part_holds(&self, k: usize, i: usize) -> bool {
        self.parts.get(k).is_some_and(|(a, _, share)| self.accepted(*a, i) && 2.0 * share.get(i).copied().unwrap_or(0.0) >= 1.0)
    }
    /// The aspects piece `i` holds.
    #[must_use]
    pub fn covers(&self, i: usize) -> Vec<usize> {
        (0..self.aspects.len()).filter(|a| self.holds(*a, i)).collect()
    }
    /// The chance piece `i` settles part `k` (`settle`).
    #[must_use]
    pub fn settles(&self, k: usize, i: usize) -> f32 {
        self.settle.get(k).and_then(|v| v.get(i)).copied().unwrap_or(0.0)
    }
    /// The chances piece `i` settles each part, added.
    #[must_use]
    pub fn settle_sum(&self, i: usize) -> f32 {
        (0..self.settle.len()).map(|k| self.settles(k, i)).sum()
    }
    /// The share of part `k` piece `i` holds anywhere in it (comments included), where it can
    /// settle the part (`settling`).
    #[must_use]
    pub fn held(&self, k: usize, i: usize) -> f32 {
        self.settle_why.get(k).and_then(|v| v.get(i)).map_or(0.0, |h| h.0[2].max(0.0))
    }
    /// How many of `cands` are plausible answers: those more likely to settle some part of the
    /// request than that no piece of the material settles it (the content-free share of the part's
    /// mass, `settle_mass`); at least one. The joint order's fusion constant: few plausible pieces
    /// make the top of each ranking count most.
    #[must_use]
    pub fn plausible(&self, cands: &[usize]) -> usize {
        let none: Vec<f32> = self.settle_mass.iter().map(|(posed, null)| 1.0 / (1.0 + (posed - null).exp())).collect();
        cands.iter().filter(|c| none.iter().enumerate().any(|(k, n)| self.settles(k, **c) > *n)).count().max(1)
    }
    /// Each part's base rate for the joint evidence: its content-free rate (`settle_base`); where
    /// the poses give none, the share of `cands` holding the part; 0 (the part left out) where no
    /// candidate holds it either.
    #[must_use]
    pub fn joint_base(&self, cands: &[usize]) -> Vec<f32> {
        (0..self.settle_why.len())
            .map(|k| {
                let b = self.settle_base.get(k).copied().unwrap_or(0.0);
                if b > 0.0 { b } else { cands.iter().filter(|c| self.held(k, **c) > 0.0).count() as f32 / cands.len().max(1) as f32 }
            })
            .collect()
    }
    /// Where the request's parts meet in piece `i`: over the parts, the log likelihood ratio of
    /// what it holds of each against the part's base rate, log((hold + b) / b), added. A missing
    /// rare part costs much, a missing common part almost nothing.
    #[must_use]
    pub fn joint(&self, i: usize, base: &[f32]) -> f32 {
        let class = self.sizes.get(i).copied();
        base.iter()
            .enumerate()
            .filter(|(_, b)| **b > 0.0)
            .map(|(k, b)| {
                // With size classes: the part's rate among units of this one's size, where it has one.
                let b = class.and_then(|c| self.settle_base_class.get(k).and_then(|v| v.get(c))).copied().filter(|x| *x > 0.0).unwrap_or(*b);
                let w = if frame_on() { self.content_weight(k) } else { 1.0 };
                w * ((self.held(k, i) + b) / b).ln()
            })
            .sum()
    }
    /// How far part `k` stands from a content-free part (`associations`): the share of its
    /// content-free poses whose holders hold the request's other parts less than its own holders
    /// do (a tie counts half). A part naming the request's subject stands above them all (1); the
    /// words of how a person asks stand among them. 1 where the test has nothing to read.
    #[must_use]
    pub fn content_weight(&self, k: usize) -> f32 {
        let Some(a) = self.association.get(k).copied().flatten() else { return 1.0 };
        let nulls: Vec<f32> = self.association_null.get(k).map_or_else(Vec::new, |v| v.iter().flatten().copied().collect());
        if nulls.is_empty() {
            return 1.0;
        }
        #[expect(clippy::cast_precision_loss, reason = "pose counts")]
        let share = nulls.iter().map(|x| if *x < a { 1.0 } else if (*x - a).abs() <= f32::EPSILON { 0.5 } else { 0.0 }).sum::<f32>() / nulls.len() as f32;
        share
    }
    /// `cands` in the joint order, each with its value: the reciprocal rank fusion (constant `k`)
    /// of the ranking by joint evidence (ties by the settle sum) and the ranking by the settle sum
    /// times its file's mass (the settle sums of the file's candidates, added; `file` names a
    /// candidate's file). Ties keep the order of `cands`.
    #[must_use]
    pub fn joint_order(&self, cands: &[usize], file: &dyn Fn(usize) -> usize, k: f32) -> Vec<(usize, f32)> {
        let base = self.joint_base(cands);
        let ss: Vec<f32> = cands.iter().map(|c| self.settle_sum(*c)).collect();
        let jv: Vec<f32> = cands.iter().map(|c| self.joint(*c, &base)).collect();
        let mut mass: HashMap<usize, f32> = HashMap::new();
        for (c, s) in cands.iter().zip(&ss) {
            *mass.entry(file(*c)).or_default() += s;
        }
        let fm: Vec<f32> = cands.iter().zip(&ss).map(|(c, s)| s * mass.get(&file(*c)).copied().unwrap_or(0.0)).collect();
        let keys: [Vec<(f32, f32)>; 2] = [jv.iter().zip(&ss).map(|(j, s)| (*j, *s)).collect(), fm.iter().map(|v| (*v, 0.0)).collect()];
        let mut score = vec![0.0_f32; cands.len()];
        for key in &keys {
            let mut idx: Vec<usize> = (0..cands.len()).collect();
            idx.sort_by(|a, b| {
                let (x, y) = (key.get(*a).copied().unwrap_or_default(), key.get(*b).copied().unwrap_or_default());
                y.0.total_cmp(&x.0).then(y.1.total_cmp(&x.1)).then(a.cmp(b))
            });
            for (r, i) in idx.into_iter().enumerate() {
                if let Some(x) = score.get_mut(i) {
                    *x += 1.0 / (k + r as f32);
                }
            }
        }
        let mut out: Vec<(usize, f32, usize)> = cands.iter().zip(&score).enumerate().map(|(p, (c, s))| (*c, *s, p)).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.2.cmp(&b.2)));
        out.into_iter().map(|(c, s, _)| (c, s)).collect()
    }
}

/// Whether the served plain path orders its packets by the joint order: on, unless `NANO_JOINT=off`
/// (for measurement).
#[must_use]
pub fn joint_on() -> bool {
    std::env::var("NANO_JOINT").map_or(true, |v| v != "off")
}

/// Whether the joint order weighs each part by how far it stands from a content-free part
/// (`Evidence::content_weight`): only with `NANO_FRAME=on` (for measurement).
#[must_use]
pub fn frame_on() -> bool {
    std::env::var("NANO_FRAME").is_ok_and(|v| v == "on")
}

/// Whether the frame test is read (`Evidence::concentration`, `Evidence::association`): with
/// `NANO_FRAME=on`, or `NANO_FRAME=measure` for the bench's dump alone.
#[must_use]
pub fn frame_measured() -> bool {
    std::env::var("NANO_FRAME").is_ok_and(|v| v == "on" || v == "measure")
}

/// What a piece holds of a part, by role, each the rarity-weighted share of the part's stems:
/// on its definition lines or in its file's path (what it defines: code items, a document's
/// headings, the thing its file is named for); on its code
/// lines (defined or used; a document's prose and a piece of notes only mention, so here they
/// hold what their headings hold); anywhere in it (comments and prose too). Each is at most the
/// next. A code name of the part (`names`: the forms the name is written in as one, with its
/// stems) names one thing: where a piece does not hold it whole (on its definition lines or in
/// its file's path for the first role, in it or its path for the others) it holds none of its
/// stems for it.
fn hold(rec: &Record, document: bool, stems: &[(String, f32)], names: &[(Vec<String>, Vec<String>)]) -> [f32; 3] {
    let total: f32 = stems.iter().map(|s| s.1).sum();
    if total <= 0.0 {
        return [0.0; 3];
    }
    let lacks = |text: &str| -> Vec<&String> {
        names.iter().filter(|(forms, _)| !forms.iter().any(|n| text.contains(n.as_str()) || rec.path_low.contains(n.as_str()))).flat_map(|(_, st)| st.iter()).collect()
    };
    // An item's attached comment describes it: its stems count as its definition lines' do.
    let defined_text = if rec.desc_low.is_empty() { rec.def_low.clone() } else { format!("{}
{}", rec.def_low, rec.desc_low) };
    let (missing, undefined) = (lacks(&rec.low), lacks(&defined_text));
    let share = |f: &dyn Fn(&str) -> bool| stems.iter().filter(|(s, _)| !missing.contains(&s) && f(s)).map(|s| s.1).sum::<f32>() / total;
    let d = share(&|s| !undefined.iter().any(|u| *u == s) && (rec.def_stems.contains(s) || rec.desc_stems.contains(s) || rec.path_stems.contains(s)));
    let c = if document || rec.notes {
        d
    } else {
        share(&|s| rec.def_stems.contains(s) || rec.desc_stems.contains(s) || rec.use_stems.contains(s) || rec.path_stems.contains(s))
    };
    let m = share(&|s| rec.stems.contains(s) || rec.path_stems.contains(s)).max(c);
    [d, c, m]
}

/// How far a part's holders concentrate in files. Over every pair of distinct holding pieces,
/// weighted by what each holds (`held`), the share of the pairs that lie in one file, against the
/// share `same_file` that two pieces drawn at random from the material show; scaled so that 1 is
/// every pair in one file and 0 is a random spread: (observed - random) / (1 - random). A part a
/// project's own subject names gathers where that subject lives; the words of how a person asks
/// lie wherever prose does. `None` with fewer than two holders, or a material of one file.
fn concentration(held: &[f32], file: &[usize], same_file: f64) -> Option<f32> {
    let mut per: HashMap<usize, f64> = HashMap::new();
    let (mut total, mut squares) = (0.0_f64, 0.0_f64);
    for (h, f) in held.iter().zip(file).filter(|(h, _)| **h > 0.0) {
        let h = f64::from(*h);
        *per.entry(*f).or_default() += h;
        total += h;
        squares += h * h;
    }
    let pairs = total * total - squares;
    if pairs <= 0.0 || same_file >= 1.0 {
        return None;
    }
    let observed = (per.values().map(|x| x * x).sum::<f64>() - squares) / pairs;
    #[expect(clippy::cast_possible_truncation, reason = "a share")]
    Some(((observed - same_file) / (1.0 - same_file)) as f32)
}

/// Whether each part's holders hold the request's other parts more than chance. For part `k`:
/// every piece's joint value over the other parts (`Evidence::joint`'s terms, log((hold + b) /
/// b), with each part's content-free base rate `b`), averaged over the pieces holding `k`
/// weighted by what each holds of it, less its average over every piece. A part naming the
/// request's subject is held where the rest of the request is; the words of how a person asks are
/// held wherever, as a content-free word would be. The same for each content-free pose of `k`
/// (its pose's holdings in place of the part's). `None` with no other part or no holder.
fn associations(held: &[Vec<f32>], null_held: &[Vec<Vec<f32>>], base: &[f32]) -> (Vec<Option<f32>>, Vec<Vec<Option<f32>>>) {
    let n = held.first().map_or(0, Vec::len);
    let term = |j: usize, i: usize| -> f64 {
        let b = f64::from(base.get(j).copied().unwrap_or(0.0));
        if b <= 0.0 { 0.0 } else { ((f64::from(held.get(j).and_then(|v| v.get(i)).copied().unwrap_or(0.0)) + b) / b).ln() }
    };
    let mut posed = Vec::with_capacity(held.len());
    let mut nulls = Vec::with_capacity(held.len());
    for k in 0..held.len() {
        let others: Vec<usize> = (0..held.len()).filter(|j| *j != k && base.get(*j).is_some_and(|b| *b > 0.0)).collect();
        if others.is_empty() || n == 0 {
            posed.push(None);
            nulls.push(Vec::new());
            continue;
        }
        let rest: Vec<f64> = (0..n).map(|i| others.iter().map(|j| term(*j, i)).sum()).collect();
        #[expect(clippy::cast_precision_loss, reason = "piece counts")]
        let chance = rest.iter().sum::<f64>() / n as f64;
        let among = |h: &[f32]| -> Option<f32> {
            let w: f64 = h.iter().map(|x| f64::from(*x)).sum();
            #[expect(clippy::cast_possible_truncation, reason = "nats")]
            (w > 0.0).then(|| (h.iter().zip(&rest).map(|(x, r)| f64::from(*x) * r).sum::<f64>() / w - chance) as f32)
        };
        posed.push(held.get(k).and_then(|h| among(h)));
        nulls.push(null_held.get(k).map_or_else(Vec::new, |v| v.iter().map(|h| among(h)).collect()));
    }
    (posed, nulls)
}

/// What each piece holds of a part where it can settle it: on definition or code lines. A
/// document's text and a piece of notes (`restates`) name the part in prose; where the
/// material's code holds any of the part, such a text only restates what the code holds and
/// counts by its headings alone (its holding anywhere drops to them); where no code holds any of
/// it (material of documents), the passage's text is where it is stated and counts as its code
/// lines. The second role of the result is the share of the part the
/// piece holds where it can settle it.
fn settling(holds: &[[f32; 3]], restates: &dyn Fn(usize) -> bool) -> Vec<[f32; 3]> {
    let code = holds.iter().enumerate().filter(|(i, _)| !restates(*i)).map(|(_, h)| h[1]).fold(0.0_f32, f32::max);
    holds
        .iter()
        .enumerate()
        .map(|(i, h)| match (restates(i), code <= 0.0) {
            (false, _) => *h,
            (true, true) => [h[0], h[2], h[2]],
            (true, false) => [h[0], h[1], h[1]],
        })
        .collect()
}

/// Distinct holdings of a part with their counts.
type Tally = Vec<([f32; 3], usize)>;

/// The chance each piece settles a part, from what the pieces hold of it by role (`hold`), with
/// no fitted weight. A piece settles the part when it holds the thing the part names, not only its
/// words: defining it weighs more than using it, using it more than mentioning it, and how much
/// more is read from the material itself. One piece of the material settles the part, or none
/// does. A piece's evidence `e` counts by how rarely a content-free pose reaches it: its
/// likelihood is the inverse of the share of the content-free poses' (piece, pose) holdings that
/// match or pass `e` in every role (Pareto dominance, so no role is traded against another). The
/// chance piece `i` settles the part is its likelihood over the likelihoods of every piece that
/// holds the part plus the mass a part nobody asked about reaches in this material (each
/// content-free pose's pieces scored against the other poses, averaged): that mass is the chance
/// no piece settles it. A part held the same way by many pieces spreads its chance over them; a
/// part whose holder stands where no content-free pose reaches gets most of it. With `sizes` (each
/// unit's size class, `size_classes`) a unit is compared with the holders of its own class only;
/// empty, every unit is in one class. Returns the chance per piece and (the posed mass, the
/// content-free mass).
fn settle(holds: &[[f32; 3]], nulls: &[Vec<[f32; 3]>], prior: &[f32], null_prior: &[Vec<f32>], sizes: &[usize], counted: &[bool]) -> (Vec<f32>, (f32, f32)) {
    // A piece kept out of the material's counts (`counted` false) is scored against the others'
    // holdings and adds to no pool or mass.
    let counts = |i: usize| counted.get(i).copied().unwrap_or(true);
    // Size classes (`size_classes`): a unit is compared with the holders of its own class; with
    // no classes every unit is in one.
    let classes = sizes.iter().copied().max().map_or(1, |m| m + 1);
    let class = |i: usize| sizes.get(i).copied().unwrap_or(0);
    // Distinct holdings with their counts, per content-free pose (most pieces hold nothing).
    let key = |h: &[f32; 3]| (h[0].to_bits(), h[1].to_bits(), h[2].to_bits());
    let tally_in = |v: &[[f32; 3]], b: usize| -> Vec<([f32; 3], usize)> {
        let mut m: HashMap<(u32, u32, u32), ([f32; 3], usize)> = HashMap::new();
        for (_, h) in v.iter().enumerate().filter(|(i, h)| h[2] > 0.0 && class(*i) == b && counts(*i)) {
            m.entry(key(h)).or_insert((*h, 0)).1 += 1;
        }
        let mut out: Vec<([f32; 3], usize)> = m.into_values().collect();
        out.sort_by(|a, b| a.0[2].total_cmp(&b.0[2]).then(a.0[1].total_cmp(&b.0[1])).then(a.0[0].total_cmp(&b.0[0])));
        out
    };
    let pools: Vec<Vec<Tally>> = nulls.iter().map(|v| (0..classes).map(|b| tally_in(v, b)).collect()).collect();
    let dominates = |x: &[f32; 3], e: &[f32; 3]| x[0] >= e[0] && x[1] >= e[1] && x[2] >= e[2];
    // The likelihood of holding `e` among the pieces that hold the same part (`pool`): one over
    // the share of them at or past `e` in every role. Among the material's holders of a part, a
    // piece that defines it is rare and a piece that names it is common.
    #[expect(clippy::cast_precision_loss, reason = "piece counts")]
    let lik = |e: &[f32; 3], pool: &[([f32; 3], usize)]| -> f32 {
        let held: usize = pool.iter().map(|(_, c)| c).sum();
        let past: usize = pool.iter().filter(|(x, _)| dominates(x, e)).map(|(_, c)| c).sum();
        held as f32 / past.max(1) as f32
    };
    let own: Vec<Vec<([f32; 3], usize)>> = (0..classes).map(|b| tally_in(holds, b)).collect();
    let none: Vec<([f32; 3], usize)> = Vec::new();
    // In nats: each holding piece's log likelihood plus its prior (the drones' surprise for the
    // aspect), so the drones' evidence weighs how strongly and the roles whether it settles.
    let mut memo: HashMap<(usize, (u32, u32, u32)), f32> = HashMap::new();
    let log: Vec<Option<f64>> = holds
        .iter()
        .enumerate()
        .zip(prior.iter().chain(std::iter::repeat(&0.0)))
        .map(|((i, h), p)| (h[2] > 0.0).then(|| f64::from(memo.entry((class(i), key(h))).or_insert_with(|| lik(h, own.get(class(i)).unwrap_or(&none))).ln() + p)))
        .collect();
    let lse = |v: &mut dyn Iterator<Item = f64>| -> f64 {
        let xs: Vec<f64> = v.collect();
        let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if m.is_finite() { m + xs.iter().map(|x| (x - m).exp()).sum::<f64>().ln() } else { f64::NEG_INFINITY }
    };
    let posed = lse(&mut log.iter().enumerate().filter(|(i, _)| counts(*i)).filter_map(|(_, l)| *l));
    // The content-free mass: each pose's holding pieces, by the same rule (among that pose's
    // holders, its own drones' surprise as the prior), averaged over the poses.
    #[expect(clippy::cast_precision_loss, reason = "pose counts")]
    let null_mass = if pools.is_empty() {
        f64::NEG_INFINITY
    } else {
        let per: Vec<f64> = nulls
            .iter()
            .zip(&pools)
            .enumerate()
            .map(|(j, (hs, pool))| {
                let mut m: HashMap<(usize, (u32, u32, u32)), f32> = HashMap::new();
                let np = null_prior.get(j);
                lse(&mut hs.iter().enumerate().filter(|(i, h)| h[2] > 0.0 && counts(*i)).map(|(i, h)| {
                    f64::from(m.entry((class(i), key(h))).or_insert_with(|| lik(h, pool.get(class(i)).unwrap_or(&none))).ln() + np.and_then(|v| v.get(i)).copied().unwrap_or(0.0))
                }))
            })
            .collect();
        lse(&mut per.iter().copied()) - (per.len() as f64).ln()
    };
    let denom = lse(&mut [posed, null_mass].into_iter());
    #[expect(clippy::cast_possible_truncation, reason = "a chance in [0, 1]")]
    let chance = log.iter().map(|l| l.map_or(0.0, |x| if denom.is_finite() { (x - denom).exp() as f32 } else { 0.0 })).collect();
    #[expect(clippy::cast_possible_truncation, reason = "nats for the audit")]
    (chance, (posed as f32, null_mass as f32))
}

/// The epistemic class of piece `rec` for an aspect with stems `stems`: a test, a document or
/// notes, a definition of what the aspect names, a use of it in code, or a mention in comments
/// and strings.
#[must_use]
pub fn tag(rec: &Record, stems: &[String], document: bool) -> Tag {
    if rec.test {
        Tag::Tests
    } else if document || rec.notes {
        Tag::Documents
    } else if stems.iter().any(|s| rec.def_stems.contains(s)) {
        Tag::Defines
    } else if stems.iter().any(|s| rec.use_stems.contains(s)) {
        Tag::Uses
    } else {
        Tag::Mentions
    }
}

/// FNV-1a over the bytes of `parts`, for choosing null words that repeat run to run.
fn hash(parts: &[&str]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for p in parts {
        for b in p.bytes().chain(std::iter::once(0)) {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// A word the material holds in as many pieces as `w`, not one of `avoid` where the material
/// allows, never one of `asked`, chosen by `seed`.
fn null_word(rarity: &Rarity, w: &str, asked: &[String], avoid: &[String], seed: u64) -> String {
    let like = rarity.like(rarity.bag_df(w), (NULLS + 1) * (asked.len() + 1));
    if like.is_empty() {
        return String::new();
    }
    let n = like.len() as u64;
    let pick = |ok: &dyn Fn(&String) -> bool| {
        (0..n).find_map(|step| like.get(usize::try_from(seed.wrapping_add(step) % n).unwrap_or(0)).map(|x| &x.1).filter(|c| *c != w && ok(c)))
    };
    // A word not used yet; in a small material whose words of that frequency are all used, one
    // used before (never a word of the request itself, which `avoid` lists first).
    pick(&|c| !avoid.contains(c)).or_else(|| pick(&|c| !asked.contains(c))).cloned().unwrap_or_default()
}

/// The words of an aspect as poses read them, each split into its identifier parts (a quantity
/// stays whole), lower-cased.
fn aspect_parts(a: &crate::intake::Aspect) -> Vec<Vec<String>> {
    a.words
        .iter()
        .map(|w| if a.form == Form::Quantity { vec![w.to_lowercase()] } else { nmlex::name_parts(&w.replace(['/', '.'], "_")) })
        .filter(|p| !p.is_empty())
        .collect()
}

/// The exact forms an aspect is written in: each word written as an identifier (parts joined),
/// and the whole aspect when it has two words or more. `written` gives each word as written.
fn phrases(parts: &[Vec<String>], written: &[String]) -> Vec<String> {
    let mut out: Vec<String> = parts.iter().zip(written).filter(|(p, w)| p.len() > 1 || w.contains(['_', '.', ':'])).map(|(_, w)| w.to_lowercase()).collect();
    if written.len() >= 2 {
        out.push(written.iter().map(|w| w.to_lowercase()).collect::<Vec<_>>().join(" "));
    }
    out
}

/// The query table the poses share: unit token rows, appended per pose.
struct Queries {
    width: usize,
    rows: Vec<f32>,
}
impl Queries {
    fn add(&mut self, drone: &Drone, rarity: &Rarity, text: &str) -> Result<(Vec<usize>, Vec<f32>), Fault> {
        let toks = drone.tokens.tokens(text)?;
        let mut seen: Vec<i32> = Vec::new();
        let mut queries = Vec::new();
        let mut weights = Vec::new();
        let mut buf = vec![0.0_f32; self.width];
        for t in toks {
            if seen.contains(&t) {
                continue;
            }
            seen.push(t);
            let Ok(id) = usize::try_from(t) else { continue };
            let w = rarity.token(t);
            if w <= 0.0 || !drone.table.row(id, &mut buf) {
                continue;
            }
            let n = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n <= 0.0 {
                continue;
            }
            queries.push(self.rows.len() / self.width.max(1));
            self.rows.extend(buf.iter().map(|x| x / n));
            weights.push(w);
        }
        Ok((queries, weights))
    }
}

/// A pose for an aspect's words (each as its parts) and exact forms, its query rows added to `q`.
fn pose(drone: &Drone, rarity: &Rarity, q: &mut Queries, parts: &[Vec<String>], phrase: Vec<String>) -> Result<Pose, Fault> {
    let flat: Vec<&String> = parts.iter().flatten().collect();
    let text = flat.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ");
    let bm25 = bm25_words(&text).into_iter().map(|w| (rarity.bm25_idf(&w), w)).map(|(i, w)| (w, i)).collect();
    let mut stems: Vec<(String, f32)> = Vec::new();
    for w in flat {
        let s = nmlex::stem(w);
        if !stems.iter().any(|x| x.0 == s) {
            let r = rarity.stem(&s);
            stems.push((s, r));
        }
    }
    let (queries, weights) = q.add(drone, rarity, &text)?;
    Ok(Pose { bm25, stems, phrase, queries, weights })
}

/// The raw signals of one piece under every pose, given the best cosine of each query row over
/// the piece's tokens.
fn raw(rec: &Record, poses: &[Pose], best: &[f32], rarity: &Rarity) -> Vec<[f32; 6]> {
    let len = |l: u32| f32::from(u16::try_from(l).unwrap_or(u16::MAX));
    let norm = BM25_K1 * (1.0 - BM25_B + BM25_B * len(rec.bag_len) / rarity.avg_len().max(1.0));
    let normf = BM25_K1 * (1.0 - BM25_B + BM25_B * len(rec.bagf_len) / rarity.avg_len_f().max(1.0));
    let bm = |bag: &HashMap<String, u32>, n: f32, words: &[(String, f32)]| -> f32 {
        words.iter().filter_map(|(w, idf)| bag.get(w).map(|f| idf * len(*f) * (BM25_K1 + 1.0) / (len(*f) + n))).sum()
    };
    let share = |sets: &[&std::collections::HashSet<String>], stems: &[(String, f32)]| -> f32 {
        let total: f32 = stems.iter().map(|s| s.1).sum();
        if total > 0.0 { stems.iter().filter(|(s, _)| sets.iter().any(|x| x.contains(s))).map(|s| s.1).sum::<f32>() / total } else { 0.0 }
    };
    poses
        .iter()
        .map(|p| {
            let wsum: f32 = p.weights.iter().sum();
            let affinity = if wsum > 0.0 && !rec.distinct.is_empty() {
                p.queries.iter().zip(&p.weights).map(|(q, w)| best.get(*q).copied().unwrap_or(0.0).max(0.0) * w).sum::<f32>() / wsum
            } else {
                0.0
            };
            [
                bm(&rec.bagf, normf, &p.bm25),
                bm(&rec.bag, norm, &p.bm25),
                if p.phrase.is_empty() { 0.0 } else { p.phrase.iter().filter(|x| rec.low.contains(x.as_str())).count() as f32 / p.phrase.len() as f32 },
                share(&[&rec.def_stems, &rec.desc_stems], &p.stems),
                share(&[&rec.def_stems, &rec.use_stems, &rec.desc_stems], &p.stems),
                affinity,
            ]
        })
        .collect()
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v.get(n / 2).copied().unwrap_or(0.0) } else { (v.get(n / 2 - 1).copied().unwrap_or(0.0) + v.get(n / 2).copied().unwrap_or(0.0)) / 2.0 }
}

/// The unit a signal is measured in: the typical size of a content-free difference, the median
/// of the null poses' differences (each null read as if posed, against the other two) at the
/// pieces a null pose fired on; the mean of them where that median is zero, else 1. One unit is
/// what a word of like frequency typically scores at a piece it hits.
fn unit(nulls: &[Vec<f32>], n: usize, counted: &[bool]) -> f32 {
    let at = |v: &[f32], i: usize| v.get(i).copied().unwrap_or(0.0);
    let counts = |i: usize| counted.get(i).copied().unwrap_or(true);
    let others = nulls.len().saturating_sub(1).max(1) as f32;
    let mut sizes: Vec<f32> = Vec::new();
    for (j, nj) in nulls.iter().enumerate() {
        for i in (0..n).filter(|i| at(nj, *i) > 0.0 && counts(*i)) {
            let rest = nulls.iter().enumerate().filter(|(o, _)| *o != j).map(|(_, v)| at(v, i)).sum::<f32>() / others;
            sizes.push((at(nj, i) - rest).abs());
        }
    }
    if sizes.is_empty() {
        return 1.0;
    }
    let mean = sizes.iter().sum::<f32>() / sizes.len() as f32;
    let m = median(&mut sizes);
    if m > 0.0 {
        m
    } else if mean > 0.0 {
        mean
    } else {
        1.0
    }
}

/// The robust fit of the content-free level on the nuisance basis, from `samples` (a piece and a
/// null pose's calibrated value there, each null read as if posed): the slope on the logarithm of
/// the tokens by Tukey's resistant line (the medians of the outer thirds), then the median of each
/// class-and-place cell (a cell no sample falls in takes the median of all). Returns each of the
/// `n` pieces' fitted value: what a content-free request scores at a piece of that length, kind
/// and place.
fn fit(samples: &[(usize, f32)], n: usize, places: &[Place]) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; n];
    }
    let lx = |i: usize| places.get(i).map_or(0.0, |p| (1.0 + p.tokens as f32).ln());
    // Lengths outside those the fit saw are read at the nearest end of that range (no
    // extrapolation).
    let lo_x = samples.iter().map(|s| lx(s.0)).fold(f32::INFINITY, f32::min);
    let hi_x = samples.iter().map(|s| lx(s.0)).fold(f32::NEG_INFINITY, f32::max);
    let x = |i: usize| lx(i).clamp(lo_x, hi_x);
    let mut by_x: Vec<(usize, f32)> = samples.to_vec();
    by_x.sort_by(|a, b| x(a.0).total_cmp(&x(b.0)));
    let third = by_x.len() / 3;
    let slope = if third > 0 {
        let lo: Vec<(usize, f32)> = by_x.iter().take(third).copied().collect();
        let hi: Vec<(usize, f32)> = by_x.iter().rev().take(third).copied().collect();
        let mx = |s: &[(usize, f32)]| median(&mut s.iter().map(|v| x(v.0)).collect::<Vec<_>>());
        let mz = |s: &[(usize, f32)]| median(&mut s.iter().map(|v| v.1).collect::<Vec<_>>());
        let dx = mx(&hi) - mx(&lo);
        if dx > 0.0 { (mz(&hi) - mz(&lo)) / dx } else { 0.0 }
    } else {
        0.0
    };
    let cell = |i: usize| places.get(i).map(|p| (p.class, p.first));
    let mut cells: HashMap<(Class, bool), Vec<f32>> = HashMap::new();
    for &(i, v) in samples {
        if let Some(c) = cell(i) {
            cells.entry(c).or_default().push(v - slope * x(i));
        }
    }
    let all = median(&mut samples.iter().map(|(i, v)| v - slope * x(*i)).collect::<Vec<_>>());
    let offsets: HashMap<(Class, bool), f32> = cells.into_iter().map(|(c, mut v)| (c, median(&mut v))).collect();
    (0..n).map(|i| slope * x(i) + cell(i).and_then(|c| offsets.get(&c).copied()).unwrap_or(all)).collect()
}

/// Whether the surprise past every content-free value reads the fitted tail (`tail_scale`) and the
/// settle prior keeps it (no ceiling at the logarithm of the sample count): off, unless
/// `NANO_TAIL=on` (off is the earlier form, `upper_spread` and the ceiling). Measured
/// 2026-11-17: ties at a part's top fall (parts with a tied top 422 to 47 of 559), stack first 92
/// to 96, every range 131 to 126 of 188, not significant; build plan "Documents and the surprise
/// ceiling".
#[must_use]
pub fn tail_fit_on() -> bool {
    std::env::var("NANO_TAIL").is_ok_and(|v| v == "on")
}

/// The earlier tail scale: the content-free values' mean excess over their median, above it.
fn upper_spread(desc: &[f32]) -> f32 {
    let mid = median(&mut desc.to_vec());
    let upper: Vec<f32> = desc.iter().filter(|v| **v > mid).map(|v| v - mid).collect();
    if upper.is_empty() { 1.0 } else { (upper.iter().sum::<f32>() / upper.len() as f32).max(f32::MIN_POSITIVE) }
}

/// The scale of an exponential fitted to the top of the content-free values (`desc`, largest
/// first), peaks over a threshold: the threshold is where their own distribution marks its upper
/// tail (Tukey's upper fence, the third quartile plus one and a half spreads between the
/// quartiles), and the scale is the exceedances' mean excess over it (the exponential's maximum
/// likelihood scale). Fallback when fewer than two values pass the fence: the top square root of
/// their count over the next one; then the earlier form (`upper_spread`).
fn tail_scale(desc: &[f32]) -> f32 {
    let n = desc.len();
    if n < 4 {
        return upper_spread(desc);
    }
    let q = |p: f32| -> f32 {
        // `desc` is largest first: the p-quantile from below.
        #[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a rank from a share")]
        let r = ((1.0 - p) * (n - 1) as f32).round() as usize;
        desc.get(r.min(n - 1)).copied().unwrap_or(0.0)
    };
    let (q1, q3) = (q(0.25), q(0.75));
    let excess = |u: f32| -> Option<f32> {
        let e: Vec<f32> = desc.iter().take_while(|v| **v > u).map(|v| v - u).collect();
        (e.len() >= 2).then(|| e.iter().sum::<f32>() / e.len() as f32).filter(|s| *s > 0.0)
    };
    #[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a count's square root")]
    let k = ((n as f32).sqrt().ceil() as usize).clamp(2, n - 1);
    excess(q3 + 1.5 * (q3 - q1)).or_else(|| desc.get(k).and_then(|u| excess(*u))).unwrap_or_else(|| upper_spread(desc))
}

/// One signal of one aspect, calibrated: the residual values, their surprise, the band, and each
/// content-free pose's surprise at every piece (read as if posed, by the same rule).
type Calibrated = (Vec<f32>, Vec<f32>, f32, Vec<Vec<f32>>);

/// Calibrate one signal of one aspect: `posed` and the three `nulls` over every piece. Returns the
/// residual values (no evidence where the posed signal did not fire), their surprise, the band, and
/// the surprise each null pose reaches at each piece (0 where it did not fire).
fn calibrate(posed: &[f32], nulls: &[Vec<f32>], places: &[Place], counted: &[bool]) -> Calibrated {
    let n = posed.len();
    // Pieces kept out of the material's counts (`counted` false: a project's own hidden
    // configuration) are scored against the content-free values of the others and add none.
    let counts = |i: usize| counted.get(i).copied().unwrap_or(true);
    let at = |v: &[f32], i: usize| v.get(i).copied().unwrap_or(0.0);
    let k = nulls.len().max(1) as f32;
    let others = nulls.len().saturating_sub(1).max(1) as f32;
    let fired: Vec<usize> = (0..n).filter(|i| at(posed, *i) > 0.0).collect();
    if fired.is_empty() {
        return (vec![f32::NEG_INFINITY; n], vec![0.0; n], f32::INFINITY, vec![vec![0.0; n]; nulls.len()]);
    }
    let sd = unit(nulls, n, counted);
    // Each null pose read as if posed, against the other two, in the same unit, where it fired:
    // the content-free level the nuisance fit is taken from.
    let null_as_posed = |j: usize, i: usize| -> f32 {
        let rest = nulls.iter().enumerate().filter(|(o, _)| *o != j).map(|(_, v)| at(v, i)).sum::<f32>() / others;
        (nulls.get(j).map_or(0.0, |v| at(v, i)) - rest) / sd
    };
    let samples: Vec<(usize, f32)> =
        (0..nulls.len()).flat_map(|j| (0..n).filter(move |i| counts(*i) && nulls.get(j).is_some_and(|v| at(v, *i) > 0.0)).map(move |i| (i, j))).map(|(i, j)| (i, null_as_posed(j, i))).collect();
    let f = fit(&samples, n, places);
    let mut r = vec![f32::NEG_INFINITY; n];
    for &i in &fired {
        let z = (at(posed, i) - nulls.iter().map(|v| at(v, i)).sum::<f32>() / k) / sd;
        if let Some(x) = r.get_mut(i) {
            *x = z - f.get(i).copied().unwrap_or(0.0);
        }
    }
    // The band: the largest residual a null pose reaches at the pieces the posed signal fired on.
    let band = (0..nulls.len())
        .flat_map(|j| fired.iter().filter(|i| counts(**i)).map(move |&i| (j, i)))
        .map(|(j, i)| null_as_posed(j, i) - f.get(i).copied().unwrap_or(0.0))
        .fold(f32::NEG_INFINITY, f32::max);
    // The content-free values over every piece, each null pose read as if posed; where a null
    // did not fire it holds nothing (below any value).
    let mut null_values: Vec<f32> = samples.iter().map(|(i, v)| v - f.get(*i).copied().unwrap_or(0.0)).collect();
    null_values.sort_by(|a, b| b.total_cmp(a));
    let all = (nulls.len() * (0..n).filter(|i| counts(*i)).count() + 1) as f32;
    // Beyond the largest content-free value the empirical share cannot fall further; there the
    // tail goes on as an exponential fitted to the top of the content-free values (peaks over a
    // threshold, `tail_scale`), anchored at the largest one, so values past every null are told
    // apart by how far past they stand in the tail's own unit.
    let top = null_values.first().copied().unwrap_or(0.0);
    let tail = if tail_fit_on() { tail_scale(&null_values) } else { upper_spread(&null_values) };
    let surprise_of = |v: f32| -> f32 {
        if v.is_finite() {
            let reach = null_values.partition_point(|x| *x >= v) as f32;
            -((1.0 + reach) / all).ln() + (v - top).max(0.0) / tail
        } else {
            0.0
        }
    };
    let surprise: Vec<f32> = r.iter().map(|v| surprise_of(*v)).collect();
    let mut null_surprise = vec![vec![0.0_f32; n]; nulls.len()];
    for (j, row) in null_surprise.iter_mut().enumerate() {
        for (i, x) in row.iter_mut().enumerate() {
            if nulls.get(j).is_some_and(|v| at(v, i) > 0.0) {
                *x = surprise_of(null_as_posed(j, i) - f.get(i).copied().unwrap_or(0.0));
            }
        }
    }
    (r, surprise, band, null_surprise)
}

/// Sense every piece of `records` for `request` (its drone's aspects), calibrated. `places` gives
/// each piece's nuisance, one per record; `meaning`, when the meaning lane runs, each piece's
/// embedding cosine with the question, then with each content-free question. Runs in parallel over
/// the processor cores.
/// # Errors
/// The tokenizer refuses an aspect.
#[expect(clippy::too_many_arguments, reason = "the drone, the request and its label, the material, its rarity and nuisance, and the meaning lane")]
pub fn sense(drone: &Drone, label: &str, request: &Request, records: &[Record], rarity: &Rarity, places: &[Place], meaning: Option<&[Vec<f32>]>) -> Result<Evidence, Fault> {
    sense_sized(drone, label, request, records, rarity, places, meaning, false)
}

/// Size classes of units of very different lengths (items: a constant of one line, a function of
/// hundreds): the units ranked by their tokens and cut into classes of equal count, as many as
/// Sturges' rule gives for their number (the base-2 logarithm plus one). What a unit holds is
/// read against units of its own class: the content-free rate of holding a part (`joint`) and the
/// holders it is compared with (`settle`); a long unit holds most words somewhere by its length
/// alone.
#[must_use]
pub fn size_classes(places: &[Place]) -> Vec<usize> {
    let n = places.len();
    if n == 0 {
        return Vec::new();
    }
    #[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a class count from the unit count")]
    let classes = ((n as f64).log2().floor() as usize + 1).max(1);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|i| (places.get(*i).map_or(0, |p| p.tokens), *i));
    let mut out = vec![0_usize; n];
    for (rank, i) in order.into_iter().enumerate() {
        if let Some(x) = out.get_mut(i) {
            *x = rank * classes / n;
        }
    }
    out
}

/// `sense`, with `sized`: units of very different lengths are read in size classes
/// (`size_classes`).
/// # Errors
/// As `sense`.
#[expect(clippy::too_many_arguments, reason = "sense's arguments and the size switch")]
pub fn sense_sized(
    drone: &Drone,
    label: &str,
    request: &Request,
    records: &[Record],
    rarity: &Rarity,
    places: &[Place],
    meaning: Option<&[Vec<f32>]>,
    sized: bool,
) -> Result<Evidence, Fault> {
    let sizes: Vec<usize> = if sized { size_classes(places) } else { Vec::new() };
    // A project's own hidden configuration is scored but kept out of every content-free count
    // (`nmtype::in_hidden_folder`), as it is out of the rarities: the other pieces' evidence is
    // what it was before it was read.
    let counted: Vec<bool> = records.iter().map(|r| !nmtype::in_hidden_folder(&r.path_low)).collect();
    let width = drone.table.width();
    let mut q = Queries { width, rows: Vec::new() };
    let asked: Vec<String> = request.aspects.iter().flat_map(aspect_parts).flatten().collect();
    let mut poses: Vec<Pose> = Vec::new();
    let mut nulls: Vec<Vec<String>> = Vec::new();
    for a in &request.aspects {
        let parts = aspect_parts(a);
        poses.push(pose(drone, rarity, &mut q, &parts, phrases(&parts, &a.words))?);
        let mut used: Vec<String> = Vec::new();
        for k in 0..NULLS {
            let mut avoid = asked.clone();
            avoid.extend(used.iter().cloned());
            let mut np: Vec<Vec<String>> = Vec::new();
            for (j, word) in parts.iter().enumerate() {
                let mut w2 = Vec::new();
                for (m, part) in word.iter().enumerate() {
                    let seed = hash(&[label, &a.text, part, &k.to_string(), &j.to_string(), &m.to_string()]);
                    let c = null_word(rarity, part, &asked, &avoid, seed);
                    avoid.push(c.clone());
                    w2.push(c);
                }
                np.push(w2);
            }
            let written: Vec<String> = np.iter().map(|w| w.join("_")).collect();
            used.extend(np.iter().flatten().cloned());
            poses.push(pose(drone, rarity, &mut q, &np, phrases(&np, &written))?);
            nulls.push(written);
        }
    }
    let nq = q.rows.len() / width.max(1);
    // The cosine of every token the material holds with every query row, once, the tokens split
    // over the processor cores; then every piece's raw signals under every pose, the pieces split
    // the same way, each piece reading the best cosine of each query over its own tokens.
    // Partitions fixed by the processor's thread count (read once); the workers lent by the one
    // budget (`nmtype::budget`, 2026-11-18 Phase C1), so values never depend on the load.
    let cores = nmtype::budget::total();
    let ids = rarity.token_ids();
    let cos = drone.cosines_for(&ids, &q.rows, -1.0)?;
    let cos_ref = &cos;
    let poses_ref = &poses;
    let run = records.len().div_ceil(cores.max(1)).max(1);
    let runs = records.len().div_ceil(run);
    let parts = nmtype::budget::run(runs, |p| {
        let slice = records.get(p * run..((p + 1) * run).min(records.len())).unwrap_or(&[]);
        let mut best = vec![-1.0_f32; nq];
        slice
            .iter()
            .map(|rec| {
                best.fill(-1.0_f32);
                for &t in &rec.distinct {
                    let Some(c) = cos_ref.get(t) else { continue };
                    for (b, x) in best.iter_mut().zip(c) {
                        if *x > *b {
                            *b = *x;
                        }
                    }
                }
                raw(rec, poses_ref, &best, rarity)
            })
            .collect::<Vec<_>>()
    });
    let per_piece: Vec<Vec<[f32; 6]>> = parts.into_iter().flatten().flatten().collect();
    if per_piece.len() != records.len() {
        return Err(Fault::new("nmpod.evidence", nmtype::FaultKind::Execution, "drone sensing: a run was lost"));
    }
    let stride = 1 + NULLS;
    // `null_s[a * SIGNALS + s][pose][i]`: each content-free pose's surprise at each piece.
    let mut null_s: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut value = Vec::with_capacity(request.aspects.len());
    let mut band = Vec::with_capacity(request.aspects.len());
    let mut surprise = Vec::with_capacity(request.aspects.len());
    // The meaning lane's cosine against the content-free questions', as every signal is calibrated
    // (a piece with no vector did not fire).
    let lane = meaning.filter(|m| m.len() >= 3 && m.iter().all(|v| v.len() == records.len())).map(|m| {
        let fin = |v: &Vec<f32>| -> Vec<f32> { v.iter().map(|x| if x.is_finite() { *x } else { 0.0 }).collect() };
        let nulls: Vec<Vec<f32>> = m.iter().skip(1).map(fin).collect();
        m.first().map_or_else(|| calibrate(&[], &[], places, &counted), |p| calibrate(&fin(p), &nulls, places, &counted))
    });
    for a in 0..request.aspects.len() {
        let mut va = Vec::with_capacity(SIGNALS.len());
        let mut ba = Vec::with_capacity(SIGNALS.len());
        let mut sa = Vec::with_capacity(SIGNALS.len());
        for s in 0..RAW {
            let col = |p: usize| -> Vec<f32> { per_piece.iter().map(|r| r.get(p).map_or(0.0, |x| x.get(s).copied().unwrap_or(0.0))).collect() };
            let posed = col(a * stride);
            let ns: Vec<Vec<f32>> = (1..stride).map(|k| col(a * stride + k)).collect();
            let (r, sp, b, nsp) = calibrate(&posed, &ns, places, &counted);
            va.push(r);
            sa.push(sp);
            ba.push(b);
            null_s.push(nsp);
        }
        let (r, sp, b, nsp) = lane.clone().unwrap_or_else(|| (vec![f32::NEG_INFINITY; records.len()], vec![0.0; records.len()], f32::INFINITY, vec![vec![0.0; records.len()]; NULLS]));
        va.push(r);
        sa.push(sp);
        ba.push(b);
        null_s.push(nsp);
        value.push(va);
        band.push(ba);
        surprise.push(sa);
    }
    let ident = SIGNALS.iter().position(|n| *n == "ident").unwrap_or(0);
    let share: Vec<Vec<f32>> = (0..request.aspects.len())
        .map(|a| per_piece.iter().map(|r| r.get(a * stride).and_then(|x| x.get(ident)).copied().unwrap_or(0.0)).collect())
        .collect();
    // What each piece holds of each part by role, posed and under each content-free pose (the
    // pose's null word in place of each of the part's words), and the chance each piece settles it.
    let stem_w = |s: String| {
        let r = rarity.held(&s);
        (s, r)
    };
    let mut settle_v: Vec<Vec<f32>> = Vec::new();
    let mut settle_mass: Vec<(f32, f32)> = Vec::new();
    let mut settle_why: Vec<Vec<([f32; 3], f32)>> = Vec::new();
    let mut settle_base: Vec<f32> = Vec::new();
    let mut settle_base_class: Vec<Vec<f32>> = Vec::new();
    let measured = frame_measured();
    // Each piece's file (by its path), and the chance two distinct pieces drawn at random from the
    // material share a file: what a part's holders would show with no concentration.
    let file_ids: Vec<usize> = if !measured {
        Vec::new()
    } else {
        let mut ids: HashMap<&str, usize> = HashMap::new();
        records.iter().map(|r| { let n = ids.len(); *ids.entry(r.path_low.as_str()).or_insert(n) }).collect()
    };
    let same_file = {
        let mut per: HashMap<usize, f64> = HashMap::new();
        for f in &file_ids {
            *per.entry(*f).or_default() += 1.0;
        }
        #[expect(clippy::cast_precision_loss, reason = "piece counts")]
        let n = records.len() as f64;
        if n < 2.0 { 1.0 } else { per.values().map(|c| c * (c - 1.0)).sum::<f64>() / (n * (n - 1.0)) }
    };
    let mut concentration: Vec<Option<f32>> = Vec::new();
    let mut concentration_null: Vec<Vec<Option<f32>>> = Vec::new();
    let mut holders: Vec<(usize, usize)> = Vec::new();
    // Per part, what each piece holds of it anywhere under each content-free pose (`associations`).
    let mut null_held: Vec<Vec<Vec<f32>>> = Vec::new();
    for (a, asp) in request.aspects.iter().enumerate() {
        // The aspect's words with their parts, as `aspect_parts` keeps them (the null poses' order).
        let quantity = asp.form == Form::Quantity;
        let kept: Vec<(&String, Vec<String>)> = asp
            .words
            .iter()
            .map(|w| (w, if quantity { vec![w.to_lowercase()] } else { nmlex::name_parts(&w.replace(['/', '.'], "_")) }))
            .filter(|(_, ps)| !ps.is_empty())
            .collect();
        for p in &asp.parts {
            let idx: Vec<usize> = p.words.iter().filter_map(|w| kept.iter().position(|x| x.0.eq_ignore_ascii_case(w))).collect();
            let stems_of = |ps: &[String]| -> Vec<String> { ps.iter().map(|x| if quantity { x.clone() } else { nmlex::stem(x) }).collect() };
            let mut posed: Vec<(String, f32)> = Vec::new();
            for j in &idx {
                for s in stems_of(kept.get(*j).map_or(&[][..], |x| x.1.as_slice())) {
                    if !posed.iter().any(|x| x.0 == s) {
                        posed.push(stem_w(s));
                    }
                }
            }
            let null_sets: Vec<Vec<(String, f32)>> = (0..NULLS)
                .map(|k| {
                    let written = nulls.get(a * NULLS + k);
                    let mut v: Vec<(String, f32)> = Vec::new();
                    for j in &idx {
                        let ps: Vec<String> = written.and_then(|w| w.get(*j)).map(|w| w.split('_').filter(|x| !x.is_empty()).map(str::to_owned).collect()).unwrap_or_default();
                        for s in stems_of(&ps) {
                            if !v.iter().any(|x| x.0 == s) {
                                v.push(stem_w(s));
                            }
                        }
                    }
                    v
                })
                .collect();
            // The part's code names: a word written as one name of several parts, in the forms it
            // is written in as one (as asked, its parts joined by an underscore, joined whole).
            let name = |w: &str| -> Option<(Vec<String>, Vec<String>)> {
                let ps = nmlex::name_parts(&w.replace(['/', '.'], "_"));
                (!quantity && ps.len() >= 2 && !w.contains(char::is_whitespace)).then(|| {
                    let mut forms = vec![w.to_lowercase(), ps.join("_"), ps.concat()];
                    forms.dedup();
                    (forms, stems_of(&ps))
                })
            };
            let names: Vec<(Vec<String>, Vec<String>)> = idx.iter().filter_map(|j| kept.get(*j)).filter_map(|x| name(x.0)).collect();
            let null_names: Vec<Vec<(Vec<String>, Vec<String>)>> = (0..NULLS)
                .map(|k| idx.iter().filter_map(|j| nulls.get(a * NULLS + k).and_then(|w| w.get(*j))).filter(|w| w.contains('_')).filter_map(|w| name(w)).collect())
                .collect();
            let doc = |i: usize| places.get(i).is_some_and(|pl| pl.class == Class::Document);
            let holds: Vec<[f32; 3]> = records.iter().enumerate().map(|(i, r)| hold(r, doc(i), &posed, &names)).collect();
            let null_holds: Vec<Vec<[f32; 3]>> = null_sets
                .iter()
                .zip(&null_names)
                .map(|(ns, nn)| records.iter().enumerate().map(|(i, r)| hold(r, doc(i), ns, nn)).collect())
                .collect();
            // The drones' prior on each piece for this part's aspect (posed, and under each
            // content-free pose): the wording and meaning families' surprise, added in nats (the
            // naming family reads the stems the roles read, so it is left out here). A family
            // counts at most what the content-free sample resolves, the logarithm of its size: past
            // every content-free value the surprise keeps order, but no larger likelihood is
            // measured.
            let meaning: Vec<usize> = (0..SIGNALS.len()).filter(|s| FAMILY.get(*s) == Some(&Family::Meaning)).collect();
            #[expect(clippy::cast_precision_loss, reason = "piece counts")]
            let resolved = ((NULLS * records.len() + 1) as f32).ln();
            // With the fitted tail (`tail_fit_on`) past the sample is measured by the fit, so no
            // ceiling.
            let ceiling = if tail_fit_on() { f32::INFINITY } else { resolved };
            let fam = |get: &dyn Fn(usize) -> f32, fs: &[usize]| fs.iter().map(|s| get(*s)).fold(0.0_f32, f32::max).min(ceiling);
            let prior: Vec<f32> = (0..records.len())
                .map(|i| {
                    let g = |s: usize| surprise.get(a).and_then(|x| x.get(s)).and_then(|x| x.get(i)).copied().unwrap_or(0.0);
                    fam(&g, &meaning)
                })
                .collect();
            let null_prior: Vec<Vec<f32>> = (0..NULLS)
                .map(|j| {
                    (0..records.len())
                        .map(|i| {
                            let g = |s: usize| null_s.get(a * SIGNALS.len() + s).and_then(|x| x.get(j)).and_then(|x| x.get(i)).copied().unwrap_or(0.0);
                            fam(&g, &meaning)
                        })
                        .collect()
                })
                .collect();
            // Whether the part's holders (anywhere) concentrate in files, posed and per pose (read
            // only for the frame test's measurement, `frame_measured`).
            if measured {
                let anywhere = |h: &[[f32; 3]]| -> Vec<f32> { h.iter().map(|x| x[2].max(0.0)).collect() };
                concentration.push(self::concentration(&anywhere(&holds), &file_ids, same_file));
                concentration_null.push(null_holds.iter().map(|h| self::concentration(&anywhere(h), &file_ids, same_file)).collect());
            }
            if measured {
                let mut fs: Vec<usize> = holds.iter().zip(&file_ids).filter(|(h, _)| h[2] > 0.0).map(|(_, f)| *f).collect();
                let m = fs.len();
                fs.sort_unstable();
                fs.dedup();
                holders.push((m, fs.len()));
            }
            // Where a piece can settle the part (`settling`), and the drones' prior counted for the
            // share of the part it holds there.
            let restates = |i: usize| doc(i) || records.get(i).is_some_and(|r| r.notes);
            let holds = settling(&holds, &restates);
            let null_holds: Vec<Vec<[f32; 3]>> = null_holds.iter().map(|h| settling(h, &restates)).collect();
            // What each content-free pose holds where it can settle the part (`associations`).
            if measured {
                null_held.push(null_holds.iter().map(|v| v.iter().map(|h| h[2].max(0.0)).collect()).collect());
            }
            let (chance, mass) = settle(&holds, &null_holds, &prior, &null_prior, &sizes, &counted);
            let cells = (null_holds.len() * counted.iter().filter(|c| **c).count()).max(1) as f32;
            settle_base.push(null_holds.iter().flat_map(|v| v.iter().zip(&counted).filter(|(_, c)| **c).map(|(h, _)| h[2].max(0.0))).sum::<f32>() / cells);
            // Per size class, the same rate over the class's units only.
            if !sizes.is_empty() {
                let classes = sizes.iter().copied().max().map_or(0, |m| m + 1);
                settle_base_class.push(
                    (0..classes)
                        .map(|b| {
                            let vals: Vec<f32> = null_holds.iter().flat_map(|v| v.iter().enumerate().filter(|(i, _)| sizes.get(*i) == Some(&b)).map(|(_, h)| h[2].max(0.0))).collect();
                            vals.iter().sum::<f32>() / vals.len().max(1) as f32
                        })
                        .collect(),
                );
            }
            settle_why.push(holds.iter().copied().zip(prior.iter().copied()).collect());
            settle_v.push(chance);
            settle_mass.push(mass);
        }
    }
    let (association, association_null) = if measured {
        let held_now: Vec<Vec<f32>> = settle_why.iter().map(|v| v.iter().map(|(h, _)| h[2].max(0.0)).collect()).collect();
        associations(&held_now, &null_held, &settle_base)
    } else {
        (Vec::new(), Vec::new())
    };
    let parts: Vec<(usize, String, Vec<f32>)> = request
        .aspects
        .iter()
        .enumerate()
        .flat_map(|(a, asp)| asp.parts.iter().map(move |p| (a, p)))
        .map(|(a, p)| {
            let weights: Vec<(String, f32)> = p.stems.iter().map(|st| (st.clone(), rarity.stem(st))).collect();
            let total: f32 = weights.iter().map(|w| w.1).sum();
            let share = records
                .iter()
                .map(|r| {
                    if total > 0.0 {
                        weights.iter().filter(|(st, _)| r.def_stems.contains(st) || r.desc_stems.contains(st) || r.use_stems.contains(st)).map(|w| w.1).sum::<f32>() / total
                    } else {
                        0.0
                    }
                })
                .collect();
            (a, p.text.clone(), share)
        })
        .collect();
    // The names the request writes as identifiers that can be called (a path or a file name is
    // not), each by its last part.
    let names: Vec<String> = request
        .aspects
        .iter()
        .flat_map(|a| a.parts.iter())
        .filter(|p| p.form == Form::Identifier)
        .flat_map(|p| p.words.iter())
        .filter(|w| !w.contains(['/', '.']))
        .filter_map(|w| w.rsplit("::").next().map(str::to_lowercase))
        .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_alphanumeric() || c == '_'))
        .collect();
    let facts: Vec<(usize, Vec<usize>)> = request
        .relations
        .iter()
        .enumerate()
        .filter(|(_, r)| **r == Asks::Callers && !names.is_empty())
        .map(|(k, _)| (k, records.iter().enumerate().filter(|(_, r)| names.iter().any(|n| r.calls(n))).map(|(i, _)| i).collect::<Vec<usize>>()))
        .filter(|(_, v)| !v.is_empty())
        .collect();
    Ok(Evidence {
        aspects: request.aspects.iter().map(|a| a.text.clone()).collect(),
        value,
        band,
        surprise,
        nulls,
        stems: request.aspects.iter().map(|a| a.stems.clone()).collect(),
        share,
        parts,
        settle: settle_v,
        settle_mass,
        settle_base,
        settle_base_class,
        concentration,
        concentration_null,
        holders,
        association,
        association_null,
        sizes,
        settle_why,
        facts,
    })
}

/// Section check: a two-dimensional table and a character tokenizer over a small material; the
/// piece defining the asked identifier is accepted and tagged as defining it, a piece that only
/// mentions it in a comment is tagged as a mention, an aspect the material does not hold accepts
/// no piece, the run repeats exactly, and the null words are words of the material.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    use crate::intake::{Bf16Table, Kind, Source};
    struct Chars;
    impl crate::intake::Tokens for Chars {
        fn tokens(&self, text: &str) -> Result<Vec<i32>, Fault> {
            Ok(text.bytes().map(|b| i32::from(b % 4)).collect())
        }
    }
    let table = Bf16Table::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![-1.0, 0.0]]);
    let drone = Drone::new(std::sync::Arc::new(Chars), table);
    let texts = [
        "pub fn growth_limit(x: usize) -> usize {\n    x * 2\n}\n",
        "fn caller() {\n    // growth_limit is read here\n    let y = 3;\n}\n",
        "pub struct Ledger { pub rows: Vec<u8> }\n",
        "fn other_thing() { let rows = 1; }\n",
        "fn table_walk() { let span = 2; }\n",
        "fn header_line() { let index = 4; }\n",
    ];
    let mut recs = Vec::new();
    for (i, t) in texts.iter().enumerate() {
        let r = drone
            .take_in(Kind::Code, Source::Lines { path: format!("src/m{i}.rs"), start: 1, end: 3 }, i + 1, t)
            .map_err(|e| e.to_string())?
            .into_inner();
        recs.push(r);
    }
    let rarity = Rarity::from_records(recs.iter());
    let places: Vec<Place> = recs.iter().map(|r| Place { tokens: r.len(), class: Class::Code, first: true }).collect();
    let req = Request::read("Where is growth_limit computed, and what is the zqxv wobble?", &[]);
    let ev = sense(&drone, "q", &req, &recs, &rarity, &places, None).map_err(|e| e.to_string())?;
    let again = sense(&drone, "q", &req, &recs, &rarity, &places, None).map_err(|e| e.to_string())?;
    if ev.aspects.len() != 2 || ev.value != again.value || ev.band != again.band {
        return Err(format!("nmpod.evidence: aspects {:?} or a run that does not repeat", ev.aspects));
    }
    if !ev.accepted(0, 0) || ev.piece(0) <= ev.piece(2) {
        return Err(format!("nmpod.evidence: the defining piece is not accepted ({} against {})", ev.piece(0), ev.piece(2)));
    }
    if (0..recs.len()).any(|i| (0..SIGNALS.len()).filter(|s| FAMILY.get(*s) != Some(&Family::Meaning)).any(|s| ev.excess(1, s, i) > 0.0)) {
        return Err("nmpod.evidence: an aspect the material does not hold accepted a piece by its words".into());
    }
    let stems: Vec<String> = req.aspects.first().map(|a| a.stems.clone()).unwrap_or_default();
    let (t0, t1) = (recs.first().map(|r| tag(r, &stems, false)), recs.get(1).map(|r| tag(r, &stems, false)));
    if t0 != Some(Tag::Defines) || t1 != Some(Tag::Mentions) {
        return Err(format!("nmpod.evidence: tags {t0:?} {t1:?}"));
    }
    let vocab: Vec<String> = recs.iter().flat_map(|r| r.bag.keys().cloned()).collect();
    if ev.nulls.iter().flatten().flat_map(|w| w.split('_')).any(|w| w.is_empty() || !vocab.iter().any(|v| v == w)) {
        return Err(format!("nmpod.evidence: null words outside the material {:?}", ev.nulls));
    }
    // The meaning signal: a piece whose cosine stands far above the material's is accepted by
    // meaning alone, the others are not, and the words' signals are as without it.
    let cos = vec![vec![0.9_f32, 0.30, 0.31, 0.29, 0.32, f32::NAN], vec![0.30, 0.31, 0.30, 0.30, 0.31, f32::NAN], vec![0.31, 0.30, 0.31, 0.29, 0.30, f32::NAN], vec![0.30, 0.30, 0.29, 0.31, 0.31, f32::NAN]];
    let lane = sense(&drone, "q", &req, &recs, &rarity, &places, Some(&cos)).map_err(|e| e.to_string())?;
    let m = SIGNALS.iter().position(|n| *n == "meaning").unwrap_or(0);
    if lane.excess(1, m, 0) <= 0.0 || (1..recs.len()).any(|i| lane.excess(1, m, i) > 0.0) || (0..m).any(|s| lane.value[0][s] != ev.value[0][s]) {
        return Err("nmpod.evidence: the meaning signal accepts the wrong pieces or moves the words' signals".into());
    }
    // The describes role: an item whose attached comment holds the part's word holds it as its
    // definition lines would; the same text with no attached comment holds it only anywhere.
    let item = "/// Grows the ledger.
fn step() {}
";
    let src = || Source::Lines { path: "src/s.rs".into(), start: 1, end: 2 };
    let described = drone.take_in_described(Kind::Code, src(), 1, item, "", "/// Grows the ledger.").map_err(|e| e.to_string())?.into_inner();
    let plain = drone.take_in(Kind::Code, src(), 1, item).map_err(|e| e.to_string())?.into_inner();
    let part = [(nmlex::stem("ledger"), 1.0_f32)];
    let (hd, hp) = (hold(&described, false, &part, &[]), hold(&plain, false, &part, &[]));
    if hd != [1.0, 1.0, 1.0] || hp != [0.0, 0.0, 1.0] {
        return Err(format!("nmpod.evidence: the describes role holds {hd:?} described, {hp:?} plain"));
    }
    Ok(())
}
