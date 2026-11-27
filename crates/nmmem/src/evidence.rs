//! Memory's evidence for one request: the pieces that verified outcomes of like requests rest on,
//! the pieces used together with them, and the refuted ones, each read fresh before it is handed
//! on. This is the memory family of evidence for the drones and the seeds for allocation; the
//! context pack takes it as recalled reads.
//!
//! Rules:
//! - Only outcomes whose newest verdict is verified are evidence for a piece. An outcome whose
//!   newest verdict is refuted comes back as `Refuted`, to be shown as refuted and never served.
//! - A request matches an outcome on the processor, twice and apart: the share of the request's
//!   key terms (weighted by their rarity in the material) among the stored question's terms
//!   (request against request), and the share among the names and words of the outcome's places as
//!   they read now (material against the request). An outcome is recalled when either share stands
//!   out from the same share over the other outcomes; its strength is the higher one that does.
//!   Nothing here loads a model; recall by meaning is the caller's, when its engines are open.
//! - Every place is read fresh through the caller's reader. A place whose lines no longer read
//!   back (changed or gone) is returned with `fresh` false and no text.
//! - Recall always completes: no deadline skips it (context is never cut for a time); its time is
//!   measured and reported by the caller.
//! - What counts as a match is derived per request from its own null (`recalled`).
use crate::outcomes::{Known, Place};
use nmtype::class::{FromMaterial, FromRequest};
use std::collections::{BTreeMap, BTreeSet};

/// The stated fallback when a request's own null cannot decide (too few other stored items, or a
/// request too short for a match to stand out from them): half of the request's rarity-weighted
/// terms.
pub const FALLBACK_FLOOR: f32 = 0.5;
/// Robust deviations above the request's null a match must stand to be recalled (the conventional
/// outlier band, as the recent-gain rule of muscle memory uses).
pub const NULL_BAND: f32 = 3.0;
/// Other stored items a null needs (fewer: the fallback decides).
pub const NULL_MIN: usize = 5;
/// The median absolute deviation of a normal distribution in units of its standard deviation.
const MAD_SCALE: f32 = 1.4826;

/// Whether a match is recalled, derived from the request's own null: `others` are the matches of
/// the request with every other stored item, which it is mostly not about. A match is recalled when
/// it stands at least `NULL_BAND` robust deviations above their median, a deviation never taken
/// finer than the request's resolution: the share its lightest term carries (`resolution`), so a
/// null of identical matches cannot make a one-term difference stand out. When the null cannot
/// decide (fewer than `NULL_MIN` others, or even a full match could not stand out at this
/// resolution), the stated fallback `FALLBACK_FLOOR` decides.
#[must_use]
pub fn recalled(share: f32, others: &[f32], resolution: f32) -> bool {
    if others.len() < NULL_MIN {
        return share >= FALLBACK_FLOOR;
    }
    let mut o = others.to_vec();
    let med = median(&mut o);
    let mut dev: Vec<f32> = others.iter().map(|x| (x - med).abs()).collect();
    let spread = (MAD_SCALE * median(&mut dev)).max(resolution);
    let bar = med + NULL_BAND * spread;
    if bar > 1.0 {
        return share >= FALLBACK_FLOOR;
    }
    share >= bar
}

/// Whether `x` stands out from `others` by the request's own null: at least `NULL_BAND` robust
/// deviations above their median; `None` when there are fewer than `NULL_MIN` others to judge by.
#[must_use]
pub fn above_null(x: f32, others: &[f32]) -> Option<bool> {
    if others.len() < NULL_MIN {
        return None;
    }
    let mut o = others.to_vec();
    let med = median(&mut o);
    let mut dev: Vec<f32> = others.iter().map(|v| (v - med).abs()).collect();
    Some(x >= med + NULL_BAND * MAD_SCALE * median(&mut dev))
}

/// The bar a value must reach to stand out from `scores` (median plus `NULL_BAND` robust
/// deviations); `None` with fewer than `NULL_MIN` scores.
#[must_use]
pub fn null_bar(scores: &[f32]) -> Option<f32> {
    if scores.len() < NULL_MIN {
        return None;
    }
    let mut o = scores.to_vec();
    let med = median(&mut o);
    let mut dev: Vec<f32> = scores.iter().map(|v| (v - med).abs()).collect();
    Some(med + NULL_BAND * MAD_SCALE * median(&mut dev))
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    let n = v.len();
    match n {
        0 => 0.0,
        _ if n % 2 == 1 => v.get(n / 2).copied().unwrap_or(0.0),
        _ => (v.get(n / 2 - 1).copied().unwrap_or(0.0) + v.get(n / 2).copied().unwrap_or(0.0)) / 2.0,
    }
}

/// What memory has on a piece.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// A place a verified outcome of a like request rests on.
    Verified,
    /// A place used together with a `Verified` piece in another verified outcome.
    CoCited,
    /// A place of a like request's outcome whose newest verdict is refuted: shown, never served.
    Refuted,
}

impl Kind {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::CoCited => "co-cited",
            Self::Refuted => "refuted",
        }
    }
}

/// One piece memory has evidence on.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    /// Where the piece is now (moved lines carry their new numbers) and its hash when verified.
    pub place: Place,
    pub kind: Kind,
    /// The match of the request to the outcome (0 to 1); for a co-cited piece, that match scaled by
    /// how often the two places were used together.
    pub strength: f32,
    /// The outcome record the piece comes from.
    pub record: String,
    /// The outcome's question.
    pub question: String,
    /// Verifications and refutations of that outcome, and its strongest verification.
    pub verified: u32,
    pub refuted: u32,
    pub evidence: String,
    /// Whether the lines read back now as they were when verified; `text` holds them when they do.
    pub fresh: bool,
    pub text: Option<String>,
}

/// Memory's evidence for one request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Evidence {
    /// Strongest first; `Verified` and `CoCited` pieces only when fresh.
    pub pieces: Vec<Piece>,
    /// Outcomes examined.
    pub examined: usize,
    /// Matched places whose lines no longer read back.
    pub stale: usize,
    /// The match of the request to every answered outcome examined, by record (for diagnosis).
    pub shares: Vec<(String, f32)>,
}

impl Evidence {
    /// The pieces that may be served (verified or co-cited, fresh).
    pub fn served(&self) -> impl Iterator<Item = &Piece> {
        self.pieces.iter().filter(|p| p.kind != Kind::Refuted && p.fresh)
    }
    /// The refuted pieces.
    pub fn refuted(&self) -> impl Iterator<Item = &Piece> {
        self.pieces.iter().filter(|p| p.kind == Kind::Refuted)
    }
}

/// A place read fresh: where its lines are now (the same or moved) and the lines, as material.
pub type Fresh = (Place, FromMaterial<String>);
/// How memory reads a place fresh, or `None` when its lines no longer read back.
pub type Reader<'a> = &'a dyn Fn(&Place) -> Option<Fresh>;

/// A request as memory matches it.
pub struct Request<'a> {
    pub text: &'a str,
    /// Rarity of a term in the material (the substrate's inverse document frequency).
    pub idf: &'a dyn Fn(&str) -> f32,
    pub read: Reader<'a>,
}

/// The stems of an earlier request's key terms (request data, compared only with the new request).
fn request_stems(q: &FromRequest<&str>) -> BTreeSet<String> {
    nmlex::key_terms(q.get()).into_iter().map(|w| nmlex::stem(&w)).collect()
}

/// The stems of a place's names and words: each word and its name parts, and the path's parts
/// (material data, compared only with the new request).
fn place_stems(path: &str, text: &FromMaterial<String>) -> BTreeSet<String> {
    let text = text.get().as_str();
    let mut out = BTreeSet::new();
    for w in nmlex::words(text).into_iter().chain(path.split(['/', '.']).map(str::to_lowercase)) {
        out.insert(nmlex::stem(&w));
    }
    for w in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        for p in nmlex::name_parts(w) {
            out.insert(nmlex::stem(&p));
        }
    }
    out
}

/// Stems of the places read in the latest call, by path and the text's hash (RAM only): a place
/// whose lines did not change is not stemmed again. Each call keeps exactly the places it read.
type StemMemo = BTreeMap<(String, u64), BTreeSet<String>>;
static STEMS: std::sync::Mutex<StemMemo> = std::sync::Mutex::new(BTreeMap::new());

fn text_key(path: &str, text: &FromMaterial<String>) -> (String, u64) {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.get().hash(&mut h);
    (path.to_owned(), h.finish())
}

fn overlaps(a: &Place, b: &Place) -> bool {
    a.path == b.path && a.start <= b.end && b.start <= a.end
}

/// Memory's evidence for `req` from the outcomes memory knows (`outcomes::known_in`, or `known`).
#[must_use]
pub fn evidence(known: &[Known], req: &Request) -> Evidence {
    let asked: Vec<String> = nmlex::key_terms(req.text);
    let weights: Vec<(String, f32)> = asked.iter().map(|t| (nmlex::stem(t), (req.idf)(t))).collect();
    let total: f32 = weights.iter().map(|w| w.1).sum();
    let mut out = Evidence::default();
    if total <= 0.0 {
        return out;
    }
    // Places read once per call: (record, place) to the fresh place and its lines.
    let mut read: BTreeMap<(String, String), Option<Fresh>> = BTreeMap::new();
    // Per answered outcome: (share on its stored question, share on its places, outcome).
    let mut candidates: Vec<(f32, f32, &Known)> = Vec::new();
    let (mut on_question, mut on_places): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
    let mut memo = STEMS.lock().map(|mut m| std::mem::take(&mut *m)).unwrap_or_default();
    let mut used: StemMemo = BTreeMap::new();
    for k in known.iter().filter(|k| k.kind == "answered") {
        out.examined += 1;
        let Some(q) = k.question.as_deref() else { continue };
        // The stored question and the places are matched apart: request against request, material
        // against the request.
        let qs = request_stems(&FromRequest::new(q));
        // The outcome's places, as the keys of their stems in `used`.
        let mut keys: Vec<(String, u64)> = Vec::new();
        for p in &k.places {
            let r = read.entry((k.record.clone(), p.label())).or_insert_with(|| (req.read)(p));
            if let Some((now, text)) = r {
                let key = text_key(&now.path, text);
                if !used.contains_key(&key) {
                    let x = memo.remove(&key).unwrap_or_else(|| place_stems(&now.path, text));
                    used.insert(key.clone(), x);
                }
                keys.push(key);
            }
        }
        let on_places_has = |s: &String| keys.iter().any(|k| used.get(k).is_some_and(|x| x.contains(s)));
        let (mut on_q, mut on_p, mut n) = (0.0_f32, 0.0_f32, 0_usize);
        for (s, w) in &weights {
            let (a, b) = (qs.contains(s), on_places_has(s));
            on_q += if a { *w } else { 0.0 };
            on_p += if b { *w } else { 0.0 };
            n += usize::from(a || b);
        }
        let (on_q, on_p) = (on_q / total, on_p / total);
        out.shares.push((k.record.clone(), on_q.max(on_p)));
        on_question.push(on_q);
        on_places.push(on_p);
        if n >= weights.len().min(2) {
            candidates.push((on_q, on_p, k));
        }
    }
    if let Ok(mut m) = STEMS.lock() {
        *m = used;
    }
    // An outcome is recalled when either of its shares stands out from the same share over the other
    // outcomes (`recalled`, at the resolution of the request's lightest term); its strength is the
    // higher share that does.
    let resolution = weights.iter().map(|w| w.1).filter(|w| *w > 0.0).fold(f32::INFINITY, f32::min) / total;
    let stands_out = |share: f32, all: &[f32]| {
        let mut others: Vec<f32> = all.to_vec();
        if let Some(i) = others.iter().position(|x| (*x - share).abs() < f32::EPSILON) {
            others.swap_remove(i);
        }
        recalled(share, &others, resolution)
    };
    let mut matched: Vec<(f32, &Known)> = candidates
        .into_iter()
        .filter_map(|(q, p, k)| {
            let a = stands_out(q, &on_question).then_some(q);
            let b = stands_out(p, &on_places).then_some(p);
            a.into_iter().chain(b).reduce(f32::max).map(|s| (s, k))
        })
        .collect();
    matched.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.record.cmp(&b.1.record)));
    let add = |out: &mut Evidence, piece: Piece| match out.pieces.iter_mut().find(|x| x.kind == piece.kind && overlaps(&x.place, &piece.place)) {
        Some(x) if piece.strength > x.strength => *x = piece,
        Some(_) => {}
        None => out.pieces.push(piece),
    };
    for (share, k) in &matched {
        let kind = if k.standing { Kind::Verified } else { Kind::Refuted };
        for p in &k.places {
            let r = read.get(&(k.record.clone(), p.label())).cloned().flatten();
            if r.is_none() && kind != Kind::Refuted {
                out.stale += 1;
            }
            let piece = Piece {
                place: r.as_ref().map_or_else(|| p.clone(), |x| x.0.clone()),
                kind,
                strength: *share,
                record: k.record.clone(),
                question: k.question.clone().unwrap_or_default(),
                verified: k.verified,
                refuted: k.refuted,
                evidence: k.strength.clone(),
                fresh: r.is_some(),
                text: r.map(|x| x.1.into_inner()),
            };
            if kind == Kind::Refuted || piece.fresh {
                add(&mut out, piece);
            }
        }
    }
    // Co-citations: a place verified together with a verified piece found here, in another
    // verified outcome, scaled by how often the two were used together.
    let links = cocitations(known);
    let found: Vec<Piece> = out.pieces.iter().filter(|p| p.kind == Kind::Verified).cloned().collect();
    for v in &found {
        for (a, b, count) in &links {
            let other = if overlaps(a, &v.place) {
                b
            } else if overlaps(b, &v.place) {
                a
            } else {
                continue;
            };
            if out.pieces.iter().any(|x| x.kind == Kind::Verified && overlaps(&x.place, other)) {
                continue;
            }
            let Some((now, text)) = (req.read)(other) else { continue };
            let text = text.into_inner();
            #[expect(clippy::cast_precision_loss, reason = "small counts")]
            let strength = v.strength * (*count as f32 / (*count as f32 + 1.0));
            let piece = Piece { place: now, kind: Kind::CoCited, strength, fresh: true, text: Some(text), ..v.clone() };
            add(&mut out, piece);
        }
    }
    out.pieces.sort_by(|a, b| a.kind.cmp(&b.kind).then(b.strength.total_cmp(&a.strength)));
    out
}

/// Places used together in verified outcomes: each pair once (by path and lines), with how many
/// verified outcomes used both. The reference graph's co-citation edges.
#[must_use]
pub fn cocitations(known: &[Known]) -> Vec<(Place, Place, u32)> {
    let mut pairs: BTreeMap<(String, String), (Place, Place, u32)> = BTreeMap::new();
    for k in known.iter().filter(|k| k.standing && k.verified > 0) {
        for (i, a) in k.places.iter().enumerate() {
            for b in k.places.iter().skip(i + 1) {
                let (x, y) = if a.label() <= b.label() { (a, b) } else { (b, a) };
                pairs.entry((x.label(), y.label())).or_insert_with(|| (x.clone(), y.clone(), 0)).2 += 1;
            }
        }
    }
    pairs.into_values().collect()
}

/// Section check: a repeated and a reworded request find a verified outcome, an unrelated one does
/// not; a request that matches only through the place's names is found at the place share; a
/// refuted outcome comes back as refuted; a changed place is stale and not served; a co-cited place
/// comes with its partner; a deadline already past marks the result late.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    let place = |path: &str, a: usize, b: usize| Place { path: path.into(), start: a, end: b, hash: format!("h-{path}-{a}") };
    let known = |rec: &str, q: &str, places: Vec<Place>, standing: bool| Known {
        record: rec.into(),
        kind: "answered".into(),
        question: Some(q.into()),
        failure: None,
        places,
        verified: u32::from(standing),
        refuted: u32::from(!standing),
        strength: "indirect".into(),
        standing,
        in_memory: true,
    };
    let total = place("src/order.rs", 10, 20);
    let tax = place("src/tax.rs", 1, 9);
    let ks = vec![
        known("oc:a", "where is the order total computed", vec![total.clone()], true),
        known("oc:b", "how is the order total taxed", vec![total.clone(), tax.clone()], true),
        known("oc:c", "which function parses the config file", vec![place("src/config.rs", 1, 30)], false),
        known("oc:d", "where are refunds booked", vec![place("src/refund.rs", 5, 8)], true),
    ];
    let texts: BTreeMap<String, String> = [
        ("src/order.rs", "pub fn order_total(items: &[Item]) -> u64 {\n    items.iter().map(price).sum()\n}"),
        ("src/tax.rs", "pub fn apply_tax(total: u64) -> u64 { total * 120 / 100 }"),
        ("src/config.rs", "pub fn parse_config(text: &str) -> Config { todo() }"),
    ]
    .into_iter()
    .map(|(a, b)| (a.to_owned(), b.to_owned()))
    .collect();
    // `src/refund.rs` changed since it was verified: it does not read back.
    let read = |p: &Place| texts.get(&p.path).map(|t| (p.clone(), FromMaterial::new(t.clone())));
    let idf = |_: &str| 1.0_f32;
    let ask = |q: &str| evidence(&ks, &Request { text: q, idf: &idf, read: &read });
    let e = ask("where is the order total computed");
    let first = e.served().next().ok_or("evidence: a repeated request found nothing")?;
    if first.record != "oc:a" || (first.strength - 1.0).abs() > 1e-6 || first.text.is_none() {
        return Err(format!("evidence: repeated request gave {first:?}"));
    }
    if !e.pieces.iter().any(|p| p.kind == Kind::CoCited && p.place.path == "src/tax.rs") && !e.served().any(|p| p.place.path == "src/tax.rs") {
        return Err("evidence: the place used together with the answer is missing".into());
    }
    let r = ask("which code computes the total of an order");
    if !r.served().any(|p| p.place.path == "src/order.rs") {
        return Err("evidence: a reworded request found nothing".into());
    }
    if !ask("rename the logging macro").pieces.is_empty() {
        return Err("evidence: an unrelated request found pieces".into());
    }
    let refuted = ask("which function parses the config file");
    if refuted.served().count() != 0 || refuted.refuted().next().map(|p| p.record.as_str()) != Some("oc:c") {
        return Err("evidence: a refuted outcome was served or not shown as refuted".into());
    }
    let stale = ask("where are refunds booked");
    if stale.served().count() != 0 || stale.stale != 1 {
        return Err(format!("evidence: a changed place was served ({} served, {} stale)", stale.served().count(), stale.stale));
    }
    let names = ask("apply_tax total");
    if !names.served().any(|p| p.place.path == "src/tax.rs") {
        return Err("evidence: a request naming what a place defines found nothing".into());
    }
    // The null decides when it can: a match standing out from many unrelated items is recalled,
    // one in their bulk is not; a full match a short request cannot raise above its null falls back.
    let null = [0.0_f32, 0.1, 0.0, 0.1, 0.0, 0.1];
    let zeros = [0.0_f32; 12];
    if !recalled(0.8, &null, 0.125)
        || recalled(0.2, &null, 0.125)
        || !recalled(1.0, &null, 0.5)
        || recalled(0.4, &null, 0.5)
        || !recalled(0.5, &[0.1], 0.3)
        || !recalled(0.7, &zeros, 0.15)
        || recalled(0.3, &zeros, 0.15)
    {
        return Err("evidence: the recall rule from the request's null".into());
    }
    if cocitations(&ks).iter().map(|x| x.2).sum::<u32>() != 1 {
        return Err("evidence: co-citations of verified outcomes".into());
    }
    Ok(())
}
