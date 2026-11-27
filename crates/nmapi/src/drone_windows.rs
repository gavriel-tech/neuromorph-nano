//! Tentacle windows from drone flags, build step T1 (2026-11-10; Gavriel Technologies,
//! architecture by Luis Jake Gabriel III; design row 2 and build step 2 of
//! `docs/2026-11-10-neumor-design-anatomy-components-strategy.md`). Selected with
//! `--tentacle-windows drones`; off by default (`WindowMode::Lens`, today's eight lens windows).
//!
//! 1. The drones sense every chunk of the tree against the request (`nmpod::intake`: exact,
//!    split-identifier, structural and token-affinity signals fused by reciprocal rank). Every
//!    chunk gets a flag strength (its fused score); none is dropped.
//! 2. Strong flags: a flag at or above the mean plus one standard deviation of the flag
//!    strengths over the chunks with any signal (derived per request, no fixed count).
//! 3. The windows are filled one after the other, without overlap and up to the full window
//!    (`lens::BUDGET` tokens each, first fit: a piece too large for the room left is skipped,
//!    not cut, and the next one is tried). Each step takes the higher of two candidates: the
//!    next piece in the drone order (priority: its flag strength), or a piece related to a strong
//!    flag already in this window over the specific reference graph (priority: its own flag
//!    strength plus the flag's strength x `lens::HOP_DECAY` x the relation's weight relative to
//!    the flag's heaviest relation, with the graph's derived edge-kind weights; contributions of
//!    several flags add up, as in `lens::relation_window`). One hop: the relations are around
//!    the strong flags themselves. So window 1 holds the strongest flags and their connected
//!    pieces, window 2 the next, and so on: the windows are ranked by their strongest flag.
//! 4. The allocation map records who holds what (drone order as `Coverage`, related pieces as
//!    `Relation` with the flag they came from), the strong flags as each window's seeds, and
//!    every chunk not placed as a passed-over reference with its priority: nothing is dropped.
//!
//! Lens kinds are not used to compose these windows; a tentacle keeps its lens name only for its
//! pose (`tentacle_pass::FOCUS` under `--tentacle-pose focus`). RAM only; nothing is written.
use nmbrain::allocation::{AllocationMap, Mode as Take};
use nmpod::intake::{self, Bf16Table, Drone, Kind, Rarity, Record, Source};
use nmsubs::Substrate;
use nmsubs::lens::HOP_DECAY;
use nmsubs::refgraph::Specific;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.drone_windows", FaultKind::Execution, m)
}

/// How the tentacle windows are composed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WindowMode {
    /// Today's eight lens windows (`nmsubs::lens::assign`).
    #[default]
    Lens,
    /// From the drones' flags plus relations (this module).
    Drones,
}
impl WindowMode {
    /// `lens` or `drones`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "lens" => Some(Self::Lens),
            "drones" => Some(Self::Drones),
            _ => None,
        }
    }
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Lens => "lens",
            Self::Drones => "drones",
        }
    }
}

/// Windows composed from drone flags.
#[derive(Clone, Debug, Default)]
pub struct Composed {
    /// Per window its chunks in the order taken; window 0 holds the strongest flag.
    pub windows: Vec<Vec<usize>>,
    pub map: AllocationMap,
    /// The strong-flag threshold and the strong flags placed.
    pub threshold: f32,
    pub strong: usize,
    /// Chunks taken by a relation to a strong flag (the rest by drone order).
    pub related: usize,
}

/// Mean plus one standard deviation of the flag strengths above zero (0 when none).
#[must_use]
pub fn strong_threshold(fused: &[f32]) -> f32 {
    let v: Vec<f64> = fused.iter().filter(|x| **x > 0.0).map(|x| f64::from(*x)).collect();
    if v.is_empty() {
        return 0.0;
    }
    #[expect(clippy::cast_precision_loss, reason = "chunk counts are small")]
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    #[expect(clippy::cast_possible_truncation, reason = "a score")]
    let t = (mean + var.sqrt()) as f32;
    t
}

/// Compose `tentacles` windows of at most `budget` tokens from the flag strengths `fused` (one
/// per chunk) and the specific relations `graph` (module notes).
#[must_use]
pub fn compose(fused: &[f32], graph: &Specific, tokens: &dyn Fn(usize) -> usize, budget: usize, tentacles: usize) -> Composed {
    compose_with(fused, graph, tokens, budget, tentacles, None)
}

/// The state of one composition while windows fill.
struct Fill<'a> {
    fused: &'a [f32],
    graph: &'a Specific,
    kw: [f32; nmsubs::refgraph::KINDS],
    threshold: f32,
    map: AllocationMap,
    placed: Vec<bool>,
    out: Composed,
    rooms: Vec<usize>,
    /// Per window, related candidates: chunk to (relation priority, flag it came from).
    nears: Vec<BTreeMap<usize, (f32, usize)>>,
    seeds: Vec<Vec<usize>>,
}
impl Fill<'_> {
    fn f(&self, c: usize) -> f32 {
        self.fused.get(c).copied().unwrap_or(0.0)
    }
    fn is_placed(&self, c: usize) -> bool {
        self.placed.get(c).copied().unwrap_or(true)
    }
    /// Place chunk `c` (of `t` tokens) in window `w`; `from` names the flag a relation came from.
    fn place(&mut self, w: usize, c: usize, from: Option<usize>, t: usize) {
        if !self.map.claim(w, c, if from.is_some() { Take::Relation } else { Take::Coverage }, from, t, false) {
            // Cannot happen (only unplaced chunks are offered); mark it so the caller moves on.
            if let Some(p) = self.placed.get_mut(c) {
                *p = true;
            }
            return;
        }
        if let Some(p) = self.placed.get_mut(c) {
            *p = true;
        }
        if let Some(r) = self.rooms.get_mut(w) {
            *r = r.saturating_sub(t);
        }
        if let Some(n) = self.nears.get_mut(w) {
            n.remove(&c);
        }
        if let Some(src) = from {
            self.map.follow(w, src, c);
            self.out.related += 1;
        }
        if let Some(win) = self.out.windows.get_mut(w) {
            win.push(c);
        }
        let fc = self.f(c);
        if fc >= self.threshold && fc > 0.0 {
            if let Some(s) = self.seeds.get_mut(w) {
                s.push(c);
            }
            self.out.strong += 1;
            let edges = self.graph.edges_of(c);
            let top = edges.iter().map(|e| e.weight * self.kw.get(e.kind.index()).copied().unwrap_or(1.0)).fold(0.0_f32, f32::max);
            if top > 0.0 {
                for e in edges {
                    let to = e.to as usize;
                    if self.is_placed(to) {
                        continue;
                    }
                    let add = fc * HOP_DECAY * e.weight * self.kw.get(e.kind.index()).copied().unwrap_or(1.0) / top;
                    if let Some(n) = self.nears.get_mut(w) {
                        let x = n.entry(to).or_insert((0.0, c));
                        x.0 += add;
                    }
                }
            }
        }
    }
}

/// `compose`, with an optional deal of the strong files across the tentacles (2026-11-10).
/// `file_of` (chunk to file) switches the deal on: the files holding a strong flag are ranked by
/// their strongest flag and dealt in serpentine order (windows 1 to n, then n to 1, and so on),
/// and each window first takes the strong pieces of its own files, so every tentacle starts from
/// strong material instead of the first window taking it all. Then every window fills its
/// remaining room as `compose` does (drone order plus relations), so windows stay full and no
/// chunk is in two. Without `file_of` the result is `compose`'s, unchanged.
/// Method: larger unit first (Zoekt ranks files, then lines, Apache-2.0, method only); spread
/// in proportion to strength (honeybee recruitment, Seeley et al. 2000); research record
/// `docs/2026-11-10-neumor-research-protocol-pass-all-parts.md`, part 3 and solution 4.
#[must_use]
pub fn compose_with(
    fused: &[f32],
    graph: &Specific,
    tokens: &dyn Fn(usize) -> usize,
    budget: usize,
    tentacles: usize,
    file_of: Option<&dyn Fn(usize) -> usize>,
) -> Composed {
    let n = fused.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| fused.get(*b).copied().unwrap_or(0.0).total_cmp(&fused.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
    let threshold = strong_threshold(fused);
    let mut st = Fill {
        fused,
        graph,
        kw: graph.kind_weights(),
        threshold,
        map: AllocationMap::new(n, tentacles),
        placed: vec![false; n],
        out: Composed { windows: vec![Vec::new(); tentacles], threshold, ..Composed::default() },
        rooms: vec![budget; tentacles],
        nears: vec![BTreeMap::new(); tentacles],
        seeds: vec![Vec::new(); tentacles],
    };
    // The deal: strong pieces go to the window their file was dealt to, in drone order.
    if let (Some(file_of), true) = (file_of, tentacles > 0) {
        let mut dealt: BTreeMap<usize, usize> = BTreeMap::new();
        for &c in &order {
            let fc = st.f(c);
            if fc < threshold || fc <= 0.0 {
                break;
            }
            let k = dealt.len();
            let w = *dealt.entry(file_of(c)).or_insert_with(|| {
                let (round, pos) = (k / tentacles, k % tentacles);
                if round % 2 == 0 { pos } else { tentacles - 1 - pos }
            });
            let t = tokens(c);
            if !st.is_placed(c) && st.rooms.get(w).is_some_and(|r| t <= *r) {
                st.place(w, c, None, t);
            }
        }
    }
    let mut head = 0_usize;
    for w in 0..tentacles {
        loop {
            while order.get(head).is_some_and(|c| st.is_placed(*c)) {
                head += 1;
            }
            let room = st.rooms.get(w).copied().unwrap_or(0);
            let drone = order.iter().skip(head).copied().find(|c| !st.is_placed(*c) && tokens(*c) <= room);
            let rel = st.nears.get(w).and_then(|near| {
                near.iter()
                    .filter(|(c, _)| !st.is_placed(**c) && tokens(**c) <= room)
                    .map(|(c, (p, from))| (*c, st.f(*c) + p, *from))
                    .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
            });
            let (c, from) = match (drone, rel) {
                (Some(d), Some(r)) if r.1 > st.f(d) => (r.0, Some(r.2)),
                (Some(d), _) => (d, None),
                (None, Some(r)) => (r.0, Some(r.2)),
                (None, None) => break,
            };
            st.place(w, c, from, tokens(c));
            if st.rooms.get(w).copied().unwrap_or(0) == 0 {
                break;
            }
        }
        let left: Vec<(usize, f32)> = st
            .nears
            .get(w)
            .map(|near| near.iter().filter(|(c, _)| !st.is_placed(**c)).map(|(c, (p, _))| (*c, st.f(*c) + p)).collect())
            .unwrap_or_default();
        for (c, v) in left {
            st.map.pass_over(c, v);
        }
        let seeds = st.seeds.get(w).cloned().unwrap_or_default();
        st.map.label(w, "drones", &seeds);
    }
    for c in 0..n {
        if !st.is_placed(c) {
            let v = st.f(c);
            st.map.pass_over(c, v);
        }
    }
    let mut out = st.out;
    out.map = st.map;
    out
}

/// The drone and the tree as it took it in, kept in RAM while the chunks are unchanged.
pub struct Keep {
    drone: Drone,
    keys: Vec<String>,
    records: Vec<Record>,
    rarity: Rarity,
    /// Each piece's nuisance (tokens, kind class, first in its file) for the evidence's robust fit.
    places: Vec<nmpod::evidence::Place>,
    /// The material's items (`nmsubs::items`), their records, rarity and nuisance: the unit of
    /// belief when items are on (`items_mode`); built once per material version, in RAM.
    items: Arc<nmsubs::items::Items>,
    item_records: Vec<Record>,
    item_rarity: Rarity,
    item_places: Vec<nmpod::evidence::Place>,
    item_digest: String,
    /// Memory areas (2026-11-27): which items are taken in (a slot not taken in holds a
    /// placeholder record of its path only), the bytes the records hold, and the last step's state.
    item_taken: Vec<bool>,
    /// A slot not taken in yet counts the tokens its text will have (characters / 4), so the
    /// windows and budgets that read a piece's size see its real size, not the placeholder's.
    est: Vec<usize>,
    item_est: Vec<usize>,
    held: usize,
    item_held: usize,
    state: KeepState,
}

/// Where the drones' intake stands after its last step (memory areas, 2026-11-27).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeepState {
    /// Pieces of the material taken in, and still waiting (their slot holds a placeholder record
    /// of the piece's path, so every index stays aligned with the material's chunks).
    pub taken: usize,
    pub pending: usize,
    /// Items taken in and waiting, the same way.
    pub items_taken: usize,
    pub items_pending: usize,
    /// The last step stopped because the `content` memory area is full.
    pub full: bool,
}

impl KeepState {
    /// Nothing waits: every piece and item is taken in.
    #[must_use]
    pub fn complete(&self) -> bool {
        self.pending == 0 && self.items_pending == 0
    }
}

/// Pieces one bounded intake step takes in at most (memory areas, 2026-11-27): about half a
/// second of the processor's work on a 12-thread machine, so a call waits at most one step.
pub const STEP_PIECES: usize = 1_024;

/// Whether the `content` memory area has no room left (`nmengn::limits::area_room`).
#[must_use]
pub fn content_full() -> bool {
    nmengn::limits::area_room("content") == Some(0)
}

/// How many of `batch` fit `left` bytes of the content area, by the estimate `est` of what each
/// adds (2026-11-27: a step checks the room before it reads, so the area cannot pass its budget
/// by a step). Every record is still admitted only while it fits (`admit`).
fn fits(batch: &[usize], left: u64, est: impl Fn(usize) -> u64) -> usize {
    let mut sum = 0_u64;
    batch
        .iter()
        .take_while(|&&i| {
            sum = sum.saturating_add(est(i));
            sum <= left
        })
        .count()
}

/// Admits a record that grows the content area by `grows` bytes while it fits `left`; `false`
/// (and `full` set) when it does not, so the piece stays waiting.
fn admit(grows: u64, left: &mut u64, full: &mut bool) -> bool {
    if *full || grows > *left {
        *full = true;
        return false;
    }
    *left -= grows;
    true
}

/// Whether belief, order, the set and the pack are per item (`nmsubs::items`) or per piece.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemsMode {
    /// Per piece (the drones' intake cut), as before.
    Off,
    /// Belief and order per item; what is written stays the pieces.
    Order,
    /// Belief, order, the set and the pack per item.
    On,
}

/// The items mode from `NANO_ITEMS` (`off`, `order`, `on`); off when unset.
#[must_use]
pub fn items_mode() -> ItemsMode {
    match std::env::var("NANO_ITEMS").ok().as_deref() {
        Some("on") => ItemsMode::On,
        Some("order") => ItemsMode::Order,
        _ => ItemsMode::Off,
    }
}
impl std::fmt::Debug for Keep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Keep").field("chunks", &self.records.len()).finish_non_exhaustive()
    }
}
impl Keep {
    /// A drone with the release tokenizer and the central token embedding table of the weight
    /// folder `weights` (`tokenizer.json`, safetensors).
    /// # Errors
    /// A missing or unreadable file.
    pub fn open(weights: &Path) -> Result<Self, Fault> {
        let tok = nmtokz::Tokenizer::open(&weights.join("tokenizer.json")).map_err(|e| bad(format!("tokenizer: {e}")))?;
        let table = match Bf16Table::open(weights) {
            Ok(t) => t,
            Err(open_err) => crate::neural::embed_tokens_from_base(crate::neural::POPULATION).map_err(|_| open_err)?,
        };
        Ok(Self {
            drone: Drone::new(Arc::new(tok), table),
            keys: Vec::new(),
            records: Vec::new(),
            rarity: Rarity::default(),
            places: Vec::new(),
            items: Arc::new(nmsubs::items::Items::default()),
            item_records: Vec::new(),
            item_rarity: Rarity::default(),
            item_places: Vec::new(),
            item_digest: String::new(),
            item_taken: Vec::new(),
            est: Vec::new(),
            item_est: Vec::new(),
            held: 0,
            item_held: 0,
            state: KeepState::default(),
        })
    }
    /// Where the intake stands after its last step.
    #[must_use]
    pub fn state(&self) -> KeepState {
        self.state
    }
    /// The bytes the drones' records hold in RAM (pieces and items).
    #[must_use]
    pub fn held_bytes(&self) -> usize {
        self.held.saturating_add(self.item_held)
    }
    fn report_held(&self) {
        nmengn::limits::set_area_used("content", "drone_intake", self.held_bytes() as u64);
    }
    /// A placeholder record for a piece not taken in yet: its path only (empty text).
    fn placeholder(&self, path: &str, start: usize, end: usize, position: usize) -> Result<Record, Fault> {
        Ok(self.drone.take_in_with(Kind::Code, Source::Lines { path: path.to_owned(), start, end }, position, "", "")?.into_inner())
    }
    /// Build the items of `sub` and take each in through the drones, when the material changed
    /// since they were built (`Substrate::digest`); the rarity and nuisance are the items' own.
    /// An item is taken in as its own lines, a document section with its heading path, a
    /// definition with its attached comment as its description (`Drone::take_in_described`).
    /// Stops early when the `content` memory area is full (`item_step`).
    /// # Errors
    /// The tokenizer refuses a text.
    pub fn sync_items(&mut self, sub: &Substrate) -> Result<usize, Fault> {
        self.item_step(sub, usize::MAX)
    }
    /// One bounded step of the item intake (memory areas, 2026-11-27): when the material changed
    /// the items are built and every slot holds a placeholder record of its path; then at most
    /// `max` items not taken in yet are taken in, in batches of `STEP_PIECES`, stopping when the
    /// `content` memory area is full. Returns the items taken in by this step.
    /// # Errors
    /// The tokenizer refuses a text.
    pub fn item_step(&mut self, sub: &Substrate, max: usize) -> Result<usize, Fault> {
        let digest = sub.digest();
        if digest != self.item_digest {
            let items = nmsubs::items::build(sub);
            let list = &items.items;
            let made: Vec<Option<Result<Record, Fault>>> = nmtype::budget::run(list.len(), |i| {
                let it = list.get(i).ok_or_else(|| bad("item intake: an item was lost"))?;
                let f = sub.files.get(it.file).ok_or_else(|| bad("item intake: a file was lost"))?;
                self.placeholder(&f.path, it.start(), it.end(), i + 1)
            });
            let mut records = Vec::with_capacity(made.len());
            for r in made {
                records.push(r.ok_or_else(|| bad("item intake: a worker stopped"))??);
            }
            self.item_held = records.iter().map(Record::held_bytes).sum();
            self.item_taken = vec![false; records.len()];
            self.item_est = items.items.iter().map(|it| sub.files.get(it.file).map_or(1, |f| (it.text(&f.text).chars().count() / 4).max(1))).collect();
            self.item_records = records;
            self.items = Arc::new(items);
            self.item_digest = digest;
            self.item_places = Vec::new();
        }
        let todo: Vec<usize> = self.item_taken.iter().enumerate().filter(|(_, t)| !**t).map(|(i, _)| i).take(max).collect();
        let mut changed = 0_usize;
        let mut full = false;
        // Bytes held per item taken in so far estimate what each item of the next step adds.
        let mut seen = (self.item_held as u64, self.item_taken.iter().filter(|t| **t).count() as u64);
        for batch in todo.chunks(STEP_PIECES) {
            self.report_held();
            let room = nmengn::limits::area_room("content");
            if room == Some(0) {
                full = true;
                break;
            }
            let batch = match room {
                Some(left) if seen.1 > 0 => batch.get(..fits(batch, left, |_| seen.0 / seen.1)).unwrap_or(batch),
                _ => batch,
            };
            if batch.is_empty() {
                full = true;
                break;
            }
            let mut left = room.unwrap_or(u64::MAX);
            // Items dealt over the one processor budget (`nmtype::budget`, 2026-11-18 Phase C1):
            // each item is taken in alone, so the records equal one drone taking them in order.
            let drone = &self.drone;
            let list = &self.items.items;
            let take_one = |at: usize| -> Result<Record, Fault> {
                let i = batch.get(at).copied().ok_or_else(|| bad("item intake: an item was lost"))?;
                let it = list.get(i).ok_or_else(|| bad("item intake: an item was lost"))?;
                let f = sub.files.get(it.file).ok_or_else(|| bad("item intake: a file was lost"))?;
                let heading = if f.is_code() { String::new() } else { nmsubs::doc::heading_path_of(f, it.start()) };
                let r = drone
                    .take_in_described(
                        Kind::Code,
                        Source::Lines { path: f.path.clone(), start: it.start(), end: it.end() },
                        i + 1,
                        &it.text(&f.text),
                        &heading,
                        &it.described_text(&f.text),
                    )?
                    .into_inner();
                nmtype::class::inlets::MATERIAL_DRONES.admit(&r)?;
                Ok(r)
            };
            let taken: Vec<Option<Result<Record, Fault>>> = nmtype::budget::run(batch.len(), take_one);
            for (&i, r) in batch.iter().zip(taken) {
                let r = r.ok_or_else(|| bad("item intake: a worker stopped"))??;
                if let (Some(slot), Some(t)) = (self.item_records.get_mut(i), self.item_taken.get_mut(i)) {
                    if !admit((r.held_bytes() as u64).saturating_sub(slot.held_bytes() as u64), &mut left, &mut full) {
                        continue;
                    }
                    seen = (seen.0 + r.held_bytes() as u64, seen.1 + 1);
                    self.item_held = self.item_held.saturating_sub(slot.held_bytes()).saturating_add(r.held_bytes());
                    *slot = r;
                    *t = true;
                    if let Some(e) = self.item_est.get_mut(i) {
                        *e = 0;
                    }
                    changed += 1;
                }
            }
            if full {
                break;
            }
        }
        self.report_held();
        if changed > 0 || self.item_places.len() != self.item_records.len() {
            self.item_rarity = Rarity::from_records(self.item_records.iter());
            let kind_at: BTreeMap<(usize, usize), &'static str> = {
                let mut m = BTreeMap::new();
                for c in &sub.chunks {
                    m.entry((c.file, c.start)).or_insert(c.kind);
                }
                m
            };
            self.item_places = self
                .items
                .items
                .iter()
                .enumerate()
                .map(|(i, it)| {
                    let tokens = self.item_tokens(i);
                    let path = sub.files.get(it.file).map_or("", |f| f.path.as_str());
                    let kind = match it.kind {
                        nmsubs::items::Kind::Definition => "item",
                        nmsubs::items::Kind::Head => "notes",
                        nmsubs::items::Kind::Section => "paragraph",
                        nmsubs::items::Kind::Piece => kind_at.get(&(it.file, it.start())).copied().unwrap_or("window"),
                    };
                    nmpod::evidence::Place { tokens, class: nmpod::evidence::Class::of(kind, nmtype::is_document(path)), first: it.start() <= 1 }
                })
                .collect();
        }
        let waiting = self.item_taken.iter().filter(|t| !**t).count();
        self.state.items_taken = self.item_taken.len() - waiting;
        self.state.items_pending = waiting;
        self.state.full |= full;
        Ok(changed)
    }
    /// The items built by the last `sync_items`.
    #[must_use]
    pub fn items(&self) -> Arc<nmsubs::items::Items> {
        Arc::clone(&self.items)
    }
    /// The record of item `i` as the drones took it in.
    #[must_use]
    pub fn item_record(&self, i: usize) -> Option<&Record> {
        self.item_records.get(i)
    }
    /// Tokens of item `i` as the drone took it in.
    #[must_use]
    pub fn item_tokens(&self, i: usize) -> usize {
        match self.item_est.get(i) {
            Some(&e) if e > 0 => e,
            _ => self.item_records.get(i).map_or(0, Record::len),
        }
    }
    /// The drones' calibrated evidence of every item for `request`, as `evidence_with` reads every
    /// piece: each item scored from its own text by the same signals and roles, calibrated against
    /// the same content-free poses over the items. `meaning` (per piece, as `evidence_with` takes
    /// it) is credited to the items: an item meeting several pieces takes the strongest reading of
    /// each row, a piece meeting several items gives each its reading.
    /// # Errors
    /// As `evidence`.
    pub fn item_evidence(&self, request: &str, meaning: Option<&[Vec<f32>]>) -> Result<nmpod::evidence::Evidence, Fault> {
        let credited: Option<Vec<Vec<f32>>> = meaning.map(|rows| {
            rows.iter()
                .map(|row| {
                    self.items
                        .pieces_of
                        .iter()
                        .map(|ps| ps.iter().filter_map(|p| row.get(*p).copied()).filter(|x| x.is_finite()).fold(f32::NAN, |m, x| if m.is_nan() || x > m { x } else { m }))
                        .collect()
                })
                .collect()
        });
        let req = self.drone.take_request("request", request)?;
        let read = intake::Request::read(&req.get().text, &[]);
        nmpod::evidence::sense_sized(&self.drone, request, &read, &self.item_records, &self.item_rarity, &self.item_places, credited.as_deref(), true)
    }
    /// Take in every chunk of `sub` whose content changed (all on the first call); the rarity
    /// is recomputed when anything changed. The changed chunks are taken in by as many drones in
    /// parallel as there are processor cores, dealt in turn so that large and small pieces spread
    /// evenly; the records are kept in chunk order, equal to one drone taking them in one after
    /// another. Stops early, the rest waiting, when the `content` memory area is full
    /// (`sync_step`).
    /// # Errors
    /// The tokenizer refuses a text.
    pub fn sync(&mut self, sub: &Substrate) -> Result<usize, Fault> {
        self.sync_step(sub, &[], usize::MAX)
    }
    /// One bounded step of the drones' intake (memory areas, 2026-11-27). Every chunk of `sub`
    /// has a slot: a slot not taken in yet holds a placeholder record of the chunk's path, so the
    /// records stay aligned with the chunks and a call can read the material while the intake
    /// goes on. The step takes in at most `max` chunks whose content changed, those in `first`
    /// (the call's question's likely pieces) before the rest in material order, in batches of
    /// `STEP_PIECES`, and stops when the `content` memory area is full; the rarity and nuisance
    /// follow what is taken in. Once every chunk is in, the items follow the same way
    /// (`item_step`). Returns the chunks taken in by this step.
    /// # Errors
    /// The tokenizer refuses a text.
    pub fn sync_step(&mut self, sub: &Substrate, first: &[usize], max: usize) -> Result<usize, Fault> {
        let n = sub.chunks.len();
        // Realigned by content key whenever a slot holds another chunk's record: a chunk whose
        // content is unchanged keeps its record wherever it moved (its place and lines set to the
        // new ones, which is what taking it in again would give); every other slot holds a
        // placeholder of its path until it is taken in, never another chunk's record.
        let aligned = self.keys.len() == n && self.keys.iter().zip(&sub.chunks).all(|(k, ch)| k.is_empty() || *k == ch.key);
        if !aligned {
            let old_records = std::mem::take(&mut self.records);
            let old_keys = std::mem::take(&mut self.keys);
            let mut by_key: std::collections::HashMap<String, Record> = std::collections::HashMap::with_capacity(old_keys.len());
            for (k, r) in old_keys.into_iter().zip(old_records) {
                if !k.is_empty() {
                    by_key.entry(k).or_insert(r);
                }
            }
            let mut slots: Vec<Option<Record>> = Vec::with_capacity(n);
            let mut keys: Vec<String> = Vec::with_capacity(n);
            for (i, ch) in sub.chunks.iter().enumerate() {
                match by_key.remove(&ch.key) {
                    Some(mut r) => {
                        r.position = i + 1;
                        r.source = Source::Lines { path: sub.files.get(ch.file).map(|f| f.path.clone()).unwrap_or_default(), start: ch.start, end: ch.end };
                        slots.push(Some(r));
                        keys.push(ch.key.clone());
                    }
                    None => {
                        slots.push(None);
                        keys.push(String::new());
                    }
                }
            }
            drop(by_key);
            let missing: Vec<usize> = slots.iter().enumerate().filter(|(_, r)| r.is_none()).map(|(i, _)| i).collect();
            let made: Vec<Option<Result<Record, Fault>>> = nmtype::budget::run(missing.len(), |k| {
                let i = missing.get(k).copied().ok_or_else(|| bad("drone intake: a chunk was lost"))?;
                let ch = sub.chunks.get(i).ok_or_else(|| bad("drone intake: a chunk was lost"))?;
                let path = sub.files.get(ch.file).map_or("", |f| f.path.as_str());
                self.placeholder(path, ch.start, ch.end, i + 1)
            });
            for (&i, r) in missing.iter().zip(made) {
                let r = r.ok_or_else(|| bad("drone intake: a worker stopped"))??;
                if let Some(slot) = slots.get_mut(i) {
                    *slot = Some(r);
                }
            }
            self.records = slots.into_iter().collect::<Option<Vec<Record>>>().ok_or_else(|| bad("drone intake: a slot was lost"))?;
            self.keys = keys;
            self.held = self.records.iter().map(Record::held_bytes).sum();
            self.est = sub.chunks.iter().zip(&self.keys).map(|(c, k)| if k.is_empty() { c.tokens.max(1) } else { 0 }).collect();
            self.places = Vec::new();
        }
        let stale = |i: usize| !self.keys.get(i).zip(sub.chunks.get(i)).is_some_and(|(k, ch)| *k == ch.key);
        let mut queued = vec![false; n];
        let mut todo: Vec<usize> = Vec::new();
        for &i in first {
            if i < n && stale(i) && !queued.get(i).copied().unwrap_or(true) {
                if let Some(q) = queued.get_mut(i) {
                    *q = true;
                }
                todo.push(i);
            }
        }
        let rest: Vec<usize> = (0..n).filter(|i| stale(*i) && !queued.get(*i).copied().unwrap_or(true)).collect();
        let waiting_before = todo.len() + rest.len();
        todo.extend(rest);
        todo.truncate(max);
        let mut changed = 0_usize;
        let mut full = false;
        // Bytes held per token over the pieces already taken in estimate what each piece of the
        // next step adds; the step takes only the pieces that fit the room left.
        let (mut seen_bytes, mut seen_tokens) = (0_u64, 0_u64);
        for (i, k) in self.keys.iter().enumerate() {
            if let (false, Some(r), Some(ch)) = (k.is_empty(), self.records.get(i), sub.chunks.get(i)) {
                seen_bytes += r.held_bytes() as u64;
                seen_tokens += ch.tokens.max(1) as u64;
            }
        }
        for batch in todo.chunks(STEP_PIECES) {
            self.report_held();
            let room = nmengn::limits::area_room("content");
            if room == Some(0) {
                full = true;
                break;
            }
            let batch = match room {
                Some(left) if seen_tokens > 0 => {
                    let est = |i: usize| (sub.chunks.get(i).map_or(1, |ch| ch.tokens.max(1)) as u64).saturating_mul(seen_bytes) / seen_tokens;
                    batch.get(..fits(batch, left, est)).unwrap_or(batch)
                }
                _ => batch,
            };
            if batch.is_empty() {
                full = true;
                break;
            }
            let mut left = room.unwrap_or(u64::MAX);
            // Changed chunks dealt over the one processor budget (`nmtype::budget`, 2026-11-18
            // Phase C1), one partition per chunk, so large and small pieces spread evenly.
            let drone = &self.drone;
            let take_one = |k: usize| -> Result<Record, Fault> {
                let i = batch.get(k).copied().ok_or_else(|| bad("drone intake: a chunk was lost"))?;
                let ch = sub.chunks.get(i).ok_or_else(|| bad("drone intake: a chunk was lost"))?;
                let path = sub.files.get(ch.file).map(|f| f.path.clone()).unwrap_or_default();
                let r = drone.take_in_with(Kind::Code, Source::Lines { path, start: ch.start, end: ch.end }, i + 1, &ch.text, &ch.heading)?.into_inner();
                // The drones keep material only.
                nmtype::class::inlets::MATERIAL_DRONES.admit(&r)?;
                Ok(r)
            };
            let taken: Vec<Option<Result<Record, Fault>>> = nmtype::budget::run(batch.len(), take_one);
            for (&i, r) in batch.iter().zip(taken) {
                let r = r.ok_or_else(|| bad("drone intake: a worker stopped"))??;
                let Some(ch) = sub.chunks.get(i) else { continue };
                if let (Some(slot), Some(k)) = (self.records.get_mut(i), self.keys.get_mut(i)) {
                    if !admit((r.held_bytes() as u64).saturating_sub(slot.held_bytes() as u64), &mut left, &mut full) {
                        continue;
                    }
                    seen_bytes += r.held_bytes() as u64;
                    seen_tokens += ch.tokens.max(1) as u64;
                    self.held = self.held.saturating_sub(slot.held_bytes()).saturating_add(r.held_bytes());
                    *slot = r;
                    ch.key.clone_into(k);
                    if let Some(e) = self.est.get_mut(i) {
                        *e = 0;
                    }
                    changed += 1;
                }
            }
            if full {
                break;
            }
        }
        self.report_held();
        if changed > 0 || self.places.len() != n {
            self.rarity = Rarity::from_records(self.records.iter());
            self.places = sub
                .chunks
                .iter()
                .enumerate()
                .map(|(i, ch)| {
                    let path = sub.files.get(ch.file).map_or("", |f| f.path.as_str());
                    nmpod::evidence::Place {
                        tokens: self.tokens(i),
                        class: nmpod::evidence::Class::of(ch.kind, nmtype::is_document(path)),
                        first: ch.start <= 1,
                    }
                })
                .collect();
        }
        let pending = waiting_before - changed;
        self.state = KeepState { taken: n - pending, pending, full, ..self.state };
        // Items are also built for G2's belief per item (lane M's G2-u, on by default since
        // 2026-11-19; `NANO_ITEM_BELIEF=off`), once every chunk is taken in.
        if pending == 0 && !full && (items_mode() != ItemsMode::Off || !std::env::var("NANO_ITEM_BELIEF").is_ok_and(|v| v == "off")) {
            self.item_step(sub, max.saturating_sub(changed).max(STEP_PIECES))?;
        }
        nmengn::limits::set_area_full("content", self.state.full);
        Ok(changed)
    }
    /// The drones' calibrated evidence of every piece for `request` (`nmpod::evidence`): the
    /// request enters through its own drone, which reads its kind and aspects; every piece is
    /// scored per aspect in each family against the content-free pose.
    /// # Errors
    /// The tokenizer refuses the request or an aspect.
    pub fn evidence(&self, request: &str) -> Result<nmpod::evidence::Evidence, Fault> {
        self.evidence_with(request, None)
    }
    /// `evidence` with the meaning lane's cosine of every piece with the request, then with each
    /// content-free question (`meaning`, each in piece order), as one more signal of the meaning
    /// family.
    /// # Errors
    /// As `evidence`.
    pub fn evidence_with(&self, request: &str, meaning: Option<&[Vec<f32>]>) -> Result<nmpod::evidence::Evidence, Fault> {
        let req = self.drone.take_request("request", request)?;
        let read = intake::Request::read(&req.get().text, &[]);
        nmpod::evidence::sense(&self.drone, request, &read, &self.records, &self.rarity, &self.places, meaning)
    }
    /// The record of piece `i` as the drones took it in.
    #[must_use]
    pub fn record(&self, i: usize) -> Option<&Record> {
        self.records.get(i)
    }
    /// Flag strength of every chunk for `request` (the drones' fused order score).
    /// # Errors
    /// The tokenizer refuses the request.
    pub fn sense(&self, request: &str) -> Result<Vec<f32>, Fault> {
        self.sense_with(request, &intake::FIRST_FORM)
    }
    /// Every signal of every chunk for `request` (2026-11-10): the drones sense in parallel, the
    /// tree split into as many runs as there are processor cores, one probe per run; the result
    /// is in chunk order and equal to a single run.
    /// # Errors
    /// The tokenizer refuses the request.
    pub fn signals(&self, request: &str) -> Result<Vec<intake::Signals>, Fault> {
        let req = self.drone.take_request("request", request)?;
        self.drone.sense_records(&req, &self.records, &self.rarity)
    }
    /// Tokens of chunk `i` as the drone took it in.
    #[must_use]
    pub fn tokens(&self, i: usize) -> usize {
        match self.est.get(i) {
            Some(&e) if e > 0 => e,
            _ => self.records.get(i).map_or(0, Record::len),
        }
    }
    /// Flag strength of every chunk for `request` by reciprocal-rank fusion over the signals
    /// `get` (2026-11-10: `&[|s| s.bm25]` gives the BM25 order alone).
    /// # Errors
    /// The tokenizer refuses the request.
    pub fn sense_with(&self, request: &str, get: &[intake::Sig]) -> Result<Vec<f32>, Fault> {
        let req = self.drone.take_request("request", request)?;
        let mut probe = self.drone.probe_request(&req, &self.rarity);
        let rows: Vec<intake::Signals> = self.records.iter().map(|p| probe.sense(p)).collect();
        Ok(intake::fuse_with(&rows, get))
    }
}

/// Section check: the strongest flag leads window 0, its related piece joins it ahead of a
/// weaker drone piece, no chunk is in two windows, no window passes its budget, and every chunk
/// not placed stays referenced in the map.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let files: Vec<(String, String)> = vec![
        ("a/seed.rs".into(), "pub fn derive_window_limit() -> usize { pick_reader_span() }\n".into()),
        ("z/far.rs".into(), "pub fn pick_reader_span() -> usize { 4 }\n".into()),
        ("m/one.rs".into(), "pub fn unrelated_one() -> u8 { 1 }\n".into()),
        ("m/two.rs".into(), "pub fn unrelated_two() -> u8 { 2 }\n".into()),
        ("m/three.rs".into(), "pub fn unrelated_three() -> u8 { 3 }\n".into()),
    ];
    let embed: nmsubs::Embed = Arc::new(|_: &str| vec![0.0]);
    let borrowed: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let sub = nmsubs::from_files(&borrowed, embed);
    let g = sub.refs().specific_with(&sub, nmsubs::refgraph::Limit::Fixed(nmsubs::refgraph::RARE_FALLBACK));
    let n = sub.chunks.len();
    let at = |p: &str| sub.chunks.iter().position(|c| sub.files.get(c.file).is_some_and(|f| f.path == p));
    let (Some(seed), Some(far), Some(one)) = (at("a/seed.rs"), at("z/far.rs"), at("m/one.rs")) else {
        return Err("nmapi.drone_windows: check tree".into());
    };
    let mut fused = vec![0.0_f32; n];
    for (c, v) in [(seed, 1.0), (one, 0.4)] {
        if let Some(x) = fused.get_mut(c) {
            *x = v;
        }
    }
    if let Some(x) = fused.get_mut(far) {
        *x = 0.01;
    }
    let got = compose(&fused, &g, &|_| 10, 20, 2);
    let w0 = got.windows.first().cloned().unwrap_or_default();
    if w0 != vec![seed, far] {
        return Err(format!("nmapi.drone_windows: window 0 {w0:?}, wanted the flag and its related piece"));
    }
    let all: Vec<usize> = got.windows.iter().flatten().copied().collect();
    let mut uniq = all.clone();
    uniq.sort_unstable();
    uniq.dedup();
    if uniq.len() != all.len() || got.windows.iter().any(|w| w.len() * 10 > 20) {
        return Err("nmapi.drone_windows: overlap or a window over budget".into());
    }
    let passed: Vec<usize> = got.map.passed().into_iter().map(|p| p.0).collect();
    if (0..n).any(|c| !all.contains(&c) && !passed.contains(&c)) {
        return Err("nmapi.drone_windows: a chunk not placed is not referenced".into());
    }
    if WindowMode::default() != WindowMode::Lens {
        return Err("nmapi.drone_windows: not off by default".into());
    }
    Ok(())
}
