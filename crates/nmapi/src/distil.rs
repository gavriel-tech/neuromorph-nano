//! Distilled Writer cartridges: a teacher's output behaviour fitted into a low-rank map at the
//! Writer's final norm, closed form, no backward pass, and the loading path that plugs it in.
//!
//! Data: curated teacher pairs in the distil family of the training store
//! (`nmtype::examples::DISTIL`, the Nano's own `memory/training/distil.jsonl`), each a prompt, the
//! teacher's completion, the teacher's id and licence, optionally the teacher's top log-probabilities
//! per token, a split fixed by the prompt's hash (fit, tune, held out) and an outcome label; a fit
//! reads only healthy rows (`used`, `superseded`).
//!
//! Fit (the lab's `run distil build`, `nmlab::distil`): the Writer reads each pair under teacher
//! forcing; at every completion position its final-norm state `h` and its logits over the serving
//! vocabulary are taken. The position's target shift is the smallest change of `h` that moves the
//! logits of a few head rows `S` (the teacher's token or tokens and the Writer's own top choices) to
//! their targets `z*`: `Delta = W_S' (W_S W_S' + mu I)^-1 (z* - W_S h)`, a `k x k` solve
//! (`target_delta`, the push-through identity of the ridge problem `min |W_S (h + Delta) - z*|^2 +
//! mu |Delta|^2`). Then reduced-rank ridge from `h` to `h + Delta` over the fit split
//! (`adapter_fit::ridge_low_rank_sums`), its strength, rank and scale chosen on the tune split.
//!
//! Serving: the map is `y' = r16(y + scale * up(down(y)))` at the Writer's `final_norm`, an
//! `nmengn::Adapter` on the model's units (`Model::set_unit` on a norm node), applied after the
//! node's cartridge filters, so the released weights and the base cartridge are never written and
//! unplugging gives the base's outputs bit for bit. A cartridge carries the values key of the
//! Writer it was fitted on (`notes.values`) and loads only on that same base.
//!
//! The cartridge is an `nm-cartridge/1` file on the adapter shelf
//! (`library/cartridges/adapter/writer-distil-<set>/`), kind `lowrank`, model the Writer's open
//! weights id, parts `down` `[r, d]` and `up` `[d, r]` at 16 bits, its gate figures, lineage and
//! outcome on the label. It is a learned unit of the `nano` compartment: plugged in or out on the
//! Writer's deck (`units/plugs.json`), and a new cartridge id starts unplugged.
//!
//! Expected effect, stated on every label: style, format and wording preferences of the teacher;
//! not new knowledge or reasoning (a closed-form map at the last layer re-weights features the
//! Writer already has).
use crate::adapter::Part;
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::path::{Path, PathBuf};
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.distil", FaultKind::Interface, m)
}

/// The id of a distilled Writer cartridge before its set's name (`<role>-distil-<set>` for any
/// MicroModel, `cartridge_id`).
pub const PREFIX: &str = "writer-distil";
/// The distillation techniques the lab's `distil` build names (`technique`): `lens` (reduced-rank
/// ridge toward the teacher's targets at the socket, this module), `channel` (per-channel gain and
/// bias toward the same targets, the job filter's form), `unit` (a forward-only trained LoRA unit),
/// `readout` (a recursive least squares read-out), `steer` (a closed-form steering vector) and
/// `teacher-loop` (a technique refitted over rounds: each round finds the fit pairs the student
/// still follows the teacher worst on, weighs them up and fits again, stopping when the tune split
/// stops gaining; repeated trials on thirds of the data then confirm the result is stable).
pub const TECHNIQUES: [&str; 6] = ["lens", "channel", "unit", "readout", "steer", "teacher-loop"];
/// The kinds of learned cartridge a distillation fills (`target`), each written in that kind's own
/// format so it plugs in as cartridges of that kind do: `adapter` (the adapter shelf, a learned
/// unit on the MicroModel's deck), `filter` (a job filter, `nmengn-sockets/1`), `preference` (the
/// `pref.*` values of a preference cartridge) and `readout` (a judge read-out).
pub const TARGETS: [&str; 4] = ["adapter", "filter", "preference", "readout"];
/// Which techniques can fill which kind (kind by technique); any other pair is refused with its
/// reason.
pub const FILLS: [(&str, &str); 9] = [
    ("adapter", "lens"),
    ("adapter", "teacher-loop"),
    ("adapter", "unit"),
    ("adapter", "steer"),
    ("filter", "channel"),
    ("filter", "teacher-loop"),
    ("preference", "readout"),
    ("readout", "readout"),
    ("readout", "teacher-loop"),
];
/// What the lab builds today: (role, kind, technique). Every other allowed pair answers with what
/// it still needs (`not_built`), never with a stand-in. The Writer and the Reader are causal
/// language models: their targets are the teacher's next tokens. The Reranker gives one score (its
/// yes logit minus its no logit): its targets are the teacher's scores of (query, document) pairs.
pub const BUILT: [(&str, &str, &str); 19] = [
    ("writer", "adapter", "lens"),
    ("writer", "adapter", "teacher-loop"),
    ("writer", "filter", "channel"),
    ("writer", "filter", "teacher-loop"),
    ("reader", "adapter", "lens"),
    ("reader", "adapter", "teacher-loop"),
    ("reader", "filter", "channel"),
    ("reader", "filter", "teacher-loop"),
    ("reranker", "adapter", "lens"),
    ("reranker", "adapter", "teacher-loop"),
    ("reranker", "filter", "channel"),
    ("reranker", "filter", "teacher-loop"),
    ("reranker", "readout", "readout"),
    ("reranker", "readout", "teacher-loop"),
    ("embedder", "adapter", "lens"),
    ("embedder", "adapter", "teacher-loop"),
    ("writer", "adapter", "steer"),
    ("reader", "adapter", "steer"),
    ("reader", "adapter", "unit"),
];
/// Whether a role's distillation reads teacher scores (a score per pair) rather than teacher
/// next tokens.
#[must_use]
pub fn scored(role: &str) -> bool {
    matches!(role, "reranker" | "embedder")
}
/// The MicroModel roles a distillation may target.
pub const ROLES: [&str; 4] = ["reader", "writer", "embedder", "reranker"];
/// The label's `kind` of a distilled cartridge.
pub const KIND: &str = "lowrank";
/// The assembly role of the socket it plugs into.
pub const SOCKET: &str = "final_norm";
/// The end-of-turn token closing a chat completion (Qwen3 `<|im_end|>`).
pub const END_OF_TURN: i32 = 151_645;
/// The end-of-text token (Qwen3 `<|endoftext|>`).
pub const END_OF_TEXT: i32 = 151_643;
/// Writer top choices added to the teacher's tokens in each position's head rows.
pub const WRITER_TOP: usize = 12;
/// Margin (nats) the teacher's token is lifted above the Writer's best other choice when the
/// teacher gave text only.
pub const MARGIN: f64 = 2.0;
/// Ridge of the per-position solve, times the mean squared norm of the head rows in it.
pub const MU: f64 = 0.05;
/// The serving vocabulary's first rows (the Writer's active head rows, `writer_idle`).
pub const ACTIVE_FIRST: usize = 16_384;
/// The serving vocabulary's cap.
pub const ACTIVE_CAP: usize = 24_576;

/// Whether distilled cartridges load on the Writer: on by default; `NANO_DISTIL=off` for
/// diagnosis (each cartridge is plugged in or out on its own in `units/plugs.json`).
#[must_use]
pub fn on() -> bool {
    !std::env::var("NANO_DISTIL").is_ok_and(|v| v == "off")
}

/// A set's name as the cartridge id carries it: lower case letters, digits and `-`, at most 40.
#[must_use]
pub fn slug(set: &str) -> String {
    let s: String = set.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    let s: String = s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-");
    let s: String = s.chars().take(40).collect();
    if s.is_empty() { "set".to_owned() } else { s }
}

/// The cartridge id of a set distilled into a role's MicroModel: `<role>-distil-<set>`.
#[must_use]
pub fn cartridge_id(role: &str, set: &str) -> String {
    format!("{}-distil-{}", slug(role), slug(set))
}

/// The job of a distilled job filter (`job-filter-distil-<set>.json`).
#[must_use]
pub fn filter_job(set: &str) -> String {
    format!("distil-{}", slug(set))
}

/// Whether an adapter id names a distilled cartridge, and of which role.
#[must_use]
pub fn role_of(id: &str) -> Option<&'static str> {
    ROLES.into_iter().find(|r| id.starts_with(&format!("{r}-distil-")))
}

/// Why the lab cannot fill a `target` kind for `role` by `technique` yet, or None when it can.
#[must_use]
pub fn not_built(role: &str, target: &str, technique: &str) -> Option<String> {
    if !ROLES.contains(&role) {
        return Some(format!("role {role}: use reader, writer, embedder or reranker"));
    }
    if !TARGETS.contains(&target) {
        return Some(format!("target {target}: use {}", TARGETS.join(", ")));
    }
    if !TECHNIQUES.contains(&technique) {
        return Some(format!("technique {technique}: use {}", TECHNIQUES.join(", ")));
    }
    if !FILLS.contains(&(target, technique)) {
        let can: Vec<&str> = FILLS.iter().filter(|(k, _)| *k == target).map(|(_, t)| *t).collect();
        return Some(format!("technique {technique} cannot fill a {target} cartridge; a {target} cartridge is filled by {}", can.join(", ")));
    }
    if BUILT.contains(&(role, target, technique)) || (target, technique) == ("preference", "readout") {
        return None;
    }
    Some(match (role, target, technique) {
        ("embedder", "filter", _) => "a filter cartridge acts on every vector the embedder makes, the stored piece vectors included, and the dense store's header would not say so; the embedder is distilled into its request-side adapter (a distilled request lens), which leaves the stored vectors as they are".to_owned(),
        ("embedder" | "reranker", _, "steer") => format!("a steering vector moves the residual stream at every position, and the {role}'s output is {}; its distillation fits that output at its final norm (an adapter{})", if role == "embedder" { "a vector stored for every indexed piece" } else { "one score" }, if role == "embedder" { ", on the request side" } else { ", a filter or a read-out" }),
        (_, "readout", _) => format!("a read-out cartridge is a judge's: it adds to one score, and only the reranker gives one score (the {role}'s output is {}); use an adapter or a filter for the {role}", if role == "embedder" { "a vector" } else { "next tokens" }),
        (_, _, "unit") => format!("a learning unit serves on the shared model only (the Reader's, loaded by serve --unit at its learning target), so distillation by unit is built for the Reader; the {role} is distilled by the closed-form techniques"),
        _ => format!("distillation into a {target} for the {role} by {technique} is planned"),
    })
}

/// The Writer's chat template around a prompt (the same turn the idle Writer writes in, thinking
/// left empty).
#[must_use]
pub fn chat_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// The token ids of one pair under teacher forcing and the length of its prompt: the prompt (in the
/// chat template unless `raw`), the completion as plain text, then the end of turn (chat) or of text
/// (raw). The logits at position `p` score token `p + 1`, so the scored positions are
/// `prompt_len - 1 .. ids.len() - 1`.
/// # Errors
/// The tokenizer fails.
pub fn sequence(tok: &nmengn::tokenizer::Tokenizer, row: &nmtype::examples::DistilRow) -> Result<(Vec<i32>, usize), Fault> {
    let head = if row.template == "raw" { row.prompt.clone() } else { chat_prompt(&row.prompt) };
    let mut ids = tok.encode(&head).map_err(|e| bad(e.to_string()))?;
    let plen = ids.len();
    ids.extend(tok.encode_plain(&row.completion).map_err(|e| bad(e.to_string()))?);
    ids.push(if row.template == "raw" { END_OF_TEXT } else { END_OF_TURN });
    Ok((ids, plen))
}

/// The serving vocabulary of a fit: the first `ACTIVE_FIRST` rows, both end tokens and the tokens
/// the data uses, up to `ACTIVE_CAP` rows, in id order (as the idle Writer keeps its head rows).
#[must_use]
pub fn serving_rows(vocab: usize, seen: &[i32]) -> Vec<usize> {
    let mut keep: std::collections::BTreeSet<usize> = (0..ACTIVE_FIRST.min(vocab)).collect();
    for t in [END_OF_TEXT, END_OF_TURN] {
        if let Ok(u) = usize::try_from(t)
            && u < vocab
        {
            keep.insert(u);
        }
    }
    for t in seen {
        if keep.len() >= ACTIVE_CAP {
            break;
        }
        if let Ok(u) = usize::try_from(*t)
            && u < vocab
        {
            keep.insert(u);
        }
    }
    keep.into_iter().collect()
}

// ---------------------------------------------------------------- the per-position target

/// The target logits of a position's head rows when the teacher gave text only: the Writer's own
/// logits `z` (teacher's row first), with the teacher's lifted to `MARGIN` above the best other row
/// when it is not already there.
#[must_use]
pub fn margin_target(z: &[f64]) -> Vec<f64> {
    let mut t = z.to_vec();
    let best_other = z.iter().skip(1).copied().fold(f64::MIN, f64::max);
    if let Some(first) = t.first_mut()
        && best_other > f64::MIN
    {
        *first = first.max(best_other + MARGIN);
    }
    t
}

/// The target logits when the teacher's top log-probabilities are known: rows `0..teacher.len()`
/// are the teacher's tokens (log-probabilities `teacher`, most likely first), the rest the Writer's
/// own choices. The teacher's rows take its log-probabilities shifted so its best token sits at the
/// Writer's best logit among the rows; a Writer-only row is kept below the teacher's least likely
/// listed token by one nat.
#[must_use]
pub fn teacher_target(z: &[f64], teacher: &[f64]) -> Vec<f64> {
    let top = z.iter().copied().fold(f64::MIN, f64::max);
    let lead = teacher.first().copied().unwrap_or(0.0);
    let shift = top - lead;
    let floor = teacher.iter().copied().fold(f64::MAX, f64::min) + shift - 1.0;
    z.iter().enumerate().map(|(i, zi)| teacher.get(i).map_or_else(|| zi.min(floor), |lp| lp + shift)).collect()
}

/// The smallest shift of `h` (ridge `mu` times the rows' mean squared norm) that moves the logits of
/// the head rows `rows` from `z` to `target`: `W' (W W' + mu I)^-1 (target - z)`. None when the
/// small system is not solvable.
#[must_use]
pub fn target_delta(rows: &[&[f32]], z: &[f64], target: &[f64], mu: f64) -> Option<Vec<f64>> {
    let k = rows.len();
    let d = rows.first().map_or(0, |r| r.len());
    if k == 0 || d == 0 || z.len() != k || target.len() != k || rows.iter().any(|r| r.len() != d) {
        return None;
    }
    let mut a = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let v: f64 = rows[i].iter().zip(rows[j]).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
            a[i * k + j] = v;
            a[j * k + i] = v;
        }
    }
    let mean = (0..k).map(|i| a[i * k + i]).sum::<f64>() / k as f64;
    for i in 0..k {
        a[i * k + i] += mu * mean.max(1e-12);
    }
    if !crate::adapter_fit::cholesky(&mut a, k) {
        return None;
    }
    let mut alpha: Vec<f64> = target.iter().zip(z).map(|(t, x)| t - x).collect();
    crate::adapter_fit::chol_solve(&a, k, &mut alpha, 1);
    let mut delta = vec![0.0_f64; d];
    for (r, al) in rows.iter().zip(&alpha) {
        for (o, w) in delta.iter_mut().zip(r.iter()) {
            *o += al * f64::from(*w);
        }
    }
    Some(delta)
}

// ---------------------------------------------------------------- the sums of the ridge fit

/// The running sums of the reduced-rank ridge from `h` to `h + Delta`: `X'X` and `X'(X + Delta)`,
/// `d x d` row-major, over the rows added.
pub struct Sums {
    pub d: usize,
    pub g: Vec<f64>,
    pub cy: Vec<f64>,
    pub n: usize,
}

impl Sums {
    #[must_use]
    pub fn new(d: usize) -> Self {
        Self { d, g: vec![0.0; d * d], cy: vec![0.0; d * d], n: 0 }
    }
    /// Adds rows `(h, delta, weight)`, the outer products split over the processor's threads by
    /// input channel (each thread owns whole rows of both sums, so the result does not depend on
    /// the thread count).
    pub fn add(&mut self, rows: &[(&[f32], &[f32], f64)]) {
        let d = self.d;
        if d == 0 || rows.is_empty() {
            return;
        }
        let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get).clamp(1, 16);
        let per = d.div_ceil(threads);
        let gs = self.g.chunks_mut(per * d);
        let cs = self.cy.chunks_mut(per * d);
        std::thread::scope(|sc| {
            for (t, (gc, cc)) in gs.zip(cs).enumerate() {
                sc.spawn(move || {
                    let i0 = t * per;
                    for (h, dl, w) in rows {
                        if h.len() != d || dl.len() != d {
                            continue;
                        }
                        for (ii, (grow, crow)) in gc.chunks_mut(d).zip(cc.chunks_mut(d)).enumerate() {
                            let xi = w * f64::from(h[i0 + ii]);
                            if xi == 0.0 {
                                continue;
                            }
                            for ((gv, cv), (xj, dj)) in grow.iter_mut().zip(crow.iter_mut()).zip(h.iter().zip(dl.iter())) {
                                let xj = f64::from(*xj);
                                *gv += xi * xj;
                                *cv += xi * (xj + f64::from(*dj));
                            }
                        }
                    }
                });
            }
        });
        self.n += rows.len();
    }
    /// The sums of two sets of rows together.
    pub fn plus(&mut self, other: &Self) {
        self.g.iter_mut().zip(&other.g).for_each(|(a, b)| *a += b);
        self.cy.iter_mut().zip(&other.cy).for_each(|(a, b)| *a += b);
        self.n += other.n;
    }
}

/// The low-rank map's leading `r` directions out of a wider fit (`down` `[rmax, d]`, `up`
/// `[d, rmax]`), as f32: (`down [r, d]`, `up [d, r]`).
#[must_use]
pub fn leading(down: &[f64], up: &[f64], d: usize, r: usize) -> (Vec<f32>, Vec<f32>) {
    let rmax = up.len().checked_div(d).unwrap_or(0);
    let r = r.min(rmax);
    #[expect(clippy::cast_possible_truncation, reason = "parameters kept at 16 bits")]
    let dn: Vec<f32> = down.iter().take(r * d).map(|x| *x as f32).collect();
    #[expect(clippy::cast_possible_truncation, reason = "parameters kept at 16 bits")]
    let upr: Vec<f32> = (0..d).flat_map(|m| up.get(m * rmax..m * rmax + r).unwrap_or(&[]).iter().map(|x| *x as f32)).collect();
    (dn, upr)
}

// ---------------------------------------------------------------- the cartridge

/// A distilled cartridge as the engine takes it: `nmengn::Adapter` over the socket, square at the
/// Writer's width, its parameters as stored (16 bits).
/// # Errors
/// Shapes that do not fit, or a scale that is not finite.
#[expect(clippy::too_many_arguments, reason = "a unit is its key, its socket, its widths, two matrices and a scale")]
pub fn unit(key: &str, socket: &str, d: usize, rank: usize, down: &[f32], up: &[f32], scale: f32) -> Result<nmengn::Adapter, Fault> {
    let r16 = |xs: &[f32]| -> Vec<f32> { xs.iter().map(|x| nmengn::half::r16(*x)).collect() };
    nmengn::Adapter::new(key, socket, rank, d, d, r16(down), r16(up), scale)
}

/// The label of a distilled cartridge's file: the adapter label (`adapter_fit::label`) with kind
/// `lowrank`, its model the Writer's open weights id; `notes` carries the values key (`values`),
/// the socket, scale, rank, ridge, the gate figures, references, teachers, licences and lineage.
#[must_use]
pub fn label(id: &str, model: &str, weights: &str, made_from: &str, notes: Value) -> Value {
    crate::adapter_fit::label(&crate::adapter_fit::LabelIn {
        id,
        version: 1,
        model,
        weights,
        kind: KIND,
        does: "Distilled output behaviour for the Writer: a low-rank map at its final norm fitted so its next-token choices follow a teacher's completions (style, format and wording; not new knowledge)",
        made_from,
        notes,
    })
}

fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn num(v: &Value, k: &str) -> Option<f64> {
    match v.get(k) {
        Ok(Value::Number(s)) => s.parse().ok(),
        _ => None,
    }
}

/// The unit of a label and its parts (`adapter::read_bytes`), checked: kind `lowrank`, made for
/// `model`, fitted on `values`, `down` `[r, width]` and `up` `[width, r]`.
/// # Errors
/// Any of those checks.
pub fn unit_from(label: &Value, parts: &[Part], model: &str, values: &str, width: usize, socket: &str) -> Result<nmengn::Adapter, Fault> {
    if text(label, "kind") != KIND {
        return Err(bad("not a distilled low-rank cartridge"));
    }
    if text(label, "model") != model {
        return Err(bad(format!("made for {}, the Writer runs {model}", text(label, "model"))));
    }
    let notes = label.get("notes").cloned().unwrap_or(Value::Null);
    let fitted = text(&notes, "values");
    if fitted != values {
        return Err(bad(format!("fitted on {fitted}, the Writer runs on {values}")));
    }
    let key = format!("{}@{}", text(label, "id"), label.get("version").and_then(Value::integer).unwrap_or(0));
    let part = |n: &str| parts.iter().find(|p| p.0 == n);
    let (Some(dn), Some(upp)) = (part("down"), part("up")) else { return Err(bad(format!("{key}: no down or up part"))) };
    let rank = dn.1.first().copied().unwrap_or(0);
    if dn.1.get(1).copied() != Some(width) || upp.1 != [width, rank] {
        return Err(bad(format!("{key}: shapes {:?} and {:?} do not fit the Writer's width {width}", dn.1, upp.1)));
    }
    #[expect(clippy::cast_possible_truncation, reason = "a scale stored as text")]
    let scale = num(&notes, "scale").unwrap_or(1.0) as f32;
    unit(&key, socket, width, rank, &dn.2, &upp.2, scale)
}

/// Every distilled cartridge id on the adapter shelf with its newest version file and when that
/// file was written, newest first.
#[must_use]
pub fn on_shelf(home: &Path, role: &str) -> Vec<(String, PathBuf, u64)> {
    let shelf = home.join("library/cartridges/adapter");
    let Ok(rd) = std::fs::read_dir(&shelf) else { return Vec::new() };
    let mut out: Vec<(String, PathBuf, u64)> = rd
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().to_str().map(str::to_owned))
        .filter(|id| id.starts_with(&format!("{role}-distil-")))
        .filter_map(|id| {
            let (_, file) = crate::adapter::versions(home, &id).pop()?;
            let ms = std::fs::metadata(&file).and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
            Some((id, file, ms))
        })
        .collect();
    out.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    out
}

/// What each MicroModel's distilled socket holds now, by role, for the status, the live view and
/// the learned units: the key seated (`id@version`) and the note.
static SEATED: RwLock<BTreeMap<String, (Option<String>, String)>> = RwLock::new(BTreeMap::new());

/// The key of the distilled cartridge seated on the Writer, if any, and the last note.
#[must_use]
pub fn seated() -> (Option<String>, String) {
    seated_on("writer")
}

/// The key of the distilled cartridge seated on `role`'s MicroModel, if any, and the last note.
#[must_use]
pub fn seated_on(role: &str) -> (Option<String>, String) {
    SEATED.read().ok().and_then(|g| g.get(role).cloned()).unwrap_or_default()
}

fn note(role: &str, key: Option<String>, line: String) -> String {
    if let Ok(mut g) = SEATED.write() {
        g.insert(role.to_owned(), (key, line.clone()));
    }
    line
}

/// The MicroModel's name in a seat line.
fn title(role: &str) -> String {
    let mut c = role.chars();
    c.next().map_or_else(String::new, |f| f.to_ascii_uppercase().to_string() + c.as_str())
}

/// The unit of a distilled cartridge's label and parts, of either kind the socket takes: a low-rank
/// map (`lowrank`, `unit_from`) or a judge read-out (`readout`, `readout_unit_from`).
/// # Errors
/// As those two.
pub fn any_unit_from(label: &Value, parts: &[Part], model: &str, values: &str, width: usize, socket: &str) -> Result<nmengn::Adapter, Fault> {
    if text(label, "kind") == crate::adapter::READOUT_KIND { readout_unit_from(label, parts, model, values, width, socket) } else { unit_from(label, parts, model, values, width, socket) }
}

/// Seats the distilled cartridge that is plugged in on the Writer `m` (`seat`).
#[must_use]
pub fn seat_writer(home: &Path, m: &nmengn::Model) -> String {
    seat(home, m, "writer")
}

/// Seats the distilled cartridge that is plugged in on `role`'s MicroModel `m` (called as the
/// MicroModel is opened: `neural::writer_model`, `neural::model` for the Reader, the reranker's
/// role model; and for the Writer again at the start of every idle writing session, so a plug
/// change reaches a Writer kept on the card): the newest plugged cartridge on the shelf whose
/// newest usable version was fitted on this MicroModel's values. None plugged, the switch off, or
/// nothing usable: the socket is emptied and the base serves alone, bit for bit. Returns the line
/// it logged.
#[must_use]
pub fn seat(home: &Path, m: &nmengn::Model, role: &str) -> String {
    if role == "embedder" {
        return seat_embedder(home, m);
    }
    let a = seat_final(home, m, role);
    let b = seat_steer(home, m, role);
    [a, b].into_iter().filter(|x| !x.is_empty()).collect::<Vec<_>>().join("; ")
}

/// The final-norm socket's part of `seat`: an adapter or a judge read-out.
fn seat_final(home: &Path, m: &nmengn::Model, role: &str) -> String {
    let who = title(role);
    let Ok(node) = m.role_node(SOCKET) else { return note(role, None, format!("the {who}'s assembly has no final norm; nothing distilled seats")) };
    let empty = |why: String| -> String {
        let _ = m.set_unit(&node, None);
        note(role, None, why)
    };
    if !on() {
        return empty("distilled cartridges switched off (NANO_DISTIL=off)".into());
    }
    let shelf = on_shelf(home, role);
    if shelf.is_empty() {
        return empty(String::new());
    }
    let plugs = crate::plugs::read(home);
    let values = crate::neural::parameters_key(m);
    let width = m.pop.dims.hidden;
    let model = m.pop.card.id.clone();
    let mut refused = Vec::new();
    // Steering vectors have their own socket (`seat_steer`).
    let steer = format!("{role}-distil-steer-");
    for (id, _, _) in shelf.iter().filter(|(id, _, _)| plugs.acts(crate::plugs::NANO, id) && !id.starts_with(&steer)) {
        for (_, file) in crate::adapter::versions(home, id).into_iter().rev() {
            let got = crate::adapter::read_file(&file).and_then(|(l, p)| {
                // Made for these open weights (the label's `weights`), as the compatibility table reads it.
                let w = text(&l, "weights");
                if !w.is_empty() && w != m.pop.card.sha_main {
                    return Err(bad(format!("{id}: made for other open weights of {model}")));
                }
                any_unit_from(&l, &p, &model, &values, width, &node)
            });
            match got {
                Ok(u) => {
                    let key = u.unit.clone();
                    if m.units_loaded().iter().any(|(n, k)| *n == node && *k == key) {
                        return note(role, Some(key.clone()), format!("{key} seated on the {who}'s final norm"));
                    }
                    return match m.set_unit(&node, Some(Arc::new(u))) {
                        Ok(()) => {
                            let line = format!("{key} seated on the {who}'s final norm ({values})");
                            eprintln!("distil: {line}");
                            note(role, Some(key), line)
                        }
                        Err(e) => empty(format!("{key} left out: {e}")),
                    };
                }
                Err(e) => refused.push(e.message),
            }
        }
    }
    let why = if refused.is_empty() { "no distilled cartridge plugged in".to_owned() } else { format!("no distilled cartridge seats: {}", refused.join("; ")) };
    empty(why)
}

// ---------------------------------------------------------------- steer and the embedder's request lens

/// The label's `kind` of a distilled steering vector.
pub const STEER_KIND: &str = "steer";

/// The steering vector of a distilled steer cartridge's label and parts (part `v` `[d]`, `layer` and
/// `scale` in the notes), checked: made for `model`, fitted on `values`, at `width`, at a layer the
/// model has. Returns (layer, vector, scale, key).
/// # Errors
/// Any of those checks.
pub fn steer_from(label: &Value, parts: &[Part], model: &str, values: &str, width: usize, layers: usize) -> Result<(usize, Vec<f32>, f32, String), Fault> {
    if text(label, "kind") != STEER_KIND {
        return Err(bad("not a distilled steering vector"));
    }
    if text(label, "model") != model {
        return Err(bad(format!("made for {}, the MicroModel runs {model}", text(label, "model"))));
    }
    let notes = label.get("notes").cloned().unwrap_or(Value::Null);
    let fitted = text(&notes, "values");
    if fitted != values {
        return Err(bad(format!("fitted on {fitted}, the MicroModel runs on {values}")));
    }
    let key = format!("{}@{}", text(label, "id"), label.get("version").and_then(Value::integer).unwrap_or(0));
    let v = parts.iter().find(|p| p.0 == "v").map(|p| p.2.clone()).ok_or_else(|| bad(format!("{key}: no v part")))?;
    let layer = notes.get("layer").and_then(Value::integer).ok().and_then(|l| usize::try_from(l).ok()).ok_or_else(|| bad(format!("{key}: no layer")))?;
    #[expect(clippy::cast_possible_truncation, reason = "a scale stored as text")]
    let scale = num(&notes, "scale").unwrap_or(0.0) as f32;
    if v.len() != width || layer >= layers || !scale.is_finite() || v.iter().any(|x| !x.is_finite()) {
        return Err(bad(format!("{key}: a vector of {} values at layer {layer} does not fit (width {width}, {layers} layers)", v.len())));
    }
    Ok((layer, v, scale, key))
}

/// Seats the newest plugged distilled steering vector of `role` on `m` (`Model::set_steer`), or
/// takes the seated one out; runs without one are exactly as before. Returns the line it logged.
#[must_use]
pub fn seat_steer(home: &Path, m: &nmengn::Model, role: &str) -> String {
    let who = title(role);
    let clear = |why: String| -> String {
        let _ = m.set_steer(BTreeMap::new());
        why
    };
    if !on() {
        return clear(String::new());
    }
    let plugs = crate::plugs::read(home);
    let values = crate::neural::parameters_key(m);
    let mut refused = Vec::new();
    for (id, _, _) in on_shelf(home, role).iter().filter(|(id, _, _)| plugs.acts(crate::plugs::NANO, id)) {
        for (_, file) in crate::adapter::versions(home, id).into_iter().rev() {
            let Ok((l, p)) = crate::adapter::read_file(&file) else { continue };
            if text(&l, "kind") != STEER_KIND {
                break;
            }
            match steer_from(&l, &p, &m.pop.card.id, &values, m.pop.dims.hidden, m.pop.dims.layers) {
                Ok((layer, v, scale, key)) => {
                    return match m.set_steer(BTreeMap::from([(layer, (Arc::new(v), scale))])) {
                        Ok(()) => {
                            let line = format!("{key} steering the {who} at layer {layer} (scale {scale})");
                            eprintln!("distil: {line}");
                            line
                        }
                        Err(e) => clear(format!("{key} left out: {e}")),
                    };
                }
                Err(e) => refused.push(e.message),
            }
        }
    }
    clear(if refused.is_empty() { String::new() } else { format!("no distilled steer seats: {}", refused.join("; ")) })
}

/// The distilled request lens seated on the embedder, if any.
static EMBED_LENS: RwLock<Option<Arc<crate::adapter::Lens>>> = RwLock::new(None);

/// The distilled request lens in force on the embedder (none when switched off or none plugged).
#[must_use]
pub fn embed_lens() -> Option<Arc<crate::adapter::Lens>> {
    if !on() {
        return None;
    }
    EMBED_LENS.read().ok().and_then(|g| g.clone())
}

/// A request vector through every request-side map in force, in order: the distilled request lens
/// (fitted on the embedder's plain request vectors, so it reads them as they come) and then the
/// project's embedder lens. None when neither is in force. Stored piece vectors are never mapped by
/// the distilled lens, so the dense store stays valid with it plugged in or out.
#[must_use]
pub fn query_lens(v: &[f32]) -> Option<Vec<f32>> {
    let d = embed_lens().map(|l| l.apply(v));
    let base = d.as_deref().unwrap_or(v);
    crate::adapter::lens().map(|l| l.apply(base)).or(d)
}

/// The key of the request-side maps in force, for records that hold only for the request vectors
/// they were read with: the project lens's key (`none` without one), then `+` and the distilled
/// lens's key when one is seated.
#[must_use]
pub fn lens_key() -> String {
    let p = crate::adapter::lens().map_or_else(|| "none".to_owned(), |l| l.key.clone());
    embed_lens().map_or(p.clone(), |d| format!("{p}+{}", d.key))
}

/// Seats the newest plugged distilled request lens on the embedder `m` (its values key checked),
/// or takes it out; either way the lens epoch moves when what is seated changed, so caches of
/// lensed request vectors are made again. Returns the line it logged.
#[must_use]
pub fn seat_embedder(home: &Path, m: &nmengn::Model) -> String {
    let values = crate::neural::parameters_key(m);
    let plugs = crate::plugs::read(home);
    let mut refused = Vec::new();
    let mut got: Option<(crate::adapter::Lens, String)> = None;
    if on() {
        'ids: for (id, _, _) in on_shelf(home, "embedder").iter().filter(|(id, _, _)| plugs.acts(crate::plugs::NANO, id)) {
            for (_, file) in crate::adapter::versions(home, id).into_iter().rev() {
                match crate::adapter::lens_of(&file, &values) {
                    Ok(l) if l.flat.is_empty() => {
                        let line = format!("{} seated on the embedder's request vectors ({values})", l.key);
                        got = Some((l, line));
                        break 'ids;
                    }
                    Ok(l) => refused.push(format!("{}: a distilled lens has no flattening", l.key)),
                    Err(e) => refused.push(e.message),
                }
            }
        }
    }
    let line = got.as_ref().map(|(_, l)| l.clone()).unwrap_or_else(|| if refused.is_empty() { String::new() } else { format!("no distilled request lens seats: {}", refused.join("; ")) });
    let new = got.map(|(l, _)| Arc::new(l));
    if let Ok(mut g) = EMBED_LENS.write() {
        let changed = g.as_ref().map(|l| l.key.clone()) != new.as_ref().map(|l| l.key.clone());
        *g = new;
        if changed {
            crate::adapter::lens_changed();
        }
    }
    note("embedder", embed_lens().map(|l| l.key.clone()), line.clone());
    if !line.is_empty() {
        eprintln!("distil: {line}");
    }
    line
}


// ---------------------------------------------------------------- the judge's score

/// The teacher's score clamped into this open interval before its log-odds are taken, so a hard
/// 0 or 1 asks for a finite score (log-odds about 4.6).
pub const SCORE_CLAMP: f64 = 0.01;

/// The log-odds of a teacher's score (`SCORE_CLAMP` from either end).
#[must_use]
pub fn score_logit(p: f64) -> f64 {
    let p = p.clamp(SCORE_CLAMP, 1.0 - SCORE_CLAMP);
    (p / (1.0 - p)).ln()
}

/// The logistic of a log-odds score.
#[must_use]
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// The relative entropy (nats) of a judge's verdict `sigmoid(z)` from the teacher's score `p`:
/// `p ln(p / q) + (1 - p) ln((1 - p) / (1 - q))`, zero when the judge says what the teacher says.
#[must_use]
pub fn bernoulli_kl(p: f64, z: f64) -> f64 {
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    let q = sigmoid(z).clamp(1e-12, 1.0 - 1e-12);
    p * (p / q).ln() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).ln()
}

/// The judge's two label logits' targets (yes, no) when the teacher's score asks for log-odds `t`:
/// the difference moved to `t`, split evenly between the two rows (the label softmax reads only
/// the difference).
#[must_use]
pub fn judge_target(yes: f64, no: f64, t: f64) -> [f64; 2] {
    let e = t - (yes - no);
    [yes + e / 2.0, no - e / 2.0]
}

/// A judge read-out as the engine serves it: the rank-one map `y' = r16(y + alpha (w . y) u)` at the
/// final norm, with `u` the direction along which the yes row minus the no row rises by one per
/// unit (`toward`), so the judge's log-odds rise by `alpha (w . y)` and its label reads nothing else
/// that moved.
/// # Errors
/// Shapes that do not fit, or a scale that is not finite.
pub fn readout_unit(key: &str, socket: &str, w: &[f32], toward: &[f32], alpha: f32) -> Result<nmengn::Adapter, Fault> {
    if w.is_empty() || w.len() != toward.len() {
        return Err(bad(format!("{key}: a read-out of {} values toward {}", w.len(), toward.len())));
    }
    unit(key, socket, w.len(), 1, w, toward, alpha)
}

/// The direction `u` of a judge read-out from the judge's yes and no head rows: `(yes - no) /
/// |yes - no|^2`, so `(yes - no) . u = 1`. None when the rows are equal.
#[must_use]
pub fn toward(yes: &[f32], no: &[f32]) -> Option<Vec<f32>> {
    let dlt: Vec<f64> = yes.iter().zip(no).map(|(a, b)| f64::from(*a) - f64::from(*b)).collect();
    let nn: f64 = dlt.iter().map(|x| x * x).sum();
    #[expect(clippy::cast_possible_truncation, reason = "a direction kept at 32 bits")]
    (nn > 0.0 && nn.is_finite()).then(|| dlt.iter().map(|x| (x / nn) as f32).collect())
}

/// The unit of a distilled judge read-out's label and parts (the judge read-out's own form,
/// `adapter::READOUT_KIND`: part `w` `[d]`, `alpha` in the notes, and the direction it acts along,
/// `toward` `[d]`), checked: made for `model`, fitted on `values`, at `width`.
/// # Errors
/// Any of those checks.
pub fn readout_unit_from(label: &Value, parts: &[Part], model: &str, values: &str, width: usize, socket: &str) -> Result<nmengn::Adapter, Fault> {
    if text(label, "kind") != crate::adapter::READOUT_KIND {
        return Err(bad("not a distilled judge read-out"));
    }
    if text(label, "model") != model {
        return Err(bad(format!("made for {}, the judge runs {model}", text(label, "model"))));
    }
    let notes = label.get("notes").cloned().unwrap_or(Value::Null);
    let fitted = text(&notes, "values");
    if fitted != values {
        return Err(bad(format!("fitted on {fitted}, the judge runs on {values}")));
    }
    let key = format!("{}@{}", text(label, "id"), label.get("version").and_then(Value::integer).unwrap_or(0));
    let part = |n: &str| parts.iter().find(|p| p.0 == n);
    let (Some(w), Some(u)) = (part("w"), part("toward")) else { return Err(bad(format!("{key}: no w or toward part"))) };
    if w.2.len() != width || u.2.len() != width {
        return Err(bad(format!("{key}: a read-out of {} values does not fit the judge's width {width}", w.2.len())));
    }
    #[expect(clippy::cast_possible_truncation, reason = "a scale stored as text")]
    let alpha = num(&notes, "alpha").unwrap_or(0.0) as f32;
    if !alpha.is_finite() {
        return Err(bad(format!("{key}: a scale that is not finite")));
    }
    readout_unit(&key, socket, &w.2, &u.2, alpha)
}

/// What a distilled cartridge does, as its label says it, by role and kind.
#[must_use]
pub fn does(role: &str, kind: &str) -> String {
    match (role, kind) {
        ("reranker", "readout") => "Distilled judgment for the Reranker: a read-out at its final norm, fitted so its yes or no score follows a teacher's scores of (query, document) pairs".to_owned(),
        ("reranker", _) => "Distilled judgment for the Reranker: a low-rank map at its final norm fitted so its yes or no score follows a teacher's scores of (query, document) pairs".to_owned(),
        _ => format!("Distilled output behaviour for the {}: a low-rank map at its final norm fitted so its next-token choices follow a teacher's completions (style, format and wording; not new knowledge)", title(role)),
    }
}

/// Self-check: a margin target lifts only a teacher token that is not already ahead; the teacher
/// target keeps the teacher's order and pushes Writer-only rows below it; the per-position shift
/// moves the rows' logits to their targets (a tiny ridge) and is zero when they are already there;
/// the sums equal a direct computation; a cartridge's label and parts round trip into a unit that
/// refuses other values and other widths.
/// # Errors
/// The first of those that fails.
pub fn check() -> Result<(), String> {
    let t = margin_target(&[1.0, 2.0, 0.5]);
    if (t[0] - 4.0).abs() > 1e-12 || t[1] != 2.0 || margin_target(&[5.0, 1.0]) != [5.0, 1.0] {
        return Err(format!("nmapi.distil: margin target {t:?}"));
    }
    let tt = teacher_target(&[0.0, 3.0, 1.0], &[-0.1, -2.0]);
    if !(tt[0] > tt[1] && (tt[0] - 3.0).abs() < 1e-12 && tt[2] < tt[1]) {
        return Err(format!("nmapi.distil: teacher target {tt:?}"));
    }
    let r0 = [1.0_f32, 0.0, 0.0, 0.5];
    let r1 = [0.0_f32, 1.0, 0.5, 0.0];
    let rows: [&[f32]; 2] = [&r0, &r1];
    let h = [0.2_f32, -0.1, 0.3, 0.0];
    let z: Vec<f64> = rows.iter().map(|r| r.iter().zip(&h).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum()).collect();
    let goal = [z[0] + 1.0, z[1] - 0.5];
    let dl = target_delta(&rows, &z, &goal, 1e-9).ok_or("nmapi.distil: the small solve failed")?;
    for (r, g) in rows.iter().zip(goal) {
        let got: f64 = r.iter().zip(&h).zip(&dl).map(|((w, x), d)| f64::from(*w) * (f64::from(*x) + d)).sum();
        if (got - g).abs() > 1e-6 {
            return Err(format!("nmapi.distil: shifted logit {got}, wanted {g}"));
        }
    }
    if target_delta(&rows, &z, &z, MU).is_none_or(|d| d.iter().any(|x| x.abs() > 1e-12)) {
        return Err("nmapi.distil: a position already on target moved".into());
    }
    let mut s = Sums::new(4);
    let d1 = [0.5_f32, 0.0, -1.0, 0.25];
    s.add(&[(&h, &d1, 1.0), (&r0, &d1, 2.0)]);
    let want = |i: usize, j: usize| f64::from(h[i]) * (f64::from(h[j]) + f64::from(d1[j])) + 2.0 * f64::from(r0[i]) * (f64::from(r0[j]) + f64::from(d1[j]));
    if s.n != 2 || (0..4).any(|i| (0..4).any(|j| (s.cy[i * 4 + j] - want(i, j)).abs() > 1e-9)) {
        return Err("nmapi.distil: the ridge sums".into());
    }
    let notes = Value::Object([("values".to_owned(), Value::String("v".into())), ("scale".to_owned(), Value::Number("0.5".into()))].into_iter().collect());
    let lab = label("writer-distil-check", "m", "w", "self-check", notes);
    let parts: Vec<Part> = vec![("down".to_owned(), vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]), ("up".to_owned(), vec![4, 1], vec![0.0, 2.0, 0.0, 0.0])];
    let (l, p) = crate::adapter::read_bytes(&crate::adapter_fit::encode_parts(lab, &parts)).map_err(|e| e.to_string())?;
    let u = unit_from(&l, &p, "m", "v", 4, "n").map_err(|e| e.to_string())?;
    // y = [1, 0, 0, 0]: down gives 1, up adds 0.5 * 2 = 1 to channel 1.
    if u.delta(&[1.0, 0.0, 0.0, 0.0], 1) != [0.0, 1.0, 0.0, 0.0] {
        return Err(format!("nmapi.distil: unit delta {:?}", u.delta(&[1.0, 0.0, 0.0, 0.0], 1)));
    }
    if unit_from(&l, &p, "m", "other", 4, "n").is_ok() || unit_from(&l, &p, "m", "v", 8, "n").is_ok() || unit_from(&l, &p, "other-model", "v", 4, "n").is_ok() {
        return Err("nmapi.distil: a cartridge loaded on other values, width or model".into());
    }
    if cartridge_id("writer", "Terse notes, v2!") != "writer-distil-terse-notes-v2" || role_of("writer-distil-x") != Some("writer") || not_built("writer", "adapter", "lens").is_some() || not_built("embedder", "filter", "channel").is_none() || not_built("writer", "filter", "lens").is_none() || serving_rows(20_000, &[19_000, 3]).len() != ACTIVE_FIRST + 1 {
        return Err("nmapi.distil: ids or serving rows".into());
    }
    check_judge()
}

/// Self-check of the judge's parts: the targets move only the score's difference to the teacher's
/// log-odds; the relative entropy is zero on agreement and positive otherwise; a read-out raises
/// the judge's log-odds by exactly `alpha (w . y)` and round trips through its label; the table
/// builds the Reader and the Reranker and refuses a read-out anywhere but the Reranker.
/// # Errors
/// The first of those that fails.
fn check_judge() -> Result<(), String> {
    let [y, n] = judge_target(1.0, 0.5, 2.0);
    if (y - n - 2.0).abs() > 1e-12 || (y + n - 1.5).abs() > 1e-12 {
        return Err(format!("nmapi.distil: judge target {y} {n}"));
    }
    if bernoulli_kl(0.7, score_logit(0.7)).abs() > 1e-9 || bernoulli_kl(0.9, -1.0) <= 0.0 || (sigmoid(score_logit(0.25)) - 0.25).abs() > 1e-9 || score_logit(1.0) > 4.6 || score_logit(0.0) < -4.6 {
        return Err("nmapi.distil: score log-odds or relative entropy".into());
    }
    let yes = [1.0_f32, 0.0, 2.0, 0.0];
    let no = [0.0_f32, 1.0, 0.0, 0.0];
    let u = toward(&yes, &no).ok_or("nmapi.distil: no read-out direction")?;
    let dz: f32 = yes.iter().zip(&no).zip(&u).map(|((a, b), c)| (a - b) * c).sum();
    if (dz - 1.0).abs() > 1e-6 || toward(&yes, &yes).is_some() {
        return Err(format!("nmapi.distil: read-out direction moves the score by {dz}"));
    }
    let w = [0.5_f32, 0.25, 0.0, 1.0];
    let notes = Value::Object([("values".to_owned(), Value::String("v".into())), ("alpha".to_owned(), Value::Number("2".into()))].into_iter().collect());
    let lab = crate::adapter_fit::label(&crate::adapter_fit::LabelIn { id: "reranker-distil-readout-check", version: 1, model: "judge", weights: "w", kind: crate::adapter::READOUT_KIND, does: &does("reranker", "readout"), made_from: "self-check", notes });
    let parts: Vec<Part> = vec![("w".to_owned(), vec![4], w.to_vec()), ("toward".to_owned(), vec![4], u.clone())];
    let (l, p) = crate::adapter::read_bytes(&crate::adapter_fit::encode_parts(lab, &parts)).map_err(|e| e.to_string())?;
    let unit = any_unit_from(&l, &p, "judge", "v", 4, "n").map_err(|e| e.to_string())?;
    let h = [1.0_f32, 2.0, 0.0, 0.5];
    let dl = unit.delta(&h, 1);
    let rise: f32 = yes.iter().zip(&no).zip(&dl).map(|((a, b), c)| (a - b) * c).sum();
    let want = 2.0 * w.iter().zip(&h).map(|(a, b)| a * b).sum::<f32>();
    if (rise - want).abs() > 0.02 * want.abs().max(1.0) {
        return Err(format!("nmapi.distil: the read-out raised the score by {rise}, wanted {want}"));
    }
    if any_unit_from(&l, &p, "judge", "other", 4, "n").is_ok() || any_unit_from(&l, &p, "judge", "v", 8, "n").is_ok() || any_unit_from(&l, &p, "writer", "v", 4, "n").is_ok() {
        return Err("nmapi.distil: a read-out loaded on other values, width or model".into());
    }
    let built = ["lens", "teacher-loop"].iter().all(|t| not_built("reader", "adapter", t).is_none() && not_built("reranker", "adapter", t).is_none())
        && not_built("reader", "filter", "channel").is_none()
        && not_built("reranker", "readout", "readout").is_none()
        && not_built("reranker", "readout", "teacher-loop").is_none();
    let refused = not_built("writer", "readout", "readout").is_some() && not_built("reader", "readout", "teacher-loop").is_some() && not_built("embedder", "filter", "channel").is_some() && not_built("writer", "adapter", "unit").is_some() && not_built("reranker", "adapter", "steer").is_some() && not_built("embedder", "adapter", "lens").is_none() && not_built("reader", "adapter", "unit").is_none() && not_built("writer", "adapter", "steer").is_none() && not_built("reranker", "adapter", "unit").is_some();
    if !built || !refused || role_of("reranker-distil-readout-x") != Some("reranker") {
        return Err("nmapi.distil: the table of what is built".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_section_check_passes() {
        assert_eq!(super::check(), Ok(()));
    }

    #[test]
    fn a_steer_cartridge_round_trips_and_refuses_what_does_not_fit() {
        use super::*;
        let notes = Value::Object([("values".to_owned(), Value::String("v".into())), ("layer".to_owned(), Value::Number("3".into())), ("scale".to_owned(), Value::Number("0.5".into()))].into_iter().collect());
        let lab = crate::adapter_fit::label(&crate::adapter_fit::LabelIn { id: "writer-distil-steer-x", version: 1, model: "m", weights: "w", kind: STEER_KIND, does: "test", made_from: "test", notes });
        let parts: Vec<Part> = vec![("v".to_owned(), vec![4], vec![0.5, -0.25, 0.0, 1.0])];
        let (l, p) = crate::adapter::read_bytes(&crate::adapter_fit::encode_parts(lab, &parts)).unwrap();
        let (layer, v, scale, key) = steer_from(&l, &p, "m", "v", 4, 4).unwrap();
        assert_eq!((layer, v.len(), key.as_str()), (3, 4, "writer-distil-steer-x@1"));
        assert!((scale - 0.5).abs() < 1e-6);
        assert!(steer_from(&l, &p, "m", "v", 4, 3).is_err(), "a layer the model does not have");
        assert!(steer_from(&l, &p, "m", "other", 4, 4).is_err(), "other values");
        assert!(steer_from(&l, &p, "m", "v", 8, 4).is_err(), "another width");
        assert!(steer_from(&l, &p, "other", "v", 4, 4).is_err(), "another model");
        // Nothing seated: the request maps are the project lens's alone, and none is in force.
        assert!(query_lens(&[1.0, 0.0]).is_none());
        assert_eq!(lens_key(), "none");
    }
}
