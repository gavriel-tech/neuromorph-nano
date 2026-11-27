//! Distillation into a MicroModel's learned adapter cartridge (the lab's `distil` build), and the
//! local teacher that writes curated teacher pairs (`distil teach`).
//!
//! `run distil build <role> [--set <name>] [--technique lens|teacher-loop] [--base <file>|--open]
//! [--rounds <n>]`: reads the healthy rows of the distil store (`nmtype::examples::read_distil`,
//! the Nano's own `memory/training/distil.jsonl`; one set, or every set), runs the role's MicroModel
//! under teacher forcing on each pair, fits the low-rank map at its final norm toward the teacher's
//! targets (`nmapi::distil`), chooses the ridge strength, rank and scale on the tune split, and gates
//! the result on the held-out split. The Writer and the Reader are built from teacher completions
//! (`build`); the Reranker from teacher scores (`build_judge`: an adapter, a filter or a judge
//! read-out); `nmapi::distil::BUILT` lists what is built, and every other role and technique
//! answers with what it still needs.
//!
//! The gate (thresholds fixed before any run, `GATE_*`), all on the held-out split, which nothing
//! chose from:
//! 1. teacher-token loss falls (mean natural-log loss per completion token over the serving
//!    vocabulary) and top-1 agreement with the teacher rises;
//! 2. a shuffled-target control (the same fit with each position's target shift taken from another
//!    position) gains at most `GATE_CONTROL` of the real gain;
//! 3. drift on neutral text (the Nano's own source read as plain text, no teacher anywhere near it)
//!    stays within `GATE_KL` nats per token (mean KL of the base's next-token distribution to the
//!    distilled one);
//! 4. the Writer's note check is not worse: on held-out prompts, the embedder ranks the prompt's
//!    own passage above its `CHECK_NEIGHBOURS` nearest neighbours from the distilled Writer's answer
//!    at least as often as from the base's;
//! 5. stable over repeated trials: the fit refitted three times, each without one third of the fit
//!    pairs, gains on the held-out split every time.
//!
//! Passed: the cartridge is written as the next version of `<role>-distil-<set>` on the adapter
//! shelf, labelled with its model, open weights, base, socket, gate figures, lineage and outcome;
//! a new cartridge id starts unplugged. Refused: nothing is written; the report holds the figures.
//! Every run writes `runs/<date>-distil-<set>-<ms>.txt` and prints `WRITTEN:` or `REFUSED:`.
//!
//! `run distil teach --prompts <file> --out <file> [--teacher <model>] [--instruction <text>]
//! [--max-new <n>] [--topk <k>] [--limit <n>]`: a local open-weights teacher (an installed Qwen3
//! model on its open weights, never a base cartridge) answers each prompt, with the instruction
//! placed before it when one is given (the instruction is then what the student learns to follow
//! without being told), greedy, keeping its top log-probabilities per token; answers that do not end
//! within `max-new` tokens are left out. No network call is made.
use nmapi::distil as dl;
use nmengn::model::{Act, Cache, Model, RunOpts, values};
use nmengn::placement::Device;
use nmjson::Value;
use nmtype::examples::{self as ex, DistilRow, Split};
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("distil", FaultKind::Execution, m)
}

/// Ridge strengths tried (times the mean energy of the final-norm states). Widened after the first
/// run on writer-base@4, where every setting of the narrower grid (0.01 to 1, scales 0.5 to 1.5)
/// drifted past the neutral bound on the tune split.
const RIDGES: [f64; 4] = [0.01, 0.1, 1.0, 10.0];
/// Ranks tried; the widest is fitted once and the leading directions taken.
const RANKS: [usize; 3] = [8, 32, 64];
/// Scales tried on the fitted map.
const SCALES: [f64; 5] = [0.125, 0.25, 0.5, 1.0, 1.5];
/// The control may gain at most this share of the real gain on the held-out split.
const GATE_CONTROL: f64 = 0.25;
/// Neutral-text drift bound, nats per token.
const GATE_KL: f64 = 0.10;
/// Repeated trials of the stability check.
const TRIALS: usize = 3;
/// The fewest pairs per split a build takes.
const MIN_FIT: usize = 24;
const MIN_TUNE: usize = 4;
const MIN_HELD: usize = 4;
/// The longest pair a build reads (tokens).
const MAX_TOKENS: usize = 2048;
/// Held-out prompts the note check writes from (two arms each).
const NOTE_ROWS: usize = 24;
/// Longest answer of the note check (tokens), as a Writer note.
const NOTE_TOKENS: usize = 48;
/// Rows per head pass on the card.
const HEAD_CHUNK: usize = 192;
/// Rounds the `teacher-loop` technique runs at most.
const ROUNDS: usize = 4;
/// Neutral chunks and their length in tokens.
const NEUTRAL: usize = 12;
const NEUTRAL_TOKENS: usize = 128;

fn arg<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).map(String::as_str)
}
fn flag(args: &[String], k: &str) -> bool {
    args.iter().any(|a| a == k)
}

/// `run distil ...`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let sub = args.get(2).map_or("", String::as_str);
    let res = match sub {
        "build" => build(home, args),
        "teach" => teach(home, args),
        _ => Err(bad("usage: run distil build <role> [--target adapter|filter|readout|preference] [--set <name>] [--technique lens|channel|readout|teacher-loop] [--base <file>|--open] | run distil teach --prompts <file> --out <file> [--teacher <model>] [--instruction <text>]")),
    };
    match res {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("distil: {e}");
            println!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Distillation into a preference cartridge: the teacher's verdicts on completions (accept or
/// reject), each with its circumstance (the paths' file kinds and top folders, the tool), counted
/// per circumstance into the preference values the serving path reads (`nmlearn::pref_fit`), and
/// written only when the held-out verdicts are predicted better (Brier score) than with no
/// cartridge. Runs on the processor; added unplugged, beside every other preference cartridge.
fn build_preference(home: &Path, set: &str) -> Result<String, Fault> {
    use nmlearn::pref_fit::{ABSENT_BRIER, MIN_HELD, PrefModel, PrefRow, brier, features};
    let rows: Vec<DistilRow> = rows_of(home, set)?.into_iter().filter(|r| !r.verdict.is_empty()).collect();
    let as_pref = |r: &DistilRow| {
        let v = Value::Object(
            [("paths", Value::Array(r.paths.iter().map(|p| Value::String(p.clone())).collect())), ("tool", Value::String(r.tool.clone()))]
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect(),
        );
        PrefRow { features: features(&v), positive: r.verdict == "accept", held: r.split == Split::Held }
    };
    let all: Vec<PrefRow> = rows.iter().map(as_pref).collect();
    let fit: Vec<&PrefRow> = all.iter().filter(|r| !r.held).collect();
    let held: Vec<&PrefRow> = all.iter().filter(|r| r.held).collect();
    let mut out = String::new();
    let _ = writeln!(out, "distillation into a preference cartridge, set {set}, technique readout (verdict counts per circumstance)");
    let _ = writeln!(out, "judged pairs (healthy only): fit {}, held out {}", fit.len(), held.len());
    if held.len() < MIN_HELD || fit.is_empty() {
        return Err(bad(format!("too few judged pairs: fit {} (at least 1), held out {} (at least {MIN_HELD}); import teacher pairs with a verdict (accept or reject) and the paths they concern", fit.len(), held.len())));
    }
    let model = PrefModel::fit(&fit);
    let cand = brier(&held, &|f| model.predict(f)).unwrap_or(ABSENT_BRIER);
    let _ = writeln!(out, "held-out Brier score: {cand:.4} with the cartridge, {ABSENT_BRIER:.4} without it");
    let report = report_file(home, &format!("pref-{}", dl::slug(set)));
    if cand >= ABSENT_BRIER - 1e-9 {
        let _ = writeln!(out, "refused: the held-out verdicts are not predicted better than without a cartridge");
        let _ = std::fs::write(&report, crate::with_overlay(&out));
        return Err(bad(format!("held-out Brier {cand:.4} does not beat {ABSENT_BRIER:.4} without a cartridge; nothing written (report {})", report.display())));
    }
    let mut lib = nmlearn::library::Library::open(home).map_err(|e| bad(e.to_string()))?;
    let id = format!("preference-distil-{}", dl::slug(set));
    let version = lib.cartridges.iter().filter(|c| c.id == id).map(|c| c.version).max().unwrap_or(0) + 1;
    let teachers: std::collections::BTreeSet<&str> = rows.iter().map(|r| r.teacher.as_str()).collect();
    let spec = nmlearn::library::CartridgeSpec {
        id: id.clone(),
        version,
        family: nmlearn::library::PREFERENCE.to_owned(),
        domain: "preferences".to_owned(),
        scope: "general".to_owned(),
        model: "settings".to_owned(),
        weights: String::new(),
        layers: None,
        name: format!("distilled preferences, set {set}"),
        description: "How likely each circumstance (file kind, folder, tool) is to be accepted, distilled from a teacher's verdicts on completions.".to_owned(),
        made_from: format!("{} judged teacher pairs of set {set} from {} ({} held out for the gate)", rows.len(), teachers.into_iter().collect::<Vec<_>>().join(", "), held.len()),
        effect: format!("held-out Brier {cand:.4} against {ABSENT_BRIER:.4} without it"),
    };
    let notes = Value::Object(
        [
            ("distilled", Value::Bool(true)),
            ("technique", Value::String("readout".into())),
            ("set", Value::String(set.to_owned())),
            ("fit_rows", Value::Number(fit.len().to_string())),
            ("held_rows", Value::Number(held.len().to_string())),
            ("held_brier", Value::Number(format!("{cand:.6}"))),
            ("absent_brier", Value::Number(format!("{ABSENT_BRIER:.6}"))),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect(),
    );
    let key = lib.add_cartridge(&spec, &nmlearn::library::Filter::Settings(model.to_settings()), notes, Vec::new(), false).map_err(|e| bad(e.to_string()))?;
    lib.save().map_err(|e| bad(e.to_string()))?;
    let _ = writeln!(out, "written {key}, unplugged: plug it in on Cartridges (whole-engine rack) or put it in the preference stack");
    let _ = std::fs::write(&report, crate::with_overlay(&out));
    Ok(format!("written {key} (held-out Brier {cand:.4} against {ABSENT_BRIER:.4}); report {}", report.display()))
}

fn report_file(home: &Path, what: &str) -> PathBuf {
    let date = nmlearn::cartridge::date_of(nmtype::now_ms());
    let mut stamp = nmtype::now_ms();
    loop {
        let p = home.join(format!("runs/{date}-distil-{what}-{stamp}.txt"));
        if !p.exists() {
            return p;
        }
        stamp += 1;
    }
}

fn device(m: &Model) -> Device {
    if m.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu }
}

// ---------------------------------------------------------------- the teacher

/// Binds an installed model on its open weights (the teacher), linear weights at 8 bits.
fn bind_open(home: &Path, id: &str) -> Result<Model, Fault> {
    let pop = nmengn::model::Population::open(home, id, true, true)?;
    let mut a = nmengn::assembly::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?;
    for n in a.nodes.iter_mut().filter(|n| n.op == "linear") {
        n.precision = "q8".to_owned();
    }
    Model::bind(&pop, a)
}

/// Prompts from a file: JSON lines with a `prompt` field, or plain text with one prompt per block
/// (blocks split by a line holding only `---`).
fn read_prompts(file: &Path) -> Result<Vec<String>, Fault> {
    let t = std::fs::read_to_string(file).map_err(|e| bad(format!("{}: {e}", file.display())))?;
    let json: Vec<String> = t.lines().filter_map(|l| nmjson::parse(l.trim().as_bytes()).ok()).filter_map(|v| v.get("prompt").and_then(Value::string).ok().map(str::to_owned)).collect();
    if !json.is_empty() {
        return Ok(json);
    }
    Ok(t.split("\n---\n").map(str::trim).filter(|p| !p.is_empty()).map(str::to_owned).collect())
}

/// A teacher's answer: its tokens and each step's top log-probabilities (the end step included).
type Taught = (Vec<i32>, Vec<Vec<(u32, f32)>>);

/// Greedy answer of `m` to `ids` over the full vocabulary with each step's top log-probabilities;
/// None when no end token came within `max_new`.
fn answer_with_topk(m: &Model, ids: &[i32], max_new: usize, k: usize) -> Result<Option<Taught>, Fault> {
    let dev = device(m);
    let head = (m.role_node("head")?, "logits".to_owned());
    let opts = RunOpts { cache: Some(Arc::new(Cache::default())), ..RunOpts::default() };
    let mut l = values(m.forward_at(dev, ids, 0, std::slice::from_ref(&head), &opts)?.get(&head).ok_or_else(|| bad("no logits"))?)?.to_vec();
    let mut toks = Vec::new();
    let mut tops = Vec::new();
    for step in 0..max_new {
        let mx = l.iter().copied().fold(f32::MIN, f32::max);
        let lse = f64::from(mx) + l.iter().map(|x| (f64::from(*x) - f64::from(mx)).exp()).sum::<f64>().ln();
        let mut idx: Vec<usize> = (0..l.len()).collect();
        let kk = k.min(idx.len()).max(1);
        idx.select_nth_unstable_by(kk - 1, |a, b| l[*b].total_cmp(&l[*a]));
        idx.truncate(kk);
        idx.sort_by(|a, b| l[*b].total_cmp(&l[*a]));
        #[expect(clippy::cast_possible_truncation, reason = "log-probabilities kept at 32 bits")]
        let top: Vec<(u32, f32)> = idx.iter().map(|i| (u32::try_from(*i).unwrap_or(0), (f64::from(l[*i]) - lse) as f32)).collect();
        let next = i32::try_from(idx.first().copied().unwrap_or(0)).unwrap_or(0);
        tops.push(top);
        if next == dl::END_OF_TURN || next == dl::END_OF_TEXT {
            return Ok(Some((toks, tops)));
        }
        toks.push(next);
        l = values(m.forward_at(dev, &[next], ids.len() + step, std::slice::from_ref(&head), &opts)?.get(&head).ok_or_else(|| bad("no logits"))?)?.to_vec();
    }
    Ok(None)
}

fn teach(home: &Path, args: &[String]) -> Result<String, Fault> {
    let prompts_file = PathBuf::from(arg(args, "--prompts").ok_or_else(|| bad("teach needs --prompts <file>"))?);
    let out_file = PathBuf::from(arg(args, "--out").ok_or_else(|| bad("teach needs --out <file>"))?);
    let teacher = arg(args, "--teacher").unwrap_or(nmapi::neural::WRITER);
    let instruction = arg(args, "--instruction").unwrap_or("");
    let max_new = arg(args, "--max-new").and_then(|v| v.parse().ok()).unwrap_or(48_usize);
    let k = arg(args, "--topk").and_then(|v| v.parse().ok()).unwrap_or(8_usize).clamp(1, ex::DISTIL_TOPK_MAX);
    let limit = arg(args, "--limit").and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let licence = arg(args, "--licence").unwrap_or("Apache-2.0 (teacher outputs of the Qwen3 open weights)");
    let prompts: Vec<String> = read_prompts(&prompts_file)?.into_iter().take(limit).collect();
    if prompts.is_empty() {
        return Err(bad(format!("{}: no prompts", prompts_file.display())));
    }
    let t0 = Instant::now();
    let m = bind_open(home, teacher)?;
    let tok = &m.pop.tokenizer;
    let id = if instruction.is_empty() { format!("{teacher} (open weights)") } else { format!("{teacher} (open weights, instructed {})", nmhash::sha256_bytes(instruction.as_bytes()).get(..8).unwrap_or("")) };
    let mut lines = Vec::new();
    let (mut kept, mut open_ended) = (0, 0);
    for p in &prompts {
        let shown = if instruction.is_empty() { p.clone() } else { format!("{instruction}\n\n{p}") };
        let ids = tok.encode(&dl::chat_prompt(&shown)).map_err(|e| bad(e.to_string()))?;
        if ids.len() > MAX_TOKENS {
            continue;
        }
        match answer_with_topk(&m, &ids, max_new, k)? {
            Some((toks, tops)) => {
                let completion = tok.decode(&toks).trim().to_owned();
                if completion.is_empty() {
                    continue;
                }
                let row = DistilRow {
                    id: ex::distil_id(p, &completion, &id),
                    set: String::new(),
                    prompt: p.clone(),
                    completion,
                    teacher: id.clone(),
                    licence: licence.to_owned(),
                    reference: String::new(),
                    template: "chat".into(),
                    topk: tops,
                    split: ex::distil_split(p),
                    outcome: String::new(),
                    at: nmtype::now_ms(),
                    verdict: String::new(),
                    paths: Vec::new(),
                    tool: String::new(),
                    score: None,
                };
                lines.push(row.to_value().encode().map_err(|e| bad(e.to_string()))?);
                kept += 1;
            }
            None => open_ended += 1,
        }
    }
    if let Some(d) = out_file.parent() {
        std::fs::create_dir_all(d).map_err(|e| bad(format!("{}: {e}", d.display())))?;
    }
    std::fs::write(&out_file, lines.join("\n") + "\n").map_err(|e| bad(format!("{}: {e}", out_file.display())))?;
    Ok(format!(
        "teacher {id}: {kept} pairs written to {} ({open_ended} answers did not end within {max_new} tokens and were left out; {} prompts read), top {k} log-probabilities per token, {:.0} s\n",
        out_file.display(),
        prompts.len(),
        t0.elapsed().as_secs_f64()
    ))
}

// ---------------------------------------------------------------- the build

/// One scored position: the final-norm state, the teacher's token as an index into the serving
/// rows, the teacher's top tokens (indices, log-probabilities) when known, its pair.
struct Pos {
    h: Vec<f32>,
    t: usize,
    tk: Vec<(usize, f32)>,
    row: usize,
}

/// Measures of one arm over a set of positions.
#[derive(Clone, Copy, Default, Debug)]
struct Score {
    nll: f64,
    top1: f64,
    n: usize,
}

/// The serving vocabulary's head: the Writer's head rows over the serving rows `rows`, read from the
/// engine itself (unit vectors seeded at the final norm, so each value is the head's own).
struct Head<'a> {
    m: &'a Model,
    fin: String,
    head: String,
    rows: Arc<Vec<usize>>,
    /// `[rows.len(), d]` row-major.
    w: Vec<f32>,
}

impl<'a> Head<'a> {
    fn new(m: &'a Model, rows: Vec<usize>) -> Result<Self, Fault> {
        let mut h = Self { m, fin: m.role_node("final_norm")?, head: m.role_node("head")?, rows: Arc::new(rows), w: Vec::new() };
        let d = m.pop.dims.hidden;
        let na = h.rows.len();
        let mut w = vec![0.0_f32; na * d];
        let mut i = 0;
        while i < d {
            let c = HEAD_CHUNK.min(d - i);
            let mut basis = vec![0.0_f32; c * d];
            for k in 0..c {
                basis[k * d + i + k] = 1.0;
            }
            let lg = h.logits(&basis, c)?;
            for k in 0..c {
                for a in 0..na {
                    w[a * d + i + k] = lg[k * na + a];
                }
            }
            i += c;
        }
        h.w = w;
        Ok(h)
    }
    /// Logits over the serving rows for `rows` final-norm states (`hs` `[rows, d]`).
    fn logits(&self, hs: &[f32], rows: usize) -> Result<Vec<f32>, Fault> {
        let d = self.m.pop.dims.hidden;
        let na = self.rows.len();
        let target = (self.head.clone(), "logits".to_owned());
        let opts = RunOpts { all_positions: true, head_rows: Arc::clone(&self.rows), ..RunOpts::default() };
        let mut out = Vec::with_capacity(rows * na);
        let mut i = 0;
        while i < rows {
            let c = HEAD_CHUNK.min(rows - i);
            let seed = hs.get(i * d..(i + c) * d).ok_or_else(|| bad("states out of range"))?.to_vec();
            let r = self.m.run_from(device(self.m), vec![(self.fin.clone(), "y".to_owned(), Act::Cpu(seed))], std::slice::from_ref(&target), c, &opts)?;
            out.extend_from_slice(values(r.get(&target).ok_or_else(|| bad("no logits"))?)?);
            i += c;
        }
        Ok(out)
    }
    /// Calls `f` with each state's index and its logits over the serving rows, one head pass of
    /// `HEAD_CHUNK` states at a time, so the logits of a whole split are never held at once.
    fn each(&self, hs: &[f32], rows: usize, mut f: impl FnMut(usize, &[f32]) -> Result<(), Fault>) -> Result<(), Fault> {
        let d = self.m.pop.dims.hidden;
        let na = self.rows.len().max(1);
        let mut i = 0;
        while i < rows {
            let c = HEAD_CHUNK.min(rows - i);
            let lg = self.logits(hs.get(i * d..(i + c) * d).ok_or_else(|| bad("states out of range"))?, c)?;
            for (k, row) in lg.chunks_exact(na).enumerate() {
                f(i + k, row)?;
            }
            i += c;
        }
        Ok(())
    }
    fn row(&self, a: usize) -> &[f32] {
        let d = self.m.pop.dims.hidden;
        self.w.get(a * d..(a + 1) * d).unwrap_or(&[])
    }
}

fn lse(row: &[f32]) -> f64 {
    let mx = row.iter().copied().fold(f32::MIN, f32::max);
    f64::from(mx) + row.iter().map(|x| (f64::from(*x) - f64::from(mx)).exp()).sum::<f64>().ln()
}
fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().fold((0, f32::MIN), |b, (j, x)| if *x > b.1 { (j, *x) } else { b }).0
}

/// A fitted map at the final norm, in the form its cartridge kind serves it: a low-rank unit (an
/// adapter cartridge) or a per-channel gain and bias (a job filter).
enum Fitted {
    Lens(nmengn::Adapter),
    Channel { gain: Vec<f32>, bias: Vec<f32> },
}

impl Fitted {
    /// Applies the map to `rows` states in place, rounded as the engine rounds them (a unit's delta
    /// added and rounded; a filter's one fused multiply-add per value, rounded).
    fn apply(&self, hs: &mut [f32], rows: usize) {
        match self {
            Self::Lens(u) => {
                let dlt = u.delta(hs, rows);
                for (x, d) in hs.iter_mut().zip(&dlt) {
                    *x = nmengn::half::r16(*x + d);
                }
            }
            Self::Channel { gain, bias } => {
                for row in hs.chunks_mut(gain.len().max(1)) {
                    for ((x, g), b) in row.iter_mut().zip(gain).zip(bias) {
                        *x = nmengn::half::r16(x.mul_add(*g, *b));
                    }
                }
            }
        }
    }
}

/// The states with a map applied (`None`: as they are).
fn mapped(ps: &[&Pos], u: Option<&Fitted>) -> Vec<f32> {
    let mut hs: Vec<f32> = ps.iter().flat_map(|p| p.h.iter().copied()).collect();
    if let Some(u) = u {
        u.apply(&mut hs, ps.len());
    }
    hs
}

/// Each pair's summed loss and positions.
type PerPair = BTreeMap<usize, (f64, usize)>;
/// Each pair's positions before its first divergence, its positions, and whether it diverged.
type Divergence = BTreeMap<usize, (usize, usize, bool)>;

/// Teacher-token loss and top-1 over positions under a map, with each pair's loss and each pair's
/// first divergence (the share of its positions before the first one whose top choice is not the
/// teacher's).
fn score(head: &Head, ps: &[&Pos], u: Option<&Fitted>) -> Result<(Score, PerPair, Divergence), Fault> {
    let mut s = Score::default();
    let mut per = PerPair::new();
    let mut div = Divergence::new();
    head.each(&mapped(ps, u), ps.len(), |i, row| {
        let Some(p) = ps.get(i) else { return Ok(()) };
        let nll = lse(row) - f64::from(row.get(p.t).copied().unwrap_or(f32::MIN));
        let hit = argmax(row) == p.t;
        s.nll += nll;
        s.top1 += f64::from(u8::from(hit));
        s.n += 1;
        let e = per.entry(p.row).or_default();
        e.0 += nll;
        e.1 += 1;
        let dv = div.entry(p.row).or_insert((0, 0, false));
        dv.1 += 1;
        if !dv.2 {
            if hit { dv.0 += 1 } else { dv.2 = true }
        }
        Ok(())
    })?;
    let n = s.n.max(1) as f64;
    s.nll /= n;
    s.top1 /= n;
    Ok((s, per, div))
}

/// Mean share of each pair's positions before its first divergence.
fn agreeing_prefix(div: &Divergence) -> f64 {
    if div.is_empty() {
        return 0.0;
    }
    div.values().map(|(a, n, _)| *a as f64 / (*n).max(1) as f64).sum::<f64>() / div.len() as f64
}

/// Mean KL of the base's distribution to the mapped one over neutral states (`base` the base's
/// log-probabilities, `[n, na]`).
fn drift(head: &Head, hs: &[f32], n: usize, base: &[f32], u: &Fitted) -> Result<f64, Fault> {
    let na = head.rows.len();
    let mut m = hs.to_vec();
    u.apply(&mut m, n);
    let mut kl = 0.0;
    head.each(&m, n, |i, q| {
        let p = base.get(i * na..(i + 1) * na).unwrap_or(&[]);
        let lq = lse(q);
        kl += p.iter().zip(q).map(|(lp, zq)| { let lp = f64::from(*lp); lp.exp() * (lp - (f64::from(*zq) - lq)) }).sum::<f64>();
        Ok(())
    })?;
    Ok(kl / n.max(1) as f64)
}

/// Log-probabilities over the serving rows for neutral states.
fn log_probs(head: &Head, hs: &[f32], n: usize) -> Result<Vec<f32>, Fault> {
    let na = head.rows.len();
    let mut lg = head.logits(hs, n)?;
    for row in lg.chunks_exact_mut(na.max(1)) {
        let l = lse(row);
        #[expect(clippy::cast_possible_truncation, reason = "log-probabilities kept at 32 bits")]
        row.iter_mut().for_each(|x| *x = (f64::from(*x) - l) as f32);
    }
    Ok(lg)
}

/// Each fit position's target shift (`nmapi::distil::target_delta`); a position whose system does
/// not solve keeps a zero shift.
fn targets(head: &Head, ps: &[&Pos]) -> Result<(Vec<Vec<f32>>, usize, usize), Fault> {
    let na = head.rows.len();
    let mut out = Vec::with_capacity(ps.len());
    let (mut with_topk, mut moved) = (0, 0);
    head.each(&mapped(ps, None), ps.len(), |i, row| {
        let Some(p) = ps.get(i) else { return Ok(()) };
        let mut set: Vec<usize> = if p.tk.is_empty() { vec![p.t] } else { p.tk.iter().map(|(a, _)| *a).collect() };
        let teacher_n = set.len();
        let mut idx: Vec<usize> = (0..na).collect();
        let k = (dl::WRITER_TOP + teacher_n).min(na).max(1);
        idx.select_nth_unstable_by(k - 1, |a, b| row[*b].total_cmp(&row[*a]));
        idx.truncate(k);
        idx.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
        for i in idx {
            if set.len() >= teacher_n + dl::WRITER_TOP {
                break;
            }
            if !set.contains(&i) {
                set.push(i);
            }
        }
        let z: Vec<f64> = set.iter().map(|a| f64::from(row.get(*a).copied().unwrap_or(0.0))).collect();
        let goal = if p.tk.is_empty() {
            dl::margin_target(&z)
        } else {
            with_topk += 1;
            dl::teacher_target(&z, &p.tk.iter().map(|(_, lp)| f64::from(*lp)).collect::<Vec<_>>())
        };
        let rows: Vec<&[f32]> = set.iter().map(|a| head.row(*a)).collect();
        #[expect(clippy::cast_possible_truncation, reason = "shifts kept at 32 bits")]
        let delta: Vec<f32> = match dl::target_delta(&rows, &z, &goal, dl::MU) {
            Some(v) => v.into_iter().map(|x| x as f32).collect(),
            None => vec![0.0; p.h.len()],
        };
        moved += usize::from(delta.iter().any(|x| *x != 0.0));
        out.push(delta);
        Ok(())
    })?;
    Ok((out, with_topk, moved))
}

/// Per-channel sums of the channel fit (`y = g x + b` per channel toward `x + delta`): weighted sums
/// of `x^2`, `x`, `x y`, `y`, and of the weights.
#[derive(Clone)]
struct ChanSums {
    xx: Vec<f64>,
    x: Vec<f64>,
    xy: Vec<f64>,
    y: Vec<f64>,
    w: f64,
}

impl ChanSums {
    fn new(d: usize) -> Self {
        Self { xx: vec![0.0; d], x: vec![0.0; d], xy: vec![0.0; d], y: vec![0.0; d], w: 0.0 }
    }
    fn add(&mut self, h: &[f32], dl: &[f32], w: f64) {
        for (j, (xv, dv)) in h.iter().zip(dl).enumerate() {
            let (x, y) = (f64::from(*xv), f64::from(*xv) + f64::from(*dv));
            self.xx[j] += w * x * x;
            self.x[j] += w * x;
            self.xy[j] += w * x * y;
            self.y[j] += w * y;
        }
        self.w += w;
    }
    fn plus(&mut self, o: &Self) {
        for (a, b) in [(&mut self.xx, &o.xx), (&mut self.x, &o.x), (&mut self.xy, &o.xy), (&mut self.y, &o.y)] {
            a.iter_mut().zip(b).for_each(|(p, q)| *p += q);
        }
        self.w += o.w;
    }
}

/// The fit's sums of one kind by third of the fit pairs.
enum Folds {
    Low(Vec<dl::Sums>),
    Chan(Vec<ChanSums>),
}

/// A fitted map before its rank and scale are chosen: (down, up) at the widest rank, or a gain and
/// bias per channel.
#[derive(Clone)]
enum FitMap {
    Low(Vec<f64>, Vec<f64>),
    Chan(Vec<f64>, Vec<f64>),
}

/// The fit's sums by third of the fit pairs (row index modulo 3), each row weighted; `shift` takes
/// each position's target from another position (the shuffled control).
fn fold_sums(target: &str, d: usize, ps: &[&Pos], deltas: &[Vec<f32>], weights: &BTreeMap<usize, f64>, shift: bool) -> Folds {
    let n = deltas.len();
    let rot = n / 3 + 1;
    let rows_of = |f: usize| -> Vec<(&[f32], &[f32], f64)> {
        ps.iter()
            .enumerate()
            .filter(|(_, p)| p.row % TRIALS == f)
            .map(|(i, p)| {
                let j = if shift { (i + rot) % n.max(1) } else { i };
                (p.h.as_slice(), deltas.get(j).map_or(&[][..], Vec::as_slice), weights.get(&p.row).copied().unwrap_or(1.0))
            })
            .collect()
    };
    if target == "filter" {
        return Folds::Chan(
            (0..TRIALS)
                .map(|f| {
                    let mut s = ChanSums::new(d);
                    for (h, dlt, w) in rows_of(f) {
                        s.add(h, dlt, w);
                    }
                    s
                })
                .collect(),
        );
    }
    Folds::Low(
        (0..TRIALS)
            .map(|f| {
                let rows = rows_of(f);
                let mut s = dl::Sums::new(d);
                for chunk in rows.chunks(512) {
                    s.add(chunk);
                }
                s
            })
            .collect(),
    )
}

fn positions(folds: &Folds) -> usize {
    match folds {
        Folds::Low(f) => f.iter().map(|s| s.n).sum(),
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a count of weighted rows")]
        Folds::Chan(f) => f.iter().map(|s| s.w as usize).sum(),
    }
}

/// The map fitted on every third but `leave_out`, at ridge `c` (and the widest rank for a lens).
fn solve(folds: &Folds, leave_out: Option<usize>, c: f64, rank: usize) -> Option<FitMap> {
    match folds {
        Folds::Low(f) => {
            let d = f.first().map_or(0, |x| x.d);
            let mut t = dl::Sums::new(d);
            for (i, x) in f.iter().enumerate().filter(|(i, _)| Some(*i) != leave_out) {
                let _ = i;
                t.plus(x);
            }
            nmapi::adapter_fit::ridge_low_rank_sums(t.g, &t.cy, d, c, rank).map(|(a, b)| FitMap::Low(a, b))
        }
        Folds::Chan(f) => {
            let d = f.first().map_or(0, |x| x.xx.len());
            let mut t = ChanSums::new(d);
            for (_, x) in f.iter().enumerate().filter(|(i, _)| Some(*i) != leave_out) {
                t.plus(x);
            }
            // Per channel: min sum w (g x + b - y)^2 + c sxx (g - 1)^2 + c w b^2, a 2 x 2 solve.
            let (mut g, mut b) = (vec![1.0; d], vec![0.0; d]);
            for j in 0..d {
                let lg = c * t.xx[j].max(1e-12);
                let lb = c * t.w.max(1e-12);
                let (a11, a12, a22) = (t.xx[j] + lg, t.x[j], t.w + lb);
                let (r1, r2) = (t.xy[j] + lg, t.y[j]);
                let det = a11 * a22 - a12 * a12;
                if det.abs() > 1e-18 {
                    g[j] = (r1 * a22 - a12 * r2) / det;
                    b[j] = (a11 * r2 - a12 * r1) / det;
                }
            }
            Some(FitMap::Chan(g, b))
        }
    }
}

/// The fitted map at rank `r` and strength `scale`, as its cartridge kind serves it.
fn make(key: &str, node: &str, d: usize, map: &FitMap, r: usize, scale: f64) -> Result<Fitted, Fault> {
    match map {
        FitMap::Low(down, up) => {
            let (dn, upr) = dl::leading(down, up, d, r);
            let r = dn.len() / d.max(1);
            #[expect(clippy::cast_possible_truncation, reason = "a scale kept at 32 bits")]
            dl::unit(key, node, d, r, &dn, &upr, scale as f32).map(Fitted::Lens)
        }
        #[expect(clippy::cast_possible_truncation, reason = "gains and biases kept at 32 bits, as a job filter holds them")]
        FitMap::Chan(g, b) => Ok(Fitted::Channel { gain: g.iter().map(|x| (1.0 + scale * (x - 1.0)) as f32).collect(), bias: b.iter().map(|x| (scale * x) as f32).collect() }),
    }
}

/// What the build reads: the rows of one set (or all), healthy only.
fn rows_of(home: &Path, set: &str) -> Result<Vec<DistilRow>, Fault> {
    let all = ex::read_distil(&home.join("memory")).map_err(bad)?;
    Ok(all.into_iter().filter(|r| r.healthy()).filter(|r| set == "all" || r.set == set).collect())
}

/// Neutral text: the Nano's own Rust source read as plain text, in chunks of `NEUTRAL_TOKENS`,
/// alternately tune and held out. Kept in RAM only.
fn neutral_ids(home: &Path, tok: &nmengn::tokenizer::Tokenizer) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let mut files = Vec::new();
    if let Ok(rd) = std::fs::read_dir(home.join("crates")) {
        for e in rd.flatten() {
            if let Ok(src) = std::fs::read_dir(e.path().join("src")) {
                files.extend(src.flatten().map(|f| f.path()).filter(|p| p.extension().is_some_and(|x| x == "rs")));
            }
        }
    }
    files.sort();
    let (mut tune, mut held) = (Vec::new(), Vec::new());
    for f in files.iter().step_by((files.len() / NEUTRAL).max(1)).take(NEUTRAL) {
        let Ok(t) = std::fs::read_to_string(f) else { continue };
        let Ok(ids) = tok.encode_plain(&t) else { continue };
        if ids.len() < NEUTRAL_TOKENS {
            continue;
        }
        let c = ids[ids.len() / 3..ids.len() / 3 + NEUTRAL_TOKENS.min(ids.len() - ids.len() / 3)].to_vec();
        if tune.len() <= held.len() { tune.push(c) } else { held.push(c) }
    }
    (tune, held)
}

/// Offers the role's serving base cartridge (or `--base <file>`) before its MicroModel opens, at
/// the precision and arithmetic the serving process uses.
fn seat_base(home: &Path, args: &[String], role: &str) -> String {
    nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Ok(mut p) = nmengn::model::PRECISION.write() {
        p.writer = p.writer.or(Some(8));
        p.embedder = p.embedder.or(Some(8));
    }
    let model = nmtype::role_models::model(role);
    let file = if flag(args, "--open") {
        None
    } else if let Some(f) = arg(args, "--base") {
        Some(PathBuf::from(f))
    } else {
        nmlearn::library::Library::open(home).ok().and_then(|lib| lib.cartridge(&format!("{role}-base")).and_then(|c| c.files.first().map(|f| lib.cartridge_dir(c).join(&f.name))))
    };
    if let Ok(mut g) = nmapi::neural::BASE_CARTRIDGES.write() {
        g.remove(&model);
        if let Some(f) = &file {
            g.insert(model.clone(), vec![f.clone()]);
        }
    }
    file.map_or_else(|| "open weights".to_owned(), |f| f.display().to_string())
}

/// The base a model serves on: `writer-base@4`, or `open:<model id>`.
fn made_on(m: &Model) -> String {
    let src = &m.pop.source;
    src.strip_prefix("cartridge ").map_or_else(|| format!("open:{}", m.pop.card.id), str::to_owned)
}

/// A greedy answer over the serving rows (no top log-probabilities), as the note check reads it.
fn answer(m: &Model, rows: &Arc<Vec<usize>>, ids: &[i32]) -> Result<Vec<i32>, Fault> {
    let dev = device(m);
    let head = (m.role_node("head")?, "logits".to_owned());
    let opts = RunOpts { cache: Some(Arc::new(Cache::default())), head_rows: Arc::clone(rows), ..RunOpts::default() };
    let mut l = values(m.forward_at(dev, ids, 0, std::slice::from_ref(&head), &opts)?.get(&head).ok_or_else(|| bad("no logits"))?)?.to_vec();
    let mut out = Vec::new();
    for step in 0..NOTE_TOKENS {
        let t = i32::try_from(rows.get(argmax(&l)).copied().unwrap_or(0)).unwrap_or(0);
        if t == dl::END_OF_TURN || t == dl::END_OF_TEXT {
            break;
        }
        out.push(t);
        l = values(m.forward_at(dev, &[t], ids.len() + step, std::slice::from_ref(&head), &opts)?.get(&head).ok_or_else(|| bad("no logits"))?)?.to_vec();
    }
    Ok(out)
}

/// The note check over pairs: how many answers find their own prompt's passage above its nearest
/// neighbours among the pairs' prompts (`CHECK_NEIGHBOURS`), read by the embedder.
fn note_check(home: &Path, prompts: &[String], answers: &[Vec<String>]) -> Result<Vec<usize>, Fault> {
    let e = nmapi::neural::embedder_model(home)?;
    let dev = device(&e);
    let docs: Vec<Vec<f32>> = prompts.iter().map(|p| nmengn::roles::card_embed_on(&e, dev, p)).collect::<Result<_, _>>()?;
    let k = nmapi::writer_idle::CHECK_NEIGHBOURS;
    let mut landed = vec![0; answers.len()];
    for (arm, outs) in answers.iter().enumerate() {
        for (i, a) in outs.iter().enumerate() {
            if a.trim().is_empty() {
                continue;
            }
            let q = nmengn::roles::card_embed_on(&e, dev, &format!("Instruct: {}\nQuery:{a}", nmsubs::dense::task_for(a)))?;
            let own = &docs[i];
            let mut near: Vec<(f32, usize)> = docs.iter().enumerate().filter(|(j, _)| *j != i).map(|(j, v)| (nmapi::adapter::dot(own, v), j)).collect();
            near.sort_by(|a, b| b.0.total_cmp(&a.0));
            near.truncate(k);
            let mine = nmapi::adapter::dot(&q, own);
            if near.iter().all(|(_, j)| nmapi::adapter::dot(&q, &docs[*j]) < mine) {
                landed[arm] += 1;
            }
        }
    }
    Ok(landed)
}

/// The chosen fit on the tune split: (ridge, rank, scale), its map at the widest rank, and the
/// tune figures (score, drift).
struct Choice {
    c: f64,
    r: usize,
    scale: f64,
    map: FitMap,
    tune: Score,
    kl: f64,
}

/// The neutral tune states and the base's log-probabilities on them.
struct Neutral<'a> {
    hs: &'a [f32],
    n: usize,
    base: &'a [f32],
}

#[expect(clippy::too_many_arguments, reason = "the grid reads the head, the fit's sums, the tune split and its base, the socket, the neutral text and the log")]
fn choose(head: &Head, folds: &Folds, tune: &[&Pos], base_tune: Score, node: &str, neutral: &Neutral, log: &mut String) -> Result<Choice, Fault> {
    let (nt, nn, nbase) = (neutral.hs, neutral.n, neutral.base);
    let d = head.m.pop.dims.hidden;
    let rmax = *RANKS.iter().max().unwrap_or(&32);
    let ranks: &[usize] = if matches!(folds, Folds::Chan(_)) { &[0] } else { &RANKS };
    let mut best: Option<(bool, f64, Choice)> = None;
    for c in RIDGES {
        let Some(map) = solve(folds, None, c, rmax) else {
            let _ = writeln!(log, "  ridge {c}: not solvable");
            continue;
        };
        for &r in ranks {
            for scale in SCALES {
                let u = make("tune", node, d, &map, r, scale)?;
                let (s, _, _) = score(head, tune, Some(&u))?;
                let kl = drift(head, nt, nn, nbase, &u)?;
                let ok = s.top1 >= base_tune.top1 && kl <= GATE_KL;
                let gain = base_tune.nll - s.nll;
                let _ = writeln!(log, "  ridge {c} rank {r} scale {scale}: tune loss {:.4} (base {:.4}), top-1 {:.1}% (base {:.1}%), neutral drift {kl:.4}{}", s.nll, base_tune.nll, s.top1 * 100.0, base_tune.top1 * 100.0, if ok { "" } else { " (outside the tune rule)" });
                if best.as_ref().is_none_or(|b| (ok && !b.0) || (ok == b.0 && gain > b.1)) {
                    best = Some((ok, gain, Choice { c, r, scale, map: map.clone(), tune: s, kl }));
                }
            }
        }
    }
    best.map(|b| b.2).ok_or_else(|| bad("no ridge strength gave a solvable fit"))
}

fn pct(x: f64) -> String {
    format!("{:.1}%", x * 100.0)
}

/// Writes the references' `fitted` lists (`memory/training/references.json`) for a cartridge.
fn mark_fitted(home: &Path, refs: &[String], cartridge: &str) {
    let f = ex::dir(&home.join("memory")).join(ex::REFERENCES);
    let Some(mut v) = std::fs::read(&f).ok().and_then(|b| nmjson::parse(&b).ok()) else { return };
    if let Value::Object(top) = &mut v
        && let Some(Value::Array(list)) = top.get_mut("references")
    {
        for r in list.iter_mut() {
            let id = r.get("id").and_then(Value::string).map(str::to_owned).unwrap_or_default();
            if refs.contains(&id)
                && let Value::Object(m) = r
                && let Value::Array(a) = m.entry("fitted".to_owned()).or_insert_with(|| Value::Array(Vec::new()))
                && !a.iter().any(|x| x.string().ok() == Some(cartridge))
            {
                a.push(Value::String(cartridge.to_owned()));
            }
        }
    }
    let _ = std::fs::write(&f, nmjson::pretty::pretty(&v));
    let _ = nmapi::audit::audit(home, "training", vec![("label", Value::String(format!("Cartridge {cartridge} fitted from reference {}", refs.join(", ")))), ("cartridge", Value::String(cartridge.to_owned()))]);
}

/// Writes a distilled job filter in the job filters' own form (`nmengn-sockets/1`,
/// `library/parameter-maps/<role>/job-filter-distil-<set>.json`), its lineage recorded beside it
/// with its outcome, so it is listed, plugged and checked as every job filter is. A new one starts
/// unplugged (`enabled: false`); a rebuild replaces the file as a job filter rebuild does, keeping
/// its plug. Returns the file and the cartridge key.
#[expect(clippy::too_many_arguments, reason = "a filter file is its set, model, base, gains, socket, figures and lineage")]
fn write_filter(home: &Path, set: &str, model: &str, base: &str, u: &Fitted, short: &str, distil: Value, lineage: &nmapi::lineage::Record) -> Result<(PathBuf, String), Fault> {
    let Fitted::Channel { gain, bias } = u else { return Err(bad("not a per-channel map")) };
    let role = nmlearn::library::role_of(model);
    let dir = home.join("library/parameter-maps").join(role);
    let job = dl::filter_job(set);
    let name = format!("job-filter-{job}.json");
    let file = dir.join(&name);
    let old = std::fs::read(&file).ok().and_then(|b| nmjson::parse(&b).ok());
    let version = old.as_ref().and_then(|v| v.get("cartridge").and_then(Value::string).ok().and_then(|c| c.rsplit_once('@').and_then(|(_, n)| n.parse::<u64>().ok()))).map_or(1, |n| n + 1);
    let enabled = old.as_ref().and_then(|v| match v.get("enabled") { Ok(Value::Bool(b)) => Some(*b), _ => None }).unwrap_or(false);
    let key = format!("{role}-{job}@{version}");
    let nums = |xs: &[f32]| Value::Array(xs.iter().map(|x| Value::Number(format!("{x:.6}"))).collect());
    let v = obj(vec![
        ("contract", sv("nmengn-sockets/1")),
        ("cartridge", sv(&key)),
        ("base", sv(base.strip_prefix("open:").map_or(base, |_| "open weights"))),
        ("model", sv(model)),
        ("job", sv(&job)),
        ("enabled", Value::Bool(enabled)),
        ("parts", Value::Array(vec![obj(vec![("target", sv(short)), ("gain", nums(gain)), ("bias", nums(bias))])])),
        ("distil", distil),
    ]);
    std::fs::create_dir_all(&dir).map_err(|e| bad(format!("{}: {e}", dir.display())))?;
    let tmp = file.with_extension("json.tmp");
    std::fs::write(&tmp, nmjson::pretty::pretty(&v)).map_err(|e| bad(format!("{}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, &file).map_err(|e| bad(format!("{}: {e}", file.display())))?;
    let rec = lineage.clone();
    nmapi::lineage::change(&dir, |all| {
        all.insert(name.clone(), rec);
    })
    .map_err(bad)?;
    Ok((file, key))
}

fn sv(x: &str) -> Value {
    Value::String(x.to_owned())
}
fn nv(x: f64) -> Value {
    Value::Number(if x.is_finite() { format!("{x:.6}") } else { "0".to_owned() })
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

// ---------------------------------------------------------------- the judge: teacher scores

/// Neutral drift bound of the judge: mean relative entropy (nats) of its verdict on a neutral pair
/// from the base's verdict there.
const GATE_KL_JUDGE: f64 = 0.05;
/// The largest difference in log-odds allowed between the judge read with the unit seated and the
/// scores the fit measured from its states (the same arithmetic on both sides, up to rounding).
const SERVED_TOLERANCE: f64 = 0.05;
/// Tokens of a neutral chunk taken as the query of the judge's neutral pairs.
const NEUTRAL_QUERY: usize = 16;

/// Running sums of a judge read-out's ridge over one third of the fit pairs: `sum w h h'` (`d x d`,
/// row-major) and `sum w h e`, `e` the teacher's log-odds minus the judge's.
#[derive(Clone)]
struct ReadSums {
    g: Vec<f64>,
    b: Vec<f64>,
    n: usize,
}

impl ReadSums {
    fn new(d: usize) -> Self {
        Self { g: vec![0.0; d * d], b: vec![0.0; d], n: 0 }
    }
    fn add(&mut self, h: &[f32], e: f64, w: f64) {
        let d = self.b.len();
        if h.len() != d {
            return;
        }
        for (i, xi) in h.iter().enumerate() {
            let xi = w * f64::from(*xi);
            if xi == 0.0 {
                continue;
            }
            self.b[i] += xi * e;
            for (gv, xj) in self.g[i * d..(i + 1) * d].iter_mut().zip(h) {
                *gv += xi * f64::from(*xj);
            }
        }
        self.n += 1;
    }
    fn plus(&mut self, o: &Self) {
        self.g.iter_mut().zip(&o.g).for_each(|(a, b)| *a += b);
        self.b.iter_mut().zip(&o.b).for_each(|(a, b)| *a += b);
        self.n += o.n;
    }
}

/// The read-out `w = (G + c (tr G / d) I)^-1 b` over every third but `leave_out`. None when the
/// system does not solve.
fn solve_readout(folds: &[ReadSums], leave_out: Option<usize>, c: f64) -> Option<Vec<f64>> {
    let d = folds.first().map_or(0, |f| f.b.len());
    let mut t = ReadSums::new(d);
    for (_, f) in folds.iter().enumerate().filter(|(i, _)| Some(*i) != leave_out) {
        t.plus(f);
    }
    if d == 0 || t.n == 0 {
        return None;
    }
    let energy = (0..d).map(|i| t.g[i * d + i]).sum::<f64>() / d as f64;
    let mut a = t.g;
    for i in 0..d {
        a[i * d + i] += c * energy.max(1e-12);
    }
    if !nmapi::adapter_fit::cholesky(&mut a, d) {
        return None;
    }
    let mut w = t.b;
    nmapi::adapter_fit::chol_solve(&a, d, &mut w, 1);
    w.iter().all(|x| x.is_finite()).then_some(w)
}

/// The judge's read-out sums by third of the fit pairs; `shift` takes each pair's residual from
/// another pair (the shuffled control).
fn read_folds(d: usize, ps: &[&Pos], resid: &[f64], weights: &BTreeMap<usize, f64>, shift: bool) -> Vec<ReadSums> {
    let n = resid.len();
    let rot = n / 3 + 1;
    let mut folds = vec![ReadSums::new(d); TRIALS];
    for (i, p) in ps.iter().enumerate() {
        let j = if shift { (i + rot) % n.max(1) } else { i };
        if let Some(f) = folds.get_mut(p.row % TRIALS) {
            f.add(&p.h, resid.get(j).copied().unwrap_or(0.0), weights.get(&p.row).copied().unwrap_or(1.0));
        }
    }
    folds
}

/// The judge's fitted map before its scale is chosen: a map the adapter or filter fit gives, or a
/// read-out vector.
#[derive(Clone)]
enum JudgeMap {
    Map(FitMap),
    Read(Vec<f64>),
}

/// The judge's fit sums: those of a map (`Folds`), or of a read-out.
enum JudgeFolds {
    Map(Folds),
    Read(Vec<ReadSums>),
}

impl JudgeFolds {
    fn solve(&self, leave_out: Option<usize>, c: f64) -> Option<JudgeMap> {
        match self {
            Self::Map(f) => solve(f, leave_out, c, *RANKS.iter().max().unwrap_or(&32)).map(JudgeMap::Map),
            Self::Read(f) => solve_readout(f, leave_out, c).map(JudgeMap::Read),
        }
    }
}

/// The judge's head: its yes and no rows at the final norm, read from the engine itself (unit
/// vectors seeded there), and the read-out direction they give.
struct Judge<'a> {
    m: &'a Model,
    fin: String,
    head: String,
    yes: Vec<f32>,
    no: Vec<f32>,
    toward: Vec<f32>,
}

impl<'a> Judge<'a> {
    fn new(m: &'a Model) -> Result<Self, Fault> {
        let mut j = Self { m, fin: m.role_node("final_norm")?, head: m.role_node("head")?, yes: Vec::new(), no: Vec::new(), toward: Vec::new() };
        let d = m.pop.dims.hidden;
        let (mut yes, mut no) = (vec![0.0_f32; d], vec![0.0_f32; d]);
        let mut i = 0;
        while i < d {
            let c = HEAD_CHUNK.min(d - i);
            let mut basis = vec![0.0_f32; c * d];
            for k in 0..c {
                basis[k * d + i + k] = 1.0;
            }
            for (k, (y, n)) in j.logits(&basis, c)?.into_iter().enumerate() {
                #[expect(clippy::cast_possible_truncation, reason = "head rows kept at 32 bits")]
                let (yf, nf) = (y as f32, n as f32);
                yes[i + k] = yf;
                no[i + k] = nf;
            }
            i += c;
        }
        j.toward = dl::toward(&yes, &no).ok_or_else(|| bad("the judge's yes and no rows are the same"))?;
        j.yes = yes;
        j.no = no;
        Ok(j)
    }
    /// The judge's yes and no logits for `rows` final-norm states.
    fn logits(&self, hs: &[f32], rows: usize) -> Result<Vec<(f64, f64)>, Fault> {
        let d = self.m.pop.dims.hidden;
        let target = (self.head.clone(), "logits".to_owned());
        let opts = RunOpts { all_positions: true, ..RunOpts::default() };
        let mut out = Vec::with_capacity(rows);
        let mut i = 0;
        while i < rows {
            let c = HEAD_CHUNK.min(rows - i);
            let seed = hs.get(i * d..(i + c) * d).ok_or_else(|| bad("states out of range"))?.to_vec();
            let r = self.m.run_from(device(self.m), vec![(self.fin.clone(), "y".to_owned(), Act::Cpu(seed))], std::slice::from_ref(&target), c, &opts)?;
            let v = values(r.get(&target).ok_or_else(|| bad("no logits"))?)?;
            if v.len() != 2 * c {
                return Err(bad(format!("the judge's head gave {} values for {c} states; it reads its yes and no rows only", v.len())));
            }
            out.extend(v.as_chunks::<2>().0.iter().map(|p| (f64::from(p[0]), f64::from(p[1]))));
            i += c;
        }
        Ok(out)
    }
    /// The log-odds (yes minus no) of states under a map.
    fn z(&self, ps: &[&Pos], u: Option<&Fitted>) -> Result<Vec<f64>, Fault> {
        Ok(self.logits(&mapped(ps, u), ps.len())?.into_iter().map(|(y, n)| y - n).collect())
    }
}

/// The judge's state at the last token and its log-odds for one (query, document) pair, read as
/// the serving reranker reads it (its card prompt with the code instruction, the document cut to
/// the input cap).
fn judge_read(m: &Model, query: &str, doc: &str) -> Result<(Vec<f32>, f64), Fault> {
    let dev = device(m);
    let tok = &m.pop.tokenizer;
    let fixed = tok.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, query, "")).map_err(|e| bad(e.to_string()))?.len();
    let mut d_ids = tok.encode(doc).map_err(|e| bad(e.to_string()))?;
    d_ids.truncate(nmengn::roles::input_cap(m, dev, fixed));
    let ids = tok.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, query, &tok.decode(&d_ids))).map_err(|e| bad(e.to_string()))?;
    let head = (m.role_node("head")?, "logits".to_owned());
    let fin = (m.role_node("final_norm")?, "y".to_owned());
    let r = m.forward(dev, &ids, &[head.clone(), fin.clone()], &RunOpts { last_rows_only: true, ..RunOpts::default() })?;
    let v = values(r.get(&head).ok_or_else(|| bad("no label logits"))?)?;
    let z = match (v.first(), v.get(1)) {
        (Some(a), Some(b)) => f64::from(*a) - f64::from(*b),
        _ => return Err(bad("empty label logits")),
    };
    let y = values(r.get(&fin).ok_or_else(|| bad("no final-norm state"))?)?;
    let d = m.pop.dims.hidden;
    let h = y.get(y.len().saturating_sub(d)..).filter(|h| h.len() == d).ok_or_else(|| bad("no last state"))?.to_vec();
    Ok((h, z))
}

/// Measures of the judge on a set of pairs: mean relative entropy from the teacher's scores, the
/// share of pairs where the judge's verdict (log-odds above zero) is the teacher's (score above one
/// half; pairs scored exactly one half are not counted), and the pairs.
#[derive(Clone, Copy, Default, Debug)]
struct JScore {
    kl: f64,
    agree: f64,
    n: usize,
}

fn jscore(z: &[f64], p: &[f64]) -> (JScore, Vec<f64>) {
    let per: Vec<f64> = z.iter().zip(p).map(|(z, p)| dl::bernoulli_kl(*p, *z)).collect();
    let decided: Vec<bool> = z.iter().zip(p).filter(|(_, p)| (**p - 0.5).abs() > 1e-9).map(|(z, p)| (*z > 0.0) == (*p > 0.5)).collect();
    let n = z.len().max(1) as f64;
    let agree = if decided.is_empty() { 0.0 } else { decided.iter().filter(|x| **x).count() as f64 / decided.len() as f64 };
    (JScore { kl: per.iter().sum::<f64>() / n, agree, n: z.len() }, per)
}

/// The share of same-query pairs with different teacher scores that the judge orders as the
/// teacher does (reported, not gated).
fn order_agreement(z: &[f64], p: &[f64], query: &[usize]) -> f64 {
    let (mut hit, mut all) = (0_usize, 0_usize);
    for i in 0..z.len() {
        for j in i + 1..z.len() {
            if query.get(i) == query.get(j) && (p[i] - p[j]).abs() > 1e-9 {
                all += 1;
                hit += usize::from((z[i] - z[j]) * (p[i] - p[j]) > 0.0);
            }
        }
    }
    if all == 0 { 0.0 } else { hit as f64 / all as f64 }
}

/// Mean relative entropy of the judge's mapped verdicts on neutral pairs from its base verdicts.
fn judge_drift(j: &Judge, neutral: &[&Pos], base: &[f64], u: &Fitted) -> Result<f64, Fault> {
    let z = j.z(neutral, Some(u))?;
    Ok(z.iter().zip(base).map(|(z, b)| dl::bernoulli_kl(dl::sigmoid(*b), *z)).sum::<f64>() / z.len().max(1) as f64)
}

/// The judge's map at scale `scale`, in the form its cartridge kind serves it.
fn judge_make(key: &str, node: &str, j: &Judge, map: &JudgeMap, r: usize, scale: f64) -> Result<Fitted, Fault> {
    let d = j.m.pop.dims.hidden;
    match map {
        JudgeMap::Map(m) => make(key, node, d, m, r, scale),
        JudgeMap::Read(w) => {
            #[expect(clippy::cast_possible_truncation, reason = "a read-out kept at 16 bits, its scale at 32")]
            let wf: Vec<f32> = w.iter().map(|x| *x as f32).collect();
            #[expect(clippy::cast_possible_truncation, reason = "a scale kept at 32 bits")]
            dl::readout_unit(key, node, &wf, &j.toward, scale as f32).map(Fitted::Lens)
        }
    }
}

/// One choice of the judge's fit on the tune split.
struct JChoice {
    c: f64,
    r: usize,
    scale: f64,
    map: JudgeMap,
    tune: JScore,
    kl: f64,
}

/// Neutral (query, document) pairs from the Nano's own source: each chunk's first tokens as the
/// query, with its own chunk and with the next chunk as documents.
fn neutral_pairs(tok: &nmengn::tokenizer::Tokenizer, chunks: &[Vec<i32>]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (k, c) in chunks.iter().enumerate() {
        let q = tok.decode(c.get(..NEUTRAL_QUERY.min(c.len())).unwrap_or(&[]));
        out.push((q.clone(), tok.decode(c)));
        if let Some(next) = chunks.get((k + 1) % chunks.len().max(1)).filter(|_| chunks.len() > 1) {
            out.push((q, tok.decode(next)));
        }
    }
    out
}

/// Distillation into the Reranker from teacher scores of (query, document) pairs: an adapter (a
/// low-rank map at its final norm toward the per-pair shifts that move its log-odds to the
/// teacher's), a filter (per-channel gain and bias toward the same shifts) or a judge read-out (a
/// ridge read-out of the residual log-odds, served as a rank-one map along the yes-minus-no
/// direction). `teacher-loop` reweighs the fit pairs it follows worst and refits. The gate, on the
/// held-out split: relative entropy from the teacher's scores falls, verdict agreement does not
/// fall, a shuffled control gains at most `GATE_CONTROL` of the gain, neutral drift within
/// `GATE_KL_JUDGE`, and three refits without a third of the fit pairs each gain.
#[expect(clippy::too_many_lines, reason = "one build: data, features, fit, gate and the cartridge, in order")]
fn build_judge(home: &Path, args: &[String], role: &str, target: &str, technique: &str, set: &str) -> Result<String, Fault> {
    let t0 = Instant::now();
    let rounds = if technique == "teacher-loop" { arg(args, "--rounds").and_then(|v| v.parse().ok()).unwrap_or(ROUNDS).clamp(1, 8) } else { 1 };
    let what = if target == "readout" { format!("readout-{}", dl::slug(set)) } else { dl::slug(set) };
    let report = report_file(home, &format!("{role}-{what}"));
    let mut out = String::new();
    let _ = writeln!(out, "distillation into the {role}: a {target} cartridge, set {set}, technique {technique}{}", if rounds > 1 { format!(", at most {rounds} rounds") } else { String::new() });
    let rows: Vec<DistilRow> = rows_of(home, set)?.into_iter().filter(|r| r.score.is_some()).collect();
    let count = |s: Split| rows.iter().filter(|r| r.split == s).count();
    let (nf, nt, nh) = (count(Split::Fit), count(Split::Tune), count(Split::Held));
    let _ = writeln!(out, "scored pairs (healthy only): fit {nf}, tune {nt}, held out {nh}");
    if nf < MIN_FIT || nt < MIN_TUNE || nh < MIN_HELD {
        return Err(bad(format!("too few scored pairs: fit {nf} (at least {MIN_FIT}), tune {nt} (at least {MIN_TUNE}), held out {nh} (at least {MIN_HELD}); import teacher pairs with a score from 0 to 1 (prompt the query, completion the document)")));
    }
    let offered = seat_base(home, args, role);
    let m = nmapi::neural::reranker_model(home)?;
    let node = m.role_node(dl::SOCKET)?;
    // Features are the base's own: whatever distilled cartridge is plugged is taken out here.
    m.set_unit(&node, None)?;
    let d = m.pop.dims.hidden;
    let dev = device(&m);
    let values_key = nmapi::neural::parameters_key(&m);
    let base = made_on(&m);
    let _ = writeln!(out, "Reranker {} on {base} (offered {offered}), device {}, values {values_key}", m.pop.card.id, dev.name());
    let t = Instant::now();
    let j = Judge::new(&m)?;
    // Features: each pair read as the serving reranker reads it.
    let (mut fit, mut tune, mut held): (Vec<Pos>, Vec<Pos>, Vec<Pos>) = (Vec::new(), Vec::new(), Vec::new());
    let (mut pf, mut pt, mut ph): (Vec<f64>, Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new(), Vec::new());
    let mut queries: BTreeMap<String, usize> = BTreeMap::new();
    let mut qh: Vec<usize> = Vec::new();
    let mut read_z: Vec<f64> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let (h, z) = judge_read(&m, &r.prompt, &r.completion)?;
        let p = r.score.unwrap_or(0.5);
        let pos = Pos { h, t: 0, tk: Vec::new(), row: i };
        match r.split {
            Split::Fit => {
                fit.push(pos);
                pf.push(p);
            }
            Split::Tune => {
                tune.push(pos);
                pt.push(p);
            }
            Split::Held => {
                read_z.push(z);
                held.push(pos);
                ph.push(p);
                let nq = queries.len();
                qh.push(*queries.entry(r.prompt.clone()).or_insert(nq));
            }
        }
    }
    let (fit_r, tune_r, held_r): (Vec<&Pos>, Vec<&Pos>, Vec<&Pos>) = (fit.iter().collect(), tune.iter().collect(), held.iter().collect());
    // The states read back through the head give the judge's own log-odds.
    let back = j.z(&held_r, None)?;
    let readback = back.iter().zip(&read_z).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let _ = writeln!(out, "features of {} pairs in {:.1} s; held-out log-odds read back from the states within {readback:.4}", rows.len(), t.elapsed().as_secs_f64());
    // Neutral pairs.
    let tok = &m.pop.tokenizer;
    let (nt_ids, nh_ids) = neutral_ids(home, tok);
    let read_neutral = |pairs: &[(String, String)]| -> Result<Vec<Pos>, Fault> { pairs.iter().enumerate().map(|(i, (q, doc))| judge_read(&m, q, doc).map(|(h, _)| Pos { h, t: 0, tk: Vec::new(), row: i })).collect() };
    let n_tune = read_neutral(&neutral_pairs(tok, &nt_ids))?;
    let n_held = read_neutral(&neutral_pairs(tok, &nh_ids))?;
    let (n_tune_r, n_held_r): (Vec<&Pos>, Vec<&Pos>) = (n_tune.iter().collect(), n_held.iter().collect());
    let (nb_tune, nb_held) = (j.z(&n_tune_r, None)?, j.z(&n_held_r, None)?);
    let _ = writeln!(out, "neutral pairs (the Nano's own source as plain text): tune {}, held out {}", n_tune.len(), n_held.len());
    // Targets and the base on every split.
    let fit_lg = j.logits(&mapped(&fit_r, None), fit_r.len())?;
    let resid: Vec<f64> = fit_lg.iter().zip(&pf).map(|((y, n), p)| dl::score_logit(*p) - (y - n)).collect();
    let rows2: [&[f32]; 2] = [&j.yes, &j.no];
    #[expect(clippy::cast_possible_truncation, reason = "shifts kept at 32 bits")]
    let deltas: Vec<Vec<f32>> = fit_lg
        .iter()
        .zip(&pf)
        .map(|((y, n), p)| {
            let goal = dl::judge_target(*y, *n, dl::score_logit(*p));
            dl::target_delta(&rows2, &[*y, *n], &goal, dl::MU).map_or_else(|| vec![0.0; d], |v| v.into_iter().map(|x| x as f32).collect())
        })
        .collect();
    let (base_tune, _) = jscore(&j.z(&tune_r, None)?, &pt);
    let z_held = j.z(&held_r, None)?;
    let (base_held, _) = jscore(&z_held, &ph);
    let fold = |weights: &BTreeMap<usize, f64>, shift: bool| -> JudgeFolds {
        if target == "readout" { JudgeFolds::Read(read_folds(d, &fit_r, &resid, weights, shift)) } else { JudgeFolds::Map(fold_sums(target, d, &fit_r, &deltas, weights, shift)) }
    };
    let ranks: Vec<usize> = if target == "adapter" { RANKS.to_vec() } else { vec![1] };
    // Rounds: one for `lens`, `channel` and `readout`; `teacher-loop` reweighs the worst fit pairs.
    let mut weights: BTreeMap<usize, f64> = BTreeMap::new();
    let mut log = String::new();
    let mut chosen: Option<(JChoice, JudgeFolds, BTreeMap<usize, f64>)> = None;
    for round in 0..rounds {
        let folds = fold(&weights, false);
        let mut best: Option<(bool, f64, JChoice)> = None;
        let grid: Vec<(f64, usize, f64)> = match &chosen {
            None => RIDGES.iter().flat_map(|c| ranks.iter().flat_map(move |r| SCALES.iter().map(move |s| (*c, *r, *s)))).collect(),
            Some((p, _, _)) => vec![(p.c, p.r, p.scale)],
        };
        let mut maps: BTreeMap<u64, Option<JudgeMap>> = BTreeMap::new();
        for (c, r, scale) in grid {
            let Some(map) = maps.entry(c.to_bits()).or_insert_with(|| folds.solve(None, c)).clone() else {
                let _ = writeln!(log, "  ridge {c}: not solvable");
                continue;
            };
            let u = judge_make("tune", &node, &j, &map, r, scale)?;
            let (s, _) = jscore(&j.z(&tune_r, Some(&u))?, &pt);
            let kl = judge_drift(&j, &n_tune_r, &nb_tune, &u)?;
            let ok = s.agree >= base_tune.agree && kl <= GATE_KL_JUDGE;
            let gain = base_tune.kl - s.kl;
            let _ = writeln!(log, "  ridge {c} rank {r} scale {scale}: tune relative entropy {:.4} (base {:.4}), agreement {} (base {}), neutral drift {kl:.4}{}", s.kl, base_tune.kl, pct(s.agree), pct(base_tune.agree), if ok { "" } else { " (outside the tune rule)" });
            if best.as_ref().is_none_or(|b| (ok && !b.0) || (ok == b.0 && gain > b.1)) {
                best = Some((ok, gain, JChoice { c, r, scale, map, tune: s, kl }));
            }
        }
        let Some((_, _, c)) = best else { return Err(bad("no ridge strength gave a solvable fit")) };
        let better = chosen.as_ref().is_none_or(|(p, _, _)| c.tune.kl < p.tune.kl && c.kl <= GATE_KL_JUDGE);
        if !better {
            let _ = writeln!(log, "round {}: the tune split stopped gaining; the previous round stands", round + 1);
            break;
        }
        let _ = writeln!(log, "round {}: chosen ridge {} rank {} scale {}: tune relative entropy {:.4} (base {:.4}), agreement {} (base {}), neutral drift {:.4}", round + 1, c.c, c.r, c.scale, c.tune.kl, base_tune.kl, pct(c.tune.agree), pct(base_tune.agree), c.kl);
        if round + 1 < rounds {
            let u = judge_make("fit", &node, &j, &c.map, c.r, c.scale)?;
            let (_, per) = jscore(&j.z(&fit_r, Some(&u))?, &pf);
            let mut worst: Vec<(f64, usize)> = per.iter().zip(&fit_r).map(|(l, p)| (*l, p.row)).collect();
            worst.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut next = weights.clone();
            for (_, r) in worst.iter().take(worst.len().div_ceil(3)) {
                let w = next.entry(*r).or_insert(1.0);
                *w = (*w * 1.5).min(4.0);
            }
            chosen = Some((c, folds, weights));
            weights = next;
        } else {
            chosen = Some((c, folds, weights.clone()));
        }
    }
    let (c, folds, weights) = chosen.ok_or_else(|| bad("no round produced a fit"))?;
    let _ = write!(out, "{log}");
    // The held-out gate.
    let id = if target == "readout" { dl::cartridge_id(role, &format!("readout-{set}")) } else { dl::cartridge_id(role, set) };
    let u = judge_make(&format!("{id}@next"), &node, &j, &c.map, c.r, c.scale)?;
    let z_fit = j.z(&held_r, Some(&u))?;
    let (held_s, _) = jscore(&z_fit, &ph);
    let ctrl_map = fold(&weights, true).solve(None, c.c).ok_or_else(|| bad("the control did not solve"))?;
    let (ctrl_s, _) = jscore(&j.z(&held_r, Some(&judge_make("control", &node, &j, &ctrl_map, c.r, c.scale)?))?, &ph);
    let kl_held = judge_drift(&j, &n_held_r, &nb_held, &u)?;
    let mut trial_gains = Vec::new();
    for f in 0..TRIALS {
        let map = folds.solve(Some(f), c.c).ok_or_else(|| bad("a trial did not solve"))?;
        let tu = judge_make("trial", &node, &j, &map, c.r, c.scale)?;
        trial_gains.push(base_held.kl - jscore(&j.z(&held_r, Some(&tu))?, &ph).0.kl);
    }
    let gain = base_held.kl - held_s.kl;
    let ctrl_gain = base_held.kl - ctrl_s.kl;
    // The served check: the unit seated on the judge itself, the held-out pairs read through the
    // serving forward give the scores the fit measured (a filter composes with the job filters in
    // the serving process, so it is checked there, not here).
    let served = match &u {
        Fitted::Lens(x) => {
            m.set_unit(&node, Some(Arc::new(x.clone())))?;
            let got: Result<Vec<f64>, Fault> = held_r.iter().map(|p| rows.get(p.row).map_or(Ok(0.0), |r| judge_read(&m, &r.prompt, &r.completion).map(|(_, z)| z))).collect();
            m.set_unit(&node, None)?;
            Some(got?.iter().zip(&z_fit).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max))
        }
        Fitted::Channel { .. } => None,
    };
    let mut checks = vec![
        ("relative entropy from the teacher's scores falls", gain > 0.0, format!("{:.4} to {:.4} ({:+.2}%)", base_held.kl, held_s.kl, -100.0 * gain / base_held.kl.max(1e-12))),
        ("verdict agreement does not fall", held_s.agree >= base_held.agree, format!("{} to {}", pct(base_held.agree), pct(held_s.agree))),
        ("shuffled control gains nothing", ctrl_gain <= GATE_CONTROL * gain.max(0.0), format!("control relative entropy {:.4}, gain {ctrl_gain:+.4} against {gain:+.4} (at most {:.0}% of it)", ctrl_s.kl, GATE_CONTROL * 100.0)),
        ("neutral drift within the bound", kl_held <= GATE_KL_JUDGE, format!("{kl_held:.4} nats per pair (bound {GATE_KL_JUDGE})")),
        ("stable over repeated trials", trial_gains.iter().all(|g| *g > 0.0), format!("held-out gains {}", trial_gains.iter().map(|g| format!("{g:+.4}")).collect::<Vec<_>>().join(", "))),
    ];
    if let Some(dz) = served {
        checks.push(("the served map gives the fitted scores", dz <= SERVED_TOLERANCE, format!("largest difference {dz:.4} in log-odds over the held-out pairs (at most {SERVED_TOLERANCE})")));
    }
    let passed = checks.iter().all(|c| c.1);
    let _ = writeln!(out, "held-out gate ({} pairs):", held_s.n);
    for (name, ok, fig) in &checks {
        let _ = writeln!(out, "  {} {name}: {fig}", if *ok { "PASS" } else { "FAIL" });
    }
    let _ = writeln!(out, "  order agreement within a query (reported): base {}, distilled {}", pct(order_agreement(&z_held, &ph, &qh)), pct(order_agreement(&z_fit, &ph, &qh)));
    let measured = format!("held out: relative entropy {:.4} to {:.4}, agreement {} to {}, control gain {ctrl_gain:+.4}, neutral drift {kl_held:.4}", base_held.kl, held_s.kl, pct(base_held.agree), pct(held_s.agree));
    let card_sha = m.pop.card.sha_main.clone();
    let model_id = m.pop.card.id.clone();
    let short = node.rsplit('#').next().unwrap_or(&node).to_owned();
    let mut written = None;
    if passed {
        let teachers: Vec<String> = rows.iter().map(|r| r.teacher.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let licences: Vec<String> = rows.iter().map(|r| r.licence.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let refs: Vec<String> = rows.iter().map(|r| r.reference.clone()).filter(|r| !r.is_empty()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let gate = obj(checks.iter().map(|(n, ok, fig)| (*n, obj(vec![("pass", Value::Bool(*ok)), ("figures", sv(fig))]))).collect());
        let lineage = nmapi::lineage::Record {
            made_on: base.clone(),
            made_by: format!("neuromorph-nano lab run distil build {role} --target {target} --set {set} --technique {technique}"),
            made_ms: nmtype::now_ms(),
            purpose: if target == "filter" { "job-filter".into() } else { "distil".into() },
            used_by: Vec::new(),
            recorded: "tool".into(),
            note: format!("fitted from {nf} teacher scores ({})", teachers.join(", ")),
            outcome: "used".into(),
            outcome_why: format!("passed its held-out gate: {measured}"),
        };
        let strs = |v: &[String]| Value::Array(v.iter().map(|x| sv(x)).collect());
        let common = |extra: Vec<(&'static str, Value)>| -> Value {
            let mut f = vec![
                ("values", sv(&values_key)),
                ("base", sv(&base)),
                ("weights_sha256", sv(&card_sha)),
                ("role", sv(role)),
                ("technique", sv(technique)),
                ("set", sv(set)),
                ("ridge", nv(c.c)),
                ("pairs", obj(vec![("fit", Value::Number(nf.to_string())), ("tune", Value::Number(nt.to_string())), ("held", Value::Number(nh.to_string()))])),
                ("teachers", strs(&teachers)),
                ("licences", strs(&licences)),
                ("references", strs(&refs)),
                ("gate", gate.clone()),
                ("measured", sv(&measured)),
                ("acts_on", sv("the Reranker's final norm; its label reads the last token's row")),
                ("effect", sv("the judge's yes or no score moves toward the teacher's scores; nothing else it reads changes")),
                ("project", sv(&home.display().to_string())),
                ("lineage", lineage.to_value()),
            ];
            f.extend(extra);
            obj(f)
        };
        let first = nmapi::adapter::versions(home, &id).is_empty();
        let unplug_new = |id: &str| {
            let _ = nmapi::plugs::change(home, |s| {
                s.units_off.insert(nmapi::plugs::key(nmapi::plugs::NANO, id));
            });
        };
        match &c.map {
            JudgeMap::Read(w) => {
                #[expect(clippy::cast_possible_truncation, reason = "a read-out kept at 16 bits")]
                let wf: Vec<f32> = w.iter().map(|x| *x as f32).collect();
                let notes = common(vec![
                    ("alpha", nv(c.scale)),
                    ("scale", nv(c.scale)),
                    ("bias", nv(0.0)),
                    ("socket", obj(vec![("node", sv(&node)), ("role", sv(dl::SOCKET)), ("layers", sv("after the last layer (final norm)")), ("form", sv("y + alpha (w . y) toward, toward = (yes - no) / |yes - no|^2")), ("rank", Value::Number("1".into())), ("w", sv(&format!("[{d}]"))), ("toward", sv(&format!("[{d}]")))])),
                ]);
                let label = nmapi::adapter_fit::LabelIn { id: &id, version: 0, model: &model_id, weights: &card_sha, kind: nmapi::adapter::READOUT_KIND, does: &dl::does(role, "readout"), made_from: &format!("{nf} teacher scores of set {set} ({}), on {base}", teachers.join(", ")), notes };
                let parts: Vec<nmapi::adapter::Part> = vec![("w".to_owned(), vec![d], wf), ("toward".to_owned(), vec![d], j.toward.clone())];
                let (file, version) = nmapi::adapter_fit::write_next(home, label, &parts)?;
                if first {
                    unplug_new(&id);
                }
                mark_fitted(home, &refs, &format!("{id}@{version}"));
                let _ = writeln!(out, "WRITTEN: {} ({id}@{version}, a judge read-out, scale {}{})", file.display(), c.scale, if first { ", unplugged until you plug it in" } else { "" });
                written = Some(format!("{id}@{version}"));
            }
            JudgeMap::Map(FitMap::Chan(..)) => {
                let distil = common(vec![("scale", nv(c.scale))]);
                let (file, key) = write_filter(home, set, &model_id, &base, &u, &short, distil, &lineage)?;
                mark_fitted(home, &refs, &key);
                let _ = writeln!(out, "WRITTEN: {} ({key}, a job filter at {short}, scale {}; serves where the Reranker's job filters are on and it is plugged in)", file.display(), c.scale);
                written = Some(key);
            }
            JudgeMap::Map(FitMap::Low(a, b)) => {
                let (dn, upm) = dl::leading(a, b, d, c.r);
                let r = dn.len() / d.max(1);
                let notes = common(vec![
                    ("scale", nv(c.scale)),
                    ("socket", obj(vec![("node", sv(&node)), ("role", sv(dl::SOCKET)), ("layers", sv("after the last layer (final norm)")), ("form", sv("y + scale * up(down(y))")), ("rank", Value::Number(r.to_string())), ("down", sv(&format!("[{r}, {d}]"))), ("up", sv(&format!("[{d}, {r}]")))])),
                ]);
                let label = nmapi::adapter_fit::LabelIn { id: &id, version: 0, model: &model_id, weights: &card_sha, kind: dl::KIND, does: &dl::does(role, "lowrank"), made_from: &format!("{nf} teacher scores of set {set} ({}), on {base}", teachers.join(", ")), notes };
                let parts: Vec<nmapi::adapter::Part> = vec![("down".to_owned(), vec![r, d], dn), ("up".to_owned(), vec![d, r], upm)];
                let (file, version) = nmapi::adapter_fit::write_next(home, label, &parts)?;
                if first {
                    unplug_new(&id);
                }
                mark_fitted(home, &refs, &format!("{id}@{version}"));
                let _ = writeln!(out, "WRITTEN: {} ({id}@{version}, rank {r}, scale {}{})", file.display(), c.scale, if first { ", unplugged until you plug it in" } else { "" });
                written = Some(format!("{id}@{version}"));
            }
        }
    } else {
        let failed: Vec<&str> = checks.iter().filter(|c| !c.1).map(|c| c.0).collect();
        let _ = writeln!(out, "REFUSED: the held-out gate did not pass ({}); nothing was written. {measured}", failed.join("; "));
    }
    let _ = nmapi::audit::audit(
        home,
        "distil-build",
        vec![
            ("role", sv(role)),
            ("target", sv(target)),
            ("technique", sv(technique)),
            ("set", sv(set)),
            ("passed", Value::Bool(passed)),
            ("cartridge", written.as_deref().map_or(Value::Null, sv)),
            ("measured", sv(&measured)),
            ("report", sv(&report.display().to_string())),
            ("label", sv(&format!("Distillation into the {role} ({target}, {technique}, set {set}): {}", if passed { "passed" } else { "refused" }))),
        ],
    );
    let _ = writeln!(out, "elapsed: {:.0} s; report {}", t0.elapsed().as_secs_f64(), report.display());
    if let Some(dir) = report.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&report, crate::with_overlay(&out));
    Ok(out)
}


// ---------------------------------------------------------------- steer and unit: maps the engine runs

/// Scales tried on a fitted steering vector (its closed form is a first-order estimate through the
/// final norm, so the engine itself measures each scale).
const STEER_SCALES: [f64; 6] = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0];
/// The rank of a distilled learning unit.
const UNIT_RANK: usize = 8;
/// Scales tried on a fitted learning unit.
const UNIT_SCALES: [f64; 5] = [0.5, 1.0, 2.0, 4.0, 8.0];
/// The forward-only fitter's settings: perturbation, step, seed, positions per pair.
const UNIT_EPS: f64 = 1e-3;
const UNIT_LR: f64 = 1e-3;
const UNIT_SEED: u64 = 7;
const UNIT_BATCH: usize = 64;
/// The wall budget of one forward-only fit (seconds) by default, and its cap; a build runs five
/// fits (the fit, the control and three trials).
const UNIT_BUDGET_S: u64 = 20;
const UNIT_BUDGET_CAP_S: u64 = 120;

/// What `build` hands a steer or unit build: the MicroModel, its head over the serving rows, the
/// pairs and their positions, the target shifts and the base's figures.
struct Fwd<'a> {
    home: &'a Path,
    args: &'a [String],
    role: &'a str,
    technique: &'a str,
    set: &'a str,
    m: &'a Model,
    head: &'a Head<'a>,
    node: &'a str,
    at: &'a BTreeMap<usize, usize>,
    seqs: &'a [(usize, Vec<i32>, usize)],
    rows: &'a [DistilRow],
    fit_r: &'a [&'a Pos],
    deltas: &'a [Vec<f32>],
    base_tune: Score,
    base_held: Score,
    neutral: (&'a [Vec<i32>], &'a [Vec<i32>]),
    neutral_base: (&'a [f32], &'a [f32]),
    base: &'a str,
    values_key: &'a str,
    counts: (usize, usize, usize),
}

/// One scored position read under some options: its pair, the teacher's token (a serving-row
/// index) and the rows of each target there.
type Read = (usize, usize, Vec<Vec<f32>>);

/// Each scored position of the pairs of `split`, read under `opts`, in the order `build` reads them.
fn read_split(cx: &Fwd, split: Split, targets: &[(String, String)], opts: &RunOpts) -> Result<Vec<Read>, Fault> {
    let dev = device(cx.m);
    let mut out = Vec::new();
    for (i, ids, plen) in cx.seqs.iter().filter(|(i, _, _)| cx.rows.get(*i).is_some_and(|r| r.split == split)) {
        let r = cx.m.forward(dev, ids, targets, opts)?;
        let ys: Vec<&[f32]> = targets.iter().map(|t| r.get(t).ok_or_else(|| bad("no state")).and_then(|a| values(a))).collect::<Result<_, _>>()?;
        for k in 0..ids.len() - plen {
            let p = plen - 1 + k;
            let Some(t) = ids.get(p + 1).and_then(|x| usize::try_from(*x).ok()).and_then(|x| cx.at.get(&x).copied()) else { continue };
            let at: Vec<Vec<f32>> = ys.iter().map(|y| { let w = y.len() / ids.len().max(1); y.get(p * w..(p + 1) * w).map(<[f32]>::to_vec).unwrap_or_default() }).collect();
            out.push((*i, t, at));
        }
    }
    Ok(out)
}

/// The neutral chunks' rows of each target under `opts`, concatenated, and the positions.
fn read_neutral(m: &Model, chunks: &[Vec<i32>], targets: &[(String, String)], opts: &RunOpts) -> Result<(Vec<Vec<f32>>, usize), Fault> {
    let dev = device(m);
    let mut out = vec![Vec::new(); targets.len()];
    let mut n = 0;
    for c in chunks {
        let r = m.forward(dev, c, targets, opts)?;
        for (o, t) in out.iter_mut().zip(targets) {
            o.extend_from_slice(values(r.get(t).ok_or_else(|| bad("no state"))?)?);
        }
        n += c.len();
    }
    Ok((out, n))
}

/// Mean KL of the base's distribution to the one at states already mapped.
fn drift_on(head: &Head, hs: &[f32], n: usize, base: &[f32]) -> Result<f64, Fault> {
    let na = head.rows.len();
    let mut kl = 0.0;
    head.each(hs, n, |i, q| {
        let p = base.get(i * na..(i + 1) * na).unwrap_or(&[]);
        let lq = lse(q);
        kl += p.iter().zip(q).map(|(lp, zq)| { let lp = f64::from(*lp); lp.exp() * (lp - (f64::from(*zq) - lq)) }).sum::<f64>();
        Ok(())
    })?;
    Ok(kl / n.max(1) as f64)
}

fn as_pos(reads: Vec<Read>) -> Vec<Pos> {
    reads.into_iter().map(|(row, t, mut v)| Pos { h: if v.is_empty() { Vec::new() } else { v.swap_remove(0) }, t, tk: Vec::new(), row }).collect()
}

/// The figures of one candidate map the engine runs (a steer or a unit) on a split and on neutral
/// text: (score, drift).
fn engine_figures(cx: &Fwd, split: Split, neutral: &[Vec<i32>], neutral_base: &[f32], opts: &RunOpts) -> Result<(Score, f64), Fault> {
    let fin = (cx.node.to_owned(), "y".to_owned());
    let ps = as_pos(read_split(cx, split, std::slice::from_ref(&fin), opts)?);
    let refs: Vec<&Pos> = ps.iter().collect();
    let (s, _, _) = score(cx.head, &refs, None)?;
    let (hs, n) = read_neutral(cx.m, neutral, std::slice::from_ref(&fin), opts)?;
    let kl = drift_on(cx.head, hs.first().map_or(&[][..], Vec::as_slice), n, neutral_base)?;
    Ok((s, kl))
}

fn steer_opts(layer: usize, v: &[f32], scale: f64) -> RunOpts {
    #[expect(clippy::cast_possible_truncation, reason = "a scale kept at 32 bits")]
    let s = scale as f32;
    RunOpts { steer: BTreeMap::from([(layer, (Arc::new(v.to_vec()), s))]), ..RunOpts::default() }
}

/// The steering vector `v_j = sum w a_ij delta_ij / sum w a_ij^2` over every third but `leave_out`,
/// `a_ij = g_j / r_i` the final norm's first-order gain from the residual to its output (`g` its
/// weight, `r_i` the position's root mean square).
fn steer_vec(num: &[Vec<f64>], den: &[Vec<f64>], leave_out: Option<usize>) -> Vec<f32> {
    let d = num.first().map_or(0, Vec::len);
    #[expect(clippy::cast_possible_truncation, reason = "a vector kept at 16 bits")]
    (0..d)
        .map(|j| {
            let (a, b) = num.iter().zip(den).enumerate().filter(|(f, _)| Some(*f) != leave_out).fold((0.0, 0.0), |(a, b), (_, (n, d))| (a + n[j], b + d[j]));
            if b > 0.0 { nmengn::half::r16((a / b) as f32) } else { 0.0 }
        })
        .collect()
}

/// The gate's checks of a map the engine runs, in the next-token builds' words.
fn fwd_checks(base: Score, held: Score, ctrl: Score, kl: f64, trials: &[f64]) -> Vec<(&'static str, bool, String)> {
    let gain = base.nll - held.nll;
    let ctrl_gain = base.nll - ctrl.nll;
    vec![
        ("teacher-token loss falls", gain > 0.0, format!("{:.4} to {:.4} ({:+.2}%)", base.nll, held.nll, -100.0 * gain / base.nll.max(1e-12))),
        ("top-1 agreement rises", held.top1 > base.top1, format!("{} to {}", pct(base.top1), pct(held.top1))),
        ("shuffled control gains nothing", ctrl_gain <= GATE_CONTROL * gain.max(0.0), format!("control loss {:.4}, gain {ctrl_gain:+.4} against {gain:+.4} (at most {:.0}% of it)", ctrl.nll, GATE_CONTROL * 100.0)),
        ("neutral drift within the bound", kl <= GATE_KL, format!("{kl:.4} nats per token (bound {GATE_KL})")),
        ("stable over repeated trials", trials.iter().all(|g| *g > 0.0), format!("held-out gains {}", trials.iter().map(|g| format!("{g:+.4}")).collect::<Vec<_>>().join(", "))),
    ]
}

#[expect(clippy::too_many_arguments, reason = "the end of a build is its context, report, clock, verdict, cartridge and figures")]
fn finish(cx: &Fwd, mut out: String, report: &Path, t0: Instant, passed: bool, written: Option<String>, measured: &str) -> Result<String, Fault> {
    let _ = nmapi::audit::audit(
        cx.home,
        "distil-build",
        vec![
            ("role", sv(cx.role)),
            ("technique", sv(cx.technique)),
            ("set", sv(cx.set)),
            ("passed", Value::Bool(passed)),
            ("cartridge", written.as_deref().map_or(Value::Null, sv)),
            ("measured", sv(measured)),
            ("report", sv(&report.display().to_string())),
            ("label", sv(&format!("Distillation into the {} ({}, set {}): {}", cx.role, cx.technique, cx.set, if passed { "passed" } else { "refused" }))),
        ],
    );
    let _ = writeln!(out, "elapsed: {:.0} s; report {}", t0.elapsed().as_secs_f64(), report.display());
    if let Some(dir) = report.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(report, crate::with_overlay(&out));
    Ok(out)
}

fn lineage_of(cx: &Fwd, measured: &str, purpose: &str) -> nmapi::lineage::Record {
    let teachers: Vec<String> = cx.rows.iter().map(|r| r.teacher.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    nmapi::lineage::Record {
        made_on: cx.base.to_owned(),
        made_by: format!("neuromorph-nano lab run distil build {} --target adapter --set {} --technique {}", cx.role, cx.set, cx.technique),
        made_ms: nmtype::now_ms(),
        purpose: purpose.into(),
        used_by: Vec::new(),
        recorded: "tool".into(),
        note: format!("fitted from {} teacher pairs ({})", cx.counts.0, teachers.join(", ")),
        outcome: "used".into(),
        outcome_why: format!("passed its held-out gate: {measured}"),
    }
}

fn strs_of(cx: &Fwd, f: fn(&DistilRow) -> &str) -> Value {
    Value::Array(cx.rows.iter().map(|r| f(r).to_owned()).filter(|x| !x.is_empty()).collect::<std::collections::BTreeSet<_>>().into_iter().map(|x| sv(&x)).collect())
}

/// Distillation by steer: a steering vector at the last layer's residual, the mean shift that moves
/// the final norm's output toward the teacher's targets in first order, its scale measured by the
/// engine with the vector in place; written as a `steer` cartridge on the adapter shelf and served
/// by the serving path's steer seat (`Model::set_steer`).
#[expect(clippy::too_many_lines, reason = "one build: fit, scale, gate and the cartridge, in order")]
fn build_steer(cx: &Fwd, mut out: String, report: &Path, t0: Instant) -> Result<String, Fault> {
    let m = cx.m;
    let d = m.pop.dims.hidden;
    let layer = m.pop.dims.layers.checked_sub(1).ok_or_else(|| bad("a model without layers"))?;
    let res2 = m.asm.nodes.iter().find(|n| n.layer == Some(layer) && n.id.ends_with(".res2")).map(|n| n.id.clone()).ok_or_else(|| bad(format!("no residual node at layer {layer}")))?;
    let fin = (cx.node.to_owned(), "y".to_owned());
    let t = Instant::now();
    let fit = read_split(cx, Split::Fit, &[(res2.clone(), "y".to_owned()), fin.clone()], &RunOpts::default())?;
    if fit.len() != cx.deltas.len() {
        return Err(bad(format!("{} fit positions read again, {} targets", fit.len(), cx.deltas.len())));
    }
    // The final norm's weight from its own input and output: y = g x / r per channel.
    let rms = |x: &[f32]| (x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len().max(1) as f64 + 1e-6).sqrt();
    let (mut gn, mut gd) = (vec![0.0_f64; d], vec![0.0_f64; d]);
    for (_, _, v) in &fit {
        let (x, y) = (&v[0], &v[1]);
        let r = rms(x);
        for j in 0..d.min(x.len()).min(y.len()) {
            let a = f64::from(x[j]) / r;
            gn[j] += f64::from(y[j]) * a;
            gd[j] += a * a;
        }
    }
    let g: Vec<f64> = gn.iter().zip(&gd).map(|(a, b)| if *b > 0.0 { a / b } else { 0.0 }).collect();
    // The shuffled control of a mean shift: a rotation of the shifts leaves their mean as it is, so
    // the control rotates the teacher's tokens instead (each position asked to move toward another
    // position's token, from its own logits) and fits the same vector to those shifts.
    let n_fit = cx.fit_r.len();
    let swapped: Vec<Pos> = (0..n_fit)
        .map(|i| {
            let (a, b) = (cx.fit_r[i], cx.fit_r[(i + n_fit / 3 + 1) % n_fit.max(1)]);
            Pos { h: a.h.clone(), t: b.t, tk: b.tk.clone(), row: a.row }
        })
        .collect();
    let (ctrl_deltas, _, _) = targets(cx.head, &swapped.iter().collect::<Vec<_>>())?;
    let sums = |shift: bool| -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let (mut num, mut den) = (vec![vec![0.0; d]; TRIALS], vec![vec![0.0; d]; TRIALS]);
        for (i, (row, _, v)) in fit.iter().enumerate() {
            let r = rms(&v[0]);
            let dl = if shift { &ctrl_deltas[i] } else { &cx.deltas[i] };
            let f = row % TRIALS;
            for j in 0..d.min(dl.len()) {
                let a = g[j] / r;
                num[f][j] += a * f64::from(dl[j]);
                den[f][j] += a * a;
            }
        }
        (num, den)
    };
    let (num, den) = sums(false);
    let v = steer_vec(&num, &den, None);
    let norm = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    let _ = writeln!(out, "steer at layer {layer} ({res2}): vector norm {norm:.4} from {} fit positions in {:.1} s", fit.len(), t.elapsed().as_secs_f64());
    // The scale, measured by the engine with the vector in place, on the tune split.
    let mut best: Option<(bool, f64, f64, Score, f64)> = None;
    for s in STEER_SCALES {
        let (sc, kl) = engine_figures(cx, Split::Tune, cx.neutral.0, cx.neutral_base.0, &steer_opts(layer, &v, s))?;
        let ok = sc.top1 >= cx.base_tune.top1 && kl <= GATE_KL;
        let gain = cx.base_tune.nll - sc.nll;
        let _ = writeln!(out, "  scale {s}: tune loss {:.4} (base {:.4}), top-1 {} (base {}), neutral drift {kl:.4}{}", sc.nll, cx.base_tune.nll, pct(sc.top1), pct(cx.base_tune.top1), if ok { "" } else { " (outside the tune rule)" });
        if best.as_ref().is_none_or(|b| (ok && !b.0) || (ok == b.0 && gain > b.1)) {
            best = Some((ok, gain, s, sc, kl));
        }
    }
    let (_, _, scale, _, _) = best.ok_or_else(|| bad("no scale measured"))?;
    let _ = writeln!(out, "chosen scale {scale}");
    // The held-out gate, every figure read by the engine with the vector in place.
    let (held, kl) = engine_figures(cx, Split::Held, cx.neutral.1, cx.neutral_base.1, &steer_opts(layer, &v, scale))?;
    let (cn, cd) = sums(true);
    let (ctrl, _) = engine_figures(cx, Split::Held, &[], cx.neutral_base.1, &steer_opts(layer, &steer_vec(&cn, &cd, None), scale))?;
    let mut trials = Vec::new();
    for f in 0..TRIALS {
        let (s, _) = engine_figures(cx, Split::Held, &[], cx.neutral_base.1, &steer_opts(layer, &steer_vec(&num, &den, Some(f)), scale))?;
        trials.push(cx.base_held.nll - s.nll);
    }
    let mut checks = fwd_checks(cx.base_held, held, ctrl, kl, &trials);
    // The served check: the vector seated on the model (the serving path's seat), runs that give no
    // steer of their own give the figures measured with it.
    m.set_steer(steer_opts(layer, &v, scale).steer)?;
    let seated = engine_figures(cx, Split::Held, &[], cx.neutral_base.1, &RunOpts::default());
    m.set_steer(BTreeMap::new())?;
    let (seated, _) = seated?;
    checks.push(("the seated steer gives the measured figures", (seated.nll - held.nll).abs() <= 1e-6 && seated.n == held.n, format!("held-out loss {:.6} seated, {:.6} measured", seated.nll, held.nll)));
    // The Writer's note check, the steer seated on the Writer itself (the serving path's seat).
    if cx.role == "writer" {
        let note_rows: Vec<&DistilRow> = cx.rows.iter().filter(|r| r.split == Split::Held && r.template == "chat").take(NOTE_ROWS).collect();
        let mut answers: Vec<Vec<String>> = vec![Vec::new(), Vec::new()];
        for (arm, outs) in answers.iter_mut().enumerate() {
            m.set_steer(if arm == 1 { steer_opts(layer, &v, scale).steer } else { BTreeMap::new() })?;
            for r in &note_rows {
                let ids = m.pop.tokenizer.encode(&dl::chat_prompt(&r.prompt)).map_err(|e| bad(e.to_string()))?;
                outs.push(m.pop.tokenizer.decode(&answer(m, &cx.head.rows, &ids)?).trim().to_owned());
            }
        }
        m.set_steer(BTreeMap::new())?;
        let prompts: Vec<String> = note_rows.iter().map(|r| r.prompt.clone()).collect();
        let landed = if prompts.len() > nmapi::writer_idle::CHECK_NEIGHBOURS { note_check(cx.home, &prompts, &answers)? } else { vec![0, 0] };
        checks.push(("note check not worse", landed[1] >= landed[0], format!("{} of {} landed, base {}", landed[1], prompts.len(), landed[0])));
    }
    let passed = checks.iter().all(|c| c.1);
    let _ = writeln!(out, "held-out gate ({} positions of {} pairs):", held.n, cx.counts.2);
    for (name, ok, fig) in &checks {
        let _ = writeln!(out, "  {} {name}: {fig}", if *ok { "PASS" } else { "FAIL" });
    }
    let measured = format!("held out: teacher-token loss {:.4} to {:.4}, top-1 {} to {}, neutral drift {kl:.4}", cx.base_held.nll, held.nll, pct(cx.base_held.top1), pct(held.top1));
    let mut written = None;
    if passed {
        let id = dl::cartridge_id(cx.role, &format!("steer-{}", cx.set));
        let gate = obj(checks.iter().map(|(n, ok, fig)| (*n, obj(vec![("pass", Value::Bool(*ok)), ("figures", sv(fig))]))).collect());
        let notes = obj(vec![
            ("values", sv(cx.values_key)),
            ("base", sv(cx.base)),
            ("weights_sha256", sv(&m.pop.card.sha_main)),
            ("role", sv(cx.role)),
            ("technique", sv(cx.technique)),
            ("set", sv(cx.set)),
            ("layer", Value::Number(layer.to_string())),
            ("scale", nv(scale)),
            ("socket", obj(vec![("node", sv(&res2)), ("role", sv("steer")), ("layers", sv(&format!("layer {layer}, its residual output, every position"))), ("form", sv("x + scale * v")), ("v", sv(&format!("[{d}]")))])),
            ("pairs", obj(vec![("fit", Value::Number(cx.counts.0.to_string())), ("tune", Value::Number(cx.counts.1.to_string())), ("held", Value::Number(cx.counts.2.to_string()))])),
            ("teachers", strs_of(cx, |r| r.teacher.as_str())),
            ("licences", strs_of(cx, |r| r.licence.as_str())),
            ("references", strs_of(cx, |r| r.reference.as_str())),
            ("gate", gate),
            ("measured", sv(&measured)),
            ("acts_on", sv(&format!("the {}'s residual after its last layer, every position", if cx.role == "reader" { "Reader" } else { "Writer" }))),
            ("effect", sv("style, format and wording preferences of the teacher; not new knowledge or reasoning")),
            ("lineage", lineage_of(cx, &measured, "distil").to_value()),
        ]);
        let label = nmapi::adapter_fit::LabelIn {
            id: &id,
            version: 0,
            model: &m.pop.card.id,
            weights: &m.pop.card.sha_main,
            kind: dl::STEER_KIND,
            does: &format!("Distilled output behaviour for the {}: a steering vector at its last layer, the mean shift toward a teacher's completions (style, format and wording; not new knowledge)", if cx.role == "reader" { "Reader" } else { "Writer" }),
            made_from: &format!("{} curated teacher pairs of set {}, on {}", cx.counts.0, cx.set, cx.base),
            notes,
        };
        let first = nmapi::adapter::versions(cx.home, &id).is_empty();
        let (file, version) = nmapi::adapter_fit::write_next(cx.home, label, &[("v".to_owned(), vec![d], v.clone())])?;
        // The file as the serving seat reads it must give the vector that was gated.
        let back = nmapi::adapter::read_file(&file).and_then(|(l, p)| dl::steer_from(&l, &p, &m.pop.card.id, cx.values_key, d, m.pop.dims.layers));
        if back.as_ref().map_or(true, |(l, bv, _, _)| *l != layer || *bv != v) {
            let _ = std::fs::remove_file(&file);
            return Err(bad(format!("{}: the steer read back from its file is not the gated one ({}); the file was taken out again", file.display(), back.err().map_or_else(String::new, |e| e.message))));
        }
        if first {
            let _ = nmapi::plugs::change(cx.home, |s| {
                s.units_off.insert(nmapi::plugs::key(nmapi::plugs::NANO, &id));
            });
        }
        let _ = writeln!(out, "WRITTEN: {} ({id}@{version}, a steering vector at layer {layer}, scale {scale}{})", file.display(), if first { ", unplugged until you plug it in" } else { "" });
        written = Some(format!("{id}@{version}"));
    } else {
        let failed: Vec<&str> = checks.iter().filter(|c| !c.1).map(|c| c.0).collect();
        let _ = writeln!(out, "REFUSED: the held-out gate did not pass ({}); nothing was written. {measured}", failed.join("; "));
    }
    finish(cx, out, report, t0, passed, written, &measured)
}

/// The distil loss of a learning unit for the forward-only fitter: the teacher-token loss over a
/// batch of positions, the unit's `up` in place on its node, read from the seeded inputs of the
/// last layer's down projection and residual (everything after them runs per position).
struct UnitLoss<'a> {
    m: &'a Model,
    head: &'a Head<'a>,
    target: String,
    seeds: (String, String),
    rank: usize,
    k: usize,
    n: usize,
    down: Vec<f32>,
    scale: f32,
    /// Per position: the down projection's input, the residual, the teacher's token.
    pos: &'a [(Vec<f32>, Vec<f32>, usize)],
    /// The token each position is scored on (its own, or another position's for the control).
    t: Vec<usize>,
}

impl<'a> UnitLoss<'a> {
    #[expect(clippy::too_many_arguments, reason = "a loss is its model, head, node, seeds, widths, down matrix and positions")]
    fn new(m: &'a Model, head: &'a Head<'a>, target: &str, seeds: &(String, String), (k, n): (usize, usize), down: &[f32], pos: &'a [(Vec<f32>, Vec<f32>, usize)]) -> Self {
        Self { m, head, target: target.to_owned(), seeds: seeds.clone(), rank: UNIT_RANK, k, n, down: down.to_vec(), scale: 1.0, pos, t: pos.iter().map(|p| p.2).collect() }
    }
    fn unit(&self, up: &[f32], scale: f32) -> Result<Arc<nmengn::Adapter>, Fault> {
        nmengn::Adapter::new("distil-unit", &self.target, self.rank, self.k, self.n, self.down.clone(), up.to_vec(), scale).map(Arc::new)
    }
    /// Per-position logits rows over the serving vocabulary, with `unit` seated (none: the base).
    fn logits(&self, unit: Option<Arc<nmengn::Adapter>>, idx: &[usize]) -> Result<Vec<f32>, Fault> {
        self.m.set_unit(&self.target, unit)?;
        let (mut u, mut x) = (Vec::new(), Vec::new());
        for i in idx {
            let Some(p) = self.pos.get(*i) else { continue };
            u.extend_from_slice(&p.0);
            x.extend_from_slice(&p.1);
        }
        let target = (self.head.head.clone(), "logits".to_owned());
        let opts = RunOpts { all_positions: true, head_rows: Arc::clone(&self.head.rows), ..RunOpts::default() };
        let seeds = vec![(self.seeds.0.clone(), "y".to_owned(), Act::Cpu(u)), (self.seeds.1.clone(), "y".to_owned(), Act::Cpu(x))];
        let r = self.m.run_from(device(self.m), seeds, std::slice::from_ref(&target), idx.len(), &opts);
        self.m.set_unit(&self.target, None)?;
        Ok(values(r?.get(&target).ok_or_else(|| bad("no logits"))?)?.to_vec())
    }
    /// Loss and top-1 over `idx` with `unit`.
    fn score(&self, unit: Option<Arc<nmengn::Adapter>>, idx: &[usize]) -> Result<Score, Fault> {
        let na = self.head.rows.len().max(1);
        let mut s = Score::default();
        for chunk in idx.chunks(HEAD_CHUNK) {
            let lg = self.logits(unit.clone(), chunk)?;
            for (row, i) in lg.chunks_exact(na).zip(chunk) {
                let t = self.t.get(*i).copied().unwrap_or(0);
                s.nll += lse(row) - f64::from(row.get(t).copied().unwrap_or(f32::MIN));
                s.top1 += f64::from(u8::from(argmax(row) == t));
                s.n += 1;
            }
        }
        let n = s.n.max(1) as f64;
        s.nll /= n;
        s.top1 /= n;
        Ok(s)
    }
}

impl nmlearn::Loss for UnitLoss<'_> {
    fn loss(&mut self, up: &[f32], batch: &[usize]) -> Result<f64, Fault> {
        let u = self.unit(up, self.scale)?;
        Ok(self.score(Some(u), batch)?.nll)
    }
}

/// Distillation by unit: a learning unit (`nmlearn` registry, `units/<id>/`) at the Reader's
/// learning target, its `up` fitted forward-only by MeZO against the distil loss within a wall
/// budget per fit; its scale chosen on the tune split; the gate on the held-out split with the
/// unit's quantized files as they will load. Written as a fitted learning unit; it serves as every
/// learning unit does (`serve --unit <id>` on the shared model).
#[expect(clippy::too_many_lines, reason = "one build: features, fits, gate and the unit, in order")]
fn build_unit(cx: &Fwd, mut out: String, report: &Path, t0: Instant) -> Result<String, Fault> {
    use nmlearn::Fitter as _;
    let m = cx.m;
    let target = nmapi::learn::target(m)?;
    let (k, n) = nmapi::learn::widths(m, &target)?;
    let from = |port: &str, to: &str| m.asm.edges.iter().find(|e| e.to == to && e.to_port == port).map(|e| e.from.clone());
    let swig = from("x", &target).ok_or_else(|| bad(format!("nothing feeds {target}")))?;
    let res2 = m.asm.edges.iter().find(|e| e.from == target).map(|e| e.to.clone()).ok_or_else(|| bad(format!("{target} feeds nothing")))?;
    let res1 = from("a", &res2).ok_or_else(|| bad(format!("nothing feeds {res2}")))?;
    let budget_s = arg(cx.args, "--budget-s").and_then(|v| v.parse().ok()).unwrap_or(UNIT_BUDGET_S).clamp(1, UNIT_BUDGET_CAP_S);
    let ts = [(swig.clone(), "y".to_owned()), (res1.clone(), "y".to_owned())];
    let to_pos = |reads: Vec<Read>| -> Vec<(Vec<f32>, Vec<f32>, usize)> {
        reads
            .into_iter()
            .map(|(_, t, mut v)| {
                let x = v.pop().unwrap_or_default();
                let u = v.pop().unwrap_or_default();
                (u, x, t)
            })
            .collect()
    };
    let t = Instant::now();
    let fit_raw = read_split(cx, Split::Fit, &ts, &RunOpts::default())?;
    let fit_rows: Vec<usize> = fit_raw.iter().map(|r| r.0).collect();
    let fit = to_pos(fit_raw);
    let tune = to_pos(read_split(cx, Split::Tune, &ts, &RunOpts::default())?);
    let held = to_pos(read_split(cx, Split::Held, &ts, &RunOpts::default())?);
    let _ = writeln!(out, "unit at {target} ({k} in, {n} out, rank {UNIT_RANK}), seeded from {swig} and {res1}; positions fit {}, tune {}, held out {}; {:.1} s", fit.len(), tune.len(), held.len(), t.elapsed().as_secs_f64());
    let card0 = nmlearn::UnitCard::new("distil-unit", &m.pop.card.id, &m.pop.card.sha_main, cx.role, &target, UNIT_RANK, UNIT_SEED);
    let w0 = nmlearn::UnitWeights::unfitted(&card0, k, n);
    let seeds = (swig.clone(), res1.clone());
    let mk = |pos| UnitLoss::new(m, cx.head, &target, &seeds, (k, n), &w0.down, pos);
    let all: Vec<usize> = (0..fit.len()).collect();
    // Batches by third of the fit pairs, so a trial leaves its third out of every batch.
    let batches_without = |leave: Option<usize>| -> Vec<Vec<usize>> {
        let keep: Vec<usize> = all.iter().copied().filter(|i| leave.is_none_or(|f| fit_rows.get(*i).is_none_or(|r| r % TRIALS != f))).collect();
        keep.chunks(UNIT_BATCH).map(<[usize]>::to_vec).collect()
    };
    let mezo = nmlearn::Mezo { eps: UNIT_EPS, lr: UNIT_LR, seed: UNIT_SEED, max_pairs: 10_000, wall_ms: budget_s * 1000, budget_ms: budget_s * 1000 + 5000, patience: 200 };
    let fit_up = |loss: &mut UnitLoss, leave: Option<usize>| -> Result<(Vec<f32>, nmlearn::FitLog), Fault> {
        let mut up = w0.up.clone();
        let rows: Vec<usize> = (0..n).collect();
        let log = mezo.fit(&mut up, UNIT_RANK, &rows, &batches_without(leave), loss)?;
        Ok((up, log))
    };
    let mut loss = mk(&fit);
    let (up, log) = fit_up(&mut loss, None)?;
    let _ = writeln!(out, "forward-only fit: {} pairs in {} ms ({}), batch loss {:.4} first, {:.4} best", log.pairs.len(), log.wall_ms, log.stopped, log.first_loss, log.best_loss);
    // The control: the same fit toward each position's token taken from another position.
    let rot = fit.len() / 3 + 1;
    let mut ctrl_loss = mk(&fit);
    ctrl_loss.t = (0..fit.len()).map(|i| fit[(i + rot) % fit.len().max(1)].2).collect();
    let (ctrl_up, _) = fit_up(&mut ctrl_loss, None)?;
    // What will load: the quantized files.
    let q = |up: &[f32]| -> Result<nmlearn::UnitWeights, Fault> { nmlearn::UnitWeights { up: up.to_vec(), ..w0.clone() }.quantized() };
    let wq = q(&up)?;
    let tune_l = UnitLoss { down: wq.down.clone(), ..mk(&tune) };
    let tidx: Vec<usize> = (0..tune.len()).collect();
    // Neutral drift with the unit in place, read by the engine.
    let neutral_kl = |unit: &Arc<nmengn::Adapter>, chunks: &[Vec<i32>], base: &[f32]| -> Result<f64, Fault> {
        m.set_unit(&target, Some(Arc::clone(unit)))?;
        let fin = (cx.node.to_owned(), "y".to_owned());
        let got = read_neutral(m, chunks, std::slice::from_ref(&fin), &RunOpts::default());
        m.set_unit(&target, None)?;
        let (hs, nn) = got?;
        drift_on(cx.head, hs.first().map_or(&[][..], Vec::as_slice), nn, base)
    };
    let mut best: Option<(bool, f64, f64)> = None;
    for s in UNIT_SCALES {
        #[expect(clippy::cast_possible_truncation, reason = "a scale kept at 32 bits")]
        let u = tune_l.unit(&wq.up, s as f32)?;
        let sc = tune_l.score(Some(Arc::clone(&u)), &tidx)?;
        let kl = neutral_kl(&u, cx.neutral.0, cx.neutral_base.0)?;
        let ok = sc.top1 >= cx.base_tune.top1 && kl <= GATE_KL;
        let gain = cx.base_tune.nll - sc.nll;
        let _ = writeln!(out, "  scale {s}: tune loss {:.4} (base {:.4}), top-1 {} (base {}), neutral drift {kl:.4}{}", sc.nll, cx.base_tune.nll, pct(sc.top1), pct(cx.base_tune.top1), if ok { "" } else { " (outside the tune rule)" });
        if best.is_none_or(|b| (ok && !b.0) || (ok == b.0 && gain > b.1)) {
            best = Some((ok, gain, s));
        }
    }
    let (_, _, scale) = best.ok_or_else(|| bad("no scale measured"))?;
    #[expect(clippy::cast_possible_truncation, reason = "a scale kept at 32 bits")]
    let sf = scale as f32;
    let held_l = UnitLoss { down: wq.down.clone(), ..mk(&held) };
    let hidx: Vec<usize> = (0..held.len()).collect();
    let u = held_l.unit(&wq.up, sf)?;
    let held_s = held_l.score(Some(Arc::clone(&u)), &hidx)?;
    let kl = neutral_kl(&u, cx.neutral.1, cx.neutral_base.1)?;
    let ctrl_s = held_l.score(Some(held_l.unit(&q(&ctrl_up)?.up, sf)?), &hidx)?;
    let mut trials = Vec::new();
    for f in 0..TRIALS {
        let mut l = mk(&fit);
        let (tu, _) = fit_up(&mut l, Some(f))?;
        trials.push(cx.base_held.nll - held_l.score(Some(held_l.unit(&q(&tu)?.up, sf)?), &hidx)?.nll);
    }
    let checks = fwd_checks(cx.base_held, held_s, ctrl_s, kl, &trials);
    let passed = checks.iter().all(|c| c.1);
    let _ = writeln!(out, "held-out gate ({} positions of {} pairs; the unit as its quantized files load):", held_s.n, cx.counts.2);
    for (name, ok, fig) in &checks {
        let _ = writeln!(out, "  {} {name}: {fig}", if *ok { "PASS" } else { "FAIL" });
    }
    let measured = format!("held out: teacher-token loss {:.4} to {:.4}, top-1 {} to {}, neutral drift {kl:.4}", cx.base_held.nll, held_s.nll, pct(cx.base_held.top1), pct(held_s.top1));
    let mut written = None;
    if passed {
        let mut units = nmlearn::Units::open(cx.home);
        let stem = format!("lwu-distil-{}", dl::slug(cx.set));
        let id = (1..).map(|v| format!("{stem}-v{v}")).find(|c| units.get(c).is_none()).unwrap_or(stem);
        let mut card = nmlearn::UnitCard::new(&id, &m.pop.card.id, &m.pop.card.sha_main, cx.role, &target, UNIT_RANK, UNIT_SEED);
        card.scale = sf;
        // A card's lineage names the registered units it was made from (the registry refuses any
        // other entry); a distilled unit has none. Its set, base and values are in `distil.json`.
        units.register(card)?;
        let w = nmlearn::UnitWeights { up, scale: sf, ..w0 };
        let mut files = Vec::new();
        for (name, bytes) in w.files()? {
            files.push((name.clone(), units.commit_file(&id, &name, &bytes)?));
        }
        let gate = obj(checks.iter().map(|(n, ok, fig)| (*n, obj(vec![("pass", Value::Bool(*ok)), ("figures", sv(fig))]))).collect());
        let record = obj(vec![
            ("technique", sv("unit")),
            ("set", sv(cx.set)),
            ("values", sv(cx.values_key)),
            ("base", sv(cx.base)),
            ("fitter", sv(&format!("MeZO eps {UNIT_EPS} lr {UNIT_LR}, {} pairs in {} ms ({}), budget {budget_s} s per fit", log.pairs.len(), log.wall_ms, log.stopped))),
            ("scale", nv(scale)),
            ("gate", gate),
            ("measured", sv(&measured)),
            ("lineage", lineage_of(cx, &measured, "distil").to_value()),
        ]);
        let rec = nmjson::pretty::pretty(&record);
        files.push(("distil.json".to_owned(), units.commit_file(&id, "distil.json", rec.as_bytes())?));
        units.set_files(&id, files, "fitted")?;
        let _ = writeln!(out, "WRITTEN: {} ({id}, a fitted learning unit at {target}, rank {UNIT_RANK}, scale {scale}; it loads with serve --unit {id})", units.unit_dir(&id).display());
        written = Some(id);
    } else {
        let failed: Vec<&str> = checks.iter().filter(|c| !c.1).map(|c| c.0).collect();
        let _ = writeln!(out, "REFUSED: the held-out gate did not pass ({}); nothing was written. {measured}", failed.join("; "));
    }
    finish(cx, out, report, t0, passed, written, &measured)
}

// ---------------------------------------------------------------- the embedder: a request lens

/// Neutral drift bound of the embedder's request lens: the mean cosine distance between a neutral
/// request's vector with and without the lens.
const GATE_EMBED_DRIFT: f64 = 0.02;

/// The embedder's vector of a request as the dense lane makes it (its instruction, then the query).
fn request_vec(e: &Model, q: &str) -> Result<Vec<f32>, Fault> {
    nmengn::roles::card_embed_on(e, device(e), &nmengn::roles::card_query(nmsubs::dense::task_for(q), q))
}

/// A request lens of the fitted map at rank `r` and scale `scale` (no flattening: the stored
/// vectors' side is never touched).
fn embed_lens(key: &str, values: &str, d: usize, map: &(Vec<f64>, Vec<f64>), r: usize, scale: f64) -> nmapi::adapter::Lens {
    let (dn, upr) = dl::leading(&map.0, &map.1, d, r);
    let r = dn.len() / d.max(1);
    #[expect(clippy::cast_possible_truncation, reason = "a scale kept at 32 bits")]
    let s = scale as f32;
    nmapi::adapter::Lens { key: key.to_owned(), values: values.to_owned(), dim: d, flat: Vec::new(), beta: Vec::new(), rank: r, down: dn.iter().map(|x| nmengn::half::r16(*x)).collect(), up: upr.iter().map(|x| nmengn::half::r16(x * s)).collect() }
}

/// Measures of the embedder on scored pairs: the mean squared difference between the cosine and
/// its target, and the share of same-query pairs with different teacher scores ordered as the
/// teacher orders them.
#[derive(Clone, Copy, Default, Debug)]
struct EScore {
    mse: f64,
    order: f64,
    n: usize,
}

/// One scored pair of the embedder: its query's index, its document's vector, the teacher's score,
/// its target cosine, its row.
struct EPair {
    q: usize,
    doc: Vec<f32>,
    p: f64,
    target: f64,
    row: usize,
}

/// One choice of the request lens on the tune split.
struct EChoice {
    c: f64,
    r: usize,
    s: f64,
    map: (Vec<f64>, Vec<f64>),
    tune: EScore,
    kl: f64,
}

fn escore(qs: &[Vec<f32>], pairs: &[&EPair], lens: Option<&nmapi::adapter::Lens>) -> EScore {
    let mapped: BTreeMap<usize, Vec<f32>> = pairs.iter().map(|p| p.q).collect::<std::collections::BTreeSet<_>>().into_iter().map(|q| (q, lens.map_or_else(|| qs[q].clone(), |l| l.apply(&qs[q])))).collect();
    let cos: Vec<f64> = pairs.iter().map(|p| f64::from(nmapi::adapter::dot(&mapped[&p.q], &p.doc))).collect();
    let mse = cos.iter().zip(pairs).map(|(c, p)| (c - p.target).powi(2)).sum::<f64>() / pairs.len().max(1) as f64;
    let (mut hit, mut all) = (0_usize, 0_usize);
    for i in 0..pairs.len() {
        for j in i + 1..pairs.len() {
            if pairs[i].q == pairs[j].q && (pairs[i].p - pairs[j].p).abs() > 1e-9 {
                all += 1;
                hit += usize::from((cos[i] - cos[j]) * (pairs[i].p - pairs[j].p) > 0.0);
            }
        }
    }
    EScore { mse, order: if all == 0 { 0.0 } else { hit as f64 / all as f64 }, n: pairs.len() }
}

fn embed_drift(neutral: &[Vec<f32>], lens: &nmapi::adapter::Lens) -> f64 {
    neutral.iter().map(|q| 1.0 - f64::from(nmapi::adapter::dot(q, &lens.apply(q)))).sum::<f64>() / neutral.len().max(1) as f64
}

/// Distillation into the Embedder from teacher similarity scores: a request lens (the embedder
/// lens's own form, `lens` kind, no flattening) fitted so the cosine of a query's vector to its
/// documents' vectors follows the teacher's scores (mapped onto the embedder's own cosine scale by
/// matching their mean and spread on the fit split). Per query, the smallest shift of its vector
/// that moves its cosines to their targets (`target_delta`, the documents' vectors as rows), then
/// reduced-rank ridge from the vector to the shifted one. It maps request vectors only: stored
/// piece vectors stay as they are, and it is served before the project's lens with its key carried
/// into the lens key and the lens epoch (`distil::query_lens`, `distil::lens_key`).
#[expect(clippy::too_many_lines, reason = "one build: data, features, fit, gate and the cartridge, in order")]
fn build_embed(home: &Path, args: &[String], technique: &str, set: &str) -> Result<String, Fault> {
    let t0 = Instant::now();
    let role = "embedder";
    let rounds = if technique == "teacher-loop" { arg(args, "--rounds").and_then(|v| v.parse().ok()).unwrap_or(ROUNDS).clamp(1, 8) } else { 1 };
    let report = report_file(home, &format!("embedder-{}", dl::slug(set)));
    let mut out = String::new();
    let _ = writeln!(out, "distillation into the embedder: a request lens (adapter), set {set}, technique {technique}");
    let rows: Vec<DistilRow> = rows_of(home, set)?.into_iter().filter(|r| r.score.is_some()).collect();
    let count = |s: Split| rows.iter().filter(|r| r.split == s).count();
    let (nf, nt, nh) = (count(Split::Fit), count(Split::Tune), count(Split::Held));
    let _ = writeln!(out, "scored pairs (healthy only): fit {nf}, tune {nt}, held out {nh}");
    if nf < MIN_FIT || nt < MIN_TUNE || nh < MIN_HELD {
        return Err(bad(format!("too few scored pairs: fit {nf} (at least {MIN_FIT}), tune {nt} (at least {MIN_TUNE}), held out {nh} (at least {MIN_HELD}); import teacher pairs with a similarity score from 0 to 1 (prompt the query, completion the document)")));
    }
    let offered = seat_base(home, args, role);
    let e = nmapi::neural::embedder_model(home)?;
    let d = e.pop.dims.hidden;
    let values_key = nmapi::neural::parameters_key(&e);
    let base = made_on(&e);
    let _ = writeln!(out, "Embedder {} on {base} (offered {offered}), device {}, values {values_key}", e.pop.card.id, device(&e).name());
    let t = Instant::now();
    let mut qi: BTreeMap<String, usize> = BTreeMap::new();
    let mut qs: Vec<Vec<f32>> = Vec::new();
    let mut pairs: Vec<EPair> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let q = match qi.get(&r.prompt) {
            Some(q) => *q,
            None => {
                qs.push(request_vec(&e, &r.prompt)?);
                qi.insert(r.prompt.clone(), qs.len() - 1);
                qs.len() - 1
            }
        };
        let doc = nmengn::roles::card_embed_on(&e, device(&e), &r.completion)?;
        pairs.push(EPair { q, doc, p: r.score.unwrap_or(0.5), target: 0.0, row: i });
    }
    // The teacher's scores on the embedder's cosine scale: their mean and spread matched on the fit split.
    let fit_ix: Vec<usize> = pairs.iter().enumerate().filter(|(_, p)| rows[p.row].split == Split::Fit).map(|(i, _)| i).collect();
    let stats = |xs: &[f64]| {
        let m = xs.iter().sum::<f64>() / xs.len().max(1) as f64;
        (m, (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len().max(1) as f64).sqrt())
    };
    let cos0: Vec<f64> = fit_ix.iter().map(|i| f64::from(nmapi::adapter::dot(&qs[pairs[*i].q], &pairs[*i].doc))).collect();
    let (cm, cs) = stats(&cos0);
    let (pm, ps) = stats(&fit_ix.iter().map(|i| pairs[*i].p).collect::<Vec<_>>());
    for p in &mut pairs {
        p.target = cm + cs * (p.p - pm) / ps.max(1e-9);
    }
    let _ = writeln!(out, "vectors of {} queries and {} documents in {:.1} s; target cosine = {cm:.4} + {cs:.4} (score - {pm:.4}) / {ps:.4}", qs.len(), pairs.len(), t.elapsed().as_secs_f64());
    let split = |s: Split| -> Vec<&EPair> { pairs.iter().filter(|p| rows[p.row].split == s).collect() };
    let (fit, tune, held) = (split(Split::Fit), split(Split::Tune), split(Split::Held));
    // Neutral requests: the first tokens of the Nano's own source chunks.
    let tok = &e.pop.tokenizer;
    let (nt_ids, nh_ids) = neutral_ids(home, tok);
    let nvec = |chunks: &[Vec<i32>]| -> Result<Vec<Vec<f32>>, Fault> { chunks.iter().map(|c| request_vec(&e, &tok.decode(c.get(..NEUTRAL_QUERY.min(c.len())).unwrap_or(&[])))).collect() };
    let (n_tune, n_held) = (nvec(&nt_ids)?, nvec(&nh_ids)?);
    // Per fit query: the shift of its vector that moves its documents' cosines to their targets.
    let mut by_q: BTreeMap<usize, Vec<&EPair>> = BTreeMap::new();
    for p in &fit {
        by_q.entry(p.q).or_default().push(p);
    }
    let mut shifts: Vec<(usize, usize, Vec<f32>)> = Vec::new();
    for (q, ps_) in &by_q {
        let docs: Vec<&[f32]> = ps_.iter().map(|p| p.doc.as_slice()).collect();
        let z: Vec<f64> = ps_.iter().map(|p| f64::from(nmapi::adapter::dot(&qs[*q], &p.doc))).collect();
        let goal: Vec<f64> = ps_.iter().map(|p| p.target).collect();
        #[expect(clippy::cast_possible_truncation, reason = "shifts kept at 32 bits")]
        let dlt: Vec<f32> = dl::target_delta(&docs, &z, &goal, dl::MU).map_or_else(|| vec![0.0; d], |v| v.into_iter().map(|x| x as f32).collect());
        let row = ps_.first().map_or(0, |p| p.row);
        shifts.push((*q, row, dlt));
    }
    let sums = |weights: &BTreeMap<usize, f64>, shift: bool| -> Vec<dl::Sums> {
        let n = shifts.len();
        (0..TRIALS)
            .map(|f| {
                let mut s = dl::Sums::new(d);
                let rows_: Vec<(&[f32], &[f32], f64)> = shifts
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, r, _))| r % TRIALS == f)
                    .map(|(i, (q, r, _))| (qs[*q].as_slice(), shifts[if shift { (i + n / 3 + 1) % n.max(1) } else { i }].2.as_slice(), weights.get(r).copied().unwrap_or(1.0)))
                    .collect();
                s.add(&rows_);
                s
            })
            .collect()
    };
    let solve_e = |folds: &[dl::Sums], leave: Option<usize>, c: f64| -> Option<(Vec<f64>, Vec<f64>)> {
        let mut t = dl::Sums::new(d);
        for (_, f) in folds.iter().enumerate().filter(|(i, _)| Some(*i) != leave) {
            t.plus(f);
        }
        nmapi::adapter_fit::ridge_low_rank_sums(t.g, &t.cy, d, c, *RANKS.iter().max().unwrap_or(&32))
    };
    let key = format!("{}@next", dl::cartridge_id(role, set));
    let (base_tune, base_held) = (escore(&qs, &tune, None), escore(&qs, &held, None));
    let mut weights: BTreeMap<usize, f64> = BTreeMap::new();
    let mut chosen: Option<(EChoice, BTreeMap<usize, f64>)> = None;
    for round in 0..rounds {
        let folds = sums(&weights, false);
        let mut best: Option<(bool, f64, EChoice)> = None;
        let grid: Vec<(f64, usize, f64)> = match &chosen {
            None => RIDGES.iter().flat_map(|c| RANKS.iter().flat_map(move |r| SCALES.iter().map(move |s| (*c, *r, *s)))).collect(),
            Some((p, _)) => vec![(p.c, p.r, p.s)],
        };
        for (c, r, s) in grid {
            let Some(map) = solve_e(&folds, None, c) else { continue };
            let l = embed_lens(&key, &values_key, d, &map, r, s);
            let sc = escore(&qs, &tune, Some(&l));
            let kl = embed_drift(&n_tune, &l);
            let ok = sc.order >= base_tune.order && kl <= GATE_EMBED_DRIFT;
            let gain = base_tune.mse - sc.mse;
            let _ = writeln!(out, "  ridge {c} rank {r} scale {s}: tune error {:.5} (base {:.5}), order {} (base {}), neutral drift {kl:.4}{}", sc.mse, base_tune.mse, pct(sc.order), pct(base_tune.order), if ok { "" } else { " (outside the tune rule)" });
            if best.as_ref().is_none_or(|b| (ok && !b.0) || (ok == b.0 && gain > b.1)) {
                best = Some((ok, gain, EChoice { c, r, s, map, tune: sc, kl }));
            }
        }
        let Some((_, _, ch)) = best else { return Err(bad("no ridge strength gave a solvable fit")) };
        if chosen.as_ref().is_some_and(|(p, _)| !(ch.tune.mse < p.tune.mse && ch.kl <= GATE_EMBED_DRIFT)) {
            let _ = writeln!(out, "round {}: the tune split stopped gaining; the previous round stands", round + 1);
            break;
        }
        let _ = writeln!(out, "round {}: chosen ridge {} rank {} scale {}: tune error {:.5}, order {}, neutral drift {:.4}", round + 1, ch.c, ch.r, ch.s, ch.tune.mse, pct(ch.tune.order), ch.kl);
        let next = if round + 1 < rounds {
            // The third of the fit queries the lens still follows the teacher worst on, weighed up.
            let l = embed_lens(&key, &values_key, d, &ch.map, ch.r, ch.s);
            let mut worst: Vec<(f64, usize)> = by_q.values().map(|ps_| (escore(&qs, ps_, Some(&l)).mse, ps_.first().map_or(0, |p| p.row))).collect();
            worst.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut w = weights.clone();
            for (_, row) in worst.iter().take(worst.len().div_ceil(3)) {
                let x = w.entry(*row).or_insert(1.0);
                *x = (*x * 1.5).min(4.0);
            }
            w
        } else {
            weights.clone()
        };
        chosen = Some((ch, weights.clone()));
        weights = next;
    }
    let (EChoice { c, r, s, map, .. }, wts) = chosen.ok_or_else(|| bad("no round produced a fit"))?;
    let lens = embed_lens(&key, &values_key, d, &map, r, s);
    let held_s = escore(&qs, &held, Some(&lens));
    let ctrl = solve_e(&sums(&wts, true), None, c).ok_or_else(|| bad("the control did not solve"))?;
    let ctrl_s = escore(&qs, &held, Some(&embed_lens("control", &values_key, d, &ctrl, r, s)));
    let drift = embed_drift(&n_held, &lens);
    let folds = sums(&wts, false);
    let mut trials = Vec::new();
    for f in 0..TRIALS {
        let m2 = solve_e(&folds, Some(f), c).ok_or_else(|| bad("a trial did not solve"))?;
        trials.push(base_held.mse - escore(&qs, &held, Some(&embed_lens("trial", &values_key, d, &m2, r, s))).mse);
    }
    // The served check: the lens read back from its own file form gives the same request vectors.
    let parts: Vec<nmapi::adapter::Part> = vec![("down".to_owned(), vec![lens.rank, d], lens.down.clone()), ("up".to_owned(), vec![d, lens.rank], lens.up.clone())];
    let gain = base_held.mse - held_s.mse;
    let ctrl_gain = base_held.mse - ctrl_s.mse;
    let checks = [
        ("cosine error from the teacher's scores falls", gain > 0.0, format!("{:.5} to {:.5} ({:+.2}%)", base_held.mse, held_s.mse, -100.0 * gain / base_held.mse.max(1e-12))),
        ("order agreement with the teacher does not fall", held_s.order >= base_held.order, format!("{} to {}", pct(base_held.order), pct(held_s.order))),
        ("shuffled control gains nothing", ctrl_gain <= GATE_CONTROL * gain.max(0.0), format!("control error {:.5}, gain {ctrl_gain:+.5} against {gain:+.5} (at most {:.0}% of it)", ctrl_s.mse, GATE_CONTROL * 100.0)),
        ("neutral drift within the bound", drift <= GATE_EMBED_DRIFT, format!("{drift:.4} mean cosine distance of neutral requests (bound {GATE_EMBED_DRIFT})")),
        ("stable over repeated trials", trials.iter().all(|g| *g > 0.0), format!("held-out gains {}", trials.iter().map(|g| format!("{g:+.5}")).collect::<Vec<_>>().join(", "))),
    ];
    let passed = checks.iter().all(|c| c.1);
    let _ = writeln!(out, "held-out gate ({} pairs):", held_s.n);
    for (name, ok, fig) in &checks {
        let _ = writeln!(out, "  {} {name}: {fig}", if *ok { "PASS" } else { "FAIL" });
    }
    let measured = format!("held out: cosine error {:.5} to {:.5}, order {} to {}, control gain {ctrl_gain:+.5}, neutral drift {drift:.4}", base_held.mse, held_s.mse, pct(base_held.order), pct(held_s.order));
    let mut written = None;
    if passed {
        let id = dl::cartridge_id(role, set);
        let teachers: Vec<String> = rows.iter().map(|r| r.teacher.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let strs = |v: Vec<String>| Value::Array(v.into_iter().map(|x| sv(&x)).collect());
        let gate = obj(checks.iter().map(|(n, ok, fig)| (*n, obj(vec![("pass", Value::Bool(*ok)), ("figures", sv(fig))]))).collect());
        let lineage = nmapi::lineage::Record {
            made_on: base.clone(),
            made_by: format!("neuromorph-nano lab run distil build embedder --target adapter --set {set} --technique {technique}"),
            made_ms: nmtype::now_ms(),
            purpose: "distil".into(),
            used_by: Vec::new(),
            recorded: "tool".into(),
            note: format!("fitted from {nf} teacher similarity scores ({})", teachers.join(", ")),
            outcome: "used".into(),
            outcome_why: format!("passed its held-out gate: {measured}"),
        };
        let notes = obj(vec![
            ("values", sv(&values_key)),
            ("base", sv(&base)),
            ("weights_sha256", sv(&e.pop.card.sha_main)),
            ("role", sv(role)),
            ("technique", sv(technique)),
            ("set", sv(set)),
            ("scale", nv(s)),
            ("ridge", nv(c)),
            ("socket", obj(vec![("node", sv("request vector")), ("role", sv("request lens")), ("layers", sv("after pooling, the request's vector only; stored piece vectors untouched")), ("form", sv("normalize(q + up(down(q))), before the project's lens")), ("rank", Value::Number(lens.rank.to_string())), ("down", sv(&format!("[{}, {d}]", lens.rank))), ("up", sv(&format!("[{d}, {}]", lens.rank)))])),
            ("target_cosine", sv(&format!("{cm:.6} + {cs:.6} (score - {pm:.6}) / {ps:.6}"))),
            ("pairs", obj(vec![("fit", Value::Number(nf.to_string())), ("tune", Value::Number(nt.to_string())), ("held", Value::Number(nh.to_string()))])),
            ("teachers", strs(teachers.clone())),
            ("licences", strs(rows.iter().map(|r| r.licence.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().collect())),
            ("gate", gate),
            ("measured", sv(&measured)),
            ("acts_on", sv("the embedder's request vectors (the dense lane's questions), before the project's lens")),
            ("effect", sv("a request's similarity to the pieces follows the teacher's similarity judgments; the stored piece vectors and the dense store are unchanged")),
            ("lineage", lineage.to_value()),
        ]);
        let label = nmapi::adapter_fit::LabelIn {
            id: &id,
            version: 0,
            model: &e.pop.card.id,
            weights: &e.pop.card.sha_main,
            kind: nmapi::adapter::LENS_KIND,
            does: "Distilled similarity for the Embedder: a request lens fitted so a request's similarity to documents follows a teacher's similarity scores; stored vectors are untouched",
            made_from: &format!("{nf} teacher similarity scores of set {set} ({}), on {base}", teachers.join(", ")),
            notes,
        };
        let first = nmapi::adapter::versions(home, &id).is_empty();
        let (file, version) = nmapi::adapter_fit::write_next(home, label, &parts)?;
        // The file as the serving path reads it must give the lens that was gated.
        let back = nmapi::adapter::lens_of(&file, &values_key)?;
        if back.down != lens.down || back.up != lens.up {
            let _ = std::fs::remove_file(&file);
            return Err(bad(format!("{}: the lens read back from its file differs from the gated one; the file was taken out again", file.display())));
        }
        if first {
            let _ = nmapi::plugs::change(home, |st| {
                st.units_off.insert(nmapi::plugs::key(nmapi::plugs::NANO, &id));
            });
        }
        let _ = writeln!(out, "WRITTEN: {} ({id}@{version}, a request lens of rank {}, scale {s}{}; the stored piece vectors are unchanged)", file.display(), lens.rank, if first { ", unplugged until you plug it in" } else { "" });
        written = Some(format!("{id}@{version}"));
    } else {
        let failed: Vec<&str> = checks.iter().filter(|c| !c.1).map(|c| c.0).collect();
        let _ = writeln!(out, "REFUSED: the held-out gate did not pass ({}); nothing was written. {measured}", failed.join("; "));
    }
    let _ = nmapi::audit::audit(
        home,
        "distil-build",
        vec![
            ("role", sv(role)),
            ("technique", sv(technique)),
            ("set", sv(set)),
            ("passed", Value::Bool(passed)),
            ("cartridge", written.as_deref().map_or(Value::Null, sv)),
            ("measured", sv(&measured)),
            ("report", sv(&report.display().to_string())),
            ("label", sv(&format!("Distillation into the embedder ({technique}, set {set}): {}", if passed { "passed" } else { "refused" }))),
        ],
    );
    let _ = writeln!(out, "elapsed: {:.0} s; report {}", t0.elapsed().as_secs_f64(), report.display());
    if let Some(dir) = report.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&report, crate::with_overlay(&out));
    Ok(out)
}


#[expect(clippy::too_many_lines, reason = "one build: data, features, fit, gate and the cartridge, in order")]
fn build(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let role = args.get(3).map_or("writer", String::as_str);
    let technique = arg(args, "--technique").unwrap_or("lens");
    let target = arg(args, "--target").unwrap_or(if technique == "channel" { "filter" } else { "adapter" });
    if let Some(why) = dl::not_built(role, target, technique) {
        return Err(bad(why));
    }
    let set = arg(args, "--set").unwrap_or("all");
    if target == "preference" {
        return build_preference(home, set);
    }
    if role == "reranker" {
        return build_judge(home, args, role, target, technique, set);
    }
    if role == "embedder" {
        return build_embed(home, args, technique, set);
    }
    let rounds = if technique == "teacher-loop" { arg(args, "--rounds").and_then(|v| v.parse().ok()).unwrap_or(ROUNDS).clamp(1, 8) } else { 1 };
    let report = report_file(home, &dl::slug(set));
    let mut out = String::new();
    let _ = writeln!(out, "distillation into the {role}: a {target} cartridge, set {set}, technique {technique}{}", if rounds > 1 { format!(", at most {rounds} rounds") } else { String::new() });
    // Next-token targets come from teacher completions: a scored pair's completion is a document
    // the teacher judged, and a rejected completion is one the teacher would not write.
    let rows: Vec<DistilRow> = rows_of(home, set)?.into_iter().filter(|r| r.score.is_none() && r.verdict != "reject").collect();
    let count = |s: Split| rows.iter().filter(|r| r.split == s).count();
    let (nf, nt, nh) = (count(Split::Fit), count(Split::Tune), count(Split::Held));
    let _ = writeln!(out, "pairs (healthy only): fit {nf}, tune {nt}, held out {nh}");
    if nf < MIN_FIT || nt < MIN_TUNE || nh < MIN_HELD {
        return Err(bad(format!("too few pairs: fit {nf} (at least {MIN_FIT}), tune {nt} (at least {MIN_TUNE}), held out {nh} (at least {MIN_HELD}); import more teacher pairs")));
    }
    let offered = seat_base(home, args, role);
    let m = if role == "reader" { nmapi::neural::model(home)? } else { nmapi::neural::writer_model(home, true)? };
    let node = m.role_node(dl::SOCKET)?;
    // Features are the base's own: whatever distilled cartridge is plugged is taken out here.
    m.set_unit(&node, None)?;
    let d = m.pop.dims.hidden;
    let dev = device(&m);
    let values_key = nmapi::neural::parameters_key(&m);
    let base = made_on(&m);
    let _ = writeln!(out, "{} {} on {base} (offered {offered}), device {}, values {values_key}", if role == "reader" { "Reader" } else { "Writer" }, m.pop.card.id, dev.name());
    // Pairs to ids; the serving vocabulary from them.
    let tok = &m.pop.tokenizer;
    let mut seqs: Vec<(usize, Vec<i32>, usize)> = Vec::new();
    let mut seen: Vec<i32> = Vec::new();
    let mut too_long = 0;
    for (i, r) in rows.iter().enumerate() {
        let (ids, plen) = dl::sequence(tok, r)?;
        if ids.len() > MAX_TOKENS || plen == 0 {
            too_long += 1;
            continue;
        }
        seen.extend_from_slice(&ids);
        seqs.push((i, ids, plen));
    }
    let serving = dl::serving_rows(m.pop.dims.vocab, &seen);
    let at: BTreeMap<usize, usize> = serving.iter().enumerate().map(|(a, t)| (*t, a)).collect();
    let t = Instant::now();
    let head = Head::new(&m, serving)?;
    let _ = writeln!(out, "serving vocabulary {} rows; head rows read from the engine in {:.1} s; pairs over {MAX_TOKENS} tokens left out: {too_long}", head.rows.len(), t.elapsed().as_secs_f64());
    // Features under teacher forcing.
    let t = Instant::now();
    let fin = (node.clone(), "y".to_owned());
    let (mut fit, mut tune, mut held): (Vec<Pos>, Vec<Pos>, Vec<Pos>) = (Vec::new(), Vec::new(), Vec::new());
    let (mut outside, mut aligned, mut misaligned) = (0, 0, 0);
    for (i, ids, plen) in &seqs {
        let r = &rows[*i];
        let y = m.forward(dev, ids, std::slice::from_ref(&fin), &RunOpts::default())?;
        let y = values(y.get(&fin).ok_or_else(|| bad("no final-norm state"))?)?;
        let scored = ids.len() - plen;
        let topk_ok = r.topk.len() == scored && r.topk.iter().zip(&ids[*plen..]).all(|(tk, want)| tk.first().is_some_and(|(id, _)| i64::from(*id) == i64::from(*want)));
        if !r.topk.is_empty() {
            if topk_ok { aligned += 1 } else { misaligned += 1 }
        }
        for k in 0..scored {
            let p = plen - 1 + k;
            let Some(tgt) = ids.get(p + 1).and_then(|x| usize::try_from(*x).ok()).and_then(|x| at.get(&x).copied()) else {
                outside += 1;
                continue;
            };
            let tk = if topk_ok { r.topk[k].iter().filter_map(|(id, lp)| at.get(&(*id as usize)).map(|a| (*a, *lp))).collect() } else { Vec::new() };
            let pos = Pos { h: y.get(p * d..(p + 1) * d).map(<[f32]>::to_vec).unwrap_or_default(), t: tgt, tk, row: *i };
            match r.split {
                Split::Fit => fit.push(pos),
                Split::Tune => tune.push(pos),
                Split::Held => held.push(pos),
            }
        }
    }
    let _ = writeln!(
        out,
        "positions: fit {}, tune {}, held out {}; teacher tokens outside the serving vocabulary {outside}; pairs with teacher log-probabilities aligned {aligned}, not aligned (read as text) {misaligned}; features {:.1} s",
        fit.len(),
        tune.len(),
        held.len(),
        t.elapsed().as_secs_f64()
    );
    let (fit_r, tune_r, held_r): (Vec<&Pos>, Vec<&Pos>, Vec<&Pos>) = (fit.iter().collect(), tune.iter().collect(), held.iter().collect());
    // Neutral text.
    let (nt_ids, nh_ids) = neutral_ids(home, tok);
    let neutral = |chunks: &[Vec<i32>]| -> Result<(Vec<f32>, usize), Fault> {
        let mut hs = Vec::new();
        let mut n = 0;
        for c in chunks {
            let y = m.forward(dev, c, std::slice::from_ref(&fin), &RunOpts::default())?;
            hs.extend_from_slice(values(y.get(&fin).ok_or_else(|| bad("no final-norm state"))?)?);
            n += c.len();
        }
        Ok((hs, n))
    };
    let (nth, ntn) = neutral(&nt_ids)?;
    let (nhh, nhn) = neutral(&nh_ids)?;
    let (ntb, nhb) = (log_probs(&head, &nth, ntn)?, log_probs(&head, &nhh, nhn)?);
    let _ = writeln!(out, "neutral text (the Nano's own source as plain text): tune {} chunks ({ntn} positions), held out {} chunks ({nhn} positions)", nt_ids.len(), nh_ids.len());
    // Targets and the base on every split.
    let t = Instant::now();
    let (deltas, with_topk, moved) = targets(&head, &fit_r)?;
    let (base_tune, _, _) = score(&head, &tune_r, None)?;
    let (base_held, _, base_div) = score(&head, &held_r, None)?;
    let _ = writeln!(out, "targets: {} positions ({with_topk} from teacher log-probabilities, the rest a {} nat margin on the teacher's token), {moved} shifted; {:.1} s", deltas.len(), dl::MARGIN, t.elapsed().as_secs_f64());
    // A steer or a unit: maps the engine runs, measured by the engine with them in place.
    if technique == "steer" || technique == "unit" {
        let cx = Fwd {
            home,
            args,
            role,
            technique,
            set,
            m: &m,
            head: &head,
            node: &node,
            at: &at,
            seqs: &seqs,
            rows: &rows,
            fit_r: &fit_r,
            deltas: &deltas,
            base_tune,
            base_held,
            neutral: (&nt_ids, &nh_ids),
            neutral_base: (&ntb, &nhb),
            base: &base,
            values_key: &values_key,
            counts: (nf, nt, nh),
        };
        return if technique == "steer" { build_steer(&cx, out, &report, t0) } else { build_unit(&cx, out, &report, t0) };
    }
    // Rounds: one for `lens`; `teacher-loop` reweights the fit pairs the student follows worst.
    let mut weights: BTreeMap<usize, f64> = BTreeMap::new();
    let mut log = String::new();
    let mut chosen: Option<(Choice, Folds, BTreeMap<usize, f64>)> = None;
    for round in 0..rounds {
        let t = Instant::now();
        let folds = fold_sums(target, d, &fit_r, &deltas, &weights, false);
        let _ = writeln!(log, "round {}: sums over {} positions in {:.1} s", round + 1, positions(&folds), t.elapsed().as_secs_f64());
        let c = match &chosen {
            None => choose(&head, &folds, &tune_r, base_tune, &node, &Neutral { hs: &nth, n: ntn, base: &ntb }, &mut log)?,
            Some((prev, _, _)) => {
                let map = solve(&folds, None, prev.c, *RANKS.iter().max().unwrap_or(&32)).ok_or_else(|| bad("the refit did not solve"))?;
                let u = make("tune", &node, d, &map, prev.r, prev.scale)?;
                let (s, _, _) = score(&head, &tune_r, Some(&u))?;
                let kl = drift(&head, &nth, ntn, &ntb, &u)?;
                let _ = writeln!(log, "  refit at ridge {} rank {} scale {}: tune loss {:.4}, top-1 {}, neutral drift {kl:.4}", prev.c, prev.r, prev.scale, s.nll, pct(s.top1));
                Choice { c: prev.c, r: prev.r, scale: prev.scale, map, tune: s, kl }
            }
        };
        let better = chosen.as_ref().is_none_or(|(p, _, _)| c.tune.nll < p.tune.nll && c.kl <= GATE_KL);
        if !better {
            let _ = writeln!(log, "round {}: the tune split stopped gaining; the previous round stands", round + 1);
            break;
        }
        let _ = writeln!(log, "round {}: chosen ridge {} rank {} scale {}: tune loss {:.4} (base {:.4}), top-1 {} (base {}), neutral drift {:.4}", round + 1, c.c, c.r, c.scale, c.tune.nll, base_tune.nll, pct(c.tune.top1), pct(base_tune.top1), c.kl);
        // The next round's weights: the third of the fit pairs with the highest remaining loss.
        if round + 1 < rounds {
            let u = make("fit", &node, d, &c.map, c.r, c.scale)?;
            let (_, per, _) = score(&head, &fit_r, Some(&u))?;
            let mut worst: Vec<(f64, usize)> = per.iter().map(|(r, (l, n))| (l / (*n).max(1) as f64, *r)).collect();
            worst.sort_by(|a, b| b.0.total_cmp(&a.0));
            let top = worst.len().div_ceil(3);
            let mut next = weights.clone();
            for (_, r) in worst.iter().take(top) {
                let w = next.entry(*r).or_insert(1.0);
                *w = (*w * 1.5).min(4.0);
            }
            chosen = Some((c, folds, weights));
            weights = next;
        } else {
            chosen = Some((c, folds, weights.clone()));
        }
    }
    let (c, folds, weights) = chosen.ok_or_else(|| bad("no round produced a fit"))?;
    let _ = write!(out, "{log}");
    // The held-out gate.
    let key = format!("{}@next", dl::cartridge_id(role, set));
    let u = make(&key, &node, d, &c.map, c.r, c.scale)?;
    let (lens_held, _, lens_div) = score(&head, &held_r, Some(&u))?;
    let ctrl_folds = fold_sums(target, d, &fit_r, &deltas, &weights, true);
    let ctrl_map = solve(&ctrl_folds, None, c.c, *RANKS.iter().max().unwrap_or(&32)).ok_or_else(|| bad("the control did not solve"))?;
    let cu = make("control", &node, d, &ctrl_map, c.r, c.scale)?;
    let (ctrl_held, _, _) = score(&head, &held_r, Some(&cu))?;
    let kl_held = drift(&head, &nhh, nhn, &nhb, &u)?;
    let mut trial_gains = Vec::new();
    for f in 0..TRIALS {
        let map = solve(&folds, Some(f), c.c, *RANKS.iter().max().unwrap_or(&32)).ok_or_else(|| bad("a trial did not solve"))?;
        let tu = make("trial", &node, d, &map, c.r, c.scale)?;
        trial_gains.push(base_held.nll - score(&head, &held_r, Some(&tu))?.0.nll);
    }
    let gain = base_held.nll - lens_held.nll;
    let ctrl_gain = base_held.nll - ctrl_held.nll;
    // The note check: held-out prompts answered by the base and by the distilled Writer.
    let t = Instant::now();
    // The note check is the Writer's (its notes are what the embedder finds); the Reader's held-out
    // gate is the teacher-token figures, the control, the drift and the trials.
    let note_rows: Vec<&DistilRow> = rows.iter().filter(|r| role == "writer" && r.split == Split::Held && r.template == "chat").take(NOTE_ROWS).collect();
    let mut answers: Vec<Vec<String>> = vec![Vec::new(), Vec::new()];
    let rows_arc = Arc::clone(&head.rows);
    let short = node.rsplit('#').next().unwrap_or(&node).to_owned();
    for (arm, outs) in answers.iter_mut().enumerate() {
        match (&u, arm) {
            (Fitted::Lens(x), 1) => m.set_unit(&node, Some(Arc::new(x.clone())))?,
            (Fitted::Channel { gain, bias }, 1) => {
                let p = nmengn::filter::build(&m.pop.card.id, &[nmengn::filter::CartridgeFilter { cartridge: key.clone(), parts: vec![nmengn::filter::Part { target: short.clone(), gain: gain.clone(), bias: bias.clone() }] }]).map_err(bad)?;
                m.set_filters(Some(Arc::new(p)));
            }
            _ => {}
        }
        for r in &note_rows {
            let ids = tok.encode(&dl::chat_prompt(&r.prompt)).map_err(|e| bad(e.to_string()))?;
            let a = answer(&m, &rows_arc, &ids)?;
            outs.push(tok.decode(&a).trim().to_owned());
        }
    }
    m.set_unit(&node, None)?;
    m.set_filters(None);
    let samples: Vec<(String, String, String)> = note_rows.iter().zip(answers[0].iter().zip(&answers[1])).take(3).map(|(r, (a, b))| (r.completion.clone(), a.clone(), b.clone())).collect();
    let card_sha = m.pop.card.sha_main.clone();
    let model_id = m.pop.card.id.clone();
    drop(head);
    drop(m);
    let prompts: Vec<String> = note_rows.iter().map(|r| r.prompt.clone()).collect();
    let landed = if prompts.len() > nmapi::writer_idle::CHECK_NEIGHBOURS { note_check(home, &prompts, &answers)? } else { vec![0, 0] };
    let note_secs = t.elapsed().as_secs_f64();
    // The verdict.
    let mut checks = vec![
        ("teacher-token loss falls", gain > 0.0, format!("{:.4} to {:.4} ({:+.2}%)", base_held.nll, lens_held.nll, -100.0 * gain / base_held.nll.max(1e-12))),
        ("top-1 agreement rises", lens_held.top1 > base_held.top1, format!("{} to {}", pct(base_held.top1), pct(lens_held.top1))),
        ("shuffled control gains nothing", ctrl_gain <= GATE_CONTROL * gain.max(0.0), format!("control loss {:.4}, gain {ctrl_gain:+.4} against {gain:+.4} (at most {:.0}% of it)", ctrl_held.nll, GATE_CONTROL * 100.0)),
        ("neutral drift within the bound", kl_held <= GATE_KL, format!("{kl_held:.4} nats per token (bound {GATE_KL})")),
        ("note check not worse", landed[1] >= landed[0], format!("{} of {} landed, base {}", landed[1], prompts.len(), landed[0])),
        ("stable over repeated trials", trial_gains.iter().all(|g| *g > 0.0), format!("held-out gains {}", trial_gains.iter().map(|g| format!("{g:+.4}")).collect::<Vec<_>>().join(", "))),
    ];
    if role != "writer" {
        checks.retain(|c| c.0 != "note check not worse");
    }
    let passed = checks.iter().all(|c| c.1);
    let _ = writeln!(out, "held-out gate ({} positions of {nh} pairs):", lens_held.n);
    for (name, ok, fig) in &checks {
        let _ = writeln!(out, "  {} {name}: {fig}", if *ok { "PASS" } else { "FAIL" });
    }
    let _ = writeln!(out, "  first divergence (share of each pair's positions before its first top choice that is not the teacher's): base {}, distilled {}", pct(agreeing_prefix(&base_div)), pct(agreeing_prefix(&lens_div)));
    let _ = writeln!(out, "  note check {note_secs:.0} s; samples (teacher | base | distilled):");
    for (tch, b, l) in &samples {
        let _ = writeln!(out, "    {} | {} | {}", tch.replace('\n', " "), b.replace('\n', " "), l.replace('\n', " "));
    }
    let measured = format!(
        "held out: teacher-token loss {:.4} to {:.4}, top-1 {} to {}, control gain {ctrl_gain:+.4}, neutral drift {kl_held:.4}, note check {} against {} of {}",
        base_held.nll,
        lens_held.nll,
        pct(base_held.top1),
        pct(lens_held.top1),
        landed[1],
        landed[0],
        prompts.len()
    );
    let mut written = None;
    if passed {
        let id = dl::cartridge_id(role, set);
        let teachers: Vec<String> = rows.iter().map(|r| r.teacher.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let licences: Vec<String> = rows.iter().map(|r| r.licence.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let refs: Vec<String> = rows.iter().map(|r| r.reference.clone()).filter(|r| !r.is_empty()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let gate = obj(checks.iter().map(|(n, ok, fig)| (*n, obj(vec![("pass", Value::Bool(*ok)), ("figures", sv(fig))]))).collect());
        let lineage = nmapi::lineage::Record {
            made_on: base.clone(),
            made_by: format!("neuromorph-nano lab run distil build {role} --target {target} --set {set} --technique {technique}"),
            made_ms: nmtype::now_ms(),
            purpose: if target == "filter" { "job-filter".into() } else { "distil".into() },
            used_by: Vec::new(),
            recorded: "tool".into(),
            note: format!("fitted from {} teacher pairs ({})", nf, teachers.join(", ")),
            outcome: "used".into(),
            outcome_why: format!("passed its held-out gate: {measured}"),
        };
        let strs = |v: &[String]| Value::Array(v.iter().map(|x| sv(x)).collect());
        if let FitMap::Chan(..) = &c.map {
            let (file, key) = write_filter(home, set, &model_id, &base, &u, &short, obj(vec![
                ("technique", sv(technique)),
                ("set", sv(set)),
                ("values", sv(&values_key)),
                ("weights_sha256", sv(&card_sha)),
                ("scale", nv(c.scale)),
                ("ridge", nv(c.c)),
                ("pairs", obj(vec![("fit", Value::Number(nf.to_string())), ("tune", Value::Number(nt.to_string())), ("held", Value::Number(nh.to_string()))])),
                ("teachers", strs(&teachers)),
                ("licences", strs(&licences)),
                ("references", strs(&refs)),
                ("gate", gate.clone()),
                ("measured", sv(&measured)),
                ("effect", sv("style, format and wording preferences of the teacher; not new knowledge or reasoning")),
            ]), &lineage)?;
            mark_fitted(home, &refs, &key);
            let _ = writeln!(out, "WRITTEN: {} ({key}, a job filter at {short}, scale {}; serves where the {}'s job filters are on and it is plugged in)", file.display(), c.scale, if role == "reader" { "Reader" } else { "Writer" });
            written = Some(key);
        }
        let (dn, up) = match &c.map {
            FitMap::Low(a, b) => dl::leading(a, b, d, c.r),
            FitMap::Chan(..) => (Vec::new(), Vec::new()),
        };
        let r = dn.len() / d.max(1);
        let notes = obj(vec![
            ("values", sv(&values_key)),
            ("base", sv(&base)),
            ("weights_sha256", sv(&card_sha)),
            ("role", sv(role)),
            ("socket", obj(vec![("node", sv(&node)), ("role", sv(dl::SOCKET)), ("layers", sv("after the last layer (final norm)")), ("form", sv("y + scale * up(down(y))")), ("rank", Value::Number(r.to_string())), ("down", sv(&format!("[{r}, {d}]"))), ("up", sv(&format!("[{d}, {r}]")))])),
            ("scale", nv(c.scale)),
            ("ridge", nv(c.c)),
            ("technique", sv(technique)),
            ("set", sv(set)),
            ("pairs", obj(vec![("fit", Value::Number(nf.to_string())), ("tune", Value::Number(nt.to_string())), ("held", Value::Number(nh.to_string()))])),
            ("teachers", strs(&teachers)),
            ("licences", strs(&licences)),
            ("references", strs(&refs)),
            ("gate", gate),
            ("measured", sv(&measured)),
            ("acts_on", sv(if role == "reader" { "the Reader's final norm, every position" } else { "the Writer's final norm, every position" })),
            ("effect", sv("style, format and wording preferences of the teacher; not new knowledge or reasoning")),
            ("project", sv(&home.display().to_string())),
            ("lineage", lineage.to_value()),
        ]);
        let label = nmapi::adapter_fit::LabelIn {
            id: &id,
            version: 0,
            model: &model_id,
            weights: &card_sha,
            kind: dl::KIND,
            does: &dl::does(role, dl::KIND),
            made_from: &format!("{nf} curated teacher pairs of set {set} ({}), on {base}", teachers.join(", ")),
            notes,
        };
        if r > 0 {
            let parts: Vec<nmapi::adapter::Part> = vec![("down".to_owned(), vec![r, d], dn), ("up".to_owned(), vec![d, r], up)];
            let first = nmapi::adapter::versions(home, &id).is_empty();
            let (file, version) = nmapi::adapter_fit::write_next(home, label, &parts)?;
            if first {
                // A new distilled cartridge starts unplugged: the user plugs it in on the Writer's deck.
                let _ = nmapi::plugs::change(home, |s| {
                    s.units_off.insert(nmapi::plugs::key(nmapi::plugs::NANO, &id));
                });
            }
            mark_fitted(home, &refs, &format!("{id}@{version}"));
            let _ = writeln!(out, "WRITTEN: {} ({id}@{version}, rank {r}, scale {}{})", file.display(), c.scale, if first { ", unplugged until you plug it in" } else { "" });
            written = Some(format!("{id}@{version}"));
        }
    } else {
        let failed: Vec<&str> = checks.iter().filter(|c| !c.1).map(|c| c.0).collect();
        let _ = writeln!(out, "REFUSED: the held-out gate did not pass ({}); nothing was written. {measured}", failed.join("; "));
    }
    let _ = nmapi::audit::audit(
        home,
        "distil-build",
        vec![
            ("role", sv(role)),
            ("technique", sv(technique)),
            ("set", sv(set)),
            ("passed", Value::Bool(passed)),
            ("cartridge", written.as_deref().map_or(Value::Null, sv)),
            ("measured", sv(&measured)),
            ("report", sv(&report.display().to_string())),
            ("label", sv(&format!("Distillation into the {role} ({technique}, set {set}): {}", if passed { "passed" } else { "refused" }))),
        ],
    );
    let _ = writeln!(out, "elapsed: {:.0} s; report {}", t0.elapsed().as_secs_f64(), report.display());
    if let Some(dir) = report.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&report, crate::with_overlay(&out));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(h: Vec<f32>, row: usize) -> Pos {
        Pos { h, t: 0, tk: Vec::new(), row }
    }

    #[test]
    fn a_readout_ridge_recovers_a_linear_residual_and_its_control_does_not() {
        // Residuals that are a fixed linear function of the states: the ridge read-out recovers
        // it; the same fit on residuals taken from other pairs does not.
        let d = 4;
        let truth = [0.5, -1.0, 0.25, 2.0];
        let ps: Vec<Pos> = (0..60_usize)
            .map(|i| {
                let f = i as f32;
                pos(vec![(f * 0.37).sin(), (f * 0.91).cos(), (f * 0.13).sin() * 2.0, 1.0], i)
            })
            .collect();
        let refs: Vec<&Pos> = ps.iter().collect();
        let resid: Vec<f64> = ps.iter().map(|p| p.h.iter().zip(&truth).map(|(x, w)| f64::from(*x) * w).sum()).collect();
        let w = solve_readout(&read_folds(d, &refs, &resid, &BTreeMap::new(), false), None, 1e-6).unwrap();
        assert!(w.iter().zip(&truth).all(|(a, b)| (a - b).abs() < 1e-3), "{w:?}");
        let err = |w: &[f64]| ps.iter().zip(&resid).map(|(p, e)| (p.h.iter().zip(w).map(|(x, w)| f64::from(*x) * w).sum::<f64>() - e).powi(2)).sum::<f64>();
        let ctrl = solve_readout(&read_folds(d, &refs, &resid, &BTreeMap::new(), true), None, 1e-6).unwrap();
        assert!(err(&ctrl) > 10.0 * err(&w) + 1.0, "the shuffled control fits as well");
        // A third left out still recovers it.
        let trial = solve_readout(&read_folds(d, &refs, &resid, &BTreeMap::new(), false), Some(1), 1e-6).unwrap();
        assert!(trial.iter().zip(&truth).all(|(a, b)| (a - b).abs() < 1e-3));
    }

    #[test]
    fn a_steer_vector_is_the_weighted_mean_shift_and_leaves_its_third_out() {
        // Two thirds, one channel: a = 1 everywhere, shifts 1 and 3: the mean is 2; leaving the
        // second third out gives the first's.
        let num = vec![vec![1.0], vec![3.0], vec![0.0]];
        let den = vec![vec![1.0], vec![1.0], vec![0.0]];
        assert!((f64::from(steer_vec(&num, &den, None)[0]) - 2.0).abs() < 1e-3);
        assert!((f64::from(steer_vec(&num, &den, Some(1))[0]) - 1.0).abs() < 1e-3);
        assert_eq!(steer_vec(&[vec![1.0]], &[vec![0.0]], None), vec![0.0], "a channel with no gain does not move");
    }

    #[test]
    fn the_embedder_scores_and_its_request_lens_drift() {
        let qs = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
        let pairs = [
            EPair { q: 0, doc: vec![1.0, 0.0], p: 1.0, target: 0.9, row: 0 },
            EPair { q: 0, doc: vec![0.0, 1.0], p: 0.0, target: 0.1, row: 1 },
            EPair { q: 1, doc: vec![0.0, 1.0], p: 1.0, target: 0.9, row: 2 },
        ];
        let refs: Vec<&EPair> = pairs.iter().collect();
        let s = escore(&qs, &refs, None);
        assert_eq!(s.n, 3);
        assert!((s.order - 1.0).abs() < 1e-12, "the one same-query pair is ordered as the teacher orders it");
        assert!((s.mse - (0.01 + 0.01 + 0.01) / 3.0).abs() < 1e-9);
        // A lens that maps nothing does not drift; one that turns requests does.
        let still = embed_lens("k", "v", 2, &(vec![0.0; 2 * 64], vec![0.0; 2 * 64]), 1, 1.0);
        assert!(embed_drift(&qs, &still) < 1e-6);
        let turn = embed_lens("k", "v", 2, &(vec![1.0, 0.0], vec![0.0, 1.0]), 1, 1.0);
        assert!(embed_drift(&qs, &turn) > 0.1);
        assert!(turn.flat.is_empty() && turn.beta.is_empty(), "a request lens never flattens the stored side");
    }

    #[test]
    fn judge_scores_and_order_agreement() {
        let (s, per) = jscore(&[2.0, -2.0, 0.5], &[0.9, 0.1, 0.5]);
        assert_eq!(s.n, 3);
        assert!((s.agree - 1.0).abs() < 1e-12, "a score of one half is not counted");
        assert!(per.iter().all(|x| *x >= 0.0));
        let (worse, _) = jscore(&[-2.0, 2.0, 0.5], &[0.9, 0.1, 0.5]);
        assert!(worse.kl > s.kl && worse.agree < 0.5);
        assert!((order_agreement(&[1.0, 0.0, 5.0], &[0.8, 0.2, 0.1], &[0, 0, 1]) - 1.0).abs() < 1e-12);
        assert!(order_agreement(&[0.0, 1.0], &[0.8, 0.2], &[0, 0]) < 1e-12);
    }

    #[test]
    fn a_judge_build_refuses_too_few_scored_pairs_before_any_model_opens() {
        let home = std::env::temp_dir().join(format!("nm-distil-judge-{}-{}", std::process::id(), nmtype::now_ms()));
        std::fs::create_dir_all(home.join("memory")).unwrap();
        let args: Vec<String> = ["run", "distil", "build", "reranker", "--target", "readout", "--technique", "readout"].iter().map(|s| (*s).to_owned()).collect();
        let e = build(&home, &args).unwrap_err();
        assert!(e.message.contains("too few scored pairs"), "{e}");
        let e = build(&home, &["run", "distil", "build", "writer", "--target", "readout", "--technique", "readout"].iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()).unwrap_err();
        assert!(e.message.contains("only the reranker gives one score"), "{e}");
        let e = build(&home, &["run", "distil", "build", "embedder", "--technique", "lens"].iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()).unwrap_err();
        assert!(e.message.contains("too few scored pairs"), "{e}");
        let e = build(&home, &["run", "distil", "build", "embedder", "--target", "filter", "--technique", "channel"].iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()).unwrap_err();
        assert!(e.message.contains("stored piece vectors"), "{e}");
        let e = build(&home, &["run", "distil", "build", "writer", "--technique", "unit"].iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()).unwrap_err();
        assert!(e.message.contains("learning unit serves on the shared model"), "{e}");
        let _ = std::fs::remove_dir_all(&home);
    }
}
