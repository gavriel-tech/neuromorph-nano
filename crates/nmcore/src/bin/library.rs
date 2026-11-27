//! The cartridge library from the program's side: the `library` command, the stacks put in use
//! at the start of `serve` and `bench`, the audit rows of every switch, the interface's view, and
//! the reading side of the evidence ledger. The library itself (cartridges as learned parameters
//! used as filters, stacks for a use, hash checks, export and import) is `nmlearn::library`;
//! hardware identity is `nmlearn::hardware`.
//!
//! Settings filters reach the served configuration as `serve` options: the values the stacks in
//! use give for the operating keys below become the matching options when those options are not
//! given on the command line. A key without an option (an engine constant, or a size derived at
//! run time) must equal what the code uses, or it is refused. A trial filter's values are recorded
//! with what they would change and never applied. `--rack off` leaves the library out.
//!
//! Cartridges for a MicroModel plug into its sockets (`nmengn::filter`): at start, the served
//! cartridges of the stacks in use are plugged in the order of the uses (preference uses last) and
//! of each stack. A cartridge that does not fit (another MicroModel, other bare weights, another
//! socket contract, a socket the model does not have) is refused by name. A serving process watches the
//! catalogue and, when a stack is switched, stages the new rack, which the engine applies where the
//! next call starts: no restart. `--try-cartridge <id>[,<id>...]` plugs named cartridges for one run
//! even on trial (a measurement, recorded as such). Only a serving process (the MCP server) writes
//! its start to the audit and `memory/stacks/in-use.json`; bench and one-shot commands write nothing.
//!
//! First content. An empty library gets one stack for the graphics card this was built on, the
//! NVIDIA GeForce GTX 1650 with 4 GB: one settings cartridge holding the operating numbers the
//! speed passes chose on that card, with those passes' measurements as its evidence. Its values
//! equal the code's defaults, so using it changes nothing on that card.
use nmjson::Value;
use nmlearn::hardware::{self, Kind, Machine, Piece};
use nmlearn::library::{CartridgeSpec, Choice, EvidenceRow, EvidenceSource, Filter, Library, Point, Stack, Strength};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn strs(v: &[String]) -> Value {
    Value::Array(v.iter().map(|x| s(x)).collect())
}

// ---------------------------------------------------------------------------------------------
// Operating keys: the code's values, and how a value reaches `serve`.

#[derive(Clone, Copy)]
enum Form {
    /// `<flag> <value>`.
    Flag(&'static str),
    /// Only `off` has an option (`rerank_batch off` is `--rerank-single`).
    OffFlag(&'static str),
    /// A model's precision: `default` is no option; a width is set before the models bind.
    Precision(&'static str),
    /// No option: must equal the code's value.
    Fixed,
}

const OPS: [(&str, Form, &str); 15] = [
    ("precision.base", Form::Precision("base"), "base MicroModel precision (default: 16 bits)"),
    ("precision.reranker", Form::Precision("reranker"), "reranker precision (default: 16 bits)"),
    ("precision.embedder", Form::Precision("embedder"), "embedder precision (default: 16 bits)"),
    ("half_math", Form::Flag("--half-math"), "half2 products in the 16-bit linear layers"),
    ("segment_attention", Form::Flag("--segment-attention"), "the packed-sequence attention kernel"),
    ("judge_window", Form::Flag("--judge-window"), "tokens of each piece the judge reads in full mode"),
    ("fast_judge_window", Form::Flag("--fast-judge-window"), "tokens of each piece the judge reads in fast mode"),
    ("look_window", Form::Flag("--look-window"), "the second look reads the judge's windows"),
    ("look_gate", Form::Flag("--look-gate"), "the second look runs when the top two are closer than this many log-odds"),
    ("fast_k", Form::Flag("--fast-k"), "pieces of the stack's order the reranker reads in fast mode"),
    ("rerank_batch", Form::OffFlag("--rerank-single"), "reranker documents packed side by side in one forward"),
    ("batch_tokens", Form::Fixed, "tokens per reranker batch"),
    ("dense_batch", Form::Fixed, "chunks per batched embedder call"),
    ("dense_add", Form::Fixed, "dense lane pieces added to the reranker's candidates in full mode"),
    ("fast_dense", Form::Fixed, "dense lane pieces added in fast mode"),
];

fn on_off(b: bool) -> String {
    if b { "on" } else { "off" }.to_owned()
}
fn window(w: Option<usize>) -> String {
    w.map_or_else(|| "off".to_owned(), |x| x.to_string())
}

/// The value the code uses today for an operating key when nothing is given.
#[must_use]
pub fn code_value(key: &str) -> Option<String> {
    Some(match key {
        "precision.base" | "precision.reranker" | "precision.embedder" => "default".to_owned(),
        // `serve` switches half math and the segment kernel on unless told off (`configure`).
        "half_math" | "segment_attention" => "on".to_owned(),
        "judge_window" => window(nmapi::whole::JUDGE_WINDOW),
        "fast_judge_window" => window(nmapi::whole::FAST_JUDGE_WINDOW),
        "look_window" => on_off(nmapi::whole::LOOK_WINDOW),
        "look_gate" => format!("{}", nmapi::whole::LOOK_GATE),
        "fast_k" => nmapi::whole::FAST_K.to_string(),
        "rerank_batch" => on_off(nmengn::roles::RERANK_BATCH.load(std::sync::atomic::Ordering::Relaxed)),
        "batch_tokens" => format!("derived from free card memory, fallback {}", nmengn::roles::BATCH_TOKENS_FALLBACK),
        "dense_batch" => nmapi::neural::DENSE_BATCH.to_string(),
        "dense_add" => nmapi::whole::DENSE_ADD.to_string(),
        "fast_dense" => nmapi::whole::FAST_DENSE.to_string(),
        _ => return None,
    })
}

/// The code's values for every operating key: the base the settings filters apply to.
#[must_use]
pub fn code_base() -> BTreeMap<String, String> {
    OPS.iter().map(|(k, _, _)| ((*k).to_owned(), code_value(k).unwrap_or_default())).collect()
}

fn given(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// What the served settings become: options added, precision widths, keys equal to the code
/// with no option, keys left to the command line, and keys refused.
#[derive(Default)]
pub struct Applied {
    pub args: Vec<String>,
    pub precision: Vec<(String, String)>,
    pub equal: Vec<String>,
    pub kept: Vec<String>,
    pub refused: Vec<String>,
}

/// Translates settings set by filters into `serve` options; options on the command line win.
#[must_use]
pub fn translate(set: &BTreeMap<String, String>, args: &[String]) -> Applied {
    let mut a = Applied::default();
    for (key, value) in set {
        match OPS.iter().find(|o| o.0 == key).map(|o| o.1) {
            None => a.refused.push(format!("{key}: not an operating key")),
            Some(Form::Flag(flag)) if given(args, flag) => a.kept.push(format!("{key}: {flag} given")),
            Some(Form::Flag(flag)) => a.args.extend([flag.to_owned(), value.clone()]),
            Some(Form::OffFlag(flag)) => match value.as_str() {
                "on" => a.equal.push(key.clone()),
                "off" if given(args, flag) => a.kept.push(format!("{key}: {flag} given")),
                "off" => a.args.push(flag.to_owned()),
                v => a.refused.push(format!("{key}: {v} is not on or off")),
            },
            Some(Form::Precision(_)) if value == "default" => a.equal.push(key.clone()),
            Some(Form::Precision(_)) if given(args, "--precision") => a.kept.push(format!("{key}: --precision given")),
            Some(Form::Precision(part)) if ["4", "8", "16"].contains(&value.as_str()) => a.precision.push((part.to_owned(), value.clone())),
            Some(Form::Precision(_)) => a.refused.push(format!("{key}: {value} is not default, 4, 8 or 16")),
            Some(Form::Fixed) if code_value(key).as_deref() == Some(value.as_str()) => a.equal.push(key.clone()),
            Some(Form::Fixed) => a.refused.push(format!("{key}: {value} differs from the code's {} and has no option", code_value(key).unwrap_or_default())),
        }
    }
    a
}

// ---------------------------------------------------------------------------------------------
// First content.

/// The graphics card the first content was measured on.
#[must_use]
pub fn gtx_1650() -> Piece {
    hardware::gpu_piece("NVIDIA GeForce GTX 1650", 4096, "581.83", "7.5")
}

/// The speed passes' measurements on the GTX 1650 (times from the report names).
fn gtx_evidence(subject: &str) -> Vec<EvidenceRow> {
    let hw = gtx_1650().identity();
    let row = |strength: Strength, at: u64, measure: &str, source: &str| EvidenceRow {
        subject: subject.to_owned(),
        hardware: hw.clone(),
        outcome: "verified".into(),
        strength: Some(strength),
        measure: measure.to_owned(),
        source: source.to_owned(),
        at,
    };
    use Strength::{Direct, Indirect};
    vec![
        row(Direct, 1_794_723_193_418, "rerank_batch on, batch_tokens derived: batched 9.22 s against one at a time 9.80 s on the same documents, scores bit-identical, order identical on 40 of 40, card peak 3,297 of 4,096 MiB", "runs/bench-full-1794723193418.txt"),
        row(Direct, 1_794_724_674_657, "look_gate 3.0: the look ran on 37 of 40; total 16.36 to 14.01 s; right first 27 to 28, top 3 and in pack unchanged", "runs/bench-full-1794724674657.txt"),
        row(Direct, 1_794_733_410_021, "precision defaults (base 8, reranker 16, embedder 16): base at 16 and 4 bits slower in the second look, base 4 lost two first places; reranker 8 equal counts, left for a later decision", "runs/bench-full-1794733410021.txt"),
        row(Indirect, 1_794_738_266_905, "fast_k 4, dense_add 4, fast_dense 2, dense_batch 16: full 17.06 s at 26, 37, 38 and fast 4.30 s at 26, 33, 33 of 40, dense lane on", "runs/bench-full-fast-1794738266905.txt"),
        row(Direct, 1_794_745_519_190, "half_math on with segment_attention on, against the reference: full 15.22 to 11.46 s, fast 3.07 to 1.94 s, right first, top 3 and in pack equal in both modes, first places identical on 80 of 80", "runs/bench-full-fast-1794745519190.txt"),
        row(Direct, 1_794_746_117_332, "judge_window 384 against whole pieces: full 12.30 to 10.66 s, counts 24, 36, 36 to 27, 38, 38; 256 and 192 fail (pack lost d29, d33)", "runs/bench-full-fast-1794746117332.txt"),
        row(Direct, 1_794_747_451_628, "fast_judge_window off: the 384 window in fast mode gave top 3 28 against 29 (d19), so fast mode reads whole pieces", "runs/bench-full-fast-1794747451628.txt"),
        row(Direct, 1_794_747_838_295, "look_window on: joint read 5.35 to 4.48 s, total 10.15 to 9.32 s, right first 27 to 28, top 3 and in pack 38 both", "runs/bench-full-1794747838295.txt"),
        row(Indirect, 1_794_750_122_305, "all together, dense lane on: full 18.47 to 10.85 s, fast 4.70 to 2.91 s; full 26, 36, 37 to 29, 38, 38, fast 26, 32, 32 both; card peak 3,981 to 3,449 MiB", "runs/bench-full-fast-1794750122305.txt"),
        row(Direct, 1_794_849_199_582, "precision.base 8 against the default 16, dense lane off, back to back: right first, top 3 and in pack equal (full 29, 37, 37; fast 24, 30, 30 of 40); full 5.68 against 5.37 s and fast 1.59 against 1.51 s after the first question; card peak 2,649 against 2,837 MiB. Kept at 8 on this card so the embedder fits beside the base and the judge", "runs/bench-full-fast-1794849199582.txt"),
        row(Direct, 1_794_973_821_408, "base, embedder and judge at 8 bits in one served process (whole path, dense lane from 8-bit stored vectors) against base 8 with embedder and judge 16: weights on the card 2,001 to 1,212 MiB, card peak 3,241 to 2,581 MiB; large code tuning 20 questions right first, top 3, in pack 12, 13, 14 both; 7.16 to 7.02 s a question", "runs/bench-full-1794973821408.txt"),
        row(Direct, 1_794_973_384_611, "the same on the large documents tuning file, 19 questions: right first 13 to 12, top 3 and in pack 14 and 15 both; 8.70 to 6.15 s a question", "runs/bench-full-1794973384611.txt"),
        row(Indirect, 1_794_971_429_576, "served plain path with the meaning lane, embedder 8 against 16, the four large tuning files paired: right first 93 to 91, top 3 145 to 143, in pack 169 both of 201 (no change significant, p 0.69 and 0.63); seconds a little lower", "runs/bench-board-1794971429576.txt"),
    ]
}

/// Id of the stack for a hardware piece.
#[must_use]
pub fn stack_id(p: &Piece) -> String {
    match p.kind {
        Kind::Gpu => format!("gpu-{}-{}-{}mib", p.vendor, p.model, p.memory_mib),
        Kind::Ram => format!("ram-{}gib", hardware::gib(p.memory_mib)),
        _ => hardware::slug(&p.identity().replace(':', "-")),
    }
}

/// The version of the GTX 1650 settings cartridge the code writes: 3 selects the three finding and
/// ranking MicroModels (the base, the embedder and the judge) at 8 bits on this card, where the
/// built-in default is 16, so the three stay on the card together.
const GTX_VERSION: u64 = 3;

/// The values the GTX 1650 settings cartridge holds: the code's, with the base MicroModel, the
/// embedder and the judge at 8 bits.
fn gtx_values() -> BTreeMap<String, String> {
    let mut m = code_base();
    for part in ["base", "embedder", "reranker"] {
        m.insert(format!("precision.{part}"), "8".into());
    }
    m
}

/// The GTX 1650 settings cartridge's label.
fn gtx_spec() -> CartridgeSpec {
    CartridgeSpec {
        id: "gtx-1650-4gb".into(),
        version: GTX_VERSION,
        family: "fundamentals".into(),
        domain: "hardware:gpu".into(),
        scope: format!("hardware:{}", gtx_1650().identity()),
        model: "settings".into(),
        weights: String::new(),
        layers: None,
        name: "GTX 1650 operating numbers".into(),
        description: "Sets how the Nano runs its MicroModels on a GeForce GTX 1650 with 4 GB: precision per model, batch sizes, the attention kernel, how much of each piece the judge reads and when the second look runs. It is for running fast on this card without losing right answers.".into(),
        made_from: "Nine paired bench runs of the speed passes on one frozen copy of the Nano's code, on this card with driver 581 (each run is listed in its evidence); no model weights are changed.".into(),
        effect: "Selects the base MicroModel, the embedder and the judge at 8 bits on this card (the program's default is 16), so the three stay on the card together (weights 2,001 to 1,212 MiB, card peak 3,241 to 2,581 MiB); every other value equals the program's own default. The speed passes behind it took the full call from 18.5 to 10.9 s with right answers equal or better; the three at 8 bits kept the served counts within one question pooled and the seconds equal or lower.".into(),
    }
}

/// The first content of an empty library: the GTX 1650 stack and its settings cartridge.
/// # Errors
/// A write failure.
pub fn create_first(lib: &mut Library) -> Result<Vec<String>, String> {
    let card = gtx_1650();
    let subject = format!("gtx-1650-4gb@{GTX_VERSION}");
    let made_c = lib.add_cartridge(&gtx_spec(), &Filter::Settings(gtx_values()), Value::Null, gtx_evidence(&subject), false).map_err(|e| e.to_string())?;
    let made_s = lib
        .add_stack(Stack {
            id: stack_id(&card),
            version: GTX_VERSION,
            use_for: "hardware:gpu".into(),
            piece: Some(card),
            cartridges: vec![made_c.clone()],
            description: "What was learned about this graphics card. Measured with driver 581.83 (the runs did not record the driver, so the range is the one in use when the stack was made).".into(),
            hash: String::new(),
            trial: false,
        })
        .map_err(|e| e.to_string())?;
    Ok(vec![made_c, made_s])
}

// ---------------------------------------------------------------------------------------------
// Opening, locking, audit.

/// One writer of the library at a time: a lock file made with `create_new`, taken over after a
/// minute (left by a process that stopped).
pub(crate) struct Lock(PathBuf);
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn lock(dir: &Path) -> Result<Lock, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let p = dir.join("cartridges").join("catalogue.lock");
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let t = std::time::Instant::now();
    loop {
        if std::fs::OpenOptions::new().write(true).create_new(true).open(&p).is_ok() {
            return Ok(Lock(p));
        }
        let stale = std::fs::metadata(&p).and_then(|m| m.modified()).ok().and_then(|m| m.elapsed().ok()).is_some_and(|e| e.as_secs() > 60);
        if stale {
            let _ = std::fs::remove_file(&p);
            continue;
        }
        if t.elapsed().as_secs() >= 5 {
            return Err(format!("{} is held by another program", p.display()));
        }
        std::thread::yield_now();
    }
}

/// Opens the library, making the first content when there is no catalogue yet (audited).
/// # Errors
/// An unreadable catalogue or a write failure.
pub fn open(home: &Path) -> Result<Library, String> {
    let lib = Library::open(home).map_err(|e| e.to_string())?;
    if lib.exists() {
        return relabel_first(home, lib).and_then(|l| first_to_current(home, l));
    }
    let _g = lock(&lib.dir)?;
    let mut lib = Library::open(home).map_err(|e| e.to_string())?;
    if lib.exists() {
        return Ok(lib);
    }
    let made = create_first(&mut lib)?;
    lib.save().map_err(|e| e.to_string())?;
    crate::controls::audit(home, "library", vec![("action", s("create")), ("made", strs(&made)), ("label", s("Cartridge library made with its first stack"))])?;
    Ok(lib)
}

/// The first content written before cartridges carried their label in their own file is written
/// again as one labelled file, same settings and evidence (the earlier folder stays where it is).
fn relabel_first(home: &Path, lib: Library) -> Result<Library, String> {
    let earlier = |l: &Library| l.cartridge("op-gtx-1650-4gb@1").filter(|c| !c.one_file()).cloned();
    if earlier(&lib).is_none() {
        return Ok(lib);
    }
    let _g = lock(&lib.dir)?;
    let mut lib = Library::open(home).map_err(|e| e.to_string())?;
    let Some(c) = earlier(&lib) else { return Ok(lib) };
    let filter = lib.earlier_filter(&c).map_err(|e| e.to_string())?;
    lib.unlist(&c.key());
    // A file written by an earlier attempt that stopped before the shelf was saved is listed as it is.
    let name = format!("{}-v{}.{}", c.id, c.version, nmlearn::library::CARTRIDGE_EXT);
    let path = lib.dir.join("cartridges").join(&name);
    match std::fs::read(&path) {
        Ok(b) => {
            let (label, _) = nmlearn::library::read_cartridge(&b).map_err(|e| e.to_string())?;
            let rec = nmlearn::library::FileRec { name, sha256: nmhash::sha256_bytes(&b), bytes: b.len() as u64 };
            lib.cartridges.push(nmlearn::library::Cartridge::from_label(&label, rec, c.trial).map_err(|e| e.to_string())?);
        }
        Err(_) => {
            let spec = CartridgeSpec { version: c.version, effect: "Every value equals the program's own default at the time, so on this card it changed nothing.".into(), ..gtx_spec() };
            lib.add_cartridge(&spec, &filter, Value::Null, c.evidence.clone(), c.trial).map_err(|e| e.to_string())?;
        }
    }
    // The stacks holding it are hashed again over its new file.
    let hashes: Vec<(usize, String)> = lib
        .stacks
        .iter()
        .enumerate()
        .filter(|(_, st)| st.cartridges.contains(&c.key()))
        .map(|(i, st)| (i, nmlearn::library::stack_hash(st, &st.cartridges.iter().filter_map(|k| lib.cartridge(k)).collect::<Vec<_>>())))
        .collect();
    for (i, h) in hashes {
        if let Some(st) = lib.stacks.get_mut(i) {
            st.hash = h;
        }
    }
    lib.save().map_err(|e| e.to_string())?;
    crate::controls::audit(home, "library", vec![("action", s("relabel")), ("cartridge", s(&c.key())), ("label", s("Cartridge written again as one labelled file"))])?;
    Ok(lib)
}

/// A library whose GTX 1650 stack is older than the code's first content gets the current settings
/// cartridge and stack beside the earlier ones (audited); the newest stack for the piece serves.
fn first_to_current(home: &Path, lib: Library) -> Result<Library, String> {
    let id = stack_id(&gtx_1650());
    let behind = |l: &Library| l.stacks.iter().any(|st| st.id == id) && !l.stacks.iter().any(|st| st.id == id && st.version >= GTX_VERSION);
    if !behind(&lib) {
        return Ok(lib);
    }
    let _g = lock(&lib.dir)?;
    let mut lib = Library::open(home).map_err(|e| e.to_string())?;
    if !behind(&lib) {
        return Ok(lib);
    }
    let made = create_first(&mut lib)?;
    lib.save().map_err(|e| e.to_string())?;
    crate::controls::audit(home, "library", vec![("action", s("first-content")), ("made", strs(&made)), ("label", s("GTX 1650 settings: the base MicroModel, the embedder and the judge selected at 8 bits on this card"))])?;
    Ok(lib)
}

/// The library opened for a change, with its lock held while the guard lives.
/// # Errors
/// An unreadable catalogue, or the lock held by another program.
pub(crate) fn for_change(home: &Path) -> Result<(Library, Lock), String> {
    drop(open(home)?);
    let g = lock(&home.join("library"))?;
    Ok((Library::open(home).map_err(|e| e.to_string())?, g))
}

/// Switches the stack in use for `use_for` (`None`: the use off). A running server plugs the new
/// stack's socket cartridges where its next call starts; settings apply at the next start of a
/// server. Returns the audit id, or none when nothing changed.
/// # Errors
/// Unknown stack, a stack for another use, or a write failure.
pub fn switch_use(home: &Path, use_for: &str, stack: Option<&str>, by: &str) -> Result<Option<String>, String> {
    let (mut lib, _g) = for_change(home)?;
    let Some(before) = lib.switch_use(use_for, stack).map_err(|e| e.to_string())? else { return Ok(None) };
    let to = lib.active.get(use_for).cloned().unwrap_or_default();
    lib.save().map_err(|e| e.to_string())?;
    crate::controls::audit(
        home,
        "library",
        vec![
            ("action", s("use")),
            ("use", s(use_for)),
            ("from", before.as_deref().map_or(Value::Null, s)),
            ("to", s(&to)),
            ("by", s(by)),
            ("label", s(&format!("Stack for {use_for}: {to}"))),
        ],
    )
    .map(Some)
}

/// Puts a stack on trial, or serves it. Serving needs the evidence for its piece: at least five
/// verified rows measured on exactly that piece, twice as many as refused.
/// # Errors
/// Unknown stack, the evidence short (with the counts), or a write failure.
pub fn set_trial(home: &Path, name: &str, trial: bool, ledger: &dyn EvidenceSource, by: &str) -> Result<Option<String>, String> {
    let (mut lib, _g) = for_change(home)?;
    let st = lib.stack(name).cloned().ok_or_else(|| format!("no stack {name}"))?;
    if !trial {
        let rows = stack_rows(&lib, &st, ledger);
        let target = st.piece.as_ref().map_or_else(|| machine_tag(&this_machine(home, false)), Piece::identity);
        if !nmlearn::library::ready(&rows, &target) {
            let (v, r) = nmlearn::library::counts(&rows, &target);
            return Err(format!(
                "{} stays on trial: {v} verified and {r} refuted rows measured on {target} (rows from other hardware never count); serving needs {} verified and twice as many as refuted",
                st.key(),
                nmlearn::library::READY_ROWS
            ));
        }
    }
    if !lib.set_trial(name, trial).map_err(|e| e.to_string())? {
        return Ok(None);
    }
    lib.save().map_err(|e| e.to_string())?;
    let (from, to) = if trial { ("served", "trial") } else { ("trial", "served") };
    crate::controls::audit(
        home,
        "library",
        vec![("action", s("trial")), ("stack", s(&st.key())), ("from", s(from)), ("to", s(to)), ("by", s(by)), ("label", s(&format!("Stack {} {to}", st.key())))],
    )
    .map(Some)
}

/// Undoes a library switch by its audit id (the switch back is itself audited).
/// # Errors
/// Not a library switch, or the switch back fails.
pub fn undo(home: &Path, id: &str, by: &str) -> Result<Option<String>, String> {
    let all = std::fs::read_to_string(home.join("memory").join("audit.jsonl")).map_err(|e| format!("audit log: {e}"))?;
    let row = all.lines().filter(|l| l.contains(id)).filter_map(|l| nmjson::parse(l.as_bytes()).ok()).find(|v| text(v, "id") == id).ok_or_else(|| format!("no audit row {id}"))?;
    match text(&row, "action").as_str() {
        "use" => {
            let from = text(&row, "from");
            let back = if from.is_empty() || from == "off" { None } else { Some(from.as_str()) };
            if from.is_empty() {
                // The use had no switch before: take it out of the switched uses.
                let (mut lib, _g) = for_change(home)?;
                lib.active.remove(&text(&row, "use"));
                lib.save().map_err(|e| e.to_string())?;
                return crate::controls::audit(home, "library", vec![("action", s("use")), ("use", s(&text(&row, "use"))), ("to", Value::Null), ("undo_of", s(id)), ("by", s(by))]).map(Some);
            }
            switch_use(home, &text(&row, "use"), back, by)
        }
        "trial" => {
            let (mut lib, _g) = for_change(home)?;
            if !lib.set_trial(&text(&row, "stack"), text(&row, "from") == "trial").map_err(|e| e.to_string())? {
                return Ok(None);
            }
            lib.save().map_err(|e| e.to_string())?;
            crate::controls::audit(home, "library", vec![("action", s("trial")), ("stack", s(&text(&row, "stack"))), ("from", s(&text(&row, "to"))), ("to", s(&text(&row, "from"))), ("undo_of", s(id)), ("by", s(by))]).map(Some)
        }
        // 2026-11-26 (Phase B): values set by hand, a stack made or changed by hand, an import.
        "import" => undo_import(home, &row, id, by),
        "values" => {
            let (mut lib, _g) = for_change(home)?;
            let back = switch_back(&mut lib, row.get("switched").and_then(Value::array).unwrap_or_default())?;
            lib.save().map_err(|e| e.to_string())?;
            let label = format!("Undone: values of {} (back to {})", text(&row, "new"), text(&row, "cartridge"));
            crate::controls::audit(home, "library", vec![("action", s("values-undone")), ("cartridge", s(&text(&row, "new"))), ("switched", Value::Array(back)), ("undo_of", s(id)), ("by", s(by)), ("label", s(&label))]).map(Some)
        }
        "stack" => {
            let (mut lib, _g) = for_change(home)?;
            let key = text(&row, "stack");
            let use_for = text(&row, "use");
            if matches!(row.get("switched"), Ok(Value::Bool(true))) && lib.active.get(&use_for).is_some_and(|k| *k == key) {
                match row.get("from").and_then(Value::string) {
                    Ok(from) => {
                        lib.active.insert(use_for.clone(), from.to_owned());
                    }
                    Err(_) => {
                        lib.active.remove(&use_for);
                    }
                }
            }
            // The stack made goes when nothing uses it; one a use was switched to since is kept.
            let in_use = lib.active.values().any(|k| *k == key);
            if !in_use {
                lib.stacks.retain(|st| st.key() != key);
            }
            lib.save().map_err(|e| e.to_string())?;
            let label = format!("Undone: stack {key}{}", if in_use { " (kept: a use runs on it)" } else { " taken out of the catalogue" });
            crate::controls::audit(home, "library", vec![("action", s("stack-undone")), ("stack", s(&key)), ("use", s(&use_for)), ("removed", Value::Bool(!in_use)), ("undo_of", s(id)), ("by", s(by)), ("label", s(&label))]).map(Some)
        }
        // Stated preference cartridges: a write, an edit or a removal (`preferences`).
        "preference-say" | "preference-edit" | "preference-remove" => crate::preferences::undo(home, &row, id, by),
        a => Err(format!("library {a} is not a switch; nothing to undo")),
    }
}

/// `POST /api/cartridges/import {path}`: loads a stack file exported by `library export` (a stack
/// and its cartridges, each file hash-checked, the whole body hash-checked) from a path on this
/// machine into the library through the library's own import (`Library::import`, as `library
/// import` does): the stack and every new cartridge arrive on trial. A file holding a base
/// cartridge is refused (base cartridges come from the downloads and are never written here).
/// Audited as a `library` `import` row; undo takes the stack and the cartridges it added out again.
/// # Errors
/// A network or missing path, a file that is not a stack file or fails a hash, a base cartridge, a
/// stack or cartridge already in the library with other content, or a write failure.
pub fn import_api(home: &Path, path: &str, by: &str) -> Result<Value, String> {
    let src = crate::pmaps::local_file(path, 512 * 1024 * 1024)?;
    let bytes = std::fs::read(&src).map_err(|e| format!("{}: {e}", src.display()))?;
    // The refusal of base cartridges comes before anything is written.
    let v = nmjson::parse(&bytes).map_err(|e| format!("not a stack file: {e}"))?;
    let body = v.get("body").map_err(|_| "not a stack file (library export): no body".to_owned())?;
    for item in body.get("cartridges").or_else(|_| body.get("books")).and_then(Value::array).unwrap_or_default() {
        let c = item.get("cartridge").or_else(|_| item.get("book")).unwrap_or(&Value::Null);
        if text(c, "family") == "base" || text(c, "kind") == "tensors" {
            return Err(format!("{}: a base cartridge; base cartridges come from the downloads, not from an import", text(c, "id")));
        }
    }
    let (mut lib, _g) = for_change(home)?;
    let base_dir = lib.dir.join("cartridges").join("base");
    let (key, added) = lib.import(&bytes).map_err(|e| e.to_string())?;
    if let Some(c) = added.iter().filter_map(|k| lib.cartridge(k)).find(|c| c.family == "base" || lib.cartridge_dir(c).starts_with(&base_dir)) {
        // Defensive: a label that turned out to be a base cartridge; its files are taken out again.
        let c = c.clone();
        for f in &c.files {
            let _ = std::fs::remove_file(lib.cartridge_dir(&c).join(&f.name));
        }
        return Err(format!("{}: a base cartridge; nothing was imported", c.key()));
    }
    lib.save().map_err(|e| e.to_string())?;
    let audit = crate::controls::audit(
        home,
        "library",
        vec![
            ("action", s("import")),
            ("stack", s(&key)),
            ("added", strs(&added)),
            ("source", s(&src.display().to_string())),
            ("hash", s(&nmhash::sha256_bytes(&bytes))),
            ("by", s(by)),
            ("label", s(&format!("Stack {key} imported on trial"))),
        ],
    )?;
    Ok(obj(vec![
        ("audit_id", s(&audit)),
        ("stack", s(&key)),
        ("added", strs(&added)),
        ("trial", Value::Bool(true)),
        ("note", s("the stack and its new cartridges are on trial: switch a use to the stack to try it, and serve it once the evidence allows (trial)")),
    ]))
}

/// Undoes a stack import: the stack goes when no use runs it, and each cartridge it added goes
/// (its files deleted) when no other stack holds it.
fn undo_import(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let (mut lib, _g) = for_change(home)?;
    let key = text(row, "stack");
    if lib.active.values().any(|k| *k == key) {
        return Err(format!("a use runs {key}; switch it off first, then undo the import"));
    }
    lib.stacks.retain(|st| st.key() != key);
    let base_dir = lib.dir.join("cartridges").join("base");
    let mut removed = Vec::new();
    for k in row.get("added").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok()) {
        if lib.stacks.iter().any(|st| st.cartridges.iter().any(|c| c == k)) {
            continue;
        }
        let Some(c) = lib.cartridge(k).cloned() else { continue };
        let dir = lib.cartridge_dir(&c);
        if c.family == "base" || dir.starts_with(&base_dir) {
            continue;
        }
        lib.unlist(k);
        for f in &c.files {
            let _ = std::fs::remove_file(dir.join(&f.name));
        }
        removed.push(k.to_owned());
    }
    lib.save().map_err(|e| e.to_string())?;
    crate::controls::audit(home, "library", vec![("action", s("import-undone")), ("stack", s(&key)), ("removed", strs(&removed)), ("undo_of", s(id)), ("by", s(by)), ("label", s(&format!("Undone: stack {key} import")))]).map(Some)
}

/// Switches each use of a `values` row's `switched` list back to the stack it ran before, where it
/// still runs the stack the change made. Returns what was switched back.
fn switch_back(lib: &mut Library, switched: &[Value]) -> Result<Vec<Value>, String> {
    let mut back = Vec::new();
    for sw in switched {
        let (use_for, from, to) = (text(sw, "use"), text(sw, "from"), text(sw, "to"));
        if lib.active.get(&use_for).is_some_and(|k| *k == to) && lib.stack(&from).is_some() {
            lib.switch_use(&use_for, Some(&from)).map_err(|e| e.to_string())?;
            back.push(obj(vec![("use", s(&use_for)), ("from", s(&to)), ("to", s(&from))]));
        }
    }
    Ok(back)
}

// ---------------------------------------------------------------------------------------------
// Inspecting and editing cartridges and stacks (2026-11-26, Phase B item A3).

/// Whether a value key of a preference cartridge reads (`pref.global`, `pref.key.<label>`).
fn pref_key(k: &str) -> bool {
    k == "pref.global" || k.strip_prefix("pref.key.").is_some_and(|l| !l.is_empty() && l.len() <= 128 && l.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ':')))
}

/// A preference value: `<good>/<signals>`, whole numbers with good at most signals.
fn pref_value(v: &str) -> bool {
    v.split_once('/').and_then(|(a, b)| Some((a.trim().parse::<u64>().ok()?, b.trim().parse::<u64>().ok()?))).is_some_and(|(a, b)| a <= b)
}

/// `GET /api/cartridges/inspect?key=<id@version or id>`: one cartridge's label and record (family,
/// kind, model, layers, files with hashes and sizes, evidence, notes, trial), the stacks holding it
/// and the uses they run, and its contents as far as they are settings: a settings or preference
/// cartridge's key and value entries (each operating key with the value the code uses without it);
/// a socket cartridge's points (target and width, numbers left out); a base cartridge's parameter
/// sets are described by its label only. A registered learning unit is shown by its card. Every
/// answer carries `parameters`, the parameters section (`cparams`, format `nm-parameters/1`):
/// tensors and points with shapes, norms and summaries, or that the cartridge holds none; a job
/// filter (`filter:<role>/<job>`) and an adapter library version (`<id>@<version>`) are read too.
#[must_use]
pub fn inspect(home: &Path, query: &str) -> Value {
    let Some(key) = crate::project::query_arg(query, "key").filter(|k| !k.is_empty()) else {
        return obj(vec![("error", s("key: give a cartridge's id@version (or id) as /api/cartridges lists it"))]);
    };
    if let Some(v) = crate::distil::inspect(home, &key) {
        return v;
    }
    let lib = match Library::open(home) {
        Ok(l) => l,
        Err(e) => return obj(vec![("error", s(&e.to_string()))]),
    };
    let Some(c) = lib.cartridge(&key).cloned() else {
        let units = nmlearn::Units::open(home);
        // Job filters and the adapter library's versions are read through the parameters section
        // (`cparams`): they hold parameters, not values.
        if let Ok(h) = crate::cparams::held(home, &key)
            && h.source != "library"
            && h.source != "unit"
        {
            return obj(vec![
                ("key", s(&h.key)),
                ("source", s(h.source)),
                ("values", Value::Null),
                ("points", Value::Null),
                ("values_editable", Value::Bool(false)),
                ("note", s(&format!("{}; its parameters are edited under parameters (a new version)", h.note))),
                ("parameters", crate::cparams::section_of(home, &h)),
            ]);
        }
        return match units.get(&key) {
            Some(card) => obj(vec![
                ("key", s(&key)),
                ("source", s("unit")),
                ("card", card.to_value()),
                ("values", Value::Null),
                ("values_editable", Value::Bool(false)),
                ("note", s("a learning unit holds low-rank weights; it is plugged in and out (Units), it has no values to edit; its parameters are shown and edited under parameters")),
                ("parameters", crate::cparams::section(home, &key)),
            ]),
            None if !lib.exists() => obj(vec![("error", s("the cartridge library is not made yet: it is made at the first start of a server, or by the first stack change or `neuromorph-nano library list`"))]),
            None => obj(vec![("error", s(&format!("no cartridge {key} in the cartridge library or the unit registry")))]),
        };
    };
    let stacks: Vec<Value> = lib
        .stacks
        .iter()
        .filter(|st| st.cartridges.contains(&c.key()))
        .map(|st| {
            let uses: Vec<String> = lib.active.iter().filter(|(_, k)| **k == st.key()).map(|(u, _)| u.clone()).collect();
            obj(vec![("key", s(&st.key())), ("use", s(&st.use_for)), ("trial", Value::Bool(st.trial)), ("active_for", strs(&uses)), ("cartridges", strs(&st.cartridges))])
        })
        .collect();
    let (values, points, note) = match c.kind.as_str() {
        "settings" => match lib.filter(&c) {
            Ok(Filter::Settings(m)) => {
                let rows: Vec<Value> = m
                    .iter()
                    .map(|(k, v)| obj(vec![("key", s(k)), ("value", s(v)), ("code_value", code_value(k).map_or(Value::Null, |x| s(&x))), ("kind", s(if k.starts_with("pref.") { "preference" } else { "operating" }))]))
                    .collect();
                (Value::Array(rows), Value::Null, "values set here make a new version of the cartridge (its file is never written over) and switch every use that ran it to a new version of its stack; undo switches them back")
            }
            Ok(_) => (Value::Null, Value::Null, "the label says settings but the file holds another kind of filter"),
            Err(_) => (Value::Null, Value::Null, "the cartridge's file is missing or did not pass its hash check; its values are not shown"),
        },
        "affine" => match lib.filter(&c) {
            Ok(f) => (
                Value::Null,
                Value::Array(f.points().iter().map(|p| obj(vec![("target", s(&p.target)), ("gain", n(p.gain.len() as u64)), ("bias", n(p.bias.len() as u64))])).collect()),
                "a socket cartridge holds learned gains and biases per socket; their counts are shown, not edited here",
            ),
            Err(_) => (Value::Null, Value::Null, "the cartridge's file did not pass its hash check"),
        },
        _ => (Value::Null, Value::Null, "a base cartridge holds a MicroModel's parameter sets; it has no values to edit (choose or remove versions under Base versions)"),
    };
    obj(vec![
        ("key", s(&c.key())),
        ("source", s("library")),
        ("cartridge", c.to_value()),
        ("files_present", Value::Bool(lib.files_present(&c))),
        ("stacks", Value::Array(stacks)),
        ("values", values),
        ("points", points),
        ("values_editable", Value::Bool(c.kind == "settings")),
        ("note", s(note)),
        ("parameters", crate::cparams::section(home, &c.key())),
    ])
}

/// The next free version of a cartridge id or stack id.
pub(crate) fn next_version(lib: &Library, id: &str, stack: bool) -> u64 {
    let top = if stack { lib.stacks.iter().filter(|x| x.id == id).map(|x| x.version).max() } else { lib.cartridges.iter().filter(|x| x.id == id).map(|x| x.version).max() };
    top.unwrap_or(0) + 1
}

/// `POST /api/cartridges/values {key, values: {name: value or null}}`: sets values of a settings or
/// preference cartridge. A cartridge file is never written over, so the edited values become a new
/// version of the cartridge; every stack holding the old version gets a new version holding the
/// new one, and every use that ran such a stack is switched to its new version. A preference value
/// is `<good>/<signals>` (`pref.global`, `pref.key.<label>`; null takes a `pref.key.*` out); an
/// operating value keeps to the operating keys. Audited as a `library` `values` row; undo switches
/// the uses back to the stacks they ran (the new versions stay in the catalogue).
/// # Errors
/// An unknown cartridge, one that holds no values, a value that does not read, or a write failure.
pub fn set_values(home: &Path, key: &str, changes: &BTreeMap<String, Value>, by: &str) -> Result<Value, String> {
    if let Some(r) = crate::distil::set_values(home, key, changes, by) {
        return r;
    }
    let (mut lib, _g) = for_change(home)?;
    let c = lib.cartridge(key).cloned().ok_or_else(|| format!("no cartridge {key}"))?;
    if c.kind != "settings" {
        return Err(format!("{} is a {} cartridge; only settings and preference cartridges hold values", c.key(), c.kind));
    }
    let Filter::Settings(old) = lib.filter(&c).map_err(|e| e.to_string())? else { return Err(format!("{}: not a settings filter", c.key())) };
    let pref = c.family == nmlearn::library::PREFERENCE;
    if let Some(k) = changes.keys().find(|k| k.starts_with(nmlearn::stated::KEY_PREFIX)) {
        return Err(format!("{k}: a stated rule is changed through preference_edit (nm_learning), which parses your words and writes a new version"));
    }
    let mut new = old.clone();
    for (k, v) in changes {
        match v {
            Value::Null => {
                if !(pref && k.starts_with("pref.key.")) {
                    return Err(format!("{k}: only a preference key (pref.key.*) can be taken out"));
                }
                new.remove(k);
            }
            Value::String(x) => {
                let x = x.trim();
                if pref {
                    if !pref_key(k) || !pref_value(x) {
                        return Err(format!("{k} = {x}: a preference value is <good>/<signals> (whole numbers, good at most signals) under pref.global or pref.key.<label>"));
                    }
                } else if !old.contains_key(k) && code_value(k).is_none() {
                    return Err(format!("{k}: not an operating key of a settings cartridge"));
                } else if x.is_empty() || x.len() > 64 || x.chars().any(char::is_control) {
                    return Err(format!("{k}: give a plain value of at most 64 characters"));
                }
                new.insert(k.clone(), x.to_owned());
            }
            _ => return Err(format!("{k}: give the value as text, or null to take a preference key out")),
        }
    }
    if new == old {
        return Ok(obj(vec![("changed", Value::Bool(false)), ("key", s(&c.key())), ("audit_id", Value::Null)]));
    }
    let version = next_version(&lib, &c.id, false);
    let spec = CartridgeSpec {
        id: c.id.clone(),
        version,
        family: c.family.clone(),
        domain: c.domain.clone(),
        scope: c.scope.clone(),
        model: c.model.clone(),
        weights: c.weights.clone(),
        layers: c.layers,
        name: c.name.clone(),
        description: c.description.clone(),
        made_from: format!("the values of {} with {} set by hand ({by})", c.key(), changes.keys().cloned().collect::<Vec<_>>().join(", ")),
        effect: "values set by hand; not measured".to_owned(),
    };
    let notes = obj(vec![("edited_from", s(&c.key())), ("edited_by", s(by)), ("edited_keys", strs(&changes.keys().cloned().collect::<Vec<_>>()))]);
    let new_key = lib.add_cartridge(&spec, &Filter::Settings(new), notes, Vec::new(), c.trial).map_err(|e| e.to_string())?;
    let mut stacks = Vec::new();
    let mut switched = Vec::new();
    let holding: Vec<Stack> = lib.stacks.iter().filter(|st| st.cartridges.contains(&c.key())).cloned().collect();
    for st in holding {
        let next = Stack {
            version: next_version(&lib, &st.id, true),
            cartridges: st.cartridges.iter().map(|k| if *k == c.key() { new_key.clone() } else { k.clone() }).collect(),
            hash: String::new(),
            ..st.clone()
        };
        let nk = lib.add_stack(next).map_err(|e| e.to_string())?;
        stacks.push(obj(vec![("from", s(&st.key())), ("to", s(&nk))]));
        let uses: Vec<String> = lib.active.iter().filter(|(_, k)| **k == st.key()).map(|(u, _)| u.clone()).collect();
        for u in uses {
            lib.switch_use(&u, Some(&nk)).map_err(|e| e.to_string())?;
            switched.push(obj(vec![("use", s(&u)), ("from", s(&st.key())), ("to", s(&nk))]));
        }
    }
    lib.save().map_err(|e| e.to_string())?;
    let label = format!("Values of {} set by hand: {new_key}", c.key());
    let audit = crate::controls::audit(
        home,
        "library",
        vec![
            ("action", s("values")),
            ("cartridge", s(&c.key())),
            ("new", s(&new_key)),
            ("keys", strs(&changes.keys().cloned().collect::<Vec<_>>())),
            ("stacks", Value::Array(stacks.clone())),
            ("switched", Value::Array(switched.clone())),
            ("by", s(by)),
            ("label", s(&label)),
        ],
    )?;
    Ok(obj(vec![
        ("changed", Value::Bool(true)),
        ("audit_id", s(&audit)),
        ("from", s(&c.key())),
        ("key", s(&new_key)),
        ("stacks", Value::Array(stacks)),
        ("switched", Value::Array(switched)),
        ("applies", s("operating values at the next start of a server; preference values when a serving process reads the switched stack")),
    ]))
}

/// `POST /api/cartridges/trial {name, trial}`: a stack on trial or served (`set_trial`, which
/// needs the evidence to serve); audited as a `library` `trial` row, undone by the Audit page.
/// # Errors
/// As `set_trial`.
pub fn trial_api(home: &Path, name: &str, trial: bool, by: &str) -> Result<Option<String>, String> {
    set_trial(home, name, trial, &LedgerEvidence::read(home), by)
}

/// `POST /api/cartridges/stack {id, use, cartridges: [key, ...], description?, switch?}`: makes a
/// stack, or a new version of one, holding the cartridges in the order given (through the
/// library's own checks: every cartridge in the library, preference cartridges only in preference
/// stacks, the stacking order base, hardware, adapter, learned, preference). A new version keeps
/// the earlier version's piece, description and trial flag; a new stack starts on trial (serving
/// needs evidence, `trial`). With `switch` (default: when the use runs an earlier version of this
/// stack) the use is switched to it. Audited as a `library` `stack` row; undo switches the use back
/// and takes the stack out when nothing runs it.
/// # Errors
/// A bad id, an unknown cartridge, a stack id already used for another use, a cartridge order the
/// stacking order refuses, or a write failure.
#[expect(clippy::too_many_arguments, reason = "a stack names its id, use, cartridges, description, switch and who asked")]
pub fn stack_save(home: &Path, id: &str, use_for: &str, cartridges: &[String], description: Option<&str>, switch: Option<bool>, by: &str) -> Result<Value, String> {
    if use_for.trim().is_empty() || use_for.len() > 128 || use_for.chars().any(char::is_control) {
        return Err("use: name the use (general, hardware:<piece>, project:<key>, job:<name>, preference or preference:<scope>)".into());
    }
    if cartridges.is_empty() {
        return Err("cartridges: give at least one cartridge key".into());
    }
    let (mut lib, _g) = for_change(home)?;
    let prev = lib.stack(id).cloned();
    if let Some(p) = &prev
        && p.use_for != use_for
    {
        return Err(format!("stack {id} is for {}; a stack keeps its use (make another id)", p.use_for));
    }
    let st = Stack {
        id: id.to_owned(),
        version: next_version(&lib, id, true),
        use_for: use_for.to_owned(),
        piece: prev.as_ref().and_then(|p| p.piece.clone()),
        cartridges: cartridges.to_vec(),
        description: description.map(str::to_owned).or_else(|| prev.as_ref().map(|p| p.description.clone())).unwrap_or_else(|| format!("made by hand ({by})")),
        hash: String::new(),
        trial: prev.as_ref().is_none_or(|p| p.trial),
    };
    let key = lib.add_stack(st).map_err(|e| e.to_string())?;
    let made = lib.stack(&key).cloned().ok_or("the stack made is not in the catalogue")?;
    lib.validate_stack_hierarchy(&made)?;
    let before = lib.active.get(use_for).cloned();
    let runs_earlier = before.as_deref().and_then(|k| lib.stack(k)).is_some_and(|s0| s0.id == id);
    let switched = if switch.unwrap_or(runs_earlier) { lib.switch_use(use_for, Some(&key)).map_err(|e| e.to_string())?.is_some() } else { false };
    lib.save().map_err(|e| e.to_string())?;
    let label = format!("Stack {key} for {use_for}{}", if switched { ", switched on" } else { "" });
    let audit = crate::controls::audit(
        home,
        "library",
        vec![
            ("action", s("stack")),
            ("stack", s(&key)),
            ("use", s(use_for)),
            ("cartridges", strs(&made.cartridges)),
            ("from", before.as_deref().map_or(Value::Null, s)),
            ("switched", Value::Bool(switched)),
            ("by", s(by)),
            ("label", s(&label)),
        ],
    )?;
    Ok(obj(vec![
        ("audit_id", s(&audit)),
        ("stack", made.to_value()),
        ("switched", Value::Bool(switched)),
        ("previous", before.map_or(Value::Null, |b| s(&b))),
        ("applies", s("a running server plugs a switched stack's socket cartridges where its next call starts; settings apply at the next start of a server")),
    ]))
}

// ---------------------------------------------------------------------------------------------
// Evidence: recorded rows and the evidence ledger.

/// The library's reading side of the evidence ledger (`crate::ledger`, read through its own
/// `read_ledger`). A ledger row is about a context pack, not a cartridge: the pack is joined to the
/// stacks its server put in use at start (the `library-stacks` audit row of the same process, the
/// latest at or before the pack), and each cartridge in use then gets the pack's rows, tagged with
/// the identity of the machine piece it served (other uses: the whole machine). The ledger's own
/// machine tag must agree with that row's graphics card (name and memory), or the rows are left out.
pub struct LedgerEvidence {
    pub rows: Vec<EvidenceRow>,
    pub skipped: BTreeMap<String, u64>,
}

struct StartRow {
    pid: String,
    at: u64,
    loaded: Vec<(String, String)>,
    gpu: Option<(String, u64)>,
}

impl LedgerEvidence {
    /// Reads the audit log and every project's ledger under `home`.
    #[must_use]
    pub fn read(home: &Path) -> Self {
        let audit = std::fs::read_to_string(home.join("memory").join("audit.jsonl")).unwrap_or_default();
        let mut starts = Vec::new();
        let mut packs: BTreeMap<String, (String, u64)> = BTreeMap::new();
        for l in audit.lines() {
            if l.contains("\"kind\":\"library-stacks\"") {
                let Ok(v) = nmjson::parse(l.as_bytes()) else { continue };
                let loaded = v.get("loaded").and_then(Value::array).unwrap_or_default().iter().map(|x| (text(x, "key"), text(x, "hardware"))).collect();
                let gpu = v.get("gpu").ok().filter(|g| !matches!(g, Value::Null)).map(|g| (text(g, "name"), g.get("memory_mib").and_then(Value::integer).unwrap_or(0)));
                starts.push(StartRow { pid: text(&v, "pid"), at: v.get("time_ms").and_then(Value::integer).unwrap_or(0), loaded, gpu });
            } else if l.contains("\"tool\":\"nm_context_pack\"") && l.contains("\"kind\":\"tool\"") {
                let Ok(v) = nmjson::parse(l.as_bytes()) else { continue };
                let pid = text(&v, "session").split('-').next().unwrap_or("").to_owned();
                packs.insert(text(&v, "id"), (pid, v.get("time_ms").and_then(Value::integer).unwrap_or(0)));
            }
        }
        let mut files = vec![("nano".to_owned(), home.join("memory").join(crate::ledger::FILE))];
        if let Ok(rd) = std::fs::read_dir(home.join("memory").join("projects")) {
            files.extend(rd.flatten().map(|e| (e.file_name().to_string_lossy().into_owned(), e.path().join(crate::ledger::FILE))));
        }
        Self::join(&starts, &packs, &files)
    }

    fn join(starts: &[StartRow], packs: &BTreeMap<String, (String, u64)>, files: &[(String, PathBuf)]) -> Self {
        let mut out = Self { rows: Vec::new(), skipped: BTreeMap::new() };
        for (project, file) in files {
            let Ok((rows, _, _)) = crate::ledger::read_ledger(file) else { continue };
            let served_gpu: BTreeMap<String, Option<(String, u64)>> = rows
                .iter()
                .filter(|r| text(r, "row") == "served")
                .map(|r| {
                    let hw = r.get("hardware").ok();
                    (text(r, "pack"), hw.and_then(|h| h.get("gpu").and_then(Value::string).ok().map(|g| (g.to_owned(), h.get("gpu_mib").and_then(Value::integer).unwrap_or(0)))))
                })
                .collect();
            for r in rows.iter().filter(|r| text(r, "row") == "evidence") {
                let pack = text(r, "pack");
                let why = match packs.get(&pack) {
                    None => Some("pack not in the audit log"),
                    Some((pid, at)) => match starts.iter().filter(|k| &k.pid == pid && k.at <= at.saturating_add(1000)).max_by_key(|k| k.at) {
                        None => Some("no stacks recorded for the pack's server"),
                        Some(st) => {
                            let agree = match (&st.gpu, served_gpu.get(&pack).cloned().flatten()) {
                                (None, None) => true,
                                (Some((a, am)), Some((b, bm))) => hardware::slug(a) == hardware::slug(&b) && *am == bm,
                                _ => false,
                            };
                            if agree {
                                out.rows.extend(st.loaded.iter().map(|(key, hw)| EvidenceRow {
                                    subject: key.clone(),
                                    hardware: hw.clone(),
                                    outcome: text(r, "state"),
                                    strength: r.get("strength").and_then(Value::string).ok().and_then(Strength::parse),
                                    measure: format!("ledger rule {}", text(r, "rule")),
                                    source: format!("ledger:{project}:{pack}"),
                                    at: *at,
                                }));
                                None
                            } else {
                                Some("ledger machine tag and the stacks' graphics card disagree")
                            }
                        }
                    },
                };
                if let Some(w) = why {
                    *out.skipped.entry(w.to_owned()).or_insert(0) += 1;
                }
            }
        }
        out
    }
}
impl EvidenceSource for LedgerEvidence {
    fn rows_for(&self, subject: &str) -> Vec<EvidenceRow> {
        self.rows.iter().filter(|r| r.subject == subject).cloned().collect()
    }
}

/// Every evidence row about a stack's cartridges: recorded and from the ledger.
fn stack_rows(lib: &Library, st: &Stack, ledger: &dyn EvidenceSource) -> Vec<EvidenceRow> {
    st.cartridges.iter().flat_map(|k| lib.cartridge(k).map(|c| c.evidence.clone()).unwrap_or_default().into_iter().chain(ledger.rows_for(k))).collect()
}

// ---------------------------------------------------------------------------------------------
// The stacks in use for this machine.

/// This machine (cached a day; the graphics card and processor read now).
#[must_use]
pub fn this_machine(home: &Path, full: bool) -> Machine {
    hardware::machine(home, full)
}

/// The whole machine as one tag (every piece's identity joined), for uses that are not one piece.
#[must_use]
pub fn machine_tag(m: &Machine) -> String {
    m.pieces.iter().map(Piece::identity).collect::<Vec<_>>().join("+")
}

/// A machine description from a file (as tests hand one in).
/// # Errors
/// Unreadable or not a machine.
pub fn machine_file(p: &Path) -> Result<Machine, String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut m = Machine::from_value(&nmjson::parse(&b).map_err(|e| format!("{}: {e}", p.display()))?)?;
    m.how = "given".into();
    Ok(m)
}

/// The stacks in use for a machine: each use's choice, the cartridges loaded (key, on trial, the
/// hardware it serves, refused with why), the settings the served filters set (key to value and
/// the cartridge that set it last), and what the trial filters would change.
pub struct InUse {
    pub machine: Machine,
    pub choices: Vec<Choice>,
    pub loaded: Vec<(String, bool, String, Option<String>)>,
    pub set: BTreeMap<String, (String, String)>,
    pub trial: Vec<(String, String, String)>,
    /// Affine cartridges for MicroModel sockets, served and on trial, in the order they were loaded.
    pub plugs: Vec<Plug>,
}

/// An affine cartridge for a MicroModel's sockets: its key, family, MicroModel, the rank of the use
/// that brought it, its place in that stack, its points, and whether it is on trial.
#[derive(Clone, Debug)]
pub struct Plug {
    pub key: String,
    pub family: String,
    pub model: String,
    pub use_rank: (u8, u8),
    pub at: usize,
    pub points: Vec<Point>,
    pub trial: bool,
}

/// Loads the chosen stacks and applies their settings filters over the code's values.
#[must_use]
pub fn in_use(lib: &Library, machine: Machine) -> InUse {
    let choices = lib.choose(&machine);
    let whole = machine_tag(&machine);
    let mut u = InUse { machine, choices: Vec::new(), loaded: Vec::new(), set: BTreeMap::new(), trial: Vec::new(), plugs: Vec::new() };
    let mut trial_set: BTreeMap<String, (String, String)> = BTreeMap::new();
    for c in &choices {
        let Some(key) = &c.stack else { continue };
        let Ok(l) = lib.load(key) else { continue };
        let hw = c.piece.clone().unwrap_or_else(|| whole.clone());
        for (k, why) in &l.refused {
            u.loaded.push((k.clone(), true, hw.clone(), Some(why.clone())));
        }
        for (at, (k, f, on_trial)) in l.filters.iter().enumerate() {
            let on_trial = *on_trial || c.trial;
            u.loaded.push((k.clone(), on_trial, hw.clone(), None));
            if let Some(cart) = lib.cartridge(k).filter(|_| f.kind() == "affine") {
                u.plugs.push(Plug {
                    key: k.clone(),
                    family: cart.family.clone(),
                    model: cart.model.clone(),
                    use_rank: nmlearn::library::use_rank(&c.use_for),
                    at,
                    points: f.points(),
                    trial: on_trial,
                });
            }
            if let Filter::Settings(m) = f {
                for (sk, sv) in m {
                    if on_trial {
                        trial_set.insert(sk.clone(), (sv.clone(), k.clone()));
                    } else {
                        u.set.insert(sk.clone(), (sv.clone(), k.clone()));
                    }
                }
            }
        }
    }
    u.trial = trial_set.into_iter().filter(|(k, (v, _))| u.set.get(k).map(|x| &x.0) != Some(v) && code_value(k).as_ref() != Some(v) || u.set.get(k).is_some_and(|x| &x.0 != v)).map(|(k, (v, c))| (k, v, c)).collect();
    u.choices = choices;
    u
}

impl InUse {
    /// The operating values the served settings filters set (a preference cartridge's `pref.*`
    /// values are read by the serve-time preference reader instead, `pref_values`).
    fn values(&self) -> BTreeMap<String, String> {
        self.set.iter().filter(|(k, _)| !k.starts_with("pref.")).map(|(k, (v, _))| (k.clone(), v.clone())).collect()
    }
    /// The served preference cartridge's values (roadmap Phase 2 item 2, 2026-11-24), with each
    /// source cartridge's parameter overlay applied: a learned count
    /// `<good>/<signals>` is edited or masked as its tensor `pref.<label>` (`[good, signals]`) is,
    /// rounded to whole counts with good at most signals (masked: `0/0`, no evidence); a cartridge
    /// masked whole gives none of its values. Values that are not counts (stated rules) pass as they are.
    #[must_use]
    pub fn pref_values_overlaid(&self, home: &Path) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        for (k, (v, c)) in self.set.iter().filter(|(k, _)| k.starts_with("pref.")) {
            let ov = nmengn::overlay::effective(home, c);
            if ov.as_ref().is_some_and(nmengn::overlay::Overlay::whole_masked) {
                continue;
            }
            let counts = v.split_once('/').and_then(|(a, b)| Some((a.trim().parse::<f32>().ok()?, b.trim().parse::<f32>().ok()?)));
            let (Some(ov), Some((g, n))) = (ov, counts) else {
                out.insert(k.clone(), v.clone());
                continue;
            };
            let mut x = [g, n];
            ov.apply(k, "F32", &mut x);
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "whole non-negative counts, rounded and clamped")]
            let (g, n) = (x[0].max(0.0).round() as u64, x[1].max(0.0).round() as u64);
            out.insert(k.clone(), format!("{}/{n}", g.min(n)));
        }
        out
    }
    #[must_use]
    pub fn to_value(&self, applied: &Applied) -> Value {
        obj(vec![
            ("machine", self.machine.to_value()),
            (
                "uses",
                Value::Array(
                    self.choices
                        .iter()
                        .map(|c| obj(vec![("use", s(&c.use_for)), ("piece", c.piece.as_deref().map_or(Value::Null, s)), ("stack", c.stack.as_deref().map_or(Value::Null, s)), ("why", s(&c.why)), ("trial", Value::Bool(c.trial))]))
                        .collect(),
                ),
            ),
            (
                "loaded",
                Value::Array(
                    self.loaded
                        .iter()
                        .map(|(k, t, hw, r)| obj(vec![("key", s(k)), ("trial", Value::Bool(*t)), ("hardware", s(hw)), ("refused", r.as_deref().map_or(Value::Null, s))]))
                        .collect(),
                ),
            ),
            ("set", Value::Object(self.set.iter().map(|(k, (v, c))| (k.clone(), obj(vec![("value", s(v)), ("from", s(c))]))).collect())),
            ("trial", Value::Array(self.trial.iter().map(|(k, v, c)| obj(vec![("key", s(k)), ("value", s(v)), ("from", s(c))])).collect())),
            ("options", strs(&applied.args)),
            ("refused", strs(&applied.refused)),
        ])
    }
}

/// Set by `serve` before it configures: this process serves, so its start is recorded (audit row
/// `library-stacks`, `memory/stacks/in-use.json`) and it watches the catalogue for stack switches. Bench
/// and other runs leave it off and write nothing.
pub static SERVING: AtomicBool = AtomicBool::new(false);

/// The cartridges `--try-cartridge` names (comma separated, repeatable).
fn tried(args: &[String]) -> Vec<String> {
    args.windows(2)
        .filter(|w| w.first().map(String::as_str) == Some("--try-cartridge"))
        .filter_map(|w| w.get(1))
        .flat_map(|v| v.split(',').map(str::trim).filter(|x| !x.is_empty()).map(str::to_owned).collect::<Vec<_>>())
        .collect()
}

/// The rack the plugs give, per MicroModel: served cartridges (and `tried` ones, even on trial)
/// ordered by use, then their place in the stack, then the order tried. Returns the
/// rack, one line per cartridge plugged, and the cartridges refused with why.
#[must_use]
pub fn rack_of(lib: &Library, plugs: &[Plug], tried: &[String]) -> (BTreeMap<String, Vec<nmengn::filter::CartridgeFilter>>, Vec<Value>, Vec<String>) {
    let mut chosen: Vec<((u8, u8), usize, Plug)> = plugs.iter().filter(|p| !p.trial).map(|p| (p.use_rank, p.at, p.clone())).collect();
    let mut refused = Vec::new();
    for (i, t) in tried.iter().enumerate() {
        let Some(c) = lib.cartridge(t) else {
            refused.push(format!("{t}: not in the cartridge library"));
            continue;
        };
        // Base cartridges are seated by `bases_of`, not plugged into sockets.
        if chosen.iter().any(|x| x.2.key == c.key()) || c.kind == "tensors" {
            continue;
        }
        match lib.filter(c) {
            Ok(f) if f.kind() == "affine" => chosen.push((
                (9, 0),
                i,
                Plug { key: c.key(), family: c.family.clone(), model: c.model.clone(), use_rank: (9, 0), at: i, points: f.points(), trial: c.trial },
            )),
            Ok(_) => {}
            Err(e) => refused.push(format!("{t}: {e}")),
        }
    }
    chosen.sort_by_key(|a| (a.0, a.1));
    let mut rack: BTreeMap<String, Vec<nmengn::filter::CartridgeFilter>> = BTreeMap::new();
    let mut lines = Vec::new();
    let home = lib.dir.parent().unwrap_or(&lib.dir);
    for (_, _, p) in chosen {
        let (weights, contract) = lib.cartridge(&p.key).map(|c| (c.weights.clone(), c.sockets.clone())).unwrap_or_default();
        let targets: Vec<&str> = p.points.iter().map(|x| x.target.as_str()).collect();
        if let Err(e) = nmengn::filter::fits(home, &p.model, &weights, &contract, &targets) {
            refused.push(format!("{}: {e}", p.key));
            continue;
        }
        let mut parts: Vec<nmengn::filter::Part> = p.points.iter().map(|x| nmengn::filter::Part { target: x.target.clone(), gain: x.gain.clone(), bias: x.bias.clone() }).collect();
        // The cartridge's parameter overlay: masked whole, its effect is left out while it stays
        // plugged; otherwise its edits and masks apply to its points.
        if let Some(ov) = nmengn::overlay::effective(home, &p.key) {
            if ov.whole_masked() {
                refused.push(format!("{}: masked whole in its parameter overlay; left out while it stays plugged", p.key));
                continue;
            }
            ov.apply_parts(&mut parts);
        }
        lines.push(obj(vec![
            ("key", s(&p.key)),
            ("family", s(&p.family)),
            ("model", s(&p.model)),
            ("points", strs(&parts.iter().map(|x| x.target.clone()).collect::<Vec<_>>())),
            ("values", n(parts.iter().map(|x| x.gain.len() as u64 * 2).sum())),
            ("trial", Value::Bool(p.trial)),
        ]));
        rack.entry(p.model.clone()).or_default().push(nmengn::filter::CartridgeFilter { cartridge: p.key.clone(), parts });
    }
    (rack, lines, refused)
}

/// What `bases_of` returns: the files per MicroModel, a line per cartridge offered, the cartridges
/// left out with why, and those by MicroModel.
pub type Bases = (BTreeMap<String, Vec<PathBuf>>, Vec<Value>, Vec<String>, BTreeMap<String, Vec<String>>);

/// The base cartridges to offer each MicroModel when it opens, in the order to try them (the
/// first that seats is its base source; none seated leaves the open weights). A base cartridge is a
/// cartridge of parameter sets on the base shelf made for this machine's open weights of its
/// MicroModel, whose file is on disk at its recorded size; it serves once it passes its seat
/// (complete, connected, the check run), with no trial wait. Per MicroModel: the ones named by
/// `--try-cartridge` first; then the one `settings.json` names under `base_cartridges`
/// (`{"<model>": "<id@version>"}`, or `"open"` for the open weights alone), else every base
/// cartridge newest first (by the day its label says it was made, then the file's time, then its
/// version). Returns the
/// files per MicroModel, a line per cartridge offered, and the cartridges left out with why (all,
/// and by MicroModel where the shelf knows it).
#[must_use]
pub fn bases_of(lib: &Library, home: &Path, tried: &[String]) -> Bases {
    type Cart = nmlearn::library::Cartridge;
    let (mut seat, mut lines, mut refused): (BTreeMap<String, Vec<PathBuf>>, Vec<Value>, Vec<String>) = (BTreeMap::new(), Vec::new(), Vec::new());
    let mut left_out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let named: BTreeMap<String, String> = std::fs::read(home.join("settings.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("base_cartridges").ok().cloned())
        .and_then(|v| if let Value::Object(o) = v { Some(o) } else { None })
        .map(|o| o.into_iter().filter_map(|(k, v)| v.string().ok().map(|x| (k.clone(), x.to_owned()))).collect())
        .unwrap_or_default();
    let bare = |model: &str| -> Option<String> {
        let card = nmtype::models_dir(home).join(model).join("part.json");
        std::fs::read(&card).ok().and_then(|b| nmjson::parse(&b).ok()).and_then(|v| v.get("weights").and_then(|w| w.get("sha256")).and_then(Value::string).ok().map(str::to_owned))
    };
    let file_of = |c: &Cart| lib.cartridge_dir(c).join(c.files.first().map_or("", |f| f.name.as_str()));
    let usable = |c: &Cart| -> Result<(), String> {
        // The file first: a cartridge whose file is gone is named as such, not by the fields its
        // label would have given.
        if !lib.files_present(c) {
            return Err(format!("{}: its file is missing or not the size its record states", c.key()));
        }
        let open = bare(&c.model).unwrap_or_default();
        if open != c.weights {
            return Err(format!("{}: made for open weights {:.12}, this {} is {:.12}", c.key(), c.weights, c.model, open));
        }
        lib.placed(c).map_err(|e| format!("{}: {e}", c.key()))?;
        Ok(())
    };
    let stamp = |c: &Cart| std::fs::metadata(file_of(c)).and_then(|m| m.modified()).ok();
    let mut by_model: BTreeMap<String, Vec<&Cart>> = BTreeMap::new();
    for c in lib.cartridges.iter().filter(|c| c.kind == "tensors" && c.family != nmlearn::library::ADAPTER) {
        by_model.entry(c.model.clone()).or_default().push(c);
    }
    for (model, mut cs) in by_model {
        cs.sort_by(|a, b| (b.made.as_str(), stamp(b), b.version).cmp(&(a.made.as_str(), stamp(a), a.version)));
        let mut order: Vec<&Cart> = tried.iter().filter_map(|t| lib.cartridge(t)).filter(|c| c.kind == "tensors" && c.model == model).collect();
        match named.get(&model).map(String::as_str) {
            Some("open") => {}
            Some(k) => match lib.cartridge(k) {
                Some(c) if c.model == model && c.kind == "tensors" => order.push(c),
                _ => {
                    let why = format!("{k}: named in settings.json for {model}, not a base cartridge of it in the cartridge library");
                    left_out.entry(model.clone()).or_default().push(why.clone());
                    refused.push(why);
                }
            },
            None => order.extend(cs),
        }
        let mut seen = std::collections::BTreeSet::new();
        for c in order {
            if !seen.insert(c.key()) {
                continue;
            }
            match usable(c) {
                Ok(()) => {
                    lines.push(obj(vec![("key", s(&c.key())), ("model", s(&c.model)), ("bytes", n(c.bytes())), ("made", s(&c.made))]));
                    seat.entry(model.clone()).or_default().push(file_of(c));
                }
                Err(e) => {
                    left_out.entry(model.clone()).or_default().push(e.clone());
                    refused.push(e);
                }
            }
        }
    }
    // A listed cartridge whose label cannot be read and whose shelf row does not say its
    // MicroModel is still named.
    for c in lib.cartridges.iter().filter(|c| c.kind.is_empty() && c.family != nmlearn::library::ADAPTER) {
        refused.push(format!("{}: its file {} cannot be read; the cartridge library does not say which MicroModel it serves", c.key(), c.files.first().map_or("", |f| f.name.as_str())));
    }
    (seat, lines, refused, left_out)
}

/// Rolls a MicroModel's base cartridge back one version: the cartridge it serves by default now
/// (the first `bases_of` offers it) is replaced by the version it was made over (its label's
/// `notes.parent`, else the next older base cartridge of that MicroModel), named in `settings.json`
/// under `base_cartridges`, with the reason in the audit (`library-rollback`). Returns the line.
/// # Errors
/// No base cartridge for the MicroModel, nothing older to roll back to, or the settings cannot be
/// written.
pub fn rollback(home: &Path, model: &str, reason: &str) -> Result<String, String> {
    let lib = Library::open(home).map_err(|e| e.to_string())?;
    let (seat, lines, _, _) = bases_of(&lib, home, &[]);
    let offered: Vec<String> = lines.iter().filter(|l| text(l, "model") == model).map(|l| text(l, "key")).collect();
    let now = offered.first().cloned().ok_or_else(|| format!("{model}: no base cartridge serves it by default"))?;
    let file = seat.get(model).and_then(|f| f.first().cloned()).ok_or_else(|| format!("{model}: no file"))?;
    let parent = nmlearn::library::label_of(&file).ok().and_then(|l| l.get("notes").and_then(|n| n.get("parent")).and_then(Value::string).ok().map(str::to_owned));
    let to = match parent.filter(|p| lib.cartridge(p).is_some()) {
        Some(p) => p,
        None => offered.get(1).cloned().ok_or_else(|| format!("{model}: {now} has no earlier version in the cartridge library to roll back to"))?,
    };
    let path = home.join("settings.json");
    let mut v = std::fs::read(&path).ok().and_then(|b| nmjson::parse(&b).ok()).unwrap_or_else(|| Value::Object(BTreeMap::new()));
    if let Value::Object(o) = &mut v
        && let Value::Object(m) = o.entry("base_cartridges".to_owned()).or_insert_with(|| Value::Object(BTreeMap::new()))
    {
        m.insert(model.to_owned(), s(&to));
    }
    std::fs::write(&path, nmjson::pretty::pretty(&v)).map_err(|e| format!("{}: {e}", path.display()))?;
    let _ = crate::controls::audit(home, "library-rollback", vec![("model", s(model)), ("from", s(&now)), ("to", s(&to)), ("reason", s(reason))]);
    Ok(format!("{model}: base cartridge rolled back from {now} to {to} ({reason}); serves from the next start"))
}

// ---------------------------------------------------------------------------------------------
// Base cartridge versions (2026-11-25, directions of Luis Jake Gabriel III: the user controls an
// upgrade from the interface: replace the old base cartridges, or keep the new versions beside them
// in the library and choose which one serves).

/// The control that says what an upgrade does with older base cartridge versions: `keep` (default:
/// the new version is added beside the old ones and serves by default; nothing is deleted) or
/// `replace` (older versions are removed once the new one is on the shelf with the right size and
/// usable; a version the user named in `settings.json` is never removed).
pub const UPGRADE_KEY: &str = "library.upgrade";

/// The `base_cartridges` choices in `settings.json` (MicroModel to cartridge key, or `open`).
fn named_bases(home: &Path) -> BTreeMap<String, String> {
    std::fs::read(home.join("settings.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("base_cartridges").ok().cloned())
        .and_then(|v| if let Value::Object(o) = v { Some(o) } else { None })
        .map(|o| o.into_iter().filter_map(|(k, v)| v.string().ok().map(|x| (k, x.to_owned()))).collect())
        .unwrap_or_default()
}

/// Writes one MicroModel's `base_cartridges` choice into `settings.json` (`None` removes it, so the
/// newest usable version serves), keeping every other key.
fn write_named_base(home: &Path, model: &str, choice: Option<&str>) -> Result<(), String> {
    let path = home.join("settings.json");
    let mut v = std::fs::read(&path).ok().and_then(|b| nmjson::parse(&b).ok()).unwrap_or_else(|| Value::Object(BTreeMap::new()));
    if let Value::Object(o) = &mut v
        && let Value::Object(m) = o.entry("base_cartridges".to_owned()).or_insert_with(|| Value::Object(BTreeMap::new()))
    {
        match choice {
            Some(c) => {
                m.insert(model.to_owned(), s(c));
            }
            None => {
                m.remove(model);
            }
        }
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, nmjson::pretty::pretty(&v)).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Every base cartridge version on the shelf, by MicroModel: key, version, made, bytes, whether its
/// file is present, why it is refused (if it is), whether it serves (the first `bases_of` offers),
/// the user's choice for the MicroModel (`auto` when none), and the upgrade policy.
pub fn base_versions(home: &Path) -> Value {
    let lib = match Library::open(home) {
        Ok(l) => l,
        Err(e) => return obj(vec![("error", s(&e.to_string()))]),
    };
    let (_, lines, _, left_out) = bases_of(&lib, home, &[]);
    let named = named_bases(home);
    let mut by_model: BTreeMap<String, Vec<&nmlearn::library::Cartridge>> = BTreeMap::new();
    for c in lib.cartridges.iter().filter(|c| c.kind == "tensors" && c.family != nmlearn::library::ADAPTER) {
        by_model.entry(c.model.clone()).or_default().push(c);
    }
    let models: Vec<Value> = by_model
        .into_iter()
        .map(|(model, mut cs)| {
            cs.sort_by_key(|c| std::cmp::Reverse(c.version));
            let serving = lines.iter().find(|l| text(l, "model") == model).map(|l| text(l, "key"));
            let refused = left_out.get(&model).cloned().unwrap_or_default();
            let versions: Vec<Value> = cs
                .iter()
                .map(|c| {
                    let key = c.key();
                    let why = refused.iter().find(|r| r.starts_with(&format!("{key}:"))).cloned();
                    obj(vec![
                        ("key", s(&key)),
                        ("version", n(c.version)),
                        ("made", s(&c.made)),
                        ("bytes", n(c.bytes())),
                        ("present", Value::Bool(lib.files_present(c))),
                        ("refused", why.map_or(Value::Null, |w| s(&w))),
                        ("serving", Value::Bool(serving.as_deref() == Some(key.as_str()))),
                    ])
                })
                .collect();
            let choice = named.get(&model).cloned().unwrap_or_else(|| "auto".to_owned());
            obj(vec![
                ("model", s(&model)),
                ("serving", serving.map_or_else(|| s("open weights"), |k| s(&k))),
                ("choice", s(&choice)),
                ("versions", Value::Array(versions)),
            ])
        })
        .collect();
    let policy = crate::controls::current(home).get(UPGRADE_KEY).cloned().unwrap_or_else(|| "keep".to_owned());
    obj(vec![("policy", s(&policy)), ("applies", s("next start")), ("models", Value::Array(models))])
}

/// Chooses what serves for one MicroModel from the next start: a base cartridge key on the shelf,
/// `open` (its open weights), or `auto` (the newest usable version). Returns the audit id.
/// # Errors
/// A key that is not a base cartridge of that MicroModel, or a write failure.
pub fn choose_base(home: &Path, model: &str, choice: &str, by: &str) -> Result<String, String> {
    choose_base_as(home, model, choice, by, None)
}

/// `choose_base`, naming the audit row it undoes (`undo_of`) when it puts an earlier choice back.
/// The row keeps the choice before (`from`) and after (`to`), so it can be undone in its turn.
fn choose_base_as(home: &Path, model: &str, choice: &str, by: &str, undo_of: Option<&str>) -> Result<String, String> {
    let from = named_bases(home).get(model).cloned().unwrap_or_else(|| "auto".to_owned());
    let lib = Library::open(home).map_err(|e| e.to_string())?;
    let known = lib.cartridges.iter().any(|c| c.kind == "tensors" && c.model == model);
    if !known && choice != "open" && choice != "auto" {
        return Err(format!("{model}: no base cartridge of this MicroModel is in the cartridge library"));
    }
    match choice {
        "auto" => write_named_base(home, model, None)?,
        "open" => write_named_base(home, model, Some("open"))?,
        key => match lib.cartridge(key) {
            Some(c) if c.kind == "tensors" && c.model == model => write_named_base(home, model, Some(key))?,
            _ => return Err(format!("{key}: not a base cartridge of {model} in the cartridge library")),
        },
    }
    let label = format!("Base cartridge for {model}: {from} to {choice}");
    let mut fields = vec![
        ("action", s("choose")),
        ("model", s(model)),
        ("choice", s(choice)),
        ("from", s(&from)),
        ("to", s(choice)),
        ("by", s(by)),
        ("applies", s("next start")),
        ("label", s(&label)),
    ];
    if let Some(u) = undo_of {
        fields.push(("undo_of", s(u)));
    }
    crate::controls::audit(home, "library-base", fields)
}

/// Undoes a base cartridge choice (`library-base` row, action `choose`): the choice before it is
/// written back through `choose_base`, logged as a new row naming the undone one. A row written
/// before the choice before was kept (no `from`) is refused with that reason.
/// # Errors
/// A row without its earlier choice, an earlier version no longer in the cartridge library, or a
/// write failure.
pub fn undo_base(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let model = text(row, "model");
    let from = row.get("from").and_then(Value::string).map_err(|_| format!("{id}: this base cartridge choice was logged without the choice before it, so it cannot be undone"))?.to_owned();
    let _g = crate::controls::write_guard()?;
    choose_base_as(home, &model, &from, by, Some(id)).map(Some)
}

/// Undoes a role's model change (`role-model` row, action `choose`): the role goes back on the
/// model it ran on before, through `choose_role`, logged as a new row naming the undone one.
/// # Errors
/// A row without its earlier model, an earlier model that may no longer run as the role, or a
/// write failure.
pub fn undo_role(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let role = text(row, "role");
    let from = row.get("from").and_then(Value::string).map_err(|_| format!("{id}: this role change was logged without the model before it, so it cannot be undone"))?.to_owned();
    let _g = crate::controls::write_guard()?;
    choose_role_as(home, &role, &from, by, Some(id)).map(|(a, _)| Some(a))
}

/// Removes one base cartridge version from the shelf: its catalogue row and its file. The version
/// that serves now, and a version the user named in `settings.json`, are refused (choose another
/// first). Returns the audit id.
/// # Errors
/// Unknown key, a refused version, or a write failure.
pub fn remove_base(home: &Path, key: &str, by: &str) -> Result<String, String> {
    let (mut lib, _g) = for_change(home)?;
    let c = lib.cartridge(key).filter(|c| c.kind == "tensors" && c.family != nmlearn::library::ADAPTER).cloned().ok_or_else(|| format!("{key}: not a base cartridge in the cartridge library"))?;
    let (_, lines, _, _) = bases_of(&lib, home, &[]);
    if lines.iter().find(|l| text(l, "model") == c.model).is_some_and(|l| text(l, "key") == key) {
        return Err(format!("{key} serves {} now; choose another version (or auto after adding one) before removing it", c.model));
    }
    if named_bases(home).get(&c.model).is_some_and(|k| k == key) {
        return Err(format!("{key} is the version chosen for {} in settings; choose another first", c.model));
    }
    let file = lib.cartridge_dir(&c).join(c.files.first().map_or("", |f| f.name.as_str()));
    lib.unlist(key);
    lib.save().map_err(|e| e.to_string())?;
    let bytes = c.bytes();
    if file.is_file() {
        if let Ok(m) = std::fs::metadata(&file) {
            let mut p = m.permissions();
            #[expect(clippy::permissions_set_readonly_false, reason = "the shelf file is removed right after; Windows marks it read-only")]
            p.set_readonly(false);
            let _ = std::fs::set_permissions(&file, p);
        }
        std::fs::remove_file(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    }
    crate::controls::audit(home, "library-base-remove", vec![("cartridge", s(key)), ("model", s(&c.model)), ("bytes", n(bytes)), ("by", s(by))])
}

/// Merges a release catalogue (`library/cartridges/release-catalogue.json`, written by the installer) into the
/// library: cartridge rows and stacks it does not list yet are added; nothing the user has is
/// changed. Then, under the `replace` policy, older base cartridge versions of each MicroModel that
/// gained a usable new version are removed (a version named in settings is kept). Returns one line
/// per change.
/// # Errors
/// An unreadable release catalogue, or a library that does not open after the merge (the earlier
/// catalogue is put back).
pub fn merge_release(home: &Path, release: &Path, by: &str) -> Result<Vec<String>, String> {
    let rel = std::fs::read(release).ok().and_then(|b| nmjson::parse(&b).ok()).ok_or_else(|| format!("{}: not a readable catalogue", release.display()))?;
    let dir = home.join("library");
    nmlearn::library::tidy(home);
    let cat_path = nmlearn::library::catalogue_file(&dir);
    let mut out = Vec::new();
    let added_keys: Vec<String>;
    {
        let _g = lock(&dir)?;
        let before = std::fs::read(&cat_path).ok();
        let Some(before_bytes) = before.clone() else {
            std::fs::copy(release, &cat_path).map_err(|e| format!("{}: {e}", cat_path.display()))?;
            out.push("library: catalogue created from the release".to_owned());
            return Ok(out);
        };
        let mut cat = nmjson::parse(&before_bytes).map_err(|e| format!("{}: {e}", cat_path.display()))?;
        let key_of = |v: &Value| format!("{}@{}", text(v, "id"), v.get("version").and_then(Value::integer).unwrap_or(0));
        let mut added = Vec::new();
        for section in ["cartridges", "stacks"] {
            let new_rows: Vec<Value> = rel.get(section).and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default();
            if let Value::Object(o) = &mut cat
                && let Value::Array(rows) = o.entry(section.to_owned()).or_insert_with(|| Value::Array(Vec::new()))
            {
                for row in new_rows {
                    let k = key_of(&row);
                    if !rows.iter().any(|r| key_of(r) == k) {
                        out.push(format!("library: {section} row {k} added"));
                        if section == "cartridges" {
                            added.push(k);
                        }
                        rows.push(row);
                    }
                }
            }
        }
        let tmp = cat_path.with_extension("json.tmp");
        std::fs::write(&tmp, nmjson::pretty::pretty(&cat)).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &cat_path).map_err(|e| format!("{}: {e}", cat_path.display()))?;
        if let Err(e) = Library::open(home) {
            let _ = std::fs::write(&cat_path, &before_bytes);
            return Err(format!("library: the merged catalogue did not open ({e}); the earlier one is back"));
        }
        added_keys = added;
    }
    if !added_keys.is_empty() {
        let _ = crate::controls::audit(home, "library-merge", vec![("added", strs(&added_keys)), ("by", s(by))]);
    }
    let policy = crate::controls::current(home).get(UPGRADE_KEY).cloned().unwrap_or_else(|| "keep".to_owned());
    if policy == "replace" && !added_keys.is_empty() {
        let lib = Library::open(home).map_err(|e| e.to_string())?;
        let (_, lines, _, _) = bases_of(&lib, home, &[]);
        let named = named_bases(home);
        let mut old: Vec<String> = Vec::new();
        for k in &added_keys {
            let Some(newc) = lib.cartridge(k).filter(|c| c.kind == "tensors") else { continue };
            // Only a new version that is on the shelf and offered (right size, made for these open
            // weights) replaces the older ones.
            if !lines.iter().any(|l| text(l, "key") == *k) {
                out.push(format!("library: {k} is not usable yet; older versions of {} kept", newc.model));
                continue;
            }
            for c in lib.cartridges.iter().filter(|c| c.kind == "tensors" && c.model == newc.model && c.version < newc.version) {
                if named.get(&c.model).is_some_and(|x| *x == c.key()) {
                    out.push(format!("library: {} kept (chosen in settings)", c.key()));
                } else {
                    old.push(c.key());
                }
            }
        }
        for k in old {
            match remove_base(home, &k, by) {
                Ok(_) => out.push(format!("library: {k} removed (upgrade policy replace)")),
                Err(e) => out.push(format!("library: {k} kept: {e}")),
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// MicroModels per role and cartridge compatibility (2026-11-25, directions of Luis Jake Gabriel
// III: the MicroModels are choosable, changeable and swappable per role, in the code and the
// interface; the interface shows which MicroModels each cartridge fits, so it is clear what can go
// where).

/// The open weights' SHA-256 an installed model's weight card states.
fn weights_sha(home: &Path, id: &str) -> String {
    std::fs::read(nmtype::models_dir(home).join(id).join("part.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("weights").and_then(|w| w.get("sha256")).and_then(Value::string).ok().map(str::to_owned))
        .unwrap_or_default()
}

/// Every role, the model it runs on from the next start, its default, the kind it needs, whether
/// it is locked (and why), and each installed model with whether it may run as that role.
pub fn roles_view(home: &Path) -> Value {
    use nmtype::role_models as rm;
    let now = rm::resolve(home);
    let installed = rm::installed(home);
    let roles: Vec<Value> = rm::ROLES
        .iter()
        .map(|(role, default)| {
            let mut others = now.clone();
            others.remove(*role);
            let choices: Vec<Value> = installed
                .iter()
                .map(|id| {
                    let ok = rm::may_run(home, role, id, &others);
                    obj(vec![("id", s(id)), ("ok", Value::Bool(ok.is_ok())), ("why", ok.err().map_or(Value::Null, |w| s(&w)))])
                })
                .collect();
            obj(vec![
                ("role", s(role)),
                ("model", s(now.get(*role).map_or(*default, String::as_str))),
                ("default", s(default)),
                ("needs", s(rm::kind_for_role(role).map_or("", rm::Kind::name))),
                ("locked", rm::locked(role).map_or(Value::Null, s)),
                ("choices", Value::Array(choices)),
            ])
        })
        .collect();
    let models: Vec<Value> = installed
        .iter()
        .map(|id| {
            let role = now.iter().find(|(_, m)| *m == id).map(|(r, _)| r.clone());
            obj(vec![("id", s(id)), ("kind", s(rm::kind_of(home, id).name())), ("role", role.map_or(Value::Null, |r| s(&r)))])
        })
        .collect();
    obj(vec![("applies", s("next start")), ("roles", Value::Array(roles)), ("models", Value::Array(models))])
}

/// Chooses the model `role` runs on from the next start (`default` puts the role back on its
/// default). Returns the audit id and a line saying what the new model leaves aside.
/// # Errors
/// A model that may not run as the role, or a write failure.
pub fn choose_role(home: &Path, role: &str, model: &str, by: &str) -> Result<(String, String), String> {
    choose_role_as(home, role, model, by, None)
}

/// `choose_role`, naming the audit row it undoes (`undo_of`) when it puts the earlier model back.
fn choose_role_as(home: &Path, role: &str, model: &str, by: &str, undo_of: Option<&str>) -> Result<(String, String), String> {
    use nmtype::role_models as rm;
    let default = rm::default_model(role).ok_or_else(|| format!("{role}: not a role (reader, writer, embedder or reranker)"))?;
    let now = rm::resolve(home);
    let to = if model == "default" { default } else { model };
    let mut others = now.clone();
    others.remove(role);
    rm::may_run(home, role, to, &others)?;
    let from = now.get(role).cloned().unwrap_or_else(|| default.to_owned());
    let path = home.join("settings.json");
    let mut v = std::fs::read(&path).ok().and_then(|b| nmjson::parse(&b).ok()).unwrap_or_else(|| Value::Object(BTreeMap::new()));
    if let Value::Object(o) = &mut v
        && let Value::Object(m) = o.entry(rm::SETTINGS_KEY.to_owned()).or_insert_with(|| Value::Object(BTreeMap::new()))
    {
        if to == default {
            m.remove(role);
        } else {
            m.insert(role.to_owned(), s(to));
        }
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, nmjson::pretty::pretty(&v)).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    // 2026-11-25 (Luis Jake Gabriel III: cartridges are not set aside): a swap changes only the model
    // the role runs on. Every cartridge stays plugged into the model it was made for, untouched; the
    // role runs with the cartridges of its new model, and finds its earlier ones again on the way back.
    let c = compat(home);
    let left: Vec<String> = c
        .get("cartridges")
        .and_then(Value::array)
        .unwrap_or_default()
        .iter()
        .filter(|x| text(x, "made_for") == from && from != to)
        .map(|x| text(x, "name"))
        .collect();
    let line = if from == to {
        format!("the {role} stays on {to}")
    } else if left.is_empty() {
        format!("the {role} runs on {to} from the next start")
    } else {
        format!("the {role} runs on {to} from the next start, with that model's cartridges; {} stay plugged into {from}, ready when the {role} goes back to it", left.join(", "))
    };
    let mut fields = vec![("action", s("choose")), ("role", s(role)), ("from", s(&from)), ("to", s(to)), ("by", s(by)), ("applies", s("next start")), ("label", s(&line))];
    if let Some(u) = undo_of {
        fields.push(("undo_of", s(u)));
    }
    let id = crate::controls::audit(home, "role-model", fields)?;
    Ok((id, line))
}

/// Which MicroModels each cartridge fits, and why not the others: every cartridge in the
/// catalogue (base, learned, adapters, the engine's own settings) and every job filter, against
/// every installed model (the same model, made for the same open weights).
/// The base cartridge (`id@version`) a description names, as a compensation cartridge's does.
fn base_named(text: &str) -> Option<String> {
    text.split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | ',' | ';' | ':'))
        .find(|w| w.split_once('@').is_some_and(|(id, v)| id.ends_with("-base") && !v.is_empty() && v.chars().all(|c| c.is_ascii_digit())))
        .map(str::to_owned)
}

#[must_use]
pub fn compat(home: &Path) -> Value {
    use nmtype::role_models as rm;
    let installed = rm::installed(home);
    let now = rm::resolve(home);
    let shas: BTreeMap<String, String> = installed.iter().map(|m| (m.clone(), weights_sha(home, m))).collect();
    let role_of = |m: &str| now.iter().find(|(_, x)| x.as_str() == m).map(|(r, _)| r.clone());
    let judge = |made_for: &str, weights: Option<&str>| -> (Vec<String>, BTreeMap<String, Value>) {
        let mut fits = Vec::new();
        let mut not = BTreeMap::new();
        for m in &installed {
            let why = if made_for == "engine" {
                None
            } else if made_for != m {
                Some(format!("made for {made_for}"))
            } else if let Some(w) = weights.filter(|w| !w.is_empty())
                && shas.get(m).is_some_and(|x| !x.is_empty() && x != w)
            {
                Some(format!("made for other open weights of {m}"))
            } else {
                None
            };
            match why {
                None => fits.push(m.clone()),
                Some(w) => {
                    not.insert(m.clone(), s(&w));
                }
            }
        }
        (fits, not)
    };
    let mut rows = Vec::new();
    if let Ok(lib) = Library::open(home) {
        for c in &lib.cartridges {
            let engine = c.model.is_empty() || c.model == "settings";
            let made_for = if engine { "engine".to_owned() } else { c.model.clone() };
            let (mut fits, mut not) = judge(&made_for, if c.kind == "tensors" { Some(c.weights.as_str()) } else { None });
            // 2026-11-26 (first-time walkthrough): a compensation cartridge is made for one base
            // cartridge version (its description names it: "the channels embedder-base@2 leaves
            // out"); on another version it does not fit, whatever the model.
            if c.kind == "affine"
                && let Some(made_on) = base_named(&c.description)
            {
                let serving = crate::pmaps::serving_base(home, &c.model);
                if !serving.is_empty() && made_on != serving && fits.contains(&c.model) {
                    fits.retain(|m| m != &c.model);
                    not.insert(c.model.clone(), s(&format!("made for {made_on}; {serving} serves now")));
                }
            }
            let kind = if c.family == nmlearn::library::ADAPTER {
                "lens"
            } else if c.kind == "tensors" {
                "base"
            } else if engine {
                "settings"
            } else if c.family == nmlearn::library::PREFERENCE {
                "preference"
            } else {
                "learned"
            };
            rows.push(obj(vec![
                ("key", s(&c.key())),
                ("name", s(&c.id)),
                ("kind", s(kind)),
                ("made_for", s(&made_for)),
                ("role", role_of(&made_for).map_or(Value::Null, |r| s(&r))),
                ("fits", strs(&fits)),
                ("not", Value::Object(not)),
            ]));
        }
    }
    let maps = home.join("library/parameter-maps");
    for role in ["embedder", "reader", "reranker", "writer"] {
        let Ok(rd) = std::fs::read_dir(maps.join(role)) else { continue };
        let mut files: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("job-filter-") && n.ends_with(".json"))).collect();
        files.sort();
        for p in files {
            let Some(v) = std::fs::read(&p).ok().and_then(|b| nmjson::parse(&b).ok()) else { continue };
            let made_for = text(&v, "model");
            let (fits, not) = judge(&made_for, None);
            rows.push(obj(vec![
                ("key", s(&format!("filter:{}", text(&v, "cartridge")))),
                ("name", s(&text(&v, "job"))),
                ("kind", s("filter")),
                ("made_for", s(&made_for)),
                ("role", role_of(&made_for).map_or_else(|| s(role), |r| s(&r))),
                ("fits", strs(&fits)),
                ("not", Value::Object(not)),
            ]));
        }
    }
    // Distilled adapter cartridges (`nmapi::distil`), keyed as the learned unit they plug in as
    // (`unit:<compartment>/<id>`): the model they were trained for, the open weights' SHA-256 they
    // were made for, and the base cartridge version they were fitted on, against what serves now.
    // What serves each MicroModel now, as the base choice decides it (the version named in
    // `settings.json`, else the library's serving version, else the open weights).
    let serving_now: BTreeMap<String, String> = base_versions(home)
        .get("models")
        .and_then(Value::array)
        .unwrap_or_default()
        .iter()
        .map(|m| (text(m, "model"), text(m, "serving")))
        .collect();
    for role in nmapi::distil::ROLES {
        for (id, file, _) in nmapi::distil::on_shelf(home, role) {
            let label = nmlearn::library::label_of(&file).unwrap_or(Value::Null);
            let made_for = text(&label, "model");
            let (mut fits, mut not) = judge(&made_for, Some(&text(&label, "weights")));
            let made_on = label.get("notes").map(|nt| text(nt, "base")).unwrap_or_default();
            let serving = serving_now.get(&made_for).filter(|k| !k.is_empty() && k.as_str() != "open weights").cloned().unwrap_or_else(|| format!("open:{made_for}"));
            if !made_on.is_empty() && made_on != serving && fits.contains(&made_for) {
                fits.retain(|m| m != &made_for);
                not.insert(made_for.clone(), s(&format!("fitted on {made_on}; {serving} serves now")));
            }
            rows.push(obj(vec![
                ("key", s(&format!("unit:{}", nmapi::plugs::key(nmapi::plugs::NANO, &id)))),
                ("name", s(&id)),
                ("kind", s("distil")),
                ("made_for", s(&made_for)),
                ("role", role_of(&made_for).map_or_else(|| s(role), |r| s(&r))),
                ("fits", strs(&fits)),
                ("not", Value::Object(not)),
            ]));
        }
    }
    obj(vec![("models", strs(&installed)), ("cartridges", Value::Array(rows))])
}

/// Why a distilled cartridge does not fit the MicroModel it was made for, by the compatibility
/// table (`compat`); None when it fits.
#[must_use]
pub fn distil_misfit(home: &Path, id: &str) -> Option<String> {
    let key = format!("unit:{}", nmapi::plugs::key(nmapi::plugs::NANO, id));
    let c = compat(home);
    let row = c.get("cartridges").and_then(Value::array).unwrap_or_default().iter().find(|r| text(r, "key") == key).cloned()?;
    let made_for = text(&row, "made_for");
    let fits = row.get("fits").and_then(Value::array).unwrap_or_default().iter().any(|m| m.string().ok() == Some(made_for.as_str()));
    if fits {
        return None;
    }
    let why = row.get("not").ok().and_then(|n| n.get(&made_for).ok()).and_then(|w| w.string().ok()).map_or_else(|| format!("{made_for} is not installed"), str::to_owned);
    Some(format!("{id} does not fit {made_for}: {why}"))
}

/// Records one MicroModel's seating in `memory/stacks/in-use.json` (`seats`, by MicroModel) and the
/// audit, so `library stack` and the interface show what serves and why.
fn record_seat(model: &str, key: Option<&str>, line: &str, refused: &[String]) {
    let home = nmtype::nano_root();
    nmlearn::library::tidy(&home);
    let file = nmlearn::library::in_use_file(&home);
    let mut v = std::fs::read(&file).ok().and_then(|b| nmjson::parse(&b).ok()).unwrap_or_else(|| Value::Object(BTreeMap::new()));
    let row = obj(vec![("model", s(model)), ("seated", key.map_or(Value::Null, s)), ("line", s(line)), ("refused", strs(refused)), ("pid", s(&std::process::id().to_string()))]);
    if let Value::Object(o) = &mut v
        && let Value::Object(m) = o.entry("seats".to_owned()).or_insert_with(|| Value::Object(BTreeMap::new()))
    {
        m.insert(model.to_owned(), row);
    }
    let _ = std::fs::write(&file, nmjson::pretty::pretty(&v));
    let _ = crate::controls::audit(&home, "library-seat", vec![("model", s(model)), ("seated", key.map_or(Value::Null, s)), ("line", s(line)), ("refused", strs(refused))]);
}

/// The seats the serving process recorded at its last start, one line each.
fn seats_text(home: &Path) -> String {
    let v = std::fs::read(nmlearn::library::in_use_file(home)).ok().and_then(|b| nmjson::parse(&b).ok());
    let mut out = String::new();
    if let Some(Value::Object(m)) = v.as_ref().and_then(|v| v.get("seats").ok()) {
        let _ = writeln!(out, "base sources at the last start (process {}):", m.values().next().map(|r| text(r, "pid")).unwrap_or_default());
        for r in m.values() {
            let _ = writeln!(out, "  {}", text(r, "line"));
        }
    }
    out
}

/// Watches the catalogue of a serving process: when it changes (a stack switched from the command or
/// the interface), the stacks in use are worked out again and the new rack is staged; the engine
/// applies it where the next call starts. Each change is audited (`library-sockets`).
fn watch(home: PathBuf, machine: Machine, tried: Vec<String>) {
    let file = nmlearn::library::catalogue_file(&home.join("library"));
    // The catalogue's time and the parameter overlays' stamp: an overlay written for a plugged
    // cartridge or a preference cartridge reaches the next call as a stack switch does.
    let stamp = |f: &Path, h: &Path| (std::fs::metadata(f).and_then(|m| m.modified()).ok(), nmengn::overlay::stamp(h));
    let mut last = stamp(&file, &home);
    let _ = std::thread::Builder::new().name("library-watch".into()).spawn(move || {
        loop {
            std::thread::park_timeout(std::time::Duration::from_secs(1));
            let now = stamp(&file, &home);
            if now == last {
                continue;
            }
            last = now;
            let Ok(lib) = Library::open(&home) else { continue };
            let u = in_use(&lib, machine.clone());
            // A preference stack switched (a new cartridge promoted, or a rollback) reaches the
            // next reply.
            nmapi::prefs::set_served(&u.pref_values_overlaid(&home));
            let (rack, lines, mut refused) = rack_of(&lib, &u.plugs, &tried);
            if let Err(e) = nmengn::filter::stage(rack) {
                refused.push(e);
            }
            let _ = crate::controls::audit(
                &home,
                "library-sockets",
                vec![("pid", s(&std::process::id().to_string())), ("sockets", Value::Array(lines)), ("refused", strs(&refused)), ("label", s("Cartridges for the next call after a stack switch"))],
            );
        }
    });
}

/// The `serve` arguments with the stacks in use added: their settings become options not given on
/// the command line; precision widths are set before any model binds; affine cartridges are plugged
/// into their MicroModels' sockets. A serving process (`SERVING`) records its start in the audit
/// (`library-stacks`) and `memory/stacks/in-use.json` and watches the catalogue; other runs write nothing.
/// `--rack off` returns the arguments unchanged and plugs nothing; a failure is reported and leaves
/// them unchanged.
#[must_use]
pub fn apply(home: &Path, args: &[String]) -> Vec<String> {
    if args.windows(2).any(|w| w.first().map(String::as_str) == Some("--rack") && w.get(1).map(String::as_str) == Some("off")) {
        eprintln!("library: off for this run");
        return args.to_vec();
    }
    let lib = match open(home) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("library: {e}; serving without it");
            return args.to_vec();
        }
    };
    let machine = this_machine(home, false);
    let slow_missing = machine.piece(Kind::Ram).is_none() || machine.piece(Kind::Disk).is_none();
    let u = in_use(&lib, machine);
    let tried = tried(args);
    // `--base-cartridges off`: every MicroModel on its open weights for this run (a comparison arm).
    let bases_off = args.windows(2).any(|w| w.first().map(String::as_str) == Some("--base-cartridges") && w.get(1).map(String::as_str) == Some("off"));
    let (seat, base_lines, base_refused, left_out) =
        if bases_off { (BTreeMap::new(), Vec::new(), vec!["--base-cartridges off: the open weights serve".to_owned()], BTreeMap::new()) } else { bases_of(&lib, home, &tried) };
    if let Ok(mut g) = nmapi::neural::BASE_CARTRIDGES.write() {
        *g = seat;
    }
    if let Ok(mut g) = nmapi::neural::BASE_LEFT_OUT.write() {
        *g = left_out;
    }
    if let Ok(mut g) = nmapi::neural::BASE_KEYS.write() {
        *g = lib.cartridges.iter().filter(|c| c.kind == "tensors").map(|c| (lib.cartridge_dir(c).join(c.files.first().map_or("", |f| f.name.as_str())), c.key())).collect();
    }
    if SERVING.load(Ordering::Relaxed)
        && let Ok(mut h) = nmapi::neural::ON_SEAT.write()
    {
        *h = Some(record_seat);
    }
    for l in &base_lines {
        eprintln!("library: base cartridge {} offered to {} (made {})", text(l, "key"), text(l, "model"), text(l, "made"));
    }
    if !base_refused.is_empty() {
        eprintln!("library: base cartridges left out: {}", base_refused.join("; "));
    }
    let (rack, sockets, mut socket_refused) = rack_of(&lib, &u.plugs, &tried);
    if let Err(e) = nmengn::filter::replace_all(&rack) {
        socket_refused.push(e);
    }
    for l in &sockets {
        eprintln!("library: socket {} on {} ({}){}", text(l, "key"), text(l, "model"), text(l, "family"), if matches!(l.get("trial"), Ok(Value::Bool(true))) { ", tried on trial" } else { "" });
    }
    if !socket_refused.is_empty() {
        eprintln!("library: sockets refused {}", socket_refused.join("; "));
    }
    nmapi::prefs::set_served(&u.pref_values_overlaid(home));
    let applied = translate(&u.values(), args);
    if !applied.precision.is_empty()
        && crate::controls::precision_arg(home).is_none()
        && let Ok(mut g) = nmengn::model::PRECISION.write()
    {
        for (part, w) in &applied.precision {
            let bits = w.parse::<u8>().ok();
            match part.as_str() {
                "base" => g.base = bits,
                "reranker" => g.reranker = bits,
                "embedder" => g.embedder = bits,
                _ => {}
            }
        }
    }
    let mut v = u.to_value(&applied);
    if let Value::Object(o) = &mut v {
        o.insert("sockets".into(), Value::Array(sockets));
        o.insert("sockets_refused".into(), strs(&socket_refused));
        o.insert("tried".into(), strs(&tried));
        o.insert("seated".into(), Value::Array(base_lines));
        o.insert("seated_refused".into(), strs(&base_refused));
    }
    if SERVING.load(Ordering::Relaxed) {
        let mut row: Vec<(&str, Value)> = vec![
            ("pid", s(&std::process::id().to_string())),
            ("gpu", u.machine.piece(Kind::Gpu).map_or(Value::Null, |g| obj(vec![("name", s(&g.name)), ("memory_mib", n(g.memory_mib))]))),
            ("label", s("Cartridge stacks put in use at start")),
        ];
        for k in ["uses", "loaded", "set", "trial", "options", "refused", "sockets", "sockets_refused", "tried", "seated", "seated_refused"] {
            if let Ok(x) = v.get(k) {
                row.push((k, x.clone()));
            }
        }
        let _ = crate::controls::audit(home, "library-stacks", row);
        let f = nmlearn::library::in_use_file(home);
        if let Some(d) = f.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let _ = std::fs::write(&f, nmjson::pretty::pretty(&v));
        watch(home.to_path_buf(), u.machine.clone(), tried);
    }
    for c in &u.choices {
        eprintln!("library: {} {}{}: {}", c.use_for, c.stack.as_deref().unwrap_or("-"), if c.trial { " (trial)" } else { "" }, c.why);
    }
    if !applied.refused.is_empty() {
        eprintln!("library: refused {}", applied.refused.join("; "));
    }
    if slow_missing && SERVING.load(Ordering::Relaxed) {
        let h = home.to_path_buf();
        std::thread::spawn(move || {
            let _ = this_machine(&h, true);
        });
    }
    let mut out = args.to_vec();
    out.extend(applied.args);
    out
}

// ---------------------------------------------------------------------------------------------
// The interface.

/// Library cartridges as further rows of `/api/cartridges` (the fields the interface reads, and
/// `library`).
#[must_use]
pub fn api_rows(home: &Path) -> Vec<Value> {
    let Ok(lib) = Library::open(home) else { return Vec::new() };
    let in_use = std::fs::read(nmlearn::library::in_use_file(home)).ok().and_then(|b| nmjson::parse(&b).ok());
    let loaded = |key: &str| in_use.as_ref().and_then(|v| v.get("loaded").and_then(Value::array).ok()?.iter().find(|c| text(c, "key") == key).map(|c| matches!(c.get("trial"), Ok(Value::Bool(true)))));
    // 2026-11-27: a base cartridge is loaded when it is the one serving its MicroModel
    // (`bases_of`) and a serving process holds that MicroModel now (its live file's `loaded`
    // rows: the models seated at start), so a base cartridge in effect never reads as not loaded.
    let serving_bases: std::collections::BTreeSet<String> = bases_of(&lib, home, &[]).1.iter().map(|l| text(l, "key")).collect();
    let held_models: std::collections::BTreeSet<String> = crate::live::read_all(home)
        .iter()
        .flat_map(|p| p.get("loaded").and_then(Value::array).unwrap_or_default().to_vec())
        .flat_map(|m| [text(&m, "id"), text(&m, "card")])
        .collect();
    lib.cartridges
        .iter()
        .map(|c| {
            // Files on disk at their recorded sizes; their hashes are checked when seated or loaded.
            let verified = lib.files_present(c);
            let base_held = c.kind == "tensors" && c.family != nmlearn::library::ADAPTER && serving_bases.contains(&c.key()) && held_models.contains(&c.model);
            let trial_now = loaded(&c.key()).or(base_held.then_some(false));
            let bytes = c.bytes();
            let date = c.evidence.iter().map(|r| r.at).max().map_or_else(String::new, nmlearn::cartridge::date_of);
            let stacks: Vec<String> = lib.stacks.iter().filter(|s| s.cartridges.contains(&c.key())).map(Stack::key).collect();
            obj(vec![
                ("schema", s(nmlearn::library::CARTRIDGE_SCHEMA)),
                ("id", s(&c.id)),
                ("kind", s(&c.kind)),
                ("data_class", s("behavioural")),
                ("base", s("")),
                ("base_sha256", s("")),
                ("components", Value::Array(Vec::new())),
                ("scopes", Value::Array(vec![s(&c.domain)])),
                ("family", s(&c.family)),
                ("quantization", s(if c.files.iter().any(|f| f.name == "numbers.q8") { "q8" } else if c.kind == "affine" { "f32" } else { "none" })),
                ("description", s(&c.description)),
                ("version", n(c.version)),
                ("check", s(if c.trial { "trial" } else { "accept" })),
                ("date", s(&date)),
                ("loaded", Value::Bool(trial_now.is_some())),
                ("bytes", n(bytes)),
                ("loaded_bytes", n(if trial_now.is_some() { bytes } else { 0 })),
                ("model", s(&c.model)),
                ("class", s("self")),
                ("layers", c.layers.map_or(Value::Null, |(a, b)| Value::Array(vec![n(a), n(b)]))),
                ("verified", Value::Bool(verified)),
                ("switch_key", s(&format!("library:{}", stacks.first().cloned().unwrap_or_default()))),
                (
                    "library",
                    obj(vec![
                        ("key", s(&c.key())),
                        ("label", obj(vec![("name", s(&c.name)), ("does", s(&c.description)), ("model", s(&c.model)), ("loads_at", s(&c.target)), ("layers", c.layers.map_or(Value::Null, |(a, b)| Value::Array(vec![n(a), n(b)]))), ("made_from", s(&c.made_from)), ("made", s(&c.made)), ("bytes", n(c.bytes())), ("effect", s(&c.effect))])),
                        ("family", s(&c.family)),
                        ("domain", s(&c.domain)),
                        ("target", s(&c.target)),
                        ("trial", Value::Bool(c.trial)),
                        ("in_use", trial_now.map_or_else(|| s("no"), |t| s(if t { "trial" } else { "served" }))),
                        ("stacks", strs(&stacks)),
                        ("hash", s(&c.hash)),
                        ("evidence_rows", n(c.evidence.len() as u64)),
                    ]),
                ),
            ])
        })
        .collect()
}

/// `/api/cartridges/library`: the catalogue, and the stacks in use for this machine (from the
/// cached machine; nothing slow is probed on a request), and those put in use at the last start.
#[must_use]
pub fn api_view(home: &Path) -> Value {
    let lib = match Library::open(home) {
        Ok(l) => l,
        Err(e) => return obj(vec![("error", s(&e.to_string()))]),
    };
    let cached = std::fs::read(nmlearn::library::machine_file(home)).ok().and_then(|b| nmjson::parse(&b).ok()).and_then(|v| Machine::from_value(&v).ok());
    let machine = cached.unwrap_or_else(|| hardware::probe(home, false, None));
    let u = in_use(&lib, machine);
    let applied = translate(&u.values(), &[]);
    obj(vec![
        ("exists", Value::Bool(lib.exists())),
        ("catalogue", lib.to_value()),
        ("in_use", u.to_value(&applied)),
        ("last_start", std::fs::read(nmlearn::library::in_use_file(home)).ok().and_then(|b| nmjson::parse(&b).ok()).unwrap_or(Value::Null)),
    ])
}

// ---------------------------------------------------------------------------------------------
// The command.

const USAGE: &str = "usage: neuromorph-nano library <list|show|stack|switch|export|import|rollback|bases|base|base-remove|compat|parameters|parameters-edit|overlay|roles|role|merge> ...
  rollback <model> [reason]           the MicroModel's base cartridge back to the version it was
                                      made over (from the next start)
  bases [--json]                      every base cartridge version by MicroModel, which serves,
                                      the choice and the upgrade policy (library.upgrade)
  base <model> <key|auto|open>        what serves the MicroModel from the next start: a version
                                      in the cartridge library, the newest usable (auto) or its
                                      open weights
  base-remove <key> --confirm         delete one base cartridge version (not the serving or the
                                      chosen one)
  compat                              which MicroModels each cartridge and job filter fits, and
                                      why not the others (JSON)
  parameters [<key>] [--tensor <name> [--offset <n>] [--limit <n>]]
                                      every object holding parameters, or one object's tensors
                                      with shapes and norms, or one page of a tensor's values
  parameters <key> --overview [--tensor <name>] [--w N --h N] [--region r0:r1,c0:c1] [--hist]
                                      the whole-set overview, or one tensor's grid (the JSON of
                                      GET /api/parameters/overview)
  parameters-edit <key> [--scale <s>] [--layer <n>=on|off|<s>]... [--values <json>]
                                      a new version with a strength, layers switched or scaled,
                                      or values set (the source is never written; audited)
  overlay <key> [edit|mask|unmask|freeze|unfreeze|clear|undo] [--path <p>] [--id <e>]
          [--op <op> --value <x>] [--preview]
                                      an object's parameter overlay, held beside it and applied
                                      when the engine seats it (the files are never written)
  roles                               the model each role runs on, the kind it needs and the
                                      installed models it may take (JSON)
  role <role> <model id|default>      the model a role runs on from the next start
  merge <release-catalogue.json>      add a release's new cartridges and stacks to the cartridge
                                      library (the installer runs it; library.upgrade replace then
                                      removes the older base versions)
  list [--json]                       cartridges (domain, the MicroModel and layers they
                                      filter) and stacks, hash check, trial
  show <id> [--json]                  one stack or cartridge: its filters, files, evidence
  stack [<use>] [--json] [--machine <file>]
                                      the stack in use for each use on this machine, why, and
                                      what its filters set
  switch <use> <stack>|off            the stack for a use (a running server plugs its socket
                                      cartridges at its next call; settings at the next start)
  switch <stack> trial|serve          put a stack on trial, or serve it (needs evidence measured
                                      on its own piece)
  export <stack> --out <file>         the stack and its cartridges as one file
  import <file>                       check a stack file and add it on trial";

fn arg<'a>(args: &'a [String], flag: &str) -> Option<&'a String> {
    args.windows(2).find(|w| w.first().map(String::as_str) == Some(flag)).and_then(|w| w.get(1))
}

fn plain_in_use(u: &InUse, applied: &Applied, only: Option<&str>) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "machine ({} probe):", u.machine.how);
    for p in &u.machine.pieces {
        let _ = writeln!(out, "  {:<4} {}  ({}; family {})", p.kind.name(), p.identity(), p.name, p.family_key());
    }
    let _ = writeln!(out, "stacks in use, applied in this order:");
    for c in u.choices.iter().filter(|c| only.is_none_or(|o| o == c.use_for)) {
        let _ = writeln!(out, "  {:<14} {:<38} {}{}", c.use_for, c.stack.as_deref().unwrap_or("-"), if c.trial { "TRIAL " } else { "" }, c.why);
    }
    for (k, t, _, r) in &u.loaded {
        let _ = writeln!(out, "    cartridge {k}{}{}", if *t { " (trial)" } else { "" }, r.as_ref().map_or_else(String::new, |r| format!(" REFUSED {r}")));
    }
    let _ = writeln!(out, "settings the served filters set:");
    for (k, (v, c)) in &u.set {
        let _ = writeln!(out, "  {k} = {v}  ({c}; code {})", code_value(k).unwrap_or_default());
    }
    let _ = writeln!(out, "options added to serve: {}", if applied.args.is_empty() { "none".to_owned() } else { applied.args.join(" ") });
    if !applied.refused.is_empty() {
        let _ = writeln!(out, "refused: {}", applied.refused.join("; "));
    }
    let _ = writeln!(out, "trial filters would change (never applied): {}", if u.trial.is_empty() { "nothing" } else { "" });
    for (k, v, c) in &u.trial {
        let _ = writeln!(out, "  {k} = {v}  ({c})");
    }
    out
}

fn show(lib: &Library, name: &str, ledger: &dyn EvidenceSource) -> Result<Value, String> {
    if let Some(st) = lib.stack(name) {
        let mut v = st.to_value();
        let l = lib.load(name).map_err(|e| e.to_string())?;
        let rows = stack_rows(lib, st, ledger);
        if let Value::Object(o) = &mut v {
            o.insert("type".into(), s("stack"));
            o.insert("refused".into(), Value::Array(l.refused.iter().map(|(k, w)| obj(vec![("key", s(k)), ("why", s(w))])).collect()));
            if let Some(p) = &st.piece {
                let (ver, refu) = nmlearn::library::counts(&rows, &p.identity());
                o.insert("evidence_here".into(), obj(vec![("hardware", s(&p.identity())), ("verified", n(ver as u64)), ("refuted", n(refu as u64)), ("ready", Value::Bool(nmlearn::library::ready(&rows, &p.identity())))]));
            }
        }
        return Ok(v);
    }
    let c = lib.cartridge(name).ok_or_else(|| format!("no cartridge or stack {name}"))?;
    let mut v = c.to_value();
    if let Value::Object(o) = &mut v {
        o.insert("type".into(), s("cartridge"));
        o.insert("label_text".into(), s(&c.label_text()));
        if c.kind == "tensors" {
            match lib.tensor_parts(c) {
                Ok(parts) => {
                    let layers: std::collections::BTreeSet<usize> = parts.iter().filter_map(|p| p.layer).collect();
                    o.insert("filter".into(), obj(vec![("tensors", n(parts.len() as u64)), ("bytes", n(parts.iter().map(|p| p.len).sum())), ("layers", n(layers.len() as u64))]));
                }
                Err(e) => {
                    o.insert("check".into(), s(&e.to_string()));
                }
            }
            o.insert("ledger_rows".into(), n(ledger.rows_for(&c.key()).len() as u64));
            return Ok(v);
        }
        match lib.filter(c) {
            Ok(Filter::Tensors { .. } | Filter::Open { .. }) => {}
            Ok(Filter::Settings(m)) => {
                o.insert("filter".into(), Value::Object(m.into_iter().map(|(k, v)| (k, s(&v))).collect()));
            }
            Ok(f @ Filter::Points { .. }) => {
                let pts = f.points();
                o.insert(
                    "filter".into(),
                    obj(vec![
                        ("points", strs(&pts.iter().map(|p| p.target.clone()).collect::<Vec<_>>())),
                        ("values", n(pts.iter().map(|p| p.gain.len() as u64 * 2).sum())),
                        ("encoding", s(if matches!(f, Filter::Points { q8: Some(_), .. }) { "q8" } else { "f32" })),
                    ]),
                );
            }
            Err(e) => {
                o.insert("check".into(), s(&e.to_string()));
            }
        }
        o.insert("ledger_rows".into(), n(ledger.rows_for(&c.key()).len() as u64));
    }
    Ok(v)
}

fn done(r: Result<Option<String>, String>, what: &str) -> ExitCode {
    match r {
        Ok(Some(id)) => {
            println!("{what} (audit {id})");
            ExitCode::SUCCESS
        }
        Ok(None) => {
            println!("no change");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("library: {e}");
            ExitCode::FAILURE
        }
    }
}

fn fail(e: impl std::fmt::Display) -> ExitCode {
    eprintln!("library: {e}");
    ExitCode::FAILURE
}

/// `neuromorph-nano library ...`.
#[must_use]
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    let json = args.iter().any(|a| a == "--json");
    let sub = args.get(1).map_or("list", String::as_str);
    let target = args.get(2).filter(|a| !a.starts_with("--")).cloned();
    // 2026-11-25: base cartridge versions and the release merge, before `open` (which makes a first
    // catalogue when there is none; the merge makes it from the release instead).
    match (sub, target.as_deref()) {
        ("merge", Some(f)) => {
            return match merge_release(home, Path::new(f), "installer") {
                Ok(lines) => {
                    for l in &lines {
                        println!("{l}");
                    }
                    if lines.is_empty() {
                        println!("library: nothing new in the release");
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => fail(e),
            };
        }
        ("bases", _) => {
            let v = base_versions(home);
            if json {
                println!("{}", nmjson::pretty::pretty(&v));
            } else {
                println!("upgrade policy ({UPGRADE_KEY}): {}", text(&v, "policy"));
                for m in v.get("models").and_then(Value::array).unwrap_or_default() {
                    println!("{}: serves {} (choice {})", text(m, "model"), text(m, "serving"), text(m, "choice"));
                    for x in m.get("versions").and_then(Value::array).unwrap_or_default() {
                        let refused = x.get("refused").and_then(Value::string).map_or(String::new(), |r| format!("  refused: {r}"));
                        let mark = if matches!(x.get("serving"), Ok(Value::Bool(true))) { "*" } else { " " };
                        println!("  {mark} {:<22} {:>6} MiB{}", text(x, "key"), x.get("bytes").and_then(Value::integer).unwrap_or(0) >> 20, refused);
                    }
                }
            }
            return ExitCode::SUCCESS;
        }
        ("base", Some(m)) => {
            let Some(choice) = args.get(3).filter(|a| !a.starts_with("--")) else { return fail(format!("base needs <model> <key|auto|open>\n{USAGE}")) };
            return done(choose_base(home, m, choice, "command").map(Some), &format!("{m}: {choice} from the next start"));
        }
        ("compat", _) => {
            println!("{}", nmjson::pretty::pretty(&compat(home)));
            return ExitCode::SUCCESS;
        }
        // An object's parameter overlay: read it, or edit, mask, unmask, freeze, unfreeze, clear, undo.
        ("overlay", _) => {
            return match crate::overlay::cli(home, args) {
                Ok(v) => {
                    println!("{}", nmjson::pretty::pretty(&v));
                    ExitCode::SUCCESS
                }
                Err(e) => fail(e),
            };
        }
        // The parameters of any object that holds them, and their edit (a new version).
        ("parameters" | "parameters-edit", _) => {
            // `--overview`: the whole-set overview, the same JSON as `GET /api/parameters/overview`.
            let read = if sub == "parameters" && args.iter().any(|a| a == "--overview") { crate::poverview::cli(home, args) } else { crate::cparams::run(home, args) };
            return match read {
                Ok(v) => {
                    println!("{}", nmjson::pretty::pretty(&v));
                    ExitCode::SUCCESS
                }
                Err(e) => fail(e),
            };
        }
        ("roles", _) => {
            println!("{}", nmjson::pretty::pretty(&roles_view(home)));
            return ExitCode::SUCCESS;
        }
        ("role", Some(r)) => {
            let Some(m) = args.get(3).filter(|a| !a.starts_with("--")) else { return fail(format!("role needs <role> <model id|default>
{USAGE}")) };
            return match choose_role(home, r, m, "command") {
                Ok((id, line)) => {
                    println!("{line} (audit {id})");
                    ExitCode::SUCCESS
                }
                Err(e) => fail(e),
            };
        }
        ("base-remove", Some(k)) => {
            if !args.iter().any(|a| a == "--confirm") {
                return fail(format!("base-remove deletes the cartridge file; add --confirm\n{USAGE}"));
            }
            return done(remove_base(home, k, "command").map(Some), &format!("{k}: removed"));
        }
        _ => {}
    }
    let lib = match open(home) {
        Ok(l) => l,
        Err(e) => return fail(e),
    };
    match (sub, target) {
        ("list", _) => {
            if json {
                println!("{}", nmjson::pretty::pretty(&lib.to_value()));
            } else {
                println!("stacks (cartridges plugged in for a use):");
                for st in &lib.stacks {
                    let piece = st.piece.as_ref().map_or_else(String::new, |p| format!(" for {}", p.identity()));
                    println!("  {:<40} {:<14}{}{}  [{}]", st.key(), st.use_for, if st.trial { " trial" } else { "" }, piece, st.cartridges.join(", "));
                }
                println!("cartridges:");
                for c in &lib.cartridges {
                    // A parameter cartridge is read and hashed when it is seated, not to list it.
                    let check = if c.kind == "tensors" {
                        if lib.files_present(c) { "on disk; parameters checked when seated".to_owned() } else { "REFUSED: its file is missing or not the size its record states".to_owned() }
                    } else {
                        match lib.filter(c) {
                            Ok(_) => "loads".to_owned(),
                            Err(e) => format!("REFUSED: {e}"),
                        }
                    };
                    print!("{}", c.label_text().lines().map(|l| format!("  {l}\n")).collect::<String>());
                    println!("    {}{}{}", if c.family.is_empty() { String::new() } else { format!("{}; ", c.family) }, check, if c.trial { "; on trial" } else { "" });
                }
                for (u, st) in &lib.active {
                    println!("switched: {u} = {st}");
                }
            }
            ExitCode::SUCCESS
        }
        ("rollback", Some(m)) => {
            let reason = args.get(3).filter(|a| !a.starts_with("--")).map_or("asked from the command", String::as_str);
            match rollback(home, &m, reason) {
                Ok(l) => {
                    println!("{l}");
                    ExitCode::SUCCESS
                }
                Err(e) => fail(e),
            }
        }
        ("show", Some(t)) => match show(&lib, &t, &LedgerEvidence::read(home)) {
            Ok(v) => {
                if !json && let Ok(l) = v.get("label_text").and_then(Value::string) {
                    print!("{l}");
                }
                println!("{}", nmjson::pretty::pretty(&v));
                ExitCode::SUCCESS
            }
            Err(e) => fail(e),
        },
        ("stack", only) => {
            let machine = match arg(args, "--machine") {
                Some(f) => match machine_file(Path::new(f)) {
                    Ok(m) => m,
                    Err(e) => return fail(e),
                },
                None => this_machine(home, true),
            };
            let u = in_use(&lib, machine);
            let applied = translate(&u.values(), &[]);
            if json {
                println!("{}", nmjson::pretty::pretty(&u.to_value(&applied)));
            } else {
                print!("{}", plain_in_use(&u, &applied, only.as_deref()));
                print!("{}", seats_text(home));
            }
            ExitCode::SUCCESS
        }
        ("switch", Some(t)) => {
            let what = args.get(3).cloned().unwrap_or_default();
            drop(lib);
            match what.as_str() {
                "trial" | "serve" => done(set_trial(home, &t, what == "trial", &LedgerEvidence::read(home), "command"), &format!("{t}: {what}")),
                "off" => done(switch_use(home, &t, None, "command"), &format!("{t}: off")),
                "" => fail(USAGE),
                stack => done(switch_use(home, &t, Some(stack), "command"), &format!("{t}: {stack}")),
            }
        }
        ("export", Some(t)) => {
            let Some(out) = arg(args, "--out") else { return fail(format!("export needs --out <file>\n{USAGE}")) };
            match lib.export_stack(&t) {
                Ok(bytes) => match std::fs::write(out, &bytes) {
                    Ok(()) => {
                        println!("exported {t} to {out} ({} bytes, SHA-256 {})", bytes.len(), nmhash::sha256_bytes(&bytes));
                        ExitCode::SUCCESS
                    }
                    Err(e) => fail(format!("{out}: {e}")),
                },
                Err(e) => fail(e),
            }
        }
        ("import", Some(f)) => {
            let bytes = match std::fs::read(&f) {
                Ok(b) => b,
                Err(e) => return fail(format!("{f}: {e}")),
            };
            drop(lib);
            let r = for_change(home).and_then(|(mut lib, _g)| {
                let (key, added) = lib.import(&bytes).map_err(|e| e.to_string())?;
                lib.save().map_err(|e| e.to_string())?;
                let id = crate::controls::audit(
                    home,
                    "library",
                    vec![("action", s("import")), ("stack", s(&key)), ("added", strs(&added)), ("hash", s(&nmhash::sha256_bytes(&bytes))), ("label", s(&format!("Stack {key} imported on trial")))],
                )?;
                Ok((key, added, id))
            });
            match r {
                Ok((key, added, id)) => {
                    println!("imported {key} on trial; cartridges added: {} (audit {id})", if added.is_empty() { "none".into() } else { added.join(", ") });
                    ExitCode::SUCCESS
                }
                Err(e) => fail(e),
            }
        }
        _ => fail(USAGE),
    }
}

/// Section check (temporary folders, no probe): the first content's values equal the code's and
/// add only options that restate the defaults; a command-line option wins; the ledger join counts
/// rows only for the piece a cartridge served and only when the machine tags agree.
pub fn check() -> Result<(), String> {
    if nmlearn::library::SOCKETS != nmengn::filter::SOCKETS {
        return Err("library: the cartridge file's socket contract differs from the engine's".into());
    }
    let dir = nmtype::check_dir("nmcore-library").map_err(|e| e.to_string())?;
    let mut lib = Library::open(&dir).map_err(|e| e.to_string())?;
    create_first(&mut lib)?;
    lib.save().map_err(|e| e.to_string())?;
    let u = in_use(&lib, Machine { pieces: vec![gtx_1650()], probed_ms: 0, how: "given".into() });
    if u.values() != gtx_values() {
        return Err("library: the GTX 1650 stack's settings differ from the code's with the base at 8 bits".into());
    }
    let applied = translate(&u.values(), &[]);
    let want_prec = [("base".to_owned(), "8".to_owned()), ("embedder".to_owned(), "8".to_owned()), ("reranker".to_owned(), "8".to_owned())];
    if !applied.refused.is_empty() || applied.precision != want_prec {
        return Err(format!("library: applying the first content refused {:?}, precision {:?}", applied.refused, applied.precision));
    }
    if !translate(&u.values(), &["--precision".into(), "16".into()]).precision.is_empty() {
        return Err("library: the settings cartridge's precision overrode --precision".into());
    }
    let mut got: Vec<(String, String)> = applied.args.chunks(2).map(|c| (c.first().cloned().unwrap_or_default(), c.get(1).cloned().unwrap_or_default())).collect();
    got.sort();
    let mut want: Vec<(String, String)> = [("--half-math", "on"), ("--segment-attention", "on"), ("--judge-window", "384"), ("--fast-judge-window", "off"), ("--look-window", "on"), ("--look-gate", "3"), ("--fast-k", "4")]
        .iter()
        .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
        .collect();
    want.sort();
    if got != want {
        return Err(format!("library: options {:?}", applied.args));
    }
    if translate(&u.values(), &["--judge-window".into(), "off".into()]).args.iter().any(|a| a == "384") {
        return Err("library: a filter overrode a command-line option".into());
    }
    // Ledger join, over rows chained as the ledger writes them.
    let led = dir.join("proj");
    std::fs::create_dir_all(&led).map_err(|e| e.to_string())?;
    let mut prev = "0".repeat(64);
    let mut body = String::new();
    for mut v in [
        obj(vec![("row", s("served")), ("pack", s("p1")), ("hardware", obj(vec![("gpu", s("NVIDIA GeForce GTX 1650")), ("gpu_mib", n(4096))]))]),
        obj(vec![("row", s("evidence")), ("pack", s("p1")), ("rule", s("edit")), ("state", s("verified")), ("strength", s("direct"))]),
    ] {
        if let Value::Object(o) = &mut v {
            o.insert("prev".into(), s(&prev));
        }
        let h = nmhash::sha256_bytes(v.encode().map_err(|e| e.to_string())?.as_bytes());
        if let Value::Object(o) = &mut v {
            o.insert("h".into(), s(&h));
        }
        body.push_str(&v.encode().map_err(|e| e.to_string())?);
        body.push('\n');
        prev = h;
    }
    let file = led.join(crate::ledger::FILE);
    std::fs::write(&file, body).map_err(|e| e.to_string())?;
    let gtx = gtx_1650().identity();
    let packs = BTreeMap::from([("p1".to_owned(), ("7".to_owned(), 200_u64))]);
    let start = |gpu: &str, mib: u64| vec![StartRow { pid: "7".into(), at: 100, loaded: vec![("op-gtx-1650-4gb@1".into(), gtx.clone())], gpu: Some((gpu.into(), mib)) }];
    let j = LedgerEvidence::join(&start("NVIDIA GeForce GTX 1650", 4096), &packs, &[("proj".into(), file.clone())]);
    if j.rows.len() != 1 || j.rows.first().is_none_or(|r| r.hardware != gtx || r.outcome != "verified") {
        return Err(format!("library: ledger join gave {} rows, skipped {:?}", j.rows.len(), j.skipped));
    }
    if !LedgerEvidence::join(&start("NVIDIA GeForce GTX 1660", 6144), &packs, &[("proj".into(), file)]).rows.is_empty() {
        return Err("library: ledger rows counted though the machine tags disagree".into());
    }
    Ok(())
}

/// Parameter tensors of `model` resident in this process, by source, from the engine's residency
/// accounting: (source, tensors counted by name, RAM bytes, card bytes).
fn resident_of(model: &str) -> Vec<(String, usize, u64, u64)> {
    let mut by: BTreeMap<String, (usize, u64, u64)> = BTreeMap::new();
    for r in nmengn::residency::get().parameters().into_iter().filter(|r| r.model.ends_with(model)) {
        let e = by.entry(r.source).or_default();
        e.0 += usize::from(!r.tensor.starts_with('('));
        e.1 += r.ram;
        e.2 += r.device;
    }
    by.into_iter().map(|(k, (a, b, c))| (k, a, b, c)).collect()
}

fn resident_line(v: &[(String, usize, u64, u64)]) -> String {
    if v.is_empty() {
        return "no parameter tensor resident".to_owned();
    }
    #[expect(clippy::cast_precision_loss, reason = "bytes for a report")]
    let m = |b: u64| b as f64 / 1_048_576.0;
    v.iter().map(|(src, k, ram, dev)| format!("{src}: {k} tensors, RAM {:.1} MiB, card {:.1} MiB", m(*ram), m(*dev))).collect::<Vec<_>>().join("; ")
}

/// Section check `nmcore.library.served`: the base cartridge each MicroModel serves is the one the
/// tracked record names (`designs/base-cartridges.json`, which the commits carry while `settings.json`
/// and `library/` are not tracked): on the shelf, its file's SHA-256 as recorded, its label's parent as
/// recorded, named in `settings.json` (or, with nothing named, the newest base cartridge of that
/// MicroModel, which `bases_of` offers first), and its compensation on the shelf with the recorded
/// SHA-256 and plugged by default exactly when the record says (`settings.json` `base_compensation`).
/// A machine without the library (a fresh clone) passes with a note. Report in
/// `tests/section-checks/nmcore-library-served/report.txt`.
pub fn check_served() -> Result<(), String> {
    let home = nmtype::nano_root();
    let rec = nmjson::parse(&std::fs::read(home.join("designs/base-cartridges.json")).map_err(|e| format!("designs/base-cartridges.json: {e}"))?).map_err(|e| format!("designs/base-cartridges.json: {e}"))?;
    let serving = rec.get("serving").and_then(Value::object).map_err(|e| format!("designs/base-cartridges.json serving: {e}"))?;
    let dir = nmtype::check_dir("nmcore-library-served").map_err(|e| e.to_string())?;
    let mut rep = String::new();
    if !nmlearn::library::catalogue_file(&home.join("library")).exists() {
        let _ = writeln!(rep, "no cartridge library on this machine: nothing seated to compare");
        return std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string());
    }
    let lib = Library::open(&home).map_err(|e| e.to_string())?;
    let settings = std::fs::read(home.join("settings.json")).ok().and_then(|b| nmjson::parse(&b).ok()).unwrap_or(Value::Null);
    let named = |k: &str, m: &str| settings.get(k).and_then(|o| o.get(m)).and_then(Value::string).ok().map(str::to_owned);
    let file_sha = |c: &nmlearn::library::Cartridge| -> Result<String, String> {
        let f = lib.cartridge_dir(c).join(c.files.first().map_or("", |f| f.name.as_str()));
        let r = std::fs::File::open(&f).map_err(|e| format!("{}: {e}", f.display()))?;
        nmhash::sha256(std::io::BufReader::new(r)).map_err(|e| format!("{}: {e}", f.display()))
    };
    let mut faults = Vec::new();
    for (model, row) in serving {
        let key = format!("{}@{}", text(row, "cartridge"), row.get("version").and_then(Value::integer).unwrap_or(0));
        let Some(c) = lib.cartridge(&key) else {
            faults.push(format!("{model}: {key} is not in the cartridge library"));
            continue;
        };
        if c.model != *model || c.kind != "tensors" {
            faults.push(format!("{model}: {key} is a {} cartridge of {}", c.kind, c.model));
        }
        let sha = file_sha(c)?;
        if sha != text(row, "sha256") {
            faults.push(format!("{model}: {key} file SHA-256 {sha:.12}, recorded {:.12}", text(row, "sha256")));
        }
        let parent = c.notes.get("parent").and_then(Value::string).map(str::to_owned).unwrap_or_default();
        if parent != text(row, "parent") {
            faults.push(format!("{model}: {key} label parent '{parent}', recorded '{}'", text(row, "parent")));
        }
        match named("base_cartridges", model) {
            Some(k) if k == key => {}
            Some(k) => faults.push(format!("{model}: settings.json names {k}, recorded {key}")),
            None => {
                let newest = lib.cartridges.iter().filter(|x| x.kind == "tensors" && x.model == *model).max_by(|a, b| (a.made.as_str(), a.version).cmp(&(b.made.as_str(), b.version))).map(nmlearn::library::Cartridge::key);
                if newest.as_deref() != Some(key.as_str()) {
                    faults.push(format!("{model}: settings.json names none and the newest in the cartridge library is {newest:?}, recorded {key}"));
                }
            }
        }
        let comp_on = named("base_compensation", model).as_deref() == Some("on");
        let mut comp_line = "no compensation".to_owned();
        if let Ok(comp) = row.get("compensation").and_then(Value::object) {
            let ckey = comp.get("cartridge").and_then(|v| v.string().ok()).unwrap_or("").to_owned();
            let want_on = matches!(comp.get("on"), Some(Value::Bool(true)));
            match lib.cartridge(&ckey) {
                None => faults.push(format!("{model}: compensation {ckey} is not in the cartridge library")),
                Some(cc) => {
                    let s2 = file_sha(cc)?;
                    let want = comp.get("sha256").and_then(|v| v.string().ok()).unwrap_or("");
                    if s2 != want {
                        faults.push(format!("{model}: compensation {ckey} SHA-256 {s2:.12}, recorded {want:.12}"));
                    }
                    if c.notes.get("compensation").and_then(Value::string).ok() != Some(ckey.as_str()) {
                        faults.push(format!("{model}: {key}'s label does not name {ckey}"));
                    }
                }
            }
            if want_on != comp_on {
                faults.push(format!("{model}: compensation recorded {}, settings.json has it {}", if want_on { "on" } else { "off" }, if comp_on { "on" } else { "off" }));
            }
            comp_line = format!("compensation {ckey} {}", if comp_on { "on" } else { "off" });
        }
        let _ = writeln!(rep, "{model}: {key} sha256 {sha:.16} parent '{parent}'; {comp_line}; backup commit '{}'", text(row, "backup_commit"));
    }
    for f in &faults {
        let _ = writeln!(rep, "FAULT {f}");
    }
    std::fs::write(dir.join("report.txt"), &rep).map_err(|e| e.to_string())?;
    if faults.is_empty() { Ok(()) } else { Err(format!("nmcore.library.served: {}", faults.join("; "))) }
}

/// Section check of a base cartridge in its final form, on Qwen3-Reranker-0.6B, on the processor
/// and on the card when one is present. A cartridge holding every parameter set the judge's
/// blocks read, at the open weights' values, each addressed to its layer and block, is written as
/// one labelled file with no structure. Refused by name: a set missing, a set whose label gives
/// another layer, a narrowed attention set, one made for other open weights, a changed value under
/// an exact label, and a file carrying an assembly. Seated over the open weights' model (the
/// connection check and the check run), the open weights released: no tensor of theirs resident
/// (counted by name through the engine's residency accounting) and every output (P(yes), the final
/// norm, layer 13's gated feed-forward) bit-identical. Pulled: the open weights back with their
/// exact outputs and no tensor of the cartridge resident. Stacking: two synthetic additional
/// cartridges (gain and bias at `L13.swigate` and `final_norm`), one and both in each order, give the
/// same outputs on the base cartridge as on the open weights, the two orders differ, pulling them
/// gives the base back exactly, and one made for another base is refused. RAM and card bytes in
/// each state and the times to seat and pull in
/// `tests/section-checks/nmcore-library-base/report.txt`.
#[expect(clippy::too_many_lines, reason = "one check, its cases in order")]
pub fn check_base() -> Result<(), String> {
    use nmengn::model::{Model, OPEN_WEIGHTS, Population, RunOpts, values};
    use nmengn::placement::Device;
    const MODEL: &str = "qwen3-reranker-0.6b";
    let home = nmtype::nano_root();
    if !nmtype::models_dir(&home).join(MODEL).exists() {
        return Ok(());
    }
    let dir = nmtype::check_dir("nmcore-library-base").map_err(|e| e.to_string())?;
    let asm = || nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-reranker").map_err(|e| e.to_string());
    let gpu = nmengn::device::open().is_ok();
    let base_pop = Population::open(&home, MODEL, false, false).map_err(|e| e.to_string())?;
    let pairs = [
        ("Where is the configuration file parsed?", "fn parse_config(path: &Path) -> Config {\n    toml::from_str(&std::fs::read_to_string(path).unwrap())\n}"),
        ("How is the head advanced?", "impl Ring { fn push(&mut self) { self.head += 1; } }"),
    ];
    let ids: Vec<Vec<i32>> = pairs
        .iter()
        .map(|(q, d)| base_pop.tokenizer.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, q, d)).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let nid = nmengn::assembly::nid;
    let outputs = |m: &Model, dev: Device| -> Result<Vec<Vec<u32>>, String> {
        let targets = vec![(m.role_node("label").map_err(|e| e.to_string())?, "y".to_owned()), (nid("final_norm"), "y".to_owned()), (nid("L13.swigate"), "y".to_owned())];
        let mut out = Vec::new();
        for p in &ids {
            let r = m.forward(dev, p, &targets, &RunOpts::default()).map_err(|e| e.to_string())?;
            for t in &targets {
                out.push(values(r.get(t).ok_or("missing output")?).map_err(|e| e.to_string())?.iter().map(|x| x.to_bits()).collect());
            }
        }
        Ok(out)
    };
    // Every parameter set the judge's blocks read, at the open weights' values, with its address.
    let needed = nmengn::base::read_by(&asm()?);
    let copy: Vec<nmlearn::library::Tensor> = base_pop
        .tensors
        .iter()
        .filter(|t| needed.contains(&t.name))
        .map(|t| {
            use std::io::{Read, Seek, SeekFrom};
            let (layer, block) = nmengn::base::address(&t.name).ok_or_else(|| format!("{}: no address", t.name))?;
            let mut f = std::fs::File::open(base_pop.card.dir.join(&t.file)).map_err(|e| e.to_string())?;
            f.seek(SeekFrom::Start(t.start)).map_err(|e| e.to_string())?;
            let mut bytes = vec![0_u8; usize::try_from(t.end - t.start).map_err(|e| e.to_string())?];
            f.read_exact(&mut bytes).map_err(|e| e.to_string())?;
            Ok(nmlearn::library::Tensor { name: t.name.clone(), layer, block: block.to_owned(), dtype: t.dtype.clone(), shape: t.shape.clone(), bytes })
        })
        .collect::<Result<_, String>>()?;
    let weights = base_pop.card.sha_main.clone();
    drop(base_pop);
    let total: usize = copy.iter().map(|t| t.bytes.len()).sum();
    let spec = CartridgeSpec {
        id: "judge-complete".into(),
        version: 1,
        family: String::new(),
        domain: String::new(),
        scope: String::new(),
        model: MODEL.into(),
        weights: weights.clone(),
        layers: Some((0, 27)),
        name: "check judge-complete".into(),
        description: "Every parameter set the reranker's blocks read, at the open weights' values, for the check.".into(),
        made_from: "the reranker's open weights, copied".into(),
        effect: String::new(),
    };
    let mut lib = Library::open(&dir).map_err(|e| e.to_string())?;
    let t = std::time::Instant::now();
    lib.add_cartridge(&spec, &Filter::Tensors(copy), obj(vec![("exact", Value::Bool(true))]), Vec::new(), false).map_err(|e| e.to_string())?;
    lib.save().map_err(|e| e.to_string())?;
    let c = lib.cartridge("judge-complete").ok_or("lost the cartridge")?;
    let file = lib.cartridge_dir(c).join(&c.files.first().ok_or("no file")?.name);
    let mut rep = format!(
        "nmcore.library.base: {} parameter sets ({total} bytes) of every block of {MODEL}'s 28 layers, the table and the final norm, written in {:.1} s\n",
        needed.len(),
        t.elapsed().as_secs_f64()
    );
    let offer = nmapi::neural::offer_of(&file).map_err(|e| e.to_string())?;
    // A file carrying structure is refused when it is read.
    let label = obj(vec![
        ("schema", s(nmlearn::library::CARTRIDGE_SCHEMA)),
        ("id", s("structure")),
        ("version", n(1)),
        ("model", s(MODEL)),
        ("weights", s(&weights)),
        ("name", s("structure")),
        ("does", s("carries an assembly")),
        ("made_from", s("the check")),
        ("made", s("2026-11-16")),
        ("parameters_sha256", s(&nmhash::sha256_bytes(b"{}"))),
        ("parts", Value::Array(vec![obj(vec![("form", s("assembly")), ("encoding", s("json")), ("at", n(0)), ("bytes", n(2))])])),
    ]);
    let asm_file = dir.join("structure.cartridge");
    std::fs::write(&asm_file, nmlearn::library::cartridge_file(&label, b"{}")).map_err(|e| e.to_string())?;
    match nmapi::neural::offer_of(&asm_file) {
        Err(e) if e.to_string().contains("never structure") => {
            let _ = writeln!(rep, "refused, a file carrying an assembly: {e}");
        }
        r => return Err(format!("nmcore.library.base: a file carrying an assembly was not refused for its structure: {:?}", r.map(|o| o.key))),
    }
    // Two synthetic additional cartridges (socket form, gain and bias at a node's output).
    #[expect(clippy::cast_precision_loss, reason = "a fixed pattern of values")]
    let wave = |w: usize, g: f32, b: f32, k: f32| nmengn::filter::Part {
        target: String::new(),
        gain: (0..w).map(|i| 1.0 + g * (i as f32 * k).sin()).collect(),
        bias: (0..w).map(|i| b * (i as f32 * k).cos()).collect(),
    };
    let part = |t: &str, p: nmengn::filter::Part| nmengn::filter::Part { target: t.to_owned(), ..p };
    let add_a = nmengn::filter::CartridgeFilter { cartridge: "synthetic-a@1".into(), parts: vec![part("L13.swigate", wave(3072, 0.02, 0.002, 0.37))] };
    let add_b = nmengn::filter::CartridgeFilter {
        cartridge: "synthetic-b@1".into(),
        parts: vec![part("L13.swigate", wave(3072, 0.03, 0.004, 0.11)), part("final_norm", wave(1024, 0.01, 0.003, 0.53))],
    };
    let mut devices = vec![(Device::Cpu, false)];
    if gpu {
        devices.push((Device::Gpu, true));
    }
    for (dev, on_card) in devices {
        let _ = writeln!(rep, "-- {}", dev.name());
        let t = std::time::Instant::now();
        let open_pop = Population::open(&home, MODEL, false, on_card).map_err(|e| e.to_string())?;
        let open = Model::bind(&open_pop, asm()?).map_err(|e| e.to_string())?.label_rows();
        let reference = outputs(&open, dev)?;
        let _ = writeln!(rep, "open weights in service ({:.2} s to open, bind and run): {}", t.elapsed().as_secs_f64(), resident_line(&resident_of(MODEL)));
        // Refusals by name; the open weights stay in service after each.
        let up13 = "model.layers.13.mlp.up_proj.weight";
        let mut bad_offers: Vec<(&str, nmengn::base::Offer)> = Vec::new();
        let mut o = offer.clone();
        o.parts.retain(|p| p.name != up13);
        bad_offers.push(("incomplete", o));
        let mut o = offer.clone();
        o.stated.insert("model.layers.5.self_attn.k_proj.weight".into(), (Some(6), "attention".into()));
        bad_offers.push(("misaddressed", o));
        let mut o = offer.clone();
        if let Some(p) = o.parts.iter_mut().find(|p| p.name == "model.layers.3.self_attn.q_proj.weight") {
            let row = (p.end - p.start) / p.shape.first().copied().unwrap_or(1).max(1) as u64;
            if let Some(r) = p.shape.first_mut() {
                *r -= 128;
            }
            p.end -= 128 * row;
        }
        bad_offers.push(("shape", o));
        let mut o = offer.clone();
        o.weights = "0".repeat(64);
        bad_offers.push(("weights", o));
        let mut o = offer.clone();
        let up14 = o.parts.iter().find(|p| p.name == "model.layers.14.mlp.up_proj.weight").map(|p| (p.start, p.end));
        if let (Some(p), Some((a, b))) = (o.parts.iter_mut().find(|p| p.name == up13), up14) {
            (p.start, p.end) = (a, b);
        }
        bad_offers.push(("check", o));
        for (want, o) in &bad_offers {
            match nmengn::base::seat(&open, o) {
                Err(e) if e.code() == *want => {
                    let _ = writeln!(rep, "refused ({want}): {e}");
                }
                Err(e) => return Err(format!("nmcore.library.base: the {want} case was refused as {}: {e}", e.code())),
                Ok(_) => return Err(format!("nmcore.library.base: the {want} case was seated; it must be refused")),
            }
        }
        if outputs(&open, dev)? != reference {
            return Err("nmcore.library.base: a refusal changed the open weights' outputs".into());
        }
        // Additional cartridges stacked on the open weights: one, and two in each order.
        let stacked = |m: &Model| -> Result<Vec<Vec<Vec<u32>>>, String> {
            let mut out = Vec::new();
            for order in [vec![&add_a], vec![&add_a, &add_b], vec![&add_b, &add_a]] {
                let cs: Vec<nmengn::filter::CartridgeFilter> = order.into_iter().cloned().collect();
                m.set_filters(Some(std::sync::Arc::new(nmengn::filter::build(MODEL, &cs)?)));
                out.push(outputs(m, dev)?);
            }
            m.set_filters(None);
            Ok(out)
        };
        let on_open = stacked(&open)?;
        if outputs(&open, dev)? != reference || on_open.first() == Some(&reference) || on_open.get(1) == on_open.get(2) {
            return Err("nmcore.library.base: on the open weights, an additional cartridge left the outputs unchanged, two orders agreed, or pulling did not give the base back".into());
        }
        // Seat: connection check, check run, then the open weights released in one step.
        let t = std::time::Instant::now();
        let (cart, seated) = nmengn::base::seat(&open, &offer).map_err(|e| e.to_string())?;
        drop(open);
        drop(open_pop);
        let seat_s = t.elapsed().as_secs_f64();
        let now = resident_of(MODEL);
        let open_left: usize = now.iter().filter(|r| r.0 == OPEN_WEIGHTS).map(|r| r.1).sum();
        if open_left > 0 {
            return Err(format!("nmcore.library.base: {open_left} tensors of the open weights still resident with the cartridge seated"));
        }
        if outputs(&cart, dev)? != reference {
            return Err("nmcore.library.base: the cartridge at the open weights' values changed an output".into());
        }
        let _ = writeln!(
            rep,
            "seated in {seat_s:.2} s (check run {} of {} outputs bit-identical), open weights released: {}; every output bit-identical",
            seated.identical,
            seated.outputs,
            resident_line(&now)
        );
        // The same additional cartridges stacked on the base cartridge give the same outputs; one
        // made for another base is refused.
        if stacked(&cart)? != on_open || outputs(&cart, dev)? != reference {
            return Err("nmcore.library.base: additional cartridges gave other outputs on the base cartridge than on the open weights, or pulling them did not give the base back".into());
        }
        nmengn::filter::fits(&home, MODEL, &weights, nmengn::filter::SOCKETS, &["L13.swigate", "final_norm"]).map_err(|e| format!("nmcore.library.base: {e}"))?;
        let other = nmengn::filter::fits(&home, MODEL, &"ab".repeat(32), nmengn::filter::SOCKETS, &["final_norm"]).err().unwrap_or_default();
        if !other.contains("made for base") {
            return Err(format!("nmcore.library.base: an additional cartridge made for another base was not refused ({other})"));
        }
        let _ = writeln!(
            rep,
            "stacking: one additional cartridge and two in each order on the base cartridge give the same outputs as on the open weights (the two orders differ); pulled, the base's outputs exactly; made for another base refused: {other}"
        );
        // Pull: the open weights back, exactly.
        let t = std::time::Instant::now();
        drop(cart);
        nmengn::base::pull_base(MODEL);
        let back_pop = Population::open(&home, MODEL, false, on_card).map_err(|e| e.to_string())?;
        let back = Model::bind(&back_pop, asm()?).map_err(|e| e.to_string())?.label_rows();
        let pull_s = t.elapsed().as_secs_f64();
        let now = resident_of(MODEL);
        let cart_left: usize = now.iter().filter(|r| r.0 != OPEN_WEIGHTS).map(|r| r.1).sum();
        if cart_left > 0 || outputs(&back, dev)? != reference {
            return Err(format!("nmcore.library.base: pulling left {cart_left} cartridge tensors resident or changed an output"));
        }
        let _ = writeln!(rep, "pulled in {pull_s:.2} s (bind; first run after): {}; every output bit-identical", resident_line(&resident_of(MODEL)));
    }
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Home(std::path::PathBuf);
    impl Drop for Home {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn home(name: &str) -> Home {
        let d = std::env::temp_dir().join(format!("nano-library-undo-{name}-{}-{}", std::process::id(), nmtype::now_ms()));
        std::fs::create_dir_all(d.join("memory")).unwrap();
        std::fs::write(d.join("settings.json"), b"{\"memory\":\"on\"}").unwrap();
        Home(d)
    }
    fn settings(h: &Home) -> Value {
        nmjson::parse(&std::fs::read(h.0.join("settings.json")).unwrap()).unwrap()
    }
    fn row(h: &Home, id: &str) -> Value {
        let all = std::fs::read_to_string(h.0.join("memory/audit.jsonl")).unwrap();
        all.lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).find(|v| text(v, "id") == id).unwrap()
    }

    /// A base cartridge choice keeps the choice before and after; the Audit undo writes the one
    /// before back through the same writer, logged as an undo of that row, and keeps other keys.
    #[test]
    fn base_choice_is_undone_through_its_writer() {
        let h = home("base");
        let model = nmtype::role_models::ROLES[1].1;
        let first = choose_base(&h.0, model, "open", "test").unwrap();
        let r = row(&h, &first);
        assert_eq!((text(&r, "action"), text(&r, "from"), text(&r, "to")), ("choose".to_owned(), "auto".to_owned(), "open".to_owned()));
        assert_eq!(settings(&h).get("base_cartridges").and_then(|b| b.get(model)).and_then(Value::string).ok(), Some("open"));
        let undo = crate::controls::undo_by(&h.0, &first, "test").unwrap().unwrap();
        let u = row(&h, &undo);
        assert_eq!((text(&u, "kind"), text(&u, "undo_of"), text(&u, "from"), text(&u, "to")), ("library-base".to_owned(), first, "open".to_owned(), "auto".to_owned()));
        let after = settings(&h);
        assert!(after.get("base_cartridges").and_then(|b| b.get(model)).is_err(), "auto removes the named choice");
        assert_eq!(after.get("memory").and_then(Value::string).ok(), Some("on"));
        // A row logged before the choice before was kept is refused with that reason.
        let old = crate::controls::audit(&h.0, "library-base", vec![("action", s("choose")), ("model", s(model)), ("choice", s("open")), ("by", s("test"))]).unwrap();
        assert!(crate::controls::undo_by(&h.0, &old, "test").unwrap_err().contains("without the choice before"));
    }

    /// A role's model change keeps the model before and after; the Audit undo puts the role back on
    /// the model before through the same writer, logged as an undo of that row.
    #[test]
    fn role_change_is_undone_through_its_writer() {
        let h = home("role");
        let (reader, writer) = (nmtype::role_models::ROLES[0].1, nmtype::role_models::ROLES[1].1);
        for id in [reader, writer] {
            std::fs::create_dir_all(nmtype::models_dir(&h.0).join(id)).unwrap();
            std::fs::write(nmtype::models_dir(&h.0).join(id).join("part.json"), b"{}").unwrap();
        }
        // The writer may share the reader's model.
        let (first, _) = choose_role(&h.0, "writer", reader, "test").unwrap();
        let r = row(&h, &first);
        assert_eq!((text(&r, "action"), text(&r, "from"), text(&r, "to")), ("choose".to_owned(), writer.to_owned(), reader.to_owned()));
        assert_eq!(nmtype::role_models::resolve(&h.0).get("writer").map(String::as_str), Some(reader));
        let undo = crate::controls::undo_by(&h.0, &first, "test").unwrap().unwrap();
        let u = row(&h, &undo);
        assert_eq!((text(&u, "kind"), text(&u, "undo_of"), text(&u, "from"), text(&u, "to")), ("role-model".to_owned(), first, reader.to_owned(), writer.to_owned()));
        assert_eq!(nmtype::role_models::resolve(&h.0).get("writer").map(String::as_str), Some(writer));
        assert!(settings(&h).get("role_models").and_then(|m| m.get("writer")).is_err(), "the default model is not written");
    }
}
