//! The four MicroModels at serve start (2026-11-27, Luis Jake Gabriel III: "the default is that all
//! four micromodels are loaded at start up", dynamic when the card cannot hold them). After the
//! tools are installed, one background thread opens the Reader, the Embedder and the Reranker
//! through their slots (`neural::model`, `embedder_model`, `reranker_model`), then seats the
//! Writer beside them when the card's effective free memory (`limits.vram_mb`, `auto` = the
//! driver's figure) holds it with the live calls' room (`writer_idle::seat_at_start`). A call that
//! needs a MicroModel still loading waits for that MicroModel only: each slot is opened once,
//! under its own lock, and a second caller takes the same model. A MicroModel that does not fit
//! stays with the existing residency rules (the Writer loads per idle session and stays seated
//! once room frees), and `nm_status` says which and why (`view`).
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

/// The roles in load order: the finders first (every pack uses them), the Writer last.
pub const ROLES: [&str; 4] = ["reader", "embedder", "reranker", "writer"];

#[derive(Clone, Debug, Default)]
struct Seat {
    state: &'static str,
    device: String,
    card_mb: u64,
    ms: u64,
    note: String,
}

static SEATS: Mutex<BTreeMap<&'static str, Seat>> = Mutex::new(BTreeMap::new());
static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn set(role: &'static str, seat: Seat) {
    if let Ok(mut g) = SEATS.lock() {
        g.insert(role, seat);
    }
}

/// Puts a MicroModel's weights on the card now (2026-11-27), as a background job behind the
/// device gate, so the first call that uses it finds them seated instead of uploading them (a
/// MicroModel opens with its weights in RAM and uploads them at its first run otherwise). Returns
/// the tensors staged; nothing without a card.
/// # Errors
/// The copy stream or an upload failed.
pub fn stage(m: &nmengn::Model) -> Result<usize, String> {
    let Some(g) = m.pop.gpu.as_ref() else { return Ok(0) };
    let _gate = nmengn::placement::gpu_gate().enter("MicroModel staging at start (background)", false);
    let st = g.gpu.copy_stream().map_err(|e| e.to_string())?;
    // Weight placement (2026-11-27): staging drops the host copies of the linear maps the card now
    // holds (`Model::drop_host`); the weights area of host memory is told what stays.
    let r = m.stage_parts(&st, crate::joint_read::full_head()).map(|s| s.tensors).map_err(|e| e.to_string());
    note_weights();
    r
}

/// The three held finding and ranking MicroModels by role (the seated Writer is read under its
/// own lock, `writer_idle::resident_writer_view`, never held here).
fn held_models() -> Vec<(&'static str, std::sync::Arc<nmengn::Model>)> {
    crate::neural::bound_models().into_iter().map(|(slot, m)| (if slot == "base" { "reader" } else { slot }, m)).collect()
}

/// The host memory of one MicroModel's weights (2026-11-27, weight placement): RAM bytes (tensor
/// bytes in RAM plus the 8-bit packets kept), `dropped` or `resident`, and why.
fn host_of(m: &nmengn::Model) -> (u64, &'static str, String) {
    let (t, q) = m.pop.host_bytes();
    let (state, why) = m.pop.host_state();
    (t + q, state, why)
}

/// Tells the weights area of host memory (`nmengn::limits::area_budget("weights")`) what the held
/// MicroModels' weights keep in RAM now, and returns that figure in bytes.
pub fn note_weights() -> u64 {
    let mut seen: Vec<*const nmengn::model::Population> = Vec::new();
    let mut total = 0;
    for (_, m) in held_models() {
        let p = std::sync::Arc::as_ptr(&m.pop);
        if !seen.contains(&p) {
            seen.push(p);
            total += host_of(&m).0;
        }
    }
    total += crate::writer_idle::resident_writer_view(|w| host_of(w).0).unwrap_or(0);
    nmengn::limits::set_area_used("weights", "micromodels", total);
    total
}

fn seat_of(m: &nmengn::Model, t: Instant) -> Seat {
    if let Err(e) = stage(m) {
        eprintln!("micromodels: {} not staged on the card at start ({e}); it uploads at its first run", m.pop.card.id);
    }
    let on_card = m.pop.gpu.is_some();
    Seat {
        state: "held",
        device: if on_card { "card".to_owned() } else { "processor".to_owned() },
        card_mb: (m.pop.device_bytes() >> 20) as u64,
        ms: u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX),
        note: if on_card { m.pop.gpu_note.clone() } else { format!("on the processor: {}", m.pop.gpu_note) },
    }
}

/// Starts loading the four MicroModels in the background (once per process). Returns false when
/// it had already started.
pub fn start(home: PathBuf) -> bool {
    if STARTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return false;
    }
    for role in ROLES {
        set(role, Seat { state: "waiting", note: "loads at start, after the MicroModels before it".to_owned(), ..Seat::default() });
    }
    std::thread::spawn(move || {
        let clock = Instant::now();
        for role in ROLES {
            set(role, Seat { state: "loading", note: "loading at start".to_owned(), ..Seat::default() });
            let t = Instant::now();
            let seat = match role {
                "reader" => crate::neural::model(&home).map(|m| seat_of(&m, t)).map_err(|e| e.to_string()),
                "embedder" => crate::neural::embedder_model(&home).map(|m| seat_of(&m, t)).map_err(|e| e.to_string()),
                "reranker" => crate::neural::reranker_model(&home).map(|m| seat_of(&m, t)).map_err(|e| e.to_string()),
                _ => crate::writer_idle::seat_at_start(&home).map(|(held, card_mb, note)| Seat {
                    state: if held { "held" } else { "per idle session" },
                    device: if held { "card".to_owned() } else { String::new() },
                    card_mb,
                    ms: u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX),
                    note,
                }),
            };
            match seat {
                Ok(s) => set(role, s),
                Err(e) => set(role, Seat { state: "not loaded", note: e, ..Seat::default() }),
            }
        }
        let v = view();
        eprintln!(
            "micromodels: {} at start in {:.2} s",
            v.get("held").and_then(Value::string).unwrap_or_default(),
            clock.elapsed().as_secs_f64()
        );
    });
    true
}

/// Whether the start's loading has been asked for in this process.
#[must_use]
pub fn started() -> bool {
    STARTED.load(std::sync::atomic::Ordering::SeqCst)
}

/// The four MicroModels now, for `nm_status` and the live view: per role its state (`waiting`,
/// `loading`, `held`, `per idle session` or `not loaded`), device, card bytes, load time and the
/// reason in words, and `held` as "N of 4" (the Writer counts while it is seated, whether at
/// start or by an idle session since).
#[must_use]
pub fn view() -> Value {
    let s = |t: &str| Value::String(t.to_owned());
    let n = |x: u64| Value::Number(x.to_string());
    let seats = SEATS.lock().map(|g| g.clone()).unwrap_or_default();
    // 2026-11-27: the writer an idle session runs counts as held while it runs, as the kept one does.
    let writer_seated = crate::writer_idle::writer_on_card_view(|w| (w.pop.device_bytes() >> 20) as u64);
    let mut held = 0;
    let mut roles = BTreeMap::new();
    for role in ROLES {
        let mut seat = seats.get(role).cloned().unwrap_or_else(|| Seat {
            state: "not asked",
            note: "this process did not load the MicroModels at start (not a serving process)".to_owned(),
            ..Seat::default()
        });
        // The card bytes now (the weights a MicroModel holds on the card change as the
        // residency rules release and restore them), not the figure at load.
        let mut host = None;
        if let Some((_, m)) = held_models().into_iter().find(|(r, _)| *r == role) {
            seat.card_mb = (m.pop.device_bytes() >> 20) as u64;
            host = Some(host_of(&m));
        }
        if role == "writer" {
            host = crate::writer_idle::writer_on_card_view(host_of);
        }
        if role == "writer"
            && let Some(mb) = writer_seated
        {
            if seat.state != "held" {
                seat.note = "seated by an idle writer session once the card had room".to_owned();
            }
            seat.state = "held";
            seat.device = "card".to_owned();
            seat.card_mb = mb;
        } else if role == "writer" && seat.state == "held" {
            // Released since (the card ran short and the residency rules evicted it).
            seat.state = "per idle session";
            seat.note = "released when the card ran short; loads per idle session and stays seated again once room frees".to_owned();
        }
        if seat.state == "held" {
            held += 1;
        }
        // Weight placement (2026-11-27): where the weights are, card and RAM, and whether the host
        // copy of the large tensors is dropped (read again from the files on demand) or resident.
        let (ram, host_copy, host_note) = host.unwrap_or((0, "none", "not held in this process".to_owned()));
        roles.insert(
            role.to_owned(),
            Value::Object(
                [
                    ("state".to_owned(), s(seat.state)),
                    ("device".to_owned(), s(&seat.device)),
                    ("card_mb".to_owned(), n(seat.card_mb)),
                    ("ram_mb".to_owned(), n(ram >> 20)),
                    ("host_copy".to_owned(), s(host_copy)),
                    ("host_note".to_owned(), s(&host_note)),
                    ("load_ms".to_owned(), n(seat.ms)),
                    ("note".to_owned(), s(&seat.note)),
                ]
                .into_iter()
                .collect(),
            ),
        );
    }
    let mut out = BTreeMap::new();
    out.insert("held".to_owned(), s(&format!("{held} of 4")));
    out.insert("roles".to_owned(), Value::Object(roles));
    // The weights' RAM over all held MicroModels, against the weights area of host memory
    // (`limits.memory.weights_mb`, or its share of the process budget on auto).
    out.insert("weights_ram_mb".to_owned(), n(note_weights() >> 20));
    out.insert("weights_area_mb".to_owned(), nmengn::limits::area_budget("weights").map_or(Value::Null, |b| n(b >> 20)));
    Value::Object(out)
}
