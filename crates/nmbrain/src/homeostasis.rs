//! Homeostasis (guide 13, 12; PLAN 4c): after every frame, read what each node did (faults,
//! latency) against `designs/homeostasis.json` and act by reflex: quarantine a node that
//! faults again and again, throttle a slow one, suspend under memory pressure. The brain
//! decides; the organism carries the action out through RNAM.
use nmjson::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    Quarantine(String),
    Throttle(String),
    Suspend(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Thresholds {
    pub latency_ms_max: f32,
    pub anomaly_rate_max: f32,
    pub faults_to_quarantine: u32,
    pub ram_mb_max: f32,
    pub gpu_mb_max: f32,
}
impl Thresholds {
    /// From `{"thresholds": {...}}`; two faults in a row quarantine (reflex default).
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let t = v.get("thresholds").ok();
        let f = |k: &str, d: f32| {
            t.and_then(|t| t.get(k).ok()).and_then(|x| if let Value::Number(n) = x { n.parse().ok() } else { None }).unwrap_or(d)
        };
        Self {
            latency_ms_max: f("latency_ms_max", 5000.0),
            anomaly_rate_max: f("anomaly_rate_max", 0.25),
            faults_to_quarantine: 2,
            ram_mb_max: f("ram_mb_max", 24000.0),
            gpu_mb_max: f("gpu_mb_max", 3500.0),
        }
    }
}

/// Per-node fault streaks and the actions they led to.
#[derive(Debug)]
pub struct Homeostasis {
    pub thresholds: Thresholds,
    streak: BTreeMap<String, u32>,
    pub actions: Vec<Action>,
}
impl Homeostasis {
    #[must_use]
    pub fn new(thresholds: Thresholds) -> Self {
        Self { thresholds, streak: BTreeMap::new(), actions: Vec::new() }
    }
    /// One frame's outcome: the nodes that faulted and the nodes that ran cleanly, each
    /// node's latency in ms, and the resident memory of every suspendable holder (id, RAM MB,
    /// GPU MB; the slot engines' parameter views from NPCS). Returns the actions triggered.
    pub fn observe(
        &mut self,
        faulted: &[String],
        clean: &[String],
        latency_ms: &BTreeMap<String, f32>,
        resident: &[(String, f32, f32)],
    ) -> Vec<Action> {
        let mut out = Vec::new();
        for id in faulted {
            let n = self.streak.entry(id.clone()).or_default();
            *n += 1;
            if *n == self.thresholds.faults_to_quarantine {
                out.push(Action::Quarantine(id.clone()));
            }
        }
        for id in clean {
            self.streak.remove(id);
        }
        for (id, ms) in latency_ms {
            if *ms > self.thresholds.latency_ms_max {
                out.push(Action::Throttle(id.clone()));
            }
        }
        // Memory pressure: suspend the largest holder on the device over its budget (guide
        // 13: free memory by suspending idle slots; designs/homeostasis.json).
        let over = |pick: fn(&(String, f32, f32)) -> f32, max: f32| -> Option<String> {
            let total: f32 = resident.iter().map(pick).sum();
            if total <= max {
                return None;
            }
            resident.iter().max_by(|a, b| pick(a).total_cmp(&pick(b))).map(|r| r.0.clone())
        };
        for id in [over(|r| r.1, self.thresholds.ram_mb_max), over(|r| r.2, self.thresholds.gpu_mb_max)].into_iter().flatten() {
            if !out.contains(&Action::Suspend(id.clone())) {
                out.push(Action::Suspend(id));
            }
        }
        self.actions.extend(out.iter().cloned());
        out
    }
    /// Forget a node's streak (after it was restarted).
    pub fn reset(&mut self, id: &str) {
        self.streak.remove(id);
    }
}

/// Self-check of the `homeostasis` sub-component: two faults in a row quarantine, a clean
/// run in between resets the streak, a slow node is throttled, the largest holder over the
/// GPU budget is suspended.
pub(crate) fn check() -> Result<(), String> {
    let mut h = Homeostasis::new(Thresholds::from_value(&Value::Null));
    let none = BTreeMap::new();
    let a = vec!["d#2".to_owned()];
    if !h.observe(&a, &[], &none, &[]).is_empty()
        || !h.observe(&[], &a, &none, &[]).is_empty()
        || !h.observe(&a, &[], &none, &[]).is_empty()
    {
        return Err("nmbrain.homeostasis: quarantined too early".into());
    }
    if h.observe(&a, &[], &none, &[]) != [Action::Quarantine("d#2".into())] {
        return Err("nmbrain.homeostasis: two faults in a row did not quarantine".into());
    }
    let slow = BTreeMap::from([("t#1".to_owned(), 9000.0)]);
    if h.observe(&[], &[], &slow, &[]) != [Action::Throttle("t#1".into())] {
        return Err("nmbrain.homeostasis: slow node not throttled".into());
    }
    let held = [("slot#embedder".to_owned(), 100.0, 3000.0), ("slot#reranker".to_owned(), 100.0, 900.0)];
    if h.observe(&[], &[], &none, &held) != [Action::Suspend("slot#embedder".into())] {
        return Err("nmbrain.homeostasis: GPU pressure did not suspend the largest holder".into());
    }
    Ok(())
}

// ---- Homeostasis of a call's time (lane T1, 2026-11-19; Luis 2026-11-18 08:38, item 4) ----

/// The optional card stages of one served call that spend its time budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    /// The tentacles' integration read.
    Read,
    /// G2's routed duels (the judge on prose, the reader on split ties).
    Duels,
    /// The close-call judge on the fused leader (W4) and its sufficiency read.
    Judge,
    /// The Core Brain's intake chain.
    Chain,
    /// The tentacle core's deep check (one joint read with three content-free forms a tentacle).
    Deep,
}

/// One call's time budget, spent by expected value: each stage's measured cost (an exponential
/// mean over the process's calls, seconds) and its expected value (the share of calls on which it
/// changed the served order, a running mean). A stage may run while the time left, less what the
/// later stages of higher value per second are expected to need, covers its own expected cost.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimeBudget {
    /// Per stage: (mean cost s, mean value 0 to 1, calls seen).
    pub stats: std::collections::BTreeMap<Stage, (f64, f64, u64)>,
}

/// The stated weight of a new observation in the running means (about the last ten calls).
pub const BUDGET_EMA: f64 = 0.1;
/// Stated priors before a stage was measured (seconds, value): from the 2026-11-18 card timings
/// (read 0.35 s, prose duels 0.9 s, judge 0.75 s, chain 0.01 s) and the measured gains (judge
/// +7 first on 148 fired of 359; read and duels about 1 in 50; chain unmeasured).
pub const BUDGET_PRIORS: [(Stage, f64, f64); 5] =
    [(Stage::Read, 0.35, 0.02), (Stage::Duels, 0.9, 0.02), (Stage::Judge, 0.75, 0.05), (Stage::Chain, 0.01, 0.02), (Stage::Deep, 0.4, 0.02)];


impl TimeBudget {
    fn get(&self, s: Stage) -> (f64, f64) {
        self.stats.get(&s).map_or_else(|| BUDGET_PRIORS.iter().find(|p| p.0 == s).map_or((0.0, 0.0), |p| (p.1, p.2)), |x| (x.0, x.1))
    }
    /// Record one run of stage `s`: its seconds and whether it changed the served order.
    pub fn observe(&mut self, s: Stage, secs: f64, changed: bool) {
        let (c0, v0) = self.get(s);
        let e = self.stats.entry(s).or_insert((c0, v0, 0));
        e.0 = (1.0 - BUDGET_EMA) * e.0 + BUDGET_EMA * secs.max(0.0);
        e.1 = (1.0 - BUDGET_EMA) * e.1 + BUDGET_EMA * if changed { 1.0 } else { 0.0 };
        e.2 += 1;
    }
    /// Whether stage `s` may run with `left` seconds of the call: its expected cost fits after
    /// reserving the expected cost of each stage in `later` whose value per second is higher.
    #[must_use]
    pub fn allow(&self, s: Stage, left: f64, later: &[Stage]) -> bool {
        let (c, v) = self.get(s);
        let rate = |c: f64, v: f64| if c > 0.0 { v / c } else { f64::INFINITY };
        let reserve: f64 = later.iter().map(|x| self.get(*x)).filter(|(lc, lv)| rate(*lc, *lv) > rate(c, v)).map(|(lc, _)| lc).sum();
        left - reserve >= c
    }
}

/// The process's call time budget (one per process, shared by the served calls).
pub static TIME_BUDGET: std::sync::Mutex<TimeBudget> = std::sync::Mutex::new(TimeBudget { stats: std::collections::BTreeMap::new() });
