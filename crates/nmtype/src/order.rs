//! Long orders of the motor division (2026-11-12; research file Solution 3, Motor, "THE
//! SOLUTION"; open-source reference ROS 2 actions, https://design.ros2.org/articles/actions.html:
//! goal, feedback, result and cancel on a channel of their own, with a goal id and a lifecycle).
//!
//! An order that takes time (focus, re-window, joint read, run chain) carries a goal id and a
//! state. Its legal life: `Accepted` or `Rejected` when received; `Executing` once started;
//! `Succeeded` or `Aborted` when done; `Cancelled` from `Accepted` or `Executing` when the sender
//! withdraws it. Feedback lines may be added while executing. Every order carries an expiry (a
//! command from above is valid for a bounded time) and is always copied to G1, the only path
//! between parts. Illegal transitions are refused by name, never applied silently.
use crate::{Fault, FaultKind};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmtype.order", FaultKind::Interface, m)
}

/// The verbs of long orders.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verb {
    Focus,
    Rewindow,
    JointRead,
    RunChain,
}
impl Verb {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Focus => "focus",
            Self::Rewindow => "re-window",
            Self::JointRead => "joint-read",
            Self::RunChain => "run-chain",
        }
    }
}

/// Lifecycle states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum State {
    Pending,
    Accepted,
    Rejected,
    Executing,
    Succeeded,
    Aborted,
    Cancelled,
}
impl State {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
            Self::Executing => "executing",
            Self::Succeeded => "succeeded",
            Self::Aborted => "aborted",
            Self::Cancelled => "cancelled",
        }
    }
    /// Whether the order has ended.
    #[must_use]
    pub const fn done(self) -> bool {
        matches!(self, Self::Rejected | Self::Succeeded | Self::Aborted | Self::Cancelled)
    }
}

/// One long order.
#[derive(Clone, Debug, PartialEq)]
pub struct Order {
    pub goal: u64,
    pub verb: Verb,
    pub sender: String,
    pub receiver: String,
    /// Always true: every order is copied to G1.
    pub copy_to_g1: bool,
    pub issued_ms: u64,
    pub expires_ms: u64,
    pub state: State,
    pub feedback: Vec<String>,
    /// The result line when done (or the reason when rejected, aborted or cancelled).
    pub result: String,
}

impl Order {
    /// A new order, pending, valid for `ttl_ms` from `now_ms`.
    #[must_use]
    pub fn new(goal: u64, verb: Verb, sender: &str, receiver: &str, now_ms: u64, ttl_ms: u64) -> Self {
        Self {
            goal,
            verb,
            sender: sender.into(),
            receiver: receiver.into(),
            copy_to_g1: true,
            issued_ms: now_ms,
            expires_ms: now_ms.saturating_add(ttl_ms),
            state: State::Pending,
            feedback: Vec::new(),
            result: String::new(),
        }
    }
    fn to(&mut self, next: State, allowed: &[State], note: &str) -> Result<(), Fault> {
        if !allowed.contains(&self.state) {
            return Err(bad(format!("goal {}: {} to {} is not a legal transition", self.goal, self.state.as_str(), next.as_str())));
        }
        self.state = next;
        if !note.is_empty() {
            note.clone_into(&mut self.result);
        }
        Ok(())
    }
    /// The receiver takes the order (refused when it has expired).
    /// # Errors
    /// Not pending, or expired at `now_ms`.
    pub fn accept(&mut self, now_ms: u64) -> Result<(), Fault> {
        if self.expired(now_ms) {
            self.to(State::Rejected, &[State::Pending], "expired before it was taken")?;
            return Err(bad(format!("goal {}: expired", self.goal)));
        }
        self.to(State::Accepted, &[State::Pending], "")
    }
    /// # Errors
    /// Not pending.
    pub fn reject(&mut self, why: &str) -> Result<(), Fault> {
        self.to(State::Rejected, &[State::Pending], why)
    }
    /// # Errors
    /// Not accepted.
    pub fn start(&mut self) -> Result<(), Fault> {
        self.to(State::Executing, &[State::Accepted], "")
    }
    /// Progress while executing.
    /// # Errors
    /// Not executing.
    pub fn progress(&mut self, line: &str) -> Result<(), Fault> {
        if self.state != State::Executing {
            return Err(bad(format!("goal {}: feedback while {}", self.goal, self.state.as_str())));
        }
        self.feedback.push(line.to_owned());
        Ok(())
    }
    /// # Errors
    /// Not executing.
    pub fn succeed(&mut self, result: &str) -> Result<(), Fault> {
        self.to(State::Succeeded, &[State::Executing], result)
    }
    /// # Errors
    /// Not executing.
    pub fn abort(&mut self, why: &str) -> Result<(), Fault> {
        self.to(State::Aborted, &[State::Executing], why)
    }
    /// The sender withdraws the order.
    /// # Errors
    /// Already done or never accepted.
    pub fn cancel(&mut self, why: &str) -> Result<(), Fault> {
        self.to(State::Cancelled, &[State::Accepted, State::Executing], why)
    }
    #[must_use]
    pub fn expired(&self, now_ms: u64) -> bool {
        now_ms > self.expires_ms
    }
    /// One-line record (ids and words of the lifecycle only).
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "order goal {} {} {} -> {} copy-to-G1 {} state {} feedback {} result {}",
            self.goal,
            self.verb.as_str(),
            self.sender,
            self.receiver,
            self.copy_to_g1,
            self.state.as_str(),
            self.feedback.len(),
            self.result
        )
    }
}

/// A gain set from above (2026-11-12; research file Solution 3, Motor: "gains arrive as settings
/// from G2 with an expiry"; nature: reflex gain held from above, Chen and Wolpaw 2002). A reflex
/// reads its gain each time it fires; after the expiry the gain is 1 again.
#[derive(Clone, Debug, PartialEq)]
pub struct Gain {
    pub name: String,
    pub value: f32,
    pub set_by: String,
    pub expires_ms: u64,
}

/// The gains in force, by name.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gains {
    pub set: Vec<Gain>,
}
impl Gains {
    /// Set (or replace) a gain for `ttl_ms` from `now_ms`. Gains are bounded to 0.25 to 4.
    pub fn set(&mut self, name: &str, value: f32, set_by: &str, now_ms: u64, ttl_ms: u64) {
        self.set.retain(|g| g.name != name);
        self.set.push(Gain {
            name: name.into(),
            value: value.clamp(0.25, 4.0),
            set_by: set_by.into(),
            expires_ms: now_ms.saturating_add(ttl_ms),
        });
    }
    /// The gain in force for `name` at `now_ms` (1 when unset or expired).
    #[must_use]
    pub fn get(&self, name: &str, now_ms: u64) -> f32 {
        self.set.iter().find(|g| g.name == name && now_ms <= g.expires_ms).map_or(1.0, |g| g.value)
    }
}

/// Section check: the legal life runs; illegal transitions are refused; an expired order is
/// rejected on acceptance; cancel works from accepted and executing only.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let mut o = Order::new(1, Verb::JointRead, "nmbrain.g2", "nmbrain.core-brain", 1_000, 500);
    o.accept(1_100).map_err(|e| e.to_string())?;
    o.start().map_err(|e| e.to_string())?;
    o.progress("read 3 pieces").map_err(|e| e.to_string())?;
    o.succeed("moved first place").map_err(|e| e.to_string())?;
    if o.state != State::Succeeded || o.feedback.len() != 1 || !o.state.done() || !o.copy_to_g1 {
        return Err(format!("nmtype.order: life {o:?}"));
    }
    if o.cancel("late").is_ok() || o.start().is_ok() || o.progress("x").is_ok() {
        return Err("nmtype.order: a finished order moved".into());
    }
    let mut late = Order::new(2, Verb::Focus, "g2", "t1", 1_000, 10);
    if late.accept(2_000).is_ok() || late.state != State::Rejected {
        return Err("nmtype.order: an expired order was taken".into());
    }
    let mut gains = Gains::default();
    gains.set("reflex.confidence", 2.0, "nmbrain.g2", 1_000, 100);
    if (gains.get("reflex.confidence", 1_050) - 2.0).abs() > 1e-6
        || (gains.get("reflex.confidence", 1_200) - 1.0).abs() > 1e-6
        || (gains.get("other", 1_050) - 1.0).abs() > 1e-6
    {
        return Err("nmtype.order: gains and expiry".into());
    }
    gains.set("x", 99.0, "g2", 0, 10);
    if (gains.get("x", 5) - 4.0).abs() > 1e-6 {
        return Err("nmtype.order: gain bound".into());
    }
    let mut c = Order::new(3, Verb::RunChain, "g2", "core", 0, 1_000);
    if c.cancel("x").is_ok() {
        return Err("nmtype.order: a pending order was cancelled".into());
    }
    c.accept(1).map_err(|e| e.to_string())?;
    c.cancel("superseded").map_err(|e| e.to_string())?;
    if c.state != State::Cancelled {
        return Err("nmtype.order: cancel".into());
    }
    Ok(())
}
