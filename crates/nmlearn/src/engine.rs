//! Complete Continuous Learning Architecture (`nmlearn.engine`; 2026-11-20 specification by Luis
//! Jake Gabriel III): an independent, model-independent cognitive subsystem operating alongside
//! the sensory drones, tentacles, Globalized Context Stack, memory system, and Core Brain.
//!
//! Organizes continuous learning across:
//! - Three learning levels: Level 1 (`Experiential`, immediate short-term adaptation), Level 2
//!   (`Knowledge`, pattern extraction and 5-category persistent learning memory consolidation),
//!   and Level 3 (`Behavioral`, conditional scoped policies, retrieval learning, and multi-window
//!   context allocation and reconstruction learning).
//! - Two complementary cycles: Cycle A (`Immediate`, during normal interaction) and Cycle B
//!   (`Consolidation`, periodic or threshold-triggered background/on-demand synthesis).
//! - Six modular continual learning cartridges (`knowledge`, `procedural`, `behavioral`,
//!   `retrieval`, `interaction`, `evaluation`) with full versioning and rollback.
//! - Dedicated `LearningSupervisor` enforcing five evidence states (`observed`, `candidate`,
//!   `under_evaluation`, `validated`, `rejected`), scope enforcement, contradiction detection, and
//!   regression prevention across eight evaluation dimensions and ten continual learning metrics
//!   (including `correction_recurrence` and `learning_efficiency`).
use crate::unit::{n, obj, s};
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Subfolder under a project's memory folder where the continual learning subsystem persists its
/// structured state (`<project_memory>/continual/state.json`).
pub const DIR: &str = "continual";
pub const STATE_FILE: &str = "state.json";
/// Held-out experiences needed before the gated cycle measures instead of using its supervisor rule.
pub const MIN_HELD_OUT: usize = 5;

/// What one gated Cycle B did (`ContinuousLearningEngine::run_gated_cycle`).
#[derive(Clone, Debug, PartialEq)]
pub struct GatedCycle {
    /// Held-out experiences the gate scored on.
    pub held_out: usize,
    /// Held-out score of the state before the cycle, and of the candidate (`None`: not measured).
    pub current: Option<f64>,
    pub candidate: Option<f64>,
    /// Whether the candidate became the state.
    pub promoted: bool,
    /// Whether the decision was measured on held-out experiences (else the supervisor rule).
    pub measured: bool,
    /// The cycle's own report.
    pub cycle: Value,
}

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmlearn.engine", FaultKind::Policy, m)
}
fn io_err(m: impl Into<String>) -> Fault {
    Fault::new("nmlearn.engine", FaultKind::Resource, m)
}
fn fnum(x: f64) -> Value {
    Value::Number(format!("{x:.4}"))
}
fn str_of(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).unwrap_or_default().to_owned()
}
fn u64_of(v: &Value, k: &str) -> u64 {
    v.get(k).and_then(Value::integer).unwrap_or(0)
}
fn f64_of(v: &Value, k: &str, default: f64) -> f64 {
    match v.get(k) {
        Ok(Value::Number(t)) => t.parse::<f64>().unwrap_or(default),
        _ => default,
    }
}
fn fopt_of(v: &Value, k: &str) -> Option<f64> {
    match v.get(k) {
        Ok(Value::Number(t)) => t.parse::<f64>().ok(),
        _ => None,
    }
}
fn bool_of(v: &Value, k: &str) -> bool {
    matches!(v.get(k), Ok(Value::Bool(true)))
}
fn vec_str_of(v: &Value, k: &str) -> Vec<String> {
    v.get(k)
        .and_then(Value::array)
        .map(|a| a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect())
        .unwrap_or_default()
}

/// The three fundamental levels of continuous learning (Section 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LearningLevel {
    /// Level 1: short-term adaptation from individual interactions, corrections, outcomes.
    Experiential,
    /// Level 2: long-term pattern extraction, consolidation, and retrieval improvement.
    Knowledge,
    /// Level 3: system-wide adaptation of strategies, workflows, context allocation, policies.
    Behavioral,
}
impl LearningLevel {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Experiential => "experiential",
            Self::Knowledge => "knowledge",
            Self::Behavioral => "behavioral",
        }
    }
}

/// The five supervision and validation states managed by the `LearningSupervisor` (Section 13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupervisionState {
    Observed,
    Candidate,
    UnderEvaluation,
    Validated,
    Rejected,
}
impl SupervisionState {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Observed => "observed",
            Self::Candidate => "candidate",
            Self::UnderEvaluation => "under_evaluation",
            Self::Validated => "validated",
            Self::Rejected => "rejected",
        }
    }
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s {
            "candidate" => Self::Candidate,
            "under_evaluation" => Self::UnderEvaluation,
            "validated" => Self::Validated,
            "rejected" => Self::Rejected,
            _ => Self::Observed,
        }
    }
}

/// The ten experience types captured by the Experience Collector (Section 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExperienceType {
    Correction,
    Preference,
    SuccessfulOutcome,
    FailedOutcome,
    ReasoningObservation,
    RetrievalObservation,
    ContextObservation,
    AgentFeedback,
    EnvironmentalObservation,
    RepeatedBehavior,
}
impl ExperienceType {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Correction => "correction",
            Self::Preference => "preference",
            Self::SuccessfulOutcome => "successful_outcome",
            Self::FailedOutcome => "failed_outcome",
            Self::ReasoningObservation => "reasoning_observation",
            Self::RetrievalObservation => "retrieval_observation",
            Self::ContextObservation => "context_observation",
            Self::AgentFeedback => "agent_feedback",
            Self::EnvironmentalObservation => "environmental_observation",
            Self::RepeatedBehavior => "repeated_behavior",
        }
    }
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s {
            "correction" => Self::Correction,
            "preference" => Self::Preference,
            "successful_outcome" => Self::SuccessfulOutcome,
            "failed_outcome" => Self::FailedOutcome,
            "reasoning_observation" => Self::ReasoningObservation,
            "retrieval_observation" => Self::RetrievalObservation,
            "context_observation" => Self::ContextObservation,
            "agent_feedback" => Self::AgentFeedback,
            "environmental_observation" => Self::EnvironmentalObservation,
            _ => Self::RepeatedBehavior,
        }
    }
}

/// The four feedback signal channels (Section 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedbackSignal {
    Explicit,
    Implicit,
    Internal,
    Delayed,
}
impl FeedbackSignal {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Implicit => "implicit",
            Self::Internal => "internal",
            Self::Delayed => "delayed",
        }
    }
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s {
            "explicit" => Self::Explicit,
            "internal" => Self::Internal,
            "delayed" => Self::Delayed,
            _ => Self::Implicit,
        }
    }
}

/// Surrounding circumstances of an experience (Section 2).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ExperienceContext {
    pub problem: String,
    pub initial_approach: String,
    pub correction: String,
    pub places: Vec<String>,
    pub provenance: String,
}
impl ExperienceContext {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("problem", s(&self.problem)),
            ("initial_approach", s(&self.initial_approach)),
            ("correction", s(&self.correction)),
            ("places", Value::Array(self.places.iter().map(|p| s(p)).collect())),
            ("provenance", s(&self.provenance)),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        Self {
            problem: str_of(v, "problem"),
            initial_approach: str_of(v, "initial_approach"),
            correction: str_of(v, "correction"),
            places: vec_str_of(v, "places"),
            provenance: str_of(v, "provenance"),
        }
    }
}

/// Outcome of an experience, supporting delayed updates when new evidence arrives (Sections 2 & 3).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ExperienceOutcome {
    pub initial_success: bool,
    pub corrected_success: bool,
    pub delayed_updated: bool,
    pub tokens_used: u64,
    pub ms_used: u64,
}
impl ExperienceOutcome {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("initial_success", Value::Bool(self.initial_success)),
            ("corrected_success", Value::Bool(self.corrected_success)),
            ("delayed_updated", Value::Bool(self.delayed_updated)),
            ("tokens_used", n(self.tokens_used)),
            ("ms_used", n(self.ms_used)),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        Self {
            initial_success: bool_of(v, "initial_success"),
            corrected_success: bool_of(v, "corrected_success"),
            delayed_updated: bool_of(v, "delayed_updated"),
            tokens_used: u64_of(v, "tokens_used"),
            ms_used: u64_of(v, "ms_used"),
        }
    }
}

/// Structured experience record stored in Episodic Learning Memory (Section 2).
#[derive(Clone, Debug, PartialEq)]
pub struct ExperienceRecord {
    pub experience_id: String,
    pub timestamp_ms: u64,
    pub experience_type: ExperienceType,
    pub signal: FeedbackSignal,
    pub task: String,
    pub scope: String,
    pub context: ExperienceContext,
    pub outcome: ExperienceOutcome,
    pub learning_candidates: Vec<String>,
    pub confidence: f64,
    pub status: SupervisionState,
    pub occurrences: u64,
}
impl ExperienceRecord {
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut cands = self.learning_candidates.clone();
        cands.sort();
        let raw = format!(
            "{}|{}|{}|{}|{}|{}",
            self.experience_type.name(),
            self.task.trim().to_lowercase(),
            self.scope.trim().to_lowercase(),
            self.context.problem.trim().to_lowercase(),
            self.context.correction.trim().to_lowercase(),
            cands.join(",")
        );
        nmhash::sha256_bytes(raw.as_bytes())
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("experience_id", s(&self.experience_id)),
            ("timestamp_ms", n(self.timestamp_ms)),
            ("experience_type", s(self.experience_type.name())),
            ("signal", s(self.signal.name())),
            ("task", s(&self.task)),
            ("scope", s(&self.scope)),
            ("context", self.context.to_value()),
            ("outcome", self.outcome.to_value()),
            ("learning_candidates", Value::Array(self.learning_candidates.iter().map(|c| s(c)).collect())),
            ("confidence", fnum(self.confidence)),
            ("status", s(self.status.name())),
            ("occurrences", n(self.occurrences)),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        Self {
            experience_id: str_of(v, "experience_id"),
            timestamp_ms: u64_of(v, "timestamp_ms"),
            experience_type: ExperienceType::parse(&str_of(v, "experience_type")),
            signal: FeedbackSignal::parse(&str_of(v, "signal")),
            task: str_of(v, "task"),
            scope: str_of(v, "scope"),
            context: v.get("context").map_or_else(|_| ExperienceContext::default(), ExperienceContext::from_value),
            outcome: v.get("outcome").map_or_else(|_| ExperienceOutcome::default(), ExperienceOutcome::from_value),
            learning_candidates: vec_str_of(v, "learning_candidates"),
            confidence: f64_of(v, "confidence", 0.7),
            status: SupervisionState::parse(&str_of(v, "status")),
            occurrences: u64_of(v, "occurrences").max(1),
        }
    }
}

/// Three types of extracted knowledge (Section 5: Declarative, Procedural, Strategic).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnowledgeKind {
    Declarative,
    Procedural,
    Strategic,
}
impl KnowledgeKind {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Declarative => "declarative",
            Self::Procedural => "procedural",
            Self::Strategic => "strategic",
        }
    }
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s {
            "declarative" => Self::Declarative,
            "strategic" => Self::Strategic,
            _ => Self::Procedural,
        }
    }
}

/// Generalized pattern extracted across related experiences (Section 5).
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedPattern {
    pub id: String,
    pub category: String,
    pub kind: KnowledgeKind,
    pub observation: String,
    pub generalized_lesson: String,
    pub recommended_procedure: Vec<String>,
    pub evidence_ids: Vec<String>,
    pub confidence: f64,
    pub scope: String,
    pub status: SupervisionState,
}
impl LearnedPattern {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("id", s(&self.id)),
            ("category", s(&self.category)),
            ("kind", s(self.kind.name())),
            ("observation", s(&self.observation)),
            ("generalized_lesson", s(&self.generalized_lesson)),
            ("recommended_procedure", Value::Array(self.recommended_procedure.iter().map(|p| s(p)).collect())),
            ("evidence_ids", Value::Array(self.evidence_ids.iter().map(|e| s(e)).collect())),
            ("confidence", fnum(self.confidence)),
            ("scope", s(&self.scope)),
            ("status", s(self.status.name())),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        Self {
            id: str_of(v, "id"),
            category: str_of(v, "category"),
            kind: KnowledgeKind::parse(&str_of(v, "kind")),
            observation: str_of(v, "observation"),
            generalized_lesson: str_of(v, "generalized_lesson"),
            recommended_procedure: vec_str_of(v, "recommended_procedure"),
            evidence_ids: vec_str_of(v, "evidence_ids"),
            confidence: f64_of(v, "confidence", 0.8),
            scope: str_of(v, "scope"),
            status: SupervisionState::parse(&str_of(v, "status")),
        }
    }
}

/// Category 2 of Persistent Learning Memory: Semantic Memory (consolidated facts, concepts,
/// relationships, provenance, and separate retention vs. retrieval priority; Section 6).
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticEntry {
    pub id: String,
    pub concept: String,
    pub fact: String,
    pub related_concepts: Vec<String>,
    pub source_experiences: Vec<String>,
    pub scope: String,
    pub confidence: f64,
    pub retrieval_priority: f64,
    pub permanent_retention: bool,
    pub uses: u64,
    pub updated_ms: u64,
}
impl SemanticEntry {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("id", s(&self.id)),
            ("concept", s(&self.concept)),
            ("fact", s(&self.fact)),
            ("related_concepts", Value::Array(self.related_concepts.iter().map(|c| s(c)).collect())),
            ("source_experiences", Value::Array(self.source_experiences.iter().map(|e| s(e)).collect())),
            ("scope", s(&self.scope)),
            ("confidence", fnum(self.confidence)),
            ("retrieval_priority", fnum(self.retrieval_priority)),
            ("permanent_retention", Value::Bool(self.permanent_retention)),
            ("uses", n(self.uses)),
            ("updated_ms", n(self.updated_ms)),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        Self {
            id: str_of(v, "id"),
            concept: str_of(v, "concept"),
            fact: str_of(v, "fact"),
            related_concepts: vec_str_of(v, "related_concepts"),
            source_experiences: vec_str_of(v, "source_experiences"),
            scope: str_of(v, "scope"),
            confidence: f64_of(v, "confidence", 0.8),
            retrieval_priority: f64_of(v, "retrieval_priority", 0.8),
            permanent_retention: bool_of(v, "permanent_retention"),
            uses: u64_of(v, "uses"),
            updated_ms: u64_of(v, "updated_ms"),
        }
    }
}

/// Category 3 of Persistent Learning Memory: Procedural Memory (traceable workflows and verification
/// procedures; Section 6).
#[derive(Clone, Debug, PartialEq)]
pub struct ProceduralEntry {
    pub id: String,
    pub category: String,
    pub trigger: String,
    pub steps: Vec<String>,
    pub source_experiences: Vec<String>,
    pub pattern_id: String,
    pub scope: String,
    pub confidence: f64,
    pub applications: u64,
    pub successes: u64,
    pub status: SupervisionState,
}
impl ProceduralEntry {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("id", s(&self.id)),
            ("category", s(&self.category)),
            ("trigger", s(&self.trigger)),
            ("steps", Value::Array(self.steps.iter().map(|st| s(st)).collect())),
            ("source_experiences", Value::Array(self.source_experiences.iter().map(|e| s(e)).collect())),
            ("pattern_id", s(&self.pattern_id)),
            ("scope", s(&self.scope)),
            ("confidence", fnum(self.confidence)),
            ("applications", n(self.applications)),
            ("successes", n(self.successes)),
            ("status", s(self.status.name())),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        Self {
            id: str_of(v, "id"),
            category: str_of(v, "category"),
            trigger: str_of(v, "trigger"),
            steps: vec_str_of(v, "steps"),
            source_experiences: vec_str_of(v, "source_experiences"),
            pattern_id: str_of(v, "pattern_id"),
            scope: str_of(v, "scope"),
            confidence: f64_of(v, "confidence", 0.85),
            applications: u64_of(v, "applications"),
            successes: u64_of(v, "successes"),
            status: SupervisionState::parse(&str_of(v, "status")),
        }
    }
}

/// Category 4 of Persistent Learning Memory: Preference and Interaction Memory (Section 6).
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceEntry {
    pub id: String,
    pub scope: String,
    pub instruction: String,
    pub prefer: Vec<String>,
    pub avoid: Vec<String>,
    pub source_experiences: Vec<String>,
    pub confidence: f64,
}
impl PreferenceEntry {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("id", s(&self.id)),
            ("scope", s(&self.scope)),
            ("instruction", s(&self.instruction)),
            ("prefer", Value::Array(self.prefer.iter().map(|p| s(p)).collect())),
            ("avoid", Value::Array(self.avoid.iter().map(|a| s(a)).collect())),
            ("source_experiences", Value::Array(self.source_experiences.iter().map(|e| s(e)).collect())),
            ("confidence", fnum(self.confidence)),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        Self {
            id: str_of(v, "id"),
            scope: str_of(v, "scope"),
            instruction: str_of(v, "instruction"),
            prefer: vec_str_of(v, "prefer"),
            avoid: vec_str_of(v, "avoid"),
            source_experiences: vec_str_of(v, "source_experiences"),
            confidence: f64_of(v, "confidence", 0.9),
        }
    }
}

/// Conditional, scoped Behavioral Policy (Section 8).
#[derive(Clone, Debug, PartialEq)]
pub struct BehavioralPolicy {
    pub id: String,
    pub name: String,
    pub condition: String,
    pub applies_to: String,
    pub does_not_apply_to: String,
    pub previous_behavior: String,
    pub observed_problem: String,
    pub new_behavior: Vec<String>,
    pub expected_improvement: String,
    pub confidence: f64,
    pub evidence_count: u64,
    pub status: SupervisionState,
}
impl BehavioralPolicy {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("id", s(&self.id)),
            ("name", s(&self.name)),
            ("condition", s(&self.condition)),
            ("applies_to", s(&self.applies_to)),
            ("does_not_apply_to", s(&self.does_not_apply_to)),
            ("previous_behavior", s(&self.previous_behavior)),
            ("observed_problem", s(&self.observed_problem)),
            ("new_behavior", Value::Array(self.new_behavior.iter().map(|b| s(b)).collect())),
            ("expected_improvement", s(&self.expected_improvement)),
            ("confidence", fnum(self.confidence)),
            ("evidence_count", n(self.evidence_count)),
            ("status", s(self.status.name())),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        Self {
            id: str_of(v, "id"),
            name: str_of(v, "name"),
            condition: str_of(v, "condition"),
            applies_to: str_of(v, "applies_to"),
            does_not_apply_to: str_of(v, "does_not_apply_to"),
            previous_behavior: str_of(v, "previous_behavior"),
            observed_problem: str_of(v, "observed_problem"),
            new_behavior: vec_str_of(v, "new_behavior"),
            expected_improvement: str_of(v, "expected_improvement"),
            confidence: f64_of(v, "confidence", 0.85),
            evidence_count: u64_of(v, "evidence_count"),
            status: SupervisionState::parse(&str_of(v, "status")),
        }
    }
}

/// Retrieval Learner state across the 7 retrieval dimensions (Section 7).
#[derive(Clone, Debug, PartialEq)]
pub struct RetrievalAdaptation {
    pub semantic_weight: f64,
    pub lexical_weight: f64,
    pub structural_weight: f64,
    pub recency_weight: f64,
    pub relationship_weight: f64,
    pub diversity_penalty: f64,
    pub similarity_threshold: f64,
    pub retrieval_depth: u64,
    pub retrieved_total: u64,
    pub used_total: u64,
    pub irrelevant_total: u64,
    pub missing_essential_total: u64,
}
impl Default for RetrievalAdaptation {
    fn default() -> Self {
        Self {
            semantic_weight: 0.35,
            lexical_weight: 0.30,
            structural_weight: 0.20,
            recency_weight: 0.08,
            relationship_weight: 0.07,
            diversity_penalty: 0.15,
            similarity_threshold: 0.62,
            retrieval_depth: 6,
            retrieved_total: 0,
            used_total: 0,
            irrelevant_total: 0,
            missing_essential_total: 0,
        }
    }
}
impl RetrievalAdaptation {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("semantic_weight", fnum(self.semantic_weight)),
            ("lexical_weight", fnum(self.lexical_weight)),
            ("structural_weight", fnum(self.structural_weight)),
            ("recency_weight", fnum(self.recency_weight)),
            ("relationship_weight", fnum(self.relationship_weight)),
            ("diversity_penalty", fnum(self.diversity_penalty)),
            ("similarity_threshold", fnum(self.similarity_threshold)),
            ("retrieval_depth", n(self.retrieval_depth)),
            ("retrieved_total", n(self.retrieved_total)),
            ("used_total", n(self.used_total)),
            ("irrelevant_total", n(self.irrelevant_total)),
            ("missing_essential_total", n(self.missing_essential_total)),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let d = Self::default();
        Self {
            semantic_weight: f64_of(v, "semantic_weight", d.semantic_weight),
            lexical_weight: f64_of(v, "lexical_weight", d.lexical_weight),
            structural_weight: f64_of(v, "structural_weight", d.structural_weight),
            recency_weight: f64_of(v, "recency_weight", d.recency_weight),
            relationship_weight: f64_of(v, "relationship_weight", d.relationship_weight),
            diversity_penalty: f64_of(v, "diversity_penalty", d.diversity_penalty),
            similarity_threshold: f64_of(v, "similarity_threshold", d.similarity_threshold),
            retrieval_depth: u64_of(v, "retrieval_depth").max(1),
            retrieved_total: u64_of(v, "retrieved_total"),
            used_total: u64_of(v, "used_total"),
            irrelevant_total: u64_of(v, "irrelevant_total"),
            missing_essential_total: u64_of(v, "missing_essential_total"),
        }
    }
}

/// Backward trace of a context compression or reconstruction omission across the 5 stages
/// (Section 9.C).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReconstructionTrace {
    pub present_in_source: bool,
    pub identified_in_compression: bool,
    pub preserved_in_representation: bool,
    pub retrieved_in_reconstruction: bool,
    pub interpreted_in_reasoning: bool,
}
impl ReconstructionTrace {
    /// Names the exact stage where the technical constraint or detail was lost.
    #[must_use]
    pub const fn failure_stage(&self) -> &'static str {
        if !self.present_in_source {
            "source_acquisition"
        } else if !self.identified_in_compression {
            "compression_importance"
        } else if !self.preserved_in_representation {
            "compressed_representation"
        } else if !self.retrieved_in_reconstruction {
            "reconstruction_retrieval"
        } else if !self.interpreted_in_reasoning {
            "final_reasoning"
        } else {
            "none"
        }
    }
}

/// Multi-window / 96K Context Architecture Learner state (Section 9: individual window, cross-window
/// union, compression/reconstruction backward trace, and context allocation).
#[derive(Clone, Debug, PartialEq)]
pub struct ContextAdaptation {
    pub code_definitions_share: f64,
    pub technical_docs_share: f64,
    pub historical_memory_share: f64,
    pub current_instructions_share: f64,
    pub window_contributions: BTreeMap<String, u64>,
    pub window_redundancies: BTreeMap<String, u64>,
    pub stage_losses: BTreeMap<String, u64>,
}
impl Default for ContextAdaptation {
    fn default() -> Self {
        Self {
            code_definitions_share: 0.50,
            technical_docs_share: 0.20,
            historical_memory_share: 0.15,
            current_instructions_share: 0.15,
            window_contributions: BTreeMap::new(),
            window_redundancies: BTreeMap::new(),
            stage_losses: BTreeMap::new(),
        }
    }
}
impl ContextAdaptation {
    #[must_use]
    pub fn to_value(&self) -> Value {
        let map_val = |m: &BTreeMap<String, u64>| Value::Object(m.iter().map(|(k, v)| (k.clone(), n(*v))).collect());
        obj(vec![
            ("code_definitions_share", fnum(self.code_definitions_share)),
            ("technical_docs_share", fnum(self.technical_docs_share)),
            ("historical_memory_share", fnum(self.historical_memory_share)),
            ("current_instructions_share", fnum(self.current_instructions_share)),
            ("window_contributions", map_val(&self.window_contributions)),
            ("window_redundancies", map_val(&self.window_redundancies)),
            ("stage_losses", map_val(&self.stage_losses)),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let d = Self::default();
        let read_map = |k: &str| -> BTreeMap<String, u64> {
            v.get(k)
                .and_then(Value::object)
                .map(|o| o.iter().filter_map(|(k, val)| Some((k.clone(), val.integer().ok()?))).collect())
                .unwrap_or_default()
        };
        Self {
            code_definitions_share: f64_of(v, "code_definitions_share", d.code_definitions_share),
            technical_docs_share: f64_of(v, "technical_docs_share", d.technical_docs_share),
            historical_memory_share: f64_of(v, "historical_memory_share", d.historical_memory_share),
            current_instructions_share: f64_of(v, "current_instructions_share", d.current_instructions_share),
            window_contributions: read_map("window_contributions"),
            window_redundancies: read_map("window_redundancies"),
            stage_losses: read_map("stage_losses"),
        }
    }
}

/// The 8 Performance Evaluation Dimensions (Section 4).
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationDimensions {
    pub accuracy: f64,
    pub relevance: f64,
    pub completeness: f64,
    pub efficiency: f64,
    pub consistency: f64,
    pub instruction_adherence: f64,
    pub reconstruction_fidelity: f64,
    pub generalization: f64,
}
impl Default for EvaluationDimensions {
    fn default() -> Self {
        Self {
            accuracy: 0.80,
            relevance: 0.80,
            completeness: 0.80,
            efficiency: 0.80,
            consistency: 0.85,
            instruction_adherence: 0.90,
            reconstruction_fidelity: 0.85,
            generalization: 0.80,
        }
    }
}
impl EvaluationDimensions {
    #[must_use]
    pub fn mean(&self) -> f64 {
        (self.accuracy
            + self.relevance
            + self.completeness
            + self.efficiency
            + self.consistency
            + self.instruction_adherence
            + self.reconstruction_fidelity
            + self.generalization)
            / 8.0
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("accuracy", fnum(self.accuracy)),
            ("relevance", fnum(self.relevance)),
            ("completeness", fnum(self.completeness)),
            ("efficiency", fnum(self.efficiency)),
            ("consistency", fnum(self.consistency)),
            ("instruction_adherence", fnum(self.instruction_adherence)),
            ("reconstruction_fidelity", fnum(self.reconstruction_fidelity)),
            ("generalization", fnum(self.generalization)),
            ("overall", fnum(self.mean())),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let d = Self::default();
        Self {
            accuracy: f64_of(v, "accuracy", d.accuracy),
            relevance: f64_of(v, "relevance", d.relevance),
            completeness: f64_of(v, "completeness", d.completeness),
            efficiency: f64_of(v, "efficiency", d.efficiency),
            consistency: f64_of(v, "consistency", d.consistency),
            instruction_adherence: f64_of(v, "instruction_adherence", d.instruction_adherence),
            reconstruction_fidelity: f64_of(v, "reconstruction_fidelity", d.reconstruction_fidelity),
            generalization: f64_of(v, "generalization", d.generalization),
        }
    }
}

/// The 10 Continuous Learning Benchmark Metrics (Section 18), including `correction_recurrence`
/// and `learning_efficiency`.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkMetrics {
    pub learning_retention: f64,
    pub correction_recurrence: f64,
    pub retrieval_precision: f64,
    pub retrieval_recall: f64,
    pub context_fidelity: f64,
    pub reconstruction_accuracy: f64,
    pub policy_effectiveness: f64,
    pub transfer_performance: f64,
    pub regression_rate: f64,
    pub learning_efficiency: f64,
    pub corrections_total: u64,
    pub recurrences_total: u64,
    pub last_recurrence_stage: String,
}
impl Default for BenchmarkMetrics {
    fn default() -> Self {
        Self {
            learning_retention: 1.0,
            correction_recurrence: 0.0,
            retrieval_precision: 0.85,
            retrieval_recall: 0.85,
            context_fidelity: 0.90,
            reconstruction_accuracy: 0.90,
            policy_effectiveness: 0.85,
            transfer_performance: 0.80,
            regression_rate: 0.0,
            learning_efficiency: 0.85,
            corrections_total: 0,
            recurrences_total: 0,
            last_recurrence_stage: "none".to_owned(),
        }
    }
}
impl BenchmarkMetrics {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("learning_retention", fnum(self.learning_retention)),
            ("correction_recurrence", fnum(self.correction_recurrence)),
            ("retrieval_precision", fnum(self.retrieval_precision)),
            ("retrieval_recall", fnum(self.retrieval_recall)),
            ("context_fidelity", fnum(self.context_fidelity)),
            ("reconstruction_accuracy", fnum(self.reconstruction_accuracy)),
            ("policy_effectiveness", fnum(self.policy_effectiveness)),
            ("transfer_performance", fnum(self.transfer_performance)),
            ("regression_rate", fnum(self.regression_rate)),
            ("learning_efficiency", fnum(self.learning_efficiency)),
            ("corrections_total", n(self.corrections_total)),
            ("recurrences_total", n(self.recurrences_total)),
            ("last_recurrence_stage", s(&self.last_recurrence_stage)),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let d = Self::default();
        Self {
            learning_retention: f64_of(v, "learning_retention", d.learning_retention),
            correction_recurrence: f64_of(v, "correction_recurrence", d.correction_recurrence),
            retrieval_precision: f64_of(v, "retrieval_precision", d.retrieval_precision),
            retrieval_recall: f64_of(v, "retrieval_recall", d.retrieval_recall),
            context_fidelity: f64_of(v, "context_fidelity", d.context_fidelity),
            reconstruction_accuracy: f64_of(v, "reconstruction_accuracy", d.reconstruction_accuracy),
            policy_effectiveness: f64_of(v, "policy_effectiveness", d.policy_effectiveness),
            transfer_performance: f64_of(v, "transfer_performance", d.transfer_performance),
            regression_rate: f64_of(v, "regression_rate", d.regression_rate),
            learning_efficiency: f64_of(v, "learning_efficiency", d.learning_efficiency),
            corrections_total: u64_of(v, "corrections_total"),
            recurrences_total: u64_of(v, "recurrences_total"),
            last_recurrence_stage: {
                let st = str_of(v, "last_recurrence_stage");
                if st.is_empty() { "none".to_owned() } else { st }
            },
        }
    }
}

/// One of the 6 specialized Continual Learning Cartridges (Section 10: `knowledge`, `procedural`,
/// `behavioral`, `retrieval`, `interaction`, `evaluation`).
pub const CARTRIDGE_DOMAINS: [&str; 6] = ["knowledge", "procedural", "behavioral", "retrieval", "interaction", "evaluation"];

#[derive(Clone, Debug, PartialEq)]
pub struct ContinualCartridge {
    pub id: String,
    pub domain: String,
    pub version: u64,
    pub status: SupervisionState,
    pub entries: u64,
    pub evidence_count: u64,
    pub confidence: f64,
    pub summary: String,
    pub updated_ms: u64,
    pub archived: bool,
}
impl ContinualCartridge {
    #[must_use]
    pub fn new(domain: &str) -> Self {
        let summary = match domain {
            "knowledge" => "Consolidated facts, concepts, relationships, and semantic project knowledge.",
            "procedural" => "Validated methods, debugging workflows, verification procedures, and step checklists.",
            "behavioral" => "Conditional decision rules, context allocation policies, and adaptive operating strategies.",
            "retrieval" => "Learned ranking weights, similarity thresholds, diversity penalties, and memory priorities.",
            "interaction" => "User preferences, explicit corrections, communication rules, and scoped constraints.",
            "evaluation" => "Performance baselines, 8-dimension evaluations, known failure guards, and recurrence metrics.",
            _ => "Continual learning domain cartridge.",
        };
        Self {
            id: format!("clc-{domain}"),
            domain: domain.to_owned(),
            version: 1,
            status: SupervisionState::Observed,
            entries: 0,
            evidence_count: 0,
            confidence: 0.85,
            summary: summary.to_owned(),
            updated_ms: 0,
            archived: false,
        }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("id", s(&self.id)),
            ("domain", s(&self.domain)),
            ("version", n(self.version)),
            ("status", s(self.status.name())),
            ("entries", n(self.entries)),
            ("evidence_count", n(self.evidence_count)),
            ("confidence", fnum(self.confidence)),
            ("summary", s(&self.summary)),
            ("updated_ms", n(self.updated_ms)),
            ("archived", Value::Bool(self.archived)),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let domain = str_of(v, "domain");
        let d = Self::new(&domain);
        Self {
            id: {
                let i = str_of(v, "id");
                if i.is_empty() { d.id } else { i }
            },
            domain,
            version: u64_of(v, "version").max(1),
            status: SupervisionState::parse(&str_of(v, "status")),
            entries: u64_of(v, "entries"),
            evidence_count: u64_of(v, "evidence_count"),
            confidence: f64_of(v, "confidence", d.confidence),
            summary: {
                let sum = str_of(v, "summary");
                if sum.is_empty() { d.summary } else { sum }
            },
            updated_ms: u64_of(v, "updated_ms"),
            archived: bool_of(v, "archived"),
        }
    }
}

/// One measured figure of the engine (`ContinuousLearningEngine::measures`).
#[derive(Clone, Debug, PartialEq)]
pub struct Measure {
    pub key: &'static str,
    pub label: &'static str,
    pub what: &'static str,
    pub value: f64,
    /// The observations it rests on.
    pub n: u64,
    /// For the held-out score: the score before the first measured cycle.
    pub baseline: Option<f64>,
}
impl Measure {
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    fn ratio(key: &'static str, label: &'static str, what: &'static str, part: u64, of: u64) -> Self {
        Self { key, label, what, value: part as f64 / of.max(1) as f64, n: of, baseline: None }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("key", s(self.key)),
            ("label", s(self.label)),
            ("what", s(self.what)),
            ("value", fnum(self.value)),
            ("n", n(self.n)),
            ("baseline", self.baseline.map_or(Value::Null, fnum)),
        ])
    }
}

/// Complete state of the Continuous Learning Engine & Supervisor for a repository (Sections 1-20).
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousLearningEngine {
    pub version: u64,
    pub cycles_run: u64,
    pub last_cycle_ms: u64,
    pub episodic: Vec<ExperienceRecord>,
    pub patterns: Vec<LearnedPattern>,
    pub semantic: Vec<SemanticEntry>,
    pub procedural: Vec<ProceduralEntry>,
    pub preferences: Vec<PreferenceEntry>,
    pub policies: Vec<BehavioralPolicy>,
    pub retrieval: RetrievalAdaptation,
    pub context: ContextAdaptation,
    pub baseline_eval: EvaluationDimensions,
    pub current_eval: EvaluationDimensions,
    pub metrics: BenchmarkMetrics,
    pub cartridges: Vec<ContinualCartridge>,
    /// 2026-11-25 (Luis Jake Gabriel III: nothing pseudo, every figure real and live): the measured
    /// held-out score the gate keeps (`run_gated_cycle`), the score of the state before the first
    /// measured cycle (the baseline), the held-out experiences it rests on, and how many measured
    /// cycles ran and how many of them the gate held back.
    pub held_out_score: Option<f64>,
    pub held_out_baseline: Option<f64>,
    pub held_out_n: u64,
    pub gated_measured: u64,
    pub gated_held_back: u64,
    /// Training store rows (`exp_<id>`) already taken in, whether they became a new experience or
    /// merged into an identical one, so each row is taken once (`idle::sync_engine`).
    pub synced: BTreeSet<String>,
    /// Previous validated snapshots ( encoded JSON ) for instant supervisor rollback (Section 13).
    pub history: Vec<(u64, String)>,
}

impl Default for ContinuousLearningEngine {
    fn default() -> Self {
        Self {
            version: 1,
            cycles_run: 0,
            last_cycle_ms: 0,
            episodic: Vec::new(),
            patterns: Vec::new(),
            semantic: Vec::new(),
            procedural: Vec::new(),
            preferences: Vec::new(),
            policies: Vec::new(),
            retrieval: RetrievalAdaptation::default(),
            context: ContextAdaptation::default(),
            baseline_eval: EvaluationDimensions::default(),
            current_eval: EvaluationDimensions::default(),
            metrics: BenchmarkMetrics::default(),
            cartridges: CARTRIDGE_DOMAINS.iter().map(|d| ContinualCartridge::new(d)).collect(),
            held_out_score: None,
            held_out_baseline: None,
            held_out_n: 0,
            gated_measured: 0,
            gated_held_back: 0,
            synced: BTreeSet::new(),
            history: Vec::new(),
        }
    }
}

impl ContinuousLearningEngine {
    /// Path to `<memory_dir>/continual/state.json`.
    #[must_use]
    pub fn state_path(memory_dir: &Path) -> PathBuf {
        memory_dir.join(DIR).join(STATE_FILE)
    }

    /// Load the engine state from `<memory_dir>/continual/state.json` along with persisted
    /// supervisor rollback snapshots under `<memory_dir>/continual/snapshots/`, or initialize a
    /// fresh state.
    #[must_use]
    pub fn open(memory_dir: &Path) -> Self {
        let p = Self::state_path(memory_dir);
        let parsed = std::fs::read(&p).ok().and_then(|b| nmjson::parse(&b).ok());
        let mut eng = parsed.as_ref().map_or_else(Self::default, Self::from_value);
        // A state saved before rows taken in were recorded (2026-11-25): every sync then took the
        // whole store, so when it holds experiences, the store's rows as they stand were taken.
        if parsed.as_ref().is_some_and(|v| v.get("synced").is_err()) && !eng.episodic.is_empty() {
            for file in [nmtype::examples::EXAMPLES, nmtype::examples::PREFERENCES] {
                for row in nmtype::examples::read_rows(memory_dir, file).unwrap_or_default() {
                    if let Ok(id) = row.get("id").and_then(Value::string) {
                        eng.synced.insert(format!("exp_{id}"));
                    }
                }
            }
        }
        let snap_dir = memory_dir.join(DIR).join("snapshots");
        if let Ok(rd) = std::fs::read_dir(&snap_dir) {
            let mut snaps: Vec<(u64, String)> = rd
                .flatten()
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    let ver = name.strip_prefix('v')?.strip_suffix(".json")?.parse::<u64>().ok()?;
                    let text = std::fs::read_to_string(e.path()).ok()?;
                    Some((ver, text))
                })
                .collect();
            snaps.sort_by_key(|(v, _)| *v);
            if snaps.len() > 8 {
                let drop_n = snaps.len() - 8;
                snaps.drain(0..drop_n);
            }
            eng.history = snaps;
        }
        eng
    }

    /// Persist the engine state atomically to `<memory_dir>/continual/state.json` and sync
    /// supervisor rollback snapshots in `<memory_dir>/continual/snapshots/`.
    /// # Errors
    /// If the directory or file cannot be written.
    pub fn save(&self, memory_dir: &Path) -> Result<(), Fault> {
        let dir = memory_dir.join(DIR);
        let snap_dir = dir.join("snapshots");
        std::fs::create_dir_all(&snap_dir).map_err(|e| io_err(format!("{}: {e}", snap_dir.display())))?;
        let keep_names: BTreeSet<String> = self.history.iter().map(|(v, _)| format!("v{v}.json")).collect();
        if let Ok(rd) = std::fs::read_dir(&snap_dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.ends_with(".json") && !keep_names.contains(&name) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        for (v, snap) in &self.history {
            let sp = snap_dir.join(format!("v{v}.json"));
            if !sp.exists() {
                let stmp = snap_dir.join(format!("v{v}.json.tmp"));
                std::fs::write(&stmp, snap).map_err(|e| io_err(format!("{}: {e}", stmp.display())))?;
                std::fs::rename(&stmp, &sp).map_err(|e| io_err(format!("{}: {e}", sp.display())))?;
            }
        }
        let p = dir.join(STATE_FILE);
        let tmp = dir.join("state.json.tmp");
        let text = nmjson::pretty::pretty(&self.to_value());
        std::fs::write(&tmp, text).map_err(|e| io_err(format!("{}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, &p).map_err(|e| io_err(format!("{}: {e}", p.display())))
    }

    /// Experience Filtering (Section 2): checks whether an incoming experience is meaningful,
    /// non-empty, and reliable enough to enter the learning pipeline.
    #[must_use]
    pub fn passes_filter(rec: &ExperienceRecord) -> bool {
        let has_content = !rec.context.problem.trim().is_empty()
            || !rec.context.correction.trim().is_empty()
            || !rec.learning_candidates.is_empty();
        has_content && rec.confidence >= 0.25 && !rec.task.trim().is_empty()
    }

    /// Cycle A (Immediate learning, Section 11 & 16): observe and filter a new experience,
    /// deduplicating repeated identical observations, checking for correction recurrence
    /// (Section 18), and extracting immediate high-confidence preferences or corrections.
    /// # Errors
    /// If the experience fails the filter criteria.
    pub fn observe(&mut self, mut rec: ExperienceRecord) -> Result<String, Fault> {
        if !Self::passes_filter(&rec) {
            return Err(bad("experience rejected by ExperienceCollector filter (low confidence or empty context)"));
        }
        if rec.timestamp_ms == 0 {
            rec.timestamp_ms = nmtype::now_ms();
        }
        if rec.scope.trim().is_empty() {
            rec.scope = "project".to_owned();
        }
        // Section 18: check whether this failure or correction is a recurrence of a previously
        // corrected candidate!
        if matches!(rec.experience_type, ExperienceType::Correction | ExperienceType::FailedOutcome) {
            let prior_corrected: BTreeSet<String> = self
                .episodic
                .iter()
                .filter(|e| e.experience_type == ExperienceType::Correction && e.outcome.corrected_success)
                .flat_map(|e| e.learning_candidates.iter().cloned())
                .collect();
            let recurred = rec.learning_candidates.iter().find(|c| prior_corrected.contains(*c)).cloned();
            if rec.experience_type == ExperienceType::Correction {
                self.metrics.corrections_total = self.metrics.corrections_total.saturating_add(1);
            }
            if let Some(cand) = recurred {
                self.metrics.recurrences_total = self.metrics.recurrences_total.saturating_add(1);
                // Diagnose which stage failed (Section 18: retention, extraction, retrieval, application)
                let in_patterns = self.patterns.iter().any(|p| p.category == cand);
                let in_procedures = self.procedural.iter().any(|p| p.category == cand && p.status == SupervisionState::Validated);
                self.metrics.last_recurrence_stage = if !in_patterns {
                    "extraction".to_owned()
                } else if !in_procedures {
                    "application".to_owned()
                } else {
                    "retrieval".to_owned()
                };
            }
            let denom = self.metrics.corrections_total.max(1) as f64;
            self.metrics.correction_recurrence = (self.metrics.recurrences_total as f64 / denom).clamp(0.0, 1.0);
        }

        // Section 2: consolidate repeated identical observations instead of duplicating endlessly.
        let fp = rec.fingerprint();
        if let Some(existing) = self.episodic.iter_mut().find(|e| e.fingerprint() == fp) {
            existing.occurrences = existing.occurrences.saturating_add(1);
            existing.timestamp_ms = rec.timestamp_ms;
            existing.confidence = (existing.confidence + 0.05).min(0.99);
            if rec.outcome.corrected_success {
                existing.outcome.corrected_success = true;
            }
            let id = existing.experience_id.clone();
            self.refresh_cartridge_counts(rec.timestamp_ms);
            return Ok(id);
        }

        let id = if rec.experience_id.trim().is_empty() {
            format!("exp_{:06}", self.episodic.len() + 1)
        } else {
            rec.experience_id.clone()
        };
        rec.experience_id.clone_from(&id);

        // Cycle A immediate extraction for explicit preferences and corrections (Section 11 table).
        if rec.experience_type == ExperienceType::Preference && !rec.context.correction.is_empty() {
            let pref_id = format!("pref_{:04}", self.preferences.len() + 1);
            self.preferences.push(PreferenceEntry {
                id: pref_id,
                scope: rec.scope.clone(),
                instruction: rec.context.correction.clone(),
                prefer: rec.learning_candidates.clone(),
                avoid: if rec.context.initial_approach.is_empty() { Vec::new() } else { vec![rec.context.initial_approach.clone()] },
                source_experiences: vec![id.clone()],
                confidence: rec.confidence,
            });
        }

        self.episodic.push(rec);
        self.refresh_cartridge_counts(nmtype::now_ms());
        Ok(id)
    }

    /// Delayed feedback update (Section 3): updates an existing experience when delayed outcome
    /// evidence arrives (e.g., a build/test verdict or subsequent correction).
    pub fn record_delayed_outcome(&mut self, exp_id: &str, success: bool, correction_note: Option<&str>) -> bool {
        let Some(rec) = self.episodic.iter_mut().find(|e| e.experience_id == exp_id) else {
            return false;
        };
        rec.outcome.corrected_success = success;
        rec.outcome.delayed_updated = true;
        rec.signal = FeedbackSignal::Delayed;
        if let Some(note) = correction_note.filter(|n| !n.trim().is_empty()) {
            rec.context.correction = note.to_owned();
        }
        if !success {
            rec.confidence = (rec.confidence * 0.7).max(0.25);
        } else {
            rec.confidence = (rec.confidence + 0.08).min(0.99);
        }
        true
    }

    /// Retrieval feedback loop (Section 7): records how many retrieved items were used vs.
    /// irrelevant vs. missing essential, and adapts the retrieval weights accordingly.
    pub fn record_retrieval_feedback(&mut self, retrieved: u64, used: u64, irrelevant: u64, missing_essential: u64) {
        self.retrieval.retrieved_total = self.retrieval.retrieved_total.saturating_add(retrieved);
        self.retrieval.used_total = self.retrieval.used_total.saturating_add(used);
        self.retrieval.irrelevant_total = self.retrieval.irrelevant_total.saturating_add(irrelevant);
        self.retrieval.missing_essential_total = self.retrieval.missing_essential_total.saturating_add(missing_essential);

        let total_ret = self.retrieval.retrieved_total.max(1) as f64;
        let precision = (self.retrieval.used_total as f64 / total_ret).clamp(0.0, 1.0);
        let recall_denom = (self.retrieval.used_total + self.retrieval.missing_essential_total).max(1) as f64;
        let recall = (self.retrieval.used_total as f64 / recall_denom).clamp(0.0, 1.0);

        self.metrics.retrieval_precision = precision;
        self.metrics.retrieval_recall = recall;
        self.current_eval.relevance = (0.5 * precision + 0.5 * recall).clamp(0.0, 1.0);

        // Lightweight retrieval adaptation (Section 7):
        if irrelevant > used {
            // Too many irrelevant semantic matches: raise threshold and boost lexical/structural precision.
            self.retrieval.similarity_threshold = (self.retrieval.similarity_threshold + 0.02).min(0.85);
            self.retrieval.lexical_weight = (self.retrieval.lexical_weight + 0.02).min(0.50);
            self.retrieval.structural_weight = (self.retrieval.structural_weight + 0.02).min(0.40);
            self.retrieval.semantic_weight = (self.retrieval.semantic_weight - 0.01).max(0.15);
            self.retrieval.diversity_penalty = (self.retrieval.diversity_penalty + 0.02).min(0.35);
        }
        if missing_essential > 0 {
            // Missed an essential connected memory: deepen retrieval and boost semantic + relationship traversal.
            self.retrieval.retrieval_depth = (self.retrieval.retrieval_depth + 1).min(12);
            self.retrieval.semantic_weight = (self.retrieval.semantic_weight + 0.02).min(0.55);
            self.retrieval.relationship_weight = (self.retrieval.relationship_weight + 0.02).min(0.25);
            self.retrieval.similarity_threshold = (self.retrieval.similarity_threshold - 0.01).max(0.45);
        }
    }

    /// Appends a compact structured `(task, retrieval/cartridge state -> outcome)` telemetry row
    /// to `<memory>/continual/param-telemetry.jsonl` to build the long-term training dataset for
    /// parameter-system learning and future parameter printing.
    /// `outcome` is `(success, tokens_used, ms_used)`.
    pub fn append_parameter_telemetry(
        &self,
        memory_dir: &Path,
        task: &str,
        scope: &str,
        outcome: (bool, u64, u64),
        active_cartridges: &[String],
    ) {
        use std::io::Write as _;
        let (success, tokens_used, ms_used) = outcome;
        let dir = memory_dir.join(DIR);
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let row = obj(vec![
            ("ts_ms", n(nmtype::now_ms())),
            ("version", n(self.version)),
            ("scope", s(scope)),
            ("task", s(task.chars().take(200).collect::<String>().as_str())),
            ("success", Value::Bool(success)),
            ("tokens_used", n(tokens_used)),
            ("ms_used", n(ms_used)),
            ("retrieval", self.retrieval.to_value()),
            ("cartridges", Value::Array(active_cartridges.iter().map(|c| s(c)).collect())),
        ]);
        let Ok(mut line) = row.encode() else {
            return;
        };
        line.push('\n');
        let path = dir.join("param-telemetry.jsonl");
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = f.write_all(line.as_bytes());
        }
    }

    /// Context compression and reconstruction backward trace (Section 9.C) and window contribution
    /// tracking (Sections 9.A & 9.B).
    pub fn record_context_trace(
        &mut self,
        useful_windows: &[&str],
        redundant_windows: &[&str],
        trace: Option<&ReconstructionTrace>,
    ) {
        for w in useful_windows {
            *self.context.window_contributions.entry((*w).to_owned()).or_insert(0) += 1;
        }
        for w in redundant_windows {
            *self.context.window_redundancies.entry((*w).to_owned()).or_insert(0) += 1;
        }
        if let Some(tr) = trace {
            let stage = tr.failure_stage();
            if stage != "none" {
                *self.context.stage_losses.entry(stage.to_owned()).or_insert(0) += 1;
                // Adapt context allocation based on where the omission occurred (Section 9.D):
                match stage {
                    "compression_importance" | "compressed_representation" => {
                        self.context.code_definitions_share = (self.context.code_definitions_share + 0.02).min(0.65);
                    }
                    "reconstruction_retrieval" => {
                        self.context.historical_memory_share = (self.context.historical_memory_share + 0.02).min(0.30);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Cycle B (Consolidation learning, Sections 5, 6, 8, 10, 11, 13, 16):
    /// 1.Snapshots current validated state for rollback.
    /// 2. Extracts recurring patterns across experiences (`PatternExtractor` + `KnowledgeExtractor`).
    /// 3. Consolidates into Semantic, Procedural, and Preference memory without destroying original
    ///    episodic records (`MemoryConsolidator`).
    /// 4. Proposes conditional, scoped Behavioral Policies (`PolicyLearner`).
    /// 5. Runs the `LearningSupervisor` validation & regression checks before activating any
    ///    candidate adaptation (`LearningSupervisor::validate`).
    /// 6. Updates the 6 Continual Learning Cartridges (`CartridgeManager`) and recomputes the
    ///    8 Evaluation Dimensions and 10 Continuous Learning Metrics.
    pub fn run_learning_cycle(&mut self) -> Value {
        let now = nmtype::now_ms();
        if let Ok(snap) = self.to_value().encode() {
            self.history.push((self.version, snap));
            if self.history.len() > 8 {
                self.history.remove(0);
            }
        }

        // Group experiences by learning candidate key.
        let mut by_candidate: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (idx, exp) in self.episodic.iter().enumerate() {
            for cand in &exp.learning_candidates {
                by_candidate.entry(cand.clone()).or_default().push(idx);
            }
        }

        let mut new_patterns = 0_u64;
        let mut new_procedures = 0_u64;
        let mut new_policies = 0_u64;
        let mut new_semantic = 0_u64;

        for (cand, indices) in &by_candidate {
            let total_occ: u64 = indices.iter().filter_map(|i| self.episodic.get(*i)).map(|e| e.occurrences).sum();
            let has_correction = indices
                .iter()
                .filter_map(|i| self.episodic.get(*i))
                .any(|e| e.experience_type == ExperienceType::Correction && e.outcome.corrected_success);

            // Promote evaluated experiences from Observed -> Candidate -> Validated.
            for &i in indices {
                if let Some(e) = self.episodic.get_mut(i)
                    && e.status == SupervisionState::Observed
                {
                    e.status = if total_occ >= 2 || has_correction {
                        SupervisionState::Validated
                    } else {
                        SupervisionState::Candidate
                    };
                }
            }

            // Section 5: extract a generalized pattern when >= 2 experiences (or a verified correction)
            // share a candidate concept.
            if (total_occ >= 2 || has_correction) && !self.patterns.iter().any(|p| p.category == *cand) {
                let ev_ids: Vec<String> = indices
                    .iter()
                    .filter_map(|i| self.episodic.get(*i))
                    .map(|e| e.experience_id.clone())
                    .collect();
                let first_exp = indices.first().and_then(|i| self.episodic.get(*i));
                let scope = first_exp.map_or_else(|| "project".to_owned(), |e| e.scope.clone());
                let problem = first_exp.map_or("", |e| e.context.problem.as_str());
                // 2026-11-25 (nothing pseudo): with no correction recorded, the lesson is what the
                // experiences themselves show (how many held), never a fixed sentence.
                let held_n = indices.iter().filter_map(|i| self.episodic.get(*i)).filter(|e| e.outcome.initial_success || e.outcome.corrected_success).count();
                let observed = format!("no correction recorded; {held_n} of {} experiences on {cand} held", indices.len());
                let correction = first_exp
                    .and_then(|e| (!e.context.correction.is_empty()).then_some(e.context.correction.as_str()))
                    .unwrap_or(observed.as_str());
                let mean_conf = indices
                    .iter()
                    .filter_map(|i| self.episodic.get(*i))
                    .map(|e| e.confidence)
                    .sum::<f64>()
                    / (indices.len().max(1) as f64);
                let conf = (mean_conf + 0.05 * (total_occ.saturating_sub(1) as f64)).min(0.99);

                // Supervisor validation check (Section 13):
                let status = if self.supervisor_validate_candidate(cand, &scope, conf, total_occ) {
                    SupervisionState::Validated
                } else {
                    SupervisionState::Candidate
                };

                let pat_id = format!("pat_{:04}", self.patterns.len() + 1);
                let steps = vec![
                    format!("Identify required preconditions for {cand}."),
                    format!("Apply lesson: {correction}."),
                    "Verify resulting state against tests or citations before finalizing.".to_owned(),
                ];
                self.patterns.push(LearnedPattern {
                    id: pat_id.clone(),
                    category: cand.clone(),
                    kind: KnowledgeKind::Procedural,
                    observation: if problem.is_empty() {
                        format!("{total_occ} experiences converged on {cand}.")
                    } else {
                        format!("Repeated observation ({total_occ} experiences): {problem}")
                    },
                    generalized_lesson: correction.to_owned(),
                    recommended_procedure: steps.clone(),
                    evidence_ids: ev_ids.clone(),
                    confidence: conf,
                    scope: scope.clone(),
                    status,
                });
                new_patterns += 1;

                // Consolidate into Semantic Memory (Section 6.2):
                if !self.semantic.iter().any(|s| s.concept == *cand) {
                    let related: Vec<String> = by_candidate
                        .keys()
                        .filter(|k| *k != cand)
                        .take(3)
                        .cloned()
                        .collect();
                    self.semantic.push(SemanticEntry {
                        id: format!("sem_{:04}", self.semantic.len() + 1),
                        concept: cand.clone(),
                        fact: correction.to_owned(),
                        related_concepts: related,
                        source_experiences: ev_ids.clone(),
                        scope: scope.clone(),
                        confidence: conf,
                        retrieval_priority: conf,
                        permanent_retention: has_correction || total_occ >= 3,
                        uses: total_occ,
                        updated_ms: now,
                    });
                    new_semantic += 1;
                }

                // Consolidate into Procedural Memory (Section 6.3):
                if status == SupervisionState::Validated && !self.procedural.iter().any(|p| p.category == *cand) {
                    self.procedural.push(ProceduralEntry {
                        id: format!("proc_{:04}", self.procedural.len() + 1),
                        category: cand.clone(),
                        trigger: format!("When working on {scope} tasks involving {cand}"),
                        steps: steps.clone(),
                        source_experiences: ev_ids.clone(),
                        pattern_id: pat_id,
                        scope: scope.clone(),
                        confidence: conf,
                        applications: total_occ,
                        // The experiences behind it whose outcome held (2026-11-25: counted, not
                        // set equal to the applications).
                        successes: held_n as u64,
                        status: SupervisionState::Validated,
                    });
                    new_procedures += 1;
                }

                // Propose a conditional, scoped Behavioral Policy (Section 8) when evidence >= 2:
                if status == SupervisionState::Validated && !self.policies.iter().any(|p| p.name == *cand) {
                    self.policies.push(BehavioralPolicy {
                        id: format!("pol_{:04}", self.policies.len() + 1),
                        name: cand.clone(),
                        condition: format!("Task in scope '{scope}' touching {cand}"),
                        applies_to: scope.clone(),
                        does_not_apply_to: "unrelated general conversation".to_owned(),
                        previous_behavior: first_exp
                            .map_or_else(|| "Direct execution without precondition check".to_owned(), |e| {
                                if e.context.initial_approach.is_empty() {
                                    "Direct execution without precondition check".to_owned()
                                } else {
                                    e.context.initial_approach.clone()
                                }
                            }),
                        observed_problem: if problem.is_empty() {
                            format!("Omission or failure around {cand}")
                        } else {
                            problem.to_owned()
                        },
                        new_behavior: steps,
                        expected_improvement: format!("Prevent recurring errors on {cand} within scope '{scope}'"),
                        confidence: conf,
                        evidence_count: total_occ,
                        status: SupervisionState::Validated,
                    });
                    new_policies += 1;
                }
            }
        }

        // Update evaluation dimensions & benchmark metrics (Sections 4 & 18):
        let total_exp = self.episodic.len();
        if total_exp > 0 {
            let success_count = self.episodic.iter().filter(|e| e.outcome.corrected_success || e.outcome.initial_success).count();
            let first_try_ok = self.episodic.iter().filter(|e| e.outcome.initial_success).count();
            let validated_count = self.episodic.iter().filter(|e| e.status == SupervisionState::Validated).count();
            // Counted from the experiences themselves (2026-11-25: no fixed formulas): the share
            // whose outcome held (first try or after a correction), the share validated, and the
            // share right the first time.
            self.current_eval.accuracy = success_count as f64 / total_exp as f64;
            self.metrics.learning_retention = validated_count as f64 / total_exp as f64;
            self.metrics.learning_efficiency = first_try_ok as f64 / total_exp as f64;
        }

        self.cycles_run = self.cycles_run.saturating_add(1);
        self.version = self.version.saturating_add(1);
        self.last_cycle_ms = now;
        self.refresh_cartridge_counts(now);

        obj(vec![
            ("version", n(self.version)),
            ("cycles_run", n(self.cycles_run)),
            ("new_patterns", n(new_patterns)),
            ("new_semantic", n(new_semantic)),
            ("new_procedures", n(new_procedures)),
            ("new_policies", n(new_policies)),
            ("validated_policies", n(self.policies.iter().filter(|p| p.status == SupervisionState::Validated).count() as u64)),
        ])
    }

    /// Learning Supervisor validation gate (Section 13): checks evidence trustworthiness, confidence
    /// threshold, scope enforcement, and ensures no regression against the baseline evaluation.
    #[must_use]
    pub fn supervisor_validate_candidate(&self, candidate: &str, scope: &str, confidence: f64, evidence_count: u64) -> bool {
        if candidate.trim().is_empty() || scope.trim().is_empty() {
            return false;
        }
        // Scope enforcement: prevent overly generic universal rules without multi-experience evidence.
        if scope == "*" && evidence_count < 3 {
            return false;
        }
        // Regression prevention: the measured held-out score must not have dropped more than 0.05
        // below the baseline (the score before the first measured cycle). Unmeasured: no figure to
        // compare, so the held-out gate of `run_gated_cycle` is the check.
        let no_regression = match (self.held_out_score, self.held_out_baseline) {
            (Some(now), Some(base)) => now + 0.05 >= base,
            _ => true,
        };
        no_regression && confidence >= 0.65 && evidence_count >= 1
    }

    /// Supervisor rollback (Section 13): restores the previous validated working state (or a
    /// specific historical version).
    /// # Errors
    /// If no snapshot exists for the requested version.
    pub fn rollback(&mut self, target_version: Option<u64>) -> Result<u64, Fault> {
        let idx = match target_version {
            Some(v) => self.history.iter().rposition(|(ver, _)| *ver == v),
            None => (!self.history.is_empty()).then(|| self.history.len() - 1),
        }
        .ok_or_else(|| bad("no previous continuous learning snapshot available to roll back to"))?;
        let (ver, snap) = self.history.remove(idx);
        let parsed = nmjson::parse(snap.as_bytes()).map_err(|e| bad(format!("snapshot parse: {e}")))?;
        let keep_history = self.history.clone();
        *self = Self::from_value(&parsed);
        self.history = keep_history;
        Ok(ver)
    }

    /// Whether an experience belongs to the held-out part of the measured gate. A record synced
    /// from the training store carries the store's own split (`split=held-out` in its provenance),
    /// so the same question stays held out whichever session captured it; any other record is
    /// split by the hash of its id (10 in 100 held out, `nmtype::examples::split_of`).
    #[must_use]
    pub fn is_held_out(rec: &ExperienceRecord) -> bool {
        if let Some(rest) = rec.context.provenance.split('|').find_map(|p| p.strip_prefix("split=")) {
            return nmtype::examples::Split::parse(rest) == Some(nmtype::examples::Split::Held);
        }
        nmtype::examples::split_of(&rec.experience_id, "engine") == nmtype::examples::Split::Held
    }

    /// Measured held-out score of what this state has learned (1 minus the mean squared error of
    /// its outcome predictions, so 1 is perfect and 0.75 is the score of knowing nothing). For each
    /// held-out experience, every validated pattern on one of its learning candidates predicts the
    /// chance of a first-try success from the outcomes of the experiences it was learned from
    /// (held-out ones never count as evidence), smoothed as (successes + 1) / (count + 2); with no
    /// such pattern the prediction is one half. `None` when `held` is empty.
    #[must_use]
    pub fn held_out_score(&self, held: &[&ExperienceRecord]) -> Option<f64> {
        if held.is_empty() {
            return None;
        }
        let held_ids: BTreeSet<&str> = held.iter().map(|h| h.experience_id.as_str()).collect();
        let by_id: BTreeMap<&str, &ExperienceRecord> = self.episodic.iter().map(|e| (e.experience_id.as_str(), e)).collect();
        let mut err = 0.0_f64;
        for h in held {
            let mut preds: Vec<f64> = Vec::new();
            for p in self.patterns.iter().filter(|p| p.status == SupervisionState::Validated && h.learning_candidates.contains(&p.category)) {
                let (mut ok, mut all) = (0_u64, 0_u64);
                for id in p.evidence_ids.iter().filter(|id| !held_ids.contains(id.as_str())) {
                    if let Some(e) = by_id.get(id.as_str()) {
                        all += 1;
                        ok += u64::from(e.outcome.initial_success);
                    }
                }
                #[expect(clippy::cast_precision_loss, reason = "counts")]
                preds.push((ok as f64 + 1.0) / (all as f64 + 2.0));
            }
            #[expect(clippy::cast_precision_loss, reason = "counts")]
            let p = if preds.is_empty() { 0.5 } else { preds.iter().sum::<f64>() / preds.len() as f64 };
            let y = if h.outcome.initial_success { 1.0 } else { 0.0 };
            err += (p - y) * (p - y);
        }
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let mse = err / held.len() as f64;
        Some(1.0 - mse)
    }

    /// Cycle B behind the measured held-out gate (the idle learning orchestrator and
    /// `POST /api/learning/cycle`). A candidate is learned from the experiences outside the
    /// held-out part only; when at least `MIN_HELD_OUT` held-out experiences exist, both the
    /// current state and the candidate are scored on them (`held_out_score`) and the candidate
    /// replaces the current state only if it scores higher. The measured score replaces the
    /// fixed-formula accuracy and generalization of whichever state is kept. With fewer held-out
    /// experiences there is nothing to measure on, and the plain cycle runs with its supervisor
    /// rule. The previous state is kept as a rollback snapshot either way.
    pub fn run_gated_cycle(&mut self) -> GatedCycle {
        let held: Vec<ExperienceRecord> = self.episodic.iter().filter(|e| Self::is_held_out(e)).cloned().collect();
        if held.len() < MIN_HELD_OUT {
            let cycle = self.run_learning_cycle();
            return GatedCycle { held_out: held.len(), current: None, candidate: None, promoted: true, measured: false, cycle };
        }
        let before = self.to_value().encode().ok();
        let mut cand = self.clone();
        cand.episodic.retain(|e| !Self::is_held_out(e));
        let cycle = cand.run_learning_cycle();
        cand.episodic.extend(held.iter().cloned());
        // The rollback snapshot the cycle pushed lacks the held-out experiences: the full state
        // before the cycle is kept in its place.
        cand.history.clone_from(&self.history);
        if let Some(snap) = before {
            cand.history.push((self.version, snap));
            if cand.history.len() > 8 {
                cand.history.remove(0);
            }
        }
        let refs: Vec<&ExperienceRecord> = held.iter().collect();
        let current = self.held_out_score(&refs).unwrap_or(0.75);
        let candidate = cand.held_out_score(&refs).unwrap_or(0.75);
        let promoted = candidate > current + 1e-9;
        let baseline = self.held_out_baseline.unwrap_or(current);
        let (measured, held_back) = (self.gated_measured + 1, self.gated_held_back + u64::from(!promoted));
        if promoted {
            *self = cand;
        }
        let kept = if promoted { candidate } else { current };
        self.held_out_score = Some(kept);
        self.held_out_baseline = Some(baseline);
        self.held_out_n = held.len() as u64;
        self.gated_measured = measured;
        self.gated_held_back = held_back;
        self.current_eval.generalization = kept;
        GatedCycle { held_out: held.len(), current: Some(current), candidate: Some(candidate), promoted, measured: true, cycle }
    }

    /// Recalls active validated procedures and behavioral policies matching `query` or `scope`
    /// so runtime retrieval and priming can apply what Nano has learned (Sections 7 & 8).
    #[must_use]
    pub fn active_guidance_for(&self, query: &str, scope: &str) -> Vec<String> {
        let low = query.to_lowercase();
        let mut notes = Vec::new();
        for pol in &self.policies {
            if pol.status != SupervisionState::Validated {
                continue;
            }
            let scope_ok = pol.applies_to == scope || pol.applies_to == "project" || low.contains(&pol.applies_to.to_lowercase());
            let name_words: Vec<&str> = pol.name.split(['_', '-', ' ']).filter(|w| w.len() >= 3).collect();
            if scope_ok && name_words.iter().any(|w| low.contains(&w.to_lowercase())) {
                notes.push(format!(
                    "policy [{} · scope {}]: {} -> {}",
                    pol.name,
                    pol.applies_to,
                    pol.observed_problem,
                    pol.new_behavior.join(" ")
                ));
            }
        }
        notes
    }

    /// Every figure this state has actually measured (2026-11-25, Luis Jake Gabriel III: nothing
    /// pseudo): each with its key, what it says, its value and the count it rests on. A figure with
    /// nothing under it is left out, never shown as a default.
    #[must_use]
    pub fn measures(&self) -> Vec<Measure> {
        let mut out = Vec::new();
        let total = self.episodic.len() as u64;
        if total > 0 {
            let held = self.episodic.iter().filter(|e| e.outcome.corrected_success || e.outcome.initial_success).count() as u64;
            let first = self.episodic.iter().filter(|e| e.outcome.initial_success).count() as u64;
            let valid = self.episodic.iter().filter(|e| e.status == SupervisionState::Validated).count() as u64;
            out.push(Measure::ratio("outcomes_held", "Outcomes that held", "experiences whose outcome held, first time or after a correction", held, total));
            out.push(Measure::ratio("first_try", "Right the first time", "experiences that succeeded without a correction", first, total));
            out.push(Measure::ratio("validated", "Experiences validated", "experiences the supervisor validated", valid, total));
        }
        let r = &self.retrieval;
        if r.retrieved_total > 0 {
            out.push(Measure::ratio("retrieval_precision", "Served places then used", "places served in context packs that were then read or edited", r.used_total, r.retrieved_total));
        }
        if r.used_total + r.missing_essential_total > 0 {
            out.push(Measure::ratio(
                "retrieval_recall",
                "Needed places that were served",
                "places used after a pack that the pack had served",
                r.used_total,
                r.used_total + r.missing_essential_total,
            ));
        }
        if self.metrics.corrections_total > 0 {
            out.push(Measure::ratio(
                "correction_recurrence",
                "Corrections that came back",
                "corrected mistakes that happened again",
                self.metrics.recurrences_total,
                self.metrics.corrections_total,
            ));
        }
        if let Some(sc) = self.held_out_score {
            out.push(Measure {
                key: "held_out",
                label: "Held-out prediction score",
                what: "1 minus the mean squared error of outcome predictions on held-out experiences (0.75 is knowing nothing)",
                value: sc,
                n: self.held_out_n,
                baseline: self.held_out_baseline,
            });
        }
        if self.gated_measured > 0 {
            out.push(Measure::ratio("held_back", "Cycles held back by the gate", "measured cycles whose candidate did not beat the current state", self.gated_held_back, self.gated_measured));
        }
        out
    }

    /// The state as the interface, the API and MCP show it: `to_value` with every evaluation and
    /// metric figure replaced by its measured value or null (never a starting default), the list of
    /// measured figures, and each continual cartridge's standing from what it holds.
    #[must_use]
    pub fn view_value(&self) -> Value {
        let ms = self.measures();
        let get = |k: &str| ms.iter().find(|m| m.key == k).map(|m| m.value);
        let num = |x: Option<f64>| x.map_or(Value::Null, fnum);
        let relevance = match (get("retrieval_precision"), get("retrieval_recall")) {
            (Some(p), Some(r)) => Some(0.5 * (p + r)),
            _ => None,
        };
        let dims = [("accuracy", get("outcomes_held")), ("relevance", relevance), ("generalization", get("held_out"))];
        let measured: Vec<f64> = dims.iter().filter_map(|(_, x)| *x).collect();
        #[expect(clippy::cast_precision_loss, reason = "a handful of figures")]
        let overall = (!measured.is_empty()).then(|| measured.iter().sum::<f64>() / measured.len() as f64);
        let mut cur: Vec<(&str, Value)> = ["accuracy", "relevance", "completeness", "efficiency", "consistency", "instruction_adherence", "reconstruction_fidelity", "generalization"]
            .iter()
            .map(|k| (*k, num(dims.iter().find(|(d, _)| d == k).and_then(|(_, x)| *x))))
            .collect();
        cur.push(("overall", num(overall)));
        let base = vec![("generalization", num(self.held_out_baseline)), ("overall", num(self.held_out_baseline))];
        let metrics = obj(vec![
            ("learning_retention", num(get("validated"))),
            ("correction_recurrence", num(get("correction_recurrence"))),
            ("retrieval_precision", num(get("retrieval_precision"))),
            ("retrieval_recall", num(get("retrieval_recall"))),
            ("context_fidelity", Value::Null),
            ("reconstruction_accuracy", Value::Null),
            ("policy_effectiveness", Value::Null),
            ("transfer_performance", Value::Null),
            ("regression_rate", num(get("held_back"))),
            ("learning_efficiency", num(get("first_try"))),
            ("corrections_total", n(self.metrics.corrections_total)),
            ("recurrences_total", n(self.metrics.recurrences_total)),
            ("last_recurrence_stage", s(&self.metrics.last_recurrence_stage)),
        ]);
        let carts: Vec<Value> = self
            .cartridges
            .iter()
            .map(|c| {
                let mut v = c.to_value();
                let standing = if c.entries == 0 {
                    "empty"
                } else if self.cycles_run == 0 {
                    "observed"
                } else {
                    c.status.name()
                };
                if let Value::Object(m) = &mut v {
                    m.insert("status".into(), s(standing));
                    m.insert("confidence".into(), Value::Null);
                }
                v
            })
            .collect();
        let mut v = self.to_value();
        if let Value::Object(m) = &mut v {
            // Only what the serving path reads (`drone_path::RetrievalWeights`): the four fusion
            // weights and the feedback totals. The threshold, depth, diversity and recency values
            // and the context shares drive nothing yet, so they are not shown as learned settings.
            let r = &self.retrieval;
            m.insert(
                "retrieval".into(),
                obj(vec![
                    ("lexical_weight", fnum(r.lexical_weight)),
                    ("semantic_weight", fnum(r.semantic_weight)),
                    ("structural_weight", fnum(r.structural_weight)),
                    ("relationship_weight", fnum(r.relationship_weight)),
                    ("retrieved_total", n(r.retrieved_total)),
                    ("used_total", n(r.used_total)),
                    ("irrelevant_total", n(r.irrelevant_total)),
                    ("missing_essential_total", n(r.missing_essential_total)),
                ]),
            );
            m.remove("context");
            m.insert("current_eval".into(), obj(cur));
            m.insert("baseline_eval".into(), obj(base));
            m.insert("metrics".into(), metrics);
            m.insert("cartridges".into(), Value::Array(carts));
            m.insert("measures".into(), Value::Array(ms.iter().map(Measure::to_value).collect()));
        }
        v
    }

    fn refresh_cartridge_counts(&mut self, now: u64) {
        let measured = self.measures().len() as u64;
        for c in &mut self.cartridges {
            let (entries, ev) = match c.domain.as_str() {
                "knowledge" => (self.semantic.len() as u64, self.episodic.len() as u64),
                "procedural" => (self.procedural.len() as u64, self.patterns.len() as u64),
                "behavioral" => (self.policies.len() as u64, self.policies.iter().map(|p| p.evidence_count).sum()),
                "retrieval" => (u64::from(self.retrieval.retrieved_total > 0), self.retrieval.retrieved_total),
                "interaction" => (
                    self.preferences.len() as u64,
                    self.episodic
                        .iter()
                        .filter(|e| matches!(e.experience_type, ExperienceType::Correction | ExperienceType::Preference))
                        .count() as u64,
                ),
                "evaluation" => (measured, self.episodic.len() as u64),
                _ => (0, 0),
            };
            if entries != c.entries || ev != c.evidence_count {
                c.entries = entries;
                c.evidence_count = ev;
                c.version = self.version;
                c.updated_ms = now;
            }
        }
    }

    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("version", n(self.version)),
            ("cycles_run", n(self.cycles_run)),
            ("last_cycle_ms", n(self.last_cycle_ms)),
            (
                "levels",
                obj(vec![
                    ("experiential", obj(vec![("episodes", n(self.episodic.len() as u64)), ("cycle", s("immediate"))])),
                    (
                        "knowledge",
                        obj(vec![
                            ("patterns", n(self.patterns.len() as u64)),
                            ("semantic", n(self.semantic.len() as u64)),
                            ("procedural", n(self.procedural.len() as u64)),
                            ("cycle", s("consolidation")),
                        ]),
                    ),
                    (
                        "behavioral",
                        obj(vec![
                            ("policies", n(self.policies.len() as u64)),
                            ("preferences", n(self.preferences.len() as u64)),
                            ("retrieval_adapted", Value::Bool(self.retrieval.retrieved_total > 0)),
                        ]),
                    ),
                ]),
            ),
            ("episodic", Value::Array(self.episodic.iter().map(ExperienceRecord::to_value).collect())),
            ("patterns", Value::Array(self.patterns.iter().map(LearnedPattern::to_value).collect())),
            ("semantic", Value::Array(self.semantic.iter().map(SemanticEntry::to_value).collect())),
            ("procedural", Value::Array(self.procedural.iter().map(ProceduralEntry::to_value).collect())),
            ("preferences", Value::Array(self.preferences.iter().map(PreferenceEntry::to_value).collect())),
            ("policies", Value::Array(self.policies.iter().map(BehavioralPolicy::to_value).collect())),
            ("retrieval", self.retrieval.to_value()),
            ("context", self.context.to_value()),
            ("baseline_eval", self.baseline_eval.to_value()),
            ("current_eval", self.current_eval.to_value()),
            ("metrics", self.metrics.to_value()),
            ("cartridges", Value::Array(self.cartridges.iter().map(ContinualCartridge::to_value).collect())),
            ("rollback_versions", Value::Array(self.history.iter().map(|(v, _)| n(*v)).collect())),
            ("held_out_score", self.held_out_score.map_or(Value::Null, fnum)),
            ("held_out_baseline", self.held_out_baseline.map_or(Value::Null, fnum)),
            ("held_out_n", n(self.held_out_n)),
            ("gated_measured", n(self.gated_measured)),
            ("gated_held_back", n(self.gated_held_back)),
            ("synced", Value::Array(self.synced.iter().map(|x| s(x)).collect())),
        ])
    }

    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let map_arr = |k: &str| v.get(k).and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default();
        let carts: Vec<ContinualCartridge> = map_arr("cartridges").iter().map(ContinualCartridge::from_value).collect();
        Self {
            version: u64_of(v, "version").max(1),
            cycles_run: u64_of(v, "cycles_run"),
            last_cycle_ms: u64_of(v, "last_cycle_ms"),
            episodic: map_arr("episodic").iter().map(ExperienceRecord::from_value).collect(),
            patterns: map_arr("patterns").iter().map(LearnedPattern::from_value).collect(),
            semantic: map_arr("semantic").iter().map(SemanticEntry::from_value).collect(),
            procedural: map_arr("procedural").iter().map(ProceduralEntry::from_value).collect(),
            preferences: map_arr("preferences").iter().map(PreferenceEntry::from_value).collect(),
            policies: map_arr("policies").iter().map(BehavioralPolicy::from_value).collect(),
            retrieval: v.get("retrieval").map_or_else(|_| RetrievalAdaptation::default(), RetrievalAdaptation::from_value),
            context: v.get("context").map_or_else(|_| ContextAdaptation::default(), ContextAdaptation::from_value),
            baseline_eval: v.get("baseline_eval").map_or_else(|_| EvaluationDimensions::default(), EvaluationDimensions::from_value),
            current_eval: v.get("current_eval").map_or_else(|_| EvaluationDimensions::default(), EvaluationDimensions::from_value),
            metrics: v.get("metrics").map_or_else(|_| BenchmarkMetrics::default(), BenchmarkMetrics::from_value),
            cartridges: if carts.is_empty() {
                CARTRIDGE_DOMAINS.iter().map(|d| ContinualCartridge::new(d)).collect()
            } else {
                carts
            },
            held_out_score: fopt_of(v, "held_out_score"),
            held_out_baseline: fopt_of(v, "held_out_baseline"),
            held_out_n: u64_of(v, "held_out_n"),
            gated_measured: u64_of(v, "gated_measured"),
            gated_held_back: u64_of(v, "gated_held_back"),
            synced: vec_str_of(v, "synced").into_iter().collect(),
            history: Vec::new(),
        }
    }
}

/// Section check of `nmlearn.engine`: verifies Cycle A experience acquisition, duplicate
/// consolidation, delayed outcome updates, Cycle B pattern and knowledge extraction across 3
/// related dependency-initialization experiences, 5-category learning memory integration,
/// retrieval and 96K context reconstruction backward trace, conditional behavioral policy
/// generation, 6 continual learning cartridges, correction recurrence tracking, and supervisor
/// rollback.
pub(crate) fn check() -> Result<(), String> {
    let dir = nmtype::check_dir("nmlearn-engine").map_err(|e| e.to_string())?;
    let mut eng = ContinuousLearningEngine::open(&dir);

    // 1. Three related dependency-initialization experiences (Section 5 example):
    for (i, prob) in [
        (1, "Dependencies initialized in the wrong order"),
        (2, "Service accessed before initialization"),
        (3, "Component accessed unavailable configuration object"),
    ] {
        let rec = ExperienceRecord {
            experience_id: format!("exp_{i:06}"),
            timestamp_ms: 1_700_000_000_000 + i * 1000,
            experience_type: ExperienceType::Correction,
            signal: FeedbackSignal::Explicit,
            task: "typescript_implementation".to_owned(),
            scope: "software_implementation".to_owned(),
            context: ExperienceContext {
                problem: prob.to_owned(),
                initial_approach: "Direct implementation".to_owned(),
                correction: "Validate initialization and availability before executing dependent operations".to_owned(),
                places: vec![format!("src/service_{i}.ts:10-25")],
                provenance: "user_correction".to_owned(),
            },
            outcome: ExperienceOutcome {
                initial_success: false,
                corrected_success: true,
                delayed_updated: false,
                tokens_used: 1400,
                ms_used: 550,
            },
            learning_candidates: vec!["dependency_initialization".to_owned(), "precondition_validation".to_owned()],
            confidence: 0.95,
            status: SupervisionState::Observed,
            occurrences: 1,
        };
        eng.observe(rec).map_err(|e| e.to_string())?;
    }

    // 2. Duplicate experience is consolidated rather than duplicated (Section 2):
    let dup = eng.episodic.first().cloned().ok_or("nmlearn.engine: missing first episode")?;
    eng.observe(dup).map_err(|e| e.to_string())?;
    if eng.episodic.len() != 3 || eng.episodic.first().map_or(0, |e| e.occurrences) != 2 {
        return Err(format!("nmlearn.engine: duplicate experience not consolidated (len {})", eng.episodic.len()));
    }

    // 3. Delayed outcome update (Section 3):
    if !eng.record_delayed_outcome("exp_000002", true, Some("Verified in integration suite")) {
        return Err("nmlearn.engine: delayed outcome update failed".into());
    }

    // 4. Retrieval and 96K context reconstruction backward trace (Sections 7 & 9.C):
    eng.record_retrieval_feedback(5, 3, 2, 1);
    let trace = ReconstructionTrace {
        present_in_source: true,
        identified_in_compression: true,
        preserved_in_representation: false,
        retrieved_in_reconstruction: false,
        interpreted_in_reasoning: false,
    };
    if trace.failure_stage() != "compressed_representation" {
        return Err(format!("nmlearn.engine: unexpected failure stage {}", trace.failure_stage()));
    }
    eng.record_context_trace(&["t0", "t2"], &["t5"], Some(&trace));

    // 5. Run Cycle B consolidation (Sections 5, 6, 8, 10, 11, 13):
    let summary = eng.run_learning_cycle();
    if eng.patterns.len() < 2 || eng.procedural.len() < 2 || eng.policies.len() < 2 || eng.semantic.len() < 2 {
        return Err(format!(
            "nmlearn.engine: cycle extracted patterns={}, proc={}, pol={}, sem={}; summary={:?}",
            eng.patterns.len(),
            eng.procedural.len(),
            eng.policies.len(),
            eng.semantic.len(),
            summary
        ));
    }

    // 6. Active guidance recall (Sections 7 & 8):
    let guidance = eng.active_guidance_for("How should dependency_initialization be checked?", "software_implementation");
    if guidance.is_empty() {
        return Err("nmlearn.engine: active_guidance_for returned no validated policy".into());
    }

    // 7. Multi-session cross-reload progression: Session 2 applies validated guidance on first try,
    // raising learning_efficiency with zero new recurrences, and RetrievalAdaptation shifts weights.
    let eff_session1 = eng.metrics.learning_efficiency;
    let rec_session1 = eng.metrics.recurrences_total;
    let sem_w_before = eng.retrieval.semantic_weight;
    eng.append_parameter_telemetry(&dir, "check dependency init", "software_implementation", (true, 512, 420), &["reader-base@1".to_owned()]);
    let tel_path = dir.join(DIR).join("param-telemetry.jsonl");
    if !tel_path.is_file() || std::fs::read_to_string(&tel_path).unwrap_or_default().trim().is_empty() {
        return Err("nmlearn.engine: parameter telemetry log was not written".into());
    }
    eng.save(&dir).map_err(|e| e.to_string())?;
    let mut loaded = ContinuousLearningEngine::open(&dir);
    if loaded.patterns.len() != eng.patterns.len() || loaded.cartridges.len() != 6 || loaded.history.is_empty() {
        return Err("nmlearn.engine: save/open round-trip mismatch or missing snapshots".into());
    }
    let session2_guidance = loaded.active_guidance_for("dependency_initialization check before service call", "software_implementation");
    if session2_guidance.is_empty() {
        return Err("nmlearn.engine: session 2 did not recall validated guidance after reload".into());
    }
    let s2_rec = ExperienceRecord {
        experience_id: "exp_session2_001".to_owned(),
        timestamp_ms: 1_700_000_090_000,
        experience_type: ExperienceType::SuccessfulOutcome,
        signal: FeedbackSignal::Explicit,
        task: "typescript_implementation".to_owned(),
        scope: "software_implementation".to_owned(),
        context: ExperienceContext {
            problem: String::new(),
            initial_approach: "Applied validated dependency_initialization policy first".to_owned(),
            correction: String::new(),
            places: vec!["src/service_4.ts:10-20".to_owned()],
            provenance: "session2_verified".to_owned(),
        },
        outcome: ExperienceOutcome {
            initial_success: true,
            corrected_success: true,
            delayed_updated: false,
            tokens_used: 900,
            ms_used: 380,
        },
        learning_candidates: vec!["dependency_initialization".to_owned()],
        confidence: 0.98,
        status: SupervisionState::Observed,
        occurrences: 1,
    };
    loaded.observe(s2_rec).map_err(|e| e.to_string())?;
    loaded.record_retrieval_feedback(8, 6, 0, 1);
    let _ = loaded.run_learning_cycle();
    if loaded.metrics.recurrences_total != rec_session1
        || loaded.metrics.learning_efficiency <= eff_session1
        || loaded.retrieval.semantic_weight <= sem_w_before
    {
        return Err(format!(
            "nmlearn.engine: session 2 progression failed (rec {}->{}, eff {:.3}->{:.3}, sem_w {:.3}->{:.3})",
            rec_session1,
            loaded.metrics.recurrences_total,
            eff_session1,
            loaded.metrics.learning_efficiency,
            sem_w_before,
            loaded.retrieval.semantic_weight
        ));
    }
    let rolled_back_to = loaded.rollback(Some(1)).map_err(|e| e.to_string())?;
    if rolled_back_to != 1 || !loaded.patterns.is_empty() {
        return Err(format!("nmlearn.engine: cross-reload rollback to version 1 gave version {rolled_back_to}"));
    }
    Ok(())
}
