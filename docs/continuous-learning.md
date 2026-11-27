# Neuromorph Nano: Continuous Learning Architecture

This document is the design of continuous learning in Neuromorph Nano, followed by what of it the
v0.1.0 code actually does. Sections 1 to 15 and 17 to 20 describe the design; where a part of the
design is not built yet, it says so. Section 16 and the section "What runs in v0.1.0" describe the
code as it is.

Continuous learning is one of the most important components of Neuromorph Nano. It is what allows the system to evolve through experience rather than behaving like a static model that starts from the same baseline every time.

The central idea is:

Neuromorph Nano should continuously improve how it understands, remembers, retrieves, reasons, and performs tasks through accumulated experience.

It should not need to retrain its foundational model every time it learns something new.

Instead, learning takes place across several interconnected layers:

* Experience acquisition
* Observation and feedback
* Performance evaluation
* Error identification
* Knowledge extraction
* Memory consolidation
* Retrieval optimization
* Behavioral adaptation
* Policy evolution
* Validation and regression prevention

There is an important distinction here: persistent memory and continuous learning are related, but they are not the same thing.

Memory determines what the system retains. Continuous learning determines how the system uses accumulated experience to improve future performance.

For Neuromorph Nano, continuous learning is designed as a system-wide process that operates alongside the context architecture, sensory drones, tentacles, Globalized Context Stack, and Core Brain.

## 1. The fundamental learning model

Neuromorph Nano's learning is divided into three levels.

### Level 1: Experiential learning (Short-term adaptation)

The system learns from individual interactions, corrections, successful outcomes, failures, and observations.

### Level 2: Knowledge learning (Long-term learning)

The system extracts useful information from experiences, consolidates it into persistent memory, identifies patterns, and improves future retrieval.

### Level 3: Behavioral learning (System-wide adaptation)

The system improves its strategies, workflows, decision-making policies, context allocation, and methods of solving problems.

These levels operate together.

For example, imagine Nano repeatedly encounters a programming problem:

* It initially makes an incorrect implementation.
* The user corrects it.
* Nano records the correction and the circumstances surrounding it.
* It recognizes that the error resulted from overlooking a particular dependency.
* It identifies similar past errors.
* It develops a better verification procedure.
* It tests that procedure against previous examples.
* It adopts the improved procedure if it performs better.

The important part is that Nano does not just remember the correction. It learns something about how to avoid making that type of mistake.

That is the difference between accumulating information and actually learning.

## 2. Experience acquisition: collecting learning material

Continuous learning begins with collecting experiences from the system's normal operation.

Nano does not treat every interaction as an equally valuable learning event. It captures different types of experiences:

| Experience | Example |
|---|---|
| User correction | User identifies an incorrect answer |
| User preference | User prefers a particular response structure |
| Successful outcome | A generated solution passes its tests |
| Failed outcome | A generated solution fails validation |
| Reasoning observation | A particular approach consistently works |
| Retrieval observation | A retrieved memory was irrelevant |
| Context observation | Important information was omitted during compression |
| Agent feedback | One component identifies another component's mistake |
| Environmental observation | An external tool returns unexpected results |
| Repeated behavior | A pattern emerges across multiple interactions |

### The experience record

Each meaningful experience produces a structured record. For example:

```json
{
  "experience_id": "exp_000184",
  "timestamp": "2026-11-20T10:30:00Z",
  "experience_type": "correction",
  "task": "typescript_implementation",
  "context": {
    "problem": "Incorrect dependency handling",
    "initial_approach": "Direct implementation",
    "correction": "Check dependency initialization first"
  },
  "outcome": {
    "initial_success": false,
    "corrected_success": true
  },
  "learning_candidates": [
    "dependency_initialization",
    "precondition_validation"
  ],
  "confidence": 0.95,
  "status": "pending_evaluation"
}
```

This is an illustrative schema, not a requirement to store every interaction in exactly this format.

The system preserves enough original context to understand why something happened. A correction without its surrounding circumstances can become misleading knowledge. For example, a user preference might apply to one particular project rather than every situation. Therefore, every experience retains its provenance, context, scope, and outcome.

### Experience filtering

Before an experience enters the learning pipeline, Nano evaluates:

* Is this information genuinely new?
* Is it a correction or just a different phrasing?
* Is it likely to be useful again?
* Does it contradict existing knowledge?
* Is it specific to one task?
* Is it a temporary circumstance?
* Is it reliable enough to retain?

Repeated identical observations are consolidated rather than endlessly duplicated.

## 3. Observation and feedback

This is the stage where Nano determines what happened, not just what was said. It distinguishes between several different signals:

### Explicit feedback

The user directly says something was wrong, correct, useful, or undesirable.

### Implicit feedback

The outcome provides evidence about performance. For example:

* A program compiles successfully.
* A test suite fails.
* A search produces irrelevant results.
* A task requires repeated corrections.
* A user accepts a proposed solution without changes.

### Internal feedback

One part of Neuromorph Nano evaluates another part. For example:

* A retrieval component identifies that the selected context was irrelevant.
* A reasoning component identifies missing information.
* A verification component identifies an unsupported conclusion.
* A context compression component identifies information that was lost during reconstruction.

### Delayed feedback

Some outcomes cannot be evaluated immediately. A solution might initially appear successful but fail when used in a larger project. Nano supports delayed outcome records so that the original experience can be updated when new evidence becomes available.

In v0.1.0, delayed evidence arrives through the evidence ledger: each context pack is judged by what followed it in the same session (the cited places read or edited, a later build or test passing or failing, an edit undone, the next prompt). The engine also has a call for updating an experience after the fact (`record_delayed_outcome`), but the serving path does not call it yet.

### Why this matters

Nano must not confuse an answer that sounds convincing with an answer that is correct. It uses outcome-based feedback wherever possible. A successful execution is stronger evidence than an internal confidence estimate.

## 4. Performance evaluation

After collecting feedback, Nano evaluates its own performance as a dedicated process rather than allowing the reasoning model to decide unilaterally whether it has improved.

| Dimension | What it measures |
|---|---|
| Accuracy | Was the result correct? |
| Relevance | Was the right information retrieved? |
| Completeness | Was important information omitted? |
| Efficiency | How much context and computation were required? |
| Consistency | Does performance remain stable? |
| Instruction adherence | Were requirements followed? |
| Reconstruction fidelity | Did compressed information retain its meaning? |
| Generalization | Does the improvement work on similar unseen tasks? |

### Establishing a performance baseline

Nano retains previous performance measurements (for example, before adaptation vs. after adaptation) and measures whether performance on unrelated tasks deteriorates. An improvement in one category is not automatically considered a system-wide improvement.

## 5. Knowledge extraction: turning experience into learning

Raw experiences are not yet knowledge. Nano identifies what can be learned from them.

Consider these three experiences:

* A TypeScript application failed because dependencies were initialized in the wrong order.
* A second application failed because a service was accessed before initialization.
* A third application failed because a component accessed an unavailable configuration object.

Individually, these are three separate incidents. However, Nano identifies a shared underlying pattern:

> Accessing a dependency before its initialization or availability has been established.

It then extracts a generalized lesson:

```text
LEARNED PATTERN
Category:
  Dependency initialization
Observation:
  Several implementation failures resulted from
  accessing dependencies before confirming availability.
Generalized lesson:
  Validate initialization and availability before
  executing dependent operations.
Recommended procedure:
  1. Identify required dependencies.
  2. Verify initialization.
  3. Verify availability.
  4. Execute dependent operations.
  5. Validate the resulting state.
Evidence:
  3 independent experiences.
Confidence:
  High
Scope:
  Software implementation.
```

The system transforms individual experiences into reusable concepts, procedures, and behavioral improvements.

### Three types of extracted knowledge

* **A. Declarative knowledge**: Information about things (e.g., a particular API requires authentication; a project uses a specific database; a user prefers a particular terminology).
* **B. Procedural knowledge**: Information about how to do things (e.g., how to debug a particular class of errors; how to reconstruct compressed context; how to validate a software implementation).
* **C. Strategic knowledge**: Information about which approach works best under particular circumstances (e.g., when to perform additional retrieval; when to split a task into smaller operations; when to request additional evidence; when a shorter context packet produces better results).

## 6. Persistent memory and knowledge consolidation

Nano organizes learning memory into five interconnected categories:

1. **Episodic memory**: Individual experiences, events, corrections, task outcomes, and their original context.
2. **Semantic memory**: Consolidated facts, concepts, relationships, and generalized knowledge.
3. **Procedural memory**: Learned procedures, successful workflows, problem-solving methods, and reusable strategies.
4. **Preference and interaction memory**: User-specific instructions, preferences, interaction patterns, and scope-specific behavioral requirements.
5. **Adaptive memory**: Learned retrieval strategies, confidence adjustments, decision policies, and performance improvements.

These are linked rather than isolated databases: a procedural memory entry is traceable to the experiences that established it; a semantic fact retains its source and confidence; a preference retains its scope.

### Memory consolidation

Consolidation transforms accumulated experiences into organized knowledge:

* Identifying duplicate experiences.
* Identifying related experiences.
* Finding common patterns.
* Extracting generalized knowledge.
* Resolving contradictions.
* Updating existing knowledge when justified.
* Establishing relationships between concepts.
* Updating retrieval indexes.
* Preserving original evidence and provenance.

Consolidation never destroys the original experiences; original records remain available for verification and future reinterpretation.

### Memory decay and reinforcement

Nano adjusts memory priority according to recency, frequency of use, historical usefulness, reliability, repeated confirmation, relevance to current tasks, and whether information has been superseded. Low-frequency information is not automatically deleted: a rare but important fact can retain permanent retention through separate retention and retrieval-priority mechanisms.

## 7. Retrieval learning: improving what Nano remembers at runtime

Nano improves not only its memory contents, but also its ability to locate the right memories through actual task outcomes.

| Retrieval dimension | Learning objective |
|---|---|
| Relevance | Select information related to the actual task |
| Accuracy | Avoid retrieving misleading information |
| Completeness | Retrieve all necessary supporting context |
| Priority | Surface the most important memories first |
| Diversity | Avoid retrieving redundant versions of the same information |
| Temporal relevance | Prefer current information when appropriate |
| Relationship awareness | Retrieve connected information that may not share obvious wording |

### Retrieval feedback loop

* Nano retrieves five memories.
* The reasoning component uses three.
* Two of those memories prove irrelevant.
* A missing sixth memory turns out to have been essential.
* The system records the retrieval outcome.
* The retrieval system identifies why the selection was poor and updates its retrieval strategy.
* Similar future tasks use the improved selection procedure.

### What adapts

Nano learns and adjusts:

* Retrieval ranking weights.
* Similarity thresholds.
* Memory importance scores.
* Query expansion strategies.
* Context selection policies.
* Memory relationship traversal.
* Retrieval depth.
* The balance between recent and historical information.

## 8. Behavioral learning: improving how Nano operates

This layer turns accumulated knowledge into better operating behavior:

```text
BEHAVIORAL POLICY
Condition:
  Complex task with relevant historical context.
Previous behavior:
  Begin reasoning immediately.
Observed problem:
  Important historical information is frequently omitted.
New behavior:
  1. Identify relevant context requirements.
  2. Search persistent memory.
  3. Retrieve supporting experiences.
  4. Evaluate information completeness.
  5. Begin reasoning when sufficient context exists.
Expected improvement:
  Reduced context-related errors.
```

### Types of behavioral adaptations

Nano can improve task decomposition, reasoning strategies, retrieval decisions, context allocation, information prioritization, verification procedures, tool selection, error recovery, decision thresholds, communication strategies, and computational resource allocation.

### Policy learning is conditional

Nano does not turn every successful experience into a universal rule. Policies carry conditions and scopes:

```text
Policy:
  Use additional dependency validation.
Applies to:
  Software implementation.
Does not automatically apply to:
  General conversation.
Confidence:
  0.91
Evidence:
  12 successful applications.
Status:
  Validated.
```

## 9. Continuous learning and the multi-window context architecture

Continuous learning is designed to operate across Nano's parallel tentacle context windows rather than as an external feature. The Neuropod has 8 tentacles, and each tentacle window holds 12,000 tokens by default (the `budget.tentacle_tokens` control, 2,000 to 32,000), so about 96,000 tokens are read per question at the default.

Status in v0.1.0: of this section, only retrieval feedback per context pack is learned (which served places were then used, passed over or missed; section 7). Learning per window, across windows, from reconstruction and of context allocation (A to D below) is design, not built. The engine has a call for recording a five-stage reconstruction trace (`record_context_trace`), but nothing in the serving path feeds it yet.

### A. Learning at the individual context level

Each context window produces localized learning signals about which information was relevant, redundant, lost, relational, or worthy of retention and handover.

### B. Learning across context windows

Nano evaluates the collective union across windows: which windows provided useful vs. redundant information, which missed relationships, what was lost during aggregation, and which combinations produced the best final result.

### C. Learning from context reconstruction

When compressed context omits a technical constraint during reconstruction, Nano traces the omission backward:

1. Was the constraint present in the original source?
2. Was it identified as important during compression?
3. Was it preserved in the compressed representation?
4. Was it retrieved during reconstruction?
5. Was it interpreted correctly during final reasoning?

This identifies the exact stage where information was lost so the learning process improves that stage.

### D. Learning context allocation

Nano measures and learns task-specific context allocation policies across historical information, current instructions, technical documentation, and code definitions.

## 10. Learning cartridges: separating different types of accumulated learning

Nano maintains specialized modular continual learning cartridges rather than placing everything into one monolithic structure:

* **Knowledge cartridge**: Facts, concepts, relationships, learned information.
* **Procedural cartridge**: Methods, workflows, techniques, procedures.
* **Behavioral cartridge**: Policies, decision rules, adaptive strategies.
* **Retrieval cartridge**: Ranking, relevance, retrieval preferences.
* **Interaction cartridge**: User preferences, corrections, interaction patterns.
* **Evaluation cartridge**: Benchmarks, historical performance, known failures.

One experience (such as a programming correction) can contribute to multiple learning domains simultaneously.

In v0.1.0 these six continual learning cartridges are records inside the engine's state file (`continual/state.json` in the project's memory folder; ids `clc-knowledge`, `clc-procedural`, `clc-behavioral`, `clc-retrieval`, `clc-interaction`, `clc-evaluation`), each with a version, a status, its entry and evidence counts and a confidence. They hold no model weights. Weights learned from your work are a different thing: the learned units (section "Learned units and compartments" below).

### Cartridge lifecycle

Every cartridge supports creation, experience accumulation, knowledge extraction, consolidation, validation, versioning, updating, rollback, and archiving. Cartridges require evidence and validation before receiving authority.

## 11. The actual continuous learning cycle

Nano implements two complementary learning cycles:

* **Cycle A: Immediate learning**: Operates during normal interaction and captures useful feedback immediately without waiting for a separate training process.
* **Cycle B: Consolidation learning**: Operates periodically or when sufficient evidence accumulates, turning individual observations into general improvements.

| Operation | Timing |
|---|---|
| Capture an explicit correction | Immediately |
| Record a task outcome | When available |
| Update a temporary task context | Immediately |
| Store a verified fact | After validation |
| Extract recurring patterns | Background |
| Consolidate related experiences | Background |
| Generate behavioral improvements | Background |
| Run regression evaluations | Before activation |
| Update retrieval indexes | After memory changes |

## 12. External continuous learning and internal model adaptation

* **A. External continuous learning (Foundation)**: The foundational model weights remain unchanged. Nano learns through structured external memory, retrieval adaptation, preference and context policies, consolidation, workflow improvements, and supervisory rules.
* **B. Internal model adaptation (Optional independent mechanism)**: Low-rank or parameter-mapped adapter cartridges over frozen base weights, kept independent so catastrophic forgetting, parameter interference, and regressions are isolated and reversible.

In v0.1.0, A is the continual learning engine (section 16), and B is the learned units and the preference cartridge: each sits beside the base cartridge it was fitted on and is never merged into it, so pulling it out gives the MicroModel back as it was.

## 13. Learning supervision and protection against bad learning

A dedicated **Learning Supervisor** oversees the learning process:

| Responsibility | Function |
|---|---|
| Evidence verification | Determine whether learning material is trustworthy |
| Confidence management | Assign confidence based on supporting evidence |
| Contradiction detection | Identify conflicting memories or policies |
| Scope enforcement | Prevent specialized knowledge becoming universal rules |
| Regression detection | Identify degraded performance |
| Change management | Approve, reject, or defer adaptations |
| Version management | Preserve previous working states |
| Rollback | Restore previous behavior when necessary |

### Confidence and evidence states

1. **Observed**: An experience has been recorded but not evaluated.
2. **Candidate**: A potentially useful lesson has been extracted.
3. **Under evaluation**: The proposed learning is being tested.
4. **Validated**: The improvement has passed its applicable evaluation criteria.
5. **Rejected**: The proposed learning was incorrect or produced regressions.

### Preventing regression

Before deploying a behavioral improvement, Nano evaluates whether it improves the target task, preserves previous performance, introduces no new errors, generalizes to unseen examples, and remains effective within its scope.

## 14. Component communication

The continuous learning subsystem is designed as twelve modular components exchanging structured learning events. In v0.1.0 these are responsibilities inside one engine (`crates/nmlearn/src/engine.rs`) and the idle learning orchestrator (`crates/nmlearn/src/idle.rs`), not twelve separate modules:

| Component | Responsibility |
|---|---|
| Experience Collector | Captures learning events |
| Feedback Processor | Interprets explicit, implicit, internal, and delayed feedback |
| Outcome Evaluator | Measures actual task performance |
| Pattern Extractor | Identifies recurring patterns |
| Knowledge Extractor | Converts experiences into reusable knowledge |
| Memory Consolidator | Organizes and integrates knowledge |
| Retrieval Learner | Improves memory selection |
| Context Learner | Improves context compression, allocation, and reconstruction |
| Policy Learner | Develops improved behavioral strategies |
| Learning Supervisor | Oversees validation and change management |
| Evaluation Engine | Tests learning outcomes |
| Cartridge Manager | Maintains learning modules and versions |

## 15. Five development stages and local implementation

These stages are the design's order of work. Where each stands in v0.1.0 is in section 16 and "What runs in v0.1.0".

* **Stage 1: Experience and memory** (structured experience records, episodic, semantic, procedural, and preference memory, provenance, confidence, persistent storage).
* **Stage 2: Feedback and evaluation** (explicit feedback capture, outcome tracking, automated evaluation, retrieval-quality evaluation, error classification, performance baselines).
* **Stage 3: Knowledge consolidation** (duplicate detection, pattern extraction, knowledge generalization, relationship discovery, contradiction detection, memory prioritization).
* **Stage 4: Adaptive intelligence** (retrieval adaptation, context allocation adaptation, compression and reconstruction improvements, procedural learning, behavioral policy generation, strategy optimization).
* **Stage 5: Autonomous learning supervision** (background consolidation, learning cartridge management, regression testing, confidence calibration, versioning and rollback, continuous performance monitoring).

## 16. Orchestration engine (`nmlearn::engine::ContinuousLearningEngine`)

The learning engine (`crates/nmlearn/src/engine.rs`) operates alongside the primary cognitive architecture. Each project has its own engine state, `continual/state.json` in the project's memory folder.

1. `observe(experience)`: captures, filters, and stores episodic records while deduplicating repeated observations and tracking correction recurrence. The experiences come from the training store: each settled example (a context pack the ledger judged) and each preference signal from the hooks is taken in once, by id (`nmlearn::idle::sync_engine`).
2. `record_retrieval_feedback(...)`: updates the retrieval adaptation from how many served places were used, passed over or missed. `record_delayed_outcome(...)` and `record_context_trace(...)` exist but are not called by the serving path yet.
3. `run_learning_cycle()`: snapshots current state, groups experiences by learning candidate, extracts patterns and procedures, consolidates into semantic, procedural and preference memory, proposes scoped behavioral policies, validates candidates through the supervisor rule, and updates the 6 continual learning cartridges.
4. `run_gated_cycle()`: Cycle B behind a measured held-out gate. The candidate is learned without the held-out experiences (the same split the training store uses), then the current state and the candidate are both scored on the held-out experiences (1 minus the mean squared error of their outcome predictions, where 0.75 is knowing nothing). The candidate replaces the current state only when it scores higher. With fewer than 5 held-out experiences there is nothing to measure on, and the supervisor rule decides alone.
5. `rollback(version)`: restores an earlier snapshot (the last 8 are kept).

What the engine changes while serving, with memory and the `learning` control on:

* The retrieval adaptation weights the ordering on the whole path (`serve --whole`).
* Validated behavioral policies whose names match the question are added to the context pack as short guidance lines (`policy [<name> · scope <scope>]: ...`).
* Each call appends compact telemetry (whether the pack met its floor, its tokens and time, the filter generation and mode), never the question or code text.

## 17. Three timescales of continuous learning

* **Milliseconds to seconds (Immediate adaptation)**: Capture feedback, update active task context, adjust temporary retrieval priorities, and record outcomes.
* **Minutes to hours (Experience consolidation)**: Identify patterns, consolidate knowledge, update indexes, and generate candidate behavioral improvements.
* **Hours to days (Long-term adaptation)**: Evaluate accumulated improvements, update learning policies, run regression checks, and reorganize long-term knowledge.

## 18. Measuring whether Nano is getting smarter

The continuous learning benchmark tracks ten metrics over time:

| Metric | What it tells us |
|---|---|
| Learning retention | Whether acquired knowledge persists |
| Correction recurrence | Whether previously corrected mistakes return |
| Retrieval precision | Whether relevant memories are selected |
| Retrieval recall | Whether important memories are missed |
| Context fidelity | Whether compression preserves essential information |
| Reconstruction accuracy | Whether stored context can be accurately recovered |
| Policy effectiveness | Whether learned strategies improve outcomes |
| Transfer performance | Whether knowledge helps with new related tasks |
| Regression rate | Whether learning damages existing capabilities |
| Learning efficiency | Improvement relative to computational cost |

When a previously corrected mistake recurs (`correction_recurrence`), Nano is designed to diagnose whether the failure occurred at retention, extraction, retrieval, application, or recognition (the diagnosis is not built in v0.1.0).

In v0.1.0 the engine reports only figures it has measured, each with the count it rests on; a figure with nothing under it is left out, never shown as a default. The measured figures are: outcomes that held, right the first time, experiences validated, retrieval precision (served places then used), retrieval recall (needed places that were served), correction recurrence, the held-out prediction score with its baseline, and the share of measured cycles the gate held back. Context fidelity, reconstruction accuracy, policy effectiveness, transfer performance and regression rate are not measured yet.

## 19. System-wide architecture summary

* **Primary cognitive architecture**: Sensory processing (drones), parallel tentacle context windows, Globalized Context Stack (G0, G1, G2, G3), Core Brain reasoning, memory retrieval, task execution.
* **Experience and feedback exchange**: Structured events between the cognitive path and the learning engine.
* **Continuous learning engine**: Experience acquisition, feedback processing, performance evaluation, knowledge extraction, pattern recognition, memory consolidation, retrieval adaptation, context adaptation, behavioral adaptation, policy generation.
* **Persistent learning memory**: Episodic memory, semantic memory, procedural memory, interaction memory, adaptive memory, evaluation history.
* **Learning supervisor**: Validation, confidence, versioning, regression prevention, rollback.

## 20. Core architectural principle

Continuous learning in Neuromorph Nano is a permanent, independent, model-independent cognitive subsystem, not merely an additional feature of the memory system. It retains its accumulated experiences, consolidated knowledge, procedures, retrieval adaptations, and validated behavioral policies across sessions, restarts, and changes to the underlying models.

## What runs in v0.1.0

### The idle learning cycle

One learning cycle (`crates/nmlearn/src/idle.rs`) runs its learners in a fixed order, each behind its own held-out gate, and records under one cycle id which learners ran, what each measured and whether its candidate was kept:

1. G3's prior replay (muscle memory; only when the `muscle` control is on).
2. The preference cartridge fit, from the preference signals the hooks record (edits kept or reverted, corrections, stated rules, refusals).
3. The continual learning engine: it takes in the training store's new examples, then runs its gated Cycle B (section 16).

When it runs is set by the `learning.cycle` control:

| Value | What happens |
|---|---|
| `auto` (default) | Each judged outcome is recorded as an experience as it comes (Cycle A). A cycle runs once 20 verified examples have settled in the training store since the last cycle and the serving process is idle (the environment variable `NANO_LEARN_THRESHOLD` changes the 20) |
| `immediate` | Experiences are recorded as they come; a cycle runs only when you ask for one |
| `manual` | A cycle runs only when you ask for one, from the Learning page |

The `learning` control (`on` by default) switches the whole of it on or off. Nothing is learned with memory off.

Before a learner runs, the files its state lives in are copied to the cycle's snapshot folder (`learning/rollback/<cycle id>/<learner>/` in the project's memory folder). Rolling a cycle back restores every learner that cycle touched and removes the files it created. The records are `learning/cycles.jsonl` (one row per cycle and per rollback) and `learning/last-cycle.json`; they hold numbers only, no question, passage or code text.

### Learned units and compartments

Everything learned that acts on a MicroModel is a learned unit. There are three kinds, each listed from its own store with its exact path, size and date:

| Kind | What it is | Where it is kept | Compartment |
|---|---|---|---|
| Project adapter | The Embedder's lens and the Reranker's code read-out (the judge read-out), fitted at idle from one project's own material and kept only if they beat a shuffled-label control on held-out files | `library/cartridges/adapter/<id>/` | the project it was fitted on |
| Learned piece | The pieces of `nm_unit` (the calibrator of when to ask for more, and learned settings) | `units/learned/` for the Nano's own folder, `memory/projects/<key>/learned/` for a project | `nano` or the project |
| Learning unit | A registered low-rank adapter beside one node of a MicroModel, fitted from checked outcomes; loaded by `serve --unit <id>` | `units/registry.json` | `shared` |

A unit loads only in the compartment it was fitted in. Each unit has its own plug, and each compartment has a switch for all of its units: switching a compartment off takes every unit in it out while each keeps its own plug, so switching it on again brings back exactly what was plugged in. A unit can also be plugged in and still not act (for example, an adapter fitted on an earlier base cartridge version); the list says why.

The plugs of adapters and pieces, and every compartment switch, are kept in `units/plugs.json`, which the serving process reads again when it changes. Each change is an audit row (`learned:<compartment>/<id>` or `compartment:<key>`; a learning unit uses `cartridge:<id>`), and Audit undoes it.

Where to control them:

* Interface: Cartridges, side panel tab Learned (lenses, learning units, and every learned unit by compartment). The Learning page shows them in brief.
* API: `GET /api/learned-units`; `POST /api/learned-units` with `{"unit": "<compartment>/<id>", "on": true|false}` or `{"compartment": "<key>", "on": true|false}`. Also `GET /api/units` and `GET /api/lenses`.
* MCP: `nm_learning` with `op` `units`, `unit_on`, `unit_off` (`unit`: `<compartment>/<id>`), `compartment_on` or `compartment_off` (`compartment`: `nano`, `shared` or a project key; the project by default).

### Stated preferences: a learning source beside usage and distillation

The user's stated rules are the third source of preference learning, beside the usage signals the
idle fitter learns from (`nmlearn::pref_fit`) and teacher distillation. All three write `preference`
cartridges through the same library path (layer 4 of the stacking order), and one cartridge may hold
stated rules and learned values together: stated rules live under their own settings keys
(`pref.stated.<id>.r<NNN>`), learned values under `pref.global` and `pref.key.*`, and neither reads
the other's keys. The fitter's own cartridge (`pref-<project>`) is written from learned values only,
so stated rules are kept in cartridges of their own.

| Step | What happens |
|---|---|
| Parse | The user's words are split into statements; each matches a known form (replace, vocabulary, ban, indentation, line length, naming) or is kept verbatim as guidance. A scope clause narrows it to a project, file kinds or folders. |
| Preview | Each rule is shown with what it does and how it is held (enforced, checked, guidance); nothing is written. |
| Write | On confirmation, a new version of the cartridge (`Library::add_cartridge`); the stacks that held the earlier version get a new version and stay switched on. Audited as a `library` row with undo. |
| Apply | In a serving process the plugged rules reach the Writer's own text and the `nm_context_pack` block within a second of the switch (the library is watched). `preference_apply` checks a draft. |

A stated rule is the user's word, not a measured change, so it serves when plugged unless it is
written on trial (recorded, never applied). Nothing is fitted from it, and no prompt or code text
enters the cartridge: only the user's statements and the rules made from them.

### The Learning page

The middle of the page is the loop working: the `learning` and `learning.cycle` switches with a project picker, the recorded calls flowing toward the cartridge slot, the engine's state (what came in, the gate, what is training or waiting, what was kept) and the learned units in brief. Below them, Preferences: the stated preference builder (a chat box, the parsed rules previewed, the cartridges with their rules and plug state, and a draft check). The side panel has the tabs Evidence (the ledger), Primes, Training data, Cartridges and Preferences (how rules are held and the forms the parser reads), and the About tab.

### Commands and tools

| Where | What |
|---|---|
| `nm_learning` (MCP) | `op`: `view` (default), `cycle`, `sync` (take stored examples into the engine), `rollback` (`version` or `cycle`; needs `confirm: true`), `examples`, `ledger`, `outcomes`, `export`, `import`, `references`, `reference_add`, `clear_examples`, `clear_ledger` (both need `confirm: true`), `units`, `unit_on`, `unit_off`, `compartment_on`, `compartment_off` |
| API | `GET /api/learning?project=<key>`, `POST /api/learning/cycle` (`action`: `cycle`, `sync` or `rollback`), `/api/examples`, `/api/ledger`, `/api/outcomes`, `/api/training/export`, `/api/training/import`, `/api/training/references` |
| Command line | `neuromorph-nano ledger`, `outcomes` and `examples --project <repo>` (see `docs/user-guide.md`) |
| Stated preferences | MCP `nm_learning` `op` `preference_say`, `preference_list`, `preference_edit`, `preference_remove`, `preference_plug`, `preference_unplug`, `preference_apply`; API `/api/preferences` and `/api/preferences/*`; command line `neuromorph-nano preferences` (`docs/user-guide.md`, Stated preferences) |
