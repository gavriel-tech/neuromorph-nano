//! One API contract for the Rust core, served over MCP (stdio). (The HTTP transport of the
//! original contract is not built in the mini.)
//! Gavriel Technologies.
//!
//! The contract table is the single source: the `OpenAPI` document, the MCP tool list and
//! both transports are generated from it, and every call is checked against it before its
//! handler runs.
pub mod adapter;
pub mod adapter_fit;
pub mod anatomy;
pub mod ask;
pub mod assemble;
// Roadmap Phase 4.5 (2026-11-24): the audit writer (moved from the program's controls), the
// MCP tools matching the web interface (`parity`) and the hard-drive storage view (`storage`).
pub mod audit;
pub mod brain_side;
pub mod cache_reads;
pub mod build;
pub mod coordinator;
pub mod core_read;
pub mod decide;
pub mod delta;
pub mod distil;
pub mod drone_path;
pub mod drone_dendrite;
pub mod drone_windows;
pub mod forage;
pub mod g3;
pub mod joint_read;
pub mod key_scan;
pub mod learn;
// Roadmap Phase 4.5 (2026-11-25): the user's hardware limits and this machine's recommended values.
pub mod limits;
pub mod lineage;
// Roadmap Phase 4.5 (2026-11-24): the live activity recorder (`GET /api/live`, `nm_live`).
pub mod live;
pub mod mcp;
// 2026-11-27: the memory areas (content, caches, weights), their use and budgets.
pub mod memory_areas;
// 2026-11-26 (Phase B): the memory and objective writes of `nm_memory` and `nm_goal`, shared with
// the interface's API (queued for a serving process that holds the memory journal).
pub mod memory_write;
pub mod neural;
pub mod parity;
pub mod plugs;
pub mod engine;
pub mod preload;
pub mod prefs;
pub mod reader;
pub mod request_read;
pub mod scan_dump;
pub mod session;
pub mod storage;
pub mod tentacle_body;
pub mod tentacle_core;
pub mod tentacle_pass;
pub mod tentacle_read;
pub mod tentacle_unit;
pub mod tools;
pub mod whole;
pub mod writer_idle;

use nmjson::Value;
use std::collections::BTreeMap;

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of the nmapi sub-components that run without an organism.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmapi.mcp", mcp::check),
        ("nmapi.forage", forage::check),
        ("nmapi.session.origins", session::check_meaning_origins),
        ("nmapi.session.family", session::check_family),
        ("nmapi.assemble", assemble::check),
        ("nmapi.delta", delta::check),
        ("nmapi.coordinator", coordinator::check),
        ("nmapi.build", build::check),
        ("nmapi.tentacle_pass", tentacle_pass::check),
        ("nmapi.tentacle_pass.layers", tentacle_pass::check_layers),
        ("nmapi.learn.names", learn::check_names),
        ("nmapi.tentacle_pass.drones", tentacle_pass::check_drones),
        ("nmapi.tentacle_pass.served", tentacle_pass::check_served),
        ("nmapi.tentacle_pass.kept_reads", tentacle_pass::check_kept_reads),
        ("nmapi.tentacle_pass.calibrated", tentacle_pass::check_calibrated),
        ("nmapi.tentacle_pass.keys", tentacle_pass::check_keys),
        ("nmapi.tentacle_core", tentacle_core::check),
        ("nmapi.core_read.look", core_read::check_look),
        ("nmapi.core_read.votes", core_read::check_votes),
        ("nmapi.core_read.pose", core_read::check_pose),
        ("nmapi.joint_read", joint_read::check),
        ("nmapi.decide", decide::check),
        ("nmapi.tentacle_body", tentacle_body::check),
        ("nmapi.tentacle_unit", tentacle_unit::check),
        ("nmapi.scan_dump", scan_dump::check),
        ("nmapi.reader", reader::check),
        ("nmapi.key_scan", key_scan::check),
        ("nmapi.brain_side", brain_side::check),
        ("nmapi.brain_side.intake", brain_side::check_intake),
        ("nmapi.drone_windows", drone_windows::check),
        ("nmapi.drone_dendrite", drone_dendrite::check),
        ("nmapi.drone_path", drone_path::check),
        ("nmapi.cache_reads", cache_reads::check),
        ("nmapi.whole", whole::check),
        ("nmapi.g3", g3::check),
        ("nmapi.ask", ask::check),
        ("nmapi.adapter", adapter::check),
        ("nmapi.adapter_fit", adapter_fit::check),
        ("nmapi.distil", distil::check),
        ("nmapi.writer_idle", writer_idle::check),
    ]
}

/// JSON type of a contract field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    /// A string.
    String,
    /// An array.
    Array,
    /// An object.
    Object,
    /// A boolean.
    Bool,
    /// A whole number.
    Integer,
}
impl FieldKind {
    /// JSON Schema type name.
    pub fn schema(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Array => "array",
            Self::Object => "object",
            Self::Bool => "boolean",
            Self::Integer => "integer",
        }
    }
    fn matches(self, v: &Value) -> bool {
        matches!(
            (self, v),
            (Self::String, Value::String(_))
                | (Self::Array, Value::Array(_))
                | (Self::Object, Value::Object(_))
                | (Self::Bool, Value::Bool(_))
                | (Self::Integer, Value::Number(_))
        )
    }
}
/// One input field.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    /// Name.
    pub name: &'static str,
    /// JSON type.
    pub kind: FieldKind,
    /// Must be present.
    pub required: bool,
    /// What it is.
    pub doc: &'static str,
    /// The value used when the field is not given, shown in the input schema.
    pub default: Option<usize>,
}
impl Field {
    /// The same field with the value used when it is not given.
    #[must_use]
    pub const fn defaulting(self, value: usize) -> Self {
        Self { default: Some(value), ..self }
    }
}
/// A typed failure, shared by both transports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    /// Stable code.
    pub code: &'static str,
    /// What failed.
    pub message: String,
}
impl ApiError {
    pub(crate) fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }
}
/// An operation handler.
pub type Handler = fn(&Value) -> Result<Value, ApiError>;
/// One operation of the contract.
#[derive(Clone, Copy, Debug)]
pub struct Operation {
    /// Name, for example `context.ask`.
    pub name: &'static str,
    /// One-line summary.
    pub summary: &'static str,
    /// Input fields.
    pub inputs: &'static [Field],
    /// The handler, run only after the input check.
    pub handler: Handler,
}
/// Watches every dispatched call (Neuromorph Nano, 2026-11-15, build plan E2: the audit of real
/// work): operation name, input, outcome and milliseconds, after the handler returns.
pub type Observer = fn(&str, &Value, &Result<Value, ApiError>, u128);
/// The contract table.
#[derive(Clone, Debug, Default)]
pub struct Api {
    ops: Vec<Operation>,
    site: Option<std::path::PathBuf>,
    observer: Option<Observer>,
}
fn text(s: &str) -> Value {
    Value::String(s.to_owned())
}
fn object(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
impl Api {
    /// The contract with one more operation (a library part serving its own capabilities).
    #[must_use]
    pub fn with(mut self, op: Operation) -> Self {
        self.ops.push(op);
        self
    }
    /// The contract with an observer that sees every dispatched call (the audit log).
    #[must_use]
    pub fn observed(mut self, f: Observer) -> Self {
        self.observer = Some(f);
        self
    }
    /// The static site folder, if any.
    #[must_use]
    pub fn site(&self) -> Option<&std::path::Path> {
        self.site.as_deref()
    }
    /// Every operation, in table order.
    pub fn operations(&self) -> &[Operation] {
        &self.ops
    }
    /// Check the input against the operation's fields, then run its handler.
    /// # Errors
    /// `api-404` unknown operation, `api-400` missing, unknown or wrongly typed fields,
    /// and handler errors.
    pub fn dispatch(&self, name: &str, input: &Value) -> Result<Value, ApiError> {
        let Some(watch) = self.observer else {
            return self.dispatch_inner(name, input);
        };
        let start = std::time::Instant::now();
        let out = self.dispatch_inner(name, input);
        watch(name, input, &out, start.elapsed().as_millis());
        out
    }
    fn dispatch_inner(&self, name: &str, input: &Value) -> Result<Value, ApiError> {
        let op = self.ops.iter().find(|o| o.name == name).ok_or_else(|| ApiError::new("api-404", format!("unknown operation {name}")))?;
        let Value::Object(fields) = input else {
            return Err(ApiError::new("api-400", "input must be an object"));
        };
        for f in op.inputs {
            match fields.get(f.name) {
                None if f.required => {
                    return Err(ApiError::new("api-400", format!("missing field {}", f.name)));
                }
                Some(v) if !f.kind.matches(v) => {
                    return Err(ApiError::new("api-400", format!("field {} must be {}", f.name, f.kind.schema())));
                }
                _ => {}
            }
        }
        if let Some(extra) = fields.keys().find(|k| !op.inputs.iter().any(|f| f.name == k.as_str())) {
            return Err(ApiError::new("api-400", format!("unknown field {extra}")));
        }
        // A parameter overlay changed since the last call reaches the held models here, between
        // calls (`neural::settle_overlays`).
        crate::neural::settle_installed();
        (op.handler)(input)
    }
    /// JSON Schema of an operation's input.
    pub fn input_schema(op: &Operation) -> Value {
        let props: BTreeMap<String, Value> = op
            .inputs
            .iter()
            .map(|f| {
                let mut p = vec![("type", text(f.kind.schema()))];
                match f.default {
                    Some(d) => {
                        p.push(("description", text(&format!("{}, default {d}", f.doc))));
                        p.push(("default", Value::Number(d.to_string())));
                    }
                    None => p.push(("description", text(f.doc))),
                }
                (f.name.to_owned(), object(p))
            })
            .collect();
        let required = op.inputs.iter().filter(|f| f.required).map(|f| text(f.name)).collect();
        object(vec![
            ("type", text("object")),
            ("properties", Value::Object(props)),
            ("required", Value::Array(required)),
            ("additionalProperties", Value::Bool(false)),
        ])
    }
    /// The `OpenAPI` 3.1 document: one POST path per operation.
    pub fn openapi(&self) -> Value {
        let paths: BTreeMap<String, Value> = self
            .ops
            .iter()
            .map(|op| {
                let body = object(vec![
                    ("required", Value::Bool(true)),
                    ("content", object(vec![("application/json", object(vec![("schema", Self::input_schema(op))]))])),
                ]);
                let ok = object(vec![("description", text("result"))]);
                let post = object(vec![
                    ("operationId", text(op.name)),
                    ("summary", text(op.summary)),
                    ("requestBody", body),
                    ("responses", object(vec![("200", ok), ("400", object(vec![("description", text("typed error"))]))])),
                ]);
                (format!("/v1/{}", op.name), object(vec![("post", post)]))
            })
            .collect();
        object(vec![
            ("openapi", text("3.1.0")),
            ("info", object(vec![("title", text("EMMA core API")), ("version", text("0.1.0"))])),
            ("paths", Value::Object(paths)),
        ])
    }
}
