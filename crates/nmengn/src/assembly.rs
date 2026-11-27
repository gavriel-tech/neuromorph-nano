//! Execution assemblies (PLAN 4c, 4d item 12): an assembly is data, a sub-graph of neural
//! components in the NTE graph format (`designs/assemblies/*.json`: nodes with kind, typed
//! ports and config; edges between ports). Each node names its operation and the parameter
//! clusters it binds (a role is which clusters a component binds; one weight system). There
//! is no Block type and no fixed layer count or order in code: `qwen3-reference.json`
//! reproduces the released layer order and is one assembly among others; the generators here
//! only write such files, the run reads the file. The port check runs before any tensor:
//! every node's operation is known, every required input has exactly one producer earlier in
//! the list, ports exist and are tensors, and tensor widths agree along every edge.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::{BTreeMap, BTreeSet};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmengn.assembly", FaultKind::Interface, m)
}

/// The operations a neural component can carry, with their input and output ports.
pub const OPS: &[(&str, &[&str], &[&str])] = &[
    ("source", &[], &["y"]),
    ("pass", &["x"], &["y"]),
    ("embed", &["ids"], &["y"]),
    ("norm", &["x"], &["y"]),
    ("headnorm", &["x"], &["y"]),
    ("linear", &["x"], &["y"]),
    ("rope", &["q", "k", "pos"], &["q", "k"]),
    ("score", &["q", "k", "v"], &["y"]),
    ("swigate", &["gate", "up"], &["y"]),
    ("add", &["a", "b"], &["y"]),
    ("head", &["x"], &["logits"]),
    // 2026-11-09: output components for the released embedding and reranker models.
    ("lastpool", &["x"], &["y"]),
    ("labelsoftmax", &["logits"], &["y"]),
];

/// One node of an assembly.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub id: String,
    pub op: String,
    pub kind: String,
    pub layer: Option<usize>,
    pub clusters: Vec<String>,
    /// Query and key-value head ranges for `score` (global head numbers).
    pub heads: Option<(usize, usize)>,
    pub kv: Option<(usize, usize)>,
    /// `rows` (stack outputs) or `cols` (stack inputs) for `linear` and `head`.
    pub stack: String,
    /// Weight precision of a `linear`: `q8` (int8 per output row, the recording's precision)
    /// or `f16`.
    pub precision: String,
    /// Label tokens of a `labelsoftmax` (each must be one token), in output order.
    pub labels: Vec<String>,
}
/// One edge: producer node and port to consumer node and port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    pub from: String,
    pub from_port: String,
    pub to: String,
    pub to_port: String,
}
/// An assembly: its name, nodes in execution order, edges, and the nodes it names for roles.
#[derive(Clone, Debug, PartialEq)]
pub struct Assembly {
    pub name: String,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// Role to node instance (2026-11-10, step P0; file key `names`): which node plays
    /// `final_norm`, `head`, `pool`, `label`, `learn_target` and so on. A role not named here
    /// is the node of today's default instance (`ROLE_DEFAULTS`).
    pub names: BTreeMap<String, String>,
}

/// Today's node instance per role, the stated default when an assembly names none
/// (2026-11-10, step P0). Any other role defaults to an instance of its own name.
pub const ROLE_DEFAULTS: &[(&str, &str)] =
    &[("final_norm", "final_norm"), ("head", "head"), ("pool", "pool"), ("label", "label"), ("learn_target", "L27.down")];

fn ports(op: &str) -> Option<(&'static [&'static str], &'static [&'static str])> {
    OPS.iter().find(|(o, _, _)| *o == op).map(|(_, i, o)| (*i, *o))
}
fn kind_of(op: &str) -> &'static str {
    match op {
        "norm" | "headnorm" => "norm",
        "linear" | "rope" | "score" => "attention",
        "swigate" => "feedforward",
        "head" | "lastpool" | "labelsoftmax" => "head",
        _ => "other",
    }
}
fn port_list(names: &[&str]) -> Value {
    Value::Array(
        names
            .iter()
            .map(|n| {
                let m: BTreeMap<String, Value> =
                    [("name", Value::String((*n).into())), ("kind", Value::String("tensor".into())), ("required", Value::Bool(true))]
                        .into_iter()
                        .map(|(k, v)| (k.to_owned(), v))
                        .collect();
                Value::Object(m)
            })
            .collect(),
    )
}

impl Assembly {
    /// Parse an assembly file (NTE graph format).
    /// # Errors
    /// Malformed JSON or a node without an operation.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let name = v.get("name").and_then(Value::string).unwrap_or("unnamed").to_owned();
        let mut nodes = Vec::new();
        for n in v.get("nodes").and_then(Value::array).map_err(|e| bad(e.to_string()))? {
            let id = n.get("id").and_then(Value::string).map_err(|e| bad(e.to_string()))?.to_owned();
            let cfg = n.get("config").map_err(|e| bad(format!("{id}: {e}")))?;
            let op = cfg.get("op").and_then(Value::string).map_err(|e| bad(format!("{id}: op: {e}")))?.to_owned();
            let int = |k: &str| cfg.get(k).and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok());
            let pair = |k: &str| -> Option<(usize, usize)> {
                let a = cfg.get(k).and_then(Value::array).ok()?;
                Some((usize::try_from(a.first()?.integer().ok()?).ok()?, usize::try_from(a.get(1)?.integer().ok()?).ok()?))
            };
            let clusters = cfg
                .get("clusters")
                .and_then(Value::array)
                .map(|a| a.iter().filter_map(|c| c.string().ok().map(str::to_owned)).collect())
                .unwrap_or_default();
            nodes.push(Node {
                kind: n.get("kind").and_then(Value::string).unwrap_or("other").to_owned(),
                layer: int("layer"),
                heads: pair("heads"),
                kv: pair("kv"),
                stack: cfg.get("stack").and_then(Value::string).unwrap_or("rows").to_owned(),
                precision: cfg.get("precision").and_then(Value::string).unwrap_or("f16").to_owned(),
                labels: cfg
                    .get("labels")
                    .and_then(Value::array)
                    .map(|a| a.iter().filter_map(|c| c.string().ok().map(str::to_owned)).collect())
                    .unwrap_or_default(),
                clusters,
                op,
                id,
            });
        }
        let mut edges = Vec::new();
        for e in v.get("edges").and_then(Value::array).map_err(|e| bad(e.to_string()))? {
            let s = |k: &str| e.get(k).and_then(Value::string).map(str::to_owned).map_err(|x| bad(x.to_string()));
            edges.push(Edge { from: s("from")?, from_port: s("from_port")?, to: s("to")?, to_port: s("to_port")? });
        }
        let mut names = BTreeMap::new();
        if let Ok(Value::Object(m)) = v.get("names") {
            for (role, inst) in m {
                names.insert(role.clone(), inst.string().map_err(|e| bad(format!("names.{role}: {e}")))?.to_owned());
            }
        }
        Ok(Self { name, nodes, edges, names })
    }
    /// The node id playing `role`: the instance named under `names`, else the stated default
    /// (`ROLE_DEFAULTS`, or the role's own name). An instance holding `#` is a whole node id.
    #[must_use]
    pub fn named(&self, role: &str) -> String {
        let inst =
            self.names.get(role).map_or_else(|| ROLE_DEFAULTS.iter().find(|r| r.0 == role).map_or(role, |r| r.1).to_owned(), Clone::clone);
        if inst.contains('#') { inst } else { nid(&inst) }
    }
    /// Load `designs/assemblies/<name>.json` from `dir`. A file with a `base` is an addition
    /// to another assembly (2026-11-10): the base is loaded, `linear_precision` (if given) sets
    /// the precision of its linear components, and the file's own nodes and edges are added.
    /// One shared structure serves every model of that layout; only what differs is written.
    /// # Errors
    /// A missing or malformed file, or a base chain deeper than four.
    pub fn load(dir: &std::path::Path, name: &str) -> Result<Self, Fault> {
        Self::load_depth(dir, name, 0)
    }
    fn load_depth(dir: &std::path::Path, name: &str, depth: usize) -> Result<Self, Fault> {
        if depth > 4 {
            return Err(bad(format!("{name}: base chain too deep")));
        }
        let path = dir.join(format!("{name}.json"));
        let v = nmjson::parse(&std::fs::read(&path).map_err(|e| bad(format!("{}: {e}", path.display())))?)
            .map_err(|e| bad(format!("{name}: {e}")))?;
        let Ok(base) = v.get("base").and_then(Value::string) else { return Self::from_value(&v) };
        let mut a = Self::load_depth(dir, base, depth + 1)?;
        if let Ok(p) = v.get("linear_precision").and_then(Value::string) {
            for n in a.nodes.iter_mut().filter(|n| n.op == "linear") {
                n.precision = p.to_owned();
            }
        }
        let own = Self::from_value(&v)?;
        a.name = own.name;
        a.nodes.extend(own.nodes);
        a.edges.extend(own.edges);
        a.names.extend(own.names);
        Ok(a)
    }
    /// An addition file over `base`: the nodes and edges of `self` that `base` lacks, and the
    /// linear precision. The inverse of `load` for one level.
    #[must_use]
    pub fn addition_value(&self, base: &Self, base_name: &str, linear_precision: &str, note: &str) -> Value {
        let own = Self {
            name: self.name.clone(),
            nodes: self.nodes.iter().filter(|n| base.node(&n.id).is_none()).cloned().collect(),
            edges: self.edges.iter().filter(|e| !base.edges.contains(e)).cloned().collect(),
            names: self.names.iter().filter(|(k, v)| base.names.get(*k) != Some(*v)).map(|(k, v)| (k.clone(), v.clone())).collect(),
        };
        let mut v = own.to_value(note);
        if let Value::Object(m) = &mut v {
            m.insert("base".into(), Value::String(base_name.into()));
            m.insert("linear_precision".into(), Value::String(linear_precision.into()));
        }
        v
    }
    /// The assembly as an NTE graph file.
    #[must_use]
    pub fn to_value(&self, note: &str) -> Value {
        let nodes = self
            .nodes
            .iter()
            .map(|n| {
                let (i, o) = ports(&n.op).unwrap_or((&[], &[]));
                let mut cfg: BTreeMap<String, Value> = BTreeMap::new();
                cfg.insert("op".into(), Value::String(n.op.clone()));
                if let Some(l) = n.layer {
                    cfg.insert("layer".into(), Value::Number(l.to_string()));
                }
                if !n.clusters.is_empty() {
                    cfg.insert("clusters".into(), Value::Array(n.clusters.iter().map(|c| Value::String(c.clone())).collect()));
                }
                for (k, r) in [("heads", n.heads), ("kv", n.kv)] {
                    if let Some((a, b)) = r {
                        cfg.insert(k.into(), Value::Array(vec![Value::Number(a.to_string()), Value::Number(b.to_string())]));
                    }
                }
                if n.op == "linear" || n.op == "head" {
                    cfg.insert("stack".into(), Value::String(n.stack.clone()));
                    cfg.insert("precision".into(), Value::String(n.precision.clone()));
                }
                if !n.labels.is_empty() {
                    cfg.insert("labels".into(), Value::Array(n.labels.iter().map(|c| Value::String(c.clone())).collect()));
                }
                let m: BTreeMap<String, Value> = [
                    ("id", Value::String(n.id.clone())),
                    ("kind", Value::String(n.kind.clone())),
                    ("ports_in", port_list(i)),
                    ("ports_out", port_list(o)),
                    ("config", Value::Object(cfg)),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect();
                Value::Object(m)
            })
            .collect();
        let edges = self
            .edges
            .iter()
            .map(|e| {
                let mode: BTreeMap<String, Value> = [("kind".to_owned(), Value::String("direct".into()))].into_iter().collect();
                let m: BTreeMap<String, Value> = [
                    ("from", Value::String(e.from.clone())),
                    ("from_port", Value::String(e.from_port.clone())),
                    ("to", Value::String(e.to.clone())),
                    ("to_port", Value::String(e.to_port.clone())),
                    ("mode", Value::Object(mode)),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect();
                Value::Object(m)
            })
            .collect();
        let mut m: BTreeMap<String, Value> = [
            ("name", Value::String(self.name.clone())),
            ("note", Value::String(note.into())),
            ("version", Value::Number("1".into())),
            ("nodes", Value::Array(nodes)),
            ("edges", Value::Array(edges)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
        // Written only when the assembly names a role, so existing files keep their text.
        if !self.names.is_empty() {
            m.insert("names".into(), Value::Object(self.names.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect()));
        }
        Value::Object(m)
    }
    #[must_use]
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }
    /// The producer of `(node, port)`.
    #[must_use]
    pub fn producer(&self, node: &str, port: &str) -> Option<(&str, &str)> {
        self.edges.iter().find(|e| e.to == node && e.to_port == port).map(|e| (e.from.as_str(), e.from_port.as_str()))
    }
    /// Structural port check (before widths): known operations, unique node ids, every edge
    /// between existing ports, every required input fed exactly once by a node listed earlier.
    /// # Errors
    /// The first broken rule.
    pub fn check_ports(&self) -> Result<(), Fault> {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut at: BTreeMap<&str, usize> = BTreeMap::new();
        for (i, n) in self.nodes.iter().enumerate() {
            if ports(&n.op).is_none() {
                return Err(bad(format!("{}: unknown operation {}", n.id, n.op)));
            }
            if !seen.insert(n.id.as_str()) {
                return Err(bad(format!("{}: duplicate node id", n.id)));
            }
            at.insert(n.id.as_str(), i);
        }
        for e in &self.edges {
            let (f, t) = (
                self.node(&e.from).ok_or_else(|| bad(format!("edge from unknown node {}", e.from)))?,
                self.node(&e.to).ok_or_else(|| bad(format!("edge to unknown node {}", e.to)))?,
            );
            let (_, fo) = ports(&f.op).unwrap_or((&[], &[]));
            let (ti, _) = ports(&t.op).unwrap_or((&[], &[]));
            if !fo.contains(&e.from_port.as_str()) || !ti.contains(&e.to_port.as_str()) {
                return Err(bad(format!("edge {}.{} -> {}.{}: no such port", e.from, e.from_port, e.to, e.to_port)));
            }
            if at.get(e.from.as_str()) >= at.get(e.to.as_str()) {
                return Err(bad(format!("edge {} -> {}: producer is not listed before its consumer", e.from, e.to)));
            }
        }
        for n in &self.nodes {
            let (ins, _) = ports(&n.op).unwrap_or((&[], &[]));
            for p in ins {
                let feeds = self.edges.iter().filter(|e| e.to == n.id && e.to_port == *p).count();
                if feeds != 1 {
                    return Err(bad(format!("{}: required input {p} has {feeds} producers (needs exactly one)", n.id)));
                }
            }
        }
        Ok(())
    }
    /// Channel-block mask on the feed-forward of layer `layer` (2026-11-11, cartridge build step
    /// C): only the listed blocks of `block` intermediate channels stay switched on. The gate and
    /// up nodes bind `rows` ranges of their tensors and the down node binds the matching `cols`
    /// ranges (stack `cols`); contiguous blocks are coalesced into one range, so keeping every
    /// block gives one range equal to the whole tensor. The released weights are read, never
    /// altered. Returns the ranges bound, as `(first channel, end channel)`.
    /// # Errors
    /// An empty block list, a zero block size, or a layer with no gate, up or down node.
    pub fn keep_ffn_blocks(&mut self, layer: usize, kept: &[usize], block: usize) -> Result<Vec<(usize, usize)>, Fault> {
        let mut b: Vec<usize> = kept.to_vec();
        b.sort_unstable();
        b.dedup();
        if b.is_empty() || block == 0 {
            return Err(bad(format!("L{layer}: a channel-block mask needs at least one block of nonzero size")));
        }
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for i in b {
            match ranges.last_mut() {
                Some(r) if r.1 == i * block => r.1 = (i + 1) * block,
                _ => ranges.push((i * block, (i + 1) * block)),
            }
        }
        for (part, axis) in [("gate", "rows"), ("up", "rows"), ("down", "cols")] {
            let suffix = format!("L{layer}.{part}");
            let mut hits = 0;
            for n in self.nodes.iter_mut().filter(|n| n.op == "linear" && n.id.ends_with(&suffix)) {
                let stem = n
                    .clusters
                    .first()
                    .and_then(|c| c.rsplit_once('/'))
                    .map(|(s, _)| s.to_owned())
                    .ok_or_else(|| bad(format!("{}: no cluster", n.id)))?;
                n.clusters = ranges.iter().map(|(a, e)| format!("{stem}/{axis}{a}-{e}")).collect();
                axis.clone_into(&mut n.stack);
                hits += 1;
            }
            if hits == 0 {
                return Err(bad(format!("no node {suffix}")));
            }
        }
        Ok(ranges)
    }
    /// The nodes needed to produce `targets` (backward reachability), in assembly order.
    #[must_use]
    pub fn needed(&self, targets: &[String], seeded: &BTreeSet<String>) -> Vec<usize> {
        let mut want: BTreeSet<String> = targets.iter().cloned().collect();
        let mut stack: Vec<String> = targets.to_vec();
        while let Some(id) = stack.pop() {
            if seeded.contains(&id) {
                continue;
            }
            for e in self.edges.iter().filter(|e| e.to == id) {
                if want.insert(e.from.clone()) {
                    stack.push(e.from.clone());
                }
            }
        }
        self.nodes.iter().enumerate().filter(|(_, n)| want.contains(&n.id) && !seeded.contains(&n.id)).map(|(i, _)| i).collect()
    }
}

/// Node id of a component in an assembly.
#[must_use]
pub fn nid(inst: &str) -> String {
    format!("nmengn.components#{inst}")
}

/// Population dimensions an assembly generator needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

struct Builder {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}
impl Builder {
    fn node(&mut self, inst: &str, op: &str, layer: Option<usize>, clusters: Vec<String>) -> String {
        let id = nid(inst);
        let precision = if op == "linear" { "q8" } else { "f16" };
        self.nodes.push(Node {
            id: id.clone(),
            op: op.into(),
            kind: kind_of(op).into(),
            layer,
            clusters,
            heads: None,
            kv: None,
            stack: "rows".into(),
            precision: precision.into(),
            labels: Vec::new(),
        });
        id
    }
    fn wire(&mut self, from: &str, fp: &str, to: &str, tp: &str) {
        self.edges.push(Edge { from: from.into(), from_port: fp.into(), to: to.into(), to_port: tp.into() });
    }
    /// A cluster reference: tensor and region, with no population. The same structure binds
    /// any Qwen3 population of this shape; the population is supplied when it binds.
    fn c(&self, tensor: &str, region: &str) -> String {
        format!("{tensor}/{region}")
    }
    fn last(&mut self) -> Option<&mut Node> {
        self.nodes.last_mut()
    }
    /// One attention path of layer `l` over query heads `h0..h1` (key-value heads follow).
    #[expect(clippy::too_many_arguments, reason = "layer, tag, head range, shape and two inputs name one path")]
    fn attention(&mut self, l: usize, tag: &str, h: (usize, usize), s: Shape, hin: &str, pos: &str) -> String {
        let d = s.head_dim;
        let g = (h.0 * s.kv_heads / s.heads, h.1 * s.kv_heads / s.heads);
        let p = format!("model.layers.{l}.self_attn");
        let rows = |r: (usize, usize)| (r.0..r.1).map(|i| format!("rows{}-{}", i * d, (i + 1) * d)).collect::<Vec<_>>();
        let q = self.node(
            &format!("L{l}.q{tag}"),
            "linear",
            Some(l),
            rows(h).iter().map(|r| self.c(&format!("{p}.q_proj.weight"), r)).collect(),
        );
        let k = self.node(
            &format!("L{l}.k{tag}"),
            "linear",
            Some(l),
            rows(g).iter().map(|r| self.c(&format!("{p}.k_proj.weight"), r)).collect(),
        );
        let v = self.node(
            &format!("L{l}.v{tag}"),
            "linear",
            Some(l),
            rows(g).iter().map(|r| self.c(&format!("{p}.v_proj.weight"), r)).collect(),
        );
        for x in [&q, &k, &v] {
            self.wire(hin, "y", x, "x");
        }
        let qn = self.node(&format!("L{l}.qnorm{tag}"), "headnorm", Some(l), vec![self.c(&format!("{p}.q_norm.weight"), "whole")]);
        let kn = self.node(&format!("L{l}.knorm{tag}"), "headnorm", Some(l), vec![self.c(&format!("{p}.k_norm.weight"), "whole")]);
        self.wire(&q, "y", &qn, "x");
        self.wire(&k, "y", &kn, "x");
        let rope = self.node(&format!("L{l}.rope{tag}"), "rope", Some(l), vec![]);
        self.wire(&qn, "y", &rope, "q");
        self.wire(&kn, "y", &rope, "k");
        self.wire(pos, "y", &rope, "pos");
        let sc = self.node(&format!("L{l}.score{tag}"), "score", Some(l), vec![]);
        if let Some(n) = self.last() {
            n.heads = Some(h);
            n.kv = Some(g);
        }
        self.wire(&rope, "q", &sc, "q");
        self.wire(&rope, "k", &sc, "k");
        self.wire(&v, "y", &sc, "v");
        let cols = (h.0..h.1).map(|i| self.c(&format!("{p}.o_proj.weight"), &format!("cols{}-{}", i * d, (i + 1) * d))).collect();
        let o = self.node(&format!("L{l}.o{tag}"), "linear", Some(l), cols);
        if let Some(n) = self.last() {
            n.stack = "cols".into();
        }
        self.wire(&sc, "y", &o, "x");
        o
    }
    /// The feed-forward of layer `l` after the residual `res`: norm, gate, up, swigate, down,
    /// residual. Returns the layer output node.
    fn ffn(&mut self, l: usize, res: &str) -> String {
        let p = format!("model.layers.{l}");
        let n =
            self.node(&format!("L{l}.ffn_norm"), "norm", Some(l), vec![self.c(&format!("{p}.post_attention_layernorm.weight"), "whole")]);
        self.wire(res, "y", &n, "x");
        let gate = self.node(&format!("L{l}.gate"), "linear", Some(l), vec![self.c(&format!("{p}.mlp.gate_proj.weight"), "whole")]);
        let up = self.node(&format!("L{l}.up"), "linear", Some(l), vec![self.c(&format!("{p}.mlp.up_proj.weight"), "whole")]);
        self.wire(&n, "y", &gate, "x");
        self.wire(&n, "y", &up, "x");
        let sw = self.node(&format!("L{l}.swigate"), "swigate", Some(l), vec![]);
        self.wire(&gate, "y", &sw, "gate");
        self.wire(&up, "y", &sw, "up");
        let down = self.node(&format!("L{l}.down"), "linear", Some(l), vec![self.c(&format!("{p}.mlp.down_proj.weight"), "whole")]);
        self.wire(&sw, "y", &down, "x");
        let r2 = self.node(&format!("L{l}.res2"), "add", Some(l), vec![]);
        self.wire(res, "y", &r2, "a");
        self.wire(&down, "y", &r2, "b");
        r2
    }
}

/// Generate an assembly: `paths` attention paths per layer (1 = the released order; 2 = query
/// heads split in halves, two attention paths whose outputs add before one shared SwiGLU).
/// One chain of single operations wired by ports, with no per-layer wrapper (2026-11-10): the
/// previous layer's residual feeds the next attention norm and residual directly. It names no
/// population, so every Qwen3 population with the same layer and head layout binds it.
#[must_use]
pub fn generate(name: &str, s: Shape, paths: usize) -> Assembly {
    let mut b = Builder { nodes: Vec::new(), edges: Vec::new() };
    let ids = b.node("ids", "source", None, vec![]);
    let pos = b.node("pos", "source", None, vec![]);
    let emb = b.node("embed", "embed", None, vec![b.c("model.embed_tokens.weight", "whole")]);
    b.wire(&ids, "y", &emb, "ids");
    let mut x = emb;
    for l in 0..s.layers {
        let an =
            b.node(&format!("L{l}.attn_norm"), "norm", Some(l), vec![b.c(&format!("model.layers.{l}.input_layernorm.weight"), "whole")]);
        b.wire(&x, "y", &an, "x");
        let per = s.heads / paths.max(1);
        let outs: Vec<String> = (0..paths.max(1))
            .map(|p| {
                let tag = if paths > 1 { ["a", "b", "c", "d"].get(p).copied().unwrap_or("x") } else { "" };
                b.attention(l, tag, (p * per, (p + 1) * per), s, &an, &pos)
            })
            .collect();
        let mut attn = outs.first().cloned().unwrap_or_default();
        for (i, o) in outs.iter().enumerate().skip(1) {
            let sum = b.node(&format!("L{l}.attn_sum{i}"), "add", Some(l), vec![]);
            b.wire(&attn, "y", &sum, "a");
            b.wire(o, "y", &sum, "b");
            attn = sum;
        }
        let r1 = b.node(&format!("L{l}.res1"), "add", Some(l), vec![]);
        b.wire(&x, "y", &r1, "a");
        b.wire(&attn, "y", &r1, "b");
        x = b.ffn(l, &r1);
    }
    let fin = b.node("final_norm", "norm", None, vec![b.c("model.norm.weight", "whole")]);
    b.wire(&x, "y", &fin, "x");
    let head = b.node("head", "head", None, vec![b.c("model.embed_tokens.weight", "whole")]);
    b.wire(&fin, "y", &head, "x");
    Assembly { name: name.to_owned(), nodes: b.nodes, edges: b.edges, names: BTreeMap::new() }
}

/// A role assembly over a released Qwen3 population (2026-11-09): the reference layer order
/// with linear weights at `precision`, then one output component. `embed`: `lastpool` on the
/// final norm (the last token's state, L2-normalised; the embedding model card's pooling).
/// `rerank`: `labelsoftmax` over the head's "yes" and "no" logits (the reranker model card's
/// P(yes)). The scoring is part of the assembly, not code around a model call.
/// # Errors
/// An unknown role.
pub fn generate_role(name: &str, s: Shape, role: &str, precision: &str) -> Result<Assembly, Fault> {
    let mut a = generate(name, s, 1);
    for n in a.nodes.iter_mut().filter(|n| n.op == "linear") {
        n.precision = precision.to_owned();
    }
    let (inst, op, from, port, labels) = match role {
        "embed" => ("pool", "lastpool", nid("final_norm"), "y", Vec::new()),
        "rerank" => ("label", "labelsoftmax", nid("head"), "logits", vec!["yes".to_owned(), "no".to_owned()]),
        other => return Err(bad(format!("unknown role {other}"))),
    };
    let id = nid(inst);
    a.nodes.push(Node {
        id: id.clone(),
        op: op.into(),
        kind: kind_of(op).into(),
        layer: None,
        clusters: Vec::new(),
        heads: None,
        kv: None,
        stack: "rows".into(),
        precision: "f16".into(),
        labels,
    });
    a.edges.push(Edge { from, from_port: port.into(), to: id, to_port: if op == "labelsoftmax" { "logits".into() } else { "x".into() } });
    Ok(a)
}

/// Self-check of `assembly`: a two-layer reference and split assembly pass the port check and
/// round-trip through JSON; removing `swigate` fails the port check (the down projection loses
/// its only input) before anything runs.
pub(crate) fn check() -> Result<(), String> {
    let s = Shape { layers: 2, heads: 4, kv_heads: 2, head_dim: 2 };
    for paths in [1, 2] {
        let a = generate("t", s, paths);
        a.check_ports().map_err(|e| e.to_string())?;
        let back = Assembly::from_value(&a.to_value("t")).map_err(|e| e.to_string())?;
        if back != a {
            return Err("nmengn.assembly: JSON round trip".into());
        }
    }
    // Channel-block mask: blocks 0, 1 and 3 of 2 channels coalesce into rows0-4 and rows6-8.
    let mut m = generate("t", s, 1);
    let r = m.keep_ffn_blocks(1, &[3, 0, 1], 2).map_err(|e| e.to_string())?;
    let down = m.node(&nid("L1.down")).ok_or("nmengn.assembly: no L1.down")?;
    if r != vec![(0, 4), (6, 8)] || down.stack != "cols" || !down.clusters.iter().any(|c| c.ends_with("/cols6-8")) {
        return Err(format!("nmengn.assembly: channel-block mask gave {r:?}, down {:?}", down.clusters));
    }
    m.check_ports().map_err(|e| e.to_string())?;
    let mut cut = generate("t", s, 1);
    cut.nodes.retain(|n| !n.id.ends_with("L0.swigate"));
    cut.edges.retain(|e| !e.from.ends_with("L0.swigate") && !e.to.ends_with("L0.swigate"));
    if cut.check_ports().is_ok() {
        return Err("nmengn.assembly: an assembly without swigate passed the port check".into());
    }
    Ok(())
}
