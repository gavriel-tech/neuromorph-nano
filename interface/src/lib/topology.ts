/** The topology of the Neuromorph Nano, built from real data: the anatomy graph
 *  (`designs/anatomy.json`, served inside `/api/settings`), the crate cards (`/api/parts`), the
 *  models (`/api/status`) and the learned cartridges (`/api/cartridges`).
 *
 *  The anatomy and the cards do not group the system the way the organism is described, so this
 *  file holds the one mapping table: which anatomy nodes and which crate sub-components belong to
 *  which main part (Core Brain, GCX, Neuropod, substrate, memory, learning, engine, runtime), and
 *  the plain names and descriptions shown for them. Every sub-component a card lists is placed
 *  somewhere: anything the table does not name falls into its crate's home part. */
import type { Cartridge, Model, Part, Port, Status } from "./api";

export type AnatomyNode = {
  id: string;
  kind: string;
  parent: string | null;
  ports_in: Port[];
  ports_out: Port[];
  config: Record<string, unknown>;
};
export type AnatomyEdge = {
  from: string;
  to: string;
  from_port: string;
  to_port: string;
  mode: { kind: string; bound?: number; predicate?: string; max_iter?: number };
};
export type Anatomy = { nodes: AnatomyNode[]; edges: AnatomyEdge[]; note?: string; version?: number };

export type Role = "organism" | "brain" | "gcx" | "pod" | "tentacle" | "substrate" | "memory" | "learning" | "engine" | "runtime" | "model" | "cartridge" | "group" | "part";
export type Tone = "violet" | "cyan" | "neutral";
export type Layout = "organism" | "fan" | "stack" | "auto";

export type TNode = {
  key: string;
  path: string[];
  label: string;
  /** Compact label for narrow screens. */
  short?: string;
  /** Second line: a short role, e.g. "spinal cord and brain stem". */
  sub?: string;
  about: string;
  role: Role;
  tone: Tone;
  layout: Layout;
  /** Engine ids this node stands for: anatomy ids, card sub-component ids, crate ids. */
  ids: string[];
  /** Every anatomy id in this node's subtree (for edges between siblings). */
  anatomyIds: Set<string>;
  crate?: string;
  card?: Part;
  anatomy?: AnatomyNode;
  sub_hash?: string;
  sub_version?: number;
  model?: Model;
  cartridge?: Cartridge;
  /** Design files in `designs/` that set this part. */
  settings: string[];
  /** Control keys (`/api/controls`) that apply to this part. */
  controls: string[];
  /** The crate is present in this build (`/api/status` parts). */
  present: boolean;
  children: TNode[];
  parent?: TNode;
};

export type Tree = {
  root: TNode;
  byPath: Map<string, TNode>;
  /** Engine id to the nodes that hold it. */
  index: Map<string, TNode[]>;
  /** Crate id to the node that is its home. */
  crateHome: Map<string, TNode>;
  edges: AnatomyEdge[];
  tentacles: TNode[];
};

/* ---------------------------------------------------------------------------------------------
 * Mapping table
 * ------------------------------------------------------------------------------------------- */

/** The main parts at level 0, the crates each one is home to, and what it does. */
const MAIN: { key: string; label: string; short: string; sub: string; role: Role; tone: Tone; layout: Layout; crates: string[]; about: string; settings: string[]; controls: string[] }[] = [
  {
    key: "core-brain",
    short: "Core Brain",
    label: "Core Brain",
    sub: "MicroModels in chains",
    role: "brain",
    tone: "violet",
    layout: "auto",
    crates: ["nmbrain"],
    about: "Where the tentacles converge. One brain whose regions are MicroModels, each with its own job, working as one: one orchestration, one arbitration, one output.",
    settings: ["chains", "homeostasis", "reflexes"],
    controls: ["mode"],
  },
  {
    key: "gcx",
    short: "GCX",
    label: "GCX",
    sub: "spinal cord and brain stem",
    role: "gcx",
    tone: "cyan",
    layout: "stack",
    crates: ["nmmusc"],
    about: "The Globalized Context Stack. It carries what the tentacles read up to the Core Brain and carries control back down. Every level is held in RAM.",
    settings: [],
    controls: [],
  },
  {
    key: "neuropod",
    short: "Neuropod",
    label: "Neuropod",
    sub: "eight tentacles",
    role: "pod",
    tone: "cyan",
    layout: "fan",
    crates: ["nmpod"],
    about: "The sensory side. Each tentacle is a small brain that reads its own window in parallel; drones at its tip are the first contact with the raw material.",
    settings: ["drones", "tentacle-core"],
    controls: ["precision.base"],
  },
  {
    key: "substrate",
    short: "Substrate",
    label: "Context Substrate",
    sub: "the material, in RAM",
    role: "substrate",
    tone: "neutral",
    layout: "auto",
    crates: ["nmsubs", "nmdoc"],
    about: "The material the organism reads, as it holds it: documents, code, notes and data, with the links between pieces and views over them. Held in RAM, redone when it changes.",
    settings: [],
    controls: [],
  },
  {
    key: "memory",
    short: "Memory",
    label: "Memory",
    sub: "kept across sessions",
    role: "memory",
    tone: "neutral",
    layout: "auto",
    crates: ["nmmem"],
    about: "Records kept per project across sessions: core, active and archive tiers, verified episodes, temporary notes and one structural summary per version of the material. Separate from learning.",
    settings: ["memory"],
    controls: ["memory"],
  },
  {
    key: "learning",
    short: "Learning",
    label: "Continuous learning",
    sub: "neuroplasticity cartridges",
    role: "learning",
    tone: "violet",
    layout: "auto",
    crates: ["nmlearn", "nmnpcs"],
    about: "Learns from verified work and from your preferences into removable neuroplasticity cartridges that help comprehension for the job at hand. Separate from memory; the base cartridges and upstream open weights are never altered.",
    settings: ["cartridges", "train-rerank"],
    controls: ["learning"],
  },
  {
    key: "engine",
    short: "Engine",
    label: "Engine",
    sub: "MicroModels",
    role: "engine",
    tone: "neutral",
    layout: "auto",
    crates: ["nmengn", "nmtokz", "nmsafe", "nmffi"],
    about: "The neural engine that runs the MicroModels from their Neuromorph base cartridges on the graphics card and the processor.",
    settings: ["models", "placement", "precision-judge-mix"],
    controls: ["precision.base", "precision.writer", "precision.reranker", "precision.embedder"],
  },
  {
    key: "runtime",
    short: "MCP",
    label: "MCP and runtime",
    sub: "agents connect here",
    role: "runtime",
    tone: "neutral",
    layout: "auto",
    crates: ["nmapi", "nmnte", "nmtool", "nmtype", "nmjson", "nmhash", "nmlex"],
    about: "The MCP server and its tools that your coding agents call (Claude Code, Cursor, VS Code, Antigravity, Windsurf and any MCP client), the executor that runs the anatomy graph, build help, and the shared foundations.",
    settings: ["compositions", "conformance"],
    controls: [],
  },
];

/** Flows between main parts that the anatomy graph does not draw (it covers the question path
 *  only). Each is a real call path described in docs/architecture.md. */
export const FLOWS: [string, string][] = [
  ["runtime", "substrate"],
  ["substrate", "neuropod"],
  ["core-brain", "runtime"],
  ["engine", "core-brain"],
  ["learning", "engine"],
];

/** GCX levels, top to bottom, and the sub-components each holds. G3 is home to the muscle
 *  memory crate (nmmusc); g0 holds every tentacle's g0_tx anatomy node. */
const GCX_LEVELS: { key: string; label: string; sub: string; about: string; subs: string[]; anatomy?: string[]; crates?: string[] }[] = [
  {
    key: "g3",
    label: "G3",
    sub: "pattern recognition",
    about: "Recognises repeated behaviour and its effect, and applies reloaded muscle memory. Behaviour only, never what was read.",
    subs: [],
    anatomy: ["nmmusc#g3"],
    crates: ["nmmusc"],
  },
  {
    key: "g2",
    label: "G2",
    sub: "whole-context comprehension",
    about: "Decides what goes up to the Core Brain and sends instructions down, such as directing a tentacle's focus.",
    subs: ["nmbrain.g2", "nmbrain.reflex", "nmbrain.gmap", "nmapi.coordinator", "nmapi.decide"],
    anatomy: ["nmbrain.g2#g2"],
  },
  {
    key: "g1",
    label: "G1",
    sub: "coordinating communicator",
    about: "Receives every tentacle's output and is the only path between tentacles.",
    subs: ["nmbrain.g1"],
    anatomy: ["nmbrain.g1#g1"],
  },
  {
    key: "g0",
    label: "g0",
    sub: "one per tentacle",
    about: "Holds its tentacle's encoded read and reports what it found on a scale shared by all tentacles.",
    subs: ["nmbrain.g0"],
  },
];

/** Groups inside parts that hold many sub-components. Unlisted ones fall into "Other". */
const GROUPS: Record<string, { key: string; label: string; about: string; subs: string[] }[]> = {
  memory: [
    { key: "stores", label: "Stores", about: "Where records live: working, episodic and structural memory, identity, the tiers and temporary notes.", subs: ["nmmem.working", "nmmem.episodic", "nmmem.structural", "nmmem.identity", "nmmem.tiers", "nmmem.temp", "nmmem.hierarchy"] },
    { key: "write-recall", label: "Write and recall", about: "What may be kept and how it comes back: the write gate, data classes, recall, consolidation and retention.", subs: ["nmmem.gate", "nmmem.classes", "nmmem.recall2", "nmmem.pass", "nmmem.consolidation", "nmmem.retention"] },
    { key: "records", label: "Records and upkeep", about: "The change journal, records and their keeping, run days and migration between layouts.", subs: ["nmmem.journal", "nmmem.record", "nmmem.book", "nmmem.runday", "nmmem.migrate", "nmmem.mem2"] },
  ],
  learning: [
    { key: "units", label: "Learning units", about: "Small learned judgments with their isolation, outcomes, read-outs and calibration.", subs: ["nmlearn.unit", "nmlearn.registry", "nmlearn.isolation", "nmlearn.outcomes", "nmlearn.stepspec", "nmlearn.forward", "nmlearn.rls", "nmlearn.readout", "nmlearn.adapt", "nmlearn.calibrate", "nmlearn.tuner", "nmlearn.learned"] },
    { key: "preferences", label: "Preferences and routing", about: "Preference learning, the cartridge label and stacking, routing between learned pieces, and the ledger of what was learned and why.", subs: ["nmlearn.preference", "nmlearn.cartridge", "nmlearn.route", "nmlearn.ledger"] },
  ],
  "core-brain": [
    { key: "chains", label: "Chains and evidence", about: "How MicroModels are chained for a request, the instructions they receive, and the evidence each region weighs.", subs: ["nmbrain.chain", "nmbrain.instruct", "nmbrain.evidence", "nmbrain.stack", "nmbrain.slot"] },
  ],
  runtime: [],
};

/** Crates shown as their own component inside a main part, with a plain name. */
const CRATE_COMPONENT: Record<string, { label: string; about: string }> = {
  nmapi: { label: "MCP server and tools", about: "The served contract: the MCP server and its tools, assembly of the cited context pack, and the paths that answer." },
  nmnte: { label: "Executor", about: "Runs the anatomy graph: snapshots, frames, the scheduler, the registry and lifecycle of components." },
  nmtool: { label: "Build help", about: "Runs one cargo command and returns compact, located errors; Step Check, improvement findings and verification." },
  nmtype: { label: "Shared contracts", about: "The Context Packet, components, cards, ports, folder and project paths shared by every component." },
  nmjson: { label: "JSON", about: "Strict JSON reader and canonical encoder." },
  nmhash: { label: "Hashing", about: "SHA-256 for seals, cards and weight checks." },
  nmlex: { label: "Lexer", about: "Token and term helpers." },
  nmengn: { label: "Neural engine", about: "Runs the MicroModels (Embedder, Reranker, Reader, Writer) from their Neuromorph base cartridges on the processor and graphics card: placement, decode, adapters, KV store and residency." },
  nmtokz: { label: "Tokenizer", about: "The MicroModels' byte-level BPE tokenizer (shared with the optional upstream open weights)." },
  nmsafe: { label: "Weight reader", about: "Reads safetensors weight files and their quantization; never writes them." },
  nmffi: { label: "Graphics card bridge", about: "The NVIDIA driver bridge; the only crate with unsafe code." },
  nmlearn: { label: "Learning", about: "Further components of continuous learning." },
  nmnpcs: { label: "Parameter map", about: "Model cards, parameter clusters and the map that places learned additions; never cuts the upstream open weights." },
  nmmusc: { label: "Muscle memory", about: "The behaviour log and muscle memory packs, saved to disk as their own store." },
  nmdoc: { label: "Document readers", about: "Reads each kind of material into text the substrate can hold: plain text, Markdown, CSV, HTML, XML, JSON, YAML and logs." },
};

/** Sub-components that serve the tentacles but live in other crates. */
const NEUROPOD_EXTRA = ["nmapi.tentacle_pass", "nmapi.tentacle_core", "nmapi.drone_windows", "nmapi.drone_path", "nmapi.core_read", "nmapi.joint_read"];

/** Plain names and descriptions for sub-components and anatomy nodes. */
const PLAIN: Record<string, [string, string]> = {
  // Core Brain
  "nmbrain.objectives": ["Objectives", "Holds the standing objectives that shape every plan."],
  "nmbrain.rivers": ["Rivers", "Where every tentacle's packets converge before arbitration."],
  "nmbrain.arbiter": ["Arbiter", "Weighs competing packets and decides what is kept."],
  "nmbrain.contradiction": ["Contradiction check", "Finds packets that disagree and marks the conflict."],
  "nmbrain.homeostasis": ["Homeostasis", "Keeps the organism within its limits of time, memory and load."],
  "nmbrain.development": ["Development", "Adjusts the organism over time from what homeostasis reports."],
  "nmbrain.synthesis": ["Synthesis", "Combines the kept packets, with the reranker's scores, into one result."],
  "nmbrain.stack#plan": ["Plan", "Turns the task and objectives into a plan sent to every tentacle."],
  "nmbrain.stack#emit": ["Emit", "Sends the finished pack out and to the memory write gate."],
  "nmbrain.slot#router": ["Router", "MicroModel slot that reads the task and routes it."],
  "nmbrain.slot#writer": ["Writer", "MicroModel slot that writes, when writing is called for."],
  "nmbrain.slot#embedder": ["Embedder", "MicroModel slot that places text by meaning."],
  "nmbrain.slot#reranker": ["Reranker", "MicroModel slot that scores candidates against the question."],
  "nmbrain.chain": ["Chains", "Chains of MicroModels per kind of request."],
  "nmbrain.instruct": ["Instructions", "The instructions each MicroModel receives."],
  "nmbrain.evidence": ["Evidence", "The evidence each region weighs."],
  "nmbrain.stack": ["Stack", "The brain's processing stack."],
  "nmbrain.slot": ["Slots", "The MicroModel slots and their binding to the engine."],
  "nmbrain.g0": ["g0 transmitter", "A tentacle's report on the shared scale."],
  "nmbrain.g1": ["G1 coordinator", "The only path between tentacles."],
  "nmbrain.g2": ["G2 comprehension", "Decides what goes up and what is directed down."],
  "nmbrain.gmap": ["Context map", "The map of what has been read and where to look next."],
  "nmbrain.reflex": ["Reflexes", "Controlled reflexes sent down to the tentacles."],
  "nmapi.coordinator": ["Map coordinator", "Sets budgets and refinement from the context map."],
  "nmapi.decide": ["Decide", "The decision of what the pack holds."],
  // Neuropod
  "nmpod.unit": ["Tentacle unit", "One tentacle and its drone tip as a unit."],
  "nmpod.tip": ["Drone tip", "The readers at a tentacle's tip."],
  "nmpod.pipeline": ["Pipeline", "A tentacle's read, from window to findings."],
  "nmpod.record": ["Record", "What a tentacle read and found."],
  "nmpod.recursion": ["Recursion", "A tentacle reaching deeper into a promising span."],
  "nmpod.territory": ["Territory", "How the material is divided into windows."],
  "nmpod.attention": ["Attention", "Attention-head readers used as drones."],
  "nmpod.pod": ["Pod", "The Neuropod that holds the tentacles."],
  "nmapi.tentacle_pass": ["Tentacle pass", "Encodes each tentacle's window once and keeps it in RAM."],
  "nmapi.tentacle_core": ["Tentacle core", "Integrates what a tentacle's drones found."],
  "nmapi.drone_windows": ["Drone windows", "The windows the drones read."],
  "nmapi.drone_path": ["Drone path", "The whole path from drones to the brain."],
  "nmapi.core_read": ["Core read", "The tentacle core's re-read."],
  "nmapi.joint_read": ["Joint read", "A read across tentacles, through G1."],
  // Memory
  "nmmem.working": ["Working memory", "What is in use for the current question."],
  "nmmem.episodic": ["Episodes", "Verified outcomes; recall re-reads the cited lines fresh."],
  "nmmem.structural": ["Structural summary", "One summary per version of the material."],
  "nmmem.identity": ["Identity", "Who the organism is and its objectives."],
  "nmmem.gate": ["Write gate", "Decides what may be written; refuses secrets."],
  "nmmem.gate#m0": ["Write gate", "Decides what may be written; refuses secrets."],
  "nmmem.tiers": ["Tiers", "Core, active and archive."],
  "nmmem.temp": ["Temporary notes", "Three uses save one; 24 hours unused clears it."],
  "nmmem.hierarchy": ["Hierarchy", "Topics by subject, most used at the top."],
  "nmmem.consolidation": ["Consolidation", "Moves what holds up into lasting memory."],
  "nmmem.retention": ["Retention", "What fades and moves to the archive."],
  "nmmem.classes": ["Data classes", "What kind of data a record is, and what is never saved."],
  "nmmem.recall2": ["Recall", "Matches a question to past episodes."],
  "nmmem.journal": ["Journal", "The change journal and its undo."],
  // Substrate
  "nmsubs.views#g0": ["Views", "Views over the substrate that answer a query with packets."],
  "nmsubs.views": ["Views", "Views over the substrate that answer a query with packets."],
  "nmsubs.repo": ["Material", "Files and folders as read: any kind of document, data or code."],
  "nmsubs.refgraph": ["Reference graph", "Real links between pieces: definitions, references and citations."],
  "nmsubs.dense": ["Dense lane", "Meaning-based search over the material."],
  "nmsubs.lens": ["Lenses", "The lens each tentacle takes for a question."],
  "nmsubs.zoom": ["Zoom", "Views from one passage up to the whole collection."],
  "nmsubs.sync": ["Sync", "Redoes reads when the material changes."],
  // Learning
  "nmlearn.cartridge": ["Cartridge label", "The label, stacking and checks of a cartridge."],
  "nmlearn.preference": ["Preferences", "Your preferences, per project or everywhere."],
  "nmlearn.ledger": ["Ledger", "What was learned, from what, and why."],
  "nmlearn.outcomes": ["Outcomes", "Right, wrong and verified outcomes that learning weighs."],
  // Engine
  "nmengn.gpu": ["Graphics card", "Kernels on the graphics card."],
  "nmengn.cpu": ["Processor", "The processor path."],
  "nmengn.decode": ["Decode", "Token-by-token decoding."],
  "nmengn.adapter": ["Adapters", "Where learned cartridges attach to a model."],
  "nmengn.kvstore": ["KV store", "Kept encoded chunks for the tentacle windows."],
  "nmengn.residency": ["Residency", "Which weights stay on the card."],
  "nmengn.placement": ["Placement", "Where each component runs."],
  // Runtime
  "nmapi.mcp": ["MCP server", "The stdio server your coding agents connect to."],
  "nmapi.tools": ["Tools", "The 19 nm_* tools."],
  "nmapi.assemble": ["Assemble", "Builds the cited context pack within the token budget."],
  "nmapi.telemetry": ["Telemetry", "What each component did, for the audit log."],
  "nmtool.verify": ["Verify", "Checks a proposed change and counts its findings."],
  "nmtool.stepcheck": ["Step Check", "Checks a proposed diff on a scratch copy before it is applied."],
};

const READER_NAMES: [RegExp, string, string][] = [
  [/^drones/, "Attention-head drones", "Attention heads of the base model read the window; central weights used by reference."],
  [/^lexical/, "Window readers", "Lexical, semantic and deep readers inside the tentacle's window."],
  [/^symbols/, "Whole-base readers", "Symbols, history, build, tests, uses, rules and interfaces across the whole material."],
];

/** The control key of a model folder's precision. */
export function precisionKey(modelId: string): string {
  if (modelId.includes("rerank")) return "precision.reranker";
  if (modelId.includes("embed")) return "precision.embedder";
  if (modelId.includes("1.7b")) return "precision.writer";
  return "precision.base";
}

/** Precision role of an anatomy MicroModel slot. */
const SLOT_PRECISION: Record<string, string> = {
  router: "precision.base",
  writer: "precision.writer",
  embedder: "precision.embedder",
  reranker: "precision.reranker",
};

/** Abbreviations in an engine id, as they are written in plain text. */
const ABBREV: Record<string, string> = { md: "Markdown", csv: "CSV", html: "HTML", xml: "XML", json: "JSON", yaml: "YAML", rls: "RLS", gpu: "GPU", cpu: "CPU", kv: "KV", mcp: "MCP", api: "API", ffi: "FFI", bpe: "BPE" };

function human(id: string): string {
  const last = id.split("#")[0].split(".").pop() ?? id;
  const words = last.replace(/[_-]+/g, " ").replace(/\d+$/, (d) => ` ${d}`).split(" ").map((w) => ABBREV[w.toLowerCase()] ?? w);
  const s = words.join(" ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function slug(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

/* ---------------------------------------------------------------------------------------------
 * Builder
 * ------------------------------------------------------------------------------------------- */

/** `roleOf` names the roles a model serves ("reader + writer"), from GET /api/models/roles; each
 *  MicroModel then carries its kind (Reader, Writer, Embedder, Reranker) as its second line. */
type Input = { anatomy: Anatomy | null; parts: Record<string, Part> | null; status: Status | null; cartridges: Cartridge[] | null; roleOf?: (model: string) => string };

/** A MicroModel's kind from its roles ("reader + writer" gives "Reader + Writer"). */
export function kindTitle(roles: string): string {
  return roles
    .split(/\s*\+\s*/)
    .filter(Boolean)
    .map((r) => r.charAt(0).toUpperCase() + r.slice(1))
    .join(" + ");
}

export function buildTree({ anatomy, parts, status, cartridges, roleOf }: Input): Tree {
  const nodes = anatomy?.nodes ?? [];
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const cards = parts ?? {};
  const present = new Set(status?.parts ?? Object.keys(cards));
  const claimed = new Set<string>();
  const stemCount = new Map<string, number>();
  for (const n of nodes) stemCount.set(n.id.split("#")[0], (stemCount.get(n.id.split("#")[0]) ?? 0) + 1);

  const make = (o: Partial<TNode> & { key: string; label: string; role: Role }): TNode => ({
    sub: undefined,
    about: "",
    tone: "neutral",
    layout: "auto",
    ids: [],
    anatomyIds: new Set(),
    settings: [],
    controls: [],
    present: true,
    children: [],
    path: [],
    ...o,
  });

  const subOf = (id: string) => {
    const crate = id.split(".")[0];
    return cards[crate]?.subcomponents.find((s) => s.id === id);
  };

  /** A leaf for one card sub-component. */
  const subLeaf = (id: string): TNode => {
    claimed.add(id);
    const crate = id.split(".")[0];
    const sc = subOf(id);
    const [label, about] = PLAIN[id] ?? [human(id), `A sealed sub-component of ${CRATE_COMPONENT[crate]?.label ?? crate}.`];
    return make({ key: slug(id.split(".").slice(1).join("-") || id), label, about, role: "part", ids: [id], crate, sub_hash: sc?.hash, sub_version: sc?.version, present: present.has(crate) });
  };

  /** A leaf for one anatomy node (and the card sub-component it is an instance of). */
  const anatomyLeaf = (a: AnatomyNode, key?: string): TNode => {
    const stem = a.id.split("#")[0];
    const crate = stem.split(".")[0];
    const plainCrate = CRATE_COMPONENT[stem] ? ([CRATE_COMPONENT[stem].label, CRATE_COMPONENT[stem].about] as [string, string]) : undefined;
    const [label, about] = PLAIN[a.id] ?? PLAIN[stem] ?? plainCrate ?? [human(a.id), `Anatomy node of kind ${a.kind}.`];
    const ids = [a.id];
    // A sub-component joins the anatomy node only when that node is its one instance.
    if (subOf(stem) && !claimed.has(stem) && (stemCount.get(stem) ?? 0) === 1) {
      ids.push(stem);
      claimed.add(stem);
    }
    const sc = subOf(stem);
    const inst = a.id.split("#")[1];
    const last = stem.split(".").slice(1).join("-");
    const auto = slug((stemCount.get(stem) ?? 0) > 1 && inst ? `${last}-${inst}` : last);
    return make({ key: key ?? auto, label, about, role: "part", ids, anatomy: a, crate, sub_hash: sc?.hash, sub_version: sc?.version, present: present.has(crate) });
  };

  /** A component node for a whole crate, its unclaimed sub-components as children. */
  const crateComponent = (crate: string, key?: string): TNode | null => {
    const card = cards[crate];
    if (!card) return null;
    const info = CRATE_COMPONENT[crate] ?? { label: crate, about: "" };
    const kids = card.subcomponents.filter((s) => !claimed.has(s.id)).map((s) => subLeaf(s.id));
    return make({ key: key ?? crate, label: info.label, about: info.about, role: "group", ids: [crate], crate, card, children: kids, present: present.has(crate) });
  };

  const root = make({ key: "", label: "Organism", sub: "Neuromorph Nano", role: "organism", tone: "neutral", layout: "organism", about: "The digital organism: a Core Brain, the GCX and a Neuropod of eight tentacles, with memory, continuous learning and the engine that runs its MicroModels.", ids: ["nmapi#n0"], anatomy: byId.get("nmapi#n0") });
  const mains = new Map<string, TNode>();
  for (const m of MAIN) {
    const n = make({ key: m.key, label: m.label, short: m.short, sub: m.sub, role: m.role, tone: m.tone, layout: m.layout, about: m.about, ids: [...m.crates], crate: m.crates[0], card: cards[m.crates[0]], settings: m.settings, controls: m.controls, present: m.crates.some((c) => present.has(c)) });
    mains.set(m.key, n);
  }
  const main = (k: string) => mains.get(k)!;

  /* Neuropod: eight tentacles, each with its drone tip (and its readers), core and g0. */
  const pod = main("neuropod");
  pod.anatomy = byId.get("nmpod.pod#p0");
  if (pod.anatomy) pod.ids.push(pod.anatomy.id);
  const tentacles: TNode[] = [];
  const tentacleNodes = nodes.filter((n) => n.kind === "tentacle").sort((a, b) => a.id.localeCompare(b.id));
  for (const t of tentacleNodes) {
    const n = Number(t.id.split("#t")[1] ?? tentacles.length);
    const lens = typeof t.config.default_dimension === "string" ? t.config.default_dimension : undefined;
    const kids: TNode[] = [];
    for (const c of nodes.filter((x) => x.parent === t.id)) {
      const role = typeof c.config.role === "string" ? c.config.role : c.kind;
      const key = c.kind === "drone" ? "tip" : role === "core" ? "core" : role === "g0_tx" ? "g0" : slug(c.id);
      const leaf = anatomyLeaf(c, key);
      if (c.kind === "drone") {
        leaf.label = "Drone tip";
        leaf.about = "The dendrites at the tentacle's tip: the first contact with the raw material. They sense, classify evidence and judge confidence.";
        const readers = Array.isArray(c.config.readers) ? (c.config.readers as unknown[]).filter((r): r is string => typeof r === "string") : [];
        leaf.children = readers.map((r, i) => {
          const hit = READER_NAMES.find(([re]) => re.test(r));
          return make({ key: `r${i}`, label: hit?.[1] ?? human(r), about: `${hit?.[2] ?? ""} Listed in the anatomy as: ${r}.`.trim(), role: "part", ids: [], present: true });
        });
      } else if (role === "core") {
        leaf.label = "Tentacle core";
        leaf.about = "Integrates what the tentacle's drones found into one report. Neural when a model is loaded, else the exact fallback.";
      } else if (role === "g0_tx") {
        leaf.label = "g0";
        leaf.about = "The tentacle's g0_tx: holds its encoded read and reports to G1 on the shared scale.";
      }
      kids.push(leaf);
    }
    const order = ["tip", "core", "g0"];
    kids.sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key));
    const tn = make({
      key: `t${n}`,
      label: `Tentacle ${n + 1}`,
      // No fixed job as its second line (2026-11-27): the plan gives a tentacle its job per call,
      // and the figures name the job only from the call's own steps. The anatomy's default view
      // stays in the description.
      about: `One autonomous reading pathway with its own window over the material. Every tentacle is general: ${tentacleNodes.length === 1 ? "it uses" : `all ${tentacleNodes.length} share`} the same weights and reading machinery, and each can take any job; the Core Brain's plan gives it one per question.${lens ? ` The anatomy sets ${lens} as its default view.` : ""} It owns no weights: the central copy is used by reference.`,
      role: "tentacle",
      tone: "cyan",
      ids: [t.id],
      anatomy: t,
      crate: "nmpod",
      children: kids,
      present: present.has("nmpod"),
    });
    tentacles.push(tn);
  }
  // The count the anatomy reports, not a fixed number.
  if (tentacleNodes.length > 0) pod.sub = `${tentacleNodes.length} tentacle${tentacleNodes.length === 1 ? "" : "s"}`;
  const mech = [...NEUROPOD_EXTRA.filter((id) => subOf(id)), ...(cards.nmpod?.subcomponents.map((s) => s.id) ?? [])].filter((id) => !claimed.has(id)).map(subLeaf);
  pod.children = [...tentacles, make({ key: "shared", label: "Shared mechanisms", about: "What every tentacle uses: the tentacle pass, territories, attention readers, recursion and the pipeline.", role: "group", ids: [], children: mech })];

  /* GCX: G3, G2, G1, g0 top to bottom. */
  const gcx = main("gcx");
  gcx.children = GCX_LEVELS.map((l) => {
    const kids: TNode[] = [];
    let anat: AnatomyNode | undefined;
    for (const a of l.anatomy ?? []) anat = byId.get(a);
    // Each tentacle's g0_tx lives inside its tentacle (its anatomy parent); the g0 level is drawn as
    // the ring those eight join, and holds only the shared g0 sub-component.
    for (const s of l.subs) if (subOf(s) && !claimed.has(s)) kids.push(subLeaf(s));
    for (const c of l.crates ?? []) {
      const comp = crateComponent(c);
      if (comp) kids.push(...comp.children);
    }
    const ids = [...(l.anatomy ?? []), ...(l.crates ?? [])];
    // G1 is one anatomy node: show it as a leaf with that node's ports.
    if (l.key === "g1" && anat) {
      return make({ key: l.key, label: l.label, sub: l.sub, about: l.about, role: "part", tone: "cyan", ids: [...ids, ...l.subs], anatomy: anat, crate: "nmbrain", present: present.has("nmbrain") });
    }
    return make({ key: l.key, label: l.label, sub: l.sub, about: l.about, role: "group", tone: "cyan", ids, anatomy: anat, children: kids, crate: l.crates?.[0], card: l.crates ? cards[l.crates[0]] : undefined, settings: l.key === "g3" ? ["muscle"] : l.key === "g2" ? ["reflexes"] : [] });
  });

  /* Core Brain: the brain's anatomy nodes, its MicroModel slots as one component, then groups. */
  const brain = main("core-brain");
  brain.anatomy = byId.get("nmbrain.stack#b0");
  if (brain.anatomy) brain.ids.push(brain.anatomy.id);
  const brainKids: TNode[] = [];
  const slots: TNode[] = [];
  // The GCX levels' anatomy nodes sit under the brain's stack in the anatomy; they are drawn in the GCX.
  const gcxAnatomy = new Set(GCX_LEVELS.flatMap((l) => l.anatomy ?? []));
  for (const a of nodes.filter((x) => x.parent === "nmbrain.stack#b0" && !gcxAnatomy.has(x.id))) {
    if (a.kind === "slot") {
      const role = a.id.split("#")[1];
      const leaf = anatomyLeaf(a, role);
      leaf.role = "model";
      leaf.tone = "violet";
      leaf.controls = SLOT_PRECISION[role] ? [SLOT_PRECISION[role]] : [];
      if (typeof a.config.card === "string") leaf.sub = `runs on ${a.config.card}`;
      slots.push(leaf);
    } else {
      brainKids.push(anatomyLeaf(a));
    }
  }
  for (const s of ["nmbrain.router", "nmbrain.writer", "nmbrain.embedder", "nmbrain.reranker"]) {
    // The slot roles also exist as card entries; they are the same parts as the anatomy slots.
    const slot = slots.find((x) => x.key === s.split(".")[1]);
    if (slot && subOf(s)) {
      slot.ids.push(s);
      claimed.add(s);
    }
  }
  if (slots.length) {
    brainKids.push(
      make({ key: "micromodels", label: "MicroModels", sub: "brain regions", about: "The Core Brain's regions: a MicroModel slot per job, each bound to a released model by the engine. Cartridges dock here.", role: "group", tone: "violet", ids: [], children: slots, controls: slots.flatMap((s) => s.controls) }),
    );
  }

  /* Claim everything GCX and Neuropod hold before groups and leftovers take the rest. */
  for (const l of GCX_LEVELS) for (const s of l.subs) claimed.add(s);

  const addGroups = (part: TNode) => {
    for (const g of GROUPS[part.key] ?? []) {
      const kids = g.subs.filter((s) => subOf(s) && !claimed.has(s)).map(subLeaf);
      // An anatomy node that is an instance of a grouped sub-component joins that leaf.
      for (const k of kids) {
        const inst = nodes.find((a) => a.id.split("#")[0] === k.ids[0] && a.parent === "nmapi#n0");
        if (inst && !mains.has(inst.id) && ![...mains.values()].some((m) => m.ids.includes(inst.id))) {
          k.ids.push(inst.id);
          k.anatomy = inst;
        }
      }
      if (kids.length) part.children.push(make({ key: g.key, label: g.label, about: g.about, role: "group", tone: part.tone, ids: [], children: kids }));
    }
  };

  brain.children = brainKids;
  addGroups(brain);

  /* Substrate: the views anatomy node plus the nmsubs sub-components. */
  const subs = main("substrate");
  const views = byId.get("nmsubs.views#g0");
  if (views) subs.children.push(anatomyLeaf(views, "views"));
  subs.children.push(...(crateComponent("nmsubs")?.children ?? []));
  const docs = crateComponent("nmdoc");
  if (docs) subs.children.push(docs);

  /* Memory: grouped sub-components (the write gate carries its anatomy node). */
  const mem = main("memory");
  addGroups(mem);

  /* Learning: the cartridges (learned, loaded or not), learning groups, the parameter map. */
  const learn = main("learning");
  // 2026-11-26 (first-time walkthrough: base cartridges were listed as learned): the base, hardware and
  // compensation cartridges the MicroModels run on are kept apart from what was learned.
  const isBase = (c: Cartridge) => c.kind === "tensors" || c.kind === "settings" || c.kind === "affine";
  const leaf = (c: Cartridge) =>
    make({ key: slug(c.id) || "cartridge", label: c.id, sub: c.kind === "tensors" ? "base cartridge" : c.kind === "settings" ? "hardware settings" : c.kind === "affine" ? "base compensation" : c.kind, about: c.description || `A ${c.kind} cartridge for ${c.base}.`, role: "cartridge", tone: "violet", ids: [`cartridge:${c.id}`], cartridge: c });
  const baseCarts = (cartridges ?? []).filter(isBase).map(leaf);
  const carts = (cartridges ?? []).filter((c) => !isBase(c)).map(leaf);
  learn.children.push(make({ key: "cartridges", label: "Cartridges", sub: "learned", about: "Learned cartridges: small, removable stacks of quantized parameters that attach beside a MicroModel and help comprehension for the job at hand. Removing one gives the model back exactly. Learning and preference cartridges are kept apart. Lenses, read-outs and learned pieces are listed under Learned units on Learning and Cartridges.", role: "group", tone: "violet", ids: [], children: carts }));
  if (baseCarts.length) learn.children.push(make({ key: "base-cartridges", label: "Base and hardware cartridges", sub: "what the MicroModels run on", about: "The Neuromorph base cartridges each MicroModel runs on, the hardware settings cartridge and the base compensation cartridges. They are not learned from use; they are built from the upstream open weights and chosen on Cartridges, Base versions.", role: "group", tone: "violet", ids: [], children: baseCarts }));
  addGroups(learn);

  /* Engine: the MicroModels (from /api/status), then its crates as components. */
  const engine = main("engine");
  const models = (status?.models ?? []).map((m) =>
    make({ key: slug(m.id), label: m.id, sub: kindTitle(roleOf?.(m.id) ?? "") || "MicroModel", about: `${m.id}: ${roleOf?.(m.id) ? `the ${kindTitle(roleOf(m.id))} MicroModel,` : "a MicroModel"} served from its Neuromorph base cartridge; the optional upstream open weights it was built from stay read only and are never altered.`, role: "model", tone: "violet", ids: [`model:${m.id}`], model: m, controls: [precisionKey(m.id)] }),
  );
  engine.children.push(make({ key: "micromodels", label: "MicroModels", sub: "from base cartridges", about: "The MicroModels the engine runs (Embedder, Reranker, Reader, Writer), each from its Neuromorph base cartridge and at its precision. Learned cartridges attach beside them and can be removed.", role: "group", tone: "violet", ids: [], children: models, controls: models.flatMap((m) => m.controls) }));

  /* Leftovers, then the crate components of each part. */
  for (const m of MAIN) {
    const part = main(m.key);
    if (m.key === "gcx" || m.key === "neuropod" || m.key === "substrate") continue;
    const homeIsGrouped = (GROUPS[m.key] ?? []).length > 0;
    for (const crate of m.crates) {
      const card = cards[crate];
      if (!card) continue;
      const left = card.subcomponents.filter((s) => !claimed.has(s.id));
      if (left.length === 0) continue;
      if (m.crates.length === 1 && homeIsGrouped) {
        part.children.push(make({ key: "other", label: "Other", about: "Further sealed sub-components of this system.", role: "group", tone: part.tone, ids: [], children: left.map((s) => subLeaf(s.id)) }));
      } else if (m.crates.length === 1) {
        part.children.push(...left.map((s) => subLeaf(s.id)));
      } else {
        const comp = crateComponent(crate);
        if (comp) part.children.push(comp);
      }
    }
  }
  // Runtime foundations: four small crates under one component.
  const rt = main("runtime");
  const foundations = rt.children.filter((c) => ["nmtype", "nmjson", "nmhash", "nmlex"].includes(c.crate ?? ""));
  if (foundations.length) {
    rt.children = rt.children.filter((c) => !foundations.includes(c));
    rt.children.push(make({ key: "foundations", label: "Foundations", about: "Shared contracts, JSON, hashing and the lexer: the ground every other component stands on.", role: "group", ids: [], children: foundations }));
  }

  root.children = MAIN.map((m) => main(m.key));

  /* Paths, parents, indexes, anatomy subtree sets. */
  const byPath = new Map<string, TNode>();
  const index = new Map<string, TNode[]>();
  const crateHome = new Map<string, TNode>();
  const walk = (n: TNode, path: string[], parent?: TNode) => {
    n.path = path;
    n.parent = parent;
    byPath.set(path.join("/"), n);
    for (const id of n.ids) {
      const list = index.get(id) ?? [];
      list.push(n);
      index.set(id, list);
    }
    const seen = new Set<string>();
    n.children = n.children.filter((c) => {
      let k = c.key || "x";
      while (seen.has(k)) k += "-";
      c.key = k;
      seen.add(k);
      return true;
    });
    for (const c of n.children) walk(c, [...path, c.key], n);
    for (const id of n.ids) if (byId.has(id)) n.anatomyIds.add(id);
    for (const c of n.children) c.anatomyIds.forEach((x) => n.anatomyIds.add(x));
  };
  walk(root, []);
  for (const m of MAIN) for (const c of m.crates) crateHome.set(c, main(m.key));

  return { root, byPath, index, crateHome, edges: anatomy?.edges ?? [], tentacles };
}

/** The tree nodes a part id from an audit row (`parts`) stands for. Ids are anatomy ids
 *  (`nmpod.unit#t0`), instances of a sub-component (`nmpod.pipeline#t3`, `nmmem.working#t2`),
 *  sub-components (`nmbrain.synthesis`) or crates (`nmengn`). */
export function resolvePart(tree: Tree, id: string): TNode[] {
  const exact = tree.index.get(id);
  if (exact?.length) return exact;
  const [stem, inst] = id.split("#");
  const crate = stem.split(".")[0];
  if (inst) {
    const t = /^[td](\d+)(?:-.*)?$/.exec(inst);
    if (t && (crate === "nmpod" || stem.startsWith("nmapi.tentacle") || stem.startsWith("nmapi.drone"))) {
      const tn = tree.tentacles.find((x) => x.key === `t${t[1]}`);
      if (tn) return [tn];
    }
  }
  const viaStem = tree.index.get(stem);
  if (viaStem?.length) return viaStem;
  const prefixed = [...tree.index.entries()].find(([k]) => k.startsWith(stem + "#"));
  if (prefixed) return prefixed[1];
  const home = tree.crateHome.get(crate);
  return home ? [home] : [];
}

/** True when `n` is `anc` or lies inside it. */
export function within(n: TNode, anc: TNode): boolean {
  for (let x: TNode | undefined = n; x; x = x.parent) if (x === anc) return true;
  return false;
}

/** One number for a node's hover: what it holds, or its precision. Only numbers the API gave. */
export function figure(n: TNode, status: Status | null): string {
  if (n.model && status?.loaded?.some((m) => m.id === n.model?.id)) return "held now";
  if (n.model) return n.model.precision === "default" ? "default precision" : `${n.model.precision}-bit`;
  if (n.cartridge) return n.cartridge.loaded ? "loaded" : "not loaded";
  if (n.role === "pod") return `${n.children.filter((c) => c.role === "tentacle").length} tentacles`;
  if (n.role === "gcx") return `${n.children.length} levels`;
  if (n.role === "organism") return `${n.children.length} systems, ${n.children.reduce((a, c) => a + c.children.length, 0)} components`;
  if (n.key === "cartridges" && n.role === "group") return n.children.length ? `${n.children.length} learned` : "no cartridges loaded";
  if (n.children.length) {
    const leaves = n.children.every((c) => c.children.length === 0);
    return `${n.children.length} ${leaves ? "sub-components" : "components"}`;
  }
  if (n.anatomy && (n.anatomy.ports_in.length || n.anatomy.ports_out.length)) return `${n.anatomy.ports_in.length} in · ${n.anatomy.ports_out.length} out`;
  if (n.sub_version !== undefined) return n.sub_version > 0 ? `version ${n.sub_version}` : "not built yet";
  return n.crate ?? "";
}

/** Ports of a node: its anatomy node's, else its card's. */
export function portsOf(n: TNode): { ins: Port[]; outs: Port[] } {
  if (n.anatomy && (n.anatomy.ports_in.length || n.anatomy.ports_out.length)) return { ins: n.anatomy.ports_in, outs: n.anatomy.ports_out };
  if (n.card && n.path.length <= 1) return { ins: n.card.ports_in, outs: n.card.ports_out };
  if (n.card && n.role === "group") return { ins: n.card.ports_in, outs: n.card.ports_out };
  return { ins: [], outs: [] };
}

export function pathHash(path: string[]): string {
  return path.length ? `#/${path.join("/")}` : "#/";
}
export function parseHash(hash: string): string[] {
  const h = hash.replace(/^#\/?/, "");
  return h ? h.split("/").filter(Boolean).map(decodeURIComponent) : [];
}

export type { Cartridge };

/** Rank of each anatomy node along the question's flow (longest path from a source), used to
 *  play a row's parts in the order a question runs through them. */
export function flowRanks(tree: Tree): Map<string, number> {
  const out = new Map<string, string[]>();
  const indeg = new Map<string, number>();
  for (const e of tree.edges) {
    // Feedback edges run against the flow (control back down); they would close loops.
    if (e.mode.kind === "feedback") continue;
    (out.get(e.from) ?? out.set(e.from, []).get(e.from)!).push(e.to);
    indeg.set(e.to, (indeg.get(e.to) ?? 0) + 1);
    if (!indeg.has(e.from)) indeg.set(e.from, indeg.get(e.from) ?? 0);
  }
  const rank = new Map<string, number>();
  const queue = [...indeg.entries()].filter(([, d]) => d === 0).map(([k]) => k);
  for (const k of queue) rank.set(k, 0);
  const left = new Map(indeg);
  while (queue.length) {
    const k = queue.shift()!;
    for (const t of out.get(k) ?? []) {
      rank.set(t, Math.max(rank.get(t) ?? 0, (rank.get(k) ?? 0) + 1));
      left.set(t, (left.get(t) ?? 1) - 1);
      if (left.get(t) === 0) queue.push(t);
    }
  }
  return rank;
}

/** Where a tree node sits along the flow: its earliest anatomy node, else by its part. */
export function nodeRank(n: TNode, ranks: Map<string, number>): number {
  let best = Infinity;
  for (const id of n.anatomyIds) best = Math.min(best, ranks.get(id) ?? Infinity);
  if (Number.isFinite(best)) return best;
  const main = n.path[0];
  const max = Math.max(0, ...ranks.values());
  if (main === "runtime") return n.ids.some((i) => i === "nmapi.tools" || i === "nmapi.mcp" || i === "nmapi.contract") ? -1 : max + 1;
  if (main === "substrate") return 0;
  if (main === "memory") return 2;
  if (main === "engine" || main === "learning") return max - 1;
  return max / 2;
}
