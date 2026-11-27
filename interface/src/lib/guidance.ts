// The agent's guidance as `GET /api/guidance` gives it (reminders, loops, the step graph and the
// deliveries), the words the Guidance section shows for each, and the step graph's layout. Pure:
// no React and no imports, so `tests/guidance.test.mjs` runs it under plain node.

export type Who = "user" | "agent" | "nano" | string;

export type Reminder = {
  id: string;
  text: string;
  by: Who;
  via?: string;
  trigger: string;
  every: number;
  glob: string | null;
  repeat: string;
  max: number;
  files: string | null;
  /** For `error-repeats`: the error code or a piece of its message. */
  error?: string | null;
  /** For `error-repeats`: tool results carrying the error before it fires (2 to 5). */
  count?: number;
  enabled: boolean;
  created_ms?: number;
  updated_ms?: number;
  reset_ms?: number;
  delivered?: number;
};

export type LoopNow = { met: boolean; detail: string; session?: string };

export type Loop = {
  id: string;
  prompt: string;
  by: Who;
  via?: string;
  until: string;
  /** For `all-of` and `any-of`: the conditions combined. */
  parts?: string[];
  goal: number | null;
  max: number;
  iterations: number;
  /** The same unmet reason this many times in a row ends the loop `stalled`. */
  stall?: number;
  budget_ms?: number | null;
  budget_tools?: number | null;
  /** Stops in a row that gave the same unmet reason as the one before. */
  repeats?: number;
  unmet_seq?: string[];
  /** `active`, `done`, `exhausted`, `stalled`, `budget-spent` or `ended`. */
  state: string;
  enabled: boolean;
  since_ms?: number;
  created_ms?: number;
  ended_ms?: number | null;
  end_note?: string | null;
  last_unmet?: string | null;
  now?: LoopNow;
  blocks?: number;
  not_acted_antigravity?: number;
};

export type StepStatus = "done" | "ready" | "blocked";

export type Step = {
  id: string;
  title: string;
  by: Who;
  goal: number | null;
  depends_on: string[];
  /** An OR-join: at least one of these steps must be done as well. */
  depends_any?: string[];
  /** A probe: done when its check passes while its dependencies are met, with no edit. */
  probe?: boolean;
  done: boolean;
  done_by?: string | null;
  done_ms?: number | null;
  evidence?: Record<string, unknown> | null;
  check: string | null;
  paths: string[];
  rev?: number;
  status: StepStatus | string;
};

/** The evidence ledger's finding for one delivery (`ledger::GUIDANCE_RULES`). */
export type Finding = {
  rule: string;
  state: string;
  strength: string | null;
  why?: string;
  detail?: Record<string, unknown>;
  steps_after?: number | null;
};

/** A delivery's judgment: the ledger id, whether its window (the rest of its session) is closed,
 *  and the finding. */
export type DeliveryVerdict = {
  delivery: string;
  kind: string;
  item: string;
  window: "open" | "closed" | string;
  closed?: string | null;
  verdict: Finding;
};

export type Delivery = {
  t: number;
  session?: string;
  event?: string;
  kind: string;
  id: string;
  channel: string;
  host?: string;
  text: string;
  verdict?: DeliveryVerdict | null;
};

/** One ended loop, as `guidance-outcomes.jsonl` keeps it. */
export type Outcome = {
  id: string;
  by?: Who;
  prompt?: string;
  condition: { until: string; parts?: string[]; goal?: number | null };
  iterations: number;
  max?: number;
  stall?: number;
  budget?: { ms?: number | null; tools?: number | null; used_ms?: number; used_tools?: number };
  /** `met`, `cap`, `stalled`, `budget-spent`, `ended` or `removed`. */
  end: string;
  end_note?: string | null;
  unmet?: string[];
  verdicts?: { rule?: string; state?: string; continuations?: number };
  started_ms?: number;
  ended_ms?: number;
};

export type Verdicts = { rules?: string[]; judged?: number; by_kind?: Record<string, Record<string, number>> };

export type GuidanceReply = {
  project?: string;
  file?: string;
  version?: number;
  settings?: { reminders?: string; loops?: string; memory?: string };
  limits?: {
    loop_max_default?: number;
    loop_cap?: number;
    text_chars?: number;
    stall_default?: number;
    stall_max?: number;
    budget_ms_min?: number;
    budget_ms_max?: number;
    budget_tools_max?: number;
    parts_min?: number;
    parts_max?: number;
    error_count_min?: number;
    error_count_max?: number;
  };
  vocabulary?: { triggers?: string[]; repeats?: string[]; conditions?: string[]; parts?: string[]; loop_ends?: string[]; checks?: string[] };
  hosts?: { host: string; reminders: string; loops: string }[];
  reminders?: Reminder[];
  loops?: Loop[];
  steps?: Step[];
  counts?: { steps?: number; done?: number; ready?: number; blocked?: number };
  next_step?: string | null;
  deliveries?: Delivery[];
  delivered_total?: number;
  outcomes?: Outcome[];
  outcomes_total?: number;
  verdicts?: Verdicts;
  error?: string;
};

export const TRIGGERS = ["prompt", "session-start", "compact", "stop", "edit", "failure", "tools", "error-repeats"] as const;
export const REPEATS = ["always", "once", "max"] as const;
export const CONDITIONS = ["check-passes", "goal-done", "steps-done", "manual", "all-of", "any-of"] as const;
/** The conditions a combined condition (`all-of`, `any-of`) is made of. */
export const PARTS = ["check-passes", "goal-done", "steps-done", "manual"] as const;
export const LOOP_CAP = 20;
export const LOOP_DEFAULT = 3;
export const STALL_DEFAULT = 2;
export const STALL_MAX = 10;
export const ERROR_COUNT_MIN = 2;
export const ERROR_COUNT_MAX = 5;

/** "1st", "2nd", "3rd", "11th". */
export function nth(n: number): string {
  const k = Math.round(n);
  const teen = k % 100 >= 11 && k % 100 <= 13;
  const suf = teen ? "th" : k % 10 === 1 ? "st" : k % 10 === 2 ? "nd" : k % 10 === 3 ? "rd" : "th";
  return `${k}${suf}`;
}

/** When a reminder is delivered, in words. */
export function triggerWords(r: Pick<Reminder, "trigger" | "every" | "glob"> & Partial<Pick<Reminder, "error" | "count">>): string {
  const every = Math.max(1, r.every || 1);
  switch (r.trigger) {
    case "session-start":
      return "at a session start";
    case "prompt":
      return every === 1 ? "at every prompt" : `at every ${nth(every)} prompt`;
    case "compact":
      return "after a compaction";
    case "stop":
      return "at a stop";
    case "edit":
      return r.glob ? `before an edit of ${r.glob}` : "before every edit";
    case "failure":
      return "after a failed build or test";
    case "tools":
      return every === 1 ? "after every tool call" : `after every ${nth(every)} tool call`;
    case "error-repeats": {
      const k = Math.min(ERROR_COUNT_MAX, Math.max(ERROR_COUNT_MIN, r.count || ERROR_COUNT_MIN));
      return r.error ? `when ${r.error} comes back in ${k} tool results` : "when the same error comes back";
    }
    default:
      return r.trigger;
  }
}

/** How often per session, in words. */
export function repeatWords(r: Pick<Reminder, "repeat" | "max">): string {
  if (r.repeat === "once") return "once a session";
  if (r.repeat === "max") return `at most ${r.max} ${r.max === 1 ? "time" : "times"} a session`;
  return "every time";
}

/** One condition's short name, for a combined condition. */
export function partWords(c: string): string {
  switch (c) {
    case "check-passes":
      return "a check passes";
    case "goal-done":
      return "the goal is closed";
    case "steps-done":
      return "the steps are done";
    case "manual":
      return "ended by hand";
    default:
      return c;
  }
}

/** A loop's condition, in words. */
export function untilWords(l: Pick<Loop, "until" | "goal"> & Partial<Pick<Loop, "parts">>): string {
  switch (l.until) {
    case "all-of":
    case "any-of": {
      const parts = (l.parts ?? []).map(partWords).join(l.until === "all-of" ? " and " : " or ");
      const goal = l.goal !== null && l.goal !== undefined ? ` (goal ${l.goal})` : "";
      return `until ${l.until === "all-of" ? "all of" : "any of"}: ${parts || "no parts"}${goal}`;
    }
    case "check-passes":
      return "until the last build or test after the last edit passes";
    case "goal-done":
      return `until goal ${l.goal ?? "?"} is closed`;
    case "steps-done":
      return l.goal !== null && l.goal !== undefined ? `until every step of goal ${l.goal} is done` : "until every step is done";
    case "manual":
      return "until it is ended";
    default:
      return l.until;
  }
}

/** A duration in milliseconds, in words: "45 s", "12 min", "2 h". */
export function msWords(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 120) return `${s} s`;
  const m = Math.round(s / 60);
  if (m < 120) return `${m} min`;
  return `${Math.round(m / 60)} h`;
}

/** A loop's budgets, in words, or "" when it has none. */
export function budgetWords(l: Partial<Pick<Loop, "budget_ms" | "budget_tools">>): string {
  const out: string[] = [];
  if (l.budget_ms) out.push(msWords(l.budget_ms));
  if (l.budget_tools) out.push(`${l.budget_tools} tool ${l.budget_tools === 1 ? "call" : "calls"}`);
  return out.length ? `budget ${out.join(", ")}` : "";
}

/** The tone of a loop's state: done verified, an end without the condition attention, active
 *  signal, ended by hand muted. */
export function loopTone(state: string): "verified" | "attention" | "signal" | "muted" {
  if (state === "done") return "verified";
  if (state === "exhausted" || state === "stalled" || state === "budget-spent") return "attention";
  if (state === "active") return "signal";
  return "muted";
}

/** Why a loop ended, in words, from its outcome row's `end`. */
export function endWords(end: string): string {
  switch (end) {
    case "met":
      return "condition met";
    case "cap":
      return "maximum reached";
    case "stalled":
      return "stalled";
    case "budget-spent":
      return "budget spent";
    case "ended":
      return "ended by hand";
    case "removed":
      return "removed while active";
    default:
      return end;
  }
}

/** The tone of an outcome row's end. */
export function endTone(end: string): "verified" | "attention" | "muted" {
  if (end === "met") return "verified";
  if (end === "cap" || end === "stalled" || end === "budget-spent") return "attention";
  return "muted";
}

/** Who set an item, as the interface says it to the user. */
export function whoWords(by: Who): string {
  if (by === "agent") return "set by the agent";
  if (by === "nano") return "by the Nano";
  return "set by the user";
}

/** A comma-separated list as ids or globs, empty entries dropped, each once. */
export function listOf(text: string): string[] {
  const out: string[] = [];
  for (const x of text.split(",")) {
    const t = x.trim();
    if (t && !out.includes(t)) out.push(t);
  }
  return out;
}

/** The share of a loop's maximum used, 0 to 1. */
export function loopShare(l: Pick<Loop, "iterations" | "max">): number {
  const m = Math.max(1, l.max || 1);
  return Math.min(1, Math.max(0, (l.iterations || 0) / m));
}

export type GraphNode = { id: string; title: string; status: string; probe: boolean; layer: number; row: number; x: number; y: number };
/** `any` is an OR-join edge (`depends_any`); the others are AND edges (`depends_on`). */
export type GraphEdge = { from: string; to: string; any: boolean; x1: number; y1: number; x2: number; y2: number };
export type GraphLayout = { nodes: GraphNode[]; edges: GraphEdge[]; width: number; height: number };

export const NODE_W = 150;
export const NODE_H = 34;
const GAP_X = 48;
const GAP_Y = 14;
const PAD = 8;

/** Both dependency lists of a step: the AND-join, then the OR-join. */
function depsOf(s: Pick<Step, "depends_on"> & Partial<Pick<Step, "depends_any">>): string[] {
  return [...(s.depends_on ?? []), ...(s.depends_any ?? [])];
}

/** The step graph laid out left to right: a step's column is the length of its longest chain of
 *  dependencies over both lists (a dependency that is not a step, or a cycle, counts as none);
 *  within a column, steps keep the order they were added. */
export function layout(steps: (Pick<Step, "id" | "title" | "status" | "depends_on"> & Partial<Pick<Step, "depends_any" | "probe">>)[]): GraphLayout {
  const byId = new Map(steps.map((s) => [s.id, s]));
  const depth = new Map<string, number>();
  const visiting = new Set<string>();
  const depthOf = (id: string): number => {
    const known = depth.get(id);
    if (known !== undefined) return known;
    const s = byId.get(id);
    if (!s || visiting.has(id)) return -1;
    visiting.add(id);
    let d = 0;
    for (const p of depsOf(s)) {
      const pd = depthOf(p);
      if (pd >= 0) d = Math.max(d, pd + 1);
    }
    visiting.delete(id);
    depth.set(id, d);
    return d;
  };
  const rows = new Map<number, number>();
  const nodes: GraphNode[] = steps.map((s) => {
    const layer = Math.max(0, depthOf(s.id));
    const row = rows.get(layer) ?? 0;
    rows.set(layer, row + 1);
    return { id: s.id, title: s.title, status: s.status, probe: Boolean(s.probe), layer, row, x: PAD + layer * (NODE_W + GAP_X), y: PAD + row * (NODE_H + GAP_Y) };
  });
  const at = new Map(nodes.map((n) => [n.id, n]));
  const edges: GraphEdge[] = [];
  for (const s of steps) {
    const to = at.get(s.id);
    if (!to) continue;
    const any = new Set(s.depends_any ?? []);
    for (const p of depsOf(s)) {
      const from = at.get(p);
      if (!from) continue;
      edges.push({ from: p, to: s.id, any: any.has(p), x1: from.x + NODE_W, y1: from.y + NODE_H / 2, x2: to.x, y2: to.y + NODE_H / 2 });
    }
  }
  const layers = nodes.reduce((m, n) => Math.max(m, n.layer + 1), 0);
  const tallest = Math.max(0, ...Array.from(rows.values()));
  return {
    nodes,
    edges,
    width: layers === 0 ? 0 : PAD * 2 + layers * NODE_W + (layers - 1) * GAP_X,
    height: tallest === 0 ? 0 : PAD * 2 + tallest * NODE_H + (tallest - 1) * GAP_Y,
  };
}

/** The SVG path of an edge: a horizontal S-curve from the dependency's right side to the step's
 *  left side. */
export function edgePath(e: Pick<GraphEdge, "x1" | "y1" | "x2" | "y2">): string {
  const mid = (e.x1 + e.x2) / 2;
  return `M${e.x1} ${e.y1} C${mid} ${e.y1} ${mid} ${e.y2} ${e.x2} ${e.y2}`;
}

/** A title cut to fit a node, with an ellipsis when it was longer (the full title is the node's
 *  tooltip). */
export function fit(title: string, chars = 20): string {
  const t = title.trim();
  return t.length <= chars ? t : `${t.slice(0, Math.max(1, chars - 3))}...`;
}

/** A delivery's channel, in words. */
export function channelWords(d: Pick<Delivery, "channel" | "host">): string {
  switch (d.channel) {
    case "context":
      return "added to the agent's context";
    case "inject":
      return "injected into Antigravity";
    case "block":
      return "continued the agent at a stop";
    case "note":
      return "final note";
    case "none":
      return d.host === "antigravity" ? "not acted (Antigravity)" : "recorded";
    default:
      return d.channel;
  }
}

/** What a rule's state means for each delivery kind, in words. */
const STATE_WORDS: Record<string, Record<string, string>> = {
  step: { used: "followed", "passed-over": "passed over", none: "open", unknown: "not judged" },
  loop: { verified: "condition met before the cap", refuted: "ended without its condition met (cap, stalled or budget spent)", none: "open", unknown: "not judged" },
  reminder: { verified: "check passed after edits under its glob", refuted: "check failed after edits under its glob", none: "open", unknown: "not judged" },
};

/** A delivery's verdict in words: the state for its kind, the strength, and why when the rule
 *  could not judge it. `null` for a delivery no rule judges. */
export function verdictWords(v: DeliveryVerdict | null | undefined): string | null {
  if (!v) return null;
  const f = v.verdict;
  const said = STATE_WORDS[v.kind]?.[f.state] ?? f.state;
  const strength = f.strength ? ` (${f.strength})` : "";
  const why = (f.state === "unknown" || f.state === "none") && f.why ? `: ${f.why}` : "";
  return `${said}${strength}${why}`;
}

/** The tone of a verdict: verified for a delivery that held, attention for one that did not,
 *  muted while open or not judged. */
export function verdictTone(v: DeliveryVerdict | null | undefined): "verified" | "attention" | "muted" {
  const st = v?.verdict.state;
  if (st === "used" || st === "verified") return "verified";
  if (st === "passed-over" || st === "refuted") return "attention";
  return "muted";
}

/** The judged deliveries counted for one line: held, did not hold, open, not judged. */
export function verdictTally(v: Verdicts | undefined): { held: number; missed: number; open: number; unjudged: number } {
  const out = { held: 0, missed: 0, open: 0, unjudged: 0 };
  for (const states of Object.values(v?.by_kind ?? {})) {
    for (const [st, k] of Object.entries(states)) {
      if (st === "used" || st === "verified") out.held += k;
      else if (st === "passed-over" || st === "refuted") out.missed += k;
      else if (st === "none") out.open += k;
      else out.unjudged += k;
    }
  }
  return out;
}
