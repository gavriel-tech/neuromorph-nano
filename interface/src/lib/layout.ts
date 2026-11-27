/** Screen-space layouts for one level of the topology: the children of the node in view, sized
 *  to the space the view has. Pure functions: the same level and size give the same layout, so a
 *  zoom can compute where a child sat in its parent's layout.
 *
 *  Level 0 is the organism drawn as one body from the anatomy: the Core Brain as a mantle with one
 *  cell per component, the GCX as its spine from g0 up to G3, the Neuropod as a nerve ring with
 *  the eight tentacles leaving it, the material they read below, and memory, learning, the engine
 *  and the MCP runtime as organs on the body. Labels sit outside the figure on leader lines.
 *  Deeper levels are the inside of one part: cards with plain names first, the arm of a
 *  tentacle, the plates of the GCX, the models with their cartridge docks. */
import type { Cartridge } from "./api";
import { flowRanks, nodeRank, type TNode, type Tree, within } from "./topology";

export type Pt = { x: number; y: number };
export type Box = { x: number; y: number; w: number; h: number };

/** A label set outside the figure, joined to it by a leader line. */
export type Label = { x: number; y: number; anchor: "start" | "end" | "middle"; title: string; sub?: string; leader?: string; dot?: Pt };

/** A small lit-able mark inside a drawn part, standing for one deeper node. */
export type Cell = { key: string; x: number; y: number; r: number; kind: "cell" | "slot" | "g0" | "core" | "tip" | "dendrite" | "window" | "chip"; label?: string; w?: number; h?: number };

export type ArmGeo = { d: string; body: string; base: Pt; tip: Pt; marks: Cell[]; dendrites: Cell[]; window?: Cell; at: (s: number) => Pt };

export type Item =
  | { kind: "mantle"; node: TNode; key: string; box: Box; d: string; cells: Cell[]; links: string[]; label: Label }
  | { kind: "spine"; node: TNode; key: string; box: Box; x: number; top: number; bottom: number; vertebrae: (Cell & { node: TNode; title: string; sub?: string })[]; label: Label; vlabels: Label[] }
  | { kind: "ring"; node: TNode; key: string; box: Box; cx: number; cy: number; rx: number; ry: number; joints: Pt[]; label?: Label }
  | { kind: "arm"; node: TNode; key: string; box: Box; arm: ArmGeo; label?: Label }
  | { kind: "organ"; node: TNode; key: string; box: Box; organ: "memory" | "learning" | "engine" | "runtime" | "substrate"; label: Label; extra: OrganExtra }
  | { kind: "card"; node: TNode; key: string; box: Box }
  | { kind: "plate"; node: TNode; key: string; box: Box; chips: Cell[] }
  | { kind: "model"; node: TNode; key: string; box: Box; column: Column };

export type Column = { x: number; y: number; w: number; h: number; layers: number; model: string };
export type OrganExtra = { columns?: Column[]; band?: { x0: number; x1: number; y: number }; windows?: Cell[]; tray?: Box };

/** A drawn connection. `route` connections also carry signals; `mode` follows the anatomy edge. */
export type Edge = { from: string; to: string; d: string; kind: "flow" | "down" | "leader" | "body"; mode?: string; tick?: { x: number; y: number; a: number } };
/** An invisible path a signal can travel, between two live keys. */
export type Route = { from: string; to: string; d: string; down?: boolean };
export type Stub = { key: string; dir: "in" | "out"; label: string; d: string; x: number; y: number };

export type Level = {
  focus: TNode;
  W: number;
  H: number;
  items: Item[];
  edges: Edge[];
  routes: Route[];
  stubs: Stub[];
  /** 0: names only, 1: names and roles, 2: everything. */
  detail: 0 | 1 | 2;
  /** A section heading inside the level ("Also in this part"). */
  band?: { y: number; label: string };
  /** Small quiet notes that orient the reader (where a path comes from or goes). */
  notes?: Note[];
  /** The parent drawn as the frame of its own inside (the Core Brain's mantle). */
  frame?: string;
  /** The material under the tentacles, drawn as a ground line with windows. */
  ground?: { y: number; x0: number; x1: number; label?: string };
};
export type Note = { x: number; y: number; text: string; anchor: "start" | "end" | "middle" };

export const keyOf = (n: TNode) => n.path.join("/");
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const f = (n: number) => Math.round(n * 10) / 10;

/** Qwen3 models of every size used here have 28 layers; the engine may give `layers` instead. */
export function layersOf(model: { id: string; layers?: number } | undefined): number {
  if (model?.layers) return model.layers;
  return 28;
}

/* ---------- geometry helpers ---------- */

type Cubic = [Pt, Pt, Pt, Pt];
function bez(c: Cubic, s: number): Pt {
  const [a, b, d, e] = c;
  const m = 1 - s;
  return { x: m * m * m * a.x + 3 * m * m * s * b.x + 3 * m * s * s * d.x + s * s * s * e.x, y: m * m * m * a.y + 3 * m * m * s * b.y + 3 * m * s * s * d.y + s * s * s * e.y };
}
function tangent(c: Cubic, s: number): Pt {
  const p = bez(c, Math.max(0, s - 0.002));
  const q = bez(c, Math.min(1, s + 0.002));
  const l = Math.hypot(q.x - p.x, q.y - p.y) || 1;
  return { x: (q.x - p.x) / l, y: (q.y - p.y) / l };
}
const cubicD = (c: Cubic) => `M ${f(c[0].x)} ${f(c[0].y)} C ${f(c[1].x)} ${f(c[1].y)}, ${f(c[2].x)} ${f(c[2].y)}, ${f(c[3].x)} ${f(c[3].y)}`;

/** The part of a cubic between s0 and s1 as a polyline path (for signal routes along an arm). */
function segD(c: Cubic, s0: number, s1: number, n = 14): string {
  const pts: string[] = [];
  for (let i = 0; i <= n; i++) {
    const p = bez(c, s0 + ((s1 - s0) * i) / n);
    pts.push(`${i ? "L" : "M"} ${f(p.x)} ${f(p.y)}`);
  }
  return pts.join(" ");
}

/** A tapered arm: the outline of a cubic centreline whose width falls from w0 to w1. */
function taper(c: Cubic, w0: number, w1: number, n = 28): string {
  const left: Pt[] = [];
  const right: Pt[] = [];
  for (let i = 0; i <= n; i++) {
    const s = i / n;
    const p = bez(c, s);
    const t = tangent(c, s);
    const w = (w0 + (w1 - w0) * Math.pow(s, 0.8)) / 2;
    left.push({ x: p.x - t.y * w, y: p.y + t.x * w });
    right.push({ x: p.x + t.y * w, y: p.y - t.x * w });
  }
  const tip = bez(c, 1);
  return `M ${left.map((p) => `${f(p.x)} ${f(p.y)}`).join(" L ")} L ${f(tip.x)} ${f(tip.y)} L ${right
    .reverse()
    .map((p) => `${f(p.x)} ${f(p.y)}`)
    .join(" L ")} Z`;
}

/** A superellipse: the Core Brain's mantle, softer than an ellipse and not a box. */
function mantle(cx: number, cy: number, rx: number, ry: number, n = 2.6, gather = 0.1): string {
  const pts: string[] = [];
  const N = 72;
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const c = Math.cos(a);
    const s = Math.sin(a);
    // The lower half is drawn a little narrower, so the form gathers toward the spine.
    const k = s > 0 ? 1 - gather * s : 1;
    pts.push(`${f(cx + Math.sign(c) * Math.pow(Math.abs(c), 2 / n) * rx * k)} ${f(cy + Math.sign(s) * Math.pow(Math.abs(s), 2 / n) * ry)}`);
  }
  return `M ${pts.join(" L ")} Z`;
}

/** Splits text into at most `max` lines that fit `width` at a rough average glyph width. */
export function wrap(text: string, width: number, size: number, max = 2, mono = false): string[] {
  const per = size * (mono ? 0.6 : 0.52);
  const cap = Math.max(4, Math.floor(width / per));
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let cur = "";
  for (const w of words) {
    const next = cur ? `${cur} ${w}` : w;
    if (next.length <= cap) cur = next;
    else {
      if (cur) lines.push(cur);
      cur = w.length > cap ? w.slice(0, cap - 1) + "…" : w;
      if (lines.length === max) break;
    }
  }
  if (cur && lines.length < max) lines.push(cur);
  if (lines.length === max && words.join(" ").length > lines.join(" ").length) {
    const last = lines[max - 1];
    lines[max - 1] = last.length >= cap ? last.slice(0, cap - 1) + "…" : last + "…";
  }
  return lines;
}

/** Shortens text to fit a width. */
export function fit(text: string, width: number, size: number, mono = false): string {
  const per = size * (mono ? 0.6 : 0.52);
  const max = Math.floor(width / per);
  if (text.length <= max) return text;
  return max <= 1 ? "" : text.slice(0, Math.max(1, max - 1)) + "…";
}

/** First sentence of a description, for a card's role line. */
export function roleOf(n: TNode): string {
  if (n.sub && n.role !== "part") return n.sub;
  const s = n.about.split(/(?<=\.)\s/)[0] ?? "";
  return s.replace(/\.$/, "");
}

/* --------------------------------------------------------------------------------------------- */

export function layoutLevel(tree: Tree, focus: TNode, W: number, H: number, carts: Cartridge[] = []): Level {
  const detail: 0 | 1 | 2 = W < 560 || H < 380 ? 0 : W < 980 || H < 560 ? 1 : 2;
  if (focus.layout === "organism") return organism(tree, focus, W, H, detail, carts);
  if (focus.layout === "fan") return fan(focus, W, H, detail);
  if (focus.layout === "stack") return stack(tree, focus, W, H, detail);
  if (focus.role === "tentacle") return tentacle(focus, W, H, detail);
  if (focus.key === "micromodels" && focus.children.some((c) => c.model)) return models(focus, W, H, detail);
  return auto(tree, focus, W, H, detail);
}

/* ---------- Arms ---------- */

/** One tentacle's arm from its base to its tip, with its own parts marked along it: g0 near the
 *  base, the core halfway, the drone tip at the end with one dendrite per reader. */
function arm(t: TNode, c: Cubic, w0: number, scale: number, window?: { y: number; w: number }): ArmGeo {
  const at = (s: number) => bez(c, s);
  const marks: Cell[] = [];
  const dendrites: Cell[] = [];
  const kid = (k: string) => t.children.find((x) => x.key === k);
  const g0 = kid("g0");
  const core = kid("core");
  const tip = kid("tip");
  if (g0) marks.push({ key: keyOf(g0), ...at(0.16), r: 2.4 * scale, kind: "g0" });
  if (core) marks.push({ key: keyOf(core), ...at(0.52), r: 2.6 * scale, kind: "core" });
  if (tip) {
    const p = at(1);
    marks.push({ key: keyOf(tip), ...p, r: 3 * scale, kind: "tip" });
    const n = Math.max(1, tip.children.length);
    for (let i = 0; i < n; i++) {
      // Dendrites in an alternating row, smaller toward the tip.
      const s = 0.74 + (i / Math.max(1, n)) * 0.2;
      const q = at(s);
      const tg = tangent(c, s);
      const side = i % 2 ? 1 : -1;
      const off = (w0 * (1 - s * 0.7)) / 2 + 2.6 * scale;
      dendrites.push({ key: keyOf(tip.children[i] ?? tip), x: q.x - tg.y * off * side, y: q.y + tg.x * off * side, r: (1.9 - i * 0.25) * scale, kind: "dendrite" });
    }
  }
  const tipPt = at(1);
  const win = window ? { key: keyOf(t), x: tipPt.x - window.w / 2, y: window.y, r: 0, w: window.w, h: 2, kind: "window" as const } : undefined;
  return { d: cubicD(c), body: taper(c, w0, 0.8 * scale), base: at(0), tip: tipPt, marks, dendrites, window: win, at };
}

function armBox(g: ArmGeo): Box {
  let x0 = Infinity,
    x1 = -Infinity,
    y0 = Infinity,
    y1 = -Infinity;
  for (let i = 0; i <= 12; i++) {
    const p = g.at(i / 12);
    x0 = Math.min(x0, p.x);
    x1 = Math.max(x1, p.x);
    y0 = Math.min(y0, p.y);
    y1 = Math.max(y1, p.y);
  }
  return { x: x0 - 12, y: y0 - 4, w: x1 - x0 + 24, h: y1 - y0 + 20 };
}

/** Signal routes along one arm, between its parts, and from its base to a point above. */
function armRoutes(t: TNode, c: Cubic, up: { key: string; d: string } | null, down: { key: string; d: string } | null): Route[] {
  const r: Route[] = [];
  const k = keyOf(t);
  const has = (x: string) => t.children.some((c2) => c2.key === x);
  if (has("tip") && has("core")) r.push({ from: `${k}/tip`, to: `${k}/core`, d: segD(c, 1, 0.52) });
  if (has("core") && has("g0")) r.push({ from: `${k}/core`, to: `${k}/g0`, d: segD(c, 0.52, 0.16) });
  if (up) r.push({ from: has("g0") ? `${k}/g0` : k, to: up.key, d: `${segD(c, 0.16, 0)} ${up.d.replace(/^M/, "L")}` });
  if (down) r.push({ from: down.key, to: has("tip") ? `${k}/tip` : k, d: `${down.d} ${segD(c, 0, 1).replace(/^M/, "L")}`, down: true });
  // The tentacle as a whole reads out to its tip.
  if (has("tip")) r.push({ from: k, to: `${k}/tip`, d: segD(c, 0, 1), down: true });
  return r;
}

/* ---------- Level 0: the organism ---------- */
function organism(tree: Tree, focus: TNode, W: number, H: number, detail: 0 | 1 | 2, carts: Cartridge[]): Level {
  const narrow = W < 640;
  const cx = W / 2;
  const kids = new Map(focus.children.map((c) => [c.key, c]));
  const items: Item[] = [];
  const edges: Edge[] = [];
  const routes: Route[] = [];
  const sc = detail === 0 ? 0.8 : detail === 1 ? 0.9 : 1;

  const top = H * (narrow ? 0.07 : 0.06);
  const bh = clamp(H * (narrow ? 0.15 : 0.19), 54, 210);
  const bw = clamp(Math.min(bh * 1.62, W * (narrow ? 0.44 : 0.22)), 96, 330);
  const yB = top + 24 + bh / 2;
  const ringY = H * (narrow ? 0.6 : 0.6);
  const rx = clamp(bw * 0.36, 34, 104);
  const ry = Math.max(8, rx * 0.26);
  const tipY = H * (narrow ? 0.84 : 0.83);
  const bandY = Math.min(H - 14, tipY + H * (narrow ? 0.09 : 0.08));
  const spread = narrow ? W * 0.43 : Math.min(W * 0.36, H * 0.7);
  // Label columns: left and right of the body, outside the figure.
  const gutter = narrow ? 10 : clamp(W * 0.04, 24, 64);
  let xL = cx - bw / 2 - (narrow ? 14 : clamp(W * 0.075, 56, 130));
  let xR = cx + bw / 2 + (narrow ? 14 : clamp(W * 0.075, 56, 130));
  xL = Math.max(gutter + (narrow ? 60 : 120), xL);
  xR = Math.min(W - gutter - (narrow ? 60 : 120), xR);
  const lab = (x: number, y: number, anchor: "start" | "end", title: string, sub: string | undefined, from: Pt): Label => {
    const end = { x: anchor === "end" ? x + 8 : x - 8, y };
    const mid = Math.abs(from.y - y) < 2 ? "" : ` L ${f(from.x + (end.x - from.x) * 0.35)} ${f(y)}`;
    const leader = `M ${f(from.x)} ${f(from.y)}${mid || ""} L ${f(end.x)} ${f(end.y)}`;
    return { x, y, anchor, title, sub: detail > 0 ? sub : undefined, leader, dot: from };
  };

  /* Core Brain: a mantle with one cell per component. */
  const brain = kids.get("core-brain");
  if (brain) {
    const d = mantle(cx, yB, bw / 2, bh / 2);
    const cells: Cell[] = [];
    const ranks = flowRanks(tree);
    const flow = brain.children.filter((c) => c.children.length === 0).map((c, i) => ({ c, i, r: nodeRank(c, ranks) })).sort((a, b) => a.r - b.r || a.i - b.i).map((x) => x.c);
    const slots = brain.children.find((c) => c.key === "micromodels")?.children ?? [];
    const groups = brain.children.filter((c) => c.children.length > 0 && c.key !== "micromodels").flatMap((g) => g.children);
    const rows: [TNode[], number, number, Cell["kind"]][] = [
      [flow, -0.2, 0.66, "cell"],
      [groups, 0.04, 0.5, "cell"],
      [slots, 0.26, 0.42, "slot"],
    ];
    for (const [list, fy, fw, kind] of rows) {
      const n = list.length;
      list.forEach((c, i) => {
        const u = n > 1 ? i / (n - 1) - 0.5 : 0;
        const x = cx + u * bw * fw;
        // Rows follow the mantle's curve.
        const y = yB + fy * bh + u * u * bh * 0.16 * (fy < 0 ? 1 : -0.4);
        cells.push({ key: keyOf(c), x, y, r: (kind === "slot" ? 3.4 : 2.6) * sc, kind, label: c.label });
      });
    }
    const label = lab(xL, yB - bh * 0.22, "end", "Core Brain", brain.sub, { x: cx - bw / 2 + 4, y: yB - bh * 0.22 });
    // The brain's own wiring: the anatomy's edges between its components, drawn inside it.
    const cellOf = (id: string) => cells.find((c) => tree.byPath.get(c.key)?.anatomyIds.has(id));
    const links: string[] = [];
    for (const e of tree.edges) {
      const a = cellOf(e.from);
      const b = cellOf(e.to);
      if (!a || !b || a === b) continue;
      const my = (a.y + b.y) / 2 - (Math.abs(a.y - b.y) < 2 ? bh * 0.07 : 0);
      const d = `M ${f(a.x)} ${f(a.y)} Q ${f((a.x + b.x) / 2)} ${f(my)} ${f(b.x)} ${f(b.y)}`;
      links.push(d);
      routes.push({ from: a.key, to: b.key, d });
    }
    items.push({ kind: "mantle", node: brain, key: keyOf(brain), box: { x: cx - bw / 2, y: yB - bh / 2, w: bw, h: bh }, d, cells, links, label });
  }

  /* MCP and runtime: the port at the crown where agents connect. */
  const runtime = kids.get("runtime");
  const crownY = top;
  if (runtime) {
    const label = lab(xR, crownY, "start", narrow ? "MCP" : "MCP and runtime", runtime.sub, { x: cx + 5, y: crownY });
    items.push({ kind: "organ", node: runtime, key: keyOf(runtime), box: { x: cx - 10, y: crownY - 10, w: 20, h: 20 }, organ: "runtime", label, extra: {} });
    edges.push({ from: keyOf(runtime), to: "core-brain", d: `M ${cx} ${crownY + 4} L ${cx} ${yB - bh / 2}`, kind: "body" });
    routes.push({ from: "runtime", to: "core-brain", d: `M ${cx} ${crownY} L ${cx} ${yB}` });
    routes.push({ from: "core-brain", to: "runtime", d: `M ${cx} ${yB} L ${cx} ${crownY}` });
  }

  /* GCX: the spine from the ring up to the brain, one vertebra per level. */
  const gcx = kids.get("gcx");
  const spineTop = yB + bh / 2;
  const spineBot = ringY - ry;
  if (gcx) {
    const levels = gcx.children; // G3 first
    const n = levels.length || 1;
    const vertebrae = levels.map((c, i) => {
      const y = spineTop + ((i + 0.65) / (n + 0.3)) * (spineBot - spineTop);
      const w = (24 - i * 3) * sc;
      return { key: keyOf(c), node: c, x: cx, y, r: 0, w, h: 5 * sc, kind: "cell" as const, title: c.label, sub: c.sub };
    });
    const vlabels = vertebrae.map((v) => lab(xL, v.y, "end", v.title, v.sub, { x: cx - (v.w ?? 20) / 2 - 3, y: v.y }));
    const mid = (spineTop + spineBot) / 2;
    const label = lab(xR, mid, "start", narrow ? "GCX" : "Globalized Context Stack", "GCX: g0 up to G3, in RAM", { x: cx + 4, y: mid });
    items.push({ kind: "spine", node: gcx, key: keyOf(gcx), box: { x: cx - 18, y: spineTop, w: 36, h: spineBot - spineTop }, x: cx, top: spineTop, bottom: spineBot, vertebrae, label, vlabels });
    // Sensory up through the levels; G1 hands up into the brain.
    for (let i = vertebrae.length - 1; i > 0; i--) routes.push({ from: vertebrae[i].key, to: vertebrae[i - 1].key, d: `M ${cx} ${vertebrae[i].y} L ${cx} ${vertebrae[i - 1].y}` });
    const g1 = vertebrae.find((v) => v.node.key === "g1");
    if (g1) routes.push({ from: g1.key, to: "core-brain", d: `M ${cx} ${g1.y} L ${cx} ${yB + bh * 0.1}` });
    const top3 = vertebrae[0];
    if (top3) routes.push({ from: "core-brain", to: top3.key, d: `M ${cx} ${yB + bh * 0.1} L ${cx} ${top3.y}`, down: true });
  }

  /* Neuropod: the nerve ring, and the eight tentacles leaving it toward the material. */
  const pod = kids.get("neuropod");
  const tentacles = pod?.children.filter((c) => c.role === "tentacle") ?? [];
  const g0v = gcx?.children.find((c) => c.key === "g0");
  if (pod) {
    const joints: Pt[] = tentacles.map((_, i) => {
      const t = (i + 0.5) / Math.max(1, tentacles.length);
      const a = Math.PI * (1 - t);
      return { x: cx + rx * Math.cos(a) * 0.92, y: ringY + ry * Math.sin(a) * 0.92 };
    });
    const label = lab(xR, ringY, "start", "Neuropod", pod.sub, { x: cx + rx + 3, y: ringY });
    items.push({ kind: "ring", node: pod, key: keyOf(pod), box: { x: cx - rx - 6, y: ringY - ry - 6, w: rx * 2 + 12, h: ry * 2 + 12 }, cx, cy: ringY, rx, ry, joints, label });
    const winW = ((spread * 2) / Math.max(1, tentacles.length)) * 0.62;
    tentacles.forEach((t, i) => {
      const n = tentacles.length;
      const u = n > 1 ? (i - (n - 1) / 2) / ((n - 1) / 2) : 0;
      const base = joints[i];
      const tip = { x: cx + u * spread, y: tipY - Math.abs(u) * H * 0.035 };
      const c: Cubic = [base, { x: base.x + u * rx * 0.9, y: base.y + (tip.y - base.y) * 0.42 }, { x: tip.x - u * spread * 0.32, y: tip.y - (tip.y - base.y) * 0.22 }, tip];
      const geo = arm(t, c, 4.2 * sc, sc * 1.15, { y: bandY, w: winW });
      const spacing = (2 * spread) / Math.max(1, n - 1);
      const name = detail === 0 || spacing < 72 ? `T${i + 1}` : t.label;
      const label: Label = { x: tip.x, y: tip.y + 15 * sc, anchor: "middle", title: name, sub: detail === 2 && spacing > 80 ? t.sub?.replace(/ lens$/, "") : undefined };
      items.push({ kind: "arm", node: t, key: keyOf(t), box: armBox(geo), arm: geo, label });
      const toRing = `M ${f(base.x)} ${f(base.y)} L ${f(cx)} ${f(ringY - ry)} L ${f(cx)} ${f(spineBot)}`;
      routes.push(...armRoutes(t, c, g0v ? { key: keyOf(g0v), d: toRing } : null, brain ? { key: "core-brain/stack-plan", d: `M ${cx} ${yB} L ${cx} ${spineBot} L ${f(base.x)} ${f(base.y)}` } : null));
      routes.push({ from: "substrate", to: `${keyOf(t)}/tip`, d: `M ${f(tip.x)} ${f(bandY)} L ${f(tip.x)} ${f(tip.y)}` });
    });
    edges.push({ from: keyOf(pod), to: "gcx", d: `M ${cx} ${ringY - ry} L ${cx} ${spineBot}`, kind: "body" });
  }

  /* Context Substrate: the material below the tentacle tips, a window under each. */
  const subs = kids.get("substrate");
  if (subs) {
    const x0 = cx - spread - (narrow ? 4 : 30);
    const x1 = cx + spread + (narrow ? 4 : 30);
    const windows = items.flatMap((it) => (it.kind === "arm" && it.arm.window ? [it.arm.window] : []));
    const label: Label = narrow
      ? { x: cx, y: bandY + 16, anchor: "middle", title: "Context Substrate", sub: undefined }
      : lab(Math.max(gutter + 110, x0 - 16), bandY, "end", "Context Substrate", subs.sub, { x: x0, y: bandY });
    if (!narrow) label.leader = undefined;
    items.push({ kind: "organ", node: subs, key: keyOf(subs), box: { x: x0, y: bandY - 8, w: x1 - x0, h: 16 }, organ: "substrate", label, extra: { band: { x0, x1, y: bandY }, windows } });
  }

  /* Memory: the tiers beside the brain, fed by its emit. */
  const mem = kids.get("memory");
  if (mem) {
    const mw = 30 * sc;
    const ox = cx - bw / 2 - (narrow ? 22 : clamp(W * 0.05, 36, 70)) - mw;
    const oy = yB + bh * 0.26;
    const box = { x: ox, y: oy - 12 * sc, w: mw, h: 26 * sc };
    const label = lab(Math.min(xL, ox - 16), oy + 4, "end", "Memory", mem.sub, { x: ox - 3, y: oy });
    items.push({ kind: "organ", node: mem, key: keyOf(mem), box, organ: "memory", label, extra: {} });
    const a = { x: cx - bw / 2 + bw * 0.03, y: oy };
    const b = { x: ox + mw + 3, y: oy };
    edges.push({ from: "core-brain", to: keyOf(mem), d: `M ${f(a.x)} ${f(a.y)} L ${f(b.x)} ${f(b.y)}`, kind: "body", mode: "buffered" });
    routes.push({ from: "core-brain", to: keyOf(mem), d: `M ${cx} ${yB} L ${f(a.x)} ${f(a.y)} L ${f(b.x)} ${f(b.y)}` });
  }

  /* Engine: one layer column per MicroModel, each with a dock for cartridges. */
  const engine = kids.get("engine");
  const modelsNode = engine?.children.find((c) => c.key === "micromodels");
  const ex = cx + bw / 2 + (narrow ? 16 : clamp(W * 0.045, 30, 64));
  let engineRight = ex;
  if (engine) {
    const ms = modelsNode?.children ?? [];
    const colH = bh * 0.78;
    const colW = 4 * sc;
    const gap = (narrow ? 9 : 13) * sc;
    const columns: Column[] = ms.map((m, i) => ({ x: ex + i * gap, y: yB - colH / 2, w: colW, h: colH, layers: layersOf(m.model), model: m.model?.id ?? m.label }));
    engineRight = ex + Math.max(1, ms.length) * gap;
    const box = { x: ex - 4, y: yB - colH / 2 - 4, w: engineRight - ex + 4, h: colH + 8 };
    const label: Label = { x: Math.max(xR, engineRight + 18), y: yB - bh * 0.22, anchor: "start", title: "Engine", sub: detail > 0 ? (ms.length ? `${ms.length} MicroModels` : engine.sub) : undefined };
    items.push({ kind: "organ", node: engine, key: keyOf(engine), box, organ: "engine", label, extra: { columns } });
    edges.push({ from: keyOf(engine), to: "core-brain", d: `M ${f(ex - 6)} ${f(yB)} L ${f(cx + bw / 2 - 2)} ${f(yB)}`, kind: "body" });
    routes.push({ from: keyOf(engine), to: "core-brain", d: `M ${f(ex)} ${f(yB)} L ${f(cx)} ${f(yB)}` });
  }

  /* Continuous learning: the cartridge tray under the engine; cartridges dock from here. */
  const learn = kids.get("learning");
  if (learn) {
    const n = Math.max(1, Math.min(6, carts.length));
    const tw = Math.max(26, n * 9 + 8) * sc;
    const tx = ex - 4;
    const ty = yB + bh / 2 + (narrow ? 14 : 22);
    const tray = { x: tx, y: ty, w: tw, h: 14 * sc };
    const loaded = carts.filter((c) => c.loaded).length;
    const label: Label = {
      x: Math.max(xR, engineRight + 18, tx + tw + 18),
      y: ty + 7 * sc,
      anchor: "start",
      title: narrow ? "Learning" : "Continuous learning",
      sub: detail > 0 ? (carts.length ? `${loaded} of ${carts.length} cartridges loaded` : "no cartridges loaded") : undefined,
    };
    items.push({ kind: "organ", node: learn, key: keyOf(learn), box: { ...tray, x: tray.x - 3, y: tray.y - 3, w: tray.w + 6, h: tray.h + 6 }, organ: "learning", label, extra: { tray } });
    if (engine) {
      edges.push({ from: keyOf(learn), to: keyOf(engine), d: `M ${f(tx + 6)} ${f(ty)} L ${f(tx + 6)} ${f(yB + bh * 0.39 + 4)}`, kind: "body", mode: "buffered" });
      routes.push({ from: keyOf(learn), to: keyOf(engine), d: `M ${f(tx + 6)} ${f(ty)} L ${f(tx + 6)} ${f(yB)}` });
    }
  }

  // Labels stay inside the view.
  const est = (l: Label) => l.title.length * (detail === 0 ? 11.5 : 13) * 0.55;
  for (const it of items) {
    const l = "label" in it ? (it.label as Label | undefined) : undefined;
    if (!l) continue;
    if (l.anchor === "start") l.x = Math.min(l.x, W - 6 - est(l));
    if (l.anchor === "end") l.x = Math.max(l.x, 6 + est(l));
  }
  // The engine and learning labels share one column; keep it outside both organs.
  for (const it of items) if (it.kind === "organ" && (it.organ === "engine" || it.organ === "learning")) it.label.x = Math.max(...items.filter((o) => o.kind === "organ" && (o.organ === "engine" || o.organ === "learning")).map((o) => (o as { label: Label }).label.x));
  void tree;
  return { focus, W, H, items, edges, routes, stubs: [], detail };
}

/* ---------- Neuropod: the ring and its eight arms ---------- */
function fan(focus: TNode, W: number, H: number, detail: 0 | 1 | 2): Level {
  const items: Item[] = [];
  const routes: Route[] = [];
  const edges: Edge[] = [];
  const narrow = W < 640;
  const tentacles = focus.children.filter((c) => c.role === "tentacle");
  const others = focus.children.filter((c) => c.role !== "tentacle");
  const cardW = clamp(W * 0.2, 180, 260);
  const cardH = 92;
  const right = !narrow && others.length > 0;
  const areaW = right ? W - cardW - 40 : W;
  const cx = areaW / 2;
  const sc = detail === 0 ? 0.9 : 1.15;
  const ringY = H * (narrow ? 0.12 : 0.13);
  const rx = clamp(areaW * 0.09, 40, 120);
  const ry = rx * 0.24;
  const tipY = H * (narrow ? 0.7 : 0.78);
  const bandY = H * (narrow ? 0.79 : 0.9);
  const spread = areaW * (narrow ? 0.43 : 0.41);
  const joints: Pt[] = tentacles.map((_, i) => {
    const a = Math.PI * (1 - (i + 0.5) / Math.max(1, tentacles.length));
    return { x: cx + rx * Math.cos(a) * 0.92, y: ringY + ry * Math.sin(a) * 0.92 };
  });
  items.push({ kind: "ring", node: focus, key: keyOf(focus) + "/.ring", box: { x: cx - rx, y: ringY - ry, w: rx * 2, h: ry * 2 }, cx, cy: ringY, rx, ry, joints });
  const winW = ((spread * 2) / Math.max(1, tentacles.length)) * 0.62;
  tentacles.forEach((t, i) => {
    const n = tentacles.length;
    const u = n > 1 ? (i - (n - 1) / 2) / ((n - 1) / 2) : 0;
    const base = joints[i];
    const tip = { x: cx + u * spread, y: tipY - Math.abs(u) * H * 0.05 };
    const c: Cubic = [base, { x: base.x + u * rx * 1.1, y: base.y + (tip.y - base.y) * 0.45 }, { x: tip.x - u * spread * 0.22, y: tip.y - (tip.y - base.y) * 0.25 }, tip];
    const geo = arm(t, c, 4.4 * sc, sc, { y: bandY, w: winW });
    const spacing = (2 * spread) / Math.max(1, n - 1);
    const label: Label = { x: tip.x, y: tip.y + 22, anchor: "middle", title: detail === 0 || spacing < 72 ? `T${i + 1}` : t.label, sub: detail > 0 && spacing > 86 ? t.sub : undefined };
    items.push({ kind: "arm", node: t, key: keyOf(t), box: armBox(geo), arm: geo, label });
    routes.push(...armRoutes(t, c, { key: keyOf(focus), d: `M ${f(base.x)} ${f(base.y)} L ${f(cx)} ${f(ringY)}` }, null));
  });
  others.forEach((o, i) => {
    const x = right ? W - cardW - 16 : 16;
    const y = right ? 20 + i * (cardH + 14) : bandY + 26 + i * (cardH + 10);
    items.push({ kind: "card", node: o, key: keyOf(o), box: { x, y, w: right ? cardW : W - 32, h: cardH } });
  });
  const notes: Note[] = detail > 0 ? [{ x: cx, y: ringY - ry - 12, text: "nerve ring: each tentacle reports to G1 through its g0", anchor: "middle" }] : [];
  return { focus, W, H, items, edges, routes, stubs: [], detail, notes, ground: { y: bandY, x0: cx - spread - 20, x1: cx + spread + 20, label: detail > 0 ? "the material: one window per tentacle" : undefined } };
}

/* ---------- One tentacle: its arm, with its parts named beside it ---------- */
function tentacle(focus: TNode, W: number, H: number, detail: 0 | 1 | 2): Level {
  const items: Item[] = [];
  const edges: Edge[] = [];
  const narrow = W < 640;
  const ax = narrow ? 34 : clamp(W * 0.3, 120, 420);
  const y0 = H * 0.1;
  const y1 = H * 0.8;
  const bandY = H * 0.9;
  const sc = narrow ? 1.2 : 1.6;
  const c: Cubic = [
    { x: ax, y: y0 },
    { x: ax + (narrow ? 10 : 34), y: y0 + (y1 - y0) * 0.35 },
    { x: ax - (narrow ? 10 : 30), y: y0 + (y1 - y0) * 0.7 },
    { x: ax + (narrow ? 4 : 12), y: y1 },
  ];
  const geo = arm(focus, c, 5 * sc, sc, { y: bandY, w: narrow ? 46 : 120 });
  items.push({ kind: "arm", node: focus, key: keyOf(focus) + "/.arm", box: armBox(geo), arm: geo });
  const cardX = ax + (narrow ? 34 : clamp(W * 0.08, 70, 140));
  const cardW = clamp(W - cardX - (narrow ? 12 : 40), 180, 380);
  const cardH = narrow ? 84 : 104;
  const routes: Route[] = armRoutes(focus, c, null, null);
  for (const m of geo.marks) {
    const node = focus.children.find((k) => keyOf(k) === m.key);
    if (!node) continue;
    const y = clamp(m.y - cardH / 2, 8, H - cardH - 8);
    items.push({ kind: "card", node, key: m.key, box: { x: cardX, y, w: cardW, h: cardH } });
    edges.push({ from: focus.key, to: m.key, d: `M ${f(m.x + m.r + 4)} ${f(m.y)} L ${f(cardX)} ${f(clamp(m.y, y + 14, y + cardH - 14))}`, kind: "leader" });
  }
  const lens = focus.sub ? `its window over the material, ${focus.sub}` : "its window over the material";
  const tipNode = focus.children.find((k) => k.key === "tip");
  const readerNotes: Note[] = narrow || !tipNode ? [] : geo.dendrites.map((sk, i) => ({ x: Math.min(sk.x, ax) - 18, y: sk.y + 4, text: tipNode.children[i]?.label ?? "", anchor: "end" as const }));
  return {
    focus,
    W,
    H,
    items,
    edges,
    routes,
    stubs: [],
    detail,
    notes: [{ x: ax, y: y0 - 14, text: "from the nerve ring; reports to G1 through g0", anchor: narrow ? "start" : "middle" }, ...readerNotes],
    ground: { y: bandY, x0: Math.max(8, ax - (narrow ? 26 : 150)), x1: ax + (narrow ? 60 : 150), label: lens },
  };
}

/* ---------- GCX: plates stacked from G3 down to g0, sensory up and behaviour down ---------- */
function stack(tree: Tree, focus: TNode, W: number, H: number, detail: 0 | 1 | 2): Level {
  const items: Item[] = [];
  const edges: Edge[] = [];
  const routes: Route[] = [];
  const n = focus.children.length;
  const narrow = W < 640;
  const gap = clamp(H * 0.05, 14, 36);
  const bh = clamp((H - 70 - gap * (n - 1)) / Math.max(1, n), 64, 124);
  const side = narrow ? 22 : 64;
  const bw = clamp(W - side * 2 - (narrow ? 8 : 60), 240, 900);
  const x = (W - bw) / 2;
  const top = (H - (bh * n + gap * (n - 1))) / 2;
  focus.children.forEach((c, i) => {
    const y = top + i * (bh + gap);
    const chips: Cell[] = [];
    const inner = c.children;
    if (inner.length) {
      const textW = narrow ? 0 : clamp(bw * 0.36, 180, 300);
      const cx0 = x + (narrow ? 14 : textW + 24);
      const cw = x + bw - 14 - cx0;
      const per = inner.filter((k) => /^t\d+$/.test(k.key));
      if (per.length) {
        // One g0 per tentacle, numbered by its tentacle.
        const step = Math.min(44, cw / per.length);
        per.forEach((ch, j) => chips.push({ key: keyOf(ch), x: cx0 + step * (j + 0.5), y: y + (narrow ? bh - 22 : bh / 2 + 4), r: narrow ? 5 : 7, kind: "g0", label: `T${Number(ch.key.slice(1)) + 1}` }));
        let ox = cx0 + step * per.length + 12;
        for (const ch of inner.filter((k) => !per.includes(k))) {
          const w = ch.label.length * 6.4 + 22;
          if (ox + w > cx0 + cw) break;
          chips.push({ key: keyOf(ch), x: ox, y: y + (narrow ? bh - 34 : bh / 2 - 8), r: 0, w, h: 24, kind: "chip", label: ch.label });
          ox += w + 6;
        }
      } else {
        const chipH = 24;
        let cxp = cx0;
        let cy = y + (narrow ? bh - 34 : 14);
        for (const ch of inner) {
          const w = Math.min(cw, ch.label.length * 6.4 + 22);
          if (cxp + w > cx0 + cw) {
            cxp = cx0;
            cy += chipH + 6;
          }
          if (cy + chipH > y + bh - 6) break;
          chips.push({ key: keyOf(ch), x: cxp, y: cy, r: 0, w, h: chipH, kind: "chip", label: ch.label });
          cxp += w + 6;
        }
      }
    }
    items.push({ kind: "plate", node: c, key: keyOf(c), box: { x, y, w: bw, h: bh }, chips });
  });
  const plates = items as Extract<Item, { kind: "plate" }>[];
  // Sensory data travels up the left side; behaviour comes down the right side.
  const lx = x - (narrow ? 11 : 30);
  const rx = x + bw + (narrow ? 11 : 30);
  for (let i = plates.length - 1; i > 0; i--) {
    const a = plates[i];
    const b = plates[i - 1];
    const ya = a.box.y + a.box.h / 2;
    const yb = b.box.y + b.box.h / 2;
    edges.push({ from: a.key, to: b.key, d: `M ${f(lx)} ${f(ya)} L ${f(lx)} ${f(yb)}`, kind: "flow", tick: { x: lx, y: yb + 6, a: -90 } });
    edges.push({ from: b.key, to: a.key, d: `M ${f(rx)} ${f(yb)} L ${f(rx)} ${f(ya)}`, kind: "down", mode: "buffered", tick: { x: rx, y: ya - 6, a: 90 } });
    routes.push({ from: a.key, to: b.key, d: `M ${f(x)} ${f(ya)} L ${f(lx)} ${f(ya)} L ${f(lx)} ${f(yb)} L ${f(x)} ${f(yb)}` });
    routes.push({ from: b.key, to: a.key, d: `M ${f(x + bw)} ${f(yb)} L ${f(rx)} ${f(yb)} L ${f(rx)} ${f(ya)} L ${f(x + bw)} ${f(ya)}`, down: true });
  }
  void tree;
  const notes: Note[] =
    detail > 0 && !narrow
      ? [
          { x: lx, y: top - 14, text: "sensory, up", anchor: "middle" },
          { x: rx, y: top - 14, text: "behaviour, down", anchor: "middle" },
        ]
      : [];
  return { focus, W, H, items, edges, routes, stubs: [], detail, notes };
}

/* ---------- MicroModels in the engine: each model's layer column and its cartridge dock ---------- */
function models(focus: TNode, W: number, H: number, detail: 0 | 1 | 2): Level {
  const items: Item[] = [];
  const n = focus.children.length;
  const narrow = W < 640;
  const cols = narrow ? 1 : n <= 2 ? n : n <= 4 ? 2 : 3;
  const gap = 16;
  const cw = clamp((W - 48 - gap * (cols - 1)) / cols, 220, 520);
  const rows = Math.ceil(n / cols);
  const ch = clamp((H - 48 - gap * (rows - 1)) / rows, 110, 190);
  const ox = (W - (cw * cols + gap * (cols - 1))) / 2;
  const oy = (H - (ch * rows + gap * (rows - 1))) / 2;
  focus.children.forEach((m, i) => {
    const x = ox + (i % cols) * (cw + gap);
    const y = oy + Math.floor(i / cols) * (ch + gap);
    const column: Column = { x: x + 18, y: y + 16, w: 10, h: ch - 32, layers: layersOf(m.model), model: m.model?.id ?? m.label };
    items.push({ kind: "model", node: m, key: keyOf(m), box: { x, y, w: cw, h: ch }, column });
  });
  return { focus, W, H, items, edges: [], routes: [], stubs: [], detail };
}

/* ---------- Everything else: a flow when the children are connected, else a grid ---------- */
type Probe = Extract<Item, { kind: "card" }>;

/** For each anatomy edge, the visible items holding its two ends (deepest holder wins). */
function holders(items: Probe[], id: string): Probe[] {
  const hit = items.filter((it) => it.node.anatomyIds.has(id));
  if (hit.length <= 1) return hit;
  return hit.filter((a) => !hit.some((b) => b !== a && within(b.node, a.node)));
}

function siblingPairs(tree: Tree, items: Probe[]): [Probe, Probe, string][] {
  const seen = new Set<string>();
  const out: [Probe, Probe, string][] = [];
  for (const e of tree.edges) {
    for (const a of holders(items, e.from))
      for (const b of holders(items, e.to)) {
        if (a === b) continue;
        const k = a.key + ">" + b.key;
        if (seen.has(k)) continue;
        seen.add(k);
        out.push([a, b, e.mode.kind]);
      }
  }
  return out;
}

function auto(tree: Tree, focus: TNode, W: number, H: number, detail: 0 | 1 | 2): Level {
  const kids = focus.children;
  if (kids.length === 0) return { focus, W, H, items: [], edges: [], routes: [], stubs: [], detail };
  const probe: Probe[] = kids.map((c) => ({ kind: "card", node: c, key: keyOf(c), box: { x: 0, y: 0, w: 0, h: 0 } }));
  const pairs = siblingPairs(tree, probe);
  const lvl = pairs.length === 0 ? grid(tree, focus, probe, W, H, detail, 0, H) : flow(tree, focus, probe, pairs, W, H, detail);
  if (focus.role === "brain" && W >= 640) lvl.frame = mantle(W / 2, H / 2, W / 2 - 8, H / 2 - 8, 6, 0);
  return lvl;
}

const CARD_H = { 0: 64, 1: 92, 2: 104 } as const;

function grid(tree: Tree, focus: TNode, items: Probe[], W: number, H: number, detail: 0 | 1 | 2, y0: number, y1: number, edges: Edge[] = [], routes: Route[] = [], extra?: Partial<Level>): Level {
  const n = items.length;
  const pad = W < 560 ? 12 : 32;
  const gap = W < 560 ? 10 : 16;
  const aw = W - pad * 2;
  const ah = y1 - y0 - pad * 2;
  let best = { c: 1, w: 0, h: 0, score: -1 };
  for (let c = 1; c <= n; c++) {
    const r = Math.ceil(n / c);
    const w = Math.min(detail === 2 ? 300 : 260, (aw - gap * (c - 1)) / c);
    const h = Math.min(CARD_H[detail], (ah - gap * (r - 1)) / r);
    if (w < 150 || h < 52) continue;
    const score = Math.min(w, h * 3.2) * Math.min(h, w / 2.6);
    if (score > best.score) best = { c, w, h, score };
  }
  if (best.score < 0) {
    const c = Math.max(1, Math.floor((aw + gap) / (150 + gap)));
    best = { c, w: (aw - gap * (c - 1)) / c, h: 52, score: 0 };
  }
  const r = Math.ceil(n / best.c);
  const totalW = best.c * best.w + (best.c - 1) * gap;
  const totalH = r * best.h + (r - 1) * gap;
  const ox = (W - totalW) / 2;
  const oy = y0 + Math.max(pad, (y1 - y0 - totalH) / 2);
  items.forEach((it, i) => {
    const row = Math.floor(i / best.c);
    const col = i % best.c;
    const inRow = Math.min(best.c, n - row * best.c);
    const rowOx = ox + ((best.c - inRow) * (best.w + gap)) / 2;
    it.box = { x: rowOx + col * (best.w + gap), y: oy + row * (best.h + gap), w: best.w, h: best.h };
  });
  return { focus, W, H, items, edges, routes, stubs: stubsFor(tree, focus, items, W), detail, ...extra };
}

function flow(tree: Tree, focus: TNode, items: Probe[], pairs: [Probe, Probe, string][], W: number, H: number, detail: 0 | 1 | 2): Level {
  const connected = new Set(pairs.flatMap(([a, b]) => [a, b]));
  const free = items.filter((i) => !connected.has(i));
  const flowItems = items.filter((i) => connected.has(i));
  const rank = new Map<Probe, number>(flowItems.map((i) => [i, 0]));
  for (let pass = 0; pass < flowItems.length; pass++) {
    let changed = false;
    for (const [a, b] of pairs) {
      const r = (rank.get(a) ?? 0) + 1;
      if (r > (rank.get(b) ?? 0) && r < flowItems.length) {
        rank.set(b, r);
        changed = true;
      }
    }
    if (!changed) break;
  }
  const cols: Probe[][] = [];
  for (const it of flowItems) (cols[rank.get(it) ?? 0] ??= []).push(it);
  const columns = cols.filter(Boolean);
  for (let c = 1; c < columns.length; c++) {
    const prev = columns[c - 1];
    const pos = (it: Probe) => {
      const ins = pairs.filter(([, b]) => b === it).map(([a]) => prev.indexOf(a)).filter((x) => x >= 0);
      return ins.length ? ins.reduce((s, x) => s + x, 0) / ins.length : 0;
    };
    columns[c].sort((a, b) => pos(a) - pos(b));
  }
  const portrait = W / H < 0.9;
  const freeBand = free.length ? clamp((CARD_H[detail] + 60) / H, 0.18, 0.3) : 0;
  const flowH = H * (1 - freeBand);
  const pad = W < 560 ? 12 : 36;
  const edges: Edge[] = [];
  const routes: Route[] = [];
  const tickAt = (p: Pt, dx: number, dy: number) => ({ x: p.x, y: p.y, a: (Math.atan2(dy, dx) * 180) / Math.PI });
  if (!portrait) {
    const nc = columns.length;
    const colGap = clamp(W * 0.045, 28, 80);
    const cw = clamp((W - pad * 2 - colGap * (nc - 1)) / nc, 120, detail === 2 ? 280 : 230);
    const totalW = nc * cw + (nc - 1) * colGap;
    const ox = (W - totalW) / 2;
    columns.forEach((col, c) => {
      const gap = 16;
      const ch = clamp((flowH - pad * 2 - gap * (col.length - 1)) / col.length, 52, CARD_H[detail]);
      const totalH = col.length * ch + (col.length - 1) * gap;
      const oy = (flowH - totalH) / 2;
      col.forEach((it, r) => (it.box = { x: ox + c * (cw + colGap), y: oy + r * (ch + gap), w: cw, h: ch }));
    });
    for (const [a, b, mode] of pairs) {
      const A = a.box;
      const B = b.box;
      const back = B.x <= A.x;
      let d: string;
      let tick;
      if (!back) {
        const pa = { x: A.x + A.w, y: A.y + A.h / 2 };
        const pb = { x: B.x, y: B.y + B.h / 2 };
        const dx = (pb.x - pa.x) * 0.5;
        d = `M ${f(pa.x)} ${f(pa.y)} C ${f(pa.x + dx)} ${f(pa.y)}, ${f(pb.x - dx)} ${f(pb.y)}, ${f(pb.x)} ${f(pb.y)}`;
        tick = tickAt(pb, 1, 0);
      } else {
        const pa = { x: A.x + A.w / 2, y: A.y + (B.y > A.y ? A.h : 0) };
        const pb = { x: B.x + B.w / 2, y: B.y + (B.y > A.y ? 0 : B.h) };
        const dy = (pb.y - pa.y) * 0.5;
        d = `M ${f(pa.x)} ${f(pa.y)} C ${f(pa.x)} ${f(pa.y + dy)}, ${f(pb.x)} ${f(pb.y - dy)}, ${f(pb.x)} ${f(pb.y)}`;
        tick = tickAt(pb, 0, Math.sign(dy) || 1);
      }
      edges.push({ from: a.key, to: b.key, d, kind: "flow", mode, tick });
      routes.push({ from: a.key, to: b.key, d });
    }
  } else {
    const perRow = Math.max(1, Math.floor((W - pad * 2 + 10) / (150 + 10)));
    const wrapped: Probe[][] = [];
    for (const col of columns) for (let i = 0; i < col.length; i += perRow) wrapped.push(col.slice(i, i + perRow));
    const nr = wrapped.length;
    const rowGap = clamp(H * 0.025, 12, 32);
    const rh = clamp((flowH - pad * 2 - rowGap * (nr - 1)) / nr, 46, 72);
    const totalH = nr * rh + (nr - 1) * rowGap;
    const oy = (flowH - totalH) / 2;
    wrapped.forEach((row, r) => {
      const gap = 10;
      const cw = clamp((W - pad * 2 - gap * (row.length - 1)) / row.length, 110, 260);
      const totalW = row.length * cw + (row.length - 1) * gap;
      const ox = (W - totalW) / 2;
      row.forEach((it, c) => (it.box = { x: ox + c * (cw + gap), y: oy + r * (rh + rowGap), w: cw, h: rh }));
    });
    for (const [a, b, mode] of pairs) {
      const pa = { x: a.box.x + a.box.w / 2, y: a.box.y + a.box.h };
      const pb = { x: b.box.x + b.box.w / 2, y: b.box.y };
      const dy = (pb.y - pa.y) * 0.5;
      const d = `M ${f(pa.x)} ${f(pa.y)} C ${f(pa.x)} ${f(pa.y + dy)}, ${f(pb.x)} ${f(pb.y - dy)}, ${f(pb.x)} ${f(pb.y)}`;
      edges.push({ from: a.key, to: b.key, d, kind: "flow", mode, tick: tickAt(pb, 0, 1) });
      routes.push({ from: a.key, to: b.key, d });
    }
  }
  if (!free.length) return { focus, W, H, items, edges, routes, stubs: stubsFor(tree, focus, items, W), detail };
  const lvl = grid(tree, focus, free, W, H, detail, flowH, H, edges, routes, { band: { y: flowH + 4, label: "Also in this system" } });
  lvl.items = items;
  lvl.stubs = stubsFor(tree, focus, items, W);
  return lvl;
}

/** Short stubs for edges that come into this level from outside it, or leave it, named by the
 *  part at the other end. Few, and only where they meet nothing. */
function stubsFor(tree: Tree, focus: TNode, items: Probe[], W: number): Stub[] {
  const out: Stub[] = [];
  const nameOf = (id: string) => {
    const n = tree.index.get(id)?.[0];
    if (!n) return null;
    const main = tree.byPath.get(n.path[0] ?? "");
    if (!main || main === n) return n.label;
    if (n.role === "tentacle" || n.parent?.role === "tentacle") return main.short ?? main.label;
    return `${main.short ?? main.label}: ${n.label}`;
  };
  for (const e of tree.edges) {
    const fromIn = focus.anatomyIds.has(e.from);
    const toIn = focus.anatomyIds.has(e.to);
    if (fromIn === toIn) continue;
    const inside = fromIn ? e.from : e.to;
    const other = fromIn ? e.to : e.from;
    const it = holders(items, inside)[0];
    if (!it) continue;
    const label = nameOf(other);
    if (!label) continue;
    const dir = fromIn ? "out" : "in";
    const y = it.box.y + it.box.h / 2;
    const len = 26;
    const d = dir === "in" ? `M ${f(it.box.x - len)} ${f(y)} L ${f(it.box.x)} ${f(y)}` : `M ${f(it.box.x + it.box.w)} ${f(y)} L ${f(it.box.x + it.box.w + len)} ${f(y)}`;
    out.push({ key: it.key, dir, label, d, x: dir === "in" ? it.box.x - len - 4 : it.box.x + it.box.w + len + 4, y });
  }
  const hits = (x0: number, y0: number, x1: number, y1: number) => items.some((it) => x0 < it.box.x + it.box.w + 2 && x1 > it.box.x - 2 && y0 < it.box.y + it.box.h && y1 > it.box.y);
  const per = new Map<string, Stub>();
  for (const s of out) {
    if (per.has(s.key + s.dir)) continue;
    const tw = Math.min(160, (s.label.length + 5) * 5.6);
    const lx0 = s.dir === "in" ? s.x - tw : s.x;
    const lx1 = s.dir === "in" ? s.x : s.x + tw;
    if (lx0 < 4 || lx1 > W - 4) continue;
    if (hits(Math.min(lx0, s.x - 30), s.y - 16, Math.max(lx1, s.x + 30), s.y + 2)) continue;
    per.set(s.key + s.dir, s);
  }
  return [...per.values()];
}

/** Where a child sits in its parent's layout, for the zoom. */
export function rectOf(level: Level, key: string): Box | null {
  const it = level.items.find((i) => i.key === key);
  if (!it) return null;
  return it.box;
}
