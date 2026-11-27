/** The plates a part opens into. Two kinds, chosen by what the part is:
 *  - a floorplan, like a chip layout, for a part with components and wiring: the die fills the
 *    frame, each component a block of cells (one band per sub-component), traces in the channels
 *    between blocks for the real anatomy edges, pads on the die's edge for its ports and for each
 *    connection that leaves it, and a legend of what each trace means;
 *  - a ring or nested-line chart for a part with levels or quantities: the GCX as four rings with
 *    their exploded stack, the MicroModels as their 28-layer matrices with the docks.
 *  A tentacle opens as an exploded view along its own axis. Labels are chips on the drawing;
 *  nothing is a box with text. */
import type { Cartridge, Status } from "./api";
import { clamp, emptyScene, hash01, HEADS_FALLBACK, keyOf, LAYERS_FALLBACK, type Box, type Line, type Pt, type Scene, type XY } from "./scene";
import { flowRanks, nodeRank, portsOf, type TNode, type Tree } from "./topology";

type Trace = { from: string; to: string; mode: string };
type Pad = { label: string; dir: "in" | "out"; inner?: string };

const HUE: Record<string, Line["hue"]> = { direct: "sense", buffered: "silver", conditional: "behav", asynchronous: "silver" };
const DASH: Record<string, number[] | undefined> = { buffered: [4, 3], asynchronous: [1, 3] };

/** Cells of one sub-component's band inside a block: an even array, every cell the same. The
 *  engine reports no per-cell quantity, so nothing varies; only the live layer lights a band. */
function fillBand(S: Scene, key: string, b: Box, sp: number, z: number) {
  const cols = Math.max(1, Math.floor(b.w / sp));
  const rows = Math.max(1, Math.floor(b.h / sp));
  const ox = b.x + (b.w - cols * sp) / 2 + sp / 2;
  const oy = b.y + (b.h - rows * sp) / 2 + sp / 2;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) S.pts.push({ x: ox + c * sp, y: oy + r * sp, r: sp * 0.22, a: 0.36, k: key, t: "b", z, cell: true });
}

/** A block's outline as a fine dotted rim of points: sharp, the nearest thing on the plate. */
function rim(S: Scene, key: string, b: Box, step: number) {
  const per = 2 * (b.w + b.h);
  for (let d = 0; d < per; d += step) {
    let x: number, y: number;
    if (d < b.w) [x, y] = [b.x + d, b.y];
    else if (d < b.w + b.h) [x, y] = [b.x + b.w, b.y + d - b.w];
    else if (d < 2 * b.w + b.h) [x, y] = [b.x + b.w - (d - b.w - b.h), b.y + b.h];
    else [x, y] = [b.x, b.y + b.h - (d - 2 * b.w - b.h)];
    S.pts.push({ x, y, r: 0.55, a: 0.62, k: key, t: "b", z: 0.05 });
  }
}

/** Splits an ordered list into `n` runs of nearly equal weight, keeping the order. */
function runs<T>(items: T[], weight: (t: T) => number, n: number): T[][] {
  const total = items.reduce((a, t) => a + weight(t), 0);
  const out: T[][] = [];
  let cur: T[] = [];
  let acc = 0;
  items.forEach((t, i) => {
    cur.push(t);
    acc += weight(t);
    const left = items.length - i - 1;
    const colsLeft = n - out.length - 1;
    if (colsLeft > 0 && (acc >= (total * (out.length + 1)) / n || left === colsLeft)) {
      out.push(cur);
      cur = [];
    }
  });
  if (cur.length) out.push(cur);
  return out;
}

/** The ports of a part and each anatomy edge that leaves it, as pads. */
function padsOf(tree: Tree, focus: TNode): Pad[] {
  const out: Pad[] = [];
  const { ins, outs } = portsOf(focus);
  for (const p of ins) out.push({ label: p.name, dir: "in" });
  for (const p of outs) out.push({ label: p.name, dir: "out" });
  const seen = new Set<string>();
  for (const e of tree.edges) {
    const fi = focus.anatomyIds.has(e.from);
    const ti = focus.anatomyIds.has(e.to);
    if (fi === ti) continue;
    const other = tree.index.get(fi ? e.to : e.from)?.[0];
    const main = other ? tree.byPath.get(other.path[0] ?? "") : undefined;
    if (!main) continue;
    const innerId = fi ? e.from : e.to;
    const inner = focus.children.find((c) => c.anatomyIds.has(innerId));
    const label = `${fi ? "to" : "from"} ${main.short ?? main.label}`;
    const k = `${label}|${inner ? keyOf(inner) : ""}`;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push({ label, dir: fi ? "out" : "in", inner: inner ? keyOf(inner) : undefined });
  }
  return out.slice(0, 20);
}

/** A floorplan that fills its frame: die, pad ring, blocks of cells, traces, pads, legend. */
function floorplan(S: Scene, tree: Tree, focus: TNode, caption: string): Scene {
  const { W, H } = S;
  const narrow = W < 640;
  const kids = focus.children;
  if (!kids.length) return S;
  const ranks = flowRanks(tree);
  const order = kids.map((c, i) => ({ c, i, r: nodeRank(c, ranks) })).sort((a, b) => a.r - b.r || a.i - b.i).map((x) => x.c);
  const weight = (n: TNode) => 1 + Math.min(12, n.children.length) * 0.45;
  const pads = narrow ? [] : padsOf(tree, focus);
  const ml = narrow ? 14 : clamp(W * 0.09, 76, 140);
  const top = narrow ? 18 : 40;
  const bottom = narrow ? 44 : clamp(H * 0.09, 64, 96);
  const die: Box = { x: ml, y: top, w: W - ml * 2, h: H - top - bottom };
  const ring = clamp(Math.min(W, H) * 0.04, 14, 34);
  const core: Box = { x: die.x + ring, y: die.y + ring, w: die.w - ring * 2, h: die.h - ring * 2 };
  const gut = clamp(Math.min(W, H) * 0.024, 9, 22);
  const n = order.length;
  const ncol = clamp(Math.round(Math.sqrt((n * core.w) / core.h / 1.4)), 1, n);
  const cols = runs(order, weight, ncol);
  const colW = cols.map((c) => c.reduce((a, t) => a + Math.sqrt(weight(t)), 0) / Math.max(1, c.length) + 0.6);
  const sumW = colW.reduce((a, b) => a + b, 0);
  const avail = core.w - gut * (cols.length - 1);
  const blocks: { node: TNode; box: Box; col: number }[] = [];
  let x = core.x;
  cols.forEach((col, ci) => {
    const w = (avail * colW[ci]) / sumW;
    const tot = col.reduce((a, t) => a + weight(t), 0);
    const ah = core.h - gut * (col.length - 1);
    let y = core.y;
    col.forEach((node) => {
      const h = (ah * weight(node)) / tot;
      blocks.push({ node, box: { x, y, w, h }, col: ci });
      y += h + gut;
    });
    x += w + gut;
  });

  // The die: a chamfered outline, the pad ring inside it, and a fine inner rule.
  const ch = 14;
  const outline = (b: Box, c: number): XY[] => [
    { x: b.x + c, y: b.y },
    { x: b.x + b.w - c, y: b.y },
    { x: b.x + b.w, y: b.y + c },
    { x: b.x + b.w, y: b.y + b.h - c },
    { x: b.x + b.w - c, y: b.y + b.h },
    { x: b.x + c, y: b.y + b.h },
    { x: b.x, y: b.y + b.h - c },
    { x: b.x, y: b.y + c },
  ];
  S.lines.push({ pts: outline(die, ch), a: 0.55, closed: true, w: 0.9 });
  S.lines.push({ pts: outline({ x: die.x + 5, y: die.y + 5, w: die.w - 10, h: die.h - 10 }, ch - 3), a: 0.22, closed: true, w: 0.6, dash: [1, 3] });
  S.lines.push({ pts: outline({ x: core.x - 6, y: core.y - 6, w: core.w + 12, h: core.h + 12 }, 6), a: 0.3, closed: true, w: 0.6 });

  const sp = clamp(Math.min(W, H) / 210, 3, 5);
  const ranksOrder = new Map(order.map((c, i) => [keyOf(c), i]));
  for (const b of blocks) {
    const key = keyOf(b.node);
    const inner: Box = { x: b.box.x + 3, y: b.box.y + 22, w: b.box.w - 6, h: b.box.h - 26 };
    const subs = b.node.children.length ? b.node.children : [b.node];
    const horiz = inner.w >= inner.h * 0.9;
    const m = subs.length;
    subs.forEach((sub, si) => {
      const band: Box = horiz
        ? { x: inner.x + (si / m) * inner.w + (si ? 1.5 : 0), y: inner.y, w: inner.w / m - (si ? 1.5 : 0) - (si < m - 1 ? 1.5 : 0), h: inner.h }
        : { x: inner.x, y: inner.y + (si / m) * inner.h + (si ? 1.5 : 0), w: inner.w, h: inner.h / m - (si ? 1.5 : 0) - (si < m - 1 ? 1.5 : 0) };
      fillBand(S, keyOf(sub), band, sp, 0.3);
      if (m > 1) rim(S, keyOf(sub), band, 3.2);
      if (S.detail === 2 && m > 1 && band.w > 70 && band.h > 26) S.texts.push({ x: band.x + 4, y: band.y + band.h - 5, text: sub.label.length * 6 > band.w - 8 ? sub.label.slice(0, Math.max(3, Math.floor((band.w - 14) / 6))) + "…" : sub.label, anchor: "start", kind: "mono", size: 9, quiet: true });
    });
    rim(S, key, b.box, 2.4);
    S.annots.push({ key, label: b.node.label, mx: b.box.x, my: b.box.y, x: b.box.x, y: b.box.y, w: b.box.w, h: b.box.h, side: "l", kind: "block", count: b.node.children.length || undefined });
    S.regions.push({ key, node: b.node, box: b.box });
    void ranksOrder;
  }

  // Traces in the channels between blocks, one per real connection, with vias at the corners.
  const boxOf = (k: string) => blocks.find((b) => keyOf(b.node) === k);
  const traces: Trace[] = [];
  const seenT = new Set<string>();
  for (const e of tree.edges) {
    const A = blocks.filter((b) => b.node.anatomyIds.has(e.from));
    const B = blocks.filter((b) => b.node.anatomyIds.has(e.to));
    for (const a of A)
      for (const b of B) {
        if (a === b) continue;
        const k = `${keyOf(a.node)}>${keyOf(b.node)}>${e.mode.kind}`;
        if (seenT.has(k)) continue;
        seenT.add(k);
        traces.push({ from: keyOf(a.node), to: keyOf(b.node), mode: e.mode.kind });
      }
  }
  const used = new Set<string>();
  const exitN = new Map<string, number>();
  const nextSlot = (k: string) => {
    const v = exitN.get(k) ?? 0;
    exitN.set(k, v + 1);
    return v;
  };
  traces.forEach((t, i) => {
    const a = boxOf(t.from);
    const b = boxOf(t.to);
    if (!a || !b) return;
    const lane = ((i % 7) - 3) * 2.2;
    const ya = a.box.y + 28 + ((nextSlot(t.from + ">") * 9) % Math.max(10, a.box.h - 36));
    const yb = b.box.y + 28 + ((nextSlot(t.to + "<") * 9 + 4) % Math.max(10, b.box.h - 36));
    let pts: XY[];
    if (b.col > a.col) {
      const cxh = a.box.x + a.box.w + gut / 2 + lane;
      const cxb = b.box.x - gut / 2 + lane;
      pts = [{ x: a.box.x + a.box.w, y: ya }, { x: cxh, y: ya }, { x: cxh, y: (ya + yb) / 2 }, { x: cxb, y: (ya + yb) / 2 }, { x: cxb, y: yb }, { x: b.box.x, y: yb }];
    } else if (b.col < a.col) {
      const cxh = a.box.x - gut / 2 + lane;
      const cxb = b.box.x + b.box.w + gut / 2 + lane;
      pts = [{ x: a.box.x, y: ya }, { x: cxh, y: ya }, { x: cxh, y: (ya + yb) / 2 }, { x: cxb, y: (ya + yb) / 2 }, { x: cxb, y: yb }, { x: b.box.x + b.box.w, y: yb }];
    } else {
      const cxh = a.box.x + a.box.w + gut / 2 + lane;
      pts = [{ x: a.box.x + a.box.w, y: ya }, { x: cxh, y: ya }, { x: cxh, y: yb }, { x: b.box.x + b.box.w, y: yb }];
    }
    used.add(t.mode);
    S.lines.push({ pts, a: 0.8, w: 1, hue: HUE[t.mode] ?? "silver", dash: DASH[t.mode], k: t.from });
    for (let q = 1; q < pts.length - 1; q++) S.pts.push({ x: pts[q].x, y: pts[q].y, r: 1.1, a: 0.85, k: t.from, t: "p", z: 0.05 });
    for (const e of [pts[0], pts[pts.length - 1]]) S.pts.push({ x: e.x, y: e.y, r: 1.6, a: 0.95, k: t.from, t: "p", z: 0.05 });
    S.routes.push({ from: t.from, to: t.to, pts });
  });

  // Pads on the die's edge: ins on the left, outs on the right, overflow on top and bottom; each
  // with a stub trace to the block inside that holds its end.
  const ins = pads.filter((p) => p.dir === "in");
  const outs = pads.filter((p) => p.dir === "out");
  const maxSide = Math.max(1, Math.floor((die.h - 40) / 30));
  const placed: { p: Pad; x: number; y: number; side: "l" | "r" | "t" | "b" }[] = [];
  const along = (list: Pad[], side: "l" | "r") => {
    const own = list.slice(0, maxSide);
    own.forEach((p, i) => placed.push({ p, side, x: side === "l" ? die.x : die.x + die.w, y: die.y + 30 + ((i + 0.5) / own.length) * (die.h - 60) }));
    return list.slice(maxSide);
  };
  const extra = [...along(ins, "l"), ...along(outs, "r")];
  extra.forEach((p, i) => placed.push({ p, side: i % 2 ? "b" : "t", x: die.x + 40 + ((Math.floor(i / 2) + 0.5) / Math.ceil(extra.length / 2)) * (die.w - 80), y: i % 2 ? die.y + die.h : die.y }));
  for (const { p, x, y, side } of placed) {
    const k = p.inner ?? keyOf(focus);
    for (let a = -2; a <= 2; a++) for (let b = -2; b <= 2; b++) S.pts.push({ x: x + a * 1.9, y: y + b * 1.9, r: 0.7, a: Math.abs(a) === 2 || Math.abs(b) === 2 ? 0.9 : 0.45, k, t: "p", z: 0.05 });
    const tx = side === "l" ? x - 10 : side === "r" ? x + 10 : x;
    const ty = side === "t" ? y - 10 : side === "b" ? y + 16 : y;
    S.annots.push({ key: k, label: p.label, mx: x, my: y, x: tx, y: ty, side: side === "l" ? "r" : side === "r" ? "l" : "c", kind: "pad" });
    const target = p.inner ? boxOf(p.inner) : undefined;
    if (target) {
      const tb = target.box;
      const inward: XY[] =
        side === "l"
          ? [{ x: x + 5, y }, { x: core.x - 3, y }, { x: core.x - 3, y: clamp(y, tb.y + 24, tb.y + tb.h - 6) }, { x: tb.x, y: clamp(y, tb.y + 24, tb.y + tb.h - 6) }]
          : side === "r"
            ? [{ x: x - 5, y }, { x: core.x + core.w + 3, y }, { x: core.x + core.w + 3, y: clamp(y, tb.y + 24, tb.y + tb.h - 6) }, { x: tb.x + tb.w, y: clamp(y, tb.y + 24, tb.y + tb.h - 6) }]
            : [{ x, y: side === "t" ? y + 5 : y - 5 }, { x, y: side === "t" ? core.y - 3 : core.y + core.h + 3 }];
      S.lines.push({ pts: inward, a: 0.55, w: 0.8, hue: p.dir === "in" ? "sense" : "behav", k: p.inner });
      if (p.dir === "in") S.routes.push({ from: keyOf(focus), to: p.inner!, pts: inward });
      else S.routes.push({ from: p.inner!, to: keyOf(focus), pts: [...inward].reverse(), down: true });
    } else {
      const stub: XY[] = side === "l" ? [{ x: x + 5, y }, { x: core.x - 3, y }] : side === "r" ? [{ x: x - 5, y }, { x: core.x + core.w + 3, y }] : [{ x, y: side === "t" ? y + 5 : y - 5 }, { x, y: side === "t" ? core.y - 3 : core.y + core.h + 3 }];
      S.lines.push({ pts: stub, a: 0.4, w: 0.7 });
    }
  }
  // The pad ring itself: a fine row of points between the die's edge and its core.
  const rr: Box = { x: die.x + ring / 2, y: die.y + ring / 2, w: die.w - ring, h: die.h - ring };
  rim(S, keyOf(focus), rr, 4.5);

  // Legend: what each colour of trace means, and what the marks are.
  const legend: [string, Line["hue"] | "pad" | "cells", number[] | undefined][] = [];
  if (used.has("direct")) legend.push(["direct", "sense", undefined]);
  if (used.has("buffered")) legend.push(["buffered", "silver", [4, 3]]);
  if (used.has("conditional")) legend.push(["conditional", "behav", undefined]);
  if (used.has("asynchronous")) legend.push(["asynchronous", "silver", [1, 3]]);
  if (pads.length) legend.push(["pad: port or outside link", "pad", undefined]);
  legend.push(["cells: one band per sub-component", "cells", undefined]);
  if (!narrow) {
    // 2026-11-27 (layout gap D1): the legend keeps two rows of 190 px columns at the right of the
    // die when they fit; in a narrower figure (1366x768) each column is as wide as its longest label
    // and the legend takes more rows, so it never runs over the caption and the counts at the left.
    const capEnd = die.x + Math.max(caption.length * 8.4, 44 * 6.1) + 20;
    const colW = (items: typeof legend) => 30 + Math.max(...items.map(([l]) => l.length * 6.1)) + 16;
    let rows = 2;
    let cols: (typeof legend)[] = [];
    for (; rows <= legend.length; rows++) {
      cols = Array.from({ length: Math.ceil(legend.length / rows) }, (_, c) => legend.slice(c * rows, c * rows + rows));
      const wide = cols.length * 190 <= die.x + die.w - capEnd;
      if (wide || cols.reduce((a, c) => a + colW(c), 0) <= die.x + die.w - capEnd) break;
    }
    rows = Math.min(rows, legend.length);
    const wide = cols.length * 190 <= die.x + die.w - capEnd;
    const widths = cols.map((c) => (wide ? 190 : colW(c)));
    const lx0 = die.x + die.w - widths.reduce((a, b) => a + b, 0);
    const step = rows > 2 ? 14 : 16;
    legend.forEach(([label, hue, dash], i) => {
      const c = Math.floor(i / rows);
      const lx = lx0 + widths.slice(0, c).reduce((a, b) => a + b, 0);
      const ly = die.y + die.h + 22 + (i % rows) * step;
      if (hue === "pad") for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) S.pts.push({ x: lx + 8 + a * 2, y: ly + b * 2, r: 0.7, a: 0.8, k: "", t: "p", z: 0.05 });
      else if (hue === "cells") for (let a = 0; a < 5; a++) S.pts.push({ x: lx + 2 + a * 4, y: ly, r: 1, a: 0.6, k: "", t: "b", z: 0.1 });
      else S.lines.push({ pts: [{ x: lx, y: ly }, { x: lx + 22, y: ly }], a: 0.85, w: 1, hue, dash });
      S.texts.push({ x: lx + 30, y: ly + 3.5, text: label, anchor: "start", kind: "mono", size: 10, quiet: true });
    });
  }
  S.texts.push({ x: die.x, y: die.y + die.h + (narrow ? 22 : 26), text: caption, anchor: "start", kind: "caps" });
  if (!narrow) S.texts.push({ x: die.x, y: die.y + die.h + 42, text: `${blocks.length} blocks · ${traces.length} traces · ${pads.length} pads`, anchor: "start", kind: "mono", size: 10, quiet: true });
  // Corner coordinates, as on a layout plot.
  if (!narrow) {
    S.texts.push({ x: die.x + 4, y: die.y - 8, text: "0,0", anchor: "start", kind: "num", size: 9, quiet: true });
    S.texts.push({ x: die.x + die.w - 4, y: die.y - 8, text: `${Math.round(die.w)},0`, anchor: "end", kind: "num", size: 9, quiet: true });
  }
  S.mist.push({ x: W / 2, y: H / 2, r: Math.max(die.w, die.h) * 0.75, a: 0.42 });
  return S;
}

/* --------------------------------------------------------------------------------------------- */

export function plateScene(tree: Tree, focus: TNode, W: number, H: number, carts: Cartridge[], status: Status | null, drones: Map<string, Set<string>> = new Map()): Scene {
  const droneModels = [...drones.keys()];
  const detail: Scene["detail"] = W < 560 || H < 380 ? 0 : W < 980 || H < 560 ? 1 : 2;
  const S = emptyScene(focus, W, H, "plate", detail);
  if (focus.key === "micromodels" && focus.children.length) return models(S, focus, carts, status, drones);
  if (focus.layout === "stack") return rings(S, focus, tree);
  if (focus.role === "tentacle") return tentacle(S, tree, focus, droneModels);
  const n = focus.children.length;
  return floorplan(S, tree, focus, `Plate · ${focus.label} · ${n} ${focus.children.every((c) => !c.children.length) ? "sub-components" : "components"}`);
}

/** A tentacle opened as an exploded view along its own axis, filling the frame: the drone tip
 *  with its readers reaching into the material on the left, the tentacle core in the middle, g0
 *  on the right where it joins the ring. Callouts above and below; underneath, the heads its
 *  attention drones read and the arm in cross-section. */
function tentacle(S: Scene, tree: Tree, focus: TNode, droneModels: string[]): Scene {
  const { W, H } = S;
  const narrow = W < 640;
  const kid = (k: string) => focus.children.find((c) => c.key === k);
  const tip = kid("tip");
  const core = kid("core");
  const g0 = kid("g0");
  const mx = narrow ? 26 : clamp(W * 0.07, 60, 130);
  const x0 = mx + (narrow ? 20 : 70);
  const x1 = W - mx - (narrow ? 10 : 40);
  const cy = narrow ? H * 0.3 : H * 0.31;
  const wMax = clamp(H * 0.22, 34, 230);
  const span = x1 - x0;
  // s runs from the tip (0, left) to g0 (1, right); each part has its stretch with a gap.
  const segs: [TNode | undefined, number, number][] = [
    [tip, 0, 0.24],
    [core, 0.29, 0.77],
    [g0, 0.82, 1],
  ];
  const X = (s: number) => x0 + s * span;
  const Y = (s: number) => cy + Math.sin(s * Math.PI * 1.6 + 0.4) * wMax * 0.16;
  const Wd = (s: number) => wMax * (0.1 + 0.9 * Math.pow(s, 0.7));
  S.lines.push({ pts: Array.from({ length: 60 }, (_, q) => ({ x: X(q / 59), y: Y(q / 59) })), a: 0.22, w: 0.6, dash: [6, 5] });
  const step = 2.2;
  segs.forEach(([n, s0, s1], si) => {
    if (!n) return;
    const k = keyOf(n);
    let row = 0;
    for (let x = X(s0); x <= X(s1); x += step, row++) {
      const s = (x - x0) / span;
      const w = Wd(s);
      const yc = Y(s);
      const rows = Math.max(4, Math.round(w / 2.4));
      for (let j = 0; j < rows; j++) {
        const v = j / (rows - 1) - 0.5;
        const rimF = Math.pow(Math.abs(v * 2), 6);
        const id = si * 99991 + row * 211 + j;
        if (si === 1) {
          const strand = Math.abs(v) < 0.06;
          if (!strand && rimF < 0.4 && hash01(id, 3) > 0.45) continue;
          S.pts.push({ x, y: yc + v * w, r: strand ? 0.8 : 0.45 + rimF * 0.4, a: strand ? 0.7 : 0.14 + rimF * 0.58, k, t: "b", z: strand ? 0.05 : rimF > 0.4 ? 0.12 : 0.4, cell: true });
        } else if (si === 2) {
          // g0: a dense register of cells in rows, brighter toward its middle.
          if (row % 2 || j % 2) continue;
          const f = 1 - Math.abs(v) * 1.4;
          S.pts.push({ x, y: yc + v * w, r: 0.7 + f * 0.4, a: 0.22 + f * 0.38 + rimF * 0.3, k, t: "m", z: 0.15 });
        } else {
          if (rimF < 0.4 && hash01(id, 9) > 0.4) continue;
          S.pts.push({ x, y: yc + v * w, r: 0.45 + rimF * 0.4, a: 0.14 + rimF * 0.6, k, t: "b", z: rimF > 0.4 ? 0.12 : 0.4, cell: true });
        }
        if (rimF < 0.4 && hash01(id, 11) < 0.12) S.pts.push({ x: x + 3, y: yc + v * w * 0.8 + 5, r: 1.1, a: 0.22, k, t: "b", z: 0.9 });
      }
    }
    S.regions.push({ key: k, node: n, box: { x: X(s0), y: cy - wMax * 0.75, w: X(s1) - X(s0), h: wMax * 1.5 } });
  });
  // The drone tip's dendrites reaching into the material: seven stems, each forking once; the
  // readers sit along the tip's last stretch.
  const tipX = X(0);
  const tipY = Y(0);
  if (tip) {
    const k = keyOf(tip);
    const strand = (x0: number, y0: number, a: number, len: number, f0: number, f1: number) => {
      for (let s = 0; s < len; s += 2.2) {
        const f = f0 + (f1 - f0) * (s / len);
        S.pts.push({ x: x0 + Math.cos(a) * s, y: y0 + Math.sin(a) * s, r: 0.5 + f * 0.7, a: 0.35 + f * 0.6, k, t: "b", z: 0.1 });
      }
    };
    for (let q = 0; q < 7; q++) {
      const a = Math.PI + (q / 6 - 0.5) * 1.4;
      const len = (narrow ? 26 : 64) * (1 - Math.abs(q / 6 - 0.5) * 0.5);
      const stem = len * 0.55;
      strand(tipX, tipY, a, stem, 1, 0.5);
      for (const side of [-1, 1]) strand(tipX + Math.cos(a) * stem, tipY + Math.sin(a) * stem, a + side * 0.4, len - stem, 0.5, 0);
    }
    tip.children.forEach((rd, j) => {
      const s = 0.05 + j * (0.17 / Math.max(1, tip.children.length - 1 || 1));
      const dir = j % 2 ? 1 : -1;
      const c = { x: X(s), y: Y(s) + dir * (Wd(s) / 2 + 12) };
      for (let q = 0; q < 16; q++) {
        const a = (q / 16) * Math.PI * 2;
        S.pts.push({ x: c.x + Math.cos(a) * 6, y: c.y + Math.sin(a) * 6, r: 0.8, a: 0.85, k: keyOf(rd), t: "b", z: 0.05, cell: true });
      }
      S.pts.push({ x: c.x, y: c.y, r: 1.4, a: 1, k: keyOf(rd), t: "b", z: 0.05, cell: true });
      S.annots.push({ key: keyOf(rd), label: rd.label, mx: c.x + 4, my: c.y + dir * 6, x: c.x + 20, y: c.y + dir * (34 + j * 6), side: "l", kind: "callout" });
    });
  }
  // The material: one faint column of chunks the tip reads, its window bracketed.
  const subs = tree.root.children.find((c) => c.key === "substrate");
  if (subs) {
    const mxm = mx - 4;
    for (let y = 52; y < cy + wMax * 1.3; y += 3.2)
      for (let c = 0; c < (narrow ? 2 : 6); c++) {
        const id = Math.round(y) * 13 + c;
        if (hash01(id, 21) > 0.55) continue;
        const inWin = Math.abs(y - tipY) < wMax * 0.55;
        S.pts.push({ x: mxm - c * 4, y, r: 0.6, a: inWin ? 0.6 : 0.2, k: keyOf(subs), t: "g", z: inWin ? 0.3 : 0.6 });
      }
    S.lines.push({ pts: [{ x: mx + 4, y: tipY - wMax * 0.55 }, { x: mx + 10, y: tipY - wMax * 0.55 }, { x: mx + 10, y: tipY + wMax * 0.55 }, { x: mx + 4, y: tipY + wMax * 0.55 }], a: 0.6, w: 0.8, hue: "sense" });
    if (!narrow) S.texts.push({ x: mx - 28, y: 40, text: "material", anchor: "start", kind: "caps" });
    if (!narrow) S.texts.push({ x: mx + 16, y: tipY + wMax * 0.55 + 14, text: "its window", anchor: "start", kind: "mono", size: 9.5, quiet: true });
  }
  // The ring it joins, on the right: an arc of g0.
  const ringX = W - mx + 6;
  for (let a = -0.5; a <= 0.5; a += 0.01) S.pts.push({ x: ringX - Math.cos(a) * 30 + 30, y: cy + Math.sin(a) * wMax * 1.6, r: 0.8, a: 0.55, k: g0 ? keyOf(g0) : keyOf(focus), t: "b", z: 0.2 });
  if (!narrow) S.texts.push({ x: ringX + 4, y: cy - wMax * 0.85, text: "to G1", anchor: "middle", kind: "caps" });

  // Callouts: the name above each part, what it does below.
  segs.forEach(([n, s0, s1]) => {
    if (!n) return;
    const sm = (s0 + s1) / 2;
    const xm = X(sm);
    const topY = cy - wMax * 0.5 - (narrow ? 40 : 80);
    S.annots.push({ key: keyOf(n), label: n.label, mx: xm, my: Y(sm) - Wd(sm) / 2 - 4, x: xm - 8, y: topY, side: "l", kind: "callout" });
    if (narrow) return;
    const first = n.about.split(". ")[0] ?? "";
    const per = Math.max(18, Math.floor(Math.min(X(s1) - X(s0), 300) / 7));
    const lines: string[] = [];
    let cur = "";
    for (const w of first.replace(/\.$/, "").split(" ")) {
      if ((cur + " " + w).trim().length > per) {
        lines.push(cur.trim());
        cur = w;
      } else cur += " " + w;
    }
    if (cur.trim()) lines.push(cur.trim());
    const yb = cy + wMax * 0.5 + 56;
    S.lines.push({ pts: [{ x: xm, y: Y(sm) + Wd(sm) / 2 + 6 }, { x: xm, y: yb - 16 }], a: 0.35, w: 0.6 });
    lines.slice(0, 4).forEach((l, j) => S.texts.push({ x: xm - 8, y: yb + j * 16, text: l, anchor: "start", kind: "sans", size: 12 }));
  });

  // Underneath: Fig. 2, the heads its attention drones read in each model the drones use; Fig. 3,
  // the arm in cross-section.
  const figTop = cy + wMax * 0.5 + 146;
  const figH = H - figTop - 46;
  if (!narrow && figH > 120) {
    const cell = clamp(figH / LAYERS_FALLBACK, 3, 10);
    let fx = x0;
    const att = tip?.children.find((c) => /attention/i.test(c.label)) ?? tip;
    droneModels.slice(0, 2).forEach((model) => {
      for (let l = 0; l < LAYERS_FALLBACK; l++)
        for (let h = 0; h < HEADS_FALLBACK; h++) S.pts.push({ x: fx + (h + 0.5) * cell, y: figTop + (LAYERS_FALLBACK - 1 - l + 0.5) * cell, r: cell * 0.36, a: 0.3, k: att ? keyOf(att) : keyOf(focus), t: "m", model, layer: l, head: h, z: 0.1 });
      for (let l = 0; l < LAYERS_FALLBACK; l += 4) S.texts.push({ x: fx - 6, y: figTop + (LAYERS_FALLBACK - 1 - l + 0.5) * cell + 3.5, text: String(l).padStart(2, "0"), anchor: "end", kind: "num", size: 9, quiet: true });
      S.texts.push({ x: fx, y: figTop - 8, text: model, anchor: "start", kind: "mono", size: 10 });
      fx += HEADS_FALLBACK * cell + 46;
    });
    if (droneModels.length) S.texts.push({ x: x0, y: figTop + LAYERS_FALLBACK * cell + 22, text: "Fig. 2 · heads its attention drones read · layers 00 to 27, bottom to top", anchor: "start", kind: "caps" });
    // Cross-section: skin, body, core strands, drawn as rings of points with leaders.
    const R = Math.min(figH * 0.34, 110);
    const ccx = Math.max(fx + R + 80, x0 + span * 0.7);
    const ccy = figTop + R + 4;
    const ringsX: [number, string, number, string | undefined][] = [
      [1, "skin", 0.7, keyOf(focus)],
      [0.72, "body", 0.25, keyOf(focus)],
      [0.16, "tentacle core", 0.95, core ? keyOf(core) : undefined],
    ];
    ringsX.forEach(([f, label, a, k], ri) => {
      const r = R * f;
      const nP = Math.floor((Math.PI * 2 * r) / (ri === 1 ? 5 : 2.6));
      for (let q = 0; q < nP; q++) {
        const aa = (q / nP) * Math.PI * 2;
        S.pts.push({ x: ccx + Math.cos(aa) * r, y: ccy + Math.sin(aa) * r, r: ri === 2 ? 1.1 : 0.7, a, k: k ?? keyOf(focus), t: "b", z: ri === 1 ? 0.6 : 0.1 });
      }
      const ly = ccy - R + ri * 30;
      S.lines.push({ pts: [{ x: ccx + r * 0.7, y: ccy - r * 0.7 }, { x: ccx + R + 30, y: ly }, { x: ccx + R + 44, y: ly }], a: 0.4, w: 0.6 });
      S.texts.push({ x: ccx + R + 50, y: ly + 3.5, text: label, anchor: "start", kind: "mono", size: 10.5 });
    });
    for (let q = 0; q < 220; q++) {
      const rr = R * 0.7 * Math.sqrt(hash01(q, 31));
      const aa = hash01(q, 32) * Math.PI * 2;
      if (rr < R * 0.2) continue;
      S.pts.push({ x: ccx + Math.cos(aa) * rr, y: ccy + Math.sin(aa) * rr, r: 0.5, a: 0.22, k: keyOf(focus), t: "b", z: 0.5 });
    }
    S.texts.push({ x: ccx - R, y: ccy + R + 30, text: "Fig. 3 · the arm in cross-section", anchor: "start", kind: "caps" });
  }
  S.texts.push({ x: x0, y: H - 18, text: `Fig. 1 · ${focus.label}${focus.sub ? ` · ${focus.sub}` : ""} · any job, assigned per question by the plan · read from the tip inward`, anchor: "start", kind: "caps" });
  const order = segs.map((s) => s[0]).filter(Boolean) as TNode[];
  for (const e of tree.edges) {
    const A = order.find((n) => n.anatomyIds.has(e.from));
    const B = order.find((n) => n.anatomyIds.has(e.to));
    if (!A || !B || A === B) continue;
    const ra = S.regions.find((r) => r.key === keyOf(A))?.box;
    const rb = S.regions.find((r) => r.key === keyOf(B))?.box;
    if (ra && rb) S.routes.push({ from: keyOf(A), to: keyOf(B), pts: [{ x: ra.x + ra.w / 2, y: cy }, { x: rb.x + rb.w / 2, y: cy }] });
  }
  S.mist.push({ x: W / 2, y: cy, r: W * 0.5, a: 0.45 });
  S.mist.push({ x: W * 0.4, y: H * 0.78, r: W * 0.3, a: 0.2 });
  return S;
}

/** The GCX: four rings in plan on the left, the same four levels exploded along one axis on the
 *  right, with callouts on both sides. */
function rings(S: Scene, focus: TNode, tree: Tree): Scene {
  const { W, H } = S;
  const narrow = W < 760;
  const levels = focus.children; // G3, G2, G1, g0
  const n = levels.length || 1;
  const cx = narrow ? W / 2 : W * 0.27;
  const cy = narrow ? H * 0.3 : H * 0.5;
  const Rmax = narrow ? Math.min(W * 0.42, H * 0.26) : Math.min(W * 0.23, H * 0.43);
  const ringR = (i: number) => Rmax * (0.32 + (0.68 * (i + 1)) / n);
  // Plan: each ring a band of points, its parts as arcs, each part named outside its arc.
  levels.forEach((lv, i) => {
    const R = ringR(i);
    const parts = lv.children.length ? lv.children : [lv];
    const total = Math.floor((Math.PI * 2 * R) / 2.6);
    for (let q = 0; q < total; q++) {
      const a = -Math.PI / 2 + (q / total) * Math.PI * 2;
      const pos = (q / total) * parts.length;
      const pi = Math.min(parts.length - 1, Math.floor(pos));
      if (parts.length > 1 && pos - pi < 0.05) continue;
      for (const dr of [-2.6, 0, 2.6]) S.pts.push({ x: cx + Math.cos(a) * (R + dr), y: cy + Math.sin(a) * (R + dr), r: dr === 0 ? 0.9 : 0.6, a: dr === 0 ? 0.75 : 0.4 + (q % 2) * 0.15, k: keyOf(parts[pi]), t: "b", z: dr === 0 ? 0.1 : 0.3, cell: true });
    }
    S.lines.push({ pts: circle(cx, cy, R + 9, 160), a: 0.14, closed: true, w: 0.5, dash: [1, 4] });
    S.texts.push({ x: cx + 5, y: cy - R - 6, text: lv.label, anchor: "start", kind: "mono", size: 11 });
    S.regions.push({ key: keyOf(lv), node: lv, box: { x: cx - R, y: cy - R, w: R * 2, h: R * 2 } });
  });
  // The measured scale: one tick per ring along a radius, numbered with its part count.
  const r0 = ringR(-1) * 0.6;
  S.lines.push({ pts: [{ x: cx, y: cy }, { x: cx + Rmax + 28, y: cy }], a: 0.35, w: 0.6 });
  levels.forEach((lv, i) => {
    const R = ringR(i);
    S.lines.push({ pts: [{ x: cx + R, y: cy - 6 }, { x: cx + R, y: cy + 6 }], a: 0.7, w: 0.8 });
    S.texts.push({ x: cx + R, y: cy + 19, text: String(lv.children.length), anchor: "middle", kind: "num", size: 10 });
  });
  for (let q = 0; q < 120; q++) {
    const rr = r0 * Math.sqrt((q + 0.5) / 120);
    const a = q * 2.39996;
    S.pts.push({ x: cx + Math.cos(a) * rr, y: cy + Math.sin(a) * rr, r: 0.6, a: 0.35 + 0.4 * (1 - rr / r0), k: keyOf(focus), t: "b", z: 0.5 });
  }
  S.texts.push({ x: cx, y: cy + r0 + 14, text: "brain", anchor: "middle", kind: "mono", size: 9.5, quiet: true });
  S.texts.push({ x: cx - Rmax, y: Math.min(H - 18, cy + Rmax + 34), text: "Fig. 1 · the four levels in plan · components per level", anchor: "start", kind: "caps" });

  // Exploded view: the same levels separated along one axis, G3 at the top, each a disc seen at
  // an angle (the near half brighter) with its components as cells on it, and between the discs
  // every real anatomy edge of the GCX as one fine line: up in the sensory colour, control back
  // down in the behaviour colour.
  if (!narrow) {
    const ex = W * 0.7;
    const rx = Math.min(W * 0.15, 260);
    const ry = rx * 0.26;
    const gap = Math.min((H - 160) / n, rx * 1.1);
    const y0 = H / 2 - ((n - 1) * gap) / 2 + 10;
    const yOf = (i: number) => y0 + i * gap;
    const onDisc = (i: number, a: number, f: number): XY => ({ x: ex + Math.cos(a) * rx * f, y: yOf(i) + Math.sin(a) * ry * f });
    S.lines.push({ pts: [{ x: ex, y: y0 - ry - 44 }, { x: ex, y: yOf(n - 1) + ry + 44 }], a: 0.18, w: 0.6, dash: [6, 4] });
    /** Where each anatomy id sits on the discs, for the edges. */
    const pos = new Map<string, XY>();
    levels.forEach((lv, i) => {
      const y = yOf(i);
      const parts = lv.children.length ? lv.children : [lv];
      const total = 240;
      for (const dr of [-2, 0, 2])
        for (let q = 0; q < total; q++) {
          const a = (q / total) * Math.PI * 2;
          const pi = Math.min(parts.length - 1, Math.floor((q / total) * parts.length));
          const near = Math.sin(a) > 0;
          S.pts.push({ x: ex + Math.cos(a) * (rx + dr), y: y + Math.sin(a) * (ry + dr * 0.3), r: dr === 0 ? 0.85 : 0.55, a: (near ? 0.75 : 0.35) * (dr === 0 ? 1 : 0.6), k: keyOf(parts[pi]), t: "b", z: near ? 0.1 : 0.55, cell: true });
        }
      // The disc's surface: even rings of faint points, brighter toward the near edge.
      for (let f = 0.14; f < 0.97; f += 0.11) {
        const m = Math.round(80 * f);
        for (let q = 0; q < m; q++) {
          const a = (q / m) * Math.PI * 2;
          const p = onDisc(i, a, f);
          S.pts.push({ x: p.x, y: p.y, r: 0.6, a: 0.2 + 0.14 * (0.5 + 0.5 * Math.sin(a)), k: keyOf(lv), t: "w", z: 0.5 });
        }
      }
      // Components as cells on the disc, one cluster each at its arc's middle, its size by what it holds.
      if (parts.length > 1 || parts[0] !== lv) {
        parts.forEach((p, j) => {
          const a = ((j + 0.5) / parts.length) * Math.PI * 2;
          const c = onDisc(i, a, 0.66);
          const m = 10 + Math.min(10, p.children.length) * 4;
          const rr = Math.min(18, 6 + Math.sqrt(m) * 1.4);
          for (let q = 0; q < m; q++) {
            const r0 = rr * Math.sqrt((q + 0.5) / m);
            const aa = q * 2.39996;
            S.pts.push({ x: c.x + Math.cos(aa) * r0, y: c.y + Math.sin(aa) * r0 * 0.42, r: 0.7, a: 0.75 * (Math.sin(a) > 0 ? 1 : 0.6), k: keyOf(p), t: "b", z: 0.12 });
          }
          for (const id of p.anatomyIds) pos.set(id, c);
        });
      }
      for (const id of lv.anatomyIds) if (!pos.has(id)) pos.set(id, { x: ex, y });
      S.annots.push({ key: keyOf(lv), label: `${lv.label}  ${lv.sub ?? ""}`.trim(), mx: ex - rx, my: y, x: ex - rx - 18, y: y - ry - 16, side: "l", kind: "callout" });
      parts.slice(0, 5).forEach((p, j) => {
        const a = -0.6 + (j / Math.max(1, Math.min(5, parts.length) - 1 || 1)) * 1.2;
        if (p !== lv) S.annots.push({ key: keyOf(p), label: p.label, mx: ex + Math.cos(a) * rx, my: y + Math.sin(a) * ry, x: ex + rx + 40, y: y - 26 + j * 13, side: "l", kind: "callout" });
      });
      if (parts.length > 5) S.texts.push({ x: ex + rx + 40, y: y - 26 + 5 * 13 + 3.5, text: `and ${parts.length - 5} more`, anchor: "start", kind: "mono", size: 10, quiet: true });
      S.regions.push({ key: keyOf(lv) + "/.x", node: lv, box: { x: ex - rx, y: y - ry, w: rx * 2, h: ry * 2 } });
    });
    // Ends outside the GCX: the Core Brain above the stack, the tentacles below it.
    const above: XY = { x: ex, y: y0 - ry - 40 };
    const nT = Math.max(1, tree.tentacles.length);
    const tentAngle = (t: number) => ((t + 0.5) / nT) * Math.PI * 2;
    const tentBelow = (id: string): XY | null => {
      const t = /#t(\d+)$/.exec(id);
      if (!t) return null;
      const a = tentAngle(Number(t[1]));
      return { x: ex + Math.cos(a) * rx * 0.66, y: yOf(n - 1) + ry + 40 };
    };
    // Each tentacle's g0_tx joins the g0 disc at its own place.
    for (const tn of tree.tentacles) {
      const g0n = tn.children.find((c) => c.key === "g0");
      const a = tentAngle(Number(tn.key.slice(1)));
      if (g0n?.anatomy) pos.set(g0n.anatomy.id, onDisc(n - 1, a, 0.66));
    }
    const brainNode = tree.byPath.get("core-brain");
    const podNode = tree.byPath.get("neuropod");
    let ups = 0;
    let downs = 0;
    for (const e of tree.edges) {
      const fi = focus.anatomyIds.has(e.from);
      const ti = focus.anatomyIds.has(e.to);
      // Edges between two tentacle parts stay inside the tentacle; the rest touch a level.
      if (!fi && !ti && !(pos.has(e.from) && pos.has(e.to))) continue;
      const end = (id: string, inside: boolean): XY | null => pos.get(id) ?? (inside ? null : brainNode?.anatomyIds.has(id) ? above : podNode?.anatomyIds.has(id) ? tentBelow(id) : null);
      const A = end(e.from, fi);
      const B = end(e.to, ti);
      if (!A || !B) continue;
      const down = e.mode.kind === "feedback" || B.y > A.y + 1;
      if (down) downs++;
      else ups++;
      const off = down ? 3 : -3;
      const pts = [{ x: A.x + off, y: A.y }, { x: B.x + off, y: B.y }];
      S.lines.push({ pts, a: 0.42, w: 0.7, hue: down ? "behav" : "sense", dash: e.mode.kind === "asynchronous" ? [1, 3] : e.mode.kind === "buffered" ? [4, 2] : undefined });
      const kf = tree.index.get(e.from)?.[0];
      const kt = tree.index.get(e.to)?.[0];
      if (kf && kt) S.routes.push({ from: keyOf(kf), to: keyOf(kt), pts, down });
    }
    S.texts.push({ x: above.x + 8, y: above.y - 4, text: "to the Core Brain", anchor: "start", kind: "mono", size: 10, quiet: true });
    S.texts.push({ x: ex + 8, y: yOf(n - 1) + ry + 56, text: "from and to the tentacles", anchor: "start", kind: "mono", size: 10, quiet: true });
    S.texts.push({ x: ex - rx, y: Math.min(H - 18, yOf(n - 1) + ry + 80), text: `Fig. 2 · exploded · ${ups} edges up, ${downs} down, from the anatomy`, anchor: "start", kind: "caps" });
  }
  S.mist.push({ x: cx, y: cy, r: Rmax * 1.7, a: 0.5 });
  S.mist.push({ x: W * 0.68, y: H / 2, r: W * 0.25, a: 0.25 });
  return S;
}

/** MicroModels: each model's matrix at full size, 28 layers by 16 heads, with its docks and the
 *  layer and head scales; beside each, a line chart of how many drone reader heads each layer
 *  has; under them, a nested-line chart of the models' sizes. */
function models(S: Scene, focus: TNode, carts: Cartridge[], status: Status | null, drones: Map<string, Set<string>>): Scene {
  const { W, H } = S;
  const list = focus.children;
  const n = list.length;
  const narrow = W < 640;
  const cols = narrow ? 2 : n;
  const rowsN = Math.ceil(n / cols);
  const mx = narrow ? 30 : 56;
  const chartW = narrow ? 0 : 34;
  const gapX = narrow ? 30 : 44;
  const availW = W - mx * 2 - gapX * (cols - 1) - chartW * cols;
  const availH = (H - (narrow ? 120 : 250)) / rowsN - 50;
  const cell = clamp(Math.min(availW / cols / HEADS_FALLBACK, availH / LAYERS_FALLBACK), 2, 26);
  // Rows may be taller than columns are wide, so the matrices use the height of the frame.
  const cellH = clamp(availH / LAYERS_FALLBACK, cell, cell * 1.7);
  const mW = cell * HEADS_FALLBACK;
  const mH = cellH * LAYERS_FALLBACK;
  const unitW = mW + chartW;
  const totalW = cols * unitW + (cols - 1) * gapX;
  const ox = (W - totalW) / 2;
  const blockH = rowsN * (mH + 80);
  const oy = Math.max(narrow ? 70 : 84, (H - blockH - (narrow ? 0 : 150)) / 2 + 40);
  const idOf = (m: TNode) => m.model?.id ?? (typeof m.anatomy?.config.card === "string" ? (m.anatomy.config.card as string) : m.label);
  list.forEach((m, i) => {
    const x0 = ox + (i % cols) * (unitW + gapX);
    const y0 = oy + Math.floor(i / cols) * (mH + 80);
    const id = idOf(m);
    for (let l = 0; l < LAYERS_FALLBACK; l++)
      for (let h = 0; h < HEADS_FALLBACK; h++) S.pts.push({ x: x0 + (h + 0.5) * cell, y: y0 + (LAYERS_FALLBACK - 1 - l + 0.5) * cellH, r: cell * 0.36, ex: cellH * 0.38, rot: Math.PI / 2, a: 0.3, k: keyOf(m), t: "m", model: id, layer: l, head: h, z: 0.1 });
    for (let l = 0; l < LAYERS_FALLBACK; l += 1) {
      const y = y0 + (LAYERS_FALLBACK - 1 - l + 0.5) * cellH;
      S.lines.push({ pts: [{ x: x0 - (l % 4 === 0 ? 9 : 5), y }, { x: x0 - 2, y }], a: l % 4 === 0 ? 0.55 : 0.25, w: 0.6 });
      if (l % 4 === 0 && (i % cols === 0 || cell > 8)) S.texts.push({ x: x0 - 12, y: y + 3.5, text: String(l).padStart(2, "0"), anchor: "end", kind: "num", size: 9.5, quiet: true });
    }
    for (let h = 0; h < HEADS_FALLBACK; h += 5) S.texts.push({ x: x0 + (h + 0.5) * cell, y: y0 + mH + 13, text: String(h), anchor: "middle", kind: "num", size: 9, quiet: true });
    // Reader heads per layer, as a line with round points on a faint grid.
    const set = drones.get(id);
    if (chartW && set) {
      const cx0 = x0 + mW + 12;
      const per = Array.from({ length: LAYERS_FALLBACK }, (_, l) => [...set].filter((lh) => lh.startsWith(`${l}:`)).length);
      const top = Math.max(1, ...per);
      const cw = chartW - 14;
      S.lines.push({ pts: [{ x: cx0, y: y0 }, { x: cx0, y: y0 + mH }], a: 0.22, w: 0.5 });
      S.lines.push({ pts: [{ x: cx0 + cw, y: y0 }, { x: cx0 + cw, y: y0 + mH }], a: 0.12, w: 0.5, dash: [1, 3] });
      const line = per.map((v, l) => ({ x: cx0 + (v / top) * cw, y: y0 + (LAYERS_FALLBACK - 1 - l + 0.5) * cellH }));
      S.lines.push({ pts: line, a: 0.75, w: 0.9, hue: "sense", k: keyOf(m) });
      for (const p of line) S.pts.push({ x: p.x, y: p.y, r: 1.3, a: 0.9, k: keyOf(m), t: "p", z: 0.05 });
      S.texts.push({ x: cx0, y: y0 + mH + 13, text: `0-${top}`, anchor: "start", kind: "num", size: 9, quiet: true });
      if (i === 0) S.texts.push({ x: cx0, y: y0 - 8, text: "readers", anchor: "start", kind: "mono", size: 9, quiet: true });
    }
    // Dock rail and docked cartridges.
    S.lines.push({ pts: [{ x: x0 + mW + 4, y: y0 }, { x: x0 + mW + 4, y: y0 + mH }], a: 0.25, w: 0.6, dash: [1, 2] });
    let k = 0;
    for (const c of carts) {
      if (!c.loaded || (c.model ?? c.base) !== id) continue;
      const [a, b] = c.layers ?? [0, LAYERS_FALLBACK - 1];
      S.docks.push({ cart: c.id, x: x0 + mW + 3 + k * 5, y: y0 + (LAYERS_FALLBACK - 1 - b) * cellH, w: 3, h: (b - a + 1) * cellH });
      k++;
    }
    const info = status?.models.find((x) => x.id === id);
    const gb = info ? `${(info.bytes / (1 << 30)).toFixed(2)} GB` : "";
    const prec = info ? (info.precision === "default" ? "default precision" : `${info.precision}-bit`) : "";
    S.annots.push({ key: keyOf(m), label: m.model ? m.label : `${m.label} · ${id}`, sub: [gb, prec].filter(Boolean).join(" · "), mx: x0, my: y0, x: x0, y: y0 - 30, w: mW, h: mH, side: "l", kind: "block" });
    S.regions.push({ key: keyOf(m), node: m, box: { x: x0, y: y0, w: mW, h: mH } });
  });
  S.mist.push({ x: W / 2, y: oy + mH / 2, r: totalW * 0.7, a: 0.4 });
  if (!narrow) {
    const below = oy + rowsN * (mH + 80) - 40;
    S.texts.push({ x: ox, y: below, text: "Fig. 1 · layers 00 to 27, bottom to top · heads 0 to 15 · brighter cells: drone reader heads", anchor: "start", kind: "caps" });
    // Nested-line chart: the models' sizes as nested rectangles, area by bytes.
    const sized = [...new Set(list.map(idOf))].map((id) => ({ id, b: status?.models.find((x) => x.id === id)?.bytes ?? 0 })).filter((x) => x.b > 0).sort((a, b) => b.b - a.b);
    const all = (status?.models ?? []).map((m) => ({ id: m.id, b: m.bytes })).sort((a, b) => b.b - a.b);
    const shown = all.length > sized.length ? all : sized;
    if (shown.length) {
      const maxB = shown[0].b;
      const side = Math.min(150, H - below - 40);
      if (side > 60) {
        const bx = ox;
        const by = below + 26;
        shown.forEach(({ id, b }, j) => {
          const s = side * Math.sqrt(b / maxB);
          S.lines.push({ pts: [{ x: bx, y: by + side - s }, { x: bx + s, y: by + side - s }, { x: bx + s, y: by + side }, { x: bx, y: by + side }], a: 0.6, closed: true, w: 0.7 });
          S.texts.push({ x: bx + side + 18, y: by + 10 + j * 15, text: `${id}  ${(b / (1 << 30)).toFixed(2)} GB`, anchor: "start", kind: "mono", size: 10, quiet: j > 0 });
          S.lines.push({ pts: [{ x: bx + s, y: by + side - s }, { x: bx + side + 12, y: by + 6 + j * 15 }], a: 0.25, w: 0.5 });
        });
        S.texts.push({ x: bx + side + 18, y: by + side, text: "Fig. 2 · models by size, area by bytes", anchor: "start", kind: "caps" });
      }
    }
  }
  return S;
}

function circle(cx: number, cy: number, r: number, n: number): XY[] {
  return Array.from({ length: n }, (_, i) => ({ x: cx + Math.cos((i / n) * Math.PI * 2) * r, y: cy + Math.sin((i / n) * Math.PI * 2) * r }));
}

export type { Pt };
