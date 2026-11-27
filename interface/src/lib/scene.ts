/** Scene geometry for the topology canvas. Pure functions: the same tree, focus and size give the
 *  same scene, so a camera move can find where a part sat in its parent's scene.
 *
 *  A scene is a field of points (every point stands for a real node of the tree, by its key), fine
 *  lines for chart plates, routes a signal can travel between keys, an annotation layer of markers
 *  and label chips (what is clicked), and regions (the boxes of the parts in view, for the camera).
 *
 *  The top level is the organism in plan: the Core Brain at the centre with its MicroModel
 *  matrices, the four GCX rings around it (G3 innermost, g0 outermost with one arc per tentacle),
 *  eight tentacles radiating evenly from the g0 ring to their drone tips, the material as the outer
 *  field, memory as a still pool below, the MCP port above and the cartridge rack at the side. */
import type { TNode } from "./topology";

export type XY = { x: number; y: number };
export type Box = { x: number; y: number; w: number; h: number };

/** Point tone: b body, m model head, g ground (material), w water (memory), r reflection,
 *  c cartridge slot, p port. */
export type Tone = "b" | "m" | "g" | "w" | "r" | "c" | "p";
/** A point of the drawing. `z` is its depth, 0 nearest to 1 deepest (default 0.35): nearer
 *  points are drawn sharp and bright, deeper ones soft and dim. A cell (`t` "m") may be a tile
 *  turned by `rot` with half-length `ex` along it, so a matrix can follow the body's curve.
 *  `cell` marks a point that carries its part's real activity as brightness (see activity.ts);
 *  `act` names the activity cell that drives it instead of its part (`tier:core`). */
export type Pt = { x: number; y: number; r: number; a: number; k: string; t: Tone; z?: number; ex?: number; rot?: number; model?: string; layer?: number; head?: number; cart?: string; cell?: boolean; act?: string };

/** A fine line of a chart plate or of the exploded view: a polyline, closed or not. */
export type Line = { pts: XY[]; a: number; w?: number; dash?: number[]; closed?: boolean; k?: string; hue?: "silver" | "sense" | "behav" };

/** An annotation: a square marker on the figure, a leader, and a chip with the plain name. */
export type Annot = { key: string; label: string; sub?: string; mx: number; my: number; x: number; y: number; side: "l" | "r" | "c"; kind: "chip" | "minor" | "callout" | "block" | "pad"; w?: number; h?: number; count?: number };

/** A caption or measure drawn as text on a plate. */
export type Text = { x: number; y: number; text: string; anchor: "start" | "middle" | "end"; kind: "caps" | "mono" | "num" | "sans"; size?: number; quiet?: boolean };

export type Route = { from: string; to: string; pts: XY[]; down?: boolean };
/** A real connection drawn as one fine curved line (anatomy edge), between two tree keys. */
export type Wire = { from: string; to: string; pts: XY[]; mode: string; down?: boolean };
export type Region = { key: string; node: TNode; box: Box };

export type Scene = {
  focus: TNode;
  W: number;
  H: number;
  kind: "body" | "plate";
  pts: Pt[];
  lines: Line[];
  annots: Annot[];
  texts: Text[];
  routes: Route[];
  wires: Wire[];
  regions: Region[];
  /** Mist and haze: soft light behind the figure. */
  mist: { x: number; y: number; r: number; a: number; warm?: boolean }[];
  /** Cartridge docks: where each docked cartridge's spectral bar sits. */
  docks: { cart: string; x: number; y: number; w: number; h: number }[];
  /** Where a cartridge leaves the rack when it docks. */
  rack?: XY;
  /** 0 names only, 1 names and roles, 2 everything. */
  detail: 0 | 1 | 2;
};

export const keyOf = (n: TNode) => n.path.join("/");
export const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

/** Qwen3 models of every size used here have 28 layers and 16 attention heads per layer; the
 *  engine may report `layers` (and later head counts) instead. Stated fallback. */
export const LAYERS_FALLBACK = 28;
export const HEADS_FALLBACK = 16;
export function layersOf(model: { layers?: number } | undefined): number {
  return model?.layers ?? LAYERS_FALLBACK;
}

/** A small deterministic hash in [0, 1), so the same scene always draws the same. */
export function hash01(a: number, b = 0): number {
  let h = Math.imul(a | 0, 374761393) ^ Math.imul(b | 0, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

export function emptyScene(focus: TNode, W: number, H: number, kind: Scene["kind"], detail: Scene["detail"]): Scene {
  return { focus, W, H, kind, pts: [], lines: [], annots: [], texts: [], routes: [], wires: [], regions: [], mist: [], docks: [], detail };
}

/** Spreads chips in one column so none overlaps the next; keeps the order of their wished y. */
export function spread(items: { y: number }[], gap: number, top: number, bottom: number) {
  items.sort((a, b) => a.y - b.y);
  for (let i = 1; i < items.length; i++) items[i].y = Math.max(items[i].y, items[i - 1].y + gap);
  const over = items.length ? items[items.length - 1].y - bottom : 0;
  if (over > 0) for (const it of items) it.y -= over;
  for (let i = 0; i < items.length; i++) items[i].y = Math.max(items[i].y, top + i * gap);
}

/** The region that holds a key, deepest first, for the camera. */
export function regionOf(scene: Scene, key: string): Region | null {
  let best: Region | null = null;
  for (const r of scene.regions) if ((key === r.key || key.startsWith(r.key + "/")) && (!best || r.key.length > best.key.length)) best = r;
  return best;
}

/** Length of a polyline. */
export function polyLen(pts: XY[]): number {
  let L = 0;
  for (let i = 1; i < pts.length; i++) L += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  return L;
}

/** The point at a distance along a polyline, and the direction there. */
export function polyAt(pts: XY[], d: number): XY {
  let left = d;
  for (let i = 1; i < pts.length; i++) {
    const l = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
    if (left <= l || i === pts.length - 1) {
      const u = l ? clamp(left / l, 0, 1) : 0;
      return { x: pts[i - 1].x + (pts[i].x - pts[i - 1].x) * u, y: pts[i - 1].y + (pts[i].y - pts[i - 1].y) * u };
    }
    left -= l;
  }
  return pts[pts.length - 1] ?? { x: 0, y: 0 };
}
