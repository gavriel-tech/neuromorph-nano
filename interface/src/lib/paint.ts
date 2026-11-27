/** Canvas painting of a scene. The still picture is painted once into its canvas and kept; it is
 *  painted again only when the scene, the data or the size changes. Bloom and mist come from
 *  small pre-rendered sprites and from the figure shrunk once per paint; no filter runs per
 *  frame.
 *
 *  At rest the body is near-monochrome: silver down to slate blue, shaded by point size and
 *  density, deep points soft and near points sharp. Colour arrives only with activity, drawn by
 *  the live layer on its own canvas. */
import { actFor, cellFactors, layerLevel, type ActView } from "./activity";
import type { Cartridge, Status } from "./api";
import type { Pt, Scene, XY } from "./scene";

export const COL = {
  night0: "#02030a",
  night1: "#070b24",
  silver: [201, 212, 234] as const,
  slate: [91, 108, 150] as const,
  sense: [15, 103, 245] as const,
  sensePale: [127, 176, 255] as const,
  behav: [143, 107, 255] as const,
  behavHot: [224, 92, 255] as const,
  white: [234, 242, 255] as const,
  red: [255, 59, 74] as const,
};
type RGB = readonly [number, number, number];
const rgba = (c: RGB, a: number) => `rgba(${c[0]},${c[1]},${c[2]},${a.toFixed(3)})`;
const mix = (a: RGB, b: RGB, t: number): RGB => [Math.round(a[0] + (b[0] - a[0]) * t), Math.round(a[1] + (b[1] - a[1]) * t), Math.round(a[2] + (b[2] - a[2]) * t)];

/* ---------------- Sprites, made once ---------------- */
const sprites = new Map<string, HTMLCanvasElement>();
/** A soft round glow of one colour, 64 px, drawn scaled. */
export function glow(c: RGB): HTMLCanvasElement {
  const id = c.join(",");
  let s = sprites.get(id);
  if (s) return s;
  s = document.createElement("canvas");
  s.width = s.height = 64;
  const g = s.getContext("2d")!;
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, rgba(c, 1));
  gr.addColorStop(0.18, rgba(c, 0.55));
  gr.addColorStop(0.5, rgba(c, 0.12));
  gr.addColorStop(1, rgba(c, 0));
  g.fillStyle = gr;
  g.fillRect(0, 0, 64, 64);
  sprites.set(id, s);
  return s;
}

let grainTile: HTMLCanvasElement | null = null;
/** A static grain tile, made once. */
function grain(): HTMLCanvasElement {
  if (grainTile) return grainTile;
  const c = document.createElement("canvas");
  c.width = c.height = 160;
  const g = c.getContext("2d")!;
  const img = g.createImageData(160, 160);
  let seed = 1234567;
  for (let i = 0; i < img.data.length; i += 4) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const v = seed % 255;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  grainTile = c;
  return c;
}

/** The fine spectral banding of a cartridge bar: the only spectral thing on screen. */
export function spectral(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, alpha: number) {
  const step = 1.25;
  for (let yy = 0, i = 0; yy < h; yy += step, i++) {
    const hue = (i * 23) % 300;
    ctx.fillStyle = `hsla(${hue}, 85%, 68%, ${alpha})`;
    ctx.fillRect(x, y + yy, w, Math.min(step * 0.7, h - yy));
  }
  ctx.strokeStyle = rgba(COL.silver, alpha * 0.55);
  ctx.lineWidth = 0.5;
  ctx.strokeRect(x - 0.5, y - 0.5, w + 1, h + 1);
}

export type PaintOpts = {
  status: Status | null;
  carts: Cartridge[];
  /** Drone reader heads per model id, as "layer:head". */
  drones: Map<string, Set<string>>;
  /** Cartridges still travelling to their dock: not drawn docked yet. */
  hide: Set<string>;
  /** During a drill-down: the part being opened. Its points come forward and everything in
   *  front of it thins out. */
  emph?: string;
  /** Real activity per part: the brightness of every cell (`Pt.cell`) and matrix row. */
  act?: ActView;
  /** What is left of the last call's light, 0 to 1. */
  fade?: number;
};

/** Brightness factor of every activity cell of a scene, by point index; 1 where none. */
function cellTexture(S: Scene, o: PaintOpts): Float32Array | null {
  const v = o.act;
  if (!v || !v.reported) return null;
  const count = new Map<string, number>();
  const group = (p: Pt) => p.act ?? p.k;
  for (const p of S.pts) if (p.cell && p.t !== "m") count.set(group(p), (count.get(group(p)) ?? 0) + 1);
  const levels = new Map<string, Float32Array>();
  for (const [k, n] of count) levels.set(k, cellFactors(v, k.includes(":") ? (v.cells.get(k) ?? null) : actFor(v, k), n, o.fade ?? 0, k));
  const seen = new Map<string, number>();
  const out = new Float32Array(S.pts.length).fill(1);
  S.pts.forEach((p, i) => {
    if (!p.cell || p.t === "m") return;
    const g = group(p);
    const j = seen.get(g) ?? 0;
    seen.set(g, j + 1);
    out[i] = levels.get(g)![j];
  });
  return out;
}

/** Rest brightness of one point (0 to 1), from the data where the engine reports it. */
function restAlpha(p: Pt, o: PaintOpts): number {
  if (p.t === "m") {
    const v = o.act;
    const lv = v && p.layer !== undefined ? layerLevel(v, actFor(v, p.k), p.layer, o.fade ?? 0, p.k) * 0.24 : 0;
    const rest = matrixRest(p, o);
    // Work brightens a row but stays below the drone reader heads, which keep the full light.
    return rest >= 1 ? 1 : Math.min(0.5, rest + lv);
  }
  return p.a;
}

/** A MicroModel cell at rest: drone reader heads bright, held models dim, others darker. */
function matrixRest(p: Pt, o: PaintOpts): number {
  const set = p.model ? o.drones.get(p.model) : undefined;
  const reader = !!set && set.has(`${p.layer}:${p.head}`);
  const loaded = o.status?.loaded;
  if (loaded) {
    const held = loaded.some((m) => m.id === p.model);
    if (!held) return 0.12;
    return reader ? 1 : 0.14 + p.a * 0.45;
  }
  // The engine does not report what is held: dim and still, readers marked by brightness.
  return reader ? 1 : 0.1 + p.a * 0.45;
}

function colourOf(p: Pt, a: number): RGB {
  if (p.t === "w") return mix(COL.slate, COL.sensePale, 0.3);
  if (p.t === "r") return mix(COL.slate, COL.sensePale, 0.45);
  if (p.t === "g") return mix(COL.slate, COL.silver, 0.3 + a * 0.5);
  if (p.t === "m") return a > 0.95 ? COL.white : mix(COL.slate, COL.silver, Math.min(1, 0.15 + a));
  return mix(COL.slate, COL.silver, Math.min(1, 0.22 + a * 0.85));
}

function layerCanvas(W: number, H: number, dpr: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(W * dpr));
  c.height = Math.max(1, Math.round(H * dpr));
  const g = c.getContext("2d")!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [c, g];
}

/** Halves a canvas a number of times; each halving averages four pixels, so small points keep
 *  their light. Used once per paint to make the bloom; never per frame. */
function shrink(src: HTMLCanvasElement, times: number): HTMLCanvasElement[] {
  const out: HTMLCanvasElement[] = [];
  let cur = src;
  for (let i = 0; i < times; i++) {
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.floor(cur.width / 2));
    c.height = Math.max(1, Math.floor(cur.height / 2));
    const g = c.getContext("2d")!;
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = "high";
    g.drawImage(cur, 0, 0, c.width, c.height);
    out.push(c);
    cur = c;
  }
  return out;
}

/** Paints the still picture of a scene into a context already scaled to CSS pixels. The figure
 *  is drawn back to front on its own layer (deep points soft, near points sharp), then laid on
 *  the night with its bloom, mist and grain. */
export function paintBase(ctx: CanvasRenderingContext2D, S: Scene, o: PaintOpts) {
  const { W, H } = S;
  const dpr = ctx.getTransform().a || 1;
  ctx.clearRect(0, 0, W, H);
  // Night: deep ultramarine behind the figure, near black at the edges.
  const bg = ctx.createRadialGradient(W / 2, H * 0.48, 0, W / 2, H * 0.48, Math.hypot(W, H) * 0.62);
  bg.addColorStop(0, "#0a1034");
  bg.addColorStop(0.32, COL.night1);
  bg.addColorStop(0.72, "#04061a");
  bg.addColorStop(1, COL.night0);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Mist: soft cool light behind the figure, from one sprite scaled.
  ctx.globalCompositeOperation = "lighter";
  const mistSprite = glow([44, 64, 138]);
  for (const m of S.mist) {
    ctx.globalAlpha = Math.min(1, m.a * 0.8);
    ctx.drawImage(mistSprite, m.x - m.r, m.y - m.r, m.r * 2, m.r * 2);
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";

  const [layer, g] = layerCanvas(W, H, dpr);
  const emph = o.emph;
  const thin = (k: string | undefined) => (emph === undefined ? 1 : k && (k === emph || k.startsWith(emph + "/")) ? 1.4 : 0.2);

  // Wires: one fine line per real connection; their light adds where they gather.
  g.globalCompositeOperation = "lighter";
  g.lineWidth = 0.7;
  for (const w of S.wires) {
    g.strokeStyle = rgba(COL.silver, Math.min(1, (w.mode === "asynchronous" ? 0.14 : 0.3) * thin(w.from)));
    g.setLineDash(w.mode === "conditional" ? [3, 2] : []);
    strokePoly(g, w.pts, false);
  }
  g.setLineDash([]);

  // Plate lines.
  for (const l of S.lines) {
    const c = l.hue === "sense" ? COL.sensePale : l.hue === "behav" ? COL.behav : COL.silver;
    g.strokeStyle = rgba(c, Math.min(1, l.a * thin(l.k)));
    g.lineWidth = l.w ?? 0.75;
    g.setLineDash(l.dash ?? []);
    strokePoly(g, l.pts, !!l.closed);
  }
  g.setLineDash([]);

  // Deep points first, soft: each a small glow sprite, dimmer with depth.
  const tex = cellTexture(S, o);
  const ta = (i: number) => (tex ? tex[i] : 1);
  const soft = glow(COL.silver);
  const softBlue = glow(mix(COL.slate, COL.sensePale, 0.5));
  S.pts.forEach((p, i) => {
    const z = p.z ?? 0.35;
    if (z < 0.7 || (p.t === "c" && p.cart)) return;
    const a = restAlpha(p, o) * ta(i) * thin(p.k) * (1.25 - z * 0.6);
    const s = p.r * (2.6 + z * 3.2);
    g.globalAlpha = Math.min(1, a * 0.55);
    g.drawImage(p.t === "w" || p.t === "r" ? softBlue : soft, p.x - s, p.y - s, s * 2, s * 2);
  });
  g.globalAlpha = 1;

  // Nearer points, sharp, batched by colour and brightness; light adds where parts overlap.
  const buckets = new Map<string, { round: Pt[]; square: Pt[] }>();
  const grow = new Map<Pt, number>();
  const bright: Pt[] = [];
  S.pts.forEach((p, i) => {
    if (p.t === "c" && p.cart) return;
    const z = p.z ?? 0.35;
    if (z >= 0.7) return;
    const a = Math.min(1, restAlpha(p, o) * ta(i) * thin(p.k) * (1.12 - z * 0.35));
    const lvl = Math.max(1, Math.min(24, Math.round(a * 24)));
    const c = colourOf(p, lvl / 24);
    const key = `${c.join(",")},${lvl}`;
    let b = buckets.get(key);
    if (!b) buckets.set(key, (b = { round: [], square: [] }));
    (p.t === "m" ? b.square : b.round).push(p);
    // A cell lit by real work also grows a little, so a busy slice reads as a brighter grain.
    if (tex && tex[i] > 1.2) grow.set(p, 1 + Math.min(0.9, (tex[i] - 1) * 0.45));
    if (a > 0.78 && p.r > 0.6) bright.push(p);
  });
  for (const [key, b] of buckets) {
    const parts = key.split(",").map(Number);
    g.fillStyle = rgba([parts[0], parts[1], parts[2]], parts[3] / 24);
    if (b.round.length) {
      g.beginPath();
      for (const p of b.round) {
        const r = p.r * (grow.get(p) ?? 1);
        if (r < 0.75) g.rect(p.x - r, p.y - r, r * 2, r * 2);
        else {
          g.moveTo(p.x + r, p.y);
          g.arc(p.x, p.y, r, 0, Math.PI * 2);
        }
      }
      g.fill();
    }
    if (b.square.length) {
      g.beginPath();
      for (const p of b.square) {
        if (p.ex === undefined || p.rot === undefined) {
          g.rect(p.x - p.r, p.y - p.r, p.r * 2, p.r * 2);
          continue;
        }
        const tx = Math.cos(p.rot);
        const ty = Math.sin(p.rot);
        g.moveTo(p.x + tx * p.ex - ty * p.r, p.y + ty * p.ex + tx * p.r);
        g.lineTo(p.x - tx * p.ex - ty * p.r, p.y - ty * p.ex + tx * p.r);
        g.lineTo(p.x - tx * p.ex + ty * p.r, p.y - ty * p.ex - tx * p.r);
        g.lineTo(p.x + tx * p.ex + ty * p.r, p.y + ty * p.ex - tx * p.r);
        g.closePath();
      }
      g.fill();
    }
  }
  // Slight softness on the brightest points.
  g.globalAlpha = 0.16;
  for (let i = 0; i < bright.length; i += 2) {
    const p = bright[i];
    const s = p.r * 6;
    g.drawImage(soft, p.x - s, p.y - s, s * 2, s * 2);
  }
  g.globalAlpha = 1;
  g.globalCompositeOperation = "source-over";

  // The figure on the night, then its bloom: the same layer shrunk and laid back over it, close,
  // middle and wide, so light falls off around every bright mark as it would through mist.
  ctx.globalCompositeOperation = "lighter";
  ctx.drawImage(layer, 0, 0, W, H);
  const levels = shrink(layer, 5);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  const bloom: [number, number][] = [
    [1, 0.4],
    [3, 0.55],
    [4, 0.75],
  ];
  for (const [i, a] of bloom) {
    const c = levels[i];
    if (!c) continue;
    ctx.globalAlpha = a;
    ctx.drawImage(c, 0, 0, W, H);
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";

  // Cartridges: spectral bars in the rack and on the rows they are docked on.
  const rackCarts = new Map<string, Pt[]>();
  for (const p of S.pts) if (p.t === "c" && p.cart) rackCarts.set(p.cart, [...(rackCarts.get(p.cart) ?? []), p]);
  for (const [id, ps] of rackCarts) {
    const c = o.carts.find((x) => x.id === id);
    const y0 = Math.min(...ps.map((p) => p.y));
    const y1 = Math.max(...ps.map((p) => p.y));
    spectral(ctx, ps[0].x - 1.2, y0, 2.4, y1 - y0, c?.loaded && !o.hide.has(id) ? 0.35 : 0.85);
  }
  for (const d of S.docks) if (!o.hide.has(d.cart)) spectral(ctx, d.x, d.y, d.w, d.h, 0.9);

  // Grain and the falling off of light at the frame's edge.
  ctx.globalAlpha = 0.055;
  ctx.globalCompositeOperation = "overlay";
  const pat = ctx.createPattern(grain(), "repeat");
  if (pat) {
    ctx.fillStyle = pat;
    ctx.fillRect(0, 0, W, H);
  }
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  const v = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.5, W / 2, H / 2, Math.hypot(W, H) * 0.64);
  v.addColorStop(0, "rgba(2,3,10,0)");
  v.addColorStop(1, "rgba(2,3,10,0.55)");
  ctx.fillStyle = v;
  ctx.fillRect(0, 0, W, H);
}

export function strokePoly(ctx: CanvasRenderingContext2D, pts: XY[], closed: boolean) {
  if (pts.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  if (closed) ctx.closePath();
  ctx.stroke();
}

/** Drone reader heads from the drones design (`designs/drones.json`), per model population. */
export function droneSets(settings: Record<string, unknown> | null): Map<string, Set<string>> {
  const out = new Map<string, Set<string>>();
  const pops = (settings?.drones as { populations?: Record<string, { default?: unknown }> } | undefined)?.populations;
  if (!pops) return out;
  for (const [model, pop] of Object.entries(pops)) {
    const list = Array.isArray(pop?.default) ? pop.default : [];
    const set = new Set<string>();
    for (const lh of list) if (Array.isArray(lh) && lh.length === 2) set.add(`${lh[0]}:${lh[1]}`);
    out.set(model, set);
  }
  return out;
}
