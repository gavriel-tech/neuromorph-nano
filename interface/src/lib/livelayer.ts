/** The live layer: a second canvas over the still picture that draws only activity, from real
 *  engine data. Parts light cold white while they compute, keep an afterglow in the colour of
 *  their stream (blue inward, violet outward), then fade back to silver. Signals travel along real
 *  connections at one speed. A cartridge travels from the rack to its dock.
 *
 *  It runs a frame only while something is changing; when nothing is, the loop stops and the
 *  canvas is cleared once. Idle cost is zero. */
import { COL, glow, spectral } from "./paint";
import { polyAt, polyLen, type Pt, type Route, type Scene, type XY } from "./scene";

type RGB = readonly [number, number, number];
type Lit = { start: number; hold: number; stream: "in" | "out"; why: string };
type Pulse = { pts: XY[]; len: number; start: number; dur: number; stream: "in" | "out" };
type Dock = { cart: string; from: XY; to: { x: number; y: number; w: number; h: number }; start: number; done: () => void };

const RISE = 80;
const TRACE = 900;
const FADE = 720;
export const SPEED = 600; // px per second, every connection
const SEAT = 600;
const SETTLE = 720;

/** Counters read by capture scripts to check that nothing draws while idle. */
type Counters = { base: number; frames: number; lastBaseMs: number };
declare global {
  interface Window {
    __nmScene?: Counters;
  }
}
export function counters(): Counters {
  if (typeof window === "undefined") return { base: 0, frames: 0, lastBaseMs: 0 };
  return (window.__nmScene ??= { base: 0, frames: 0, lastBaseMs: 0 });
}

const reduced = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export class LiveLayer {
  private ctx: CanvasRenderingContext2D;
  private byKey = new Map<string, number[]>();
  private centre = new Map<string, XY>();
  private lit = new Map<string, Lit>();
  private pulses: Pulse[] = [];
  private docks: Dock[] = [];
  private raf = 0;
  private dead = false;
  /** What the canvas last said it shows (`data-live-*`), so the attributes are written only on a change. */
  private told = "";

  constructor(
    private canvas: HTMLCanvasElement,
    private scene: Scene,
    private dpr: number,
  ) {
    this.ctx = canvas.getContext("2d")!;
    scene.pts.forEach((p, i) => {
      if (!p.k) return;
      // A point belongs to its own key and to every key above it.
      const parts = p.k.split("/");
      for (let n = parts.length; n >= 1; n--) {
        const k = parts.slice(0, n).join("/");
        let list = this.byKey.get(k);
        if (!list) this.byKey.set(k, (list = []));
        list.push(i);
      }
    });
    for (const [k, list] of this.byKey) {
      let x = 0;
      let y = 0;
      for (const i of list) {
        x += scene.pts[i].x;
        y += scene.pts[i].y;
      }
      this.centre.set(k, { x: x / list.length, y: y / list.length });
    }
    // A new layer starts at rest: nothing a layer before it said stays on the canvas.
    this.told = "?";
    this.tell(0);
  }

  /** Keys that have drawn points (a part and every part above it). */
  has(k: string): boolean {
    return this.byKey.has(k);
  }

  get routes(): Route[] {
    return this.scene.routes;
  }

  /** Lights a part: cold white while it computes, held until `settle` when `hold` is null. `why`
   *  names the real event behind the light (`stage:G1.dense`, `call:<id>:<part id>`); it is
   *  published on the canvas (`data-live-why`) so a check can match each light to the engine's
   *  record. */
  light(k: string, holdMs: number | null, why = "") {
    if (!this.byKey.has(k)) return;
    const now = performance.now();
    const prev = this.lit.get(k);
    const start = prev && now - prev.start < prev.hold ? prev.start : now;
    this.lit.set(k, { start, hold: holdMs === null ? Infinity : now - start + holdMs, stream: streamOf(k), why: why || prev?.why || "" });
    // Said at once, not at the next frame: the attributes are the state, whatever the frame rate.
    this.tell(now);
    this.kick();
  }

  /** The drawn state as attributes on the live canvas, written only when it changes:
   *  `data-live-lit` the keys computing now, `data-live-glow` the keys in their afterglow,
   *  `data-live-why` each of those keys with the event that lit it, `data-live-signals` the
   *  signals travelling. All empty or 0 at rest. */
  private tell(now: number) {
    const lit: string[] = [];
    const glow: string[] = [];
    const why: string[] = [];
    for (const [k, l] of this.lit) {
      (now - l.start < l.hold ? lit : glow).push(k);
      why.push(`${k}=${l.why}`);
    }
    const signals = this.pulses.filter((p) => now >= p.start).length;
    const sig = `${lit.join(" ")}|${glow.join(" ")}|${why.join(" ")}|${signals}`;
    if (sig === this.told) return;
    this.told = sig;
    const d = this.canvas.dataset;
    d.liveLit = lit.join(" ");
    d.liveGlow = glow.join(" ");
    d.liveWhy = why.join(" ");
    d.liveSignals = String(signals);
  }

  /** One part settles into its afterglow now (the engine no longer reports it running). */
  release(k: string) {
    const l = this.lit.get(k);
    if (!l) return;
    const now = performance.now();
    if (now - l.start < l.hold) l.hold = Math.max(RISE, now - l.start);
    this.kick();
    // Settled from the next instant on; the frame says so when it draws.
  }

  /** Every part held lit settles into its afterglow now. */
  releaseHeld() {
    const now = performance.now();
    for (const l of this.lit.values()) if (!Number.isFinite(l.hold)) l.hold = Math.max(RISE, now - l.start);
    this.kick();
  }

  /** A signal along a route, starting after `delay` ms. Returns its travel time. */
  pulse(r: Route, delay: number): number {
    const len = polyLen(r.pts);
    const dur = Math.max(160, (len / SPEED) * 1000);
    if (reduced() || len < 2) return reduced() ? 0 : dur;
    this.pulses.push({ pts: r.pts, len, start: performance.now() + delay, dur, stream: r.down ? "out" : "in" });
    this.kick();
    return dur;
  }

  /** A cartridge travels from the rack to its dock, then the docked rows settle. */
  dock(cart: string, done: () => void) {
    const d = this.scene.docks.find((x) => x.cart === cart);
    const from = this.scene.rack;
    if (!d || !from || reduced()) {
      done();
      return;
    }
    this.docks.push({ cart, from, to: d, start: performance.now(), done });
    this.kick();
  }

  destroy() {
    this.dead = true;
    cancelAnimationFrame(this.raf);
  }

  private kick() {
    if (this.raf || this.dead) return;
    this.raf = requestAnimationFrame(this.frame);
  }

  private frame = (now: number) => {
    this.raf = 0;
    if (this.dead) return;
    const { ctx, scene, dpr } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, scene.W, scene.H);
    let busy = false;
    ctx.globalCompositeOperation = "lighter";

    // Lit parts.
    const warm: XY[] = [];
    for (const [k, l] of this.lit) {
      const t = now - l.start;
      let on = 0;
      let trace = 0;
      if (t < l.hold) on = Math.min(1, t / RISE);
      else {
        const u = t - l.hold;
        if (u < TRACE) trace = 1 - (u / TRACE) * 0.45;
        else if (u < TRACE + FADE) trace = 0.55 * (1 - (u - TRACE) / FADE);
        else {
          this.lit.delete(k);
          continue;
        }
      }
      busy = true;
      const list = this.byKey.get(k) ?? [];
      const c: RGB = on > 0 ? COL.white : l.stream === "in" ? COL.sense : COL.behavHot;
      // The material is wide and faint: it lights at a lower level so the body stays the focus.
      const a = (on > 0 ? on : trace) * (k === "substrate" ? 0.35 : 1);
      this.drawPoints(list, c, a, on > 0 ? 1.6 : 1.3);
      if (k.startsWith("neuropod/") || k === "substrate") {
        const p = this.centre.get(k);
        if (p) warm.push(p);
      }
    }
    // A slightly warmer core where reading is happening.
    if (warm.length) {
      const s = glow([120, 80, 60]);
      ctx.globalAlpha = 0.18;
      for (const p of warm.slice(0, 8)) ctx.drawImage(s, p.x - 90, p.y - 90, 180, 180);
      ctx.globalAlpha = 1;
    }

    // Signals.
    this.pulses = this.pulses.filter((p) => {
      const t = now - p.start;
      if (t < 0) {
        busy = true;
        return true;
      }
      if (t > p.dur + 260) return false;
      busy = true;
      const head = Math.min(p.len, (t / p.dur) * p.len);
      const c: RGB = p.stream === "in" ? COL.sense : COL.behavHot;
      const sprite = glow(c);
      const core = glow(COL.white);
      const fadeOut = t > p.dur ? 1 - (t - p.dur) / 260 : 1;
      for (let i = 0; i < 9; i++) {
        const d = head - i * 7;
        if (d < 0) break;
        const q = polyAt(p.pts, d);
        const a = (1 - i / 9) * fadeOut;
        const s = 10 - i * 0.7;
        ctx.globalAlpha = a * 0.75;
        ctx.drawImage(sprite, q.x - s, q.y - s, s * 2, s * 2);
      }
      const q = polyAt(p.pts, head);
      ctx.globalAlpha = 0.9 * fadeOut;
      ctx.drawImage(core, q.x - 4, q.y - 4, 8, 8);
      ctx.globalAlpha = 1;
      return true;
    });

    // Cartridges seating.
    ctx.globalCompositeOperation = "source-over";
    this.docks = this.docks.filter((d) => {
      const t = now - d.start;
      if (t > SEAT + SETTLE) {
        d.done();
        return false;
      }
      busy = true;
      if (t <= SEAT) {
        const e = easeDock(t / SEAT);
        const x = d.from.x + (d.to.x - d.from.x) * e;
        const y = d.from.y - d.to.h / 2 + (d.to.y + d.to.h / 2 - d.from.y) * e;
        spectral(ctx, x, y, d.to.w + 1, d.to.h, 0.95);
      } else {
        spectral(ctx, d.to.x, d.to.y, d.to.w, d.to.h, 0.95);
        ctx.globalCompositeOperation = "lighter";
        const a = 1 - (t - SEAT) / SETTLE;
        const s = glow(COL.behav);
        ctx.globalAlpha = a * 0.6;
        ctx.drawImage(s, d.to.x - 14, d.to.y - 6, d.to.w + 28, d.to.h + 12);
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = "source-over";
      }
      return true;
    });
    ctx.globalCompositeOperation = "source-over";
    counters().frames++;
    this.tell(now);
    if (busy) this.kick();
    else ctx.clearRect(0, 0, scene.W, scene.H);
  };

  private drawPoints(list: number[], c: RGB, a: number, grow: number) {
    const { ctx, scene } = this;
    const pts = scene.pts;
    ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${(a * 0.95).toFixed(3)})`;
    ctx.beginPath();
    for (const i of list) {
      const p: Pt = pts[i];
      if (p.t === "g" && !p.k) continue;
      const r = p.r * grow;
      ctx.rect(p.x - r, p.y - r, r * 2, r * 2);
    }
    ctx.fill();
    const s = glow(c);
    ctx.globalAlpha = a * 0.22;
    const every = list.length > 1500 ? 6 : list.length > 400 ? 3 : 1;
    for (let j = 0; j < list.length; j += every) {
      const p = pts[list[j]];
      const r = 5 + p.r * 3;
      ctx.drawImage(s, p.x - r, p.y - r, r * 2, r * 2);
    }
    ctx.globalAlpha = 1;
  }
}

/** Inward parts carry the sensory stream; the brain, learning, memory and the runtime answer
 *  outward with behaviour. */
function streamOf(k: string): "in" | "out" {
  if (k.startsWith("neuropod") || k.startsWith("substrate")) return "in";
  if (k.startsWith("gcx") && !k.startsWith("gcx/g3")) return "in";
  return "out";
}

/** The dock curve: one small overshoot, the only bounce in the product. */
function easeDock(t: number): number {
  const c1 = 1.3;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}
