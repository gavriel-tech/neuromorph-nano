"use client";

// The whole-set parameter viewer of one object opened in the editor. One tensor at a time is the
// primary view: a large heatmap of cell summaries (min, max, mean or RMS of the weights a cell
// covers) with row and column rulers in weight indices, a vertical colour scale with its ticks, the
// view's true minimum and maximum and the zero mark, on a perceptual diverging scale around 0 or a
// sequential scale of magnitude. The wheel zooms at the cursor, a drag pans, shift and a drag zooms
// into a box (and selects it, as the drag did before), a double click goes back to the whole tensor.
// Down at one weight per cell the cells are exact values; from 6 px a cell its edges are drawn, from
// 14 px its value is printed in it. A click inspects: the inspector beside the view gives the
// weight's value, place, percentile in the tensor (with the tensor's distribution) and the related
// parameter maps' values there, and selects the weight, its row, column or head for the controls.
// Over a cell of many weights it gives their min, max, mean, RMS and count. Above the view, the
// tensor's statistics (shape, type, min, max, mean, std, share of exact zeros, L2) and a slim
// distribution. Every other tensor of the set is a compact thumbnail, grouped by layer and block; a
// click makes it the primary view.
//
// What the overlay holds (edits, masks, freezes) is drawn on the primary view as read-only marks:
// masked dimmed, frozen a light diagonal hatch, edited a thin outline. Nothing on the viewer adjusts
// them; the controls under it do (ParamFocus).
//
// Reads (owner's rule: nothing is read until the object is opened): the set when this is mounted
// with an opened object's key; a thumbnail when it scrolls into view; the primary view's visible
// region only, at the canvas resolution, debounced, the stale read aborted; at most PARALLEL reads
// at once; nothing polls. GET /api/parameters/overview (crates/nmcore/src/bin/poverview.rs; `hist=1`
// adds the region's distribution, folded in the same streamed pass) and, for an inspected weight,
// GET /api/cartridges/parameters?key&path (its maps' values there).
//
// Props:
// - objectKey: the opened object (a cartridge id@version, filter:<role>/<job>, an adapter <id>@<n>,
//   a learning unit id, or open:<model> for the upstream open weights).
// - onSelect (optional): called with the editor's path of a region the user boxes or picks
//   (`name[r0:r1, c0:c1]` for a matrix, `name[c0:c1]` for a vector, the flat range of whole rows
//   for three or more dimensions), as crates/nmcore/src/bin/pedit.rs parse_path reads it.
// - marks (optional): the overlay's entries, drawn as marks.
// - maps (optional): the related parameter maps (per-layer values), for the inspector.
// - target (optional): a path to show; each new `n` makes its tensor primary and zooms to it.

import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { createPortal } from "react-dom";
import { apiBase, post, refreshAll } from "@/lib/api";
import { diverging, indexOf, niceStep, overlaps, parseHex, percentileOf, rectsOf, sequential, shortFmt, ticks, toLab, type Palette, type Rect, type Rgb } from "@/lib/paramgeom";

/** One tensor of the set, as the set-level read returns it (no values). */
export type OverviewTensor = { name: string; layer: number | null; part: string; shape: number[]; dtype: string; rows: number; cols: number; size: number; readable: boolean; unread: string | null };
type SetReply = { error?: string; format?: string; key?: string; model?: string; kind?: string; name?: string; count?: number; weights?: number; tensors?: OverviewTensor[] };
type Cells = (number | null)[];
/** A region's distribution (`hist=1`): linear bins over `lo..hi`, the value at each percentile. */
export type Hist = { bins: number[]; lo: number | null; hi: number | null; q: (number | null)[]; zeros: number; count: number; std: number | null; l2: number | null; precision: string };
type Summary = { min: number | null; max: number | null; mean: number | null; rms: number | null; count: number; nonfinite: number };
/** One tensor's region as a grid of cells. `values` when every cell is one weight (`exact`). */
export type OverviewGrid = {
  error?: string;
  tensor: string;
  shape: number[];
  dtype: string;
  rows: number;
  cols: number;
  region: Rect;
  path: string;
  w: number;
  h: number;
  cell: { rows: number; cols: number; weights: number };
  exact: boolean;
  summary: Summary;
  read: { bytes: number; chunks: number; largest_chunk: number; chunk_cap: number; ms: number };
  hist?: Hist;
  values?: Cells;
  min?: Cells;
  max?: Cells;
  mean?: Cells;
  rms?: Cells;
};
export type Stat = "mean" | "min" | "max" | "rms";
type Scale = "diverging" | "sequential";
/** An overlay entry, as the viewer draws it. */
export type OverviewMark = { id: string; action: "edit" | "mask" | "freeze"; op: string | null; whole: boolean; regions: { path: string; tensor: string }[] };
/** A related parameter map and its per-layer values. */
export type OverviewMap = { file: string; layers: { layer: number; value: number | null }[] };
/** An active parameter map overlaid directly onto the Parameter Viewer (e.g. super-weights coordinates, job filter layers, or profile map coverage). */
export type ActiveMapOverlay = {
  id: string;
  file: string;
  role: string;
  kind: string;
  about?: string;
  pointsByTensor: Record<string, [number, number][]>;
  layers: number[];
  layerValues: Record<number, number>;
  targets: string[];
};
type Region = Rect;

const STATS: Stat[] = ["mean", "min", "max", "rms"];
const ROUTE = "/api/parameters/overview";
const CELLS_MAX = 65_536;
const SIDE_MAX = 1024;
/** A thumbnail's width in CSS pixels. */
const THUMB = 150;
/** Values per page of the value list. */
const PAGE = 256;
/** Overview reads in flight at once. */
const PARALLEL = 3;
/** Ruler and legend sizes (CSS pixels). */
const RULER_L = 46;
const RULER_T = 18;
const LEGEND = 104;
const LEGEND_GAP = 8;
/** Floating inspector window bounds (CSS pixels). */
type WinRect = { x: number; y: number; w: number; h: number };
const WIN_MIN_W = 290;
const WIN_MIN_H = 240;
const WIN_DEF_W = 360;
const WIN_DEF_H = 480;
const WIN_PAD = 12;

function clampWin(r: WinRect, minimized = false): WinRect {
  const vw = typeof window === "undefined" ? 1280 : window.innerWidth;
  const vh = typeof window === "undefined" ? 800 : window.innerHeight;
  const maxW = Math.max(WIN_MIN_W, Math.min(680, vw - WIN_PAD * 2));
  const maxH = Math.max(WIN_MIN_H, vh - WIN_PAD * 2);
  const w = Math.max(WIN_MIN_W, Math.min(maxW, Math.round(r.w)));
  const h = Math.max(WIN_MIN_H, Math.min(maxH, Math.round(r.h)));
  const effH = minimized ? 36 : h;
  const x = Math.max(WIN_PAD, Math.min(vw - w - WIN_PAD, Math.round(r.x)));
  const y = Math.max(WIN_PAD, Math.min(vh - effH - WIN_PAD, Math.round(r.y)));
  return { x, y, w, h };
}

function defaultWin(): WinRect {
  const vw = typeof window === "undefined" ? 1280 : window.innerWidth;
  const vh = typeof window === "undefined" ? 800 : window.innerHeight;
  return clampWin({
    w: WIN_DEF_W,
    h: Math.min(WIN_DEF_H, Math.max(WIN_MIN_H, vh - 120)),
    x: vw - WIN_DEF_W - 24,
    y: Math.max(72, Math.min(116, vh - WIN_DEF_H - 24)),
  });
}
/** A cell is drawn at most this wide when zoomed in all the way. */
const CELL_PX_MAX = 96;
/** From this cell size its edges are drawn; from the next its value is printed in it. */
const EDGE_PX = 6;
const TEXT_PX = 14;
/** The pause after the last zoom or pan before the view is read. */
const DEBOUNCE_MS = 70;

const enc = encodeURIComponent;
const btn = "press mono border-b border-line-strong px-1 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const on = "press mono border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.1em] text-text";

/* ---------- reads ---------- */

let running = 0;
const waiting: (() => void)[] = [];
/** Runs at most PARALLEL reads at once; a finished read hands its slot to the next. */
async function limited<T>(run: () => Promise<T>): Promise<T> {
  if (running >= PARALLEL) await new Promise<void>((go) => waiting.push(go));
  else running += 1;
  try {
    return await run();
  } finally {
    const next = waiting.shift();
    if (next) next();
    else running -= 1;
  }
}

async function getJson<T>(path: string, signal: AbortSignal): Promise<T> {
  return limited(async () => {
    // A read aborted while it waited for a slot is not sent.
    if (signal.aborted) throw new DOMException("aborted", "AbortError");
    const r = await fetch(apiBase() + path, { cache: "no-store", signal });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return (await r.json()) as T;
  });
}

/** One read per path, when the path is given, after `delay` ms (a later path aborts it); no cache and
 *  no polling. The last answer stays while the next is read (`stale`). */
function useRead<T>(path: string | null, delay = 0): { data: T | null; error: string | null; stale: boolean } {
  const [got, setGot] = useState<{ path: string; data: T | null; error: string | null } | null>(null);
  useEffect(() => {
    if (path === null) return;
    const ac = new AbortController();
    const go = () =>
      getJson<T>(path, ac.signal).then(
        (data) => {
          if (!ac.signal.aborted) setGot({ path, data, error: null });
        },
        (e: unknown) => {
          if (!ac.signal.aborted) setGot({ path, data: null, error: e instanceof Error ? e.message : String(e) });
        },
      );
    const timer = delay > 0 ? setTimeout(go, delay) : null;
    if (timer === null) void go();
    return () => {
      if (timer !== null) clearTimeout(timer);
      ac.abort();
    };
  }, [path, delay]);
  if (!got || path === null) return { data: null, error: null, stale: false };
  if (got.path !== path) return { data: got.data, error: null, stale: true };
  return { data: got.data, error: got.error, stale: false };
}

/* ---------- geometry ---------- */

function fmt(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "-";
  const a = Math.abs(x);
  if (a === 0) return "0";
  return a >= 1000 || a < 0.001 ? x.toExponential(2) : String(Number(x.toPrecision(3)));
}

function wholeOf(t: OverviewTensor): Region {
  return { r0: 0, r1: t.rows, c0: 0, c1: t.cols };
}

/** The editor's path of a region, as the engine writes it (`poverview::path_of`). */
export function pathOf(t: OverviewTensor, g: Region): string {
  if (t.shape.length <= 1) return `${t.name}[${g.c0}:${g.c1}]`;
  if (t.shape.length === 2) return `${t.name}[${g.r0}:${g.r1}, ${g.c0}:${g.c1}]`;
  return `${t.name}[${g.r0 * t.cols}:${g.r1 * t.cols}]`;
}

/** Cells for a region drawn `px` wide and high: at most one per pixel, per weight, SIDE_MAX a side
 *  and CELLS_MAX in all. */
function cellsOf(g: Region, px: { w: number; h: number }): { w: number; h: number } {
  let w = Math.max(1, Math.min(g.c1 - g.c0, Math.ceil(px.w), SIDE_MAX));
  let h = Math.max(1, Math.min(g.r1 - g.r0, Math.ceil(px.h), SIDE_MAX));
  if (w * h > CELLS_MAX) {
    const k = Math.sqrt(CELLS_MAX / (w * h));
    w = Math.max(1, Math.floor(w * k));
    h = Math.max(1, Math.floor(h * k));
  }
  return { w, h };
}

/** The weights cells `a..b` (inclusive) of `cells` cover along a side of `len` from `from`. */
function span(a: number, b: number, len: number, cells: number, from: number): [number, number] {
  return [from + Math.floor((a * len) / cells), from + Math.floor(((b + 1) * len) / cells)];
}

/** The cell of a grid that holds weight (r, c), or null outside it. */
function cellOf(g: OverviewGrid, r: number, c: number): { i: number; j: number } | null {
  const R = g.region;
  if (r < R.r0 || r >= R.r1 || c < R.c0 || c >= R.c1) return null;
  const i = Math.min(g.h - 1, Math.floor(((r - R.r0 + 1) * g.h - 1) / (R.r1 - R.r0)));
  const j = Math.min(g.w - 1, Math.floor(((c - R.c0 + 1) * g.w - 1) / (R.c1 - R.c0)));
  return { i, j };
}

function same(a: Region, b: Region): boolean {
  return a.r0 === b.r0 && a.r1 === b.r1 && a.c0 === b.c0 && a.c1 === b.c1;
}

function label(g: Region, t: OverviewTensor): string {
  if (t.rows === 1) return `cols ${g.c0}:${g.c1}`;
  return `rows ${g.r0}:${g.r1}, cols ${g.c0}:${g.c1}`;
}

/** The cells the chosen statistic colours (an exact grid: the values, RMS as magnitude). */
function cellValues(g: OverviewGrid, stat: Stat): Cells {
  if (g.exact) {
    const v = g.values ?? [];
    return stat === "rms" ? v.map((x) => (x === null ? null : Math.abs(x))) : v;
  }
  return g[stat] ?? [];
}

/** The colour scale: the 99th percentile of the cells' magnitude (beyond it, clipped). */
function scaleOf(xs: Cells): number {
  const m = xs.filter((x): x is number => x !== null && Number.isFinite(x)).map(Math.abs);
  if (m.length === 0) return 0;
  m.sort((a, b) => a - b);
  return m[Math.min(m.length - 1, Math.floor(0.99 * (m.length - 1)))] || m[m.length - 1] || 0;
}

/** The view (fractional weight bounds) kept inside the tensor, at least a few weights across. */
function clampView(v: Region, t: OverviewTensor, cw: number, ch: number): Region {
  const minC = Math.min(t.cols, Math.max(1, cw / CELL_PX_MAX));
  const minR = Math.min(t.rows, Math.max(1, ch / CELL_PX_MAX));
  const w = Math.min(t.cols, Math.max(minC, v.c1 - v.c0));
  const h = t.rows === 1 ? 1 : Math.min(t.rows, Math.max(minR, v.r1 - v.r0));
  const cx = (v.c0 + v.c1) / 2;
  const cy = (v.r0 + v.r1) / 2;
  const c0 = Math.max(0, Math.min(t.cols - w, cx - w / 2));
  const r0 = t.rows === 1 ? 0 : Math.max(0, Math.min(t.rows - h, cy - h / 2));
  return { r0, r1: r0 + h, c0, c1: c0 + w };
}

/** The whole weights a view touches. */
function intOf(v: Region, t: OverviewTensor): Region {
  const r0 = Math.max(0, Math.floor(v.r0 + 1e-9));
  const c0 = Math.max(0, Math.floor(v.c0 + 1e-9));
  return { r0, r1: Math.min(t.rows, Math.max(r0 + 1, Math.ceil(v.r1 - 1e-9))), c0, c1: Math.min(t.cols, Math.max(c0 + 1, Math.ceil(v.c1 - 1e-9))) };
}

/** A region grown to at least 32 weights a side around its centre, to see it in its place. */
function around(r: Region, t: OverviewTensor, cw: number, ch: number): Region {
  const w = Math.max(r.c1 - r.c0, Math.min(t.cols, 32));
  const h = t.rows === 1 ? 1 : Math.max(r.r1 - r.r0, Math.min(t.rows, Math.round((32 * ch) / Math.max(1, cw)) || 32));
  const cx = (r.c0 + r.c1) / 2;
  const cy = (r.r0 + r.r1) / 2;
  return clampView({ r0: cy - h / 2, r1: cy + h / 2, c0: cx - w / 2, c1: cx + w / 2 }, t, cw, ch);
}

/** The value list's pages over a region: row bands of up to PAGE values, or PAGE columns of a row. */
function pagesOf(g: Region): { count: number; at: (p: number) => Region } {
  const R = g.r1 - g.r0;
  const C = g.c1 - g.c0;
  if (C >= PAGE) {
    const per = Math.ceil(C / PAGE);
    return {
      count: R * per,
      at: (p) => {
        const r = g.r0 + Math.floor(p / per);
        const c = g.c0 + (p % per) * PAGE;
        return { r0: r, r1: r + 1, c0: c, c1: Math.min(g.c1, c + PAGE) };
      },
    };
  }
  const rowsPer = Math.max(1, Math.floor(PAGE / C));
  return {
    count: Math.ceil(R / rowsPer),
    at: (p) => {
      const r = g.r0 + p * rowsPer;
      return { r0: r, r1: Math.min(g.r1, r + rowsPer), c0: g.c0, c1: g.c1 };
    },
  };
}

function gridPath(k: string, t: OverviewTensor, g: Region, cells: { w: number; h: number }, hist = false): string {
  return `${ROUTE}?key=${enc(k)}&tensor=${enc(t.name)}&w=${cells.w}&h=${cells.h}&r0=${g.r0}&r1=${g.r1}&c0=${g.c0}&c1=${g.c1}${hist ? "&hist=1" : ""}`;
}

/* ---------- colour ---------- */

/** The theme's colours for the canvas (app/globals.css), read once on the client. */
type Theme = { pal: Palette; bg0: Rgb; text1: Rgb; text3: Rgb; signal: Rgb; verified: Rgb; line2: Rgb; mono: string };
const FALLBACK: Record<string, string> = { "--bg-0": "#03040e", "--bg-3": "#131a44", "--sense": "#0f67f5", "--sense-pale": "#7fb0ff", "--behav-hot": "#e05cff", "--learned": "#c58bff", "--signal": "#eaf2ff", "--text-1": "#eef3ff", "--text-3": "#9fb0dd", "--verified": "#7dffa0", "--line-2": "#253170" };

function readTheme(): Theme {
  const cs = typeof document === "undefined" ? null : getComputedStyle(document.documentElement);
  const c = (v: string): Rgb => parseHex(cs?.getPropertyValue(v) ?? "") ?? parseHex(FALLBACK[v]) ?? [0, 0, 0];
  const mono = (cs?.getPropertyValue("--font-geist-mono") ?? "").trim();
  return {
    pal: { zero: c("--bg-3"), neg: c("--sense"), negHi: c("--sense-pale"), pos: c("--behav-hot"), posHi: c("--learned"), top: c("--signal") },
    bg0: c("--bg-0"),
    text1: c("--text-1"),
    text3: c("--text-3"),
    signal: c("--signal"),
    verified: c("--verified"),
    line2: c("--line-2"),
    mono: mono ? `${mono}, ui-monospace, monospace` : "ui-monospace, monospace",
  };
}

const rgba = (c: Rgb, a: number) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

/** The colour tables of both scales. */
function useLuts(theme: Theme) {
  return useMemo(() => ({ diverging: diverging(theme.pal), sequential: sequential(theme.pal) }), [theme]);
}

/** A grid drawn one pixel per cell, for scaling onto a canvas. */
function paint(g: OverviewGrid, vals: Cells, lut: Rgb[], s: number, kind: Scale): HTMLCanvasElement | null {
  if (typeof document === "undefined") return null;
  const off = document.createElement("canvas");
  off.width = g.w;
  off.height = g.h;
  const ctx = off.getContext("2d");
  if (!ctx) return null;
  const img = ctx.createImageData(g.w, g.h);
  for (let n = 0; n < g.w * g.h; n++) {
    const x = vals[n];
    if (x === null || x === undefined || !Number.isFinite(x)) continue;
    const col = lut[indexOf(x, s, lut.length, kind)];
    img.data[n * 4] = col[0];
    img.data[n * 4 + 1] = col[1];
    img.data[n * 4 + 2] = col[2];
    img.data[n * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return off;
}

/* ---------- small parts ---------- */

/** A distribution's bins (square-root counts) over `lo..hi`, with the zero mark and a marked value. */
function HistBars({ h, mark, height, e2e }: { h: Hist; mark?: number | null; height: number; e2e?: string }) {
  const n = h.bins.length;
  const max = Math.max(1, ...h.bins.map((c) => Math.sqrt(c)));
  const lo = h.lo ?? 0;
  const hi = h.hi ?? 0;
  const sp = hi - lo;
  const xOf = (v: number) => (sp > 0 ? Math.max(0, Math.min(100, ((v - lo) / sp) * 100)) : 50);
  return (
    <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" className="block w-full" style={{ height }} role="img" aria-label={`distribution from ${fmt(lo)} to ${fmt(hi)}`} data-e2e={e2e}>
      <title>{`${h.count.toLocaleString()} weights in ${n} bins from ${fmt(lo)} to ${fmt(hi)}; heights are square-root counts`}</title>
      <line x1="0" x2="100" y1={height - 0.5} y2={height - 0.5} stroke="var(--line-2)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      {h.bins.map((c, i) => {
        const bh = c > 0 ? Math.max(1, (Math.sqrt(c) / max) * (height - 1)) : 0;
        return <rect key={i} x={(i * 100) / n} width={Math.max(0.1, 100 / n - 0.25)} y={height - 1 - bh} height={bh} fill="var(--text-4)" />;
      })}
      {lo < 0 && hi > 0 && <line x1={xOf(0)} x2={xOf(0)} y1="0" y2={height} stroke="var(--text-3)" strokeWidth="1" strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />}
      {mark !== null && mark !== undefined && Number.isFinite(mark) && <line x1={xOf(mark)} x2={xOf(mark)} y1="0" y2={height} stroke="var(--signal)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />}
    </svg>
  );
}

/** A tensor's statistics row and slim distribution, from a whole-tensor read with `hist=1`. */
function TensorHeader({ t, whole, mark }: { t: OverviewTensor; whole: OverviewGrid | null; mark: number | null }) {
  const h = whole?.hist;
  const s = whole?.summary;
  const shape = t.shape.length === 0 ? "scalar" : t.shape.join(" x ");
  const sparsity = h && h.count > 0 ? (100 * h.zeros) / h.count : null;
  const item = (k: string, v: string, title?: string) => (
    <span className="whitespace-nowrap" title={title}>
      <span className="text-quiet">{k}</span> <span className="text-text">{v}</span>
    </span>
  );
  return (
    <div className="grid gap-1" data-e2e="param-tensor-header">
      <div className="mono flex min-w-0 flex-wrap items-baseline gap-x-2 text-[11px]">
        <span className="min-w-0 truncate text-text" title={t.name}>
          {t.name}
        </span>
        <span className="text-faint">
          {t.layer === null ? "outside layers" : `layer ${t.layer}`} · {t.part}
        </span>
      </div>
      <div className="mono flex flex-wrap gap-x-3 gap-y-0.5 text-[10px]">
        {item("shape", shape)}
        {item("dtype", t.dtype)}
        {item("weights", t.size.toLocaleString())}
        {item("min", fmt(s?.min))}
        {item("max", fmt(s?.max))}
        {item("mean", fmt(s?.mean))}
        {item("std", fmt(h?.std))}
        {item("zeros", sparsity === null ? "-" : `${sparsity < 0.01 && sparsity > 0 ? sparsity.toExponential(1) : Number(sparsity.toPrecision(3))}%`, "share of weights exactly 0")}
        {item("L2", fmt(h?.l2))}
        {(s?.nonfinite ?? 0) > 0 && item("non-finite", String(s?.nonfinite))}
      </div>
      {h ? (
        <div className="grid gap-0.5">
          <HistBars h={h} mark={mark} height={22} e2e="param-tensor-hist" />
          <div className="mono flex justify-between text-[9px] text-quiet">
            <span>{fmt(h.lo)}</span>
            <span>distribution · {h.bins.length} bins · square-root counts{mark !== null ? " · marked: the inspected value" : ""}</span>
            <span>{fmt(h.hi)}</span>
          </div>
        </div>
      ) : (
        <div aria-hidden className="placeholder h-[22px]" />
      )}
    </div>
  );
}

/** A ruler in weight indices along one side of the view. */
function Ruler({ axis, from, to, px, length }: { axis: "rows" | "cols"; from: number; to: number; px: number; length: number }) {
  const n = to - from;
  const step = niceStep(n, Math.max(2, px / (axis === "cols" ? 70 : 40)), 1);
  const ts = ticks(Math.ceil(from), Math.floor(to - 1e-9), step).filter((x) => x >= 0 && x < length);
  const at = (i: number) => ((i + 0.5 - from) / n) * px;
  const cols = axis === "cols";
  return (
    <svg width={cols ? px : RULER_L} height={cols ? RULER_T : px} className="block overflow-visible" data-e2e="param-ruler" data-axis={axis} aria-label={`${axis} ${Math.floor(from)} to ${Math.ceil(to) - 1}`}>
      <line x1={cols ? 0 : RULER_L - 0.5} x2={cols ? px : RULER_L - 0.5} y1={cols ? RULER_T - 0.5 : 0} y2={cols ? RULER_T - 0.5 : px} stroke="var(--line-2)" strokeWidth="1" />
      {ts.map((i) => {
        const p = Math.round(at(i)) + 0.5;
        return cols ? (
          <g key={i}>
            <line x1={p} x2={p} y1={RULER_T - 5} y2={RULER_T} stroke="var(--text-4)" strokeWidth="1" />
            <text x={p} y={RULER_T - 7} textAnchor="middle" className="mono" fontSize="9" fill="var(--text-3)">
              {i}
            </text>
          </g>
        ) : (
          <g key={i}>
            <line x1={RULER_L - 5} x2={RULER_L} y1={p} y2={p} stroke="var(--text-4)" strokeWidth="1" />
            <text x={RULER_L - 7} y={p + 3} textAnchor="end" className="mono" fontSize="9" fill="var(--text-3)">
              {i}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** The vertical colour scale: its table, numeric ticks, the zero mark and the view's true min and max. */
function Legend({ lut, s, kind, height, min, max, stat }: { lut: Rgb[]; s: number; kind: Scale; height: number; min: number | null; max: number | null; stat: Stat }) {
  const bar = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = bar.current;
    if (!c) return;
    c.width = 1;
    c.height = lut.length;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(1, lut.length);
    // Top is the largest value.
    lut.forEach((col, i) => {
      const k = (lut.length - 1 - i) * 4;
      img.data[k] = col[0];
      img.data[k + 1] = col[1];
      img.data[k + 2] = col[2];
      img.data[k + 3] = 255;
    });
    ctx.putImageData(img, 0, 0);
  }, [lut]);
  const lo = kind === "diverging" ? -s : 0;
  const hi = s;
  const yOf = (v: number) => (hi > lo ? ((hi - Math.max(lo, Math.min(hi, v))) / (hi - lo)) * height : height / 2);
  const step = niceStep(hi - lo, Math.max(2, height / 44));
  const ts = s > 0 ? ticks(lo, hi, step) : [];
  const marks = [
    { k: "max", v: max },
    { k: "min", v: kind === "sequential" ? null : min },
  ].filter((m): m is { k: string; v: number } => m.v !== null && Number.isFinite(m.v));
  // Where each mark's label sits: on its line, kept inside the bar; when the two would share a
  // line (both beyond the scale, or a narrow range) the lower one steps down a line.
  const labelY: Record<string, number> = {};
  for (const m of marks) labelY[m.k] = m.k === "max" ? Math.max(8, Math.round(yOf(m.v)) + 3.5) : Math.min(height - 1, Math.round(yOf(m.v)) + 3.5);
  if (labelY.max !== undefined && labelY.min !== undefined && labelY.min - labelY.max < 10) {
    labelY.min = labelY.max + 10;
    if (labelY.min > height - 1) {
      labelY.min = height - 1;
      labelY.max = labelY.min - 10;
    }
  }
  const near = (y: number) => marks.some((m) => Math.abs(yOf(m.v) - y) < 9 || Math.abs(labelY[m.k] - 3 - y) < 9);
  return (
    <div className="relative" style={{ width: LEGEND, height }} data-e2e="param-legend" title={`colour: ${kind === "diverging" ? `diverging around 0 over ±${fmt(s)}` : `magnitude over 0 to ${fmt(s)}`} (the 99th percentile of |${stat}| in the view; beyond it clipped)`}>
      <canvas ref={bar} className="absolute top-0 left-0 border border-line-strong" style={{ width: 10, height, imageRendering: "pixelated" }} aria-hidden />
      <svg width={LEGEND} height={height} className="absolute inset-0 overflow-visible" aria-hidden>
        {ts.map((v) => {
          const y = Math.round(yOf(v)) + 0.5;
          if (near(y) && v !== 0) return null;
          return (
            <g key={v}>
              <line x1="10" x2={v === 0 ? 18 : 14} y1={y} y2={y} stroke={v === 0 ? "var(--text-2)" : "var(--text-4)"} strokeWidth="1" />
              {/* The zero line stays; its number gives way to a min or max label on the same line. */}
              {!near(y) && (
                <text x="20" y={y + 3} className="mono" fontSize="9" fill={v === 0 ? "var(--text-2)" : "var(--text-4)"}>
                  {fmt(v)}
                </text>
              )}
            </g>
          );
        })}
        {kind === "diverging" && s > 0 && <line x1="0" x2="10" y1={Math.round(yOf(0)) + 0.5} y2={Math.round(yOf(0)) + 0.5} stroke="var(--text-2)" strokeWidth="1" />}
        {marks.map((m) => {
          const clipped = m.v > hi || m.v < lo;
          const y = Math.round(yOf(m.v)) + 0.5;
          return (
            <g key={m.k}>
              <path d={`M0 ${y} L-5 ${y - 3} L-5 ${y + 3} Z`} fill="var(--signal)" />
              <line x1="0" x2="10" y1={y} y2={y} stroke="var(--signal)" strokeWidth="1" />
              <text x="20" y={labelY[m.k]} className="mono" fontSize="9" fill="var(--signal)">
                {m.k} {fmt(m.v)}
                {clipped ? "+" : ""}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ---------- the value list ---------- */

function Values({ k, t, g }: { k: string; t: OverviewTensor; g: Region }) {
  const [page, setPage] = useState(0);
  const pages = pagesOf(g);
  const p = Math.min(page, pages.count - 1);
  const at = pages.at(p);
  const q = useRead<OverviewGrid>(gridPath(k, t, at, { w: at.c1 - at.c0, h: at.r1 - at.r0 }));
  const d = q.stale ? null : q.data;
  const rows: { ix: number; r: number; c: number; v: number | null }[] = [];
  if (d && !d.error && d.exact) {
    const W = at.c1 - at.c0;
    (d.values ?? []).forEach((v, i) => {
      const r = at.r0 + Math.floor(i / W);
      const c = at.c0 + (i % W);
      rows.push({ ix: r * t.cols + c, r, c, v });
    });
  }
  return (
    <div className="grid gap-1" data-e2e="param-overview-values">
      <div className="mono flex flex-wrap items-center gap-2 text-[10px] text-faint">
        <span>
          page {p + 1} of {pages.count} · {label(at, t)}
        </span>
        <button type="button" className={btn} disabled={p === 0} onClick={() => setPage(0)}>
          first
        </button>
        <button type="button" className={btn} disabled={p === 0} onClick={() => setPage(p - 1)}>
          prev
        </button>
        <button type="button" className={btn} disabled={p >= pages.count - 1} onClick={() => setPage(p + 1)}>
          next
        </button>
        <button type="button" className={btn} disabled={p >= pages.count - 1} onClick={() => setPage(pages.count - 1)}>
          last
        </button>
        <label className="flex items-center gap-1">
          page
          <input
            type="number"
            min={1}
            max={pages.count}
            className="mono h-6 w-20 border-b border-line bg-transparent px-1 text-[10px] text-text outline-none"
            defaultValue={p + 1}
            key={p}
            onKeyDown={(e) => {
              if (e.key !== "Enter") return;
              const n = Number((e.target as HTMLInputElement).value);
              if (Number.isFinite(n)) setPage(Math.max(0, Math.min(pages.count - 1, Math.floor(n) - 1)));
            }}
          />
        </label>
      </div>
      {q.error || d?.error ? (
        <p className="mono text-[10.5px] text-error">{q.error ?? d?.error}</p>
      ) : !d ? (
        <div role="status" aria-label="loading values" data-loading="values" className="placeholder relative h-16">
          <span className="placeholder-label mono">loading values</span>
        </div>
      ) : (
        <div className="max-h-72 overflow-auto">
          <table className="mono w-full text-left text-[10px]">
            <thead className="sticky top-0 bg-bg text-faint">
              <tr>
                <th className="py-0.5 pr-3 font-normal">index</th>
                <th className="py-0.5 pr-3 font-normal">row</th>
                <th className="py-0.5 pr-3 font-normal">col</th>
                <th className="py-0.5 pr-3 font-normal">value</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((x) => (
                <tr key={x.ix} className="border-t border-line/40">
                  <td className="py-0.5 pr-3 text-faint">{x.ix}</td>
                  <td className="py-0.5 pr-3 text-faint">{x.r}</td>
                  <td className="py-0.5 pr-3 text-faint">{x.c}</td>
                  <td className="py-0.5 pr-3 text-text">{x.v === null ? "non-finite" : String(x.v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ---------- the inspector ---------- */

/** What a click picked: one weight, or a cell of many with its summary as read. */
type Pick = { reg: Region; value?: number | null; cell?: { min: number | null; max: number | null; mean: number | null; rms: number | null } };
type PointRead = { error?: string; values?: (number | null)[]; overlay?: [number | null, boolean, boolean][] | null; maps?: { map: string; at: string; values?: (number | null)[]; value?: number | null }[] };

type DragKind = "move" | "e" | "s" | "se";
type DragState = { kind: DragKind; sx: number; sy: number; start: WinRect };

const pillBtn = "press mono border border-line-strong bg-bg-2/70 px-1.5 py-0.5 text-[9.5px] uppercase tracking-[0.08em] text-muted hover:border-sense-pale hover:text-text disabled:opacity-40";

function Inspector({
  k,
  t,
  pick,
  whole,
  maps,
  mapOverlay,
  editFocusTick = 0,
  open,
  minimized,
  win,
  onWinChange,
  onResetWin,
  onToggleMin,
  onClose,
  onSelect,
  onZoomToBlock,
}: {
  k: string;
  t: OverviewTensor;
  pick: Pick | null;
  whole: OverviewGrid | null;
  maps: OverviewMap[];
  mapOverlay?: ActiveMapOverlay | null;
  editFocusTick?: number;
  open: boolean;
  minimized: boolean;
  win: WinRect;
  onWinChange: (next: WinRect) => void;
  onResetWin: () => void;
  onToggleMin: () => void;
  onClose: () => void;
  onSelect?: (path: string) => void;
  onZoomToBlock?: (reg: Region) => void;
}) {
  const [headDim, setHeadDim] = useState(128);
  const [editOp, setEditOp] = useState<"set" | "add" | "mul" | "scale">("set");
  const [draftVal, setDraftVal] = useState("");
  const [draftFor, setDraftFor] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [editNote, setEditNote] = useState<{ text: string; tone: "ok" | "err" } | null>(null);
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [dragging, setDragging] = useState<DragKind | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const editInputRef = useRef<HTMLInputElement>(null);
  const one = pick !== null && pick.reg.r1 - pick.reg.r0 === 1 && pick.reg.c1 - pick.reg.c0 === 1;
  const wPath = pick ? pathOf(t, pick.reg) : null;
  // The weight's own read: its exact value, overlay state, and the maps' values at its place.
  const pr = useRead<PointRead>(open && one && wPath ? `/api/cartridges/parameters?key=${enc(k)}&path=${enc(wPath)}&_r=${refreshNonce}` : null);
  const point = pr.stale ? null : pr.data;
  const h = whole?.hist;
  const value = pick?.value ?? point?.values?.[0] ?? null;
  const overlaid = point?.overlay?.[0] ?? null;
  const defaultDraft = one
    ? overlaid && overlaid[0] !== null
      ? String(overlaid[0])
      : value !== null
        ? String(value)
        : ""
    : pick?.cell?.mean !== null && pick?.cell?.mean !== undefined
      ? String(Number(pick.cell.mean.toPrecision(4)))
      : "0";
  if (wPath !== draftFor) {
    setDraftFor(wPath);
    setDraftVal(defaultDraft);
    setEditNote(null);
  }
  useEffect(() => {
    if (open && !minimized && editFocusTick > 0 && wPath) {
      window.setTimeout(() => {
        editInputRef.current?.focus();
        editInputRef.current?.select();
      }, 24);
    }
  }, [open, minimized, editFocusTick, wPath]);
  useEffect(() => {
    if (!open) return;
    const onResize = () => onWinChange(clampWin(win, minimized));
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      onClose();
    };
    window.addEventListener("resize", onResize);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("resize", onResize);
      window.removeEventListener("keydown", onKey);
    };
  }, [open, win, minimized, onWinChange, onClose]);

  const beginDrag = (kind: DragKind, e: ReactPointerEvent<HTMLElement>) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = { kind, sx: e.clientX, sy: e.clientY, start: win };
    setDragging(kind);
  };
  const moveDrag = (e: ReactPointerEvent<HTMLElement>) => {
    const d = dragRef.current;
    if (!d) return;
    const dx = e.clientX - d.sx;
    const dy = e.clientY - d.sy;
    if (d.kind === "move") {
      onWinChange(clampWin({ ...d.start, x: d.start.x + dx, y: d.start.y + dy }, minimized));
    } else if (d.kind === "e") {
      onWinChange(clampWin({ ...d.start, w: d.start.w + dx }, minimized));
    } else if (d.kind === "s") {
      onWinChange(clampWin({ ...d.start, h: d.start.h + dy }, minimized));
    } else {
      onWinChange(clampWin({ ...d.start, w: d.start.w + dx, h: d.start.h + dy }, minimized));
    }
  };
  const endDrag = () => {
    dragRef.current = null;
    setDragging(null);
  };

  const applyOverlay = async (body: Record<string, unknown>, okLabel: string) => {
    if (!wPath) return;
    setBusy(true);
    setEditNote(null);
    try {
      const r = await post<{ audit_id?: string; takes_effect?: { when?: string } }>("/api/parameters/overlay", { key: k, path: wPath, ...body });
      setEditNote({ text: `${okLabel}${r.takes_effect?.when ? ` (${r.takes_effect.when})` : ""}${r.audit_id ? ` · Audit ${r.audit_id}` : ""}`, tone: "ok" });
      setRefreshNonce((n) => n + 1);
      for (const p of ["/api/cartridges/parameters", "/api/parameters/overlay", "/api/audit"]) void refreshAll(p);
    } catch (e) {
      setEditNote({ text: e instanceof Error ? e.message : String(e), tone: "err" });
    }
    setBusy(false);
  };
  const applyValue = () => {
    const num = Number(draftVal.trim());
    if (!Number.isFinite(num)) {
      setEditNote({ text: "Enter a finite number.", tone: "err" });
      return;
    }
    void applyOverlay({ action: "edit", op: editOp, value: num }, `${editOp} ${num}`);
  };
  const q = (h?.q ?? []).filter((x): x is number => x !== null);
  const pct = one && value !== null && q.length === 101 ? percentileOf(value, q) : null;
  const layerMaps = t.layer === null ? [] : maps.map((m) => ({ file: m.file, v: m.layers.find((x) => x.layer === t.layer)?.value })).filter((m) => m.v !== undefined && m.v !== null);
  const pointMaps = (point?.maps ?? []).filter((m) => m.values || m.value !== undefined);
  const two = t.shape.length === 2;
  const heads = two && headDim > 0 && t.rows % headDim === 0 ? t.rows / headDim : 0;
  const detailRow = (kk: string, v: React.ReactNode) => (
    <div className="contents">
      <dt className="shrink-0 text-quiet">{kk}</dt>
      <dd className="min-w-0 break-all text-text">{v}</dd>
    </div>
  );
  const sel = (what: string, path: string) => (
    <button type="button" className={pillBtn} disabled={!onSelect} onClick={() => onSelect?.(path)} title={path} data-e2e="param-inspector-select" data-what={what}>
      {what}
    </button>
  );

  if (!open || typeof document === "undefined") return null;

  const badge = one
    ? two || t.rows > 1
      ? `r${pick.reg.r0} · c${pick.reg.c0}`
      : `idx ${pick.reg.c0}`
    : pick
      ? `${((pick.reg.r1 - pick.reg.r0) * (pick.reg.c1 - pick.reg.c0)).toLocaleString()} wts`
      : "ready";

  const panel = (
    <aside
      role="dialog"
      aria-label="Parameter Inspector"
      data-e2e="param-inspector"
      className="mono fixed z-50 flex flex-col overflow-hidden border border-line-3 bg-bg-1/96 text-[10.5px] text-text shadow-[0_18px_48px_rgba(2,3,10,0.88)] backdrop-blur-md"
      style={{
        left: win.x,
        top: win.y,
        width: win.w,
        height: minimized ? "auto" : win.h,
        maxWidth: "calc(100vw - 24px)",
        maxHeight: "calc(100vh - 24px)",
      }}
    >
      {/* Draggable title bar */}
      <div
        onPointerDown={(e) => beginDrag("move", e)}
        onPointerMove={moveDrag}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onDoubleClick={onToggleMin}
        className={`flex h-9 shrink-0 select-none items-center justify-between gap-2 border-b border-line-strong bg-bg-2 px-2.5 ${dragging === "move" ? "cursor-grabbing" : "cursor-grab"}`}
        title="Drag to move · Double-click to collapse"
      >
        <div className="flex min-w-0 items-center gap-2">
          <svg width="8" height="12" viewBox="0 0 8 12" aria-hidden className="shrink-0 text-quiet">
            <circle cx="2" cy="2.5" r="1" fill="currentColor" />
            <circle cx="6" cy="2.5" r="1" fill="currentColor" />
            <circle cx="2" cy="6" r="1" fill="currentColor" />
            <circle cx="6" cy="6" r="1" fill="currentColor" />
            <circle cx="2" cy="9.5" r="1" fill="currentColor" />
            <circle cx="6" cy="9.5" r="1" fill="currentColor" />
          </svg>
          <span className="text-[10px] font-medium uppercase tracking-[0.14em] text-text">inspector</span>
          <span className="truncate border border-line-strong bg-bg-1 px-1.5 py-0.5 text-[9px] text-sense-pale">{badge}</span>
        </div>
        <div className="flex shrink-0 items-center gap-1" onPointerDown={(e) => e.stopPropagation()}>
          <button
            type="button"
            onClick={onResetWin}
            title="Reset popup position and size"
            className="press h-6 border border-line px-1.5 text-[9px] uppercase tracking-[0.08em] text-faint hover:border-line-3 hover:text-text"
          >
            fit
          </button>
          <button
            type="button"
            onClick={onToggleMin}
            title={minimized ? "Expand inspector" : "Collapse inspector"}
            aria-label={minimized ? "Expand inspector" : "Collapse inspector"}
            className="press flex h-6 w-6 items-center justify-center border border-line text-[12px] leading-none text-faint hover:border-line-3 hover:text-text"
          >
            {minimized ? "+" : "-"}
          </button>
          <button
            type="button"
            onClick={onClose}
            title="Close inspector (Esc)"
            aria-label="Close inspector"
            className="press flex h-6 w-6 items-center justify-center border border-line text-[12px] leading-none text-faint hover:border-error/70 hover:text-error"
          >
            x
          </button>
        </div>
      </div>

      {/* Scrollable body */}
      {!minimized && (
        <>
          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto overflow-x-hidden p-3">
            {!pick ? (
              <div className="space-y-2 border border-line bg-bg-2/50 p-3 text-[10.5px] leading-relaxed text-faint">
                <p className="text-text">Nothing picked yet.</p>
                <p>Click any weight or cell on the heatmap to inspect it here, or double-click any cell/block to jump straight into manual editing.</p>
                <p className="text-[9.5px] text-quiet">Wheel zooms at cursor · drag pans · Shift+drag box-zooms.</p>
              </div>
            ) : one ? (
              <>
                <div className="space-y-1 border border-line bg-bg-2/70 p-2.5">
                  <div className="flex items-center justify-between gap-2 text-[9px] uppercase tracking-[0.12em] text-quiet">
                    <span>weight value</span>
                    <span className="text-faint">{two || t.rows > 1 ? `row ${pick.reg.r0}, col ${pick.reg.c0}` : `index ${pick.reg.c0}`}</span>
                  </div>
                  <div className="break-all text-[18px] leading-snug text-text" data-e2e="param-inspector-value">
                    {value === null ? (pr.error ? "-" : "reading...") : String(value)}
                  </div>
                  {overlaid && (
                    <div className="border-t border-line/60 pt-1 text-[10px] text-verified">
                      overlaid: {overlaid[0] === null ? "-" : String(overlaid[0])}
                      {overlaid[1] ? " · masked" : ""}
                      {overlaid[2] ? " · frozen" : ""}
                    </div>
                  )}
                </div>

                <div className="space-y-2 border border-line bg-bg-2/40 p-2.5" data-e2e="param-inspector-edit">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[9.5px] uppercase tracking-[0.12em] text-muted">edit manually (overlay)</span>
                    <span className="text-[9px] text-quiet">reversible</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <select
                      value={editOp}
                      onChange={(e) => setEditOp(e.target.value as "set" | "add" | "mul" | "scale")}
                      aria-label="Edit operation"
                      className="mono h-7 shrink-0 border border-line-strong bg-bg-1 px-1.5 text-[10px] text-text outline-none focus-visible:border-sense-pale"
                    >
                      <option value="set" className="bg-bg-1">set</option>
                      <option value="add" className="bg-bg-1">add</option>
                      <option value="mul" className="bg-bg-1">mul</option>
                      <option value="scale" className="bg-bg-1">scale</option>
                    </select>
                    <input
                      ref={editInputRef}
                      value={draftVal}
                      onChange={(e) => setDraftVal(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          applyValue();
                        }
                      }}
                      aria-label="New parameter value"
                      className="mono h-7 min-w-0 flex-1 border border-line-strong bg-bg px-2 text-[11px] text-text outline-none focus-visible:border-sense-pale"
                    />
                    <button
                      type="button"
                      disabled={busy}
                      onClick={applyValue}
                      className="press mono h-7 shrink-0 border border-sense-pale bg-sense/20 px-2.5 text-[9.5px] uppercase tracking-[0.1em] text-sense-pale hover:bg-sense/30 hover:text-text disabled:opacity-40"
                    >
                      {busy ? "..." : "apply"}
                    </button>
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    <button type="button" disabled={busy} className={pillBtn} onClick={() => void applyOverlay({ action: "mask" }, "masked")}>
                      mask
                    </button>
                    <button type="button" disabled={busy} className={pillBtn} onClick={() => void applyOverlay({ action: "freeze" }, "frozen")}>
                      freeze
                    </button>
                    <button type="button" disabled={busy} className={pillBtn} onClick={() => void applyOverlay({ action: "unmask" }, "unmasked")}>
                      unmask
                    </button>
                  </div>
                  {editNote && <p className={`break-words text-[9.5px] leading-snug ${editNote.tone === "err" ? "text-error" : "text-verified"}`}>{editNote.text}</p>}
                </div>

                <dl className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-2 gap-y-1 border-t border-line pt-2 text-[10px]">
                  {detailRow("flat index", (pick.reg.r0 * t.cols + pick.reg.c0).toLocaleString())}
                  {detailRow("tensor", t.name)}
                  {detailRow("layer", t.layer === null ? "outside layers" : `layer ${t.layer}`)}
                  {detailRow("block", t.part)}
                  {heads > 0 && detailRow("head", `${Math.floor(pick.reg.r0 / headDim)} of ${heads} (${headDim} rows each)`)}
                </dl>

                <div className="space-y-1 border-t border-line pt-2">
                  <div className="flex items-center justify-between gap-2 text-[10px]">
                    <span className="text-quiet">percentile in tensor</span>
                    <span className="text-text" data-e2e="param-inspector-percentile">
                      {pct === null ? (h ? "-" : "reading") : `${pct.toFixed(pct >= 99 || pct <= 1 ? 2 : 1)}%`}
                    </span>
                  </div>
                  {h && <HistBars h={h} mark={value} height={36} />}
                  {h && (
                    <div className="flex justify-between gap-2 text-[9px] text-quiet">
                      <span>{fmt(h.lo)}</span>
                      <span>{h.precision} buckets</span>
                      <span>{fmt(h.hi)}</span>
                    </div>
                  )}
                </div>

                <div className="space-y-1 border-t border-line pt-2 text-[10px]" data-e2e="param-inspector-maps">
                  <div className="text-[9.5px] uppercase tracking-[0.12em] text-quiet">parameter maps here</div>
                  {mapOverlay && (mapOverlay.pointsByTensor[t.name] ?? []).some(([r, c]) => r === pick.reg.r0 && c === pick.reg.c0) && (
                    <div className="flex items-center justify-between gap-2 border border-[#ffd452]/40 bg-[#ffd452]/10 px-2 py-1 text-[#ffd452]">
                      <span className="min-w-0 truncate">★ {mapOverlay.file.replace(/\.json$/, "")}</span>
                      <span className="shrink-0 text-[9px] uppercase">pinned</span>
                    </div>
                  )}
                  {pointMaps.map((m) => (
                    <div key={m.map} className="flex items-center justify-between gap-2" title={`${m.map}: ${m.at}`}>
                      <span className="min-w-0 truncate text-faint">{m.map.replace(/\.json$/, "")}</span>
                      <span className="shrink-0 text-text">{fmt(m.values ? m.values[0] : m.value)}</span>
                    </div>
                  ))}
                  {layerMaps
                    .filter((m) => !pointMaps.some((p) => p.map === m.file))
                    .map((m) => (
                      <div key={m.file} className="flex items-center justify-between gap-2" title={`${m.file}: per layer only; its value at layer ${t.layer}`}>
                        <span className="min-w-0 truncate text-faint">{m.file.replace(/\.json$/, "")}</span>
                        <span className="shrink-0 text-muted">
                          {fmt(m.v)} <span className="text-quiet">L{t.layer}</span>
                        </span>
                      </div>
                    ))}
                  {layerMaps.length > 0 && pointMaps.length === 0 && <p className="text-[9px] text-quiet">Per-layer values only (not per-weight).</p>}
                  {pointMaps.length === 0 && layerMaps.length === 0 && <p className="text-[9px] text-quiet">{pr.stale || (!point && !pr.error) ? "reading..." : "No parameter map holds a value here."}</p>}
                </div>

                <div className="space-y-1.5 border-t border-line pt-2">
                  <div className="text-[9.5px] uppercase tracking-[0.12em] text-quiet">send selection to controls</div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {sel("weight", pathOf(t, pick.reg))}
                    {two && sel("row", `${t.name}[${pick.reg.r0}:${pick.reg.r0 + 1}, :]`)}
                    {two && sel("column", `${t.name}[:, ${pick.reg.c0}:${pick.reg.c0 + 1}]`)}
                    {two && (
                      <span className="flex items-center gap-1">
                        <button
                          type="button"
                          className={pillBtn}
                          disabled={!onSelect || heads === 0}
                          title={heads ? `rows of one head of ${headDim}` : `${t.rows} rows do not split into heads of ${headDim}`}
                          onClick={() => {
                            const hd = Math.floor(pick.reg.r0 / headDim);
                            onSelect?.(`${t.name}[${hd * headDim}:${(hd + 1) * headDim}, :]`);
                          }}
                          data-e2e="param-inspector-select"
                          data-what="head"
                        >
                          head
                        </button>
                        <input
                          value={headDim}
                          onChange={(e) => setHeadDim(Math.max(1, Math.floor(Number(e.target.value) || 0)))}
                          aria-label="Head size in rows"
                          className="mono h-6 w-12 border border-line-strong bg-bg px-1 text-[10px] text-text outline-none focus-visible:border-sense-pale"
                          inputMode="numeric"
                        />
                      </span>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="space-y-2 border border-line bg-bg-2/70 p-2.5">
                  <div className="flex items-center justify-between gap-2 text-[9px] uppercase tracking-[0.12em] text-quiet">
                    <span>selected block</span>
                    <span className="text-sense-pale">{((pick.reg.r1 - pick.reg.r0) * (pick.reg.c1 - pick.reg.c0)).toLocaleString()} weights</span>
                  </div>
                  <div className="break-words text-[11px] text-text">{label(pick.reg, t)}</div>
                  <div className="grid grid-cols-2 gap-1.5 text-[10px]">
                    <div className="border border-line/70 bg-bg-1 px-2 py-1">
                      <span className="block text-[8.5px] uppercase text-quiet">min</span>
                      <span className="text-text">{fmt(pick.cell?.min)}</span>
                    </div>
                    <div className="border border-line/70 bg-bg-1 px-2 py-1">
                      <span className="block text-[8.5px] uppercase text-quiet">max</span>
                      <span className="text-text">{fmt(pick.cell?.max)}</span>
                    </div>
                    <div className="border border-line/70 bg-bg-1 px-2 py-1">
                      <span className="block text-[8.5px] uppercase text-quiet">mean</span>
                      <span className="text-text">{fmt(pick.cell?.mean)}</span>
                    </div>
                    <div className="border border-line/70 bg-bg-1 px-2 py-1">
                      <span className="block text-[8.5px] uppercase text-quiet">RMS</span>
                      <span className="text-text">{fmt(pick.cell?.rms)}</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-2 border border-line bg-bg-2/40 p-2.5" data-e2e="param-inspector-edit">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[9.5px] uppercase tracking-[0.12em] text-muted">edit block manually (overlay)</span>
                    <span className="text-[9px] text-quiet">reversible</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <select
                      value={editOp}
                      onChange={(e) => setEditOp(e.target.value as "set" | "add" | "mul" | "scale")}
                      aria-label="Block edit operation"
                      className="mono h-7 shrink-0 border border-line-strong bg-bg-1 px-1.5 text-[10px] text-text outline-none focus-visible:border-sense-pale"
                    >
                      <option value="set" className="bg-bg-1">set</option>
                      <option value="add" className="bg-bg-1">add</option>
                      <option value="mul" className="bg-bg-1">mul</option>
                      <option value="scale" className="bg-bg-1">scale</option>
                    </select>
                    <input
                      ref={editInputRef}
                      value={draftVal}
                      onChange={(e) => setDraftVal(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          applyValue();
                        }
                      }}
                      aria-label="New block value"
                      className="mono h-7 min-w-0 flex-1 border border-line-strong bg-bg px-2 text-[11px] text-text outline-none focus-visible:border-sense-pale"
                    />
                    <button
                      type="button"
                      disabled={busy}
                      onClick={applyValue}
                      className="press mono h-7 shrink-0 border border-sense-pale bg-sense/20 px-2.5 text-[9.5px] uppercase tracking-[0.1em] text-sense-pale hover:bg-sense/30 hover:text-text disabled:opacity-40"
                    >
                      {busy ? "..." : "apply"}
                    </button>
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    <button type="button" disabled={busy} className={pillBtn} onClick={() => void applyOverlay({ action: "mask" }, "block masked")}>
                      mask block
                    </button>
                    <button type="button" disabled={busy} className={pillBtn} onClick={() => void applyOverlay({ action: "freeze" }, "block frozen")}>
                      freeze block
                    </button>
                    <button type="button" disabled={busy} className={pillBtn} onClick={() => void applyOverlay({ action: "unmask" }, "block unmasked")}>
                      unmask
                    </button>
                  </div>
                  {editNote && <p className={`break-words text-[9.5px] leading-snug ${editNote.tone === "err" ? "text-error" : "text-verified"}`}>{editNote.text}</p>}
                </div>

                <dl className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-2 gap-y-1 border-t border-line pt-2 text-[10px]">
                  {detailRow("tensor", t.name)}
                  {detailRow("layer", t.layer === null ? "outside layers" : `layer ${t.layer}`)}
                  {detailRow("block", t.part)}
                </dl>

                <div className="space-y-1.5 border-t border-line pt-2">
                  <div className="text-[9.5px] uppercase tracking-[0.12em] text-quiet">block actions</div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {sel("select in controls", pathOf(t, pick.reg))}
                    {onZoomToBlock && (
                      <button type="button" className={pillBtn} onClick={() => onZoomToBlock(pick.reg)} title="Zoom into this block to inspect and edit individual weights">
                        zoom to weights
                      </button>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Slim footer with dedicated bottom-right resize handle (never overlaps content) */}
          <div className="flex h-5 shrink-0 select-none items-center justify-between border-t border-line bg-bg-2/85 pl-2.5 pr-0.5 text-[8.5px] text-quiet">
            <span className="truncate">drag title to move · drag edge or corner to resize</span>
            <div
              role="separator"
              aria-label="Resize inspector"
              onPointerDown={(e) => beginDrag("se", e)}
              onPointerMove={moveDrag}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
              className="flex h-5 w-6 shrink-0 cursor-nwse-resize items-center justify-center text-faint hover:text-sense-pale"
              title="Drag to resize"
            >
              <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
                <path d="M9 1L1 9M9 5L5 9M9 8.5L8.5 9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="square" />
              </svg>
            </div>
          </div>

          {/* Edge resize strips */}
          <div
            aria-hidden
            onPointerDown={(e) => beginDrag("e", e)}
            onPointerMove={moveDrag}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            className="absolute top-9 right-0 bottom-5 w-1.5 cursor-ew-resize"
          />
          <div
            aria-hidden
            onPointerDown={(e) => beginDrag("s", e)}
            onPointerMove={moveDrag}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            className="absolute right-6 bottom-0 left-0 h-1.5 cursor-ns-resize"
          />
        </>
      )}
    </aside>
  );

  return createPortal(panel, document.body);
}

/* ---------- the primary view ---------- */

type Press = { x: number; y: number; view: Region; shift: boolean; moved: boolean; id: number };
type MarkRect = { action: OverviewMark["action"]; rect: Rect; id: string };

function PrimaryView({
  k,
  t,
  stat,
  scale,
  theme,
  avail,
  whole,
  marks,
  maps,
  mapOverlay,
  focus,
  inspectorOpen,
  inspectorMin,
  inspectorWin,
  onInspectorWinChange,
  onResetInspectorWin,
  onOpenInspector,
  onToggleInspector,
  onToggleInspectorMin,
  onCloseInspector,
  onSelect,
}: {
  k: string;
  t: OverviewTensor;
  stat: Stat;
  scale: Scale;
  theme: Theme;
  avail: number;
  whole: OverviewGrid | null;
  marks: MarkRect[];
  maps: OverviewMap[];
  mapOverlay?: ActiveMapOverlay | null;
  focus: { rect: Rect; n: number } | null;
  inspectorOpen: boolean;
  inspectorMin: boolean;
  inspectorWin: WinRect;
  onInspectorWinChange: (next: WinRect) => void;
  onResetInspectorWin: () => void;
  onOpenInspector: () => void;
  onToggleInspector: () => void;
  onToggleInspectorMin: () => void;
  onCloseInspector: () => void;
  onSelect?: (path: string) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const press = useRef<Press | null>(null);
  const cw = Math.max(160, Math.floor(avail - RULER_L - LEGEND - LEGEND_GAP - 6));
  const ch = t.rows === 1 ? 84 : Math.max(160, Math.min(460, Math.round((cw * t.rows) / t.cols)));
  const full = wholeOf(t);
  const [view, setView] = useState<Region>(() => (focus ? around(focus.rect, t, cw, ch) : full));
  const [stack, setStack] = useState<Region[]>(() => (focus ? [full, around(focus.rect, t, cw, ch)] : [full]));
  const [seenFocus, setSeenFocus] = useState<number | null>(focus?.n ?? null);
  const [box, setBox] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const [pick, setPick] = useState<Pick | null>(() => (focus && focus.rect.r1 - focus.rect.r0 === 1 && focus.rect.c1 - focus.rect.c0 === 1 ? { reg: focus.rect } : null));
  const [list, setList] = useState(false);
  const [grabbing, setGrabbing] = useState(false);
  const [tool, setTool] = useState<"pan" | "box">("pan");
  const [overlayMode, setOverlayMode] = useState<"show" | "boost" | "off">("show");
  const [mapMode, setMapMode] = useState<"show" | "boost" | "off">("show");
  const [editFocusTick, setEditFocusTick] = useState(0);
  const mapPoints = useMemo(() => (mapOverlay ? (mapOverlay.pointsByTensor[t.name] ?? []) : []), [mapOverlay, t.name]);
  const mapLayerHit = Boolean(mapOverlay && t.layer !== null && mapOverlay.layers.includes(t.layer));
  const hasMapVisual = mapPoints.length > 0 || mapLayerHit;
  const luts = useLuts(theme);
  const lut = luts[scale];

  // A new target zooms to it (state adjusted while rendering, as React advises for a prop change).
  if (focus && focus.n !== seenFocus) {
    setSeenFocus(focus.n);
    const v = around(focus.rect, t, cw, ch);
    setView(v);
    setStack([full, v]);
    setPick(focus.rect.r1 - focus.rect.r0 === 1 && focus.rect.c1 - focus.rect.c0 === 1 ? { reg: focus.rect } : null);
  }

  const v = clampView(view, t, cw, ch);
  const reg = intOf(v, t);
  const kx = cw / (v.c1 - v.c0);
  const ky = ch / (v.r1 - v.r0);
  const cells = cellsOf(reg, { w: (reg.c1 - reg.c0) * kx, h: (reg.r1 - reg.r0) * ky });
  const q = useRead<OverviewGrid>(t.readable ? gridPath(k, t, reg, cells) : null, DEBOUNCE_MS);
  const g = q.data && !q.data.error ? q.data : null;
  const values = useMemo(() => (g ? cellValues(g, stat) : []), [g, stat]);
  const s = useMemo(() => scaleOf(values), [values]);
  // The true extent of what the view colours (the chosen statistic of its cells).
  const ext = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const x of values) {
      if (x === null || !Number.isFinite(x)) continue;
      const y = scale === "sequential" ? Math.abs(x) : x;
      if (y < lo) lo = y;
      if (y > hi) hi = y;
    }
    return lo <= hi ? { min: lo, max: hi } : { min: null, max: null };
  }, [values, scale]);
  const img = useMemo(() => (g ? paint(g, values, lut, s, scale) : null), [g, values, lut, s, scale]);
  // Values printed in the cells share one power of ten (shown under the view), so a small cell
  // still holds its sign and leading digits.
  const printing = Boolean(g?.exact) && kx >= TEXT_PX && (t.rows === 1 ? ch : ky) >= TEXT_PX;
  const big = g ? Math.max(Math.abs(g.summary.min ?? 0), Math.abs(g.summary.max ?? 0)) : 0;
  const expo = big > 0 && Number.isFinite(big) ? Math.floor(Math.log10(big)) : 0;
  const unit = Math.abs(expo) >= 1 ? expo : 0;

  // Draw: the latest grid placed where its region lies in the view, then edges, values, the
  // overlay's marks and the picked weight.
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = Math.round(cw * dpr);
    c.height = Math.round(ch * dpr);
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = rgba(theme.bg0, 1);
    ctx.fillRect(0, 0, cw, ch);
    const X = (col: number) => (col - v.c0) * kx;
    const Y = (r: number) => (t.rows === 1 ? (r === 0 ? 0 : ch) : (r - v.r0) * ky);
    if (g && img) {
      ctx.imageSmoothingEnabled = false;
      ctx.save();
      if ((overlayMode === "boost" && marks.length > 0) || (mapMode === "boost" && hasMapVisual)) ctx.globalAlpha = 0.42;
      ctx.drawImage(img, X(g.region.c0), Y(g.region.r0), (g.region.c1 - g.region.c0) * kx, t.rows === 1 ? ch : (g.region.r1 - g.region.r0) * ky);
      ctx.restore();
      const ry = t.rows === 1 ? ch : ky;
      if (g.exact && kx >= EDGE_PX && ry >= EDGE_PX) {
        ctx.strokeStyle = rgba(theme.bg0, 0.85);
        ctx.lineWidth = 1 / dpr;
        ctx.beginPath();
        for (let col = Math.max(g.region.c0, Math.floor(v.c0)); col <= Math.min(g.region.c1, Math.ceil(v.c1)); col++) {
          const x = Math.round(X(col) * dpr) / dpr + 0.5 / dpr;
          ctx.moveTo(x, Math.max(0, Y(g.region.r0)));
          ctx.lineTo(x, Math.min(ch, Y(g.region.r1)));
        }
        if (t.rows > 1)
          for (let r = Math.max(g.region.r0, Math.floor(v.r0)); r <= Math.min(g.region.r1, Math.ceil(v.r1)); r++) {
            const y = Math.round(Y(r) * dpr) / dpr + 0.5 / dpr;
            ctx.moveTo(Math.max(0, X(g.region.c0)), y);
            ctx.lineTo(Math.min(cw, X(g.region.c1)), y);
          }
        ctx.stroke();
      }
      if (printing) {
        const fs = Math.max(6, Math.min(10, Math.floor(Math.min(ry * 0.5, kx / 2.6))));
        const k10 = Math.pow(10, -unit);
        ctx.font = `${fs}px ${theme.mono}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const chars = Math.floor((kx - 2) / (fs * 0.62));
        const W = g.region.c1 - g.region.c0;
        for (let r = Math.max(g.region.r0, Math.floor(v.r0)); r < Math.min(g.region.r1, Math.ceil(v.r1)); r++) {
          for (let col = Math.max(g.region.c0, Math.floor(v.c0)); col < Math.min(g.region.c1, Math.ceil(v.c1)); col++) {
            const x = values[(r - g.region.r0) * W + (col - g.region.c0)];
            if (x === null || x === undefined) continue;
            const text = shortFmt(x * k10, chars);
            if (!text) continue;
            const L = toLab(lut[indexOf(x, s, lut.length, scale)])[0];
            ctx.fillStyle = L > 0.62 ? rgba(theme.bg0, 0.9) : rgba(theme.text1, 0.82);
            ctx.fillText(text, X(col) + kx / 2, t.rows === 1 ? ch / 2 : Y(r) + ky / 2);
          }
        }
      }
    }
    // The overlay's marks: high-contrast so masks, freezes and edits read clearly over dark heatmaps.
    if (overlayMode !== "off") {
      for (const m of marks) {
        if (!overlaps(m.rect, reg)) continue;
        const x0 = Math.max(0, X(m.rect.c0));
        const y0 = Math.max(0, Y(m.rect.r0));
        const x1 = Math.min(cw, X(m.rect.c1));
        const y1 = Math.min(ch, Y(m.rect.r1));
        const mw = x1 - x0;
        const mh = y1 - y0;
        if (mw <= 0 || mh <= 0) continue;
        if (m.action === "mask") {
          ctx.save();
          ctx.beginPath();
          ctx.rect(x0, y0, mw, mh);
          ctx.clip();
          ctx.fillStyle = rgba(theme.bg0, overlayMode === "boost" ? 0.82 : 0.68);
          ctx.fillRect(x0, y0, mw, mh);
          ctx.strokeStyle = "rgba(255, 184, 84, 0.72)";
          ctx.lineWidth = 1.25 / dpr;
          ctx.beginPath();
          for (let d = x0 - mh; d < x1 + mh; d += 8) {
            ctx.moveTo(d, y1);
            ctx.lineTo(d + mh, y0);
            ctx.moveTo(d, y0);
            ctx.lineTo(d + mh, y1);
          }
          ctx.stroke();
          ctx.restore();
          ctx.strokeStyle = "rgba(255, 196, 102, 0.96)";
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x0 + 0.75, y0 + 0.75, Math.max(0, mw - 1.5), Math.max(0, mh - 1.5));
          if (mw >= 46 && mh >= 16) {
            ctx.fillStyle = "rgba(18, 12, 4, 0.86)";
            ctx.fillRect(x0 + 2, y0 + 2, 42, 12);
            ctx.font = `9px ${theme.mono}`;
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.fillStyle = "rgba(255, 206, 120, 1)";
            ctx.fillText("MASKED", x0 + 5, y0 + 4);
          }
        } else if (m.action === "freeze") {
          ctx.save();
          ctx.beginPath();
          ctx.rect(x0, y0, mw, mh);
          ctx.clip();
          ctx.fillStyle = "rgba(110, 190, 255, 0.22)";
          ctx.fillRect(x0, y0, mw, mh);
          ctx.strokeStyle = "rgba(155, 220, 255, 0.85)";
          ctx.lineWidth = 1.25 / dpr;
          ctx.beginPath();
          for (let d = x0 - mh; d < x1; d += 7) {
            ctx.moveTo(d, y1);
            ctx.lineTo(d + mh, y0);
          }
          ctx.stroke();
          ctx.restore();
          ctx.save();
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = "rgba(155, 220, 255, 0.96)";
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x0 + 0.75, y0 + 0.75, Math.max(0, mw - 1.5), Math.max(0, mh - 1.5));
          ctx.restore();
          if (mw >= 46 && mh >= 16) {
            ctx.fillStyle = "rgba(8, 22, 42, 0.86)";
            ctx.fillRect(x0 + 2, y0 + 2, 42, 12);
            ctx.font = `9px ${theme.mono}`;
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.fillStyle = "rgba(170, 228, 255, 1)";
            ctx.fillText("FROZEN", x0 + 5, y0 + 4);
          }
        } else {
          ctx.fillStyle = rgba(theme.verified, overlayMode === "boost" ? 0.24 : 0.14);
          ctx.fillRect(x0, y0, mw, mh);
          ctx.strokeStyle = rgba(theme.verified, 0.98);
          ctx.lineWidth = 1.75;
          ctx.strokeRect(x0 + 0.75, y0 + 0.75, Math.max(0, mw - 1.5), Math.max(0, mh - 1.5));
          if (mw >= 44 && mh >= 16) {
            ctx.fillStyle = "rgba(6, 28, 16, 0.86)";
            ctx.fillRect(x0 + 2, y0 + 2, 40, 12);
            ctx.font = `9px ${theme.mono}`;
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.fillStyle = rgba(theme.verified, 1);
            ctx.fillText("EDITED", x0 + 5, y0 + 4);
          }
        }
      }
    }
    // Active parameter map overlay: exact pinned coordinates (e.g. super-weights) or layer coverage highlight.
    if (mapOverlay && mapMode !== "off") {
      if (mapPoints.length > 0) {
        let visiblePts = 0;
        for (const [pr, pc] of mapPoints) {
          if (pr < reg.r0 || pr >= reg.r1 || pc < reg.c0 || pc >= reg.c1) continue;
          visiblePts += 1;
          const px = X(pc + 0.5);
          const py = t.rows === 1 ? ch / 2 : Y(pr + 0.5);
          if (kx >= EDGE_PX && (t.rows === 1 ? ch : ky) >= EDGE_PX) {
            const bx = X(pc);
            const by = t.rows === 1 ? 1 : Y(pr);
            const bw = Math.max(2, kx);
            const bh = t.rows === 1 ? ch - 2 : Math.max(2, ky);
            ctx.strokeStyle = "rgba(10, 8, 2, 0.92)";
            ctx.lineWidth = 3;
            ctx.strokeRect(bx, by, bw, bh);
            ctx.strokeStyle = "#ffd452";
            ctx.lineWidth = 1.5;
            ctx.strokeRect(bx, by, bw, bh);
          } else {
            const rad = mapMode === "boost" ? 4.5 : 3.2;
            ctx.fillStyle = "rgba(8, 6, 2, 0.85)";
            ctx.fillRect(px - rad - 1, py - rad - 1, (rad + 1) * 2, (rad + 1) * 2);
            ctx.fillStyle = "#ffd452";
            ctx.fillRect(px - rad, py - rad, rad * 2, rad * 2);
          }
        }
        const tag = `MAP: ${mapOverlay.file.replace(/\.json$/, "")} · ${visiblePts}/${mapPoints.length} pts`;
        ctx.font = `9.5px ${theme.mono}`;
        const tw = ctx.measureText(tag).width + 12;
        ctx.fillStyle = "rgba(18, 14, 4, 0.9)";
        ctx.fillRect(cw - tw - 4, 4, tw, 15);
        ctx.strokeStyle = "rgba(255, 212, 82, 0.9)";
        ctx.lineWidth = 1;
        ctx.strokeRect(cw - tw - 4, 4, tw, 15);
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillStyle = "#ffd452";
        ctx.fillText(tag, cw - tw + 2, 7);
      } else if (mapLayerHit) {
        ctx.save();
        ctx.strokeStyle = mapMode === "boost" ? "rgba(127, 176, 255, 0.98)" : "rgba(127, 176, 255, 0.72)";
        ctx.lineWidth = mapMode === "boost" ? 2.5 : 1.5;
        ctx.strokeRect(1, 1, cw - 2, ch - 2);
        const lv = t.layer !== null ? mapOverlay.layerValues[t.layer] : undefined;
        const tag = `MAP: ${mapOverlay.file.replace(/\.json$/, "")} · L${t.layer}${lv !== undefined ? ` (${fmt(lv)})` : ""}`;
        ctx.font = `9.5px ${theme.mono}`;
        const tw = ctx.measureText(tag).width + 12;
        ctx.fillStyle = "rgba(8, 16, 36, 0.9)";
        ctx.fillRect(cw - tw - 4, 4, tw, 15);
        ctx.strokeStyle = "rgba(127, 176, 255, 0.85)";
        ctx.lineWidth = 1;
        ctx.strokeRect(cw - tw - 4, 4, tw, 15);
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillStyle = "#b8d6ff";
        ctx.fillText(tag, cw - tw + 2, 7);
        ctx.restore();
      }
    }
    if (pick) {
      const p = pick.reg;
      const x0 = X(p.c0);
      const y0 = Y(p.r0);
      const w = Math.max(3, (p.c1 - p.c0) * kx);
      const h = t.rows === 1 ? ch : Math.max(3, (p.r1 - p.r0) * ky);
      ctx.lineWidth = 1;
      ctx.strokeStyle = rgba(theme.bg0, 1);
      ctx.strokeRect(x0 - 1.5, y0 - 1.5, w + 3, h + 3);
      ctx.strokeStyle = rgba(theme.signal, 1);
      ctx.strokeRect(x0 - 0.5, y0 - 0.5, w + 1, h + 1);
    }
  }, [g, img, values, v, kx, ky, cw, ch, t.rows, t.layer, theme, marks, reg, pick, lut, s, scale, printing, unit, overlayMode, mapOverlay, mapMode, mapPoints, mapLayerHit, hasMapVisual]);

  const zoomBy = (factor: number, px = 0.5, py = 0.5) => {
    setView((prev) => {
      const p = clampView(prev, t, cw, ch);
      const wx = p.c0 + px * (p.c1 - p.c0);
      const wy = p.r0 + py * (p.r1 - p.r0);
      const nw = (p.c1 - p.c0) / factor;
      const nh = t.rows === 1 ? 1 : (p.r1 - p.r0) / factor;
      return clampView({ c0: wx - px * nw, c1: wx - px * nw + nw, r0: wy - py * nh, r1: wy - py * nh + nh }, t, cw, ch);
    });
  };

  const zoomExact = () => {
    const cx = pick ? (pick.reg.c0 + pick.reg.c1) / 2 : (v.c0 + v.c1) / 2;
    const cy = pick ? (pick.reg.r0 + pick.reg.r1) / 2 : (v.r0 + v.r1) / 2;
    const targetPx = 24;
    const nw = Math.min(t.cols, Math.max(1, cw / targetPx));
    const nh = t.rows === 1 ? 1 : Math.min(t.rows, Math.max(1, ch / targetPx));
    const next = clampView({ c0: cx - nw / 2, c1: cx + nw / 2, r0: cy - nh / 2, r1: cy + nh / 2 }, t, cw, ch);
    setView(next);
    setStack((prev) => [...prev, intOf(next, t)]);
  };

  // The wheel zooms smoothly at the cursor, keeping cursor anchor exact.
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = c.getBoundingClientRect();
      const px = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
      const py = Math.max(0, Math.min(1, (e.clientY - rect.top) / Math.max(1, rect.height)));
      const f = Math.exp(-Math.max(-240, Math.min(240, e.deltaY * (e.deltaMode === 1 ? 33 : 1))) * 0.0028);
      setView((prev) => {
        const p = clampView(prev, t, cw, ch);
        const wx = p.c0 + px * (p.c1 - p.c0);
        const wy = p.r0 + py * (p.r1 - p.r0);
        const nw = (p.c1 - p.c0) / f;
        const nh = t.rows === 1 ? 1 : (p.r1 - p.r0) / f;
        return clampView({ c0: wx - px * nw, c1: wx - px * nw + nw, r0: wy - py * nh, r1: wy - py * nh + nh }, t, cw, ch);
      });
    };
    c.addEventListener("wheel", wheel, { passive: false });
    return () => c.removeEventListener("wheel", wheel);
  }, [t, cw, ch]);

  const at = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(cw - 1e-6, e.clientX - rect.left));
    const y = Math.max(0, Math.min(ch - 1e-6, e.clientY - rect.top));
    return { x, y, c: Math.min(t.cols - 1, Math.floor(v.c0 + x / kx)), r: t.rows === 1 ? 0 : Math.min(t.rows - 1, Math.floor(v.r0 + y / ky)) };
  };
  const inspect = (r: number, c: number): Pick => {
    if (!g) {
      const next: Pick = { reg: { r0: r, r1: r + 1, c0: c, c1: c + 1 } };
      setPick(next);
      return next;
    }
    const ij = cellOf(g, r, c);
    if (!ij) {
      const next: Pick = { reg: { r0: r, r1: r + 1, c0: c, c1: c + 1 } };
      setPick(next);
      return next;
    }
    const n = ij.i * g.w + ij.j;
    const [r0, r1] = span(ij.i, ij.i, g.region.r1 - g.region.r0, g.h, g.region.r0);
    const [c0, c1] = span(ij.j, ij.j, g.region.c1 - g.region.c0, g.w, g.region.c0);
    if (g.exact || (r1 - r0 === 1 && c1 - c0 === 1)) {
      const next: Pick = { reg: { r0, r1, c0, c1 }, value: g.exact ? (g.values?.[n] ?? null) : (g.min?.[n] ?? null) };
      setPick(next);
      return next;
    }
    const next: Pick = { reg: { r0, r1, c0, c1 }, cell: { min: g.min?.[n] ?? null, max: g.max?.[n] ?? null, mean: g.mean?.[n] ?? null, rms: g.rms?.[n] ?? null } };
    setPick(next);
    return next;
  };
  const zoomTo = (r: Region, select: boolean) => {
    const z = clampView(r, t, cw, ch);
    setView(z);
    if (!same(r, stack[stack.length - 1] ?? full)) setStack([...stack, r]);
    if (select) onSelect?.(pathOf(t, r));
  };
  const down = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = at(e);
    press.current = { x: p.x, y: p.y, view: v, shift: e.shiftKey || tool === "box", moved: false, id: e.pointerId };
  };
  const move = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const p = at(e);
    setHover({ r: p.r, c: p.c });
    const pr = press.current;
    if (!pr) return;
    if (!pr.moved && Math.hypot(p.x - pr.x, p.y - pr.y) < 3) return;
    if (!pr.moved && !pr.shift) setGrabbing(true);
    pr.moved = true;
    if (pr.shift) setBox({ x0: pr.x, y0: pr.y, x1: p.x, y1: p.y });
    else {
      const dc = (p.x - pr.x) / kx;
      const dr = t.rows === 1 ? 0 : (p.y - pr.y) / ky;
      setView(clampView({ r0: pr.view.r0 - dr, r1: pr.view.r1 - dr, c0: pr.view.c0 - dc, c1: pr.view.c1 - dc }, t, cw, ch));
    }
  };
  const up = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const pr = press.current;
    press.current = null;
    setGrabbing(false);
    if (!pr) return;
    const p = at(e);
    if (!pr.moved && Math.hypot(p.x - pr.x, p.y - pr.y) < 3) {
      inspect(p.r, p.c);
      onOpenInspector();
      return;
    }
    if (pr.shift) {
      // The box from the press to the release (the drawn box state may lag a fast drag).
      setBox(null);
      const ca = Math.floor(pr.view.c0 + Math.min(pr.x, p.x) / kx);
      const cb = Math.floor(pr.view.c0 + Math.max(pr.x, p.x) / kx) + 1;
      const ra = t.rows === 1 ? 0 : Math.floor(pr.view.r0 + Math.min(pr.y, p.y) / ky);
      const rb = t.rows === 1 ? 1 : Math.floor(pr.view.r0 + Math.max(pr.y, p.y) / ky) + 1;
      zoomTo({ r0: Math.max(0, ra), r1: Math.min(t.rows, rb), c0: Math.max(0, ca), c1: Math.min(t.cols, cb) }, true);
    }
  };
  const reset = () => {
    setView(full);
    setStack([full]);
  };

  const cur = intOf(v, t);
  const top = stack[stack.length - 1] ?? full;
  const tMarks = marks.length > 0;
  const zoomX = t.cols / Math.max(1e-6, v.c1 - v.c0);
  const layerMaps = t.layer === null ? [] : maps.map((m) => ({ file: m.file, v: m.layers.find((x) => x.layer === t.layer)?.value })).filter((m): m is { file: string; v: number } => typeof m.v === "number");
  const hv = hover && g ? cellOf(g, hover.r, hover.c) : null;
  const hn = hv ? hv.i * g!.w + hv.j : -1;
  return (
    <div className="grid min-w-0 gap-2" data-e2e="param-overview-primary" data-tensor={t.name}>
      <TensorHeader t={t} whole={whole} mark={pick && pick.reg.r1 - pick.reg.r0 === 1 && pick.reg.c1 - pick.reg.c0 === 1 ? (pick.value ?? null) : null} />
      {layerMaps.length > 0 && (
        <div className="mono flex flex-wrap items-center gap-x-3 gap-y-1 border-l-2 border-sense-pale pl-2.5 text-[10px] text-faint" data-e2e="param-layer-maps">
          <span className="uppercase tracking-[0.1em] text-muted">layer {t.layer} maps</span>
          {layerMaps.map((m) => (
            <span key={m.file} className="text-text" title={`${m.file} at layer ${t.layer}`}>
              <span className="text-faint">{m.file.replace(/\.json$/, "")}</span> {fmt(m.v)}
            </span>
          ))}
        </div>
      )}
      {!t.readable ? (
        <p className="mono text-[10px] text-quiet">{t.unread ?? "values not held here"}</p>
      ) : (
        <div className="grid min-w-0 gap-2">
          <div className="grid min-w-0 gap-1.5">
            <div className="mono flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] text-faint" style={{ paddingLeft: RULER_L }} data-e2e="param-zoom-bar">
              <span className="text-quiet">zoom</span>
              <button type="button" className={btn} disabled={zoomX <= 1.01} onClick={() => zoomBy(0.5)} title="Zoom out 2x (-)">
                -
              </button>
              <span className="min-w-[3.4rem] text-center text-text" title="Current zoom factor">
                {zoomX < 10 ? `${zoomX.toFixed(1)}x` : `${Math.round(zoomX)}x`}
                {g?.exact ? " · 1:1" : ""}
              </span>
              <button type="button" className={btn} disabled={kx >= CELL_PX_MAX * 0.98} onClick={() => zoomBy(2)} title="Zoom in 2x (+)">
                +
              </button>
              <button type="button" className={g?.exact ? on : btn} onClick={zoomExact} title="Jump straight to individual weight cells with numeric values">
                1:1 exact
              </button>
              <button type="button" className={btn} disabled={stack.length === 1 && same(cur, full)} onClick={reset} title="Show the whole tensor (0)">
                fit
              </button>
              <span aria-hidden className="text-line-strong">|</span>
              <span className="text-quiet">drag</span>
              <button type="button" className={tool === "pan" ? on : btn} onClick={() => setTool("pan")} title="Drag pans the view; Shift+drag boxes a region">
                pan
              </button>
              <button type="button" className={tool === "box" ? on : btn} onClick={() => setTool("box")} title="Drag draws a zoom and selection box directly">
                box zoom
              </button>
              <span aria-hidden className="text-line-strong">|</span>
              <button
                type="button"
                className={inspectorOpen ? on : btn}
                onClick={onToggleInspector}
                title={inspectorOpen ? "Hide floating Parameter Inspector" : "Open floating Parameter Inspector"}
                data-e2e="param-inspector-toggle"
              >
                {inspectorOpen ? "inspector · open" : pick ? "open inspector" : "inspector"}
              </button>
              {tMarks && (
                <>
                  <span aria-hidden className="text-line-strong">|</span>
                  <span className="text-quiet">masks</span>
                  <button type="button" className={overlayMode === "show" ? on : btn} onClick={() => setOverlayMode("show")}>
                    on
                  </button>
                  <button type="button" className={overlayMode === "boost" ? on : btn} onClick={() => setOverlayMode("boost")} title="Dim the heatmap so masks, freezes and edits pop clearly">
                    emphasize
                  </button>
                  <button type="button" className={overlayMode === "off" ? on : btn} onClick={() => setOverlayMode("off")}>
                    hide
                  </button>
                </>
              )}
              {hasMapVisual && mapOverlay && (
                <>
                  <span aria-hidden className="text-line-strong">|</span>
                  <span className="text-quiet">map</span>
                  <button type="button" className={mapMode === "show" ? on : btn} onClick={() => setMapMode("show")}>
                    on
                  </button>
                  <button type="button" className={mapMode === "boost" ? on : btn} onClick={() => setMapMode("boost")} title="Dim the heatmap so the overlaid parameter map stands out clearly">
                    emphasize
                  </button>
                  <button type="button" className={mapMode === "off" ? on : btn} onClick={() => setMapMode("off")}>
                    hide
                  </button>
                </>
              )}
            </div>
            <div className="overflow-x-auto pb-1">
              <div className="grid" style={{ gridTemplateColumns: `${RULER_L}px ${cw}px ${LEGEND + LEGEND_GAP}px`, gridTemplateRows: `${RULER_T}px ${ch}px`, columnGap: 0 }}>
                <span />
                <Ruler axis="cols" from={v.c0} to={v.c1} px={cw} length={t.cols} />
                <span />
                {t.rows > 1 ? <Ruler axis="rows" from={v.r0} to={v.r1} px={ch} length={t.rows} /> : <span className="mono self-center pr-2 text-right text-[9px] text-quiet">row 0</span>}
                <div className="relative border border-line-strong" style={{ width: cw, height: ch }}>
                  {!g && !q.error && !q.data?.error && (
                    <div role="status" aria-label="loading region" data-loading="region" className="placeholder absolute inset-0">
                      <span className="placeholder-label mono">loading region</span>
                    </div>
                  )}
                  <canvas
                    ref={canvas}
                    tabIndex={0}
                    className={`absolute inset-0 touch-none outline-none focus-visible:ring-1 focus-visible:ring-sense-pale ${tool === "box" ? "cursor-crosshair" : grabbing ? "cursor-grabbing" : "cursor-grab"}`}
                    style={{ width: cw, height: ch }}
                    onKeyDown={(e) => {
                      if (e.key === "+" || e.key === "=") { e.preventDefault(); zoomBy(2); }
                      else if (e.key === "-") { e.preventDefault(); zoomBy(0.5); }
                      else if (e.key === "0") { e.preventDefault(); reset(); }
                    }}
                    onPointerDown={down}
                    onPointerMove={move}
                    onPointerUp={up}
                    onPointerCancel={() => {
                      press.current = null;
                      setBox(null);
                      setGrabbing(false);
                    }}
                    onPointerLeave={() => setHover(null)}
                    onDoubleClick={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      const x = Math.max(0, Math.min(cw - 1e-6, e.clientX - rect.left));
                      const y = Math.max(0, Math.min(ch - 1e-6, e.clientY - rect.top));
                      const c = Math.min(t.cols - 1, Math.floor(v.c0 + x / kx));
                      const r = t.rows === 1 ? 0 : Math.min(t.rows - 1, Math.floor(v.r0 + y / ky));
                      const next = inspect(r, c);
                      onSelect?.(pathOf(t, next.reg));
                      onOpenInspector();
                      setEditFocusTick((n) => n + 1);
                    }}
                    aria-label={`${t.name} ${label(cur, t)} heatmap`}
                    data-e2e="param-overview-canvas"
                    data-primary="1"
                    data-exact={g?.exact ? "1" : undefined}
                  />
                  {box && (
                    <span
                      aria-hidden
                      className="pointer-events-none absolute border-2 border-signal bg-signal/15"
                      style={{ left: Math.min(box.x0, box.x1), top: Math.min(box.y0, box.y1), width: Math.abs(box.x1 - box.x0), height: Math.abs(box.y1 - box.y0) }}
                    />
                  )}
                  {q.stale && <span className="mono pointer-events-none absolute top-1 right-1 bg-bg/80 px-1 text-[9px] text-quiet">reading</span>}
                </div>
                <div style={{ paddingLeft: LEGEND_GAP }}>{g && <Legend lut={lut} s={s} kind={scale} height={ch} min={ext.min} max={ext.max} stat={stat} />}</div>
              </div>
            </div>
            {(q.error || q.data?.error) && <p className="mono text-[10px] text-error">{q.error ?? q.data?.error}</p>}
            <div className="mono grid gap-0.5 text-[9.5px] text-faint" style={{ paddingLeft: RULER_L }}>
              <span className="min-h-[1.2em] text-muted">
                {hover && g && hv ? (
                  g.exact ? (
                    <>
                      row {hover.r} · col {hover.c} · index {(hover.r * t.cols + hover.c).toLocaleString()} · value {String(g.values?.[hn] ?? "non-finite")}
                    </>
                  ) : (
                    <>
                      row {hover.r} · col {hover.c} · cell min {fmt(g.min?.[hn])} · max {fmt(g.max?.[hn])} · mean {fmt(g.mean?.[hn])} · rms {fmt(g.rms?.[hn])}
                    </>
                  )
                ) : g ? (
                  `view ${label(cur, t)} · min ${fmt(g.summary.min)} · max ${fmt(g.summary.max)} · mean ${fmt(g.summary.mean)} · rms ${fmt(g.summary.rms)}`
                ) : (
                  "reading"
                )}
              </span>
              {g && (
                <span>
                  {g.w} x {g.h} cells · cell {fmt(g.cell.rows)} x {fmt(g.cell.cols)} weights{g.exact ? " · exact values" : ""} · {Math.max(0.01, kx).toPrecision(2)} px a weight{printing ? ` · printed values ${unit ? `in units of 1e${unit}` : "as is"}` : ""} · read {fmt(g.read.bytes / 1048576)} MiB in {g.read.ms} ms
                </span>
              )}
              {tMarks && (
                <span className="flex flex-wrap items-center gap-x-3 text-muted" data-e2e="param-overlay-key">
                  <span className="text-quiet">overlay:</span>
                  <span className="flex items-center gap-1">
                    <i aria-hidden className="inline-block h-2.5 w-3.5 border border-[#ffc466] bg-[#120c04]" style={{ backgroundImage: "repeating-linear-gradient(45deg, rgba(255,184,84,0.75) 0 1px, transparent 1px 4px)" }} /> masked
                  </span>
                  <span className="flex items-center gap-1">
                    <i aria-hidden className="inline-block h-2.5 w-3.5 border border-dashed border-[#9bdcff] bg-[#6ebeff]/25" style={{ backgroundImage: "repeating-linear-gradient(45deg, rgba(155,220,255,0.85) 0 1px, transparent 1px 4px)" }} /> frozen
                  </span>
                  <span className="flex items-center gap-1">
                    <i aria-hidden className="inline-block h-2.5 w-3.5 border border-verified bg-verified/20" /> edited
                  </span>
                  <span className="text-quiet">({marks.length} active)</span>
                </span>
              )}
            </div>
            <div className="mono flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] text-faint" style={{ paddingLeft: RULER_L }} data-e2e="param-overview-zoom">
              {stack.map((r, idx) => (
                <span key={idx} className="flex items-center gap-1">
                  {idx > 0 && <span aria-hidden>/</span>}
                  <button
                    type="button"
                    className={idx === stack.length - 1 && same(cur, r) ? on : btn}
                    onClick={() => {
                      setView(r);
                      setStack(stack.slice(0, idx + 1));
                    }}
                    title={pathOf(t, r)}
                  >
                    {idx === 0 ? "whole" : label(r, t)}
                  </button>
                </span>
              ))}
              {!same(cur, top) && (
                <span className="flex items-center gap-1">
                  <span aria-hidden>/</span>
                  <span className={on}>{label(cur, t)}</span>
                </span>
              )}
              <button type="button" className={btn} disabled={stack.length === 1 && same(cur, full)} onClick={reset}>
                reset
              </button>
              <button type="button" className={btn} onClick={() => onSelect?.(pathOf(t, cur))} disabled={!onSelect} title={pathOf(t, cur)}>
                select view
              </button>
              <button type="button" className={list ? on : btn} onClick={() => setList(!list)}>
                values
              </button>
              <span className="text-quiet">wheel zoom · drag pan · shift drag box · double click edit</span>
            </div>
            {list && <Values key={pathOf(t, cur)} k={k} t={t} g={cur} />}
          </div>
          <Inspector
            key={t.name}
            k={k}
            t={t}
            pick={pick}
            whole={whole}
            maps={maps}
            mapOverlay={mapOverlay}
            editFocusTick={editFocusTick}
            open={inspectorOpen}
            minimized={inspectorMin}
            win={inspectorWin}
            onWinChange={onInspectorWinChange}
            onResetWin={onResetInspectorWin}
            onToggleMin={onToggleInspectorMin}
            onClose={onCloseInspector}
            onSelect={onSelect}
            onZoomToBlock={(r) => zoomTo(r, true)}
          />
        </div>
      )}
    </div>
  );
}

/* ---------- a thumbnail ---------- */

function Thumb({ k, t, stat, scale, theme, primary, mapBadge, onPick, onWhole }: { k: string; t: OverviewTensor; stat: Stat; scale: Scale; theme: Theme; primary: boolean; mapBadge?: string | null; onPick: (name: string) => void; onWhole: (g: OverviewGrid) => void }) {
  const root = useRef<HTMLButtonElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [seen, setSeen] = useState(false);
  const w = THUMB;
  const h = t.rows === 1 ? 14 : Math.max(24, Math.min(THUMB, Math.round((THUMB * t.rows) / t.cols)));
  const cells = cellsOf(wholeOf(t), { w, h });
  // The whole tensor is read for the thumbnail; its distribution comes in the same pass.
  const q = useRead<OverviewGrid>(seen && t.readable ? gridPath(k, t, wholeOf(t), cells, true) : null);
  const g = q.data && !q.data.error ? q.data : null;
  const luts = useLuts(theme);
  const values = useMemo(() => (g ? cellValues(g, stat) : []), [g, stat]);
  const s = useMemo(() => scaleOf(values), [values]);

  useEffect(() => {
    const el = root.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (es) => {
        if (es.some((e) => e.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  useEffect(() => {
    if (g) onWhole(g);
  }, [g, onWhole]);
  useEffect(() => {
    const c = canvas.current;
    if (!c || !g) return;
    const off = paint(g, values, luts[scale], s, scale);
    c.width = g.w;
    c.height = g.h;
    const ctx = c.getContext("2d");
    if (ctx && off) ctx.drawImage(off, 0, 0);
  }, [g, values, luts, s, scale]);

  const shape = t.shape.length === 0 ? "scalar" : t.shape.join(" x ");
  return (
    <button
      ref={root}
      type="button"
      onClick={() => onPick(t.name)}
      disabled={!t.readable}
      aria-pressed={primary}
      className={`press grid min-w-0 grid-cols-[minmax(0,1fr)] content-start gap-1 border-t pt-1 text-left ${primary ? "border-sense-pale" : "border-line hover:border-line-3"}`}
      data-e2e="param-overview-tensor"
      data-tensor={t.name}
      title={`${t.name} · ${shape} · ${t.dtype}${t.readable ? " · click to view" : ""}`}
    >
      <span className={`mono flex items-center justify-between gap-1 text-[9.5px] ${primary ? "text-text" : "text-muted"}`}>
        <span className="truncate">{t.name.replace(/^model\.layers\.\d+\./, "").replace(/\.weight$/, "")}</span>
        {mapBadge && <span className={`shrink-0 text-[8.5px] ${mapBadge.startsWith("★") ? "text-[#ffd452]" : "text-sense-pale"}`}>{mapBadge}</span>}
      </span>
      {!t.readable ? (
        <span className="mono text-[9px] text-quiet">{t.unread ?? "values not held here"}</span>
      ) : (
        <span className="relative block" style={{ width: w, height: h, maxWidth: "100%" }}>
          {!g && !q.error && !q.data?.error && (
            <span role="status" aria-label={`loading ${t.name}`} data-loading="tensor" className="placeholder absolute inset-0">
              <span className="placeholder-label mono">{seen ? "loading" : "waiting to scroll in"}</span>
            </span>
          )}
          <canvas ref={canvas} className="absolute inset-0 h-full w-full" style={{ imageRendering: "pixelated", visibility: g ? "visible" : "hidden" }} aria-label={`${t.name} heatmap`} data-e2e="param-overview-canvas" />
        </span>
      )}
      <span className="mono text-[9px] text-faint">
        {shape}
        {g ? ` · ${fmt(g.summary.min)} to ${fmt(g.summary.max)}` : ""}
        {q.error || q.data?.error ? <span className="text-error"> · {q.error ?? q.data?.error}</span> : null}
      </span>
    </button>
  );
}

/* ---------- the set ---------- */

type Group = { layer: number | null; parts: { part: string; tensors: OverviewTensor[] }[] };

/** Consecutive tensors of one layer form a group, and inside it one row per block. */
function groupsOf(ts: OverviewTensor[]): Group[] {
  const out: Group[] = [];
  for (const t of ts) {
    let g = out[out.length - 1];
    if (!g || g.layer !== t.layer) {
      g = { layer: t.layer, parts: [] };
      out.push(g);
    }
    let p = g.parts.find((x) => x.part === t.part);
    if (!p) {
      p = { part: t.part, tensors: [] };
      g.parts.push(p);
    }
    p.tensors.push(t);
  }
  return out;
}

/** The first tensor worth a large view: a readable matrix inside a layer, else any readable one. */
function firstPrimary(ts: OverviewTensor[]): string | null {
  return (ts.find((t) => t.readable && t.rows > 1 && t.layer !== null) ?? ts.find((t) => t.readable))?.name ?? null;
}

/** The whole-tensor read of the primary tensor (its statistics and distribution), unless a thumbnail
 *  has brought it already. One cell, so the reply is small; the tensor is streamed once. */
function useWhole(k: string, t: OverviewTensor | null, held: OverviewGrid | undefined): OverviewGrid | null {
  const q = useRead<OverviewGrid>(t && t.readable && !held ? gridPath(k, t, wholeOf(t), { w: 1, h: 1 }, true) : null);
  if (held) return held;
  return q.data && !q.data.error && !q.stale ? q.data : null;
}

const KIND_LABEL: Record<string, string> = {
  base: "base cartridge",
  open: "open weights (upstream)",
  filter: "filter cartridge",
  adapter: "adapter cartridge",
  lens: "lens adapter",
  readout: "read-out adapter",
  unit: "learning unit",
  preference: "preference cartridge",
};

export default function ParamOverview({
  objectKey,
  onSelect,
  marks,
  maps,
  mapOverlay,
  layerFilter = null,
  onSelectLayer,
  target,
}: {
  objectKey: string;
  onSelect?: (path: string) => void;
  marks?: OverviewMark[];
  maps?: OverviewMap[];
  mapOverlay?: ActiveMapOverlay | null;
  layerFilter?: number | null;
  onSelectLayer?: (layer: number | null) => void;
  target?: { path: string; n: number } | null;
}) {
  const root = useRef<HTMLDivElement>(null);
  const primaryRef = useRef<HTMLDivElement>(null);
  const [stat, setStat] = useState<Stat>("mean");
  const [scale, setScale] = useState<Scale>("diverging");
  const [width, setWidth] = useState(0);
  const [theme] = useState<Theme>(readTheme);
  const [chosen, setChosen] = useState<string | null>(null);
  const [wholes, setWholes] = useState<Record<string, OverviewGrid>>({});
  const [focus, setFocus] = useState<{ rect: Rect; n: number } | null>(null);
  const [seenTarget, setSeenTarget] = useState<number | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [inspectorMin, setInspectorMin] = useState(false);
  const [inspectorWin, setInspectorWin] = useState<WinRect>(defaultWin);
  const q = useRead<SetReply>(`${ROUTE}?key=${enc(objectKey)}`);
  const set = q.data;
  const tensors = useMemo(() => set?.tensors ?? [], [set]);
  const availLayers = useMemo(() => {
    const ls = [...new Set(tensors.map((t) => t.layer).filter((l): l is number => l !== null))];
    return ls.sort((a, b) => a - b);
  }, [tensors]);
  const visibleTensors = useMemo(
    () => (layerFilter === null ? tensors : tensors.filter((t) => t.layer === layerFilter)),
    [tensors, layerFilter],
  );
  const groups = useMemo(() => groupsOf(visibleTensors), [visibleTensors]);
  // Few tensors per layer (a job filter's gain and bias): the layers stand side by side in columns
  // instead of one tall list, so the whole set is seen at once.
  const perLayer = Math.max(0, ...groups.map((g) => g.parts.reduce((a, p) => a + p.tensors.length, 0)));
  const compact = groups.length > 3 && perLayer <= 3;
  const chosenVisible = chosen && visibleTensors.some((t) => t.name === chosen) ? chosen : null;
  const primaryName =
    chosenVisible ??
    (mapOverlay
      ? visibleTensors.find((t) => t.readable && (mapOverlay.pointsByTensor[t.name]?.length ?? 0) > 0)?.name ??
        visibleTensors.find((t) => t.readable && t.layer !== null && mapOverlay.layers.includes(t.layer) && t.rows > 1)?.name ??
        firstPrimary(visibleTensors)
      : firstPrimary(visibleTensors));
  const primary = visibleTensors.find((t) => t.name === primaryName) ?? null;
  const badgeFor = (t: OverviewTensor): string | null => {
    if (!mapOverlay) return null;
    const pts = mapOverlay.pointsByTensor[t.name]?.length ?? 0;
    if (pts > 0) return `★ ${pts}`;
    if (t.layer !== null && mapOverlay.layers.includes(t.layer)) return "map";
    return null;
  };

  // A target from the controls (an overlay entry picked there) makes its tensor primary and zooms to it.
  if (target && target.n !== seenTarget && tensors.length > 0) {
    setSeenTarget(target.n);
    const hit = rectsOf(target.path, tensors);
    if (hit && hit.tensor.rows * hit.tensor.cols > 0) {
      const b = hit.rects.reduce((a, r) => ({ r0: Math.min(a.r0, r.r0), r1: Math.max(a.r1, r.r1), c0: Math.min(a.c0, r.c0), c1: Math.max(a.c1, r.c1) }));
      setChosen(hit.tensor.name);
      setFocus({ rect: b, n: target.n });
      if (b.r1 - b.r0 === 1 && b.c1 - b.c0 === 1) {
        setInspectorOpen(true);
        setInspectorMin(false);
      }
    }
  }

  useEffect(() => {
    const el = root.current;
    if (!el) return;
    if (el.clientWidth > 0) setWidth(Math.floor(el.clientWidth));
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((es) => {
      const next = Math.floor(es[0]?.contentRect.width ?? el.clientWidth ?? 0);
      if (next > 0) setWidth(next);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    if (chosen) primaryRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [chosen, focus]);

  // Only the statistics and distribution are kept of a thumbnail's read, not its cells.
  const onWhole = useMemo(() => (g: OverviewGrid) => setWholes((w) => (w[g.tensor] ? w : { ...w, [g.tensor]: { ...g, values: undefined, min: undefined, max: undefined, mean: undefined, rms: undefined } })), []);
  const whole = useWhole(objectKey, primary, primary ? wholes[primary.name] : undefined);
  const markRects = useMemo<MarkRect[]>(() => {
    if (!primary) return [];
    const out: MarkRect[] = [];
    for (const m of marks ?? []) {
      if (m.whole) {
        out.push({ action: m.action, rect: wholeOf(primary), id: m.id });
        continue;
      }
      for (const r of m.regions) {
        if (r.tensor !== primary.name && !r.path.startsWith(`${primary.name}[`) && r.path !== primary.name) continue;
        const hit = rectsOf(r.path, [primary]);
        for (const rect of hit?.rects ?? []) out.push({ action: m.action, rect, id: m.id });
      }
    }
    return out;
  }, [marks, primary]);

  return (
    <div ref={root} className="grid min-w-0 gap-4" data-e2e="param-overview">
      <div className="mono flex flex-wrap items-center justify-between gap-2 text-[10px] text-faint">
        <span>
          {set && !set.error ? (
            <>
              <span className="text-text">{set.name}</span> · {set.kind ? (KIND_LABEL[set.kind] ?? set.kind) : ""} · {set.count} tensors · {set.weights?.toLocaleString()} weights
            </>
          ) : (
            objectKey
          )}
        </span>
        <span className="flex flex-wrap items-center gap-3">
          <span className="flex items-center gap-1" role="radiogroup" aria-label="cell statistic">
            <span className="text-quiet">cell</span>
            {STATS.map((s) => (
              <button key={s} type="button" role="radio" aria-checked={stat === s} className={stat === s ? on : btn} onClick={() => setStat(s)}>
                {s}
              </button>
            ))}
          </span>
          <span className="flex items-center gap-1" role="radiogroup" aria-label="colour scale" data-e2e="param-scale">
            <span className="text-quiet">scale</span>
            <button type="button" role="radio" aria-checked={scale === "diverging"} className={scale === "diverging" ? on : btn} onClick={() => setScale("diverging")} title="Signed values: blue below 0, magenta above, dark at 0">
              signed
            </button>
            <button type="button" role="radio" aria-checked={scale === "sequential"} className={scale === "sequential" ? on : btn} onClick={() => setScale("sequential")} title="Magnitude: |value|, dark at 0">
              magnitude
            </button>
          </span>
        </span>
      </div>
      {q.error || set?.error ? (
        <p className="mono text-[10.5px] text-error">{q.error ?? set?.error}</p>
      ) : !set ? (
        <div role="status" aria-label="loading tensors" data-loading="tensors" className="placeholder relative h-24">
          <span className="placeholder-label mono">loading tensors</span>
        </div>
      ) : (
        <>
          {availLayers.length > 0 && onSelectLayer && (
            <div className="mono flex flex-wrap items-center gap-1 border-y border-line/60 py-1.5 text-[9.5px]" data-e2e="param-overview-layers">
              <span className="mr-1 uppercase tracking-[0.12em] text-quiet">layer</span>
              <button
                type="button"
                onClick={() => {
                  setChosen(null);
                  onSelectLayer(null);
                }}
                className={`press border px-1.5 py-0.5 ${layerFilter === null ? "border-sense-pale bg-bg-2 text-text" : "border-line text-faint hover:text-text"}`}
              >
                all ({tensors.length})
              </button>
              {availLayers.map((l) => {
                const active = layerFilter === l;
                return (
                  <button
                    key={l}
                    type="button"
                    onClick={() => {
                      setChosen(null);
                      onSelectLayer(active ? null : l);
                    }}
                    className={`press border px-1.5 py-0.5 ${active ? "border-sense-pale bg-sense/20 text-text" : "border-line text-faint hover:border-line-3 hover:text-text"}`}
                    title={active ? `Showing layer ${l} only (click for all layers)` : `Show only layer ${l} in the Parameter Viewer`}
                  >
                    L{l}
                  </button>
                );
              })}
            </div>
          )}
          <div ref={primaryRef} className="scroll-mt-4">
            {primary && width > 0 && (
              <PrimaryView
                key={primary.name}
                k={objectKey}
                t={primary}
                stat={stat}
                scale={scale}
                theme={theme}
                avail={width}
                whole={whole}
                marks={markRects}
                maps={maps ?? []}
                mapOverlay={mapOverlay}
                focus={focus && chosen === primary.name ? focus : null}
                inspectorOpen={inspectorOpen}
                inspectorMin={inspectorMin}
                inspectorWin={inspectorWin}
                onInspectorWinChange={setInspectorWin}
                onResetInspectorWin={() => {
                  setInspectorWin(defaultWin());
                  setInspectorMin(false);
                }}
                onOpenInspector={() => {
                  setInspectorOpen(true);
                  setInspectorMin(false);
                }}
                onToggleInspector={() => {
                  setInspectorOpen((o) => !o);
                  setInspectorMin(false);
                }}
                onToggleInspectorMin={() => setInspectorMin((m) => !m)}
                onCloseInspector={() => setInspectorOpen(false)}
                onSelect={onSelect}
              />
            )}
          </div>
          <div className="grid gap-3 border-t border-line-strong pt-3" data-e2e="param-overview-set">
            <span className="mono text-[9.5px] uppercase tracking-[0.12em] text-quiet">
              {layerFilter !== null ? `layer ${layerFilter} tensors (${visibleTensors.length}) · click one to view it above` : "every tensor · click one to view it above"}
            </span>
            {compact && (
              <div className="grid gap-x-6 gap-y-2.5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(17rem, 1fr))" }} data-e2e="param-overview-compact">
                {groups.map((g, gi) => (
                  <section key={`${gi}-${g.layer ?? "none"}`} className="grid grid-cols-[2.75rem_minmax(0,1fr)] items-start gap-2 border-t border-line/60 pt-1.5">
                    <h3 className="mono pt-0.5 text-[10px] uppercase tracking-[0.12em] text-muted" title={g.layer === null ? "outside layers" : `layer ${g.layer}`}>
                      {g.layer === null ? "none" : `L${g.layer}`}
                    </h3>
                    <div className="grid min-w-0 gap-2" style={{ gridTemplateColumns: `repeat(${Math.max(1, perLayer)}, minmax(0, 1fr))` }}>
                      {g.parts.flatMap((p) =>
                        p.tensors.map((t) => (
                          <Thumb key={t.name} k={objectKey} t={t} stat={stat} scale={scale} theme={theme} primary={t.name === primaryName} mapBadge={badgeFor(t)} onPick={(n) => { setChosen(n); setFocus(null); }} onWhole={onWhole} />
                        )),
                      )}
                    </div>
                  </section>
                ))}
              </div>
            )}
            {!compact && groups.map((g, gi) => (
              <section key={`${gi}-${g.layer ?? "none"}`} className="grid gap-1.5">
                <h3 className="mono text-[10px] uppercase tracking-[0.12em] text-muted">{g.layer === null ? "outside layers" : `layer ${g.layer}`}</h3>
                {g.parts.map((p) => (
                  <div key={p.part} className="grid grid-cols-[5.5rem_minmax(0,1fr)] items-start gap-2">
                    <span className="mono pt-1 text-[9px] uppercase tracking-[0.1em] text-faint">{p.part}</span>
                    <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${THUMB}px, 1fr))` }}>
                      {p.tensors.map((t) => (
                        <Thumb key={t.name} k={objectKey} t={t} stat={stat} scale={scale} theme={theme} primary={t.name === primaryName} mapBadge={badgeFor(t)} onPick={(n) => { setChosen(n); setFocus(null); }} onWhole={onWhole} />
                      ))}
                    </div>
                  </div>
                ))}
              </section>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
