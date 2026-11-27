"use client";

/** Real activity per part, as the drawings' texture and the panels' figures.
 *
 *  Source: the engine's activity endpoint (`/api/activity?window=1h` and `?window=24h`, API
 *  version 3, `docs/interface-api.md`): per anatomy component its calls, its timed total, when it
 *  last ran, and where it applies its sub-parts (`sub`), its model layers (`layers`) and its
 *  tentacle jobs (`jobs`). Only when that endpoint does not answer are the audit's tool rows
 *  counted here instead (`/api/audit`), by the same rules; the cover line says which.
 *
 *  Nothing is estimated. A step with no time is counted as a call and its time stays "not
 *  timed". Where neither source answers, `reported` is false and every drawing stays dim, even
 *  and still. */
import { useEffect, useMemo, useRef, useState } from "react";
import { refresh, useApi, type Audit, type AuditRow } from "./api";
import { LIVE_PATH, mark, sortRows } from "./live";
import { slotOf } from "./livefeed";
import { keyOf } from "./scene";
import { resolvePart, type TNode, type Tree } from "./topology";

export const ACTIVITY_DAY = "/api/activity?window=24h";
export const ACTIVITY_HOUR = "/api/activity?window=1h";
export const AUDIT_WIDE = "/api/audit?kind=tool&limit=1000";
export const HOUR = 3_600_000;
export const DAY = 24 * HOUR;

/** One call's work in one part: when, how long (null when not timed), the layers it ran. Known
 *  only when the audit is counted here. */
export type Ev = { t: number; ms: number | null; layers?: [number, number] };

/** A sub-part's figures over the day (and the hour, when the engine reports them). */
export type SubAct = { id: string; calls_day: number; calls_hour: number; ms_day: number | null };

/** Activity of one drawn part (a tree key). */
export type KeyAct = {
  calls_hour: number;
  calls_day: number;
  /** Milliseconds over the hour and the day; null when no step of the part was timed. */
  ms_hour: number | null;
  ms_day: number | null;
  /** Median of each call's timed total over the day, from the engine. */
  median_ms?: number | null;
  last: number | null;
  /** Each call in the last day, when the audit is counted here; null from the engine. */
  events: Ev[] | null;
  /** Calls per model layer over the day, for a MicroModel. */
  layers: number[] | null;
  /** Calls per job the plan gave a tentacle, over the day. */
  jobs?: Record<string, number>;
  /** Per sub-part, by its engine id or cell name, where the engine reports it. */
  subs?: SubAct[];
};

export type ActView = {
  reported: boolean;
  source: "engine" | "audit" | null;
  /** What the figures cover, in plain words. */
  cover: string;
  now: number;
  /** Activity of each key and everything inside it. */
  total: Map<string, KeyAct>;
  /** Activity of ids that resolve to this key itself (not to a part inside it). */
  own: Map<string, KeyAct>;
  /** Activity of named cells that are not tree parts: `tier:core` and so on, from memory's
   *  `read:<tier>` and `write:<tier>` cells. */
  cells: Map<string, KeyAct>;
  maxDay: number;
  maxMs: number;
  maxLayer: number;
  /** The newest call: when it ended, the keys it touched and the layers it ran per key. */
  last: { t: number; keys: Set<string>; layers: Map<string, [number, number][]> } | null;
  sig: string;
};

export const NO_ACT: ActView = { reported: false, source: null, cover: "activity not reported by the engine", now: 0, total: new Map(), own: new Map(), cells: new Map(), maxDay: 0, maxMs: 0, maxLayer: 0, last: null, sig: "none" };

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);

function blank(events: boolean): KeyAct {
  return { calls_hour: 0, calls_day: 0, ms_hour: null, ms_day: null, last: null, events: events ? [] : null, layers: null };
}

type StepLike = { ms?: unknown; us?: unknown; timed?: unknown };
/** A step's time in milliseconds (from `us` when the engine measured that finely), or null when
 *  the engine did not time it. */
export function stepMs(st: StepLike): number | null {
  if (st.timed === false) return null;
  const us = num(st.us);
  if (us !== null) return us / 1000;
  return num(st.ms);
}

/** A time as shown: "not timed" when absent; below a millisecond in microseconds. */
export function msText(ms: number | null | undefined, timed?: boolean): string {
  if (timed === false || ms === null || ms === undefined) return "not timed";
  if (ms < 1) return ms > 0 ? `${Math.max(1, Math.round(ms * 1000))} µs` : "under 1 ms";
  if (ms >= 10_000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms < 10) return `${ms.toFixed(1)} ms`;
  return `${Math.round(ms)} ms`;
}

/** A tier named by a memory cell (`read:active`, `write:core`), or null. */
export function tierOfCell(cell: string): string | null {
  const m = /^(?:read|write):(core|active|archive|temporary)$/.exec(cell);
  return m ? m[1] : null;
}

/* ---------------- Engine source ---------------- */

type Entry = { calls: number; total_ms: number | null; median_ms: number | null; last_ms: number | null; layers: Record<string, Entry> | null; jobs: Record<string, number> | null; sub: Record<string, Entry> | null };
type Reply = { components: Record<string, Entry>; parts: Record<string, Entry> };

function entry(v: unknown): Entry | null {
  const o = obj(v);
  if (!o) return null;
  const map = (x: unknown) => {
    const m = obj(x);
    if (!m) return null;
    const out: Record<string, Entry> = {};
    for (const [k, e] of Object.entries(m)) {
      const r = entry(e);
      if (r) out[k] = r;
    }
    return out;
  };
  const jobs = obj(o.jobs);
  return {
    calls: num(o.calls) ?? 0,
    total_ms: num(o.total_ms),
    median_ms: num(o.median_ms),
    last_ms: num(o.last_ms),
    layers: map(o.layers),
    jobs: jobs ? (Object.fromEntries(Object.entries(jobs).filter(([, n]) => typeof n === "number")) as Record<string, number>) : null,
    sub: map(o.sub),
  };
}

/** Reads one reply of `/api/activity`; null when it is not one. */
function readReply(j: unknown): Reply | null {
  const o = obj(j);
  const c = obj(o?.components);
  if (!o || !c) return null;
  const read = (m: Record<string, unknown> | null) => {
    const out: Record<string, Entry> = {};
    for (const [k, v] of Object.entries(m ?? {})) {
      const e = entry(v);
      if (e) out[k] = e;
    }
    return out;
  };
  return { components: read(c), parts: read(obj(o.parts)) };
}

/* ---------------- Building the view ---------------- */

function addTo(m: Map<string, KeyAct>, key: string, f: (a: KeyAct) => void, events: boolean) {
  let a = m.get(key);
  if (!a) m.set(key, (a = blank(events)));
  f(a);
}

const sumMs = (a: number | null, b: number | null) => (b === null ? a : (a ?? 0) + b);

function addLayers(a: KeyAct, l: [number, number]) {
  a.layers ??= [];
  for (let i = Math.max(0, l[0]); i <= Math.min(127, l[1]); i++) a.layers[i] = (a.layers[i] ?? 0) + 1;
}

type Partial0 = Omit<ActView, "sig" | "maxDay" | "maxMs" | "maxLayer">;

/** The newest call's touched parts and layers, from its audit row (for the fading light). */
function lastOf(tree: Tree, row: AuditRow | null): ActView["last"] {
  if (!row || row.state === "running") return null;
  const keys = new Set<string>();
  const layers = new Map<string, [number, number][]>();
  let end = row.time_ms;
  for (const st of row.steps ?? []) {
    end = Math.max(end, row.time_ms + (num(st.start_ms) ?? 0) + (stepMs(st) ?? 0));
    for (const n of resolvePart(tree, st.part))
      for (let x: TNode | undefined = n; x && x.path.length; x = x.parent) {
        keys.add(keyOf(x));
        if (Array.isArray(st.layers)) layers.set(keyOf(x), [...(layers.get(keyOf(x)) ?? []), st.layers as [number, number]]);
      }
  }
  return keys.size ? { t: end, keys, layers } : null;
}

type AuditStep = { part: string; start_ms?: number; ms?: number; us?: number; timed?: boolean; shared?: string; inside?: string; layers?: [number, number]; cells?: unknown[] };

/** Counts the audit's tool rows per drawn part, when the engine's activity does not answer: one
 *  call per row for every part it touched and every part above, time from the steps' own clocks
 *  (a shared pass counted once per part, a step timed inside another left out of the sum). */
function fromAudit(tree: Tree, rows: AuditRow[], now: number, newest: AuditRow | null): Partial0 {
  const total = new Map<string, KeyAct>();
  const own = new Map<string, KeyAct>();
  const cells = new Map<string, KeyAct>();
  const list = sortRows(rows).filter((r) => r.state !== "running" && (r.kind === "tool" || !!r.tool) && typeof r.time_ms === "number");
  let oldest = Infinity;
  for (const r of list) {
    oldest = Math.min(oldest, r.time_ms);
    const steps: AuditStep[] = Array.isArray(r.steps) && r.steps.length ? (r.steps as AuditStep[]) : (r.parts ?? []).map((p) => ({ part: p }));
    const per = new Map<string, { ms: number | null; shared: Set<string>; layers: [number, number][]; end: number; own: boolean }>();
    const tiers = new Set<string>();
    for (const st of steps) {
      const nodes = resolvePart(tree, st.part);
      const ms = st.inside ? null : stepMs(st);
      const end = r.time_ms + (num(st.start_ms) ?? 0) + (stepMs(st) ?? 0);
      for (const c of Array.isArray(st.cells) ? st.cells : []) {
        const t = typeof c === "string" ? tierOfCell(c) : null;
        if (t) tiers.add(t);
      }
      for (const n of nodes) {
        for (let x: TNode | undefined = n; x && x.path.length; x = x.parent) {
          const k = keyOf(x);
          let e = per.get(k);
          if (!e) per.set(k, (e = { ms: null, shared: new Set(), layers: [], end, own: false }));
          if (x === n) e.own = true;
          e.end = Math.max(e.end, end);
          if (ms !== null && !(st.shared && e.shared.has(st.shared))) {
            e.ms = (e.ms ?? 0) + ms;
            if (st.shared) e.shared.add(st.shared);
          }
          if (Array.isArray(st.layers) && st.layers.length === 2) e.layers.push(st.layers);
        }
      }
    }
    const inDay = now - r.time_ms < DAY;
    const inHour = now - r.time_ms < HOUR;
    const count = (a: KeyAct, ms: number | null, end: number, layers: [number, number][]) => {
      a.last = Math.max(a.last ?? 0, end);
      if (!inDay) return;
      a.calls_day++;
      a.ms_day = sumMs(a.ms_day, ms);
      if (inHour) {
        a.calls_hour++;
        a.ms_hour = sumMs(a.ms_hour, ms);
      }
      a.events!.push({ t: r.time_ms, ms, layers: layers[0] });
      for (const l of layers) addLayers(a, l);
    };
    for (const [k, e] of per) {
      addTo(total, k, (a) => count(a, e.ms, e.end, e.layers), true);
      if (e.own) addTo(own, k, (a) => count(a, e.ms, e.end, e.layers), true);
    }
    for (const t of tiers) addTo(cells, `tier:${t}`, (a) => count(a, null, r.time_ms, []), true);
  }
  const since = Number.isFinite(oldest) ? new Date(oldest) : null;
  const cover = list.length
    ? `counted here from the newest ${list.length} calls in the audit${since && now - oldest < DAY ? ` (since ${since.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })})` : ""}, because the engine's activity does not answer`
    : "no call recorded in the audit";
  return { reported: true, source: "audit", cover, now, total, own, cells, last: lastOf(tree, newest) };
}

/** Places the engine's figures on the drawn parts. A component counts for the part it resolves
 *  to, and for each part above it as the busiest part inside (calls are not summed up the tree,
 *  since one call runs many parts). Raw parts that are not components count only where no
 *  component already stands for that part (the engine folds tentacle and drone instance parts
 *  into their component). Memory's tier cells light the strata. */
function fromEngine(tree: Tree, day: Reply, hour: Reply | null, now: number, newest: AuditRow | null): Partial0 {
  const total = new Map<string, KeyAct>();
  const own = new Map<string, KeyAct>();
  const cells = new Map<string, KeyAct>();
  const ownIds = new Set<string>();
  const layersOf = (e: Entry): number[] | null => {
    if (!e.layers) return null;
    const out: number[] = [];
    for (const [k, v] of Object.entries(e.layers)) if (/^\d+$/.test(k)) out[Number(k)] = v.calls;
    return out.length ? Array.from(out, (x) => x ?? 0) : null;
  };
  const actOf = (d: Entry, h: Entry | null): KeyAct => ({
    calls_day: d.calls,
    calls_hour: h?.calls ?? 0,
    ms_day: d.total_ms,
    ms_hour: h ? h.total_ms : null,
    median_ms: d.median_ms,
    last: d.last_ms,
    events: null,
    layers: layersOf(d),
    jobs: d.jobs ?? undefined,
    subs: d.sub ? Object.entries(d.sub).map(([id, s]) => ({ id, calls_day: s.calls, calls_hour: h?.sub?.[id]?.calls ?? 0, ms_day: s.total_ms })) : undefined,
  });
  const mergeUp = (a: KeyAct, b: KeyAct) => {
    a.calls_day = Math.max(a.calls_day, b.calls_day);
    a.calls_hour = Math.max(a.calls_hour, b.calls_hour);
    a.ms_day = sumMs(a.ms_day, b.ms_day);
    a.ms_hour = sumMs(a.ms_hour, b.ms_hour);
    if (b.last !== null) a.last = Math.max(a.last ?? 0, b.last);
    if (b.layers) {
      a.layers ??= [];
      b.layers.forEach((c, i) => (a.layers![i] = Math.max(a.layers![i] ?? 0, c)));
    }
  };
  const place = (id: string, k: KeyAct) => {
    for (const n of resolvePart(tree, id)) {
      const key = keyOf(n);
      const prev = own.get(key);
      const exact = n.ids.includes(id);
      if (prev) {
        // Two ids on one drawn part: the part's own id stands, else the busier; sub-parts of both
        // are kept.
        const subs = [...(prev.subs ?? []), ...(k.subs ?? [])];
        if (exact || (!ownIds.has(key) && k.calls_day > prev.calls_day)) own.set(key, { ...k, subs });
        else prev.subs = subs;
      } else own.set(key, { ...k });
      if (exact) ownIds.add(key);
      for (let x: TNode | undefined = n; x && x.path.length; x = x.parent) addTo(total, keyOf(x), (a) => mergeUp(a, k), false);
    }
  };
  for (const [id, d] of Object.entries(day.components)) place(id, actOf(d, hour?.components[id] ?? null));
  const folded = new Set<string>();
  for (const d of Object.values(day.components)) for (const s of Object.keys(d.sub ?? {})) folded.add(s);
  for (const [id, d] of Object.entries(day.parts)) {
    if (id in day.components || folded.has(id) || d.calls === 0) continue;
    if (resolvePart(tree, id).some((n) => (own.get(keyOf(n))?.calls_day ?? 0) > 0)) continue;
    place(id, actOf(d, hour?.parts[id] ?? null));
  }
  for (const [id, d] of Object.entries(day.components)) {
    for (const [c, s] of Object.entries(d.sub ?? {})) {
      const t = tierOfCell(c);
      if (!t) continue;
      const hs = hour?.components[id]?.sub?.[c] ?? null;
      addTo(
        cells,
        `tier:${t}`,
        (a) => {
          // A call that reads and writes the same tier counts once: the larger of the two stands.
          a.calls_day = Math.max(a.calls_day, s.calls);
          a.calls_hour = Math.max(a.calls_hour, hs?.calls ?? 0);
          a.ms_day = sumMs(a.ms_day, s.total_ms);
          if (s.last_ms !== null) a.last = Math.max(a.last ?? 0, s.last_ms);
          (a.subs ??= []).push({ id: c, calls_day: s.calls, calls_hour: hs?.calls ?? 0, ms_day: s.total_ms });
        },
        false,
      );
    }
  }
  return { reported: true, source: "engine", cover: "reported by the engine (activity over the last hour and day)", now, total, own, cells, last: lastOf(tree, newest) };
}

function finish(v: Partial0): ActView {
  let maxDay = 0;
  let maxMs = 0;
  let maxLayer = 0;
  const parts: string[] = [];
  for (const [k, a] of v.own) {
    maxDay = Math.max(maxDay, a.calls_day);
    for (const e of a.events ?? []) if (e.ms !== null) maxMs = Math.max(maxMs, e.ms);
    if (!a.events && a.ms_day !== null && a.calls_day) maxMs = Math.max(maxMs, a.ms_day / a.calls_day);
    for (const c of a.layers ?? []) maxLayer = Math.max(maxLayer, c ?? 0);
    parts.push(`${k}:${a.calls_hour}/${a.calls_day}/${a.ms_day ?? "u"}/${a.last ?? ""}`);
  }
  for (const [k, a] of v.total) parts.push(`T${k}:${a.calls_day}`);
  for (const [k, a] of v.cells) parts.push(`C${k}:${a.calls_hour}/${a.calls_day}`);
  const sig = `${v.source}|${Math.floor(v.now / (15 * 60_000))}|${v.last?.t ?? ""}|${parts.join(",")}`;
  return { ...v, maxDay, maxMs, maxLayer, sig };
}

/** The activity view for a tree. The engine's figures are read every 15 s (the engine reuses a
 *  result for 3 s while its log is unchanged) and at once when a new call arrives; the returned
 *  object stays the same while nothing it shows has changed, so nothing repaints while idle. */
export function useActivity(tree: Tree | null): ActView {
  const day = useApi<unknown>(ACTIVITY_DAY, 15000);
  const hour = useApi<unknown>(ACTIVITY_HOUR, 15000);
  const live = useApi<Audit>(LIVE_PATH, 5000);
  const dayReply = useMemo(() => (day.error ? null : readReply(day.data)), [day.data, day.error]);
  const hourReply = useMemo(() => (hour.error ? null : readReply(hour.data)), [hour.data, hour.error]);
  // The audit is read wide only when the engine's activity does not answer.
  const engineDown = !!day.error && !dayReply;
  const wide = useApi<Audit>(engineDown ? AUDIT_WIDE : ACTIVITY_DAY);
  const newest = useMemo(() => {
    const rows = sortRows((live.data?.audit ?? []) as AuditRow[]);
    return rows.length ? rows[rows.length - 1] : null;
  }, [live.data]);
  const newestMark = newest ? `${mark(newest)}:${newest.state}` : null;
  const seen = useRef<string | null>(null);
  useEffect(() => {
    if (newestMark === null) return;
    if (seen.current !== null && seen.current !== newestMark) {
      void refresh(ACTIVITY_DAY);
      void refresh(ACTIVITY_HOUR);
      if (engineDown) void refresh(AUDIT_WIDE);
    }
    seen.current = newestMark;
  }, [newestMark, engineDown]);
  // The time the figures are counted at: moves in steps of a minute while the page is open.
  const [now, setNow] = useState(0);
  useEffect(() => {
    const tick = () => setNow(Math.floor(Date.now() / 60_000) * 60_000);
    const first = window.setTimeout(tick, 0);
    const t = window.setInterval(tick, 60_000);
    return () => {
      clearTimeout(first);
      clearInterval(t);
    };
  }, []);
  const view = useMemo(() => {
    if (!tree || !now) return NO_ACT;
    if (dayReply) return finish(fromEngine(tree, dayReply, hourReply, now, newest));
    if (engineDown && wide.data && Array.isArray((wide.data as Audit).audit)) return finish(fromAudit(tree, (wide.data as Audit).audit as AuditRow[], now, newest));
    return NO_ACT;
  }, [tree, now, dayReply, hourReply, engineDown, wide.data, newest]);
  const [kept, setKept] = useState<ActView>(NO_ACT);
  if (kept.sig !== view.sig) setKept(view);
  return kept.sig === view.sig ? kept : view;
}

/** The activity to draw for a key: its own figures and those of everything inside it, else the
 *  figures of the nearest part above that the engine reports itself (a cell of a part the engine
 *  names only as a whole). */
export function actFor(v: ActView, key: string): KeyAct | null {
  const t = v.total.get(key);
  if (t) return t;
  const parts = key.split("/");
  for (let n = parts.length - 1; n >= 1; n--) {
    const o = v.own.get(parts.slice(0, n).join("/"));
    if (o) return o;
  }
  return null;
}

/** How brightly a cell of a part's drawing shows, as a factor on its rest brightness.
 *
 *  The cells of a part stand for the last day as a ring of time slices: cell `i` of `n` holds the
 *  slice that `floor(t / (DAY / n)) mod n` names, so a cell keeps its place as time passes and a
 *  call lights the one cell of its moment. A lit cell is brighter for more calls in its slice, for
 *  longer timed work, and for nearer time; an idle cell is dim. The engine gives counts, not each
 *  call's time: then each call of the day lights one cell, filled in a fixed even order, brighter
 *  when the part also ran in the last hour and for longer timed work per call.
 *  Without any source every cell keeps its rest brightness: even and still. */
export function cellFactors(v: ActView, a: KeyAct | null, n: number, fade: number, key: string): Float32Array {
  const out = new Float32Array(Math.max(1, n));
  if (!v.reported) return out.fill(1);
  if (!a || a.calls_day === 0) return out.fill(0.62);
  const lit = new Float32Array(out.length);
  if (a.events) {
    const slice = DAY / out.length;
    for (const e of a.events) {
      const i = ((Math.floor(e.t / slice) % out.length) + out.length) % out.length;
      const age = Math.min(1, Math.max(0, (v.now - e.t) / DAY));
      const w = 0.45 + 0.55 * (1 - age);
      const t = e.ms !== null && v.maxMs > 0 ? 0.6 + 0.4 * (Math.log1p(e.ms) / Math.log1p(v.maxMs)) : 0.6;
      lit[i] += w * t;
    }
  } else {
    const n = out.length;
    const litN = Math.min(n, a.calls_day);
    const perCall = a.ms_day !== null && a.calls_day ? a.ms_day / a.calls_day : null;
    const t = perCall !== null && v.maxMs > 0 ? 0.6 + 0.4 * (Math.log1p(perCall) / Math.log1p(v.maxMs)) : 0.6;
    const recent = a.calls_hour > 0 ? 1 : 0.55;
    // Cell j of the fill order is the cell at golden-ratio steps round the part, so lit cells
    // spread evenly whatever their number.
    for (let j = 0; j < litN; j++) lit[Math.floor(((j * 0.6180339887) % 1) * n)] += recent * t;
  }
  const extra = fade > 0 && v.last?.keys.has(key) ? 0.6 * fade : 0;
  for (let i = 0; i < out.length; i++) out[i] = 0.62 + 1.9 * Math.min(1, lit[i]) + extra;
  return out;
}

/** Brightness added to a MicroModel cell by its layer's work over the day (0 to 1), and by the
 *  last call when it ran that layer. */
export function layerLevel(v: ActView, a: KeyAct | null, layer: number, fade: number, key: string): number {
  if (!v.reported || !a) return 0;
  let lv = 0;
  if (a.layers && v.maxLayer > 0) lv = Math.sqrt((a.layers[layer] ?? 0) / v.maxLayer);
  else if (!a.layers && a.calls_day > 0 && v.maxDay) lv = 0.35 * Math.sqrt(a.calls_day / v.maxDay);
  if (fade > 0) for (const [l0, l1] of v.last?.layers.get(key) ?? []) if (layer >= l0 && layer <= l1) lv += 0.5 * fade;
  return Math.min(1.2, lv);
}

/** How much of the last call's light is left (0 to 1), in steps of a minute over fifteen
 *  minutes, so the still picture is painted again at most fifteen times after a call. */
export function useFade(v: ActView): number {
  const t = v.last?.t ?? null;
  const [step, setStep] = useState(0);
  useEffect(() => {
    if (t === null) return;
    const STEPS = 15;
    const at = () => Math.floor((Date.now() - t) / 60_000);
    let timer = 0;
    const run = () => {
      const s = at();
      setStep(s);
      if (s < STEPS) timer = window.setTimeout(run, 60_000 - ((Date.now() - t) % 60_000) + 50);
    };
    timer = window.setTimeout(run, 0);
    return () => clearTimeout(timer);
  }, [t]);
  if (t === null) return 0;
  return Math.max(0, 1 - step / 15);
}

/** Per-tier record counts of the Nano's own memory, from `/api/memory/tiers` (`nano`). Null
 *  when not reported. */
export type Tiers = { core?: number; active?: number; archive?: number; temporary?: number };
export function tiersOf(...sources: unknown[]): Tiers | null {
  for (const s of sources) {
    const o = obj(s);
    if (!o) continue;
    const t = obj(o.nano) ?? obj(o.tiers);
    if (!t) continue;
    const out: Tiers = {};
    let any = false;
    for (const k of ["core", "active", "archive", "temporary"] as const) {
      const v = num(t[k]) ?? num(obj(t[k])?.count) ?? num(obj(t[k])?.items);
      if (v !== null) {
        out[k] = v;
        any = true;
      }
    }
    if (any) return out;
  }
  return null;
}

/** Layer levels (0 to 1) of one released model over the last day, from the MicroModel slots that
 *  run on it: per layer the calls that ran it, against the busiest layer of any model. Null when
 *  activity is not reported; all zero when the model did not run. */
export function modelLayers(v: ActView, tree: Tree | null, modelId: string, layers: number, roles?: string[]): { levels: number[]; calls: number; ms: number | null; perLayer: boolean; last: number | null; hour: number } | null {
  if (!v.reported || !tree) return null;
  // 2026-11-26 (Luis Jake Gabriel III: "jsut one micromodel is shown lit up and not dynamically or
  // based on real activity"): the anatomy design names one card (qwen3-0.6b) on every slot, so
  // matching on it lit the Reader with every call. The slot a model runs is the slot of the role it
  // serves now (the engine's role map); the design's card is only the fallback when no role is given.
  const want = roles ? new Set(roles.map((r) => slotOf(r)).filter((s): s is string => !!s)) : null;
  const slots = tree.byPath.get("core-brain/micromodels")?.children.filter((s) => (want ? want.has(s.anatomy?.id ?? "") : s.anatomy?.config.card === modelId)) ?? [];
  const levels = new Array<number>(layers).fill(0);
  let calls = 0;
  let ms: number | null = null;
  let perLayer = false;
  // 2026-11-26: when it last ran and how often in the last hour, so the card says what it shows.
  let last: number | null = null;
  let hour = 0;
  for (const s of slots) {
    const a = v.total.get(keyOf(s));
    if (!a) continue;
    calls += a.calls_day;
    hour += a.calls_hour;
    if (a.last !== null) last = Math.max(last ?? 0, a.last);
    ms = sumMs(ms, a.ms_day);
    // 2026-11-26: a single layer entry (only "0") is a slot that does not report per layer, not
    // a MicroModel whose bottom layer did all the work.
    if (a.layers && Object.keys(a.layers).length > 1 && v.maxLayer > 0) {
      perLayer = true;
      for (let l = 0; l < layers; l++) levels[l] = Math.max(levels[l], Math.sqrt((a.layers[l] ?? 0) / v.maxLayer));
    } else if (a.calls_day > 0 && v.maxDay) for (let l = 0; l < layers; l++) levels[l] = Math.max(levels[l], 0.35 * Math.sqrt(a.calls_day / v.maxDay));
  }
  return { levels, calls, ms, perLayer, last, hour };
}
