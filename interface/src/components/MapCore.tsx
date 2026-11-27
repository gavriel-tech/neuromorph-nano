"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals" and "aslo the parameters main page ui
// should be way better. more visual and interactive with real fucntionality."): what the Parameters
// page's parts share. The shapes GET /api/parameter-maps and GET /api/surgery answer with, the states
// and kinds of maps, the filter, the lineage graph of a MicroModel's maps (made on, used by), and the
// page context the right rail's detail reads, so a map opened there stays live while the data polls.

import { createContext, useContext } from "react";
import type { Tone } from "@/components/ui";

export type Lineage = { made_on: string; made_by: string; made_ms: number; purpose: string; used_by: string[]; recorded: string; note: string; outcome?: string; outcome_why?: string };

// 2026-11-26 (Luis Jake Gabriel III: "Make sure we list when a map fails. I don't want to use trash
// maps in our data"; "We at least need to keep dialed data from healthy data"): how each map turned
// out (nmapi::lineage::OUTCOMES). Healthy maps (used, superseded) are the only ones learned from by
// default; failed and control maps only as labelled negatives; duplicates and unused never.
export const OUTCOME_TONE: Record<string, Tone> = { used: "verified", superseded: "muted", failed: "error", duplicate: "error", control: "attention", unused: "muted" };
export const healthyOutcome = (o: string | undefined) => o === "used" || o === "superseded";
export type MapRow = { file: string; kind: string; about: string; bytes: number; modified_ms: number; summary: Record<string, unknown>; purpose?: string; state?: string; state_why?: string; lineage?: Lineage | null };
export type ArchivedMap = { file: string; kind: string; about: string; bytes: number; modified_ms: number; lineage?: Lineage | null };
export type RoleMaps = { role: string; model: string; serving_base?: string | null; maps: MapRow[]; archived?: ArchivedMap[]; cleanup?: Record<string, { maps: number; bytes: number }> };
/** A job filter as GET /api/surgery lists it, with its held-out split gate figures when recorded. */
export type JobFilter = { cartridge: string; job: string; enabled?: boolean; layers?: number; base?: string; base_cos?: number; filtered_cos?: number; base_max_diff?: number; filtered_max_diff?: number };
export type BuildState = { status: string; kind?: string; role: string; job: string; message: string; by?: string; updated_ms?: number };
export type Note = { text: string; tone: "verified" | "error" };
/** What the atlas and the list show: one MicroModel or all, the states switched on, a layer, a word. */
export type MapFilter = { role: string; states: string[]; layer: number | null; text: string };
export type Run = (id: string, path: string, body: unknown, done: string) => Promise<void>;

export const STATES = ["current", "stale", "origin not recorded", "log", "history", "experiment"] as const;
export const ALL_STATES: string[] = [...STATES];
export const STATE_TONE: Record<string, Tone> = { current: "verified", stale: "attention", history: "muted", experiment: "muted", log: "muted", "origin not recorded": "attention" };
export const STATE_SWATCH: Record<string, string> = { current: "bg-verified", stale: "bg-attention", "origin not recorded": "bg-attention/60", log: "bg-quiet", history: "bg-sense-pale/50", experiment: "bg-line-3" };
export const KIND_TONE: Record<string, string> = {
  "job filter": "bg-signal",
  "parameter map": "bg-sense-pale",
  "activation map": "bg-sense",
  "separation map": "bg-behav",
  tolerance: "bg-behav",
  "block map": "bg-sense",
  "role weights": "bg-learned",
  "energy record": "bg-quiet",
  "super-weights": "bg-attention",
  "surgery probe": "bg-quiet",
  "surgery ledger": "bg-quiet",
};
/** The same colours as CSS values, for the lineage graph's SVG. */
export const KIND_VAR: Record<string, string> = {
  "job filter": "var(--signal)",
  "parameter map": "var(--sense-pale)",
  "activation map": "var(--sense)",
  "separation map": "var(--behav)",
  tolerance: "var(--behav)",
  "block map": "var(--sense)",
  "role weights": "var(--learned)",
  "energy record": "var(--text-4)",
  "super-weights": "var(--attention)",
  "surgery probe": "var(--text-4)",
  "surgery ledger": "var(--text-4)",
};
export const kindTone = (kind: string) => KIND_TONE[kind] ?? "bg-quiet";
export const kindVar = (kind: string) => KIND_VAR[kind] ?? "var(--text-4)";

// What can be cleaned up: the engine refuses the rest (what serves, what the serving base is built
// from, and the files surgery keeps at a fixed place).
const CLEANABLE = new Set(["experiment", "history", "stale", "origin not recorded"]);
const KEEP_FILES = new Set(["super-weights.json", "surgery-ledger.jsonl"]);
export const stateOf = (m: { state?: string }) => m.state ?? "origin not recorded";
export const cleanable = (m: MapRow) => m.kind !== "job filter" && !KEEP_FILES.has(m.file) && CLEANABLE.has(m.state ?? "");
export const listOf = (v: unknown): string[] => (Array.isArray(v) ? v.map(String) : []);
export const layersOf = (m: MapRow): number[] => listOf(m.summary.layers).map(Number).filter((n) => Number.isFinite(n));
export const filterOf = (jf: JobFilter[], m: MapRow): JobFilter | undefined => (m.kind === "job filter" ? jf.find((x) => `job-filter-${x.job}.json` === m.file) : undefined);
export const sum = (a: { bytes: number }[]) => a.reduce((s, m) => s + m.bytes, 0);

/** Whether a map passes the filter (the role is checked by the caller). */
export function passes(m: MapRow, f: MapFilter): boolean {
  if (!f.states.includes(stateOf(m))) return false;
  if (f.layer !== null && !layersOf(m).includes(f.layer)) return false;
  const t = f.text.trim().toLowerCase();
  if (t && !`${m.file} ${m.kind} ${m.purpose ?? ""} ${m.lineage?.made_on ?? ""} ${(m.lineage?.used_by ?? []).join(" ")} ${m.lineage?.note ?? ""}`.toLowerCase().includes(t)) return false;
  return true;
}
export const filtered = (f: MapFilter) => f.role !== "all" || f.states.length !== ALL_STATES.length || f.layer !== null || f.text.trim() !== "";

// 2026-11-26 (Luis Jake Gabriel III: "keep track of what they belong to and organize them well. In
// the system visually and through the back ends"): each MicroModel's maps grouped by what they are
// for now, from their lineage (nmapi::lineage).
export const GROUPS: { id: string; label: string; short: string; about: string; fold: boolean }[] = [
  { id: "serving", label: "Serving now", short: "serving", about: "read by the engine while it serves: the job filters plugged in on this MicroModel", fold: false },
  { id: "base", label: "What the serving base cartridge is built from", short: "builds the serving base", about: "the cut that made it and what surgery restored since", fold: false },
  { id: "probe", label: "Probes and logs", short: "probes, logs", about: "measurements of a base cartridge against the open weights, and the log of surgery writes", fold: false },
  { id: "unknown", label: "Origin not recorded", short: "origin not recorded", about: "maps that do not say what they were measured on; state it with the lineage editor", fold: false },
  { id: "history", label: "History: built earlier versions", short: "history", about: "inputs of base or compensation cartridges that no longer serve; kept for the record and for rebuilding", fold: true },
  { id: "experiment", label: "Experiments", short: "experiments", about: "sweeps, comparisons and variants no build or serving path reads", fold: true },
];
export function groupOf(m: MapRow): string {
  if (m.state === "experiment") return "experiment";
  if (m.state === "history") return "history";
  if (m.purpose === "serving" || m.kind === "job filter") return "serving";
  if (m.purpose === "probe" || m.state === "log") return "probe";
  if (m.purpose === "base-build" || m.purpose === "measurement") return "base";
  return "unknown";
}

/** What a map covers, in a few words from its summary. */
export function covers(m: MapRow): string {
  const s = m.summary;
  const parts: string[] = [];
  const num = (v: unknown) => typeof v === "string" || typeof v === "number";
  if (listOf(s.jobs).length) parts.push(`jobs ${listOf(s.jobs).join(", ")}`);
  if (typeof s.job === "string" && s.job) parts.push(`job ${s.job}`);
  if (listOf(s.sockets).length) parts.push(`${listOf(s.sockets).length} sockets`);
  if (num(s.parts)) parts.push(`${s.parts} parts`);
  if (num(s.channels_off)) parts.push(`${s.channels_off} channels left out`);
  if (num(s.tensors)) parts.push(`${s.tensors} tensors pinned`);
  if (num(s.rows)) parts.push(`${s.rows} entries`);
  if (typeof s.cos_before === "string" && typeof s.cos_after === "string") parts.push(`cosine ${s.cos_before} to ${s.cos_after}`);
  if (num(s.entries)) parts.push(`${s.entries} entries`);
  return parts.join(" · ");
}

// ---------------------------------------------------------------------------------------------
// The lineage graph of one MicroModel: an edge from what a map was made on to the map, and from the
// map to everything it is used by. Nodes are map files, base and job filter cartridges, and the
// upstream open weights ("open:<model>").

export type Graph = { up: Map<string, string[]>; down: Map<string, string[]> };
export type NodeKind = "weights" | "base" | "comp" | "filter-cart" | "cartridge" | "map" | "archived" | "missing";

export function graphOf(r: RoleMaps): Graph {
  const up = new Map<string, string[]>();
  const down = new Map<string, string[]>();
  const add = (a: string, b: string) => {
    if (!a || !b || a === b) return;
    const d = down.get(a) ?? [];
    if (!d.includes(b)) down.set(a, [...d, b]);
    const u = up.get(b) ?? [];
    if (!u.includes(a)) up.set(b, [...u, a]);
  };
  for (const m of [...r.maps, ...(r.archived ?? [])]) {
    if (m.lineage?.made_on) add(m.lineage.made_on, m.file);
    for (const u of m.lineage?.used_by ?? []) add(m.file, u);
  }
  return { up, down };
}

/** Everything reached from `start` going up or down, with the number of steps to it. */
export function walk(g: Graph, start: string, dir: "up" | "down", depth = 8): Map<string, number> {
  const seen = new Map<string, number>([[start, 0]]);
  let front = [start];
  for (let d = 1; d <= depth && front.length; d++) {
    const next: string[] = [];
    for (const n of front)
      for (const x of (dir === "up" ? g.up : g.down).get(n) ?? []) {
        if (seen.has(x)) continue;
        seen.set(x, d);
        next.push(x);
      }
    front = next;
  }
  return seen;
}

export function nodeKind(id: string, r: RoleMaps, jf: JobFilter[]): NodeKind {
  if (id.startsWith("open:")) return "weights";
  if (/\.jsonl?$/.test(id)) return r.maps.some((m) => m.file === id) ? "map" : (r.archived ?? []).some((m) => m.file === id) ? "archived" : "missing";
  if (jf.some((f) => f.cartridge === id)) return "filter-cart";
  if (/-base@\d+$/.test(id)) return "base";
  if (/-comp@\d+$/.test(id)) return "comp";
  return "cartridge";
}
export const isMapId = (id: string) => /\.jsonl?$/.test(id);
export const NODE_WORD: Record<NodeKind, string> = {
  weights: "upstream open weights",
  base: "base cartridge",
  comp: "compensation cartridge",
  "filter-cart": "job filter cartridge",
  cartridge: "cartridge",
  map: "map",
  archived: "map, in the archive",
  missing: "map, not in the folder",
};
/** A cartridge key's version, to sort "writer-base@3" before "writer-base@4". */
export const versionOf = (id: string) => Number(/@(\d+)$/.exec(id)?.[1] ?? 0);

// ---------------------------------------------------------------------------------------------
// The page context: the rail's detail is rendered inside the Workspace, so it reads the live data
// from here rather than from props captured when it was opened.

export type MapsApi = {
  roles: RoleMaps[];
  layerCount: (model: string) => number | null;
  filtersOf: (role: string) => JobFilter[];
  now: number;
  busy: boolean;
  run: Run;
  onNote: (n: Note) => void;
  buildRunning: boolean;
  /** Puts a job filter's MicroModel and job into the Add form (the caller opens the Add tab). */
  rebuild: (role: string, job: string) => void;
  /** Opens a cartridge or parameter object in the full Parameter Viewer, optionally overlaying a parameter map, filtering to a layer, or jumping to masking/overlay controls. */
  openObject?: (key: string, opts?: { map?: string; layer?: number | null; controls?: boolean }) => void;
};

const noop = async () => {};
export const MapsCtx = createContext<MapsApi>({ roles: [], layerCount: () => null, filtersOf: () => [], now: 0, busy: false, run: noop, onNote: () => {}, buildRunning: false, rebuild: () => {} });
export const useMaps = () => useContext(MapsCtx);
