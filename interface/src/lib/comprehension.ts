/* The 4-level comprehension and world-model view (`GET /api/comprehension`, ui.rs
 * `comprehension_view`): notes of the idle writer per level, the outline preview, the measured
 * routing record and the drone heads per lens. Pure functions, so the page and the tests share
 * them. */

export type LevelTag = { tag: string; count: number };
export type Level = { level: number; name: string; count: number; tags: LevelTag[] };
export type RoutingRow = {
  job?: string;
  kind?: string;
  model?: string;
  stack?: string;
  accuracy?: number;
  slope?: number;
  n?: number;
  correct?: number;
  check_accuracy?: number;
  check_n?: number;
  strength?: number;
};
export type RoutingCombine = { job?: string; kind?: string; combine?: boolean; decisive?: number | null };
export type Routing = {
  present: boolean;
  made_utc?: string | null;
  version?: number | null;
  rows?: RoutingRow[];
  combine?: RoutingCombine[];
  lens?: string;
  lenses?: string[];
  base_cartridges?: string[];
};
export type LensHeads = { population: string; lens: string; heads: [number, number][]; count: number };

export interface ComprehensionReply {
  total_notes: number;
  note_counts: Record<string, number>;
  outline: string | null;
  routing_record: boolean;
  drone_default_heads: number;
  drone_lens_sets: number;
  base_cartridges: number;
  base_backed_up: number;
  /** Newer builds: which store was read (relative to the memory folder), or null for none. */
  notes_store?: string | null;
  levels?: Level[];
  unlevelled_notes?: number;
  outline_key?: string | null;
  routing?: Routing;
  heads?: LensHeads[];
}

/** The four levels, their labels and the note tags each holds (ui.rs `LEVELS`). */
export const LEVELS: { level: number; name: string; label: string; tags: string[]; about: string }[] = [
  { level: 1, name: "piece", label: "L1 piece", tags: ["note", "names", "sum"], about: "one-line notes, likely names and summaries of single pieces" },
  { level: 2, name: "file", label: "L2 file", tags: ["file"], about: "file notes rolled up from the piece notes" },
  { level: 3, name: "module", label: "L3 module", tags: ["module"], about: "folder and crate notes on the structural tree" },
  { level: 4, name: "system", label: "L4 system", tags: ["outline"], about: "the project outline" },
];

/** Plain names of the note tags. */
export const TAG_NAMES: Record<string, string> = {
  note: "piece notes",
  names: "likely names",
  sum: "summaries",
  file: "file notes",
  module: "module notes",
  outline: "outlines",
};

/** The world-model level (1 to 4) of a note tag, or null. */
export function levelOf(tag: string): number | null {
  return LEVELS.find((l) => l.tags.includes(tag))?.level ?? null;
}

/** The four levels with their counts: the engine's `levels` when it sends them, else computed
 *  from `note_counts` (an older build), so the panel always shows L1 to L4 in order. */
export function levelsOf(reply: Pick<ComprehensionReply, "note_counts" | "levels"> | null | undefined): (Level & { label: string; about: string })[] {
  const counts = reply?.note_counts ?? {};
  return LEVELS.map((def) => {
    const sent = reply?.levels?.find((l) => l.level === def.level);
    const tags = sent?.tags ?? def.tags.map((tag) => ({ tag, count: counts[tag] ?? 0 }));
    const count = sent?.count ?? tags.reduce((a, t) => a + (t.count ?? 0), 0);
    return { level: def.level, name: def.name, label: def.label, about: def.about, count, tags };
  });
}

/** Notes whose tag is on no level (another writer's job), from `note_counts`. */
export function unlevelled(counts: Record<string, number> | null | undefined): number {
  return Object.entries(counts ?? {}).reduce((a, [t, c]) => (levelOf(t) === null ? a + c : a), 0);
}

/** A fraction 0..1 as a whole percent, or a dash. */
export function pct(x: number | null | undefined): string {
  return typeof x === "number" && Number.isFinite(x) ? `${Math.round(x * 1000) / 10}%` : "-";
}

/** A slope with its sign, or a dash. */
export function slopeText(x: number | null | undefined): string {
  if (typeof x !== "number" || !Number.isFinite(x)) return "-";
  return `${x >= 0 ? "+" : ""}${x.toFixed(2)}`;
}

/** The combine line for a job: whether the two MicroModels' reads are combined, and at what
 *  decisive threshold. */
export function combineText(c: RoutingCombine): string {
  if (!c.combine) return "not combined";
  return typeof c.decisive === "number" ? `combined, decisive at ${c.decisive.toFixed(2)}` : "combined, no decisive threshold";
}

/** The routing record grouped by job: its model rows and its combine flag. */
export function routingByJob(r: Routing | null | undefined): { job: string; rows: RoutingRow[]; combine: RoutingCombine | null }[] {
  if (!r?.present) return [];
  const jobs: string[] = [];
  for (const x of [...(r.rows ?? []), ...(r.combine ?? [])]) {
    const j = x.job ?? "?";
    if (!jobs.includes(j)) jobs.push(j);
  }
  return jobs.map((job) => ({
    job,
    rows: (r.rows ?? []).filter((x) => (x.job ?? "?") === job),
    combine: (r.combine ?? []).find((x) => (x.job ?? "?") === job) ?? null,
  }));
}

/** A head list as `L13.6 L14.6 ...` (layer.head). */
export function headsText(heads: [number, number][] | null | undefined, max = 32): string {
  const list = (heads ?? []).filter((h) => Array.isArray(h) && h.length >= 2);
  const shown = list.slice(0, max).map(([l, h]) => `L${l}.${h}`);
  return list.length > max ? `${shown.join(" ")} +${list.length - max}` : shown.join(" ");
}
