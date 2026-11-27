/**
 * Memory areas (2026-11-27): the process budget divided by what the engine holds in host memory.
 * Each area is `auto` (a fixed share of the budget) or a size in MiB, set through
 * `POST /api/limits {"key": "content_mb", "value": "auto" | number}` (or the full
 * `limits.memory.<area>_mb` key). The engine reports each area's setting, budget and use under
 * `memory_areas` in `GET /api/limits`, `GET /api/live`, `nm_status` and `nm_live`.
 */

export type AreaName = "content" | "caches" | "weights";
export const AREA_NAMES: readonly AreaName[] = ["content", "caches", "weights"];
export const AREA_KEYS: Record<AreaName, string> = {
  content: "limits.memory.content_mb",
  caches: "limits.memory.caches_mb",
  weights: "limits.memory.weights_mb",
};
export const AREA_LABELS: Record<AreaName, string> = { content: "Project content", caches: "Kept caches", weights: "MicroModel weights" };
/** What each area holds, and what lowering it costs. */
export const AREA_HOLDS: Record<AreaName, string> = {
  content: "The files and pieces read, the drones' intake, the relevance keys and the dense lane's vectors.",
  caches: "The encoded-chunk cache the tentacles re-read.",
  weights: "The host copies of the MicroModels' weights.",
};
export const AREA_COST: Record<AreaName, string> = {
  content: "Lower: the intake stops sooner and the pieces not reached are found by their paths only.",
  caches: "Lower: fewer encodings kept, so repeated questions encode again and take longer.",
  weights: "Lower: fewer weights held in RAM, so a MicroModel may load when a call needs it.",
};
/** Auto shares of the process budget in percent (`nmengn::limits::AREA_SHARES`); the other 15
 *  percent stays with the engine's own working memory. */
export const AREA_SHARE: Record<AreaName, number> = { content: 25, caches: 15, weights: 45 };
export const AREA_MIN_MB = 128;

export type AreaRow = { setting: string | null; budgetMb: number | null; usedMb: number | null; full: boolean; state: string | null };
export type AreaIntake = { taken: number | null; waiting: number | null; full: boolean; running: boolean; note: string | null } | null;
export type MemoryAreas = { areas: Record<AreaName, AreaRow>; intake: AreaIntake };

const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : null);
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);

/** The `memory_areas` block of a reply (`/api/limits`, `/api/live`), or null when it has none. */
export function parseMemoryAreas(reply: unknown): MemoryAreas | null {
  const block = obj(obj(reply)?.memory_areas);
  const areas = obj(block?.areas);
  if (!block || !areas) return null;
  const row = (a: AreaName): AreaRow => {
    const r = obj(areas[a]);
    return { setting: str(r?.setting), budgetMb: num(r?.budget_mb), usedMb: num(r?.used_mb), full: r?.full === true, state: str(r?.state) };
  };
  const i = obj(block.intake);
  return {
    areas: { content: row("content"), caches: row("caches"), weights: row("weights") },
    intake: i ? { taken: num(i.pieces_taken), waiting: num(i.pieces_waiting), full: i.full === true, running: i.background_pass === true, note: str(i.note) } : null,
  };
}

/** Checks a typed area size: `auto` or a whole number of at least `AREA_MIN_MB` MiB. */
export function validateArea(raw: string): { ok: true; value: string } | { ok: false; error: string } {
  const t = raw.trim().toLowerCase();
  if (t === "auto") return { ok: true, value: "auto" };
  if (!/^\d+$/.test(t)) return { ok: false, error: "Must be auto or a whole number of MiB" };
  const n = Number(t);
  if (n < AREA_MIN_MB) return { ok: false, error: `Must be auto or at least ${AREA_MIN_MB} MiB` };
  return { ok: true, value: String(n) };
}

/** One line for an area: use against budget, and whether a pass stopped at it. */
export function areaLine(r: AreaRow): string {
  const used = r.usedMb === null ? "not measured" : `${Math.round(r.usedMb)} MiB`;
  const budget = r.budgetMb === null ? "budget not known" : `${Math.round(r.budgetMb)} MiB`;
  return `${used} / ${budget}${r.full ? " · full" : ""}`;
}
