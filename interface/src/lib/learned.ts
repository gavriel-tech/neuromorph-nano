/* What the Nano has learned, counted from the engine's own records: job-path filters
 * (`/api/surgery`), project lenses (`/api/comprehension` routing record, `/api/live` models),
 * preference cartridges (`/api/cartridges`), G3 muscle updates and the split-gate verdicts of the
 * idle learning cycles (`/api/learning` `idle_cycle.cycles[].learners`). Pure, shared by the
 * learning page, its funnel figure and the tests. A source that did not answer counts as not
 * reported, never as zero. */
import type { Cartridge } from "./api";

export type JobFilterRow = { role: string; cartridge: string; job: string; enabled: boolean; base_cos?: number; filtered_cos?: number; layers?: number };
export type LearnerRow = { name?: string; ran?: boolean; promoted?: boolean; gate?: string; held_out?: number; current?: number | null; candidate?: number | null; note?: string };
export type CycleRow = { id?: string; at?: number; trigger?: string; learners?: LearnerRow[]; rolled_back?: boolean };
export type IdleCycle = { verified_since_last_cycle?: number; threshold?: number; due?: boolean; cycles?: CycleRow[] };

export type Learned = {
  filters: number | null;
  filtersOn: number | null;
  lenses: number | null;
  /** Judge read-outs: judge read-out cartridges, and judge read-out learners the split gate
   *  promoted in cycles that were not rolled back. */
  judges: number | null;
  preference: number | null;
  muscle: number | null;
  /** Split-gate verdicts over the kept cycles: a learner's candidate promoted, or held back. */
  gatePass: number | null;
  gateHeld: number | null;
  rollbacks: number | null;
  cycles: number | null;
  /** All learned pieces the engine reports (filters, lenses, judge read-outs, preference
   *  cartridges, G3 updates). */
  total: number | null;
};

/** The job-path filters `/api/surgery` lists per role. */
export function jobFiltersOf(surgery: { roles?: Record<string, { job_filters?: { cartridge: string; job: string; enabled?: boolean; base_cos?: number; filtered_cos?: number; layers?: number }[] }> } | null | undefined): JobFilterRow[] | null {
  if (!surgery || typeof surgery !== "object" || !surgery.roles) return null;
  const out: JobFilterRow[] = [];
  for (const [role, info] of Object.entries(surgery.roles)) for (const jf of info?.job_filters ?? []) out.push({ role, cartridge: jf.cartridge, job: jf.job, enabled: jf.enabled !== false, base_cos: jf.base_cos, filtered_cos: jf.filtered_cos, layers: jf.layers });
  return out;
}

/** True for a preference cartridge (by class or family). */
export function isPreference(c: Cartridge): boolean {
  return c.class === "preference" || /pref/i.test(c.family ?? "") || /pref/i.test(c.kind ?? "");
}

/** True for a judge read-out cartridge. */
export function isJudge(c: Cartridge): boolean {
  return /judge|read-?out/i.test(`${c.kind ?? ""} ${c.id}`);
}

export function learnedCounts(src: { carts?: Cartridge[] | null; filters?: JobFilterRow[] | null; lenses?: string[] | null; idle?: IdleCycle | null }): Learned {
  const filters = src.filters ? src.filters.length : null;
  const filtersOn = src.filters ? src.filters.filter((f) => f.enabled).length : null;
  const lenses = src.lenses ? new Set(src.lenses.filter(Boolean)).size : null;
  const preference = src.carts ? src.carts.filter((c) => isPreference(c) && !isJudge(c)).length : null;
  const judgeCarts = src.carts ? src.carts.filter(isJudge).length : null;
  let judgeLearned: number | null = null;
  const cycles = src.idle?.cycles ?? null;
  let muscle: number | null = null;
  let gatePass: number | null = null;
  let gateHeld: number | null = null;
  let rollbacks: number | null = null;
  if (cycles) {
    muscle = 0;
    gatePass = 0;
    gateHeld = 0;
    rollbacks = 0;
    judgeLearned = 0;
    for (const c of cycles) {
      if (c.rolled_back) rollbacks++;
      for (const l of c.learners ?? []) {
        if (!l.ran) continue;
        if (l.promoted) gatePass++;
        else gateHeld++;
        if (/g3|muscle/i.test(l.name ?? "") && l.promoted && !c.rolled_back) muscle++;
        if (/judge|read-?out/i.test(l.name ?? "") && l.promoted && !c.rolled_back) judgeLearned++;
      }
    }
  }
  const judges = judgeCarts === null && judgeLearned === null ? null : (judgeCarts ?? 0) + (judgeLearned ?? 0);
  const parts = [filters, lenses, judges, preference, muscle];
  const total = parts.every((p) => p === null) ? null : parts.reduce<number>((a, p) => a + (p ?? 0), 0);
  return { filters, filtersOn, lenses, judges, preference, muscle, gatePass, gateHeld, rollbacks, cycles: cycles ? cycles.length : null, total };
}

const part = (n: number | null, one: string, many: string) => (n === null ? null : `${n} ${n === 1 ? one : many}`);

/** The lenses the engine names: the routing record's lenses (`/api/comprehension`) and each live
 *  MicroModel's lens (`/api/live`), the base lens left out. Null when neither answered. */
export function lensesOf(routing: { lens?: string; lenses?: string[] } | null | undefined, live: { lens: string | null }[] | null | undefined): string[] | null {
  if (!routing && !live) return null;
  // 2026-11-26: one lens named with its version in one place (`id@9`) and without in another is one
  // lens, counted once.
  const out = new Set<string>();
  const add = (l: string | null | undefined) => {
    if (l) out.add(l.replace(/@\d+$/, ""));
  };
  for (const l of routing?.lenses ?? []) add(l);
  add(routing?.lens);
  for (const m of live ?? []) add(m.lens);
  // The base lens (no project lens) is not a learned piece.
  return [...out].filter((l) => !/^(base|default|none|off)$/i.test(l));
}

/** The funnel's caption: learned pieces by kind, in place of a fixed zero. */
export function learnedLine(l: Learned): string {
  if (l.total === null) return "learned pieces not reported";
  const bits = [part(l.filters, "job filter", "job filters"), part(l.lenses, "lens", "lenses"), part(l.judges, "judge read-out", "judge read-outs"), part(l.preference, "preference cartridge", "preference cartridges"), part(l.muscle, "G3 muscle update", "G3 muscle updates")].filter(Boolean);
  return `${l.total} learned ${l.total === 1 ? "piece" : "pieces"}${bits.length ? ` (${bits.join(", ")})` : ""}`;
}

/** The split-gate line: candidates promoted, held back, cycles rolled back. */
export function gateLine(l: Learned): string {
  if (l.gatePass === null) return "split gate not reported";
  return `split gate ${l.gatePass} passed · ${l.gateHeld} held back · ${l.rollbacks} rolled back · ${l.cycles} ${l.cycles === 1 ? "cycle" : "cycles"}`;
}
