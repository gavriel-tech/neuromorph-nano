/* The per-stage breakdown of a warm question (`GET /api/stage-times`, ui.rs `stage_times_view`):
 * median and p95 milliseconds per stage, per question set, from the newest stage-bench report.
 * Pure functions shared by the panel and the tests. */

export type StageRow = { stage: string; median_ms: number; p95_ms: number; n: number };
export type StageSet = { set: string; summary?: string; warm?: string; stages: StageRow[] };
export type StageTimes = {
  source: string;
  live: boolean;
  file: string | null;
  stamp: number | null;
  header: string | null;
  sets: StageSet[];
  files?: { file: string; stamp: number }[];
  hook?: string;
  /** Recent calls a serving process published in its live file (stage name to ms), newest last;
   *  empty until the engine publishes them. */
  live_calls?: Record<string, unknown>[];
};

/** The newest live call as stage rows (numeric fields only, `at_ms` and `total` left out), or
 *  null when none is published. */
export function lastLiveCall(d: Pick<StageTimes, "live_calls"> | null | undefined): { stage: string; ms: number }[] | null {
  const last = d?.live_calls?.at(-1);
  if (!last) return null;
  return Object.entries(last)
    .filter(([k, v]) => typeof v === "number" && Number.isFinite(v) && k !== "at_ms" && k !== "total")
    .map(([stage, ms]) => ({ stage, ms: ms as number }));
}

/** Plain names of the stages the serving path times (drone_path.rs `NANO_STAGE_TIMES`). */
export const STAGE_NAMES: Record<string, string> = {
  signals_meaning: "signals and meaning",
  evidence: "evidence",
  drone_kind: "drone kind",
  sense: "sense",
  compose: "compose",
  g1: "G1",
  tentacle_read: "tentacle read",
  deep_check: "deep check",
  g2_own: "G2 own picks",
  g2_order_reads: "G2 order reads",
  judge_duels: "judge duels",
  judge_close: "judge close calls",
  pack: "pack",
  total: "total",
};

/** The stage rows of a set without `total`, and the total row (or the sum of medians when the
 *  report has none). Each row carries its share of the total median, 0..1. */
export function stageBars(set: StageSet | null | undefined): { rows: (StageRow & { share: number })[]; total: StageRow | null } {
  const all = set?.stages ?? [];
  const total = all.find((r) => r.stage === "total") ?? null;
  const rows = all.filter((r) => r.stage !== "total");
  const denom = total?.median_ms && total.median_ms > 0 ? total.median_ms : rows.reduce((a, r) => a + Math.max(0, r.median_ms), 0);
  return { rows: rows.map((r) => ({ ...r, share: denom > 0 ? Math.max(0, r.median_ms) / denom : 0 })), total };
}

/** The slowest stage by median (not `total`), or null. */
export function slowest(set: StageSet | null | undefined): StageRow | null {
  const rows = (set?.stages ?? []).filter((r) => r.stage !== "total");
  return rows.reduce<StageRow | null>((a, r) => (a === null || r.median_ms > a.median_ms ? r : a), null);
}

/** Milliseconds as `414 ms` or `3.32 s`. */
export function msShort(ms: number | null | undefined): string {
  if (typeof ms !== "number" || !Number.isFinite(ms)) return "-";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

/** The report's time from its stamp (ms since 1970), as `YYYY-MM-DD HH:MM UTC`. */
export function stampText(stamp: number | null | undefined): string {
  if (typeof stamp !== "number" || !Number.isFinite(stamp) || stamp <= 0) return "";
  return new Date(stamp).toISOString().slice(0, 16).replace("T", " ") + " UTC";
}
