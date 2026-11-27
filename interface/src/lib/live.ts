"use client";

import { useEffect, useState } from "react";
import { useApi, type Audit, type AuditRow } from "./api";

/** The tool rows the topology plays. One poller for every reader. */
/** How many of the newest tool calls the live read carries. */
export const LIVE_LIMIT = 20;
export const LIVE_PATH = `/api/audit?kind=tool&limit=${LIVE_LIMIT}`;

/** A row's place in time order. */
export const mark = (r: AuditRow) => `${String(r.time_ms).padStart(15, "0")}:${r.id}`;

/** Rows oldest first, one per id: a call written at its start and again at its end keeps the
 *  content of its last writing. */
export function sortRows(rows: AuditRow[]): AuditRow[] {
  const byId = new Map<string, AuditRow>();
  for (const r of rows) {
    // An ended writing is never replaced by the running one, whatever order they arrive in.
    const prev = byId.get(r.id);
    if (prev && prev.state !== "running" && r.state === "running") continue;
    byId.set(r.id, r);
  }
  return [...byId.values()].sort((a, b) => a.time_ms - b.time_ms || a.id.localeCompare(b.id));
}

/** A call of a named tool. A row with no tool name (a request the server refused, for one) is
 *  listed but never played as the organism's activity. */
export function isCall(r: AuditRow): boolean {
  return (r.kind === "tool" || !!r.tool) && !!r.tool?.trim();
}

/** The parts a row names: its steps in run order when the engine gave them, else `parts`. */
export function rowParts(r: AuditRow): string[] {
  return Array.isArray(r.steps) && r.steps.length ? r.steps.map((s) => s.part) : Array.isArray(r.parts) ? r.parts : [];
}

/** The newest call that went through the Core Brain (the last real question), else the newest
 *  named call. `inBrain` says whether a part id belongs to the Core Brain. */
export function lastBrainCall(rows: AuditRow[], inBrain: (part: string) => boolean): AuditRow | null {
  let named: AuditRow | null = null;
  for (let i = rows.length - 1; i >= 0; i--) {
    const r = rows[i];
    if (!isCall(r)) continue;
    named ??= r;
    if (rowParts(r).some(inBrain)) return r;
  }
  return named;
}

/** The job a tentacle step carried in a call, when the engine reports one. */
export function stepJob(step: { part: string; [k: string]: unknown }): { tentacle: number; job: string } | null {
  const m = /^(?:nmpod\.unit|nmapi\.tentacle_core|nmpod\.pipeline)#t(\d+)/.exec(step.part);
  // The job the plan gave the step; with lenses on and no job named, the lens it composed by.
  const named = typeof step.job === "string" && step.job.trim() ? step.job : typeof step.lens === "string" && step.lens.trim() ? step.lens : null;
  if (!m || named === null) return null;
  return { tentacle: Number(m[1]), job: named.trim() };
}

/** True while the engine is working: a row says `state: "running"`, or a new row arrived in the
 *  last few seconds. History present when the page opens does not count. */
export function useWorking(): boolean {
  const { data } = useApi<Audit>(LIVE_PATH, 1000);
  const rows = data?.audit ?? [];
  const sorted = sortRows(rows);
  const newest = sorted.length ? sorted[sorted.length - 1] : null;
  const running = !!newest && newest.state === "running";
  const key = newest ? mark(newest) : null;
  const [seen, setSeen] = useState<{ first: string | null; fresh: string | null }>({ first: null, fresh: null });
  if (key && seen.first === null) setSeen({ first: key, fresh: null });
  else if (key && seen.first !== null && key !== seen.first && key !== seen.fresh) setSeen({ first: seen.first, fresh: key });
  const [glow, setGlow] = useState(false);
  useEffect(() => {
    if (!seen.fresh) return;
    const on = window.setTimeout(() => setGlow(true), 0);
    const off = window.setTimeout(() => setGlow(false), 3200);
    return () => {
      clearTimeout(on);
      clearTimeout(off);
    };
  }, [seen.fresh]);
  return running || glow;
}
