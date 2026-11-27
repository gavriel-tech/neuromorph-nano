"use client";

/** Readers for the session in play, the evidence ledger, the outcomes memory keeps and the
 *  training store (API 4 to 7), and the project they are asked for. Only what the engine reports; nothing is inferred
 *  beyond counting its rows. */
import { useSyncExternalStore } from "react";
import { useApi, type Ledger, type LedgerRow, type PrimeCounts, type Projects, type SessionPrime, type SessionState } from "./api";
import type { Tone } from "@/components/ui";

/** API levels at which each reader exists. */
export const API_SESSION = 4;
export const API_LIBRARY = 5;
export const API_OUTCOMES = 6;
export const API_EXAMPLES = 7;

/** The primes of a session, under the name API 7 gives them or the earlier one. */
export function primesOf(s: SessionState | null | undefined): SessionPrime[] {
  return s?.primes ?? s?.nudges ?? [];
}
/** The ledger's prime counts, under the name API 7 gives them or the earlier one. */
export function primeCounts(l: Ledger | null | undefined): PrimeCounts | undefined {
  return l?.primes ?? l?.nudges;
}
/** A ledger row that records a prime given (`primed`; API 6 wrote `nudged`). */
export const isPrimeRow = (r: LedgerRow) => r.row === "primed" || r.row === "nudged";
/** A ledger row that records a delivery of the agent's guidance (a reminder, a step nudge, a loop
 *  continuation). */
export const isGuidedRow = (r: LedgerRow) => r.row === "guided";

/* ---------- The project asked for: picked on a page, kept for this session only. ---------- */
let picked: string | null = null;
const subs = new Set<() => void>();
export function pickProject(key: string | null) {
  picked = key;
  subs.forEach((f) => f());
}

/** The project whose session, ledger and outcomes are shown: the one picked on a page, else the
 *  project a server is serving now, else the most recently served, else the Nano's own folder. */
export function useProjectKey(): { key: string | null; picked: boolean; projects: Projects | null } {
  const projects = useApi<Projects>("/api/projects", 10000);
  const p = useSyncExternalStore(
    (f) => {
      subs.add(f);
      return () => subs.delete(f);
    },
    () => picked,
    () => null,
  );
  const d = projects.data;
  if (p) return { key: p, picked: true, projects: d };
  if (!d) return { key: null, picked: false, projects: null };
  const serving = d.serving?.find((s) => s.project_key)?.project_key;
  const newest = [...d.projects].sort((a, b) => (b.last_ms ?? 0) - (a.last_ms ?? 0))[0]?.key;
  return { key: d.current ?? serving ?? newest ?? "nano", picked: false, projects: d };
}

/* ---------- Words ---------- */

/** The colour of a ledger state, one meaning each: used is the sensory blue (the material was
 *  read), verified the checked green, refuted the failed red; passed over is pale, nothing found
 *  and unknown are quiet. */
export function stateBg(state: string | undefined | null): string {
  switch (state) {
    case "used":
      return "bg-sense-pale";
    case "verified":
      return "bg-verified";
    case "refuted":
      return "bg-error";
    case "passed-over":
      return "bg-faint";
    default:
      return "bg-line-3";
  }
}
export function stateText(state: string | undefined | null): string {
  switch (state) {
    case "used":
      return "text-sense-pale";
    case "verified":
      return "text-verified";
    case "refuted":
      return "text-error";
    case "passed-over":
      return "text-faint";
    default:
      return "text-quiet";
  }
}
/** The tone of an outcome's standing. */
export function standingTone(s: string | undefined): Tone {
  return s === "verified" ? "verified" : s === "refuted" ? "error" : "muted";
}

export function stateWord(state: string | undefined | null): string {
  if (!state) return "unknown";
  if (state === "passed-over") return "passed over";
  if (state === "none") return "nothing found";
  return state;
}

export const RULES: [string, string][] = [
  ["read", "a cited place read"],
  ["edit", "a cited place edited"],
  ["elsewhere", "other files instead"],
  ["check", "the next build or test"],
  ["revert", "an edit undone"],
  ["next-prompt", "your next prompt"],
  ["asked-again", "the same question again"],
  ["followed", "a prime followed"],
  ["step-followed", "a step nudge followed"],
  ["loop-met", "a loop's condition met before its cap"],
  ["glob-check", "a reminder's edits under its glob kept a passing check"],
];

/** The kind of a prime in plain words. */
export function primeWord(kind: string | undefined): string {
  switch (kind) {
    case "prompt-recall":
      return "at a prompt · an earlier answer";
    case "failure-fix":
      return "after a failure · the fix that worked";
    case "edit-impact":
      return "before an edit · what it touches";
    case "resume":
      return "after compaction · what was in play";
    case "open-loop":
      return "at a stop · changes not checked";
    default:
      return kind ?? "prime";
  }
}

/** A place as it may be shown: the file's own name and its lines, the folders held back until
 *  asked. `crates/shop/src/lib.rs:13-19` shows as `…/lib.rs:13-19`. */
export function placeShort(place: string | null | undefined): string {
  if (!place) return "";
  const p = place.replace(/\\/g, "/");
  const i = p.lastIndexOf("/");
  return i < 0 ? p : `…/${p.slice(i + 1)}`;
}

/** Engine wording shown as it comes, with the one word the system does not use replaced. */
export function cartridgeWords(s: string | null | undefined): string {
  if (!s) return "";
  return s.replace(/\bbooks\b/g, "cartridges").replace(/\bbook\b/g, "cartridge").replace(/\bBooks\b/g, "Cartridges").replace(/\bBook\b/g, "Cartridge");
}

/* ---------- What the ledger says overall ---------- */

export type PackSummary = { pack: string; at: number; prime: boolean; guided: boolean; kind?: string; states: Set<string>; followed: string | null; evidence: LedgerRow[] };

/** Packs and primes with their evidence, from the ledger's rows. */
export function packsOf(rows: LedgerRow[]): PackSummary[] {
  const by = new Map<string, PackSummary>();
  for (const r of rows) {
    if (!r.pack) continue;
    let p = by.get(r.pack);
    if (!p) by.set(r.pack, (p = { pack: r.pack, at: r.at ?? r.t ?? 0, prime: false, guided: false, states: new Set(), followed: null, evidence: [] }));
    if (r.row === "served") p.at = r.at ?? p.at;
    if (isPrimeRow(r)) {
      p.prime = true;
      p.kind = typeof r.kind === "string" ? r.kind : undefined;
      p.at = r.at ?? p.at;
    }
    if (isGuidedRow(r)) {
      p.guided = true;
      p.kind = typeof r.kind === "string" ? r.kind : undefined;
      p.at = r.at ?? p.at;
    }
    if (r.row === "evidence") {
      p.evidence.push(r);
      if (r.state) p.states.add(r.state);
      if (r.rule === "followed") p.followed = r.state ?? null;
    }
  }
  return [...by.values()].sort((a, b) => b.at - a.at);
}

/** How often what the Nano served was then used, verified or refuted: over packs whose window
 *  closed (their evidence is written). Primes and guidance deliveries are not packs. */
export function overall(l: Ledger | null) {
  const packs = packsOf(l?.rows ?? []).filter((p) => !p.prime && !p.guided);
  const closed = packs.filter((p) => p.evidence.length > 0);
  const n = (s: string) => closed.filter((p) => p.states.has(s)).length;
  return { served: l?.packs?.served ?? packs.length, open: l?.packs?.window_open ?? 0, closed: closed.length, used: n("used"), verified: n("verified"), refuted: n("refuted"), passed: n("passed-over") };
}

export function pct(a: number, b: number): string {
  return b > 0 ? `${Math.round((a / b) * 100)}%` : "–";
}

/* ---------- Cartridge families ---------- */

/** The three families, in the order a stack applies them: fundamentals first, then learning, then
 *  preference on top. */
export const FAMILIES = ["fundamentals", "learning", "preference"] as const;
export type Family = (typeof FAMILIES)[number];

/** A cartridge's family: the engine's `family` when it names one; otherwise its class, where a
 *  library cartridge (`self`: hardware use, the Nano's own system, code and ML fundamentals) is a
 *  fundamentals cartridge. */
export function familyOf(c: { family?: string; class?: string; kind?: string } | null | undefined): Family {
  const f = (c?.family ?? "").toLowerCase();
  if ((FAMILIES as readonly string[]).includes(f)) return f as Family;
  if (c?.class === "preference" || (!c?.class && (c?.kind ?? "").toLowerCase().includes("preference"))) return "preference";
  if (c?.class === "learning") return "learning";
  return c?.class === "self" ? "fundamentals" : "learning";
}
export const familyRank = (f: Family) => FAMILIES.indexOf(f);

/** A domain as it may be shown: a project's name held back (`project:repo-a8de…`). */
export function domainShort(d: string, mask: (s: string) => string): string {
  return d.startsWith("project:") ? `project:${mask(d.slice(8))}` : d;
}
