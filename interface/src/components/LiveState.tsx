"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel"): the live state comes apart
// into the last call (its path through the systems) and the systems now, so each can be a tab of
// the side panel; LiveState still shows both together.

/** The live state of the organism, shown where a selected part's details would be when nothing
 *  is selected: the running or last call as the path it took (each system's steps on its own row
 *  along the call's time, from the engine's own clocks), then each system's state now. Only what
 *  the API reports; a step the engine did not time is marked as untimed, never placed by guess. */
import type { ReactNode } from "react";
import { CallText, Dot, Reticle } from "@/components/ui";
import { msText, stepMs } from "@/lib/activity";
import { ago, bytes, type AuditRow, type Cartridge, type Project, type Status } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { isCall, lastBrainCall } from "@/lib/live";
import { resolvePart, type TNode, type Tree } from "@/lib/topology";

type Step = NonNullable<AuditRow["steps"]>[number];

/** The call to show: a named call running now, else the last that went through the Core Brain. */
export function currentCall(rows: AuditRow[], tree: Tree): AuditRow | null {
  const newest = [...rows].reverse().find(isCall);
  if (newest?.state === "running") return newest;
  return lastBrainCall(rows, (p) => resolvePart(tree, p).some((n) => n.path[0] === "core-brain"));
}

/** Per system: its steps in the call and the time it spent (a shared pass counted once, a step
 *  timed inside another left out of the sum). */
function bySystem(tree: Tree, row: AuditRow | null) {
  const out = new Map<string, { steps: Step[]; ms: number; timed: boolean }>();
  if (!row) return out;
  const steps: Step[] = Array.isArray(row.steps) && row.steps.length ? row.steps : (row.parts ?? []).map((p) => ({ part: p }));
  for (const st of steps) {
    const main = resolvePart(tree, st.part)[0]?.path[0];
    if (!main) continue;
    let e = out.get(main);
    if (!e) out.set(main, (e = { steps: [], ms: 0, timed: false }));
    e.steps.push(st);
  }
  for (const e of out.values()) {
    const seen = new Set<string>();
    for (const st of e.steps) {
      const ms = stepMs(st);
      if (ms === null) continue;
      e.timed = true;
      // A step timed inside another is part of that step's time: not summed twice.
      if (typeof st.inside === "string") continue;
      if (st.shared) {
        if (seen.has(st.shared)) continue;
        seen.add(st.shared);
      }
      e.ms += ms;
    }
  }
  return out;
}

/** What the steps reported, in the system's words, in the order a call runs: pieces the drones
 *  read and flagged, what the tentacles held and reported, what G1 handed up and in how many
 *  rounds, what G2 sent up and down, what memory and the reduce step counted. */
const COUNT_WORDS: [string, string][] = [
  ["chunks", "pieces read"],
  ["flagged", "flagged"],
  ["window", "in windows"],
  ["reported", "reported up"],
  ["handover", "handed up"],
  ["queue", "queued"],
  ["rounds", "rounds"],
  ["up", "sent up"],
  ["instructions", "instructions down"],
  ["episodes", "episodes"],
  ["reads", "places re-read"],
  ["packets_in", "packets in"],
  ["packets_out", "packets out"],
  ["jobs", "jobs"],
  ["tentacles", "tentacles"],
];

/** The counts of one system's steps, summed per name; null when no step reported any. A step
 *  timed inside another still counts: its figures are its own. */
export function countsOf(steps: Step[]): string | null {
  const sum = new Map<string, number>();
  for (const st of steps) {
    const c = st.counts;
    if (!c || typeof c !== "object") continue;
    for (const [k, v] of Object.entries(c)) if (typeof v === "number" && Number.isFinite(v)) sum.set(k, (sum.get(k) ?? 0) + v);
  }
  if (!sum.size) return null;
  const known = COUNT_WORDS.filter(([k]) => sum.has(k)).map(([k, w]) => `${sum.get(k)} ${k === "rounds" && sum.get(k) === 1 ? "round" : w}`);
  const other = [...sum.keys()].filter((k) => !COUNT_WORDS.some(([x]) => x === k)).map((k) => `${sum.get(k)} ${k.replace(/_/g, " ")}`);
  return [...known, ...other].join(" · ");
}

function Head({ children }: { children: ReactNode }) {
  return (
    <h3 className="eyebrow mb-3 flex items-center gap-2 text-muted">
      <Reticle />
      {children}
    </h3>
  );
}

/** One line for the sheet's handle on narrow screens. */
export function liveLine(row: AuditRow | null, now: number): string {
  if (!row) return "no call recorded yet";
  if (row.state === "running") return `${row.tool ?? "call"} · running`;
  return `${row.tool ?? "call"} · ${msText(row.ms)}${now > 0 ? ` · ${ago(row.time_ms, now)}` : ""}`;
}

/** The running or last call as the path it took: each system's steps on its own row along the
 *  call's time, from the engine's own clocks, with the time each system spent. */
export function LastCall({ tree, rows, flush = false }: { tree: Tree; rows: AuditRow[]; flush?: boolean }) {
  const now = useNow();
  const row = currentCall(rows, tree);
  const running = row?.state === "running";
  const sys = bySystem(tree, row);
  const steps: Step[] = row && Array.isArray(row.steps) ? row.steps : [];
  const span = Math.max(1, row?.ms ?? 0, ...steps.map((s) => (s.start_ms ?? 0) + (s.ms ?? 0)));
  const rowsShown = tree.root.children.filter((s) => sys.has(s.key));
  return (
    <section className={flush ? "pb-2" : "px-4 pt-4 pb-4 md:px-5"} data-e2e="last-call">
      <Head>{running ? "Running call" : "Last call"}</Head>
      {!row ? (
        <div className="text-xs text-faint">No call recorded yet. A call appears here as the path it takes, with its timing.</div>
      ) : (
        <>
          <div className="flex items-baseline justify-between gap-3">
            <span className="mono min-w-0 truncate text-[12px] uppercase tracking-[0.08em] text-text">{row.tool ?? "call"}</span>
            <span className="mono tnum shrink-0 text-[11px] text-faint">
              {running ? "running" : msText(row.ms)}
              {now > 0 ? ` · ${ago(row.time_ms, now)}` : ""}
            </span>
          </div>
          <div className="mono mt-1 truncate text-[11px] text-faint">
            <CallText id={row.id} label={row.label} tool={row.tool} />
          </div>
          {running && <div className="mt-3 text-xs text-faint">The path is reported when the call ends.</div>}
          {!running && rowsShown.length > 0 && (
            <div className="mt-4" role="img" aria-label={`Path of the call through ${rowsShown.length} systems over ${span} ms`}>
              {rowsShown.map((s) => {
                const e = sys.get(s.key)!;
                const counts = countsOf(e.steps);
                return (
                  <div key={s.key} className="py-[3px]">
                    <div className="grid grid-cols-[92px_minmax(0,1fr)_3.5rem] items-center gap-2">
                      <span className="mono truncate text-[10.5px] text-muted">{s.short ?? s.label}</span>
                      <span className="relative block h-2.5 border-b border-line">
                        {e.steps.map((st, i) =>
                          typeof st.start_ms === "number" && stepMs(st) !== null ? (
                            <i key={i} className="absolute bottom-0 block bg-sense-pale" style={{ left: `${(st.start_ms / span) * 100}%`, width: `max(2px, ${((stepMs(st) ?? 0) / span) * 100}%)`, height: "100%", opacity: 0.75 }} />
                          ) : (
                            <i key={i} className="absolute right-0 bottom-0 block h-1.5 w-1.5 border border-faint" title="not timed" />
                          ),
                        )}
                      </span>
                      <span className="mono tnum text-right text-[10px] text-faint">{e.timed ? msText(e.ms) : "untimed"}</span>
                    </div>
                    {counts && (
                      <div className="mono grid grid-cols-[92px_minmax(0,1fr)] gap-2 text-[10px] text-faint">
                        <span />
                        <span className="truncate tnum" title={counts}>
                          {counts}
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
              <div className="mono mt-1 grid grid-cols-[92px_minmax(0,1fr)_3.5rem] gap-2 text-[10px] text-quiet">
                <span />
                <span className="flex justify-between">
                  <span>0</span>
                  <span>{span} ms</span>
                </span>
                <span />
              </div>
              {steps.some((st) => stepMs(st) === null) && (
                <div className="mono mt-1 grid grid-cols-[92px_minmax(0,1fr)] gap-2 text-[10px] text-quiet">
                  <span />
                  <span className="flex items-center gap-1.5">
                    <i aria-hidden className="inline-block h-1.5 w-1.5 border border-faint" /> step not timed by the engine
                  </span>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}

/** Each system's state now, from what the API reports about it; a click opens it. */
export function SystemsNow({ tree, rows, status, carts, projects, onGo, flush = false }: { tree: Tree; rows: AuditRow[]; status: Status | null; carts: Cartridge[] | null; projects: Project[] | null; onGo: (path: string[]) => void; flush?: boolean }) {
  const row = currentCall(rows, tree);
  const running = row?.state === "running";
  const sys = bySystem(tree, row);
  const systems = tree.root.children;

  /** The state of a system now, from what the API reports about it. */
  const stateOf = (n: TNode): string => {
    const e = sys.get(n.key);
    const ran = e ? (running ? "working now" : e.timed ? `ran ${msText(e.ms)}` : "ran, not timed") : null;
    let held = "";
    if (n.role === "engine") {
      const l = status?.loaded ?? [];
      held = l.length ? `${l.length} ${l.length === 1 ? "model" : "models"} held${status?.gpu_bytes ? ` · ${bytes(status.gpu_bytes)} on the card` : ""}` : status?.loaded ? "no model held" : "";
    } else if (n.role === "memory") {
      held = projects ? `${projects.length} project ${projects.length === 1 ? "folder" : "folders"}` : "";
    } else if (n.role === "learning") {
      held = carts ? `${carts.filter((c) => c.loaded).length} of ${carts.length} cartridges loaded` : "";
    } else if (n.role === "substrate") {
      const s = status?.substrate;
      held = s ? `${s.chunks ?? 0} chunks · ${s.files ?? 0} files held` : status?.api ? "nothing held" : "";
    } else if (n.role === "runtime") {
      held = typeof status?.serving === "number" ? `${status.serving} serving` : "";
    } else if (n.role === "pod" && e) {
      const t = new Set(e.steps.map((s) => /#[td](\d+)/.exec(s.part)?.[1]).filter(Boolean));
      held = t.size ? `${t.size} of ${n.children.filter((c) => c.role === "tentacle").length} tentacles` : "";
    }
    return [ran ?? "idle", held].filter(Boolean).join(" · ");
  };

  return (
    <section className={flush ? "" : "border-t border-line px-4 py-4 md:px-5"} data-e2e="systems-now">
      <Head>{`${systems.length} systems, now`}</Head>
      <ul className="flex flex-col">
        {systems.map((s) => {
          const on = sys.has(s.key);
          return (
            <li key={s.key}>
              <button onClick={() => onGo(s.path)} className="press group grid min-h-11 w-full grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-2.5 border-b border-line/60 text-left">
                <Dot tone={on && running ? "signal" : on ? "verified" : "muted"} />
                <span className="flex min-w-0 flex-col py-1.5">
                  <span className="truncate text-[13px] text-muted group-hover:text-text">{s.label}</span>
                  <span className="mono truncate text-[11px] text-faint">{stateOf(s)}</span>
                </span>
                <span className="mono text-[10.5px] text-quiet group-hover:text-muted" title={`${s.children.length} parts inside`}>
                  {s.children.length || ""}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

export default function LiveState({ tree, rows, status, carts, projects, onGo }: { tree: Tree; rows: AuditRow[]; status: Status | null; carts: Cartridge[] | null; projects: Project[] | null; onGo: (path: string[]) => void }) {
  return (
    <div>
      <LastCall tree={tree} rows={rows} />
      <SystemsNow tree={tree} rows={rows} status={status} carts={carts} projects={projects} onGo={onGo} />
    </div>
  );
}
