"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "also the livate activity needs to be updated and all
// working and correctly. says no claude activity. we dont use just cluade so thats misleading. also
// its just missing a lot of real details"): what every coding agent asked the Nano in this project,
// from the audit log's tool rows, whichever agent made them (the MCP client names itself when it
// connects, and each call records it). Claude Code's hooks add a richer session trace when they are
// set up; this list does not depend on them.
//
// 2026-11-26 (his words: "The discriptors should be mostly hidden ... And that whole information
// database for it should be on a right side panel"): each call shows its real figures on two lines
// (tool, mode, time, reply tokens, which MicroModels ran); a click opens everything the engine
// recorded about it in the side panel: the MicroModels with their time and layers, what was read,
// the coverage, memory, and the steps that took time.

import { useMemo, type ReactNode } from "react";
import { roleTitle } from "@/components/MicroModel";
import { CallText, Dot } from "@/components/ui";
import { useRail } from "@/components/Workspace";
import { msText } from "@/lib/activity";
import { ago, useApi, type Audit, type AuditRow } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { LIVE_LIMIT, LIVE_PATH } from "@/lib/live";
import { maskName } from "@/lib/privacy";

/** An MCP client's own name, as people know it. */
export function agentName(client: unknown): string {
  const c = typeof client === "string" ? client.trim() : "";
  if (!c) return "an agent";
  const l = c.toLowerCase();
  if (l.includes("claude")) return "Claude Code";
  if (l.includes("cursor")) return "Cursor";
  if (l.includes("visual studio code") || l === "vscode" || l.includes("copilot")) return "VS Code";
  if (l.includes("antigravity")) return "Antigravity";
  if (l.includes("windsurf") || l.includes("codeium")) return "Windsurf";
  if (l.startsWith("nano-")) return `the Nano's own ${c.slice(5)}`;
  return c;
}

const num = (v: unknown) => (typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v)) ? Number(v) : null);
const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
const str = (v: unknown) => (typeof v === "string" && v.trim() ? v : null);

/** Who made the call: the MCP client's name when the row records it; older rows did not. */
export function callerOf(r: AuditRow): string {
  const c = str((r as Record<string, unknown>).client);
  if (c) return agentName(c);
  return r.via === "serve" ? "MCP client" : r.via === "call" ? "command line" : "caller not recorded";
}

const ROLES = ["reader", "writer", "embedder", "reranker"] as const;

/** The MicroModels a call ran, by role, with the time and layers each step reports: from the
 *  engine's own steps (`nmbrain.slot#<role>`) and the coordinator's list of engines used. */
export function modelsRan(r: AuditRow): { role: string; ms: number | null; layers: [number, number] | null }[] {
  const out = new Map<string, { role: string; ms: number | null; layers: [number, number] | null }>();
  for (const st of r.steps ?? []) {
    const m = /^nmbrain\.slot#(\w+)/.exec(st.part);
    const role = m?.[1];
    if (!role || !(ROLES as readonly string[]).includes(role)) continue;
    const e = out.get(role) ?? { role, ms: null, layers: null };
    const ms = num(st.ms);
    if (ms !== null) e.ms = (e.ms ?? 0) + ms;
    if (Array.isArray(st.layers) && st.layers.length === 2) e.layers = st.layers;
    out.set(role, e);
  }
  const engines = obj(obj(obj(r.context_map)?.coordinator)?.model)?.engines;
  if (Array.isArray(engines)) for (const x of engines) if (typeof x === "string" && (ROLES as readonly string[]).includes(x) && !out.has(x)) out.set(x, { role: x, ms: null, layers: null });
  return [...out.values()];
}

/** The short line of real figures under a call. */
function factsLine(r: AuditRow): string {
  const o = r as Record<string, unknown>;
  const ms = num(o.ms);
  const tokens = num(o.reply_tokens);
  const running = r.state === "running";
  const models = modelsRan(r).map((m) => roleTitle(m.role));
  return [
    String(o.tool ?? ""),
    typeof o.mode === "string" ? o.mode : null,
    running ? "running now" : r.state === "stale" ? "never ended" : ms !== null ? msText(ms) : null,
    tokens !== null ? `${tokens.toLocaleString()} tokens back` : null,
    models.length ? models.join(" + ") : null,
    r.ok === false ? "failed" : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

function Row({ k, children }: { k: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[7.5rem_minmax(0,1fr)] items-baseline gap-3 py-0.5 text-[11px]">
      <span className="text-faint">{k}</span>
      <span className="mono min-w-0 break-words text-muted">{children}</span>
    </div>
  );
}

function Block({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="grid gap-1 border-t border-line/60 py-3">
      <h4 className="eyebrow text-muted">{title}</h4>
      {children}
    </section>
  );
}

/** Everything the engine recorded about one call, for the side panel. Only fields the row has. */
export function CallDetail({ row: first }: { row: AuditRow }) {
  const now = useNow();
  // A call still running when it was opened is read again until it ends.
  const live = useApi<Audit>(LIVE_PATH, 2000);
  const row = ((live.data?.audit ?? []) as AuditRow[]).find((r) => r.id === first.id) ?? first;
  const o = row as Record<string, unknown>;
  const cm = obj(o.context_map);
  const coord = obj(cm?.coordinator);
  const model = obj(coord?.model);
  const budgets = obj(coord?.budgets);
  const read = Array.isArray(cm?.read) ? (cm!.read as Record<string, unknown>[]) : [];
  const models = modelsRan(row);
  const steps = (row.steps ?? []).filter((s) => typeof s.ms === "number" && s.ms > 0 && typeof s.inside !== "string").sort((a, b) => (b.ms ?? 0) - (a.ms ?? 0));
  const span = Math.max(1, num(o.ms) ?? 0, ...(row.steps ?? []).map((s) => (s.start_ms ?? 0) + (s.ms ?? 0)));
  const carts = Array.isArray(o.cartridges) ? (o.cartridges as unknown[]).map(String) : [];
  const coverage = num(cm?.coverage);
  return (
    <div className="grid" data-e2e="call-detail" data-call={row.id}>
      <div className="grid gap-0.5 pb-3">
        <div className="mono text-[11.5px] text-text">
          <CallText id={row.id} label={row.label} tool={row.tool} />
        </div>
        <div className="mono text-[10.5px] text-faint">
          {[callerOf(row), now > 0 ? ago(row.time_ms, now) : null, new Date(row.time_ms).toLocaleString()].filter(Boolean).join(" · ")}
        </div>
      </div>
      <Block title="Call">
        <Row k="Tool">{row.tool ?? "not named"}</Row>
        {row.mode && <Row k="Mode">{row.mode}</Row>}
        {row.via && <Row k="Through">{row.via === "serve" ? "MCP server (serve)" : row.via}</Row>}
        {row.project && <Row k="Project">{maskName(row.project)}</Row>}
        <Row k="Time">{row.state === "running" ? "running now" : row.state === "stale" ? "never ended" : msText(row.ms)}</Row>
        {num(o.reply_tokens) !== null && <Row k="Reply">{`${num(o.reply_tokens)!.toLocaleString()} tokens`}</Row>}
        <Row k="Result">{row.ok === false ? `failed${row.error ? `: ${row.error}` : ""}` : row.state === "running" ? "not yet" : "answered"}</Row>
        {str(o.session) && <Row k="Session">{String(o.session)}</Row>}
      </Block>
      <Block title="MicroModels that ran">
        {models.length === 0 ? (
          <span className="mono text-[11px] text-faint">none: this call ran without a MicroModel</span>
        ) : (
          <ul className="grid gap-0.5">
            {models.map((m) => (
              <li key={m.role} className="mono grid grid-cols-[minmax(0,1fr)_auto_auto] items-baseline gap-3 text-[11px]">
                <span className="text-text">{roleTitle(m.role)}</span>
                <span className="text-faint">{m.layers ? `layers ${m.layers[0]} to ${m.layers[1]}` : ""}</span>
                <span className="tnum text-muted">{m.ms !== null ? msText(m.ms) : "not timed"}</span>
              </li>
            ))}
          </ul>
        )}
        {model && (
          <div className="mono mt-1 text-[10.5px] text-faint">
            {[num(model.gpu_mb) !== null ? `${Math.round(num(model.gpu_mb)!)} MiB on the card` : null, num(model.ram_mb) !== null ? `${Math.round(num(model.ram_mb)!)} MiB in RAM` : null, num(model.shadow_agree) !== null ? `shadow agreed ${Math.round(num(model.shadow_agree)! * 100)}%` : null].filter(Boolean).join(" · ")}
          </div>
        )}
      </Block>
      {carts.length > 0 && (
        <Block title="Cartridges used">
          <span className="mono text-[11px] text-muted">{carts.join(", ")}</span>
        </Block>
      )}
      {cm && (
        <Block title="Context">
          {str(coord?.task_family) && <Row k="Task">{String(coord!.task_family)}</Row>}
          {coverage !== null && <Row k="Coverage">{`${Math.round(coverage * 100)}%`}</Row>}
          {num(cm.scope_files) !== null && <Row k="In scope">{`${num(cm.scope_files)} files`}</Row>}
          {read.length > 0 && <Row k="Read">{`${read.reduce((a, x) => a + (num(x.files) ?? 0), 0)} files in ${read.length} segments · ${read.reduce((a, x) => a + (num(x.findings) ?? 0), 0)} findings`}</Row>}
          {num(cm.rounds) !== null && <Row k="Rounds">{String(num(cm.rounds))}</Row>}
          {num(cm.merged) !== null && <Row k="Merged">{String(num(cm.merged))}</Row>}
          {budgets && num(budgets.pack_tokens) !== null && <Row k="Pack budget">{`${num(budgets.pack_tokens)!.toLocaleString()} tokens`}</Row>}
          {num(o.rerank_best) !== null && <Row k="Best rerank">{num(o.rerank_best)!.toFixed(3)}</Row>}
          {typeof o.floor_met === "boolean" && <Row k="Floor met">{o.floor_met ? "yes" : "no"}</Row>}
        </Block>
      )}
      {(num(o.memory_recalled) !== null || str(o.memory_write) || typeof o.withheld === "boolean") && (
        <Block title="Memory">
          {num(o.memory_recalled) !== null && <Row k="Recalled">{`${num(o.memory_recalled)} episodes`}</Row>}
          {str(o.memory_write) && <Row k="Written">{String(o.memory_write)}</Row>}
          {typeof o.withheld === "boolean" && <Row k="Text kept">{o.withheld ? "withheld (prompts off)" : "kept"}</Row>}
        </Block>
      )}
      {steps.length > 0 && (
        <Block title={`Steps that took time · ${steps.length} of ${row.steps?.length ?? 0}`}>
          <ul className="grid gap-[3px]" role="img" aria-label={`The ${steps.length} timed steps of the call over ${span} ms`}>
            {steps.slice(0, 14).map((s, i) => (
              <li key={`${s.part}-${i}`} className="mono grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_3.5rem] items-center gap-2 text-[10.5px]">
                <span className="truncate text-muted" title={s.part}>
                  {s.part}
                </span>
                <span className="relative block h-2 border-b border-line">
                  <i aria-hidden className="absolute bottom-0 block h-full bg-sense-pale/75" style={{ left: `${((s.start_ms ?? 0) / span) * 100}%`, width: `max(2px, ${((s.ms ?? 0) / span) * 100}%)` }} />
                </span>
                <span className="tnum text-right text-faint">{msText(s.ms)}</span>
              </li>
            ))}
          </ul>
        </Block>
      )}
    </div>
  );
}

export default function AgentCalls({ project, limit = 12 }: { project: string | null; limit?: number }) {
  // Every project's calls, up to the live read's 20, come from the topology's live read (one poll
  // shared by the page); a project's calls are their own read.
  const shared = !project && limit <= LIVE_LIMIT;
  const q = useApi<Audit>(shared ? LIVE_PATH : `/api/audit?kind=tool&limit=${limit}${project ? `&project=${encodeURIComponent(project)}` : ""}`, 3000);
  const now = useNow();
  const rail = useRail();
  const rows = useMemo(() => {
    const all = (q.data?.audit ?? []) as AuditRow[];
    return (shared ? all.slice(-limit) : [...all]).reverse();
  }, [q.data, shared, limit]);
  const named = [...new Set(rows.map((r) => str((r as Record<string, unknown>).client)).filter(Boolean).map(agentName))];
  const unnamed = rows.filter((r) => !str((r as Record<string, unknown>).client)).length;
  if (!q.data) return <div className="mono py-2 text-[11px] text-faint">{q.error ? `the calls did not answer (${q.error})` : "reading the calls"}</div>;
  if (rows.length === 0)
    return (
      <div className="mono py-2 text-[11px] text-faint" data-e2e="agent-calls">
        No coding agent has called the Nano on this project yet. Connect one (Claude Code, Cursor, VS Code, Antigravity, Windsurf or any MCP client) and its calls show here as they happen.
      </div>
    );
  return (
    <div className="grid min-h-0 gap-1" data-e2e="agent-calls">
      <div className="mono text-[10.5px] text-faint">
        newest {rows.length} calls
        {named.length > 0 && ` · from ${named.join(", ")}`}
        {unnamed > 0 && ` · ${unnamed} from a client the row did not name`}
      </div>
      <ol className="grid min-h-0 overflow-auto">
        {rows.map((r) => {
          const running = r.state === "running";
          const open = () => rail.show({ key: `call:${r.id}`, title: r.tool ?? "call", body: <CallDetail row={r} /> });
          // The row opens the call's details; its own button is the keyboard's way in (the label's
          // show button sits inside the row, so the row itself is not a button).
          return (
            <li key={r.id} onClick={open} className="group grid cursor-pointer grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-3 border-b border-line/50 py-1.5" data-e2e="agent-call">
              <span className="min-w-0">
                <span className="flex min-w-0 items-baseline gap-2">
                  <Dot tone={running ? "signal" : r.ok === false ? "error" : "verified"} />
                  <span className="mono shrink-0 text-[10.5px] text-muted">{callerOf(r)}</span>
                  <span className="min-w-0 truncate text-[11.5px] text-text">
                    <CallText id={r.id} label={r.label} tool={r.tool} />
                  </span>
                </span>
                <span className="mono block truncate pl-4 text-[10px] text-faint group-hover:text-muted">{factsLine(r)}</span>
              </span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  open();
                }}
                aria-label={`Open the details of the ${r.tool ?? "call"} call`}
                title="Everything the engine recorded about this call"
                className="press mono shrink-0 text-[10px] text-faint hover:text-text"
              >
                {typeof r.time_ms === "number" && now > 0 ? ago(r.time_ms, now) : "open"} ›
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
