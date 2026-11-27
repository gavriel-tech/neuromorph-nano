"use client";

/** The live instruments, all from `/api/live` and its event stream: what runs right now (or Idle /
 *  Listening), the four MicroModels' state, the idle writer, the recent real timeline, the latency
 *  waterfall of the newest question and the graphics card's memory by MicroModel. Nothing here
 *  moves unless the engine reports it; an engine build without the route says so plainly. */
import type { ReactNode } from "react";
import { Dot, Reticle } from "@/components/ui";
import { ago, useApi, type Audit, type AuditRow } from "@/lib/api";
import { isCall, LIVE_PATH, sortRows } from "@/lib/live";
import { useNow } from "@/lib/clock";
import { engineServing, eventGroup, GROUP_NAMES, modelState, stageGroup, waterfall, writerState, type LiveCapacity, type LiveEvent, type LiveReply, type LiveStage, type LiveVram, type StageGroup } from "@/lib/livefeed";
import { useLive, type FeedMode } from "@/lib/livehook";
import { AREA_NAMES, areaLine, parseMemoryAreas } from "@/lib/memoryareas";
import { msShort, STAGE_NAMES } from "@/lib/stagetimes";
import { gpuCeiling, vramBreakdown } from "@/lib/vram";

/** One colour per stage group, from the product's palette: inward search blue, outward behaviour
 *  violet, background work quiet. */
export const GROUP_HUE: Record<StageGroup, string> = {
  G0: "var(--sense-pale)",
  G1: "var(--sense)",
  G2: "var(--signal-dim)",
  G3: "var(--behav)",
  writer: "var(--learned)",
  preempt: "var(--behav-hot)",
  dense: "var(--sense)",
  filter: "var(--verified)",
  learning: "var(--learned)",
  memory: "var(--behav)",
  rebind: "var(--verified)",
  backup: "var(--text-3)",
  guidance: "var(--behav)",
  other: "var(--text-3)",
};

const stageName = (s: string) => STAGE_NAMES[s] ?? s.replace(/_/g, " ");

/** The line shown where a live figure would be when no engine is serving (the words of
 *  `NO_ENGINE`, with the commands set in monospace). */
export function NoEngine({ className = "" }: { className?: string }) {
  return (
    <div className={`mono text-[11px] text-faint ${className}`} data-e2e="no-engine">
      No engine is serving right now; start one with <code className="text-muted">neuromorph-nano serve</code> or <code className="text-muted">daemon</code>
    </div>
  );
}

function modeWord(mode: FeedMode, error: string | null, reply: LiveReply | null): string {
  if (!reply && error) return error.startsWith("404") ? "not served by this engine build" : "not answering";
  if (mode === "stream") return "event stream";
  if (mode === "polling") return "polling every 1 s";
  return "connecting";
}

/** Live elapsed time of an active stage: its reported elapsed plus the time since the snapshot. */
export function elapsedNow(a: LiveStage, snapAt: number | null, now: number): number | null {
  if (a.since_ms !== null && now > 0) return Math.max(0, now - a.since_ms);
  if (a.elapsed_ms === null) return null;
  return a.elapsed_ms + (snapAt && now > snapAt ? now - snapAt : 0);
}

function Head({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-line pb-2">
      <h2 className="eyebrow flex items-center gap-2 text-muted">
        <Reticle />
        {children}
      </h2>
      {right && <span className="mono truncate text-[10.5px] text-faint">{right}</span>}
    </div>
  );
}

/** The timeline as a list, newest first. */
export function Timeline({ events, now, max = 8 }: { events: LiveEvent[]; now: number; max?: number }) {
  const list = [...events].sort((a, b) => b.at_ms - a.at_ms).slice(0, max);
  if (!list.length) return <div className="mono py-2 text-[11px] text-faint">No event recorded yet. Each real stage, writer pass and preemption appears here as it happens.</div>;
  return (
    <ol className="min-w-0" data-e2e="live-timeline">
      {list.map((e, i) => {
        const g = eventGroup(e);
        return (
          <li key={`${e.at_ms}-${e.stage}-${i}`} className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-baseline gap-x-2.5 border-b border-line/60 py-1.5">
            <i aria-hidden className="inline-block h-1.5 w-1.5 translate-y-[-1px]" style={{ background: GROUP_HUE[g] }} />
            <span className="min-w-0">
              <span className="mono block truncate text-[11.5px] text-muted">
                {stageName(e.stage)}
                {e.model ? <span className="text-faint"> · {e.model}</span> : null}
                {e.ms !== null ? <span className="text-text"> · {msShort(e.ms)}</span> : null}
              </span>
              {e.summary && <span className="mono block truncate text-[10.5px] text-faint">{e.summary}</span>}
            </span>
            <span className="mono shrink-0 text-[10px] text-faint">{now > 0 ? ago(e.at_ms, now) : ""}</span>
          </li>
        );
      })}
    </ol>
  );
}

/** The latency waterfall of the newest question: one bar per timed stage on the question's own
 *  clock, a running stage growing as it runs. */
export function Waterfall({ events, active, snapAt, compact = false }: { events: LiveEvent[]; active: LiveStage[]; snapAt: number | null; compact?: boolean }) {
  const now = useNow();
  const live = active.map((a) => ({ ...a, elapsed_ms: elapsedNow(a, snapAt, now) }));
  // Compact (a narrow column): the ten newest bars, on the whole burst's clock.
  const { bars, span, at } = waterfall(events, live, now);
  if (!bars.length) return <div className="mono py-2 text-[11px] text-faint">No timed stage yet. The waterfall of the next question draws here from its real stage times.</div>;
  return (
    <div data-e2e="live-waterfall" role="img" aria-label={`Waterfall of ${bars.length} stages over ${msShort(span)}`}>
      {(compact ? bars.slice(-10) : bars).map((b, i) => (
        <div key={`${b.stage}-${i}`} className={`grid ${compact ? "grid-cols-[86px_minmax(0,1fr)_52px]" : "grid-cols-[132px_minmax(0,1fr)_64px]"} items-center gap-2 py-[2px]`}>
          <span className="mono truncate text-[10.5px] text-muted" title={`${GROUP_NAMES[b.group]}${b.model ? ` · ${b.model}` : ""}`}>
            {stageName(b.stage)}
          </span>
          <span className="relative block h-2 border-b border-line">
            <i className="absolute bottom-0 block h-full" style={{ left: `${(b.start / span) * 100}%`, width: `max(2px, ${(b.ms / span) * 100}%)`, background: GROUP_HUE[b.group], opacity: b.running ? 1 : 0.8, outline: b.running ? "1px solid var(--signal)" : undefined }} />
          </span>
          <span className={`mono tnum text-right text-[10.5px] ${b.running ? "text-signal" : "text-faint"}`}>{b.running ? `${msShort(b.ms)}+` : msShort(b.ms)}</span>
        </div>
      ))}
      <div className={`mono mt-1 grid ${compact ? "grid-cols-[86px_minmax(0,1fr)_52px]" : "grid-cols-[132px_minmax(0,1fr)_64px]"} gap-2 text-[10px] text-quiet`}>
        <span>{at && now > 0 ? ago(at, now) : ""}</span>
        <span className="flex justify-between">
          <span>0</span>
          <span>{msShort(span)}</span>
        </span>
        <span />
      </div>
    </div>
  );
}

const PART_HUE: Record<string, string> = { model: "var(--sense-pale)", kv: "var(--behav)", other: "var(--text-4)", reserve: "var(--verified)", free: "transparent" };
const MODEL_HUES = ["#7fb0ff", "#5b8dff", "#c58bff", "#8f6bff"];

/** The card's memory as one bar: the four MicroModels, KV caches and windows, other use, the free
 *  reserve kept for live calls, and what is free. */
/** The card's memory by MicroModel. With `capMb` (the user's VRAM limit, when one is set), a
 *  marker stands on the bar at the cap and the legend names it. */
export function VramBar({ vram, backend, capMb = null, serving = null }: { vram: LiveVram | null; backend?: "cuda" | "cpu" | null; capMb?: number | null; serving?: boolean | null }) {
  const parts = vramBreakdown(vram);
  if (!parts) {
    if (serving === false) return <NoEngine />;
    return <div className="mono text-[11px] text-faint">{backend === "cpu" ? "Running on the processor: no graphics card memory in use." : serving === null ? "Reading the live state." : "The serving engine has not opened a graphics card yet, so there is no card memory to show."}</div>;
  }
  const capShare = typeof capMb === "number" && capMb > 0 && vram?.total_mb ? Math.min(1, capMb / vram.total_mb) : null;
  const colored = parts.map((p, i) => ({ ...p, hue: p.kind === "model" ? MODEL_HUES[parts.slice(0, i).filter((x) => x.kind === "model").length % MODEL_HUES.length] : PART_HUE[p.kind] }));
  return (
    <div data-e2e="vram-breakdown">
      <div className="mono mb-1.5 flex items-baseline justify-between text-[10.5px] text-faint">
        <span>
          {Math.round(vram?.used_mb ?? 0)} of {Math.round(vram?.total_mb ?? 0)} MiB used
        </span>
        <span>
          {capShare !== null ? <span className="text-signal">limit {Math.round(capMb ?? 0)} MiB · </span> : null}
          {backend ? backend.toUpperCase() : ""}
        </span>
      </div>
      <div className="relative">
        <div className="flex h-2.5 w-full overflow-hidden border border-line" role="img" aria-label={colored.map((p) => `${p.label} ${Math.round(p.mb)} MiB`).join(", ") + (capShare !== null ? `, limit at ${Math.round(capMb ?? 0)} MiB` : "")}>
          {colored.map((p) => (
            <i key={p.id} className="block h-full" style={{ width: `${p.share * 100}%`, background: p.hue, backgroundImage: p.kind === "reserve" ? "repeating-linear-gradient(135deg, transparent 0 2px, rgba(2,3,10,0.55) 2px 4px)" : undefined }} />
          ))}
        </div>
        {capShare !== null && (
          <span aria-hidden data-e2e="vram-cap" className="pointer-events-none absolute -top-1 -bottom-1 w-px bg-signal shadow-[0_0_6px_rgb(234_242_255/0.7)]" style={{ left: `calc(${capShare * 100}% - 0.5px)` }}>
            <span className="absolute -top-[3px] -left-[3px] h-0 w-0 border-x-[3.5px] border-t-[4px] border-x-transparent border-t-signal" />
          </span>
        )}
      </div>
      <ul className="mono mt-2 grid grid-cols-2 gap-x-4 gap-y-0.5 text-[10.5px]">
        {colored.map((p) => (
          <li key={p.id} className="flex min-w-0 items-center gap-1.5">
            <i aria-hidden className="inline-block h-1.5 w-1.5 shrink-0 border border-line-3" style={{ background: p.hue }} />
            <span className="min-w-0 flex-1 truncate text-muted">{p.label}</span>
            <span className="shrink-0 text-faint">{Math.round(p.mb)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** What runs now: the live state, the active stages with their live elapsed time, the four
 *  MicroModels, the idle writer, and the recent real timeline. */
export function LiveNow({ timeline = 6, models = true, waterfall: fall = false }: { timeline?: number; models?: boolean; waterfall?: boolean }) {
  const { reply, error, at, mode, events } = useLive();
  const now = useNow();
  const busy = reply?.state === "busy";
  const off = engineServing(reply) === false;
  return (
    <section aria-label="Live engine state" data-e2e="live-now" className="min-w-0">
      <Head right={modeWord(mode, error, reply)}>Engine, now</Head>
      {!reply ? (
        <div className="mono py-2 text-[11px] text-faint">{error ? (error.startsWith("404") ? "This engine build does not serve /api/live yet. The topology still plays every recorded call." : `Live state did not answer: ${error}`) : "Reading the live state."}</div>
      ) : (
        <div className="grid gap-3 pt-2">
          <div className="flex items-center gap-2.5">
            <Dot tone={off ? "muted" : busy ? "signal" : "verified"} />
            <span className={`mono text-[12px] uppercase tracking-[0.12em] ${busy ? "text-signal" : "text-muted"}`}>{off ? (busy ? "Working (interface)" : "No engine serving") : busy ? "Working" : "Idle / Listening"}</span>
            {reply.backend && <span className="mono ml-auto text-[10.5px] text-faint">{reply.backend === "cuda" ? "CUDA" : "CPU"}</span>}
          </div>
          {off && <NoEngine />}
          {!off && reply.backend_note && <div className="mono text-[10.5px] text-faint">{reply.backend_note}</div>}
          {(reply.progress ?? []).map((p) => (
            <div key={p.kind} className="mono grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-2 text-[10.5px]" data-e2e="live-progress">
              <span className="min-w-0 truncate text-muted">{GROUP_NAMES[stageGroup(p.kind)]}</span>
              <span className="tnum text-faint">{p.done !== null && p.total !== null ? `${p.done} of ${p.total}` : p.done !== null ? `${p.done} done` : "running"}</span>
            </div>
          ))}
          {reply.active.length > 0 && (
            <ul className="grid gap-1" data-e2e="live-active">
              {reply.active.map((a, i) => {
                const g = stageGroup(a.stage);
                const el = elapsedNow(a, at, now);
                return (
                  <li key={`${a.stage}-${i}`} className="mono grid grid-cols-[auto_minmax(0,1fr)_auto] items-baseline gap-2 text-[11px]">
                    <i aria-hidden className="inline-block h-1.5 w-1.5" style={{ background: GROUP_HUE[g] }} />
                    <span className="min-w-0 truncate text-text">
                      {GROUP_NAMES[g]} · {stageName(a.stage)}
                      {a.model ? <span className="text-faint"> · {a.model}</span> : null}
                    </span>
                    <span className="tnum text-signal">{el !== null ? msShort(el) : "running"}</span>
                  </li>
                );
              })}
            </ul>
          )}
          {/* 2026-11-25: with no engine serving, only the line that says how to start one; the
              empty model list, writer line, waterfall and timeline gave the column four "nothing
              yet" blocks and pushed Recent activity off the screen. */}
          {models && !off && reply.models.length > 0 && (
            <ul className="grid gap-0.5" data-e2e="live-models">
              {reply.models.map((m) => {
                const st = modelState(m);
                return (
                  <li key={m.role} className="mono grid grid-cols-[auto_minmax(0,1fr)_auto] items-baseline gap-2 text-[10.5px]">
                    <Dot tone={st === "executing" ? "signal" : st === "resident" ? "verified" : "muted"} />
                    <span className="min-w-0 truncate text-muted" title={m.card ?? m.role}>
                      {m.card ?? m.role}
                      <span className="text-faint">
                        {m.precision ? ` · ${m.precision}` : ""}
                        {m.filters.length ? ` · ${m.filters.length} ${m.filters.length === 1 ? "socket" : "sockets"}` : ""}
                        {m.lens ? ` · lens ${m.lens}` : ""}
                      </span>
                    </span>
                    <span className="text-faint">{st === "executing" ? "executing" : m.vram_mb !== null ? `${Math.round(m.vram_mb)} MiB` : st}</span>
                  </li>
                );
              })}
            </ul>
          )}
          {reply.writer && !off && (
            <div className="mono text-[10.5px] text-faint" data-e2e="live-writer">
              idle writer: {reply.writer.queue ?? "-"} queued
              {/* `in_session` is the Writer writing now; `waiting` why it is held back; `held` only says its weights are on the card. */}
              {` · ${writerState(reply.writer)}`}
              {(["L1", "L2", "L3", "L4"] as const)
                .map((k) => {
                  const l = reply.writer!.levels[k];
                  if (!l) return null;
                  return l.done !== null && l.total !== null ? `${k} ${l.done}/${l.total}` : l.count !== null ? `${k} ${l.count}` : null;
                })
                .filter(Boolean)
                .map((s) => ` · ${s}`)
                .join("")}
              {reply.writer.preempted ? ` · ${reply.writer.preempted} preempted` : ""}
            </div>
          )}
          {fall && !off && <Waterfall events={events} active={reply.active} snapAt={at} compact />}
          {timeline > 0 && (!off || events.length > 0) && <Timeline events={events} now={now} max={timeline} />}
        </div>
      )}
    </section>
  );
}

const gib = (mbv: number | null | undefined) => (typeof mbv === "number" && Number.isFinite(mbv) ? (mbv >= 1024 ? `${(mbv / 1024).toFixed(1)} GiB` : `${Math.round(mbv)} MiB`) : "-");

/** One line of system memory by area (2026-11-27): each area's use against its budget, as the
 *  serving engine reports it under `memory_areas`; a full area is marked. */
function AreasLine({ reply }: { reply: { memory_areas?: unknown } | null }) {
  const m = parseMemoryAreas(reply);
  if (!m) return null;
  return (
    <div className="mono flex min-w-0 flex-wrap gap-x-3 text-[10.5px]" data-e2e="memory-areas-line" title={m.intake?.note ?? undefined}>
      <span className="text-faint">memory by area</span>
      {AREA_NAMES.map((a) => (
        <span key={a} className={m.areas[a].full ? "text-attention" : "text-muted"}>
          {a} {areaLine(m.areas[a])}
        </span>
      ))}
    </div>
  );
}

/** The machine's capacity as the engine measured it: the graphics card (name, total, free,
 *  compute, the reserve kept for live calls and whether it was measured), host memory and threads,
 *  and the ceilings homeostasis keeps to, with the graphics card ceiling the control resolves to
 *  (`auto` follows this card). */
/** `serving` false: no engine is serving, so `cap` is this machine's detected hardware (from
 *  `/api/limits` `profile`); the line says so, and a card read without a context (reserve not
 *  measured) shows its free memory as not measured. */
export function Capacity({ cap, gpuControl, backend, serving = null }: { cap: LiveCapacity | null; gpuControl?: string | null; backend?: "cuda" | "cpu" | null; serving?: boolean | null }) {
  if (!cap) return serving === false ? <NoEngine /> : <div className="mono text-[11px] text-faint">Reading this machine&apos;s capacity.</div>;
  const ceil = gpuCeiling(gpuControl, cap);
  const c = cap.card;
  const unmeasured = !!c && c.reserve_measured !== true && serving !== true;
  const rows: [string, string][] = c
    ? [
        ["card", c.name ?? "graphics card"],
        ["total", gib(c.total_mb)],
        ["free now", unmeasured ? "free not measured" : gib(c.free_mb)],
        ["compute", c.compute ?? "-"],
        ["reserve", c.reserve_mb !== null && !unmeasured ? `${gib(c.reserve_mb)} ${c.reserve_measured ? "measured" : "estimated"}` : unmeasured ? "not measured" : "-"],
      ]
    : [["card", backend === "cuda" ? "not reported" : "none: processor only"]];
  const threads = cap.threads !== null ? `${cap.threads}${cap.hardware_threads !== null && cap.hardware_threads !== undefined && cap.hardware_threads !== cap.threads ? ` of ${cap.hardware_threads}` : ""}` : "-";
  rows.push(["card ceiling", ceil.label], ["engine ceiling", gib(cap.gpu_ceiling_mb)], ["host memory", `${gib(cap.host_available_mb)} free of ${gib(cap.host_total_mb)}`], ["memory ceiling", gib(cap.ram_ceiling_mb)], ["threads", threads]);
  return (
    <div className="grid gap-1.5">
      {serving === false && (
        <div className="grid gap-0.5">
          <NoEngine />
          <div className="mono text-[10.5px] text-quiet">This machine&apos;s detected hardware:</div>
        </div>
      )}
      <dl className="mono grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-0.5 text-[10.5px]" data-e2e="live-capacity" data-source={serving === false ? "profile" : "live"}>
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-faint">{k}</dt>
            <dd className="min-w-0 truncate text-right text-muted" title={v}>
              {v}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** The one line a page header shows: Working (the stage running now), Idle / Listening, or No
 *  engine serving. */
export function LiveBadge() {
  const { reply, error } = useLive();
  if (!reply) return <span className="mono text-[10.5px] text-faint">{error ? `live state did not answer (${error})` : "reading live state"}</span>;
  const busy = reply.state === "busy";
  const a = reply.active[0];
  if (engineServing(reply) === false)
    return (
      <span className="mono inline-flex min-w-0 items-center gap-2 text-[10.5px]" data-e2e="live-badge" title="No engine is serving right now; start one with neuromorph-nano serve or daemon">
        <Dot tone="muted" />
        <span className="truncate text-faint">No engine serving</span>
      </span>
    );
  return (
    <span className="mono inline-flex min-w-0 items-center gap-2 text-[10.5px]" data-e2e="live-badge">
      <Dot tone={busy ? "signal" : "verified"} />
      <span className={`truncate ${busy ? "text-signal" : "text-muted"}`}>{busy ? `Working${a ? ` · ${GROUP_NAMES[stageGroup(a.stage)]}` : ""}` : "Idle / Listening"}</span>
    </span>
  );
}

/** The live waterfall and the timeline under one heading, for pages that show the newest
 *  question's stages (the stage-times panel). */
export function LiveWaterfall({ compact = false }: { compact?: boolean }) {
  const { reply, error, at, mode, events } = useLive();
  return (
    <section aria-label="Live stage waterfall" data-e2e="live-stage-waterfall" className="min-w-0">
      <Head right={modeWord(mode, error, reply)}>{reply?.state === "busy" ? "Stages, running now" : "Stages of the newest question"}</Head>
      <div className="pt-2">
        {!reply && error ? (
          <div className="mono py-2 text-[11px] text-faint">{error.startsWith("404") ? "This engine build does not serve /api/live yet; the stage bench below is the newest measure." : `Live state did not answer: ${error}`}</div>
        ) : (
          <Waterfall events={events} active={reply?.active ?? []} snapAt={at} compact={compact} />
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ the system panel */

const numOf = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const objOf = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);

/** The token budget a call ran under (`config.config.budget_tokens`, else the coordinator's pack
 *  budget), or null when its row does not carry one. */
function budgetOf(r: AuditRow): number | null {
  const o = r as Record<string, unknown>;
  const cfg = objOf(objOf(o.config)?.config);
  return numOf(cfg?.budget_tokens) ?? numOf(objOf(objOf(o.coordinator)?.budgets)?.pack_tokens);
}

/** A side panel of readings that complement a page's centre: the card's memory by MicroModel, each
 *  MicroModel's forward passes and time since the engine started, the idle Writer in full,
 *  background work with a measurable end, the tokens the recent calls sent back against their
 *  budget, this machine's capacity and the engine's recent events. All from `/api/live` and the
 *  audit; with no engine serving, the engine's own readings say so. */
export function SystemPanel({ events: maxEvents = 16 }: { events?: number }) {
  const { reply, events } = useLive();
  const audit = useApi<Audit>(LIVE_PATH, 1000);
  const now = useNow();
  const serving = engineServing(reply);
  const on = serving === true;
  const models = on ? (reply?.models ?? []) : [];
  const busiest = Math.max(1, ...models.map((m) => m.busy_ms ?? 0));
  const w = on ? (reply?.writer ?? null) : null;
  const progress = reply?.progress ?? [];
  const calls = sortRows((audit.data?.audit ?? []) as AuditRow[])
    .filter((r) => isCall(r) && r.state !== "running" && typeof r.reply_tokens === "number")
    .slice(-6)
    .reverse();
  const cap = reply?.capacity?.limits?.vram?.cap_mb ?? null;
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-6" data-e2e="system-panel">
      <section className="grid gap-2">
        <Head right={reply?.backend ? reply.backend.toUpperCase() : undefined}>Graphics card memory</Head>
        <VramBar vram={on ? (reply?.vram ?? null) : null} backend={reply?.backend ?? null} capMb={cap && cap > 0 ? cap : null} serving={serving} />
      </section>

      <section className="grid gap-2" data-e2e="system-models">
        <Head right="since the engine started">MicroModels</Head>
        {!on ? (
          <NoEngine />
        ) : (
          <ul className="grid gap-1.5">
            {models.map((m) => {
              const st = modelState(m);
              return (
                <li key={m.role} className="mono grid gap-1 text-[10.5px]" data-role={m.role} data-state={st}>
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="flex min-w-0 items-center gap-2">
                      <Dot tone={st === "executing" ? "signal" : st === "resident" ? "verified" : "muted"} />
                      <span className="uppercase tracking-[0.12em] text-muted">{m.role}</span>
                      <span className={st === "executing" ? "text-signal" : "text-faint"}>{st}</span>
                    </span>
                    <span className="tnum shrink-0 text-faint">{m.vram_mb ? `${Math.round(m.vram_mb)} MiB` : "0 MiB"}</span>
                  </div>
                  <div className="flex items-baseline justify-between gap-2 pl-3.5 text-faint">
                    <span>{typeof m.forwards === "number" ? `${m.forwards.toLocaleString()} forwards` : "forwards not measured"}</span>
                    <span className="tnum">{typeof m.busy_ms === "number" ? `${msShort(m.busy_ms)} busy` : "time not measured"}</span>
                  </div>
                  {typeof m.busy_ms === "number" && (
                    <span aria-hidden className="relative ml-3.5 block h-[2px] bg-line">
                      <i className="absolute inset-y-0 left-0 block bg-sense-pale/80" style={{ width: `${(m.busy_ms / busiest) * 100}%` }} />
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <section className="grid gap-2" data-e2e="system-writer">
        <Head right={w ? writerState(w) ?? undefined : undefined}>Idle Writer</Head>
        {!w ? (
          on ? <div className="mono text-[11px] text-faint">The engine does not report the idle Writer.</div> : <NoEngine />
        ) : (
          <dl className="mono grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-0.5 text-[10.5px]">
            {(
              [
                ["state", writerState(w) ?? "not reported"],
                ["queued", w.queue !== null ? String(w.queue) : "not measured"],
                ["written", w.results !== null ? String(w.results) : "not measured"],
                ["sessions", w.sessions !== null ? String(w.sessions) : "not measured"],
                ["preempted by calls", w.preempted !== null ? String(w.preempted) : "not measured"],
                ["card room kept for calls", w.in_session && w.headroom_mb !== null ? `${Math.round(w.headroom_mb)} MiB` : "set when a session opens"],
                ...(["L1", "L2", "L3", "L4"] as const).map((k): [string, string] => {
                  const l = w.levels[k];
                  return [`${k} notes written`, !l ? "not measured" : l.done !== null && l.total !== null ? `${l.done} of ${l.total}` : l.count !== null ? String(l.count) : "not measured"];
                }),
              ] as [string, string][]
            ).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-faint">{k}</dt>
                <dd className={`min-w-0 truncate text-right ${k === "state" && w.in_session ? "text-signal" : "text-muted"}`} title={v}>
                  {v}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </section>

      <section className="grid gap-2" data-e2e="system-progress">
        <Head>Background work</Head>
        {progress.length === 0 ? (
          <div className="mono text-[11px] text-faint">Nothing with a measurable end is running: no index build, no learning cycle, no lab build.</div>
        ) : (
          <ul className="grid gap-1.5">
            {progress.map((p) => (
              <li key={p.kind} className="mono grid gap-1 text-[10.5px]">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-muted">{p.kind === "dense" ? "index (dense lane) build" : GROUP_NAMES[stageGroup(p.kind)]}</span>
                  <span className="tnum text-signal">{p.done !== null && p.total !== null ? `${p.done} of ${p.total}` : p.done !== null ? `${p.done} done` : "running"}</span>
                </div>
                {p.done !== null && p.total ? (
                  <span aria-hidden className="relative block h-[2px] bg-line">
                    <i className="absolute inset-y-0 left-0 block bg-signal" style={{ width: `${Math.min(100, (p.done / p.total) * 100)}%` }} />
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="grid gap-2" data-e2e="system-tokens">
        <Head right="reply against budget">Tokens, recent calls</Head>
        {calls.length === 0 ? (
          <div className="mono text-[11px] text-faint">No call with a token count in the newest audit rows.</div>
        ) : (
          <ul className="grid gap-1.5">
            {calls.map((r) => {
              const out = r.reply_tokens as number;
              const tin = typeof r.request_tokens === "number" ? r.request_tokens : null;
              const budget = budgetOf(r);
              const span = Math.max(out, budget ?? 0, 1);
              return (
                <li key={r.id} className="mono grid gap-1 text-[10.5px]">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="min-w-0 truncate text-muted">{r.tool}</span>
                    <span className="tnum shrink-0 text-faint">
                      {tin !== null ? `${tin.toLocaleString()} in · ` : "in not measured · "}
                      <span className={budget !== null && out > budget ? "text-attention" : "text-text"}>{out.toLocaleString()}</span> out · {budget !== null ? `budget ${budget.toLocaleString()}` : "budget not reported"}
                      {now > 0 ? ` · ${ago(r.time_ms, now)}` : ""}
                    </span>
                  </div>
                  <span aria-hidden className="relative block h-[2px] bg-line">
                    <i className="absolute inset-y-0 left-0 block bg-sense-pale/80" style={{ width: `${(out / span) * 100}%` }} />
                    {budget !== null && <i className="absolute -top-[2px] block h-[6px] w-px bg-signal" style={{ left: `calc(${(budget / span) * 100}% - 1px)` }} />}
                  </span>
                </li>
              );
            })}
          </ul>
        )}
        <div className="mono text-[10px] text-quiet">Tokens in are the request&apos;s arguments and tokens out the reply, both as the engine counts them (characters / 4); the line marks the call&apos;s budget. The material the Nano read to answer is not measured per call.</div>
      </section>

      <section className="grid gap-2">
        <Head>This machine</Head>
        <Capacity cap={reply?.capacity ?? null} backend={reply?.backend ?? null} serving={serving} />
        <AreasLine reply={on ? reply : null} />
      </section>

      <section className="grid gap-1" data-e2e="system-events">
        <Head right={`${events.length} seen`}>Engine events</Head>
        <Timeline events={events} now={now} max={maxEvents} />
      </section>
    </div>
  );
}
