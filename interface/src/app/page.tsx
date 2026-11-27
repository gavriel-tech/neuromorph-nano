"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "Dude. The discriptors should be mostly hidden ...
// And that whole information database for it should be on a right side panel. The main core of the
// pages should be on actual work flow and core function visuals ... you have to think of it as if
// it's an interactive system for the main parts the right menu are libraries and or discription
// tools etc. left menu are the main interface nav like it is now. ... We need the interface to be
// amazing."; and of this page: "also the livate activity needs to be updated and all working and
// correctly. says no claude activity. we dont use just cluade so thats misleading. also its just
// missing a lot of real details"): Live Activity is the organism, large, and what is happening now:
// every coding agent's calls with their real figures, the engine now, the stage timings, the path
// of the last call and the session in play. The side panel holds the recent activity and learning
// events, the MicroModels, the cartridges, the set up and what the page is.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type ReactNode } from "react";
import AgentCalls from "@/components/AgentCalls";
import { LiveNow, SystemPanel } from "@/components/LivePanels";
import MicroModelName, { roleTitle } from "@/components/MicroModel";
import { cartKind, Rack } from "@/components/PagePlates";
import SessionTrace from "@/components/SessionTrace";
import SetupStrip, { useSetupNeeded } from "@/components/SetupStrip";
import Topology, { LastCallRail } from "@/components/Topology";
import { CallText, Dot, Failed, Reticle } from "@/components/ui";
import Workspace, { useRail } from "@/components/Workspace";
import { msText } from "@/lib/activity";
import { ago, bits, bytes, useApi, type Audit, type AuditRow, type Cartridge, type Status } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { familyOf, FAMILIES } from "@/lib/evidence";
import { isCall, LIVE_PATH, mark, sortRows } from "@/lib/live";
import { engineServing, liveModelFor, modelState } from "@/lib/livefeed";
import { useLive } from "@/lib/livehook";
import { maskName } from "@/lib/privacy";
import { useRoleMap } from "@/lib/roles";
import { msShort, STAGE_NAMES, stampText, type StageTimes } from "@/lib/stagetimes";

/** A panel heading in the middle: a reticle, a short name, and a link or figure at the right. */
function Head({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-line pb-2">
      <h2 className="eyebrow flex min-w-0 items-center gap-2 text-muted">
        <Reticle />
        {children}
      </h2>
      {right && <span className="mono shrink-0 text-[10.5px] text-faint">{right}</span>}
    </div>
  );
}

/* ------------------------------------------------------------------ status line */

/** The figures on the title line: the engine, the MicroModels held, the cartridges loaded, the last
 *  call, and a mark when something is missing for the Nano to work. */
function StatusLine({ status, carts, newest }: { status: Status | null; carts: Cartridge[] | null; newest: AuditRow | null }) {
  const live = useLive();
  const now = useNow();
  const rail = useRail();
  const need = useSetupNeeded(status);
  const reply = live.reply;
  const off = engineServing(reply) === false;
  const busy = reply?.state === "busy";
  const held = reply ? reply.models.filter((m) => m.resident).length : null;
  return (
    <>
      <span className="flex items-center gap-2" data-e2e="live-engine-state">
        <Dot tone={!reply ? "muted" : off ? "muted" : busy ? "signal" : "verified"} />
        <span className={busy ? "text-signal" : undefined}>{!reply ? (live.error ? "engine not answering" : "connecting") : off ? "no engine serving" : busy ? "working" : "listening"}</span>
      </span>
      {status && (
        <span title="MicroModels held in memory now, of those installed">
          {held !== null && reply ? `${held} of ${reply.models.length}` : status.models.length} MicroModels{held !== null ? " held" : ""}
        </span>
      )}
      {carts && <span>{`${carts.filter((c) => c.loaded).length} of ${carts.length} cartridges loaded`}</span>}
      {newest && now > 0 && <span className="normal-case tracking-normal">last call {ago(newest.time_ms, now)}</span>}
      {need > 0 && (
        <button type="button" onClick={() => rail.open("setup")} className="press border-b border-attention text-attention hover:text-text">
          set up: {need} {need === 1 ? "thing" : "things"} missing
        </button>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ stage timings */

/** The stage timings: the newest call a serving process published (live), else the newest stage
 *  bench median per stage, each stage a bar of its share. */
function StageTimings() {
  const q = useApi<StageTimes>("/api/stage-times", 10000);
  const d = q.data;
  const lastLive = d?.live_calls?.at(-1) as Record<string, unknown> | undefined;
  const liveRows = lastLive
    ? Object.entries(lastLive)
        .filter(([k, v]) => typeof v === "number" && Number.isFinite(v) && k !== "at_ms" && k !== "total")
        .map(([stage, ms]) => ({ stage, ms: ms as number }))
    : null;
  const set = d?.sets?.[0];
  const benchRows = set ? set.stages.filter((s) => s.stage !== "total").map((s) => ({ stage: s.stage, ms: s.median_ms, p95: s.p95_ms })) : [];
  const rows: { stage: string; ms: number; p95?: number }[] = liveRows?.length ? liveRows : benchRows;
  const total = liveRows?.length ? (typeof lastLive?.total === "number" ? (lastLive.total as number) : rows.reduce((a, r) => a + r.ms, 0)) : (set?.stages.find((s) => s.stage === "total")?.median_ms ?? rows.reduce((a, r) => a + r.ms, 0));
  const max = Math.max(1, ...rows.map((r) => r.p95 ?? r.ms));
  const source = liveRows?.length ? "live, newest call" : set ? `bench median · ${set.set}${d?.stamp ? ` · ${stampText(d.stamp)}` : ""}` : null;
  return (
    <section aria-label="Stage timings" className="grid min-w-0 gap-2" data-e2e="live-stage-times">
      <Head right={total > 0 ? `total ${msShort(total)}` : undefined}>Stage timings</Head>
      {!d ? (
        <div className="mono text-[11px] text-faint">{q.error ? `the stage times did not answer (${q.error})` : "reading the stage times"}</div>
      ) : rows.length === 0 ? (
        <div className="mono text-[11px] text-faint">No stage table yet. A serving process publishes each call&apos;s stages; the stage bench writes a table.</div>
      ) : (
        <>
          <div className="mono text-[10px] text-quiet">{source}</div>
          <ul className="grid gap-[3px]" role="img" aria-label={`${rows.length} stages, ${msShort(total)} in all`}>
            {rows.map((r) => (
              <li key={r.stage} className="mono grid grid-cols-[minmax(0,7.5rem)_minmax(0,1fr)_3.25rem] items-center gap-2 text-[10.5px]">
                <span className="truncate text-muted" title={r.stage}>
                  {STAGE_NAMES[r.stage] ?? r.stage.replace(/_/g, " ")}
                </span>
                <span className="relative block h-2 border-b border-line">
                  {r.p95 !== undefined && <i aria-hidden className="absolute bottom-0 left-0 block h-px bg-line-3" style={{ width: `${(r.p95 / max) * 100}%` }} title={`p95 ${msShort(r.p95)}`} />}
                  <i aria-hidden className="absolute bottom-0 left-0 block h-full bg-sense-pale/80" style={{ width: `max(1px, ${(r.ms / max) * 100}%)` }} />
                </span>
                <span className="tnum text-right text-faint">{msShort(r.ms)}</span>
              </li>
            ))}
          </ul>
          {!liveRows?.length && <div className="mono text-[10px] text-quiet">the faint line is the p95; live figures replace these while an engine serves</div>}
        </>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ rail: activity */

function rowText(e: AuditRow): string {
  if (e.kind === "setting") return `${e.label ?? e.key ?? "Setting changed"}`;
  return String(e.label ?? e.tool ?? e.kind);
}

const LEARNING_KINDS = "learning-cycle,library-seat,library-stacks,cartridge-source,setting";

/** A learning event in plain words, the MicroModel named by its role. */
function learningText(e: AuditRow, roleOf: (m: string) => string): string {
  const o = e as Record<string, unknown>;
  if (e.kind === "library-seat" && typeof o.line === "string") {
    const model = typeof o.model === "string" ? o.model : "";
    const role = model ? roleOf(model) : "";
    const rest = model && o.line.startsWith(model + ": ") ? o.line.slice(model.length + 2) : o.line;
    return `${role ? roleTitle(role) : "MicroModel"}: ${rest}`;
  }
  if (e.kind === "learning-cycle") {
    const restored = Array.isArray(o.restored) ? ` (restored ${(o.restored as unknown[]).join(", ")})` : "";
    return e.label ?? `learning cycle ${String(o.action ?? "")}${restored}${e.project ? ` · ${maskName(e.project)}` : ""}`;
  }
  if (e.kind === "setting") return `${e.label ?? e.key}: ${e.from} to ${e.to}`;
  return e.label ?? e.kind;
}

function ActivityRail() {
  const router = useRouter();
  const recent = useApi<Audit>("/api/audit?limit=24", 2000);
  const learn = useApi<Audit>(`/api/audit?kind=${LEARNING_KINDS}&limit=24`, 10000);
  const roles = useRoleMap();
  const now = useNow();
  const rows = sortRows(recent.data?.audit ?? []).reverse();
  const learning = sortRows((learn.data?.audit ?? []) as AuditRow[])
    .filter((e) => e.kind !== "setting" || /learn|cartridge|library|lens|filter/i.test(String(e.key ?? "")))
    .reverse()
    .slice(0, 10);
  const goAudit = (id: string) => router.push(`/audit/?id=${encodeURIComponent(id)}`);
  // Rows present when the page opened are history; later ones are marked new.
  const [firstMark, setFirstMark] = useState<string | null>(null);
  if (firstMark === null && rows[0]) setFirstMark(mark(rows[0]));
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-6">
      <section className="grid grid-cols-[minmax(0,1fr)] gap-1" data-e2e="recent-activity">
        <Head
          right={
            <Link href="/audit/" className="link">
              audit
            </Link>
          }
        >
          Recent activity
        </Head>
        {recent.error && !recent.data && <div className="mono py-2 text-[11px] text-error">the audit did not answer ({recent.error})</div>}
        {recent.data && rows.length === 0 && <div className="py-3 text-xs text-faint">Nothing yet. Calls, changes and outcomes appear here as they happen.</div>}
        <ol>
          {rows.map((e) => {
            const fresh = firstMark !== null && mark(e) > firstMark;
            return (
              <li
                key={e.id}
                onClick={() => goAudit(e.id)}
                title="Open this event in Audit"
                className={`group flex cursor-pointer items-start gap-3 border-b border-line/60 py-2 transition-colors hover:bg-bg-2/50 ${fresh ? "row-new" : ""}`}
              >
                <Dot tone={e.ok === false ? "error" : e.state === "running" ? "signal" : "muted"} className="mt-1.5" />
                <div className="min-w-0 flex-1">
                  <div className="mono truncate text-[12px] text-muted group-hover:text-text">
                    <span>{e.kind === "tool" ? <CallText id={e.id} label={e.label} tool={e.tool} onDetail={() => goAudit(e.id)} /> : rowText(e)}</span>
                  </div>
                  <div className="mono truncate text-[10.5px] text-faint">
                    {e.kind === "setting"
                      ? `${e.from} to ${e.to}${e.undo_of ? " (undo)" : ""}`
                      : [e.kind, e.mode, e.state === "running" ? "running" : e.state === "stale" ? "never ended" : e.kind === "tool" || typeof e.ms === "number" ? msText(e.ms) : null, typeof e.reply_tokens === "number" ? `${e.reply_tokens} tokens` : null].filter(Boolean).join(" · ")}
                  </div>
                </div>
                <span className="mono shrink-0 pt-0.5 text-[10.5px] text-faint group-hover:text-text">
                  {now > 0 ? ago(e.time_ms, now) : ""} ›
                </span>
              </li>
            );
          })}
        </ol>
      </section>
      <section className="grid grid-cols-[minmax(0,1fr)] gap-1" data-e2e="learning-events">
        <Head
          right={
            <Link href="/learning/" className="link">
              learning
            </Link>
          }
        >
          Learning events
        </Head>
        {learn.data && learning.length === 0 && <div className="py-3 text-xs text-faint">No learning event recorded yet: cycles, cartridges seated and learned units switched show here.</div>}
        <ol>
          {learning.map((e) => (
            <li key={e.id} className="border-b border-line/60">
              <Link
                href={`/audit/?id=${encodeURIComponent(e.id)}`}
                title="Open this event in Audit"
                className="group grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-3 py-2 transition-colors hover:bg-bg-2/50"
              >
                <i aria-hidden className={`mt-1.5 inline-block h-1.5 w-1.5 ${e.kind === "library-seat" ? "bg-learned" : e.kind === "learning-cycle" ? "bg-sense-pale" : "border border-line-3"}`} />
                <div className="min-w-0">
                  <div className="mono line-clamp-2 text-[11.5px] text-muted group-hover:text-text" title={learningText(e, roles.roleOf)}>
                    {learningText(e, roles.roleOf)}
                  </div>
                  <div className="mono text-[10px] text-faint">{e.kind.replace(/-/g, " ")}</div>
                </div>
                <span className="mono shrink-0 pt-0.5 text-[10.5px] text-faint group-hover:text-text">
                  {now > 0 ? ago(e.time_ms, now) : ""} ›
                </span>
              </Link>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ rail: MicroModels */

function ModelsRail({ status }: { status: Status | null }) {
  const live = useLive();
  const roles = useRoleMap();
  const models = status?.models ?? [];
  return (
    <section className="grid gap-1" data-e2e="live-micromodels">
      <Head
        right={
          <Link href="/models/" className="link">
            models
          </Link>
        }
      >
        MicroModels
      </Head>
      {!status && <div className="mono py-2 text-[11px] text-faint">reading the MicroModels</div>}
      <ul>
        {models.map((m) => {
          const lm = liveModelFor(live.reply?.models, m.id, roles.roleOf);
          const st = modelState(lm);
          return (
            <li key={m.id} className="grid gap-1 border-b border-line/60 py-2.5 text-[12px]" title={m.card === true ? "model card present and checked" : m.card === false ? "the engine reports no model card" : "card not reported"}>
              <div className="flex items-start gap-3">
                {/* A red mark on every model the engine reports without a card. */}
                <Dot tone={m.card === true ? "verified" : m.card === false ? "attention" : "muted"} className="mt-1" />
                <span className="min-w-0 flex-1">
                  <MicroModelName model={m.id} layout="stack" nameClass="text-text" />
                </span>
                <span className={`mono shrink-0 text-[10.5px] ${st === "executing" ? "text-signal" : st === "resident" ? "text-verified" : "text-faint"}`}>{st}</span>
              </div>
              <div className="mono flex flex-wrap gap-x-3 pl-4 text-[10.5px] text-faint">
                <span>{bits(m.precision)}</span>
                <span>{bytes(m.bytes)} on disk</span>
                {typeof m.layers === "number" && <span>{m.layers} layers</span>}
                {lm?.vram_mb ? <span>{Math.round(lm.vram_mb)} MiB on the card</span> : null}
                {lm && lm.filters.length > 0 && <span>{`${lm.filters.filter((f) => f.on).length} of ${lm.filters.length} sockets on`}</span>}
                {lm?.lens && <span>lens {lm.lens}</span>}
              </div>
            </li>
          );
        })}
      </ul>
      <div className="mono pt-1 text-[10px] text-quiet">
        <i aria-hidden className="mr-1.5 inline-block h-1.5 w-1.5 bg-learned" />
        base cartridge it runs on
        <i aria-hidden className="mr-1.5 ml-4 inline-block h-1.5 w-1.5 rounded-full border border-quiet" />
        upstream open weights
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ rail: cartridges */

function CartridgesRail({ carts, error }: { carts: Cartridge[] | null; error: string | null }) {
  const list = carts ?? [];
  const loaded = list.filter((c) => c.loaded);
  const byFam = FAMILIES.map((f) => ({ f, all: list.filter((c) => familyOf(c) === f) }));
  return (
    <section className="grid gap-3" data-e2e="live-cartridges">
      <Head
        right={
          <Link href="/cartridges/" className="link">
            cartridges
          </Link>
        }
      >
        Cartridges
      </Head>
      {error && !carts && <div className="mono text-[11px] text-error">the cartridges did not answer ({error})</div>}
      {carts && (
        <>
          <dl className="mono grid grid-cols-3 gap-3 text-[10.5px]">
            {byFam.map(({ f, all }) => (
              <div key={f} className="grid gap-0.5 border-t border-line pt-2">
                <dt className="uppercase tracking-[0.12em] text-faint">{f}</dt>
                <dd className="tnum text-[13px] text-text">
                  {all.filter((c) => c.loaded).length}
                  <span className="text-[10.5px] text-faint"> of {all.length} loaded</span>
                </dd>
              </div>
            ))}
          </dl>
          {/* The rack: every loaded cartridge as a slab against the model's layer scale. */}
          <div className="overflow-x-auto border-t border-line pt-2">
            <Rack carts={loaded} />
          </div>
          <ul>
            {[...list]
              .sort((a, b) => Number(b.loaded) - Number(a.loaded) || a.id.localeCompare(b.id) || b.version - a.version)
              .slice(0, 24)
              .map((c, i) => (
                <li key={`${c.id}@${c.version}#${i}`} className="mono grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 border-b border-line/50 py-1.5 text-[11px]" title={c.description}>
                  <span aria-hidden className={c.loaded ? "spectral-sw !h-3.5 !w-1" : "dither-slot !h-3.5 !w-2"} />
                  <span className="min-w-0 truncate text-muted">
                    {c.id}@{c.version}
                    <span className="text-faint"> · {cartKind(c)} · {c.kind}</span>
                  </span>
                  <span className={c.loaded ? "text-verified" : "text-faint"}>{c.loaded ? "loaded" : c.verified ? "verified" : "in the cartridge library"}</span>
                </li>
              ))}
          </ul>
          {list.length > 24 && <div className="mono text-[10.5px] text-faint">{list.length - 24} more on the cartridges page</div>}
        </>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ the page */

const about = (
  <div className="grid gap-3 text-[12.5px] leading-relaxed text-muted">
    <p>The Nano reads any material, finds and compresses the context your coding agents need, keeps a memory, and learns into cartridges that help comprehension for the job.</p>
    <p>The organism in the middle is drawn from the engine&apos;s own anatomy. Each call a coding agent makes (Claude Code, Cursor, VS Code, Antigravity, Windsurf or any MCP client) lights the parts it used in the order they ran; the engine&apos;s live stage events light their parts while they run. Click a part to see it here; scroll or use + and - to zoom.</p>
    <p>Under it: every agent&apos;s calls with their real figures (click one for all it recorded), the engine now, the stage timings, the path of the last call through the systems, and the session an agent&apos;s hooks record.</p>
    <p>Nothing here is placed by guess or animated without a real event. A figure the engine does not report is shown as not reported.</p>
  </div>
);

/** Live Activity: the organism as the centrepiece, what is happening now under it, the libraries
 *  and references in the side panel. */
export default function LiveActivity() {
  const status = useApi<Status>("/api/status", 5000);
  const carts = useApi<Cartridge[]>("/api/cartridges", 5000);
  // The newest call is read from the topology's own live read (the last 20 tool calls, polled
  // once a second), not from a second poll of the same rows.
  const calls = useApi<Audit>(LIVE_PATH, 2000);
  const s = status.data ?? null;
  const newest = sortRows((calls.data?.audit ?? []) as AuditRow[])
    .reverse()
    .find(isCall) ?? null;

  return (
    <Workspace
      title="Live Activity"
      railKey="live"
      defaultTab="activity"
      status={<StatusLine status={s} carts={carts.data ?? null} newest={newest} />}
      rail={[
        { id: "activity", label: "Activity", body: <ActivityRail /> },
        { id: "system", label: "System", body: <SystemPanel /> },
        { id: "models", label: "MicroModels", body: <ModelsRail status={s} />, count: s ? s.models.length : undefined },
        { id: "cartridges", label: "Cartridges", body: <CartridgesRail carts={carts.data ?? null} error={carts.error} />, count: carts.data ? carts.data.filter((c) => c.loaded).length : undefined },
        { id: "setup", label: "Set up", body: <SetupStrip status={s} variant="rail" /> },
      ]}
      about={about}
    >
      <div className="@container grid min-w-0 gap-8">
        {status.error && !s && <Failed error={status.error} />}

        <section aria-label="The organism" className="ticks flex h-[clamp(400px,calc(100dvh-15rem),900px)] flex-col overflow-hidden border border-line">
          <Topology variant="hero" panel="rail" />
        </section>

        <div className="grid min-w-0 gap-x-10 gap-y-8 @3xl:grid-cols-2 @6xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)_minmax(0,1fr)]">
          <section aria-label="Agent calls" className="grid min-w-0 content-start gap-2 @3xl:row-span-2 @6xl:row-span-1">
            <Head
              right={
                <Link href="/audit/" className="link">
                  audit
                </Link>
              }
            >
              Agent calls
            </Head>
            <AgentCalls project={null} limit={10} />
          </section>
          <div className="grid min-w-0 content-start gap-8">
            <LiveNow timeline={4} waterfall />
            <StageTimings />
          </div>
          <section aria-label="Path of the last call" className="min-w-0">
            <LastCallRail />
          </section>
        </div>

        <SessionTrace packs={3} calls={false} />
      </div>
    </Workspace>
  );
}
