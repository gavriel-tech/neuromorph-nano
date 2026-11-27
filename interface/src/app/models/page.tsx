"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "Dude. The discriptors should be mostly hidden ... And
// that whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals ... you have to think of it as if it's an
// interactive system for the main parts the right menu are libraries and or discription tools etc.
// left menu are the main interface nav like it is now. ... We need the interface to be amazing." And
// about this page: "the activity of the micromodels on the model page seem off. like the activity is
// fake. jsut one micromodel is shown lit up and not dynamically or based on real activity", "also it
// doesnt let you control the micromodels base open source weights completely or show them on the ui
// for the micromodels", "the oipen weights used is not the micromodel. that should be clearly
// diferentiated but not overyly through text"):
// the middle is the four MicroModels as live machines: an activity meter across all four (real calls
// over 24 hours and the last hour, from the engine's activity record), then each machine with its
// layers lit by those calls, its live state from the live feed (a fresh step lights its dot), the base
// cartridge it runs on and the upstream open weights it was built from, its figures and its controls.
// The drone heads are no longer drawn bright by default (they made one MicroModel look lit whatever
// ran); a switch shows them. Pressing a machine shows its detail in the side panel. The side panel's
// tabs: Downloads (`/models/#downloads` opens it), Libraries, Routing, Lenses, About.

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { NoteLine } from "@/components/Controls";
import CartridgeSources from "@/components/CartridgeSources";
import { DownloadsPanel } from "@/components/Downloads";
import { HFDownloadButton } from "@/components/HFDownloadManager";
import { LiveBadge, SystemPanel } from "@/components/LivePanels";
import { roleTitle } from "@/components/MicroModel";
import { FRESH_MS, ModelDetail, ModelMachine, liveFor, stateWord, type BaseModel, type Note, type Pulse, type StatusModel } from "@/components/ModelMachine";
import { RolesPanel, RoutingPanel } from "@/components/RolesRouting";
import { useTopology } from "@/components/Topology";
import { LensesPanel } from "@/components/UnitsLenses";
import Workspace, { RailSection, useRail } from "@/components/Workspace";
import { Failed, Masked, Placeholder } from "@/components/ui";
import { modelLayers, useActivity } from "@/lib/activity";
import { bytes, useApi, type Cartridge, type Control, type Status } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { engineServing, slotOf, type LiveEvent } from "@/lib/livefeed";
import { useLive, useLiveEvents } from "@/lib/livehook";
import { droneSets } from "@/lib/paint";
import { maskPath } from "@/lib/privacy";
import { useRoleMap } from "@/lib/roles";
import { precisionKey } from "@/lib/topology";

/** Address hashes that open a side-panel tab (other pages link to `/models/#downloads`). */
const HASH_TAB: Record<string, string> = { downloads: "downloads", libraries: "libraries", sources: "libraries", routing: "routing", roles: "routing", lenses: "lenses", about: "about" };

/** Opens the side-panel tab the address names, on arrival and on every hash change. */
function HashTabs() {
  const rail = useRail();
  // The newest `open` (its identity changes with the rail's state); the hash is read only on
  // arrival and on a hash change, so picking another tab is never undone.
  const open = useRef(rail.open);
  useEffect(() => {
    open.current = rail.open;
  });
  useEffect(() => {
    const go = () => {
      const tab = HASH_TAB[window.location.hash.replace(/^#/, "").toLowerCase()];
      if (tab) open.current(tab);
    };
    go();
    window.addEventListener("hashchange", go);
    return () => window.removeEventListener("hashchange", go);
  }, []);
  return null;
}

/** A side-panel tab that keeps its own result line for the actions inside it. */
function Noted({ children }: { children: (onNote: (n: Note) => void) => ReactNode }) {
  const [note, setNote] = useState<Note | null>(null);
  return (
    <div className="grid min-w-0 gap-6">
      <NoteLine note={note} />
      {children(setNote)}
    </div>
  );
}

/** The working system: the activity meter across the four MicroModels, then the machines. */
function Machines({ data }: { data: Status }) {
  const rail = useRail();
  const controls = useApi<Control[]>("/api/controls");
  const carts = useApi<Cartridge[]>("/api/cartridges", 5000);
  const settings = useApi<Record<string, unknown>>("/api/settings");
  const bases = useApi<{ models?: BaseModel[] }>("/api/cartridges/bases", 15000);
  const surgery = useApi<{ roles?: Record<string, { job_filters?: { job: string; enabled?: boolean }[] }> }>("/api/surgery", 15000);
  const drones = useMemo(() => droneSets(settings.data), [settings.data]);
  const { tree } = useTopology();
  const act = useActivity(tree);
  const live = useLive();
  const now = useNow();
  const roleMap = useRoleMap();
  const [note, setNote] = useState<Note | null>(null);
  const [showDrones, setShowDrones] = useState(false);
  // The newest live step per MicroModel slot, seen since the page opened (from the live stream or
  // the polled timeline): it lights that machine's dot for a few seconds.
  const [pulses, setPulses] = useState<Record<string, Pulse>>({});
  useLiveEvents(
    useCallback((e: LiveEvent) => {
      // Keyed by the slot its model or stage names, and by the model name itself (a card id).
      const keys = [slotOf(e.model), slotOf(e.stage), e.model].filter((k): k is string => !!k);
      if (keys.length) {
        const at = { seen: Date.now(), stage: e.stage };
        setPulses((p) => ({ ...p, ...Object.fromEntries(keys.map((k) => [k, at])) }));
      }
    }, []),
  );
  const models = data.models as StatusModel[];
  const total = models.reduce((a, m) => a + m.bytes, 0);
  const serving = live.reply ? engineServing(live.reply) : null;
  const filtersOn = (model: string) =>
    Object.entries(surgery.data?.roles ?? {})
      .filter(([role]) => roleMap.modelOf(role) === model)
      .flatMap(([role, r]) => (r.job_filters ?? []).filter((f) => f.enabled !== false).map((f) => `${role}/${f.job}`));
  const rows = models.map((m) => {
    const roles = roleMap.rolesOf(m.id);
    const ml = typeof m.layers === "number" ? modelLayers(act, tree, m.id, m.layers, roles) : null;
    // With no engine serving, the reply's MicroModel rows are the interface's own zeros, not readings.
    const lm = serving === false ? null : liveFor(live.reply?.models, m.id, roles);
    const slots = [...new Set([m.id, ...roles.map((r) => slotOf(r)).filter((s): s is string => !!s)])];
    const pulse = slots.map((s) => pulses[s]).filter((p): p is Pulse => !!p).sort((a, b) => b.seen - a.seen)[0] ?? null;
    return { m, roles, role: roles.join(" + "), ml, lm, pulse };
  });
  const maxCalls = Math.max(1, ...rows.map((r) => r.ml?.calls ?? 0));
  const show = (id: string, tensors: boolean) => {
    const r = rows.find((x) => x.m.id === id);
    rail.show({ key: id, title: `${r?.role ? roleTitle(r.role) : "MicroModel"} · ${tensors ? "tensors" : "detail"}`, body: <ModelDetail key={`${id}:${tensors}`} model={id} tensors={tensors} /> });
  };
  const selected = rail.active === "detail" ? rail.shown?.key : undefined;
  return (
    <div className="grid min-w-0 gap-6">
      <NoteLine note={note} />
      <div className="grid gap-2">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <p
            className="mono text-[10.5px] text-faint"
            data-e2e="models-activity-note"
            title="Each layer is as bright as the real calls that ran it over the last 24 hours (the engine's activity record, refreshed as new calls land). A MicroModel no call used is dark. A MicroModel executing right now carries a cold-white rim; a fresh live step lights its dot."
          >
            {serving ? "live · layers lit by real calls in the last 24 h · a running MicroModel carries a white rim" : serving === false ? "nothing is serving · layers show the record of real calls in the last 24 h" : "live state not reported · layers show the record of real calls in the last 24 h"}
          </p>
          <button
            type="button"
            aria-pressed={showDrones}
            onClick={() => setShowDrones((s) => !s)}
            className={`press mono border-b px-0.5 text-[10px] uppercase tracking-[0.12em] ${showDrones ? "border-sense-pale text-text" : "border-line text-faint hover:text-text"}`}
            title="Draw the attention heads the drones read with as bright cells (a fixed map, not activity)"
          >
            drone heads {showDrones ? "shown" : "hidden"}
          </button>
        </div>
        <ul className="grid gap-1" data-e2e="models-meter" aria-label="Calls in the last 24 hours by MicroModel" title={act.cover}>
          {rows.map((r) => {
            const calls = r.ml?.calls ?? 0;
            const st = stateWord(r.lm, !!live.reply, serving);
            const hot = r.lm?.executing || (!!r.pulse && now > 0 && now - r.pulse.seen < FRESH_MS);
            return (
              <li key={r.m.id}>
                <button type="button" onClick={() => show(r.m.id, false)} className="press grid w-full grid-cols-[7.5rem_minmax(0,1fr)_auto] items-center gap-3 text-left" title={`${r.role ? roleTitle(r.role) : r.m.id}: ${st}`}>
                  <span className="mono flex min-w-0 items-center gap-2 text-[10.5px] uppercase tracking-[0.14em] text-muted">
                    <span aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 ${hot ? "bg-signal motion-safe:animate-pulse" : r.lm?.resident ? "bg-verified" : "border border-quiet"}`} />
                    <span className="truncate">{r.role ? roleTitle(r.role) : "no role"}</span>
                  </span>
                  <span className="relative h-1.5 bg-line/60">
                    <span className={`absolute inset-y-0 left-0 transition-[width] duration-500 ${hot ? "bg-signal" : "bg-sense-pale/80"}`} style={{ width: `${r.ml ? Math.round((calls / maxCalls) * 100) : 0}%` }} />
                  </span>
                  <span className="mono tnum flex w-[9.5rem] justify-end gap-3 text-[10.5px] text-faint">
                    {r.ml ? (
                      <>
                        <span className={calls ? "text-muted" : ""}>{calls} · 24 h</span>
                        <span className={r.ml.hour ? "text-text" : ""}>{r.ml.hour} · 1 h</span>
                      </>
                    ) : (
                      "not reported"
                    )}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
      <div className="grid grid-cols-1 gap-x-8 gap-y-10 md:grid-cols-2 2xl:grid-cols-4">
        {rows.map(({ m, role, ml, lm, pulse }) => (
          <ModelMachine
            key={m.id}
            m={m}
            role={role}
            base={bases.data?.models?.find((b) => b.model === m.id)}
            basesError={!!bases.error}
            ml={ml}
            cover={act.cover}
            lm={lm}
            liveReported={!!live.reply}
            serving={serving}
            drones={showDrones ? drones.get(m.id) : undefined}
            carts={carts.data ?? []}
            onCard={data.loaded?.find((x) => x.id === m.id)?.layers_on_card}
            ctl={controls.data?.find((c) => c.key === precisionKey(m.id))}
            filters={filtersOn(m.id)}
            share={total ? m.bytes / total : 0}
            pulse={pulse}
            now={now}
            selected={selected === m.id}
            onSelect={() => show(m.id, false)}
            onInspect={() => show(m.id, true)}
            onNote={setNote}
          />
        ))}
      </div>
    </div>
  );
}

export default function Models() {
  const { data, error } = useApi<Status>("/api/status", 5000);
  const live = useLive();
  const total = data?.models.reduce((a, m) => a + m.bytes, 0) ?? 0;
  const resident = live.reply ? live.reply.models.filter((m) => m.resident).length : null;
  const running = live.reply ? live.reply.models.filter((m) => m.executing).length : null;
  const rail = [
    { id: "system", label: "Live", body: <SystemPanel events={8} /> },
    {
      id: "downloads",
      label: "Downloads",
      body: (
        <Noted>
          {(onNote) => (
            <>
              <DownloadsPanel onNote={onNote} />
              {data && (
                <RailSection title="Models folder">
                  <div className="mono break-all text-[11px] text-muted">
                    <Masked id="models_dir" shown={maskPath(data.models_dir, data.root)} full={data.models_dir} />
                  </div>
                  <div className="mono mt-1 text-[10px] text-quiet">set in designs/models.json or by NANO_MODELS</div>
                </RailSection>
              )}
            </>
          )}
        </Noted>
      ),
    },
    { id: "libraries", label: "Libraries", body: <CartridgeSources /> },
    {
      id: "routing",
      label: "Routing",
      body: (
        <Noted>
          {(onNote) => (
            <>
              <RolesPanel onNote={onNote} />
              <RoutingPanel onNote={onNote} />
            </>
          )}
        </Noted>
      ),
    },
    { id: "lenses", label: "Lenses", body: <LensesPanel compact /> },
  ];
  const about = (
    <div className="read grid gap-3 text-[12px] leading-relaxed text-muted">
      <p>The four MicroModels (Embedder, Reranker, Reader, Writer), each named by its role and each running on its Neuromorph base cartridge.</p>
      <p className="mono grid gap-1 text-[11px]">
        <span className="flex items-center gap-2">
          <i aria-hidden className="inline-block h-1.5 w-1.5 bg-learned" /> the base cartridge it runs on
        </span>
        <span className="flex items-center gap-2 text-faint">
          <i aria-hidden className="inline-block h-1.5 w-1.5 rounded-full border border-quiet" /> the upstream open weights it was built from: not the MicroModel, never altered, and needed to run: each base cartridge is seated against them (built against them and checked output for output before it serves, nmengn base::seat), and each MicroModel reads its tokenizer and configuration from them
        </span>
      </p>
      <p>Each machine is drawn as its layers by heads, as the engine reports them. A layer is as bright as the real calls that ran it over the last 24 hours; a MicroModel no call used is dark. A MicroModel executing right now carries a cold-white rim, and a fresh live step lights its dot. Cartridges, job filters and preference cartridges dock beside the layers they touch; a project lens runs along the left edge.</p>
      <p>
        Precision and the base cartridge choice apply at the next start and are logged in{" "}
        <Link href="/audit/" className="link">
          Audit
        </Link>
        , with undo. The model files are only read, never written.
      </p>
    </div>
  );
  return (
    <Workspace
      title="Models"
      railKey="models"
      defaultTab="downloads"
      rail={rail}
      about={about}
      status={
        <>
          <LiveBadge />
          {data && <span>{data.models.length} MicroModels</span>}
          {resident !== null && data && (
            <span title="Held in memory now, from the live feed">
              {resident} of {data.models.length} resident
            </span>
          )}
          {running ? <span className="text-signal">{running} executing</span> : null}
          {data && <span title="The model files are only read, never written">{bytes(total)} read only</span>}
          <HFDownloadButton label="HF downloads" />
        </>
      }
    >
      <HashTabs />
      {error && !data && <Failed error={error} />}
      {!data && !error && <Placeholder className="h-[480px]" what="models" />}
      {data && <Machines data={data} />}
    </Workspace>
  );
}
