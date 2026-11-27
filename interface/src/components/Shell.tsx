"use client";

import Link, { useLinkStatus } from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import DotText from "@/components/DotText";
import { Dot, Masked, Mark } from "@/components/ui";
import { maskName } from "@/lib/privacy";
import { useWorking } from "@/lib/live";
import { ago, useApi, type Projects, type Status } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { engineServing, eventGroup, GROUP_NAMES, nowLine, stageGroup, writerState } from "@/lib/livefeed";
import { useLive } from "@/lib/livehook";
import { parseStorage, STORAGE_API } from "@/lib/storage";

type NavItem = { href: string; label: string; icon: string };

const NAV: NavItem[] = [
  { href: "/", label: "Live Activity", icon: "M4 11.5 12 5l8 6.5M6.5 10v9h11v-9" },
  { href: "/topology/", label: "Topology", icon: "M12 9a3 3 0 1 1 0 6 3 3 0 0 1 0-6ZM12 3.5V7M12 17v3.5M3.5 12H7M17 12h3.5M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6" },
  { href: "/models/", label: "Models", icon: "M4 7.5 12 4l8 3.5-8 3.5-8-3.5Zm0 4.5 8 3.5 8-3.5M4 16.5 12 20l8-3.5" },
  { href: "/cartridges/", label: "Cartridges", icon: "M5 7h14v10H5zM9 7v10M13 7v10M17 7v10" },
  { href: "/parameters/", label: "Parameters", icon: "M4 6h16M4 12h16M4 18h16M8 4v4M14 10v4M18 16v4" },
  { href: "/anatomy/", label: "Anatomy", icon: "M5 5h6v6H5zM13 5h6v6h-6zM5 13h6v6H5zM13 13h6v6h-6z" },
  { href: "/memory/", label: "Memory", icon: "M6 6h12v4H6zM6 14h12v4H6zM9 8h.01M9 16h.01" },
  { href: "/learning/", label: "Learning", icon: "M4 17.5 9 12l3.5 3L20 7.5M15 7.5h5v5" },
  { href: "/storage/", label: "Storage", icon: "M4 6.5h16v4H4zM4 13.5h16v4H4zM7 8.5h.01M7 15.5h.01M11 8.5h6M11 15.5h6" },
  { href: "/audit/", label: "Audit", icon: "M6 4.5h12v15H6zM9 8.5h6M9 12h6M9 15.5h3.5" },
  { href: "/settings/", label: "Settings", icon: "M12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5ZM12 3.5v2.5M12 18v2.5M3.5 12H6M18 12h2.5M6 6l1.8 1.8M16.2 16.2 18 18M6 18l1.8-1.8M16.2 7.8 18 6" },
];

/** 2026-11-26: the pages in three groups, each under a hairline eyebrow: what you watch, the
 *  engine's parts, and the system around them. */
const SECTIONS: { title: string; hrefs: string[] }[] = [
  { title: "Observe", hrefs: ["/", "/topology/", "/anatomy/"] },
  { title: "Engine", hrefs: ["/models/", "/cartridges/", "/parameters/", "/memory/", "/learning/"] },
  { title: "System", hrefs: ["/storage/", "/audit/", "/settings/"] },
];
const NAV_SECTIONS: { title: string; items: NavItem[] }[] = SECTIONS.map((s) => ({
  title: s.title,
  items: s.hrefs.map((h) => NAV.find((n) => n.href === h)).filter((n): n is NavItem => !!n),
}));

function Icon({ d }: { d: string }) {
  return (
    <svg aria-hidden viewBox="0 0 24 24" className="h-4 w-4 shrink-0" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d={d} />
    </svg>
  );
}

function isActive(path: string | null, href: string) {
  if (!path) return false;
  return href === "/" ? path === "/" : path.startsWith(href.replace(/\/$/, ""));
}

/** When this interface was built, fixed at build time. */
const BUILD = process.env.NEXT_PUBLIC_BUILD_STAMP ?? "unknown";

type AirGap = { air_gap?: boolean; loopback_only?: boolean; lan_listeners?: number };

/** What leaves this machine, as GET /api/airgap reports it; nothing is claimed before it answers. */
function IsolationLine() {
  const { data, error } = useApi<AirGap>("/api/airgap", 30000);
  if (!data) return error ? <div className="mt-2.5 text-faint">Network isolation not reported.</div> : null;
  const lan = (data.lan_listeners ?? 0) > 0 || data.loopback_only === false;
  const words = data.air_gap
    ? "Air gap on. The Nano sends nothing off this machine."
    : lan
      ? "Shared on your local network (LAN on)."
      : data.loopback_only
        ? "Local only. The Nano sends nothing off this machine; downloads only when you ask."
        : "Network isolation not reported.";
  return <div className={`mt-2.5 ${lan && !data.air_gap ? "text-attention" : "text-faint"}`}>{words}</div>;
}

/* ------------------------------------------------------------------ system meters */

const NOT_MEASURED = "not measured";
/** On a short window (under 860 px high) the strip keeps its five main readings as plain rows, so
 *  the page list above it stays in view; the rest is in the System side panel. */
const SHORT_HIDE = "[@media(max-height:860px)]:hidden";
/** Rows that show only on a tall window (over 1040 px high); they are in the System side panel too. */
const TALL_ONLY = "[@media(max-height:1040px)]:hidden";
const mib = (v: number | null | undefined) => (typeof v === "number" && Number.isFinite(v) ? (v >= 10240 ? `${(v / 1024).toFixed(1)} GiB` : `${Math.round(v)} MiB`) : null);
const gibOf = (bytes: number | null | undefined) => (typeof bytes === "number" && Number.isFinite(bytes) ? `${(bytes / 1024 ** 3).toFixed(1)} GiB` : null);

/** One meter: a name, its reading, and a bar of the reading against its limit when both are
 *  measured. `mark` is a line on the bar (a limit or a floor inside the range). A reading the
 *  engine does not give shows as "not measured" with no bar. */
function Meter({ id, name, value, share, mark, tone = "plain", title, tall = false }: { id: string; name: string; value: string | null; share?: number | null; mark?: number | null; tone?: "plain" | "live" | "warn"; title?: string; tall?: boolean | "only" }) {
  const measured = value !== null;
  const s = typeof share === "number" && Number.isFinite(share) ? Math.max(0, Math.min(1, share)) : null;
  return (
    <div className={`grid min-w-0 grid-cols-[minmax(0,1fr)] gap-[3px] ${tall === "only" ? TALL_ONLY : tall ? SHORT_HIDE : ""}`} data-e2e={`meter-${id}`} data-measured={measured ? "true" : "false"} title={title}>
      <div className="flex min-w-0 items-baseline justify-between gap-2">
        <dt className="shrink-0 uppercase tracking-[0.14em] text-quiet">{name}</dt>
        <dd className={`tnum min-w-0 truncate text-right ${!measured ? "text-quiet" : tone === "live" ? "text-signal" : tone === "warn" ? "text-attention" : "text-muted"}`}>{value ?? NOT_MEASURED}</dd>
      </div>
      {s !== null && (
        <div aria-hidden className={`relative h-[2px] bg-line ${SHORT_HIDE}`}>
          <i className={`absolute inset-y-0 left-0 block ${tone === "warn" ? "bg-attention" : "bg-sense-pale/80"}`} style={{ width: `${s * 100}%` }} />
          {typeof mark === "number" && mark > 0 && mark < 1 && <i className="absolute -top-[2px] block h-[6px] w-px bg-signal" style={{ left: `${mark * 100}%` }} />}
        </div>
      )}
    </div>
  );
}

/** The system strip: the meters an operator reads at a glance, on every page. Card memory against
 *  its limit, system memory, processor threads, the four MicroModels' residency, calls in the last
 *  hour, the idle Writer, background work with a measurable end, and disk free against its floor.
 *  Every figure is a reading of `/api/live`, `/api/activity` or `/api/storage`, kept fresh by the
 *  live feed; a figure no process reports is shown as not measured. */
function SystemMeters() {
  const { reply, events, mode } = useLive();
  const hour = useApi<{ calls?: number }>("/api/activity?window=1h", 15000);
  const disk = useApi<unknown>(STORAGE_API, 60000);
  const serving = engineServing(reply);
  const on = serving === true;
  const cap = reply?.capacity ?? null;
  const lim = cap?.limits ?? null;
  const vram = reply?.vram ?? null;

  // Card: what the driver reports used, against the user's limit when one is set, else the card.
  const cardTotal = vram?.total_mb ?? null;
  const cardUsed = vram?.used_mb ?? null;
  const cardCap = lim?.vram?.cap_mb ?? null;
  const cardLimit = cardCap ?? cardTotal;
  const cardValue = on && cardUsed !== null && cardTotal !== null ? `${Math.round(cardUsed)} / ${Math.round(cardLimit ?? cardTotal)} MiB` : on && reply?.backend === "cpu" ? "processor only" : null;
  const cardShare = on && cardUsed !== null && cardTotal ? cardUsed / cardTotal : null;

  // System memory: the Nano's own use, and what the machine has free.
  const ramOwn = lim?.ram?.used_mb ?? null;
  const ramFree = cap?.host_available_mb ?? null;
  const ramTotal = cap?.host_total_mb ?? null;
  const ramValue = ramFree !== null && ramTotal !== null ? `${mib(ramFree)} free` : null;
  const ramShare = ramFree !== null && ramTotal ? (ramTotal - ramFree) / ramTotal : null;
  const ramCap = lim?.ram?.cap_mb ?? cap?.ram_ceiling_mb ?? null;

  const threads = cap?.threads ?? null;
  const hw = cap?.hardware_threads ?? null;

  const models = on ? (reply?.models ?? []) : [];
  const resident = models.filter((m) => m.resident).length;

  const calls = typeof hour.data?.calls === "number" ? hour.data.calls : null;
  const w = on ? (reply?.writer ?? null) : null;
  const writerValue = w ? `${w.queue ?? 0} queued · ${w.in_session ? "writing" : w.waiting ? "held back" : w.held ? "loaded" : "not loaded"}` : null;

  const st = parseStorage(disk.data);
  const free = st?.free_bytes ?? null;
  const floor = st?.floor_bytes ?? null;
  const diskTotal = st?.disk_total_bytes ?? null;
  const below = st?.below_floor === true || (free !== null && floor !== null && free < floor);
  const diskValue = free !== null ? `${gibOf(free)}${floor !== null ? ` · floor ${gibOf(floor)}` : " free"}` : null;

  const progress = reply?.progress ?? [];
  const newest = events.length ? events[events.length - 1] : null;
  const state = !reply ? "connecting" : !on && serving === false ? "no engine serving" : reply.state === "busy" ? "working" : "idle";
  const now = nowLine(reply);

  return (
    <section aria-label="System meters" className="mono min-w-0 border-t border-line pt-2.5 text-[10px]" data-e2e="system-strip" data-serving={serving === null ? "unknown" : String(serving)} data-state={state} data-feed={mode}>
      <div className="mb-1.5 flex items-center gap-2" aria-live="polite">
        <span aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 ${state === "working" ? "bg-signal" : state === "idle" ? "bg-verified" : "border border-quiet"}`} />
        <span className={`min-w-0 truncate uppercase tracking-[0.14em] ${state === "working" ? "text-signal" : "text-muted"}`} title={now ?? (newest ? `last event: ${GROUP_NAMES[eventGroup(newest)]} · ${newest.stage}` : undefined)}>
          {state === "working" && now ? now.split(" · ")[0] : state}
        </span>
      </div>
      <dl className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-[5px]">
        <Meter
          id="card"
          name="Card"
          value={cardValue}
          share={cardShare}
          mark={cardCap !== null && cardTotal ? cardCap / cardTotal : null}
          tone={cardShare !== null && cardShare > 0.95 ? "warn" : "plain"}
          title={on ? `Graphics card memory in use, as the driver reports it${cardCap !== null ? `, against your limit of ${Math.round(cardCap)} MiB` : cardTotal !== null ? `, of ${Math.round(cardTotal)} MiB` : ""}` : "Card memory is read by a serving engine; none is serving"}
        />
        <Meter id="ram" name="RAM" value={ramValue} share={ramShare} mark={ramCap !== null && ramTotal ? ramCap / ramTotal : null} title={`System memory${ramOwn !== null ? `: the Nano holds ${mib(ramOwn)}` : ""}${ramFree !== null && ramTotal !== null ? `; ${mib(ramFree)} free of ${mib(ramTotal)}` : ""}${ramCap !== null ? `; the Nano's ceiling is ${mib(ramCap)}` : ""}`} />
        <Meter tall="only" id="threads" name="Threads" value={threads !== null ? `${threads}${hw !== null ? ` of ${hw}` : ""}` : null} share={threads !== null && hw ? threads / hw : null} title="Processor threads the engine plans on, of the machine's hardware threads" />
        <div className="flex min-w-0 items-center justify-between gap-2" data-e2e="meter-models" data-measured={on ? "true" : "false"} title={on ? models.map((m) => `${m.role}: ${m.executing ? "executing" : m.resident ? "resident" : "not held"}${m.vram_mb ? `, ${Math.round(m.vram_mb)} MiB on the card` : ""}`).join("; ") : "Residency is reported by a serving engine; none is serving"}>
          <dt className="shrink-0 uppercase tracking-[0.14em] text-quiet">Models</dt>
          <dd className={`flex items-center gap-1.5 ${on ? "text-muted" : "text-quiet"}`}>
            {on ? (
              <>
                <span aria-hidden className="flex gap-[3px]">
                  {models.map((m) => (
                    <i key={m.role} data-role={m.role} data-state={m.executing ? "executing" : m.resident ? "resident" : "not-held"} className={`block h-[7px] w-[7px] ${m.executing ? "bg-signal" : m.resident ? "bg-verified" : "border border-quiet"}`} />
                  ))}
                </span>
                <span className="tnum whitespace-nowrap">
                  {resident} of {models.length} held
                </span>
              </>
            ) : (
              NOT_MEASURED
            )}
          </dd>
        </div>
        <Meter tall id="calls" name="Calls" value={calls !== null ? `${calls} in the last hour` : null} title="Calls the audit recorded in the last hour" />
        <Meter id="writer" name="Writer" value={writerValue} tone={w?.in_session ? "live" : "plain"} title={w ? `The idle Writer: ${writerState(w)}; ${w.queue ?? 0} jobs queued, ${w.results ?? 0} written, ${w.sessions ?? 0} sessions, ${w.preempted ?? 0} preempted by calls` : "The idle Writer's queue is reported by a serving engine; none is serving"} />
        {progress.map((p) => (
          <Meter key={p.kind} id={`progress-${p.kind}`} name={p.kind === "dense" ? "Index" : GROUP_NAMES[stageGroup(p.kind)]} value={p.done !== null && p.total !== null ? `${p.done} of ${p.total}` : p.done !== null ? `${p.done} done` : "running"} share={p.done !== null && p.total ? p.done / p.total : null} tone="live" title="Background work in progress, as the engine counts it" />
        ))}
        <Meter id="disk" name="Disk" value={diskValue} share={free !== null && diskTotal ? (diskTotal - free) / diskTotal : null} mark={floor !== null && diskTotal ? (diskTotal - floor) / diskTotal : null} tone={below ? "warn" : "plain"} title={free !== null ? `${gibOf(free)} free on the drive holding the Nano's folder${floor !== null ? `; below the floor of ${gibOf(floor)} free, builds refuse to start` : ""}` : "Disk space not reported"} />
      </dl>
    </section>
  );
}

function EngineStatus({ compact = false }: { compact?: boolean }) {
  const { data, error } = useApi<Status>("/api/status", 5000);
  const projects = useApi<Projects>("/api/projects", 15000);
  const now = useNow();
  const online = !!data && !error;
  // /api/status answering means the interface server is up; an engine is serving only when it
  // reports serving processes.
  const servingN = online && typeof data.serving === "number" ? data.serving : null;
  const engineOn = servingN !== null && servingN > 0;
  const current = [...(projects.data?.projects ?? [])].sort((a, b) => (b.last_ms ?? 0) - (a.last_ms ?? 0))[0];
  return (
    <div className={compact ? "text-xs" : "border-t border-line pt-4 text-xs"}>
      <div className="flex items-center gap-2">
        <Dot tone={!data && !error ? "muted" : !online ? "error" : engineOn ? "verified" : "attention"} />
        <span className={`whitespace-nowrap ${!online ? (error ? "text-error" : "text-muted") : engineOn ? "text-text" : "text-muted"}`}>
          {!data && !error
            ? "Connecting"
            : !online
              ? "Offline"
              : engineOn
                ? servingN > 1 ? `Serving ${servingN}` : "Serving"
                : servingN === null
                  ? (compact ? "Up" : "Interface up · serving not reported")
                  : compact ? "Not serving" : "Interface up · not serving"}
        </span>
      </div>
      {!compact && (
        <div className={`mono mt-1.5 text-[10px] text-quiet ${SHORT_HIDE}`} data-build>
          {data ? `v${data.version} · ` : ""}build {BUILD} UTC
        </div>
      )}
      {!compact && (
        <div className={SHORT_HIDE}>
          <div className="mt-2.5 text-faint">Last served project</div>
          {current ? (
            <div className="mt-0.5 flex items-baseline gap-2">
              <span className="mono min-w-0 flex-1 truncate text-muted">
                <Masked id={`project:${current.key}`} shown={`project ${maskName(current.key)}`} full={current.key} />
              </span>
              {current.last_ms && now > 0 && <span className="shrink-0 text-faint">{ago(current.last_ms, now)}</span>}
            </div>
          ) : (
            <div className="mt-0.5 text-muted">{projects.data ? "none yet" : "–"}</div>
          )}
          <IsolationLine />
        </div>
      )}
    </div>
  );
}

export default function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const working = useWorking();
  const [menu, setMenu] = useState<{ open: boolean; at: string | null }>({ open: false, at: path });
  // The menu closes when the page changes.
  if (menu.at !== path) setMenu({ open: false, at: path });
  useEffect(() => {
    if (!menu.open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setMenu((m) => ({ ...m, open: false }));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menu.open]);

  return (
    <div className="relative flex min-h-dvh">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-bg-2 focus:px-3 focus:py-2 focus:text-sm">
        Skip to content
      </a>
      <aside className="sticky top-0 hidden h-dvh w-[216px] shrink-0 flex-col border-r border-line px-3 py-6 lg:flex">
        <Link href="/" className="press mb-6 flex shrink-0 flex-col gap-2.5 px-3 py-1 text-muted [@media(max-height:1040px)]:mb-4 [@media(max-height:860px)]:mb-3" aria-label="Neuromorph Nano, live activity">
          <span className="flex items-center gap-2.5">
            <Mark className="h-5 w-5" working={working} />
            <span className="text-text">
              <DotText text="NEUROMORPH" size={11} label="Neuromorph" />
            </span>
          </span>
          <span className="mono pl-[30px] text-[10px] uppercase tracking-[0.32em] text-faint">Nano · local</span>
        </Link>
        <nav aria-label="Pages" className="-mr-1 flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1 [@media(max-height:860px)]:gap-2.5">
          {NAV_SECTIONS.map((sec) => (
            <div key={sec.title} className="grid gap-0.5">
              <div className="eyebrow border-b border-line/60 px-3 pb-1.5 text-[10px] text-quiet">{sec.title}</div>
              {sec.items.map((n) => {
                const active = isActive(path, n.href);
                return (
                  <Link
                    key={n.href}
                    href={n.href}
                    aria-current={active ? "page" : undefined}
                    className={`press group relative flex h-9 shrink-0 items-center gap-3 px-3 text-[13px] [@media(max-height:1040px)]:h-[34px] [@media(max-height:860px)]:h-8 ${active ? "text-text" : "text-faint hover:text-text"}`}
                  >
                    {active && <span aria-hidden className="absolute top-2 bottom-2 left-0 w-[2px] bg-sense-pale shadow-[0_0_10px_rgb(127_176_255/0.6)]" />}
                    <span className={active ? "text-text" : "text-quiet group-hover:text-muted"}>
                      <Icon d={n.icon} />
                    </span>
                    {n.label}
                    <LinkPending />
                  </Link>
                );
              })}
            </div>
          ))}
        </nav>
        <div className="mt-4 grid shrink-0 grid-cols-[minmax(0,1fr)] gap-3 [@media(max-height:860px)]:mt-2">
          <SystemMeters />
          <EngineStatus />
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-40 flex h-14 items-center gap-3 border-b border-line bg-bg px-4 lg:hidden">
          <Link href="/" className="flex items-center gap-2.5 text-muted">
            <Mark className="h-5 w-5" working={working} />
            <span className="text-text">
              <DotText text="NEUROMORPH" size={10} label="Neuromorph" />
            </span>
            <span className="flex flex-col leading-none">
              <span className="mono text-[10px] uppercase tracking-[0.2em] text-faint">Nano</span>
              <span className="mono mt-1 text-[9px] text-quiet" data-build>
                build {BUILD}
              </span>
            </span>
          </Link>
          <div className="ml-auto">
            <EngineStatus compact />
          </div>
          <button
            onClick={() => setMenu((m) => ({ ...m, open: !m.open }))}
            aria-expanded={menu.open}
            aria-controls="nm-menu"
            aria-label={menu.open ? "Close menu" : "Open menu"}
            className="press -mr-2 inline-flex h-10 w-10 items-center justify-center rounded-lg text-muted hover:text-text"
          >
            <svg aria-hidden viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.4} strokeLinecap="round">
              {menu.open ? <path d="M6 6l12 12M18 6 6 18" /> : <path d="M4 8h16M4 16h16" />}
            </svg>
          </button>
        </header>
        {menu.open && (
          <div id="nm-menu" className="fixed inset-x-0 top-14 z-40 lg:hidden">
            <button aria-label="Close menu" className="fixed inset-0 top-14 bg-bg/70" onClick={() => setMenu((m) => ({ ...m, open: false }))} />
            <nav aria-label="Pages" className="open-from relative m-0 max-h-[calc(100dvh-3.5rem)] overflow-y-auto border-b border-line bg-bg p-2" style={{ transformOrigin: "top right" }}>
              {NAV_SECTIONS.map((sec) => (
                <div key={sec.title} className="grid">
                  <div className="eyebrow mt-2 border-b border-line/60 px-3 pb-1.5 text-[10px] text-quiet">{sec.title}</div>
                  {sec.items.map((n) => {
                    const active = isActive(path, n.href);
                    return (
                      <Link
                        key={n.href}
                        href={n.href}
                        aria-current={active ? "page" : undefined}
                        className={`press flex h-11 items-center gap-3 px-3 text-sm ${active ? "text-text underline decoration-sense-pale underline-offset-4" : "text-muted"}`}
                      >
                        <span className={active ? "text-text" : "text-faint"}>
                          <Icon d={n.icon} />
                        </span>
                        {n.label}
                      </Link>
                    );
                  })}
                </div>
              ))}
              <div className="mt-2 grid grid-cols-[minmax(0,1fr)] gap-4 border-t border-line p-2 pt-3">
                <SystemMeters />
                <EngineStatus />
              </div>
            </nav>
          </div>
        )}
        <main id="main" className="w-full min-w-0 flex-1 px-4 py-6 md:px-8 md:py-7 2xl:px-10">
          {children}
        </main>
      </div>
    </div>
  );
}

/** A page switch whose code is still arriving: the link says so, and the page shown stays until
 *  the next one can draw (the centre is never blank). */
function LinkPending() {
  const { pending } = useLinkStatus();
  if (!pending) return null;
  return (
    <span role="status" data-loading="page" className="mono ml-auto text-[9.5px] uppercase tracking-[0.14em] text-quiet">
      loading
    </span>
  );
}
