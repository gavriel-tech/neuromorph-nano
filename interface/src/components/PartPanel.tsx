"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel"): the part's details can be
// shown inside the page's side panel (embedded), where the panel's own title names the part.

import Link from "next/link";
import { createContext, useContext, useState, type ReactNode } from "react";
import { ControlSurface, NoteLine } from "@/components/Controls";
import LiveState, { countsOf, currentCall, LastCall, liveLine, SystemsNow } from "@/components/LiveState";
import { CallText, Dot, Pill, Placeholder, Reticle } from "@/components/ui";
import { actFor, msText, NO_ACT, tierOfCell, type ActView } from "@/lib/activity";
import { ago, bits, bytes, useApi, type AuditRow, type Cartridge, type Control, type Project, type Status } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { sortRows, stepJob } from "@/lib/live";
import { figure, pathHash, portsOf, resolvePart, within, type TNode, type Tree } from "@/lib/topology";

/** True inside the side panel: sections drop their own side padding. */
const Flush = createContext(false);

function Section({ title, children }: { title: string; children: ReactNode }) {
  const flush = useContext(Flush);
  return (
    <section className={flush ? "border-t border-line/60 py-4" : "border-t border-line px-4 py-4 md:px-5"}>
      <h3 className="eyebrow mb-3 flex items-center gap-2 text-muted">
        <Reticle />
        {title}
      </h3>
      {children}
    </section>
  );
}

/** Short text for a design value. */
function brief(v: unknown): string {
  if (v === null || v === undefined) return "none";
  if (typeof v === "string") return v.length > 64 ? v.slice(0, 63) + "…" : v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  if (Array.isArray(v)) return `${v.length} items`;
  return `${Object.keys(v as object).length} fields`;
}

/** The side panel (a bottom sheet on narrow screens) for the selected part. With nothing
 *  selected it shows the live state instead: the breadcrumb above already names the organism. */
export default function PartPanel({
  node,
  tree,
  status,
  carts,
  rows,
  settings,
  projects,
  onGo,
  act = NO_ACT,
  embedded = false,
  topologyLink = false,
}: {
  node: TNode;
  tree: Tree;
  status: Status | null;
  carts: Cartridge[] | null;
  rows: AuditRow[];
  settings: Record<string, unknown> | null;
  projects: Project[] | null;
  onGo: (path: string[]) => void;
  act?: ActView;
  /** Inside the page's side panel: the body only, its title given by the panel. */
  embedded?: boolean;
  /** A link to the part in the full topology (for pages other than the topology). */
  topologyLink?: boolean;
}) {
  const root = node.path.length === 0;
  const controls = useApi<Control[]>("/api/controls");
  const now = useNow();
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState<{ text: string; tone: "verified" | "error" } | null>(null);
  const { ins, outs } = portsOf(node);
  const deps = node.card && (node.path.length <= 1 || node.role === "group") ? node.card.deps : [];
  const ctl = (controls.data ?? []).filter((c) => node.controls.includes(c.key));
  const activity = rows
    .filter((r) => {
      if (node.path.length === 0) return true;
      const parts = Array.isArray(r.steps) && r.steps.length ? r.steps.map((st) => st.part) : Array.isArray(r.parts) ? r.parts : [];
      return parts.some((p) => resolvePart(tree, p).some((n) => within(n, node)));
    })
    .slice(-6)
    .reverse();
  const config = node.anatomy && Object.keys(node.anatomy.config).length ? node.anatomy.config : node.card && node.path.length <= 1 ? node.card.config_default : null;
  const designs = node.settings.filter((s) => settings && s in settings);

  const recent = (
    <Section title="Recent activity">
      {activity.length === 0 ? (
        <div className="text-xs text-faint">No recent call named this component. Calls appear here as they run.</div>
      ) : (
        <ol className="flex flex-col gap-2">
          {activity.map((r) => {
            // What this component's own steps reported in the call (pieces read, handed up, rounds).
            const counts = root || !Array.isArray(r.steps) ? null : countsOf(r.steps.filter((st) => resolvePart(tree, st.part).some((n) => within(n, node))));
            return (
              <li key={r.id} className="text-xs">
                <div className="flex items-baseline gap-2">
                  <Dot tone={r.ok === false ? "error" : r.state === "running" ? "signal" : "muted"} className="translate-y-[-1px]" />
                  <span className="mono min-w-0 flex-1 truncate text-muted">
                    <CallText id={r.id} label={r.label} tool={r.tool} />
                    {r.ok === false && typeof r.error === "string" && <span className="text-faint"> · {r.error}</span>}
                  </span>
                  <span className="mono tnum shrink-0 text-faint">{msText(r.ms)}</span>
                  {now > 0 && <span className="shrink-0 text-faint">{ago(r.time_ms, now)}</span>}
                </div>
                {counts && (
                  <div className="mono truncate pl-3.5 text-[10.5px] text-faint" title={counts}>
                    {counts}
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </Section>
  );

  const activityFig = <ActivityFigures node={node} act={act} now={now} />;

  const body = root ? (
    <>
      {embedded ? (
        <>
          <LastCall tree={tree} rows={rows} flush />
          <SystemsNow tree={tree} rows={rows} status={status} carts={carts} projects={projects} onGo={onGo} flush />
        </>
      ) : (
        <LiveState tree={tree} rows={rows} status={status} carts={carts} projects={projects} onGo={onGo} />
      )}
      {activityFig}
      {recent}
    </>
  ) : (
    <>
      <div className={embedded ? "pb-4" : "px-4 pt-4 pb-4 md:px-5"}>
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            {!embedded && <h2 className="mono text-[15px] font-normal uppercase leading-snug tracking-[0.22em] text-text">{node.label}</h2>}
            {node.sub && <div className="mt-0.5 text-xs text-faint">{node.sub}</div>}
            {node.role === "tentacle" && <TentacleJob node={node} rows={rows} now={now} />}
            {embedded && figure(node, status) && <div className="mono mt-1 text-[11px] text-muted">{figure(node, status)}</div>}
          </div>
          {!node.present && <Pill tone="attention">not in this build</Pill>}
        </div>
        <p className="mt-3 text-[13px] leading-relaxed text-muted">{node.about}</p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {node.ids.slice(0, 3).map((id) => (
            <span key={id} className="mono text-[11px] text-quiet">
              {id}
            </span>
          ))}
          {node.sub_hash && (
            <span className="mono text-[11px] text-quiet" title={node.sub_hash}>
              sealed {node.sub_hash.slice(0, 8)}
            </span>
          )}
        </div>
        {(topologyLink || (embedded && node.parent)) && (
          <div className="mono mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
            {embedded && node.parent && (
              <button type="button" onClick={() => onGo(node.parent!.path)} className="press border-b border-line-strong text-muted hover:text-text">
                up to {node.parent.path.length ? node.parent.label : "the organism"}
              </button>
            )}
            {topologyLink && (
              <Link href={`/topology/${pathHash(node.path)}`} className="link">
                open in the topology
              </Link>
            )}
          </div>
        )}
      </div>

      {node.children.length > 0 && (
        <Section title={figure(node, status)}>
          <ul className="flex flex-col gap-0.5">
            {node.children.slice(0, 40).map((c) => (
              <li key={c.key} className="reveal">
                <button onClick={() => onGo(c.path)} className="press group flex min-h-11 w-full items-center gap-2.5 border-b border-line/60 text-left">
                  <Dot tone={c.cartridge?.loaded ? "learned" : "muted"} />
                  <span className="min-w-0 flex-1 truncate text-[13px] text-muted group-hover:text-text">{c.label}</span>
                  <span className="mono shrink-0 text-[11px] text-faint">{c.children.length ? c.children.length : ""}</span>
                </button>
              </li>
            ))}
          </ul>
        </Section>
      )}

      {activityFig}

      {recent}

      {node.model && (
        <Section title="Model">
          <div className="flex flex-col gap-1 text-xs">
            <Row k="Precision">{bits(node.model.precision)}</Row>
            <Row k="Size">{bytes(node.model.bytes)}</Row>
            <Row k="Card">{node.model.card ? "present" : "missing"}</Row>
            {node.model.main_file && <Row k="Main file">{node.model.main_file}</Row>}
          </div>
          <CartridgeSlots carts={(carts ?? []).filter((c) => c.base === node.model?.id)} />
        </Section>
      )}

      {node.cartridge && (
        <Section title="Cartridge">
          <div className="flex flex-col gap-1 text-xs">
            <Row k="Kind">{node.cartridge.kind}</Row>
            <Row k="Base">{node.cartridge.base}</Row>
            <Row k="Data">{node.cartridge.data_class}</Row>
            <Row k="Precision">{node.cartridge.quantization}</Row>
            <Row k="Scopes">{node.cartridge.scopes.join(", ") || "any"}</Row>
            <Row k="Check">{node.cartridge.check}</Row>
            <Row k="State">{node.cartridge.loaded ? "loaded" : "not loaded"}</Row>
          </div>
        </Section>
      )}

      {(node.key === "cartridges" || node.role === "learning") && !node.cartridge && (
        <Section title="Cartridges loaded">
          <CartridgeSlots carts={carts ?? []} />
        </Section>
      )}

      {(ins.length > 0 || outs.length > 0) && (
        <Section title="Ports">
          <ul className="flex flex-col gap-1 text-xs">
            {ins.map((p) => (
              <li key={"i" + p.name} className="flex items-center gap-2">
                <span className="w-7 text-faint">in</span>
                <span className="mono text-text">{p.name}</span>
                <span className="text-faint">{p.kind}</span>
                {!p.required && <span className="text-faint">optional</span>}
              </li>
            ))}
            {outs.map((p) => (
              <li key={"o" + p.name} className="flex items-center gap-2">
                <span className="w-7 text-faint">out</span>
                <span className="mono text-text">{p.name}</span>
                <span className="text-faint">{p.kind}</span>
              </li>
            ))}
          </ul>
        </Section>
      )}

      {deps.length > 0 && (
        <Section title="Depends on">
          <div className="flex flex-wrap gap-1.5">
            {deps.map((d) => {
              const home = tree.crateHome.get(d);
              const target = home?.children.find((c) => c.crate === d && c.role === "group") ?? home?.children.flatMap((c) => c.children).find((c) => c.crate === d && c.role === "group") ?? home;
              return (
                <button key={d} onClick={() => target && onGo(target.path)} className="press mono inline-flex h-8 items-center border-b border-line-strong text-[11px] text-muted hover:border-text hover:text-text">
                  {d}
                </button>
              );
            })}
          </div>
        </Section>
      )}

      {node.controls.length > 0 &&
        (controls.error && !controls.data ? (
          <Section title="Controls">
            <div className="mono flex items-center gap-2 text-xs text-error">
              <Dot tone="error" />
              <span>Controls did not answer ({controls.error})</span>
            </div>
          </Section>
        ) : !controls.data ? (
          <Section title="Controls">
            <Placeholder className="h-16" what="controls" />
          </Section>
        ) : ctl.length > 0 ? (
          <Section title="Controls">
            {controls.error && <div className="mono mb-2 text-[11px] text-error">Refresh failed ({controls.error}); showing last values.</div>}
            <NoteLine note={note} />
            <ControlSurface controls={ctl} compact onNote={setNote} cols={ctl.length > 1 ? "grid-cols-2" : "grid-cols-1"} />
          </Section>
        ) : null)}

      {(config || designs.length > 0) && (
        <Section title="Settings">
          {config && (
            <div className="flex flex-col gap-0.5 text-xs">
              {Object.entries(config)
                .slice(0, 8)
                .map(([k, v]) => (
                  <Row key={k} k={k}>
                    {brief(v)}
                  </Row>
                ))}
            </div>
          )}
          {designs.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {designs.map((d) => (
                <Link key={d} href={`/settings/#${d}`} className="press mono inline-flex h-8 items-center border-b border-line-strong text-[11px] text-muted hover:border-text hover:text-text">
                  designs/{d}.json
                </Link>
              ))}
            </div>
          )}
        </Section>
      )}

    </>
  );

  if (embedded)
    return (
      <Flush.Provider value>
        <div key={node.path.join("/")} className="fade-in" data-e2e="part-panel" data-part={node.path.join("/")}>
          {body}
        </div>
      </Flush.Provider>
    );

  return (
    <>
      {/* Wide screens: a side panel. */}
      <aside aria-label={root ? "Live state" : `${node.label} details`} className="hidden w-[340px] shrink-0 overflow-y-auto border-l border-line lg:block 2xl:w-[380px] lg:portrait:h-[36%] lg:portrait:w-auto lg:portrait:border-t lg:portrait:border-l-0">
        <div key={node.path.join("/")} className="fade-in">
          {body}
        </div>
      </aside>
      {/* Narrow screens: a bottom sheet with a handle that shows the name. */}
      <div className="fixed inset-x-0 bottom-0 z-30 lg:hidden">
        <div className="sheet mx-auto max-w-2xl border-t border-line-strong bg-bg" style={{ transform: open ? "translateY(0)" : "translateY(calc(100% - 64px))" }}>
          <button onClick={() => setOpen((o) => !o)} aria-expanded={open} className="flex h-16 w-full items-center gap-3 px-4 text-left">
            <span aria-hidden className="absolute top-2 left-1/2 h-px w-9 -translate-x-1/2 bg-line-3" />
            <span className="min-w-0 flex-1">
              <span className="mono block truncate text-[12px] uppercase tracking-[0.12em] text-text">{root ? "Live state" : node.label}</span>
              <span className="mono block truncate text-[11px] text-faint">{root ? liveLine(currentCall(rows, tree), now) : figure(node, status)}</span>
            </span>
            <span className="text-xs text-muted">{open ? "Close" : "Details"}</span>
          </button>
          <div className="max-h-[62vh] overflow-y-auto pb-[env(safe-area-inset-bottom)]">{body}</div>
        </div>
      </div>
    </>
  );
}

function Row({ k, children }: { k: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-0.5">
      <span className="shrink-0 text-faint">{k}</span>
      <span className="mono min-w-0 truncate text-right text-muted">{children}</span>
    </div>
  );
}

function CartridgeSlots({ carts }: { carts: Cartridge[] }) {
  if (carts.length === 0) {
    return (
      <div className="mt-3 flex items-center gap-3 text-xs text-faint" title="No cartridges loaded">
        <span aria-hidden className="dither-slot !h-4 !w-2.5" />
        No cartridges loaded
      </div>
    );
  }
  return (
    <ul className="mt-3 flex flex-col gap-1.5">
      {carts.map((c) => (
        <li key={c.id} className="flex items-center gap-2 text-xs">
          <span aria-hidden className={c.loaded ? "spectral-sw !h-4 !w-1" : "dither-slot !h-4 !w-2.5"} />
          <span className="mono min-w-0 flex-1 truncate text-muted">{c.id}</span>
          <span className="text-faint">{c.kind}</span>
          <span className="text-faint">{c.loaded ? "loaded" : "not loaded"}</span>
        </li>
      ))}
    </ul>
  );
}

/** The part's real activity: calls and time over the last hour and day, when it was last used,
 *  and where the engine reports them its sub-parts, model layers, tentacle jobs and memory tiers,
 *  and the components inside it that ran. What the drawing's cells show. */
function ActivityFigures({ node, act, now }: { node: TNode; act: ActView; now: number }) {
  if (!act.reported)
    return (
      <Section title="Activity">
        <div className="text-xs text-faint">Activity is not reported by the engine yet, so the cells of this part are drawn dim and even.</div>
      </Section>
    );
  const key = node.path.join("/");
  const a = node.path.length ? actFor(act, key) : null;
  const kids = node.children
    .map((c) => ({ c, a: act.total.get(c.path.join("/")) }))
    .filter((x) => x.a && x.a.calls_day > 0)
    .sort((x, y) => y.a!.calls_day - x.a!.calls_day)
    .slice(0, 8);
  // Sub-parts, jobs and the median are the part's own figures, not those of parts inside it.
  const o = act.own.get(key) ?? a;
  const subs = (o?.subs ?? []).filter((s) => s.calls_day > 0 && !tierOfCell(s.id)).slice(0, 10);
  const jobs = Object.entries(o?.jobs ?? {}).sort((x, y) => y[1] - x[1]);
  const ran = (a?.layers ?? []).map((c, i) => [i, c] as const).filter(([, c]) => c > 0);
  const tiers = node.role === "memory" || key.startsWith("memory/") ? [...act.cells.entries()].filter(([k]) => k.startsWith("tier:")) : [];
  const line = (label: string, calls: string, time: string, mono = false) => (
    <li key={label} className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-baseline gap-x-3 border-b border-line/60 py-1 text-xs">
      <span className={`min-w-0 truncate text-muted ${mono ? "mono" : ""}`}>{label}</span>
      <span className="mono tnum text-faint">{calls}</span>
      <span className="mono tnum w-20 text-right text-faint">{time}</span>
    </li>
  );
  return (
    <Section title="Activity">
      {node.path.length > 0 && (
        <div className="flex flex-col gap-0.5 text-xs">
          <Row k="Calls, last hour">{a ? a.calls_hour : 0}</Row>
          <Row k="Calls, last day">{a ? a.calls_day : 0}</Row>
          <Row k="Time, last hour">{a && a.calls_hour ? msText(a.ms_hour) : "none"}</Row>
          <Row k="Time, last day">{a && a.calls_day ? msText(a.ms_day) : "none"}</Row>
          {o?.median_ms !== undefined && o.calls_day > 0 && <Row k="Per call, median">{msText(o.median_ms)}</Row>}
          <Row k="Last used">{a?.last && now > 0 ? ago(a.last, now) : "not in the last day"}</Row>
        </div>
      )}
      {jobs.length > 0 && (
        <ul className="mt-3 flex flex-col">
          {jobs.map(([j, n]) => line(`job: ${j}`, `${n} calls`, ""))}
        </ul>
      )}
      {ran.length > 0 && (
        <div className="mono mt-3 text-[11px] text-faint">
          layers {ran[0][0]} to {ran[ran.length - 1][0]} ran · {Math.max(...ran.map(([, c]) => c))} calls on the busiest layer
        </div>
      )}
      {tiers.length > 0 && <ul className="mt-3 flex flex-col">{tiers.map(([k, t]) => line(`${k.slice(5)} tier, read or written`, `${t.calls_day} calls`, msText(t.ms_day)))}</ul>}
      {subs.length > 0 && <ul className="mt-3 flex flex-col">{subs.map((s) => line(s.id, `${s.calls_day} calls`, msText(s.ms_day), true))}</ul>}
      {kids.length > 0 && <ul className="mt-3 flex flex-col">{kids.map(({ c, a: ka }) => line(c.label, `${ka!.calls_day} calls`, msText(ka!.ms_day)))}</ul>}
      <div className="mono mt-3 text-[10.5px] leading-relaxed text-quiet">
        {act.source === "engine"
          ? "Each call of the last day lights one cell of the drawing, brighter when it also ran in the last hour and for longer work per call."
          : "Each cell of the drawing is a slice of the last day; a call lights the cell of its moment, brighter for longer timed work and for nearer time."}{" "}
        Figures {act.cover}.
      </div>
    </Section>
  );
}

/** A tentacle's job as the engine reported it: the job (or lens) of its step in the running call,
 *  or in the newest call that reported one, marked as such. Nothing is claimed without a step. */
function TentacleJob({ node, rows, now }: { node: TNode; rows: AuditRow[]; now: number }) {
  const idx = Number(/^t(\d+)$/.exec(node.key)?.[1] ?? NaN);
  if (!Number.isFinite(idx)) return null;
  let found: { job: string; row: AuditRow } | null = null;
  const sorted = sortRows(rows);
  for (let i = sorted.length - 1; i >= 0 && !found; i--) {
    for (const st of sorted[i].steps ?? []) {
      const j = stepJob(st);
      if (j && j.tentacle === idx) {
        found = { job: j.job, row: sorted[i] };
        break;
      }
    }
  }
  if (!found)
    return (
      <div className="mono mt-0.5 text-[11px] text-faint" data-e2e="tentacle-job" data-job="">
        no job reported yet · the plan gives one per call
      </div>
    );
  const running = found.row.state === "running";
  return (
    <div className={`mono mt-0.5 text-[11px] ${running ? "text-signal" : "text-muted"}`} data-e2e="tentacle-job" data-job={found.job}>
      {running ? `job now · ${found.job}` : `last job · ${found.job} · ${ago(found.row.time_ms ?? now, now)}`}
    </div>
  );
}
