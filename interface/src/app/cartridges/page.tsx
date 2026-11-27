"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "Dude. The discriptors should be mostly hidden ... And
// that whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals ... you have to think of it as if it's an
// interactive system for the main parts the right menu are libraries and or discription tools etc.
// left menu are the main interface nav like it is now. ... We need the interface to be amazing. You
// really went the wrong direction of the cartridge page and make it look like it was centered around
// as a tutorial page than an interactive systems control for the cartridges ... it should be centered
// on the interactive cartridge system"; earlier: "the decks and library view should be the default
// for the ... cartidge page"): the middle is the cartridge system itself. One deck per
// MicroModel (Reader, Writer, Embedder, Reranker) with its base cartridge seated in the base slot
// and every cartridge docked on it, plugged in or out on the deck; the whole-engine cartridges below;
// the stacks per use and the layers as the other two views. The side panel holds the cartridge library (drag
// a card onto a deck), the base versions, the builders (job filters, base versions), the learned
// units, the spec sheets, how it fits, the glossary and the page's description; a click on any
// cartridge opens its detail there.

import Link from "next/link";
import { maskPathsIn } from "@/lib/privacy";
import { useEffect, useMemo, useRef, useState } from "react";
import Workspace, { RailSection, useRail } from "@/components/Workspace";
import { CartridgeBuildTab } from "@/components/MapRail";
import MicroModelName, { roleTitle } from "@/components/MicroModel";
import LearnedUnits from "@/components/LearnedUnits";
import SystemFlow, { Glossary } from "@/components/SystemFlow";
import { LensesPanel, UnitsPanel } from "@/components/UnitsLenses";
import { StackEditor } from "@/components/CartridgeEdit";
import { CartridgeCard, CompatMatrix, DeckBay, DeckSkeleton, KIND_SWATCH, LibraryPanel, type Kart } from "@/components/CartridgeBay";
import { DeckDetail, KartDetail, useDecks, usePlug, type Decks } from "@/components/CartridgeDecks";
import { AboutCartridges, BackupsLine, BasesTab, KINDS, KindsGuide, SpecSheets } from "@/components/CartridgeRail";
import CartridgeSources from "@/components/CartridgeSources";
import { DownloadsPanel } from "@/components/Downloads";
import { HFDownloadButton } from "@/components/HFDownloadManager";
import { NoteLine, StackSurface, wordForUse } from "@/components/Controls";
import { LiveBadge } from "@/components/LivePanels";
import { LibraryRack, Matrix, Rack, type Bay, type RackSocket } from "@/components/PagePlates";
import { ModuleSwitches } from "@/components/RolesRouting";
import { useTopology } from "@/components/Topology";
import { Empty, Failed, Fold, Pill, Placeholder, Segmented } from "@/components/ui";
import { ago, useApi, type Cartridge, type LibCartridge, type Library, type LibStack } from "@/lib/api";
import { modelLayers, useActivity } from "@/lib/activity";
import { useRoleMap } from "@/lib/roles";
import { useNow } from "@/lib/clock";
import { API_LIBRARY, cartridgeWords, familyOf, familyRank } from "@/lib/evidence";
import { setHash, useHash } from "@/lib/hash";
import { droneSets } from "@/lib/paint";
import { bytesFor, parseStorage } from "@/lib/storage";

type Plug = ReturnType<typeof usePlug>;

/** The views of the middle; decks first. */
const VIEWS = ["decks", "stacks", "layers"] as const;
type View = (typeof VIEWS)[number];
/** The addresses that open a tab of the side panel (old section addresses included). */
const RAIL_FOR: Record<string, string> = { library: "library", compat: "library", bases: "bases", adapters: "adapters", learned: "adapters", units: "adapters", specs: "specs", flow: "flow", glossary: "flow", kinds: "flow", sources: "sources", libraries: "sources", downloads: "downloads", about: "about" };

const markOf = (c: Cartridge) => c.library ?? c.book;
const keyOf = (c: { id: string; version?: number }) => (c.version ? `${c.id}@${c.version}` : c.id);

/** The address opens its tab of the side panel (`/cartridges/#flow` opens Components). */
function AddressToRail({ onReset }: { onReset?: () => void }) {
  const rail = useRail();
  const open = useRef(rail.open);
  const clear = useRef(rail.clear);
  const resetRef = useRef(onReset);
  useEffect(() => {
    open.current = rail.open;
    clear.current = rail.clear;
    resetRef.current = onReset;
  });
  const target = RAIL_FOR[useHash().replace(/^#/, "")];
  useEffect(() => {
    if (target) open.current(target);
  }, [target]);
  useEffect(() => {
    const handle = (e: Event) => {
      const href = (e as CustomEvent<{ href?: string }>).detail?.href;
      if (href !== "/cartridges/") return;
      clear.current();
      resetRef.current?.();
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("nm:nav-reset", handle);
    return () => window.removeEventListener("nm:nav-reset", handle);
  }, []);
  return null;
}

/** Opens a cartridge, or a deck, in the side panel; a second click on the same one closes it. */
function useOpen() {
  const rail = useRail();
  const pickKey = rail.shown?.key ?? null;
  const openKart = (k: Kart) => (pickKey === k.key ? rail.clear() : rail.show({ title: <span title={k.key}>{k.name}</span>, key: k.key, body: <KartDetail kartKey={k.key} /> }));
  const openDeck = (model: string, role: string) => (pickKey === `deck:${model}` ? rail.clear() : rail.show({ title: role ? roleTitle(role) : "MicroModel (no role)", key: `deck:${model}`, body: <DeckDetail model={model} /> }));
  return { pickKey, openKart, openDeck };
}

/** The figures on the title line: what serves now, at a glance. */
function StatusLine({ d }: { d: Decks }) {
  const bases = (d.basesQ.data?.models ?? []).filter((b) => b.serving && b.serving !== "open weights").length;
  const nModels = d.models.length;
  const filters = d.karts.filter((k) => k.kind === "filter");
  const learned = d.unitsQ.data;
  return (
    <>
      <LiveBadge />
      <span title="MicroModels on a Neuromorph base cartridge, of all installed">base {nModels ? `${bases}/${nModels}` : "-"}</span>
      <span title="Job filters plugged in, of all built">filters {d.surgeryQ.data ? `${filters.filter((k) => k.plugged).length}/${filters.length}` : "-"}</span>
      <span title="Learned units acting, of all learned">learned {learned ? `${learned.acting}/${learned.total}` : "-"}</span>
      <span title="Cartridges plugged in, of every cartridge in the cartridge library">plugged {d.karts.filter((k) => k.plugged).length}/{d.karts.length}</span>
    </>
  );
}

/** The decks: one per MicroModel, then the cartridges that set values for the whole engine. */
function DecksView({ d, plug, dragging, setDragging }: { d: Decks; plug: Plug; dragging: Kart | null; setDragging: (k: Kart | null) => void }) {
  const { pickKey, openKart, openDeck } = useOpen();
  const rail = useRail();
  const now = useNow();
  const ids = new Set(d.models.map((m) => m.id));
  const engine = d.karts.filter((k) => k.model === null || !ids.has(k.model));
  const build = d.surgeryQ.data?.job_filter_build && d.surgeryQ.data.job_filter_build.status !== "idle" ? d.surgeryQ.data.job_filter_build : null;
  const doPlug = (k: Kart) => {
    setDragging(null);
    void plug.act(k, "plug");
  };
  if (d.status.error && !d.status.data) return <Failed error={d.status.error} />;
  const onDecks = d.karts.filter((k) => k.model !== null && ids.has(k.model));
  const buildText = build?.message ? maskPathsIn(cartridgeWords(build.message)) : "";
  return (
    // One plate holds the whole rack: the four decks on one line (two by two when the middle is
    // narrow), then the cartridges of the whole engine on the rule beneath them.
    <section aria-label="Cartridge rack" className="ticks @container min-w-0 border border-line bg-bg-1/25 px-4 pb-5 pt-4" data-e2e="rack-plate">
      <header className="mono mb-4 flex min-w-0 flex-wrap items-baseline gap-x-4 gap-y-1 border-b border-line pb-2.5 text-[10px] uppercase tracking-[0.14em] text-faint">
        <span className="text-[11px] tracking-[0.18em] text-text">Rack</span>
        <span aria-hidden className="text-quiet">
          #/
        </span>
        <span title="One deck per MicroModel: its base cartridge seated in the base slot, its cartridges docked on it bottom up">{d.status.data ? `${d.models.length} ${d.models.length === 1 ? "deck" : "decks"}` : "reading"}</span>
        <span className="ml-auto flex flex-wrap items-center gap-x-2.5 gap-y-1 normal-case tracking-normal">
          <span title="Cartridges plugged into a MicroModel's deck, of every cartridge made for one">
            {d.status.data ? `${onDecks.filter((k) => k.plugged).length} of ${onDecks.length} docked` : ""}
          </span>
          <HFDownloadButton label="HF downloads" />
          <button
            type="button"
            onClick={() => rail.open("sources")}
            className="press mono border border-line-strong bg-bg-2/70 px-2 py-0.5 text-[9.5px] uppercase tracking-[0.1em] text-muted hover:border-sense-pale hover:text-text"
            title="Open the Hugging Face library & publish side panel"
          >
            sources
          </button>
        </span>
        {build && (
          <span className="flex min-w-0 basis-full items-center gap-x-3 normal-case tracking-normal text-muted" data-e2e="filter-build" title={`job filter ${build.role}/${build.job}${buildText ? `: ${buildText}` : ""}`}>
            <Pill tone={build.status === "running" ? "signal" : build.status === "error" ? "error" : build.status === "done" ? "verified" : "muted"}>{build.status}</Pill>
            <span className="shrink-0">
              job filter {build.role}/{build.job}
            </span>
            {/* One line while it runs or after it finished; an error is shown whole. */}
            {buildText && <span className={`min-w-0 ${build.status === "error" ? "text-error" : "truncate text-faint"}`}>{buildText}</span>}
          </span>
        )}
      </header>
      <div className="grid min-w-0 grid-cols-1 gap-x-5 gap-y-8 @[30rem]:grid-cols-2 @[54rem]:grid-cols-4" data-e2e="decks" aria-busy={!d.status.data}>
        {!d.status.data
          ? [0, 1, 2, 3].map((i) => <DeckSkeleton key={i} />)
          : d.models.map((m, mi) => {
              const roles = d.roleMap.rolesOf(m.id);
              const role = roles.join(" + ");
              const slot = d.basesQ.data?.models?.find((b) => b.model === m.id) ?? null;
              const last = d.lastCall(m.id);
              return (
                <DeckBay
                  key={m.id}
                  model={m.id}
                  role={role}
                  layers={typeof m.layers === "number" ? m.layers : null}
                  live={d.liveFor(m.id)}
                  serving={d.serving}
                  lastUsed={last ? ago(last, now) : null}
                  slot={slot}
                  karts={d.karts.filter((k) => k.model === m.id)}
                  dragging={dragging}
                  onDropKart={doPlug}
                  pickKey={pickKey}
                  landing={plug.landing}
                  busyKey={plug.busyKey}
                  onOpen={openKart}
                  onPlug={doPlug}
                  onEject={(k) => void plug.act(k, "eject")}
                  linkOf={d.linkOf}
                  baseBusy={plug.busyKey === `choice:${m.id}`}
                  onChooseBase={(c) => void plug.run(`choice:${m.id}`, "/api/cartridges/bases", { model: m.id, choice: c }, c === "auto" ? `${m.id}: the newest usable base cartridge serves from the next start.` : c === "open" ? `${m.id}: its open weights serve from the next start.` : `${m.id}: ${c} serves from the next start.`)}
                  onOpenDeck={() => openDeck(m.id, role)}
                  switchesRight={mi % 2 === 1}
                  switches={
                    roles.length ? (
                      <div className="grid gap-2">
                        {roles.map((r) => (
                          <ModuleSwitches key={r} role={r} onNote={plug.setNote} />
                        ))}
                      </div>
                    ) : (
                      <div className="mono border border-dashed border-line px-2.5 py-2 text-[10.5px] text-quiet">no role runs on this model now; choose it for a role on Models</div>
                    )
                  }
                />
              );
            })}
      </div>
      {d.status.data && d.models.length === 0 && <Empty title="No MicroModel installed">Install them on Models, Downloads.</Empty>}
      {/* The engine's own line under the decks: the cartridges that set values for the whole engine. */}
      <section aria-label="Whole-engine cartridges" className="mt-6 min-w-0" data-e2e="engine-rack">
        <div aria-hidden className="rule-ticks" />
        <div className="mono mb-2.5 mt-1 flex items-baseline justify-between gap-3 text-[10px] uppercase tracking-[0.14em] text-faint">
          <span title="Cartridges that set values for the whole engine rather than one MicroModel; their stack for a use plugs them in">whole engine</span>
          <span className="normal-case tracking-normal">{d.cartsQ.data ? `${engine.filter((k) => k.plugged).length} of ${engine.length} in use` : "reading"}</span>
        </div>
        {!d.cartsQ.data ? (
          <Placeholder className="h-16" what="engine cartridges" />
        ) : engine.length === 0 ? (
          <div className="mono py-2 text-[11px] text-quiet">none in the cartridge library</div>
        ) : (
          <ul className="grid grid-cols-1 gap-x-5 gap-y-2 @[30rem]:grid-cols-2 @[54rem]:grid-cols-4">
            {engine.map((k) => (
              <li key={k.key} className="min-w-0">
                <CartridgeCard k={k} size="mini" selected={pickKey === k.key} busy={plug.busyKey === k.key} onOpen={() => openKart(k)} onPlug={k.plug ? () => void plug.act(k, "plug") : undefined} onEject={k.eject ? () => void plug.act(k, "eject") : undefined} />
              </li>
            ))}
          </ul>
        )}
      </section>
    </section>
  );
}

/** The stacks in apply order: the uses in use first, as the library applies them, then the stacks
 *  it holds for other uses. */
function baysOf(lib: Library, rows: Cartridge[]): Bay[] {
  const cat: LibCartridge[] = lib.catalogue?.cartridges ?? lib.catalogue?.books ?? [];
  const stacks = lib.catalogue?.stacks ?? [];
  const loaded = new Map((lib.in_use?.loaded ?? []).map((l) => [l.key, l]));
  const out: Bay[] = [];
  const done = new Set<string>();
  const add = (use: string, stackKey: string | null, trial: boolean, inUse: boolean) => {
    const s = stacks.find((x) => keyOf(x) === stackKey || x.id === stackKey);
    const keys = s?.cartridges ?? [];
    if (stackKey) done.add(stackKey);
    out.push({
      use,
      label: wordForUse(use),
      stack: stackKey,
      trial,
      inUse,
      slabs: keys
        .map((k) => {
          const c = cat.find((x) => keyOf(x) === k || x.id === k);
          const row = rows.find((r) => markOf(r)?.key === k);
          const l = loaded.get(k);
          return { key: k, family: familyOf(row ?? { family: c?.family, class: "self" }), layers: c?.layers ?? null, settings: (c?.model ?? row?.model) === "settings" || c?.target === "settings", lit: inUse && !!l && !l.refused, trial: !!(l?.trial ?? c?.trial ?? trial), verified: row?.verified ?? false, bytes: row?.bytes ?? 0 };
        })
        .sort((a, b) => familyRank(a.family) - familyRank(b.family)),
    });
  };
  for (const u of lib.in_use?.uses ?? []) if (u.stack) add(u.use, u.stack, !!u.trial, true);
  for (const s of stacks) if (!done.has(keyOf(s))) add(s.use, keyOf(s), !!s.trial, false);
  return out;
}

/** The piece a stack is for, in its own name when the library gives one. */
function pieceName(s: LibStack | undefined, lib: Library, identity?: string | null): string {
  const p = s?.piece;
  if (p && typeof p === "object" && p.name) return p.name;
  const id = identity ?? (typeof p === "string" ? p : p?.identity);
  return lib.in_use?.machine?.pieces?.find((x) => x.identity === id)?.name ?? "";
}

/** Each MicroModel's cartridges in the order a stack applies them: fundamentals, then learning, then
 *  preference on top; within a family, in the order of their stacks. */
function useOrdered(all: Cartridge[], bays: Bay[]): Cartridge[] {
  return useMemo(() => {
    const order = bays.flatMap((b) => b.slabs.map((s) => s.key));
    const rank = (c: Cartridge) => {
      const i = order.indexOf(markOf(c)?.key ?? "");
      return i < 0 ? order.length : i;
    };
    return [...all].sort((a, b) => familyRank(familyOf(a)) - familyRank(familyOf(b)) || rank(a) - rank(b));
  }, [all, bays]);
}

/** The stacks for each use: the rack, the switch per use, the editor, and each stack in detail. */
function StacksView({ d, plug }: { d: Decks; plug: Plug }) {
  const lib = d.libQ.data && !d.libQ.data.error ? d.libQ.data : null;
  const bays = useMemo(() => (lib ? baysOf(lib, d.all) : []), [lib, d.all]);
  const ordered = useOrdered(d.all, bays);
  const storageQ = useApi<unknown>("/api/storage", 30000);
  const storage = useMemo(() => parseStorage(storageQ.data), [storageQ.data]);
  const surgery = d.surgeryQ.data;
  const reply = d.live.reply;
  // Every socket the engine reports besides the cartridges: the job filters per role, the lenses and
  // preference cartridges docked on each live MicroModel (used when the library is not reported).
  const sockets = useMemo(() => {
    const out: RackSocket[] = [];
    const seen = new Set(d.all.map((c) => c.id));
    for (const [role, info] of Object.entries(surgery?.roles ?? {}))
      for (const jf of info.job_filters ?? []) {
        if (seen.has(jf.cartridge)) continue;
        seen.add(jf.cartridge);
        out.push({ id: jf.cartridge, kind: "filter", on: jf.enabled !== false, role, bytes: bytesFor(storage, jf.cartridge) });
      }
    for (const m of reply?.models ?? []) {
      for (const f of m.filters) {
        if (seen.has(f.id) || f.kind === "other") continue;
        seen.add(f.id);
        out.push({ id: f.id, kind: f.kind === "lens" ? "lens" : f.kind === "preference" ? "preference" : "filter", on: f.on, role: m.role, bytes: bytesFor(storage, f.id) });
      }
      if (m.lens && !seen.has(m.lens)) {
        seen.add(m.lens);
        out.push({ id: m.lens, kind: "lens", on: true, role: m.role, bytes: bytesFor(storage, m.lens) });
      }
    }
    return out;
  }, [d.all, surgery, reply, storage]);
  const build = surgery?.job_filter_build && surgery.job_filter_build.status !== "idle" ? surgery.job_filter_build : null;
  const stacks = lib?.catalogue?.stacks ?? [];
  const inUse = (lib?.in_use?.uses ?? []).filter((u) => u.stack).length;
  return (
    <div className="grid min-w-0 gap-8">
      {/* The same plate as the rack, the figure in its middle. */}
      <section aria-label="Stacks for each use" className="ticks min-w-0 border border-line bg-bg-1/25 px-4 pb-4 pt-4">
        <div className="mono mb-3 flex flex-wrap items-baseline gap-x-4 gap-y-1 border-b border-line pb-2.5 text-[10px] uppercase tracking-[0.14em] text-faint">
          <span className="text-[11px] tracking-[0.18em] text-text">Stacks</span>
          <span aria-hidden className="text-quiet">
            #/
          </span>
          <span>bottom up, in the order they apply</span>
          <span className="ml-auto normal-case tracking-normal">{lib ? `${bays.length} ${bays.length === 1 ? "stack" : "stacks"} · ${inUse} in use` : "cartridge library not reported"}</span>
        </div>
        <div className="mx-auto max-w-3xl">{lib ? <LibraryRack bays={bays} /> : <Rack carts={ordered} sockets={sockets} build={build} />}</div>
      </section>
      {lib && (
        <section aria-label="Switch a stack" className="min-w-0">
          <div className="mono mb-2 border-b border-line pb-1.5 text-[10px] uppercase tracking-[0.14em] text-faint">switch a stack · per use</div>
          <StackSurface lib={lib} onNote={plug.setNote} cols="grid-cols-1 sm:grid-cols-2 2xl:grid-cols-4" />
        </section>
      )}
      {lib && <StackEditor lib={lib} />}
      <Fold title="Each stack in detail" right={lib ? `${inUse} in use` : undefined}>
        {!d.hasLib ? (
          <Empty title="Not reported by this engine build" slot>
            This engine build answers API {d.level ?? "?"}. The cartridge library, its stacks for each use and the switch between them come with API {API_LIBRARY}.
          </Empty>
        ) : d.libQ.error && !d.libQ.data ? (
          <Empty title="The cartridge library did not answer">{d.libQ.error}</Empty>
        ) : d.libQ.data?.error ? (
          <Empty title="The cartridge library could not be opened">{d.libQ.data.error}</Empty>
        ) : !lib ? (
          <Placeholder className="h-24" what="cartridge library" />
        ) : (
          <>
            <ol>
              {bays.map((b) => {
                const s = stacks.find((x) => keyOf(x) === b.stack || x.id === b.stack);
                const u = lib.in_use?.uses?.find((x) => x.use === b.use && x.stack === b.stack);
                const piece = pieceName(s, lib, u?.piece);
                const set = Object.entries(lib.in_use?.set ?? {}).filter(([, v]) => b.slabs.some((sl) => sl.key === v.from));
                return (
                  <li key={b.use + (b.stack ?? "")} className="grid gap-x-8 gap-y-3 border-b border-line py-3 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.3fr)_minmax(0,1fr)]">
                    <div className="min-w-0">
                      <div className="flex min-w-0 items-center gap-3">
                        <span className="mono truncate text-[12px] uppercase tracking-[0.12em] text-text">{b.label}</span>
                        {b.trial ? <Pill tone="attention">on trial</Pill> : b.inUse ? <Pill tone="learned">in use</Pill> : <Pill>not in use</Pill>}
                      </div>
                      {piece && <div className="mono mt-1 truncate text-[11px] text-muted">{piece}</div>}
                      <div className="mono mt-1 truncate text-[11px] text-faint" title={u?.why ? cartridgeWords(u.why) : (b.stack ?? "")}>
                        stack {b.stack ?? "none"}
                      </div>
                    </div>
                    <ol className="flex min-w-0 flex-wrap items-center gap-y-2">
                      {b.slabs.map((sl, i) => (
                        <li key={sl.key} className="flex min-w-0 items-center">
                          {i > 0 && <span aria-hidden className="mx-2 inline-block h-px w-4 bg-line-strong" />}
                          <span className="mono flex min-w-0 items-center gap-2 text-[11px]">
                            <span className="text-faint">{i + 1}</span>
                            <span aria-hidden className={sl.lit ? "spectral-sw !h-4 !w-1" : "dither-slot !h-4 !w-2"} />
                            <span className={`truncate ${sl.lit ? "text-text" : "text-muted"}`}>{sl.key.replace(/@\d+$/, "")}</span>
                            <span className="shrink-0 text-quiet">{sl.settings || !sl.layers ? "settings" : `layers ${sl.layers[0]} to ${sl.layers[1]}`}</span>
                          </span>
                        </li>
                      ))}
                      {b.slabs.length === 0 && <li className="mono text-[11px] text-quiet">no cartridge in this stack</li>}
                    </ol>
                    <div className="min-w-0">
                      {set.length ? (
                        <ul>
                          {set.slice(0, 8).map(([k, v]) => (
                            <li key={k} className="flex items-baseline justify-between gap-3 py-0.5 text-[11px]">
                              <span className="mono shrink-0 text-faint">{k}</span>
                              <span className="mono min-w-0 truncate text-right text-muted" title={v.value}>
                                {v.value}
                              </span>
                            </li>
                          ))}
                          {set.length > 8 && <li className="mono text-[10.5px] text-quiet">and {set.length - 8} more</li>}
                        </ul>
                      ) : (
                        <div className="mono text-[11px] text-quiet">{b.inUse ? "nothing set" : "applies only when in use"}</div>
                      )}
                    </div>
                  </li>
                );
              })}
            </ol>
            {(lib.in_use?.trial ?? []).length > 0 && <div className="mono mt-3 text-[11px] text-faint">{lib.in_use!.trial!.length} values a trial stack would change, recorded and never applied.</div>}
            {(lib.in_use?.refused ?? []).length > 0 && <div className="mono mt-2 text-[11px] text-error">refused: {lib.in_use!.refused!.map(cartridgeWords).join(" · ")}</div>}
          </>
        )}
      </Fold>
    </div>
  );
}

/** Where the cartridges plug into each MicroModel's layers, with the layers at work. */
function LayersView({ d }: { d: Decks }) {
  const settings = useApi<Record<string, unknown>>("/api/settings");
  const drones = useMemo(() => droneSets(settings.data), [settings.data]);
  const { tree } = useTopology();
  const act = useActivity(tree);
  const roleMap = useRoleMap();
  const lib = d.libQ.data && !d.libQ.data.error ? d.libQ.data : null;
  const bays = useMemo(() => (lib ? baysOf(lib, d.all) : []), [lib, d.all]);
  const ordered = useOrdered(d.all, bays);
  if (!d.status.data) return <Placeholder className="h-64" what="engine status" />;
  return (
    // The same plate as the rack: a column per MicroModel, the names on one line and the layer
    // matrices starting on one line under them.
    <section aria-label="Layers of each MicroModel" className="ticks @container min-w-0 border border-line bg-bg-1/25 px-4 pb-5 pt-4">
      <header className="mono mb-4 flex min-w-0 flex-wrap items-baseline gap-x-4 gap-y-1 border-b border-line pb-2.5 text-[10px] uppercase tracking-[0.14em] text-faint">
        <span className="text-[11px] tracking-[0.18em] text-text">Layers</span>
        <span aria-hidden className="text-quiet">
          #/
        </span>
        <span>
          {d.models.length} {d.models.length === 1 ? "MicroModel" : "MicroModels"}
        </span>
        <span className="ml-auto normal-case tracking-normal" title="A row is a layer, a cell a head; a cartridge is drawn on the layers it plugs into">
          a row is a layer, layer 0 at the bottom · a cell is a head
        </span>
      </header>
      <div className="grid min-w-0 grid-cols-1 gap-x-6 gap-y-8 @[30rem]:grid-cols-2 @[54rem]:grid-cols-4" data-e2e="layers">
        {d.models.map((m) => {
          const ml = typeof m.layers === "number" ? modelLayers(act, tree, m.id, m.layers, roleMap.rolesOf(m.id)) : null;
          return (
            <div key={m.id} className="row-span-2 grid min-w-0 grid-cols-[minmax(0,1fr)] grid-rows-subgrid gap-y-0">
              <div className="mb-3 min-w-0">
                <MicroModelName model={m.id} layout="stack" nameClass="eyebrow" />
              </div>
              <div className="min-w-0">
                <Matrix id={m.id} cell={8} fluid drones={drones.get(m.id)} carts={ordered} scale={false} onCard={d.status.data?.loaded?.find((x) => x.id === m.id)?.layers_on_card} levels={ml?.levels ?? null} live={d.liveFor(m.id)} layers={m.layers} heads={m.heads} />
                {typeof m.layers !== "number" && <div className="mono mt-1 text-[10.5px] text-faint">layers not reported</div>}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/** The middle: the view switch with the legend, the note of the last action, then the view. */
function CartridgeSystem({ d, plug, dragging, setDragging }: { d: Decks; plug: Plug; dragging: Kart | null; setDragging: (k: Kart | null) => void }) {
  const h = useHash().replace(/^#/, "");
  const [picked, setPicked] = useState<View | null>(null);
  const view: View = (VIEWS as readonly string[]).includes(h) ? (h as View) : (picked ?? "decks");
  return (
    <div className="grid min-w-0 gap-4">
      <AddressToRail
        onReset={() => {
          setPicked("decks");
          window.history.pushState(null, "", "/cartridges/");
        }}
      />
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
        <Segmented
          label="Cartridges view"
          values={[...VIEWS]}
          value={view}
          onPick={(v) => {
            setPicked(v as View);
            setHash(`#${v}`, false);
          }}
        />
        <span className="mono text-[10px] text-quiet">{view === "decks" ? "one deck per MicroModel · bottom up" : view === "stacks" ? "the cartridges plugged in together for each use" : "where each cartridge plugs into its MicroModel's layers"}</span>
      </div>
      <NoteLine note={plug.note} />
      {view === "decks" && <DecksView d={d} plug={plug} dragging={dragging} setDragging={setDragging} />}
      {view === "stacks" && <StacksView d={d} plug={plug} />}
      {view === "layers" && <LayersView d={d} />}
      {/* The key of the figure, under it: the band colour of each kind, then how to work it. */}
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1.5 border-t border-line pt-2.5">
        <ul className="mono flex flex-wrap items-center gap-x-3 gap-y-1 text-[9.5px] uppercase tracking-[0.12em] text-faint" aria-label="Kinds of cartridges" data-e2e="kinds-legend">
          {KINDS.map((k) => (
            <li key={k.kind} className="flex items-center gap-1.5" title={k.what}>
              <i aria-hidden className={`inline-block h-2.5 w-1 ${KIND_SWATCH[k.kind]}`} />
              {k.name}
            </li>
          ))}
        </ul>
        <p className="mono text-[10px] text-quiet">
          {view === "decks" ? "drag a card from the cartridge library onto its deck, or use plug in · click a cartridge for its detail" : view === "stacks" ? "a new stack starts on trial" : "with the layers at work"}
        </p>
      </div>
    </div>
  );
}

/** The library tab: every cartridge, filterable, dragged onto a deck or plugged with its button;
 *  then which MicroModel each fits. */
function LibraryTab({ d, plug, setDragging }: { d: Decks; plug: Plug; setDragging: (k: Kart | null) => void }) {
  const { pickKey, openKart } = useOpen();
  const rail = useRail();
  if (!d.basesQ.data && !d.cartsQ.data) return <Placeholder className="h-48" what="cartridges and bases" />;
  return (
    <div className="grid gap-5">
      <LibraryPanel
        karts={d.karts}
        models={d.models.map((m) => m.id)}
        roleOf={d.roleMap.roleOf}
        pickKey={pickKey}
        busyKey={plug.busyKey}
        onOpen={openKart}
        onPlug={(k) => void plug.act(k, "plug")}
        onEject={(k) => void plug.act(k, "eject")}
        onDragStart={setDragging}
        onDragEnd={() => setDragging(null)}
      />
      <Fold title="Fit · which MicroModel each fits" right={`${(d.compatQ.data?.cartridges ?? []).length}`}>
        <CompatMatrix rows={d.compatQ.data?.cartridges ?? []} models={d.compatQ.data?.models ?? []} roleOf={d.roleMap.roleOf} />
        <p className="mono mt-2 text-[10px] text-quiet" title="A cartridge fits a MicroModel when it was made for that model and its open weights; cartridges stay with their model, so a swapped role runs with its new model's cartridges and finds its own again on the way back.">
          made for the model and its open weights
        </p>
      </Fold>
      <RailSection title="Get more">
        <span className="mono flex flex-wrap gap-x-4 gap-y-1 text-[10.5px]">
          <button type="button" className="link" onClick={() => rail.open("sources")}>
            online cartridge libraries
          </button>
          <button type="button" className="link" onClick={() => rail.open("downloads")}>
            base &amp; map downloads
          </button>
          <button type="button" className="link" onClick={() => rail.open("build")}>
            build one
          </button>
          <Link href="/parameters/#import" className="link">
            import a file
          </Link>
        </span>
      </RailSection>
      <RailSection title="Backups">
        <BackupsLine parts={["cartridges", "units"]} />
      </RailSection>
    </div>
  );
}

/** The spec sheets tab: each opens its cartridge in the detail. */
function SpecsTab({ d, plug }: { d: Decks; plug: Plug }) {
  const { openKart } = useOpen();
  if (d.cartsQ.error && !d.cartsQ.data) return <Failed error={d.cartsQ.error} />;
  if (!d.cartsQ.data) return <Placeholder className="h-48" what="cartridges" />;
  return (
    <SpecSheets
      all={d.all}
      onNote={plug.setNote}
      onOpen={(c) => {
        const k = d.karts.find((x) => x.key === `cart:${c.id}@${c.version}`);
        if (k) openKart(k);
      }}
    />
  );
}

export default function Cartridges() {
  const d = useDecks();
  const plug = usePlug();
  const [dragging, setDragging] = useState<Kart | null>(null);
  const learned = d.unitsQ.data;
  return (
    <Workspace
      title="Cartridges"
      railKey="cartridges"
      defaultTab="library"
      status={<StatusLine d={d} />}
      rail={[
        { id: "library", label: "Library", count: d.karts.length || undefined, body: <LibraryTab d={d} plug={plug} setDragging={setDragging} /> },
        { id: "bases", label: "Bases", body: <BasesTab onNote={plug.setNote} /> },
        { id: "build", label: "Build", body: <CartridgeBuildTab /> },
        { id: "sources", label: "Sources", body: <CartridgeSources /> },
        { id: "downloads", label: "Downloads", body: <DownloadsPanel onNote={plug.setNote} /> },
        {
          id: "adapters",
          label: "Learned",
          count: learned ? `${learned.acting}/${learned.total}` : undefined,
          body: (
            <div className="grid gap-8">
              <LensesPanel compact />
              <UnitsPanel />
              <LearnedUnits projectKey={null} />
            </div>
          ),
        },
        { id: "specs", label: "Specs", body: <SpecsTab d={d} plug={plug} /> },
        {
          id: "flow",
          label: "Components",
          body: (
            <div className="grid gap-6">
              <SystemFlow />
              <RailSection title="Kinds of cartridges">
                <KindsGuide />
              </RailSection>
              <Glossary />
            </div>
          ),
        },
      ]}
      about={<AboutCartridges />}
    >
      <CartridgeSystem d={d} plug={plug} dragging={dragging} setDragging={setDragging} />
    </Workspace>
  );
}
