"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals ... We need the interface to be amazing.
// You really went the wrong direction of the cartridge page and make it look like it was centered
// around as a tutorial page than an interactive systems control for the cartridges ... it should be
// centered on the interactive cartridge system"): the pieces of the cartridge system as drawn. A
// deck per MicroModel: the base cartridge seated in its base slot with its version, every cartridge
// docked on it stacked bottom up (lit when plugged in), the ones in the cartridge library that fit beside it
// with a plug, the ones that do not fit with the reason; the library in the side panel, dragged onto
// the deck it fits. The cards carry no state of their own: every plug and eject goes through the
// engine's API and the views refresh from what the engine then reports.
//
// 2026-11-25 (directions of Luis Jake Gabriel III): cartridges that look like cartridges, decks
// they plug into, and the library beside them.

import { BaseMark, WeightsMark, roleTitle } from "@/components/MicroModel";
import { useMemo, useState, type CSSProperties, type ReactNode } from "react";
import { Pill } from "@/components/ui";
import type { LiveModel } from "@/lib/livefeed";

export type KartKind = "base" | "filter" | "lens" | "adapter" | "comp" | "preference" | "settings" | "learned";

/** One change the engine makes when a cartridge is plugged in or ejected. */
export type KartAction = { path: string; body: unknown; done: string };

/** Where a cartridge's record lives in the engine's replies, so its detail reads it fresh. */
export type KartSource =
  | { type: "base"; model: string; key: string }
  | { type: "filter"; role: string; job: string; id: string }
  | { type: "unit"; key: string }
  | { type: "lens"; role: string; id: string }
  | { type: "cart"; id: string; version: number };

/** A cartridge as the decks and the library draw it. */
export type Kart = {
  key: string;
  name: string;
  kind: KartKind;
  /** The MicroModel it plugs into; none for a cartridge that sets values for the whole engine. */
  model: string | null;
  version: string;
  meta: string;
  plugged: boolean;
  /** In the cartridge library with the right size and made for these open weights. */
  ready: boolean;
  note?: string;
  /** 2026-11-25: the installed MicroModels it fits (same model, same open weights), why not the
   *  others, and one line saying so (from /api/cartridges/compat). */
  fits?: string[];
  why?: Record<string, string>;
  fitText?: string;
  plug?: KartAction;
  eject?: KartAction;
  src?: KartSource;
};

export const KIND_WORD: Record<KartKind, string> = { base: "base", filter: "job filter", lens: "lens", adapter: "adapter", comp: "compensation", preference: "preference", settings: "engine", learned: "learned" };
/** Bottom up in a deck: the base first, then compensation, filters, adapters and lens, preference on top. */
export const STACK_RANK: Record<KartKind, number> = { base: 0, comp: 1, filter: 2, lens: 3, adapter: 3, learned: 4, preference: 5, settings: 6 };
/** The band colour of the kinds the stylesheet does not name. */
const BAND: Partial<Record<KartKind, string>> = { adapter: "var(--learned)", comp: "var(--sense)" };
export const KIND_SWATCH: Record<KartKind, string> = { base: "bg-sense-pale", filter: "bg-signal", lens: "bg-learned", adapter: "bg-learned", comp: "bg-sense", preference: "bg-learned", settings: "bg-quiet", learned: "bg-quiet" };

/** Whether the running engine reports a cartridge connected where it is plugged in. */
export type Link = { state: "on" | "off" | "unknown"; why?: string };

/** Whether a cartridge fits a MicroModel, and why not when it does not. */
export function fitOf(k: Kart, model: string): { fits: boolean; why: string | null } {
  if (k.fits) return k.fits.includes(model) ? { fits: true, why: null } : { fits: false, why: k.why?.[model] ?? `made for ${k.model ?? "the whole engine"}` };
  return k.model === model || k.model === null ? { fits: true, why: null } : { fits: false, why: `made for ${k.model}` };
}

const btn = "press mono border-b px-0.5 text-[9.5px] uppercase tracking-[0.12em] disabled:opacity-40";

/** One cartridge: grip, label strip in the colour of its kind, contact pins along the bottom. */
export function CartridgeCard({ k, size = "deck", selected, landing, busy, link, onOpen, onPlug, onEject, onDragStart, onDragEnd, extra }: {
  k: Kart;
  link?: Link;
  size?: "deck" | "mini";
  selected?: boolean;
  landing?: boolean;
  busy?: boolean;
  onOpen?: () => void;
  onPlug?: () => void;
  onEject?: () => void;
  onDragStart?: () => void;
  onDragEnd?: () => void;
  /** A control carried on the label (the base slot's version choice). */
  extra?: ReactNode;
}) {
  const movable = !!(k.plug || k.eject);
  const band = BAND[k.kind];
  return (
    <div
      className={`cart ${landing ? "cart-in" : ""}`}
      data-kind={k.kind}
      data-plugged={k.plugged ? "true" : "false"}
      data-selected={selected ? "true" : "false"}
      data-e2e="cart"
      data-key={k.key}
      style={{ ...(band ? ({ "--band": band } as CSSProperties) : {}), ...(selected ? { boxShadow: "inset 0 0 0 1px var(--sense-pale), inset 0 0 18px -6px var(--sense-pale)" } : {}) }}
      draggable={movable && !k.plugged && k.ready}
      onDragStart={(e) => {
        e.dataTransfer.setData("text/plain", k.key);
        e.dataTransfer.effectAllowed = "move";
        onDragStart?.();
      }}
      onDragEnd={onDragEnd}
    >
      <div className="cart-label">
        <button type="button" onClick={onOpen} aria-pressed={selected} className="press block w-full min-w-0 text-left" title={`Open ${k.name} in the side panel`}>
          <span className={`mono block truncate ${size === "deck" ? "text-[12px]" : "text-[11px]"} text-text`}>{k.name}</span>
          <span className="mono mt-0.5 flex min-w-0 items-baseline justify-between gap-2 text-[9.5px] uppercase tracking-[0.1em] text-faint">
            <span className="min-w-0 truncate" title={`${KIND_WORD[k.kind]} · ${k.version}`}>
              {KIND_WORD[k.kind]} · {k.version}
            </span>
            {k.meta && (
              <span className="max-w-[55%] shrink-0 truncate normal-case tracking-normal" title={k.meta}>
                {k.meta}
              </span>
            )}
          </span>
          {size === "mini" && k.fitText && (
            <span className={`mono mt-0.5 block truncate text-[9.5px] ${k.fits && k.fits.length === 0 ? "text-attention" : "text-faint"}`} title={k.why ? Object.entries(k.why).map(([m, w]) => `${m}: ${w}`).join(" · ") : undefined}>
              {k.fitText}
            </span>
          )}
        </button>
        {extra}
        {(onPlug || onEject || link) && (
          <div className="mt-1.5 flex items-center justify-end gap-3">
            {link && (
              <span
                className={`mono mr-auto flex min-w-0 items-center gap-1.5 text-[9.5px] uppercase tracking-[0.1em] ${link.state === "on" ? "text-verified" : link.state === "off" ? "text-attention" : "text-quiet"}`}
                title={link.why}
                data-e2e="cart-link"
              >
                <i aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 ${link.state === "on" ? "bg-verified" : link.state === "off" ? "bg-attention" : "bg-quiet"}`} />
                <span className="truncate">{link.state === "on" ? "connected" : link.state === "off" ? "not connected" : "engine off"}</span>
              </span>
            )}
            {k.plugged
              ? onEject && (
                  <button type="button" disabled={busy} onClick={onEject} className={`${btn} border-line-strong text-muted hover:text-text`} aria-label={`Unplug ${k.name}`}>
                    {busy ? "unplugging" : "unplug"}
                  </button>
                )
              : onPlug && (
                  <button type="button" disabled={busy || !k.ready} onClick={onPlug} className={`${btn} border-sense-pale text-sense-pale hover:text-text`} aria-label={`Plug in ${k.name}`} title={k.ready ? undefined : k.note}>
                    {busy ? "plugging" : "plug in"}
                  </button>
                )}
          </div>
        )}
      </div>
      <div aria-hidden className="cart-pins" />
    </div>
  );
}

/** The base slot's state: what is chosen and what serves, with every version in the cartridge library. */
export type BaseSlot = { choice: string; serving: string; versions: { key: string; version: number; present: boolean; refused: string | null; serving: boolean }[] };

/** The version choice on the base slot: the newest usable, one version, or the open weights. */
function BaseChoice({ model, slot, busy, onChoose }: { model: string; slot: BaseSlot; busy: boolean; onChoose: (choice: string) => void }) {
  return (
    <label className="mono mt-1.5 flex min-w-0 items-center gap-2 text-[9.5px] uppercase tracking-[0.1em] text-faint" title="What serves on this MicroModel from the next start of the engine">
      <span className="shrink-0">serve</span>
      <select
        value={slot.choice}
        disabled={busy}
        aria-label={`Base cartridge for ${model}`}
        data-e2e="base-choice"
        onChange={(e) => onChoose(e.target.value)}
        className="mono min-w-0 flex-1 truncate border-b border-line-strong bg-transparent text-[11px] normal-case tracking-normal text-text outline-none focus-visible:border-sense-pale disabled:opacity-50"
      >
        <option value="auto" className="bg-bg-1">
          newest usable
        </option>
        {slot.versions.map((v) => (
          <option key={v.key} value={v.key} disabled={!v.present || !!v.refused} className="bg-bg-1">
            {v.key}
            {!v.present ? " (not downloaded)" : v.refused ? " (refused)" : ""}
          </option>
        ))}
        <option value="open" className="bg-bg-1">
          open weights only
        </option>
      </select>
    </label>
  );
}

/** A MicroModel's deck: its header (role, base, open weights, live state), the bay with the docked
 *  cartridges stacked on the base slot, then what waits in the cartridge library for it (fits, unplugged) and what does not fit. */
export function DeckBay({ model, role, layers, live, serving, lastUsed, slot, karts, dragging, onDropKart, pickKey, landing, busyKey, onOpen, onPlug, onEject, linkOf, onChooseBase, baseBusy, onOpenDeck, switches, switchesRight }: {
  model: string;
  role: string;
  /** The model's layer count as the engine reports it; null when it reports none. */
  layers: number | null;
  live: LiveModel | null;
  /** Whether an engine process is serving now (null: not known yet). */
  serving: boolean | null;
  /** The newest call the live record shows on this MicroModel, as words; null when none is shown. */
  lastUsed: string | null;
  slot: BaseSlot | null;
  karts: Kart[];
  dragging: Kart | null;
  onDropKart: (k: Kart) => void;
  pickKey: string | null;
  landing: string | null;
  busyKey: string | null;
  onOpen: (k: Kart) => void;
  onPlug: (k: Kart) => void;
  onEject: (k: Kart) => void;
  linkOf?: (k: Kart) => Link;
  onChooseBase: (choice: string) => void;
  baseBusy: boolean;
  onOpenDeck: () => void;
  /** This MicroModel's own switches (its model and precision), folded on its deck. */
  switches?: ReactNode;
  /** The switches open from the deck's right edge (a deck in a right-hand column). */
  switchesRight?: boolean;
}) {
  const [over, setOver] = useState(false);
  const fit = dragging ? fitOf(dragging, model) : null;
  const base = karts.find((k) => k.kind === "base" && k.plugged) ?? null;
  const docked = useMemo(() => karts.filter((k) => k.kind !== "base" && k.plugged).sort((a, b) => STACK_RANK[a.kind] - STACK_RANK[b.kind]), [karts]);
  const waiting = karts.filter((k) => k.kind !== "base" && !k.plugged && fitOf(k, model).fits);
  const unfit = karts.filter((k) => k.kind !== "base" && !k.plugged && !fitOf(k, model).fits);
  const spare = karts.filter((k) => k.kind === "base" && !k.plugged);
  const state = serving === false ? "engine off" : live?.executing ? "working" : live?.resident ? "on the card" : "not held";
  const tone = live?.executing ? "signal" : live?.resident ? "learned" : "muted";
  const pending = slot && slot.choice !== "auto" && slot.choice !== slot.serving ? slot.choice : null;
  const links = linkOf ? [...(base ? [base] : []), ...docked].map(linkOf) : [];
  const allOn = links.length > 0 && links.every((l) => l.state === "on");
  // Where the dragged cartridge lands: the base slot for a base cartridge, else its place in the
  // stack order (the docked list is bottom up).
  const landsBase = !!dragging && !!fit?.fits && dragging.kind === "base";
  const landAt = dragging && fit?.fits && !landsBase ? docked.filter((k) => STACK_RANK[k.kind] <= STACK_RANK[dragging.kind]).length : -1;
  const slotCls = `mono flex items-center justify-center border border-dashed px-2 py-3 text-center text-[10px] uppercase tracking-[0.14em] ${over ? "border-sense-pale bg-sense/10 text-text" : "border-line-3 text-faint"}`;
  const landingSlot = dragging ? (
    <li key="landing" className="my-1" data-e2e="deck-landing">
      <div className={slotCls}>{over ? "release to plug in" : `${dragging.name} lands here`}</div>
    </li>
  ) : null;
  return (
    // Three rows shared by every deck of the rack (header, bay, what is beside it), so the bays
    // start on one line and the base slots stand on one line.
    <article className="row-span-3 grid min-w-0 grid-cols-[minmax(0,1fr)] grid-rows-subgrid gap-y-0" data-e2e={`deck-${model}`} data-model={model}>
      <header className="mb-2 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <h3 className="mono min-w-0 text-[13px] uppercase leading-none tracking-[0.2em] text-text">
            <button type="button" onClick={onOpenDeck} className="press text-left uppercase hover:text-sense-pale" title="Open this MicroModel's deck in the side panel: its switches, base versions and every cartridge">
              {role ? roleTitle(role) : "No role"}
            </button>
          </h3>
          <span className="shrink-0">
            <Pill tone={tone}>{state}</Pill>
          </span>
        </div>
        <div className="mt-1.5 flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5">
          <BaseMark base={slot?.serving} />
          <WeightsMark model={model} />
        </div>
        <div className="mono mt-1.5 flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-[10px] text-faint">
          <span title="The newest call on this MicroModel in the engine's live record">{lastUsed ? `last call ${lastUsed}` : "no call in the live record"}</span>
          {live?.vram_mb ? <span>{live.vram_mb.toLocaleString()} MiB on the card</span> : null}
          {live?.precision ? <span>{live.precision}</span> : null}
        </div>
      </header>
      <div
        data-active={live?.executing ? "true" : undefined}
        className="deck-bay flex min-h-[15rem] min-w-0 flex-col justify-end px-2.5 pt-3"
        data-drop={dragging ? (fit?.fits ? "fits" : "no") : undefined}
        title={dragging && !fit?.fits ? `${dragging.name} does not fit: ${fit?.why ?? ""}` : undefined}
        onDragOver={(e) => {
          if (!fit?.fits) return;
          e.preventDefault();
          e.dataTransfer.dropEffect = "move";
          if (!over) setOver(true);
        }}
        onDragLeave={(e) => {
          // Leaving for a child of the bay is not leaving the bay.
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOver(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (dragging && fit?.fits) onDropKart(dragging);
        }}
      >
        {/* The free height of the bay above the stack, in the one-bit dither of an empty state. */}
        <div aria-hidden className="dither-slot mb-2 min-h-0 !w-auto flex-1 opacity-25" style={{ height: "auto" }} />
        <ol className="flex flex-col-reverse" aria-label="Docked cartridges, bottom up">
          {landAt === 0 && landingSlot}
          {docked.flatMap((k, i) => [
            <li key={k.key} className={i > 0 && landAt !== i ? "-mb-1" : ""} style={{ zIndex: i + 2 }}>
              <CartridgeCard k={k.meta === role ? { ...k, meta: "" } : k} selected={pickKey === k.key} landing={landing === k.key} busy={busyKey === k.key} link={linkOf?.(k)} onOpen={() => onOpen(k)} onEject={k.eject ? () => onEject(k) : undefined} />
            </li>,
            landAt === i + 1 ? landingSlot : null,
          ])}
        </ol>
        {docked.length === 0 && landAt < 0 && <div className="mono py-3 text-center text-[10px] text-quiet">nothing docked</div>}
        <div className="mono mb-1 mt-2 flex items-baseline justify-between text-[9px] uppercase tracking-[0.16em] text-quiet">
          <span>base slot</span>
          {pending && <span className="normal-case tracking-normal text-attention" title="Chosen; it serves from the next start of the engine">{pending === "open" ? "open weights" : pending} from next start</span>}
        </div>
        {landsBase && dragging && (
          <div className={`${slotCls} mb-1`} data-e2e="deck-landing">
            {over ? "release to seat · serves from the next start" : `${dragging.name} ${dragging.version} seats here`}
          </div>
        )}
        <div data-e2e="base-slot" style={{ zIndex: 1 }}>
          {base ? (
            <CartridgeCard
              k={base}
              selected={pickKey === base.key}
              landing={landing === base.key}
              busy={busyKey === base.key}
              link={linkOf?.(base)}
              onOpen={() => onOpen(base)}
              onEject={base.eject ? () => onEject(base) : undefined}
              extra={slot && <BaseChoice model={model} slot={slot} busy={baseBusy} onChoose={onChooseBase} />}
            />
          ) : (
            <div className="border border-dashed border-attention/60 px-3 py-3">
              <div className="mono text-[10px] uppercase tracking-[0.14em] text-attention">empty · open weights serve</div>
              {slot ? <BaseChoice model={model} slot={slot} busy={baseBusy} onChoose={onChooseBase} /> : <div className="mono mt-1 text-[10px] text-faint">base versions not reported</div>}
            </div>
          )}
        </div>
        <div aria-hidden className="deck-socket mt-1" />
        <div className="mono flex items-baseline justify-between gap-2 border-t border-line-strong px-1 py-1.5 text-[9.5px] uppercase tracking-[0.12em] text-faint">
          <span title={`${model}: the upstream open weights the base cartridge was built from`}>○ open weights</span>
          <span>{layers !== null ? `${layers} layers` : "layers not reported"}</span>
        </div>
      </div>
      <div className="min-w-0">
      <div className="mono mt-1.5 flex items-baseline justify-between gap-2 text-[10px] text-faint">
        <span>
          {docked.length + (base ? 1 : 0)} plugged in{allOn ? " · all connected" : ""}
        </span>
        {live?.executing ? <span className="text-signal">working</span> : null}
      </div>
      {(waiting.length > 0 || spare.length > 0) && (
        <div className="mt-2.5" data-e2e="deck-shelf">
          <div className="mono mb-1 text-[9px] uppercase tracking-[0.16em] text-quiet">in the cartridge library · fits</div>
          <ul className="grid gap-1">
            {[...waiting, ...spare].map((k) => (
              <li key={k.key} className={`flex min-w-0 items-center gap-2 border-l-2 py-0.5 pl-2 ${pickKey === k.key ? "border-sense-pale" : "border-line-strong"}`}>
                <i aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 ${KIND_SWATCH[k.kind]} opacity-60`} />
                <button type="button" onClick={() => onOpen(k)} className="press mono min-w-0 flex-1 truncate text-left text-[10.5px] text-muted hover:text-text" title={`${KIND_WORD[k.kind]} · ${k.version}${k.note ? ` · ${k.note}` : ""}`}>
                  {k.name} <span className="text-quiet">{k.version}</span>
                </button>
                {k.plug && (
                  <button type="button" disabled={busyKey === k.key || !k.ready} onClick={() => onPlug(k)} className={`${btn} shrink-0 border-sense-pale text-sense-pale hover:text-text`} aria-label={`Plug ${k.name} into the ${role || model}`} title={k.ready ? undefined : k.note}>
                    {busyKey === k.key ? "plugging" : k.kind === "base" ? "seat" : "plug in"}
                  </button>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
      {unfit.length > 0 && (
        <div className="mt-2.5" data-e2e="deck-unfit">
          <div className="mono mb-1 text-[9px] uppercase tracking-[0.16em] text-quiet" title="Made for this model, but it does not fit what serves on it now; the reason is on each line">does not fit now</div>
          <ul className="grid gap-1">
            {unfit.map((k) => (
              <li key={k.key} className="flex min-w-0 items-center gap-2 border-l-2 border-attention/50 py-0.5 pl-2">
                <button type="button" onClick={() => onOpen(k)} className="press mono min-w-0 flex-1 truncate text-left text-[10.5px] text-faint hover:text-text" title={fitOf(k, model).why ?? undefined}>
                  {k.name} <span className="text-quiet">{k.version}</span>
                </button>
                <span className="mono shrink-0 text-[9.5px] text-attention" title={fitOf(k, model).why ?? undefined}>
                  no fit
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {switches && (
        // The switches open over the rack, at their own width (a deck can be narrower than them),
        // from this deck's left edge or, on a deck to the right, from its right edge.
        <details className="group relative mt-2.5">
          <summary className="press mono flex cursor-pointer list-none items-center justify-between border-t border-line pt-1.5 text-[9.5px] uppercase tracking-[0.14em] text-faint hover:text-text group-open:text-text [&::-webkit-details-marker]:hidden">
            <span>model · precision</span>
            <span aria-hidden className="transition-transform group-open:rotate-180">⌄</span>
          </summary>
          <div className={`absolute top-full z-20 mt-1.5 w-[max(100%,19rem)] border border-line-strong bg-bg-1 p-1.5 ${switchesRight ? "right-0" : "left-0"}`}>{switches}</div>
        </details>
      )}
      </div>
    </article>
  );
}

/** A deck drawn while the engine has not answered yet. */
export function DeckSkeleton() {
  return (
    <div className="row-span-3 grid min-w-0 grid-cols-[minmax(0,1fr)] grid-rows-subgrid gap-y-0" aria-hidden data-e2e="deck-loading">
      <div>
        <div className="placeholder mb-2 h-3 w-24" />
        <div className="placeholder mb-3 h-2.5 w-40" />
      </div>
      <div className="deck-bay flex min-h-[15rem] flex-col justify-end gap-1.5 px-2.5 pb-2 pt-3">
        <div className="placeholder h-10 opacity-50" />
        <div className="placeholder h-12" />
        <div aria-hidden className="deck-socket mt-1" />
      </div>
      <div />
    </div>
  );
}

const GROUPS: { id: string; label: string; kinds: KartKind[] }[] = [
  { id: "all", label: "All", kinds: ["base", "filter", "lens", "adapter", "comp", "preference", "settings", "learned"] },
  { id: "base", label: "Base", kinds: ["base"] },
  { id: "filter", label: "Filters", kinds: ["filter"] },
  { id: "learned", label: "Learned", kinds: ["lens", "adapter", "preference", "comp"] },
  { id: "other", label: "Engine", kinds: ["settings", "learned"] },
];

/** The library in the side panel: every cartridge as a small card, by MicroModel, with a search,
 *  a filter by kind and by state. Drag one onto the deck it fits, or plug it in with its button. */
export function LibraryPanel({ karts, models, roleOf, pickKey, busyKey, onOpen, onPlug, onEject, onDragStart, onDragEnd }: {
  karts: Kart[];
  models: string[];
  roleOf: (model: string) => string;
  pickKey: string | null;
  busyKey: string | null;
  onOpen: (k: Kart) => void;
  onPlug: (k: Kart) => void;
  onEject: (k: Kart) => void;
  onDragStart: (k: Kart) => void;
  onDragEnd: () => void;
}) {
  const [q, setQ] = useState("");
  const [group, setGroup] = useState("all");
  const [plug, setPlug] = useState<"any" | "in" | "out">("any");
  const kinds = GROUPS.find((g) => g.id === group)?.kinds ?? [];
  const shown = karts.filter((k) => kinds.includes(k.kind) && (plug === "any" || (plug === "in") === k.plugged) && (!q || `${k.name} ${k.version} ${k.meta} ${k.model ?? ""} ${KIND_WORD[k.kind]}`.toLowerCase().includes(q.toLowerCase())));
  // The MicroModel by its role; its upstream open weights as the quiet ring, never as its name.
  const sections: { title: string; model?: string; list: Kart[] }[] = [
    ...models.map((m) => ({ title: roleOf(m) ? roleTitle(roleOf(m)) : "no role", model: m, list: shown.filter((k) => k.model === m) })),
    { title: "the whole engine", list: shown.filter((k) => k.model === null || !models.includes(k.model)) },
  ].filter((s) => s.list.length > 0);
  const chip = (on: boolean) => `press mono border-b-2 py-1 text-[10px] uppercase tracking-[0.12em] ${on ? "border-sense-pale text-text" : "border-transparent text-faint hover:text-text"}`;
  return (
    <section aria-label="Cartridge library" className="min-w-0" data-e2e="cartridge-library">
      <div className="mono mb-2 flex items-baseline justify-between gap-3 text-[10.5px] text-faint">
        <span>
          {karts.length} cartridges · {karts.filter((k) => k.plugged).length} plugged in
        </span>
        <span className="text-quiet" title="Drag a card onto the deck it fits, or use its plug button. A base cartridge serves from the next start.">
          drag onto a deck
        </span>
      </div>
      <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search the cartridge library" aria-label="Search the cartridge library" className="mono mb-2 h-9 w-full border-b border-line-strong bg-transparent text-[12px] outline-none placeholder:text-faint focus-visible:border-sense-pale" />
      <div className="mb-3 flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <ul className="flex flex-wrap gap-x-3 gap-y-1" role="radiogroup" aria-label="Kind of cartridge">
          {GROUPS.map((g) => (
            <li key={g.id}>
              <button type="button" role="radio" aria-checked={group === g.id} onClick={() => setGroup(g.id)} className={chip(group === g.id)}>
                {g.label}
              </button>
            </li>
          ))}
        </ul>
        <ul className="flex gap-x-3" role="radiogroup" aria-label="Plugged in or not">
          {(["any", "in", "out"] as const).map((p) => (
            <li key={p}>
              <button type="button" role="radio" aria-checked={plug === p} onClick={() => setPlug(p)} className={chip(plug === p)}>
                {p === "any" ? "any" : p === "in" ? "in" : "out"}
              </button>
            </li>
          ))}
        </ul>
      </div>
      <div className="grid gap-4">
        {sections.length === 0 && <div className="mono py-6 text-center text-[11px] text-quiet">nothing matches</div>}
        {sections.map((s) => (
          <section key={s.title} className="min-w-0">
            <div className="mono mb-1.5 flex items-baseline gap-3 truncate text-[10px] uppercase tracking-[0.14em] text-faint" title={s.title}>
              {s.title}
              {s.model && (
                <span className="normal-case tracking-normal">
                  <WeightsMark model={s.model} />
                </span>
              )}
            </div>
            <ul className="grid gap-1.5">
              {s.list.map((k) => (
                <li key={k.key}>
                  <CartridgeCard
                    k={k}
                    size="mini"
                    selected={pickKey === k.key}
                    busy={busyKey === k.key}
                    onOpen={() => onOpen(k)}
                    onPlug={k.plug ? () => onPlug(k) : undefined}
                    onEject={k.eject ? () => onEject(k) : undefined}
                    onDragStart={() => onDragStart(k)}
                    onDragEnd={onDragEnd}
                  />
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </section>
  );
}

/** Which MicroModel each cartridge fits, as a table: a row per cartridge, a column per installed
 *  MicroModel; a mark where it fits, the reason where it does not. */
export function CompatMatrix({ rows, models, roleOf }: { rows: { key: string; name: string; kind: string; made_for: string; fits: string[]; not: Record<string, string> }[]; models: string[]; roleOf: (m: string) => string }) {
  if (!rows.length) return <div className="mono text-[11px] text-faint">No cartridge in the cartridge library yet.</div>;
  return (
    <div className="overflow-x-auto" data-e2e="cartridge-compat">
      <table className="mono w-full text-[10.5px]">
        <thead>
          <tr className="text-left text-faint">
            <th className="py-1.5 pr-3 font-normal">cartridge</th>
            {models.map((m) => (
              <th key={m} className="px-1 py-1.5 text-center font-normal" title={`${m}: the upstream open weights this MicroModel's base cartridge was built from`}>
                <span className="uppercase tracking-[0.1em] text-muted">{roleOf(m) ? roleTitle(roleOf(m)).slice(0, 4) : "-"}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} className="border-t border-line/60">
              <td className="max-w-[11rem] py-1 pr-3" title={`${r.key} · made for ${r.made_for}`}>
                <span className="block truncate text-text">{r.name}</span>
                <span className="text-[9px] uppercase tracking-[0.1em] text-faint">{KIND_WORD[r.kind as KartKind] ?? r.kind}</span>
              </td>
              {models.map((m) => {
                const ok = r.fits.includes(m);
                return (
                  <td key={m} className="px-1 py-1 text-center" title={ok ? `fits ${m}` : `${m}: ${r.not[m] ?? "does not fit"}`}>
                    {ok ? <span aria-label="fits" className="inline-block h-2.5 w-2.5 bg-sense-pale" /> : <span aria-label="does not fit" className="text-quiet">·</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
