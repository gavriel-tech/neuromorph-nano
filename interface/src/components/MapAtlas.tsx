"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "aslo the parameters main page ui should be way better.
// more visual and interactive with real fucntionality." and "The main core of the pages should be on
// actual work flow and core function visuals"): the parameter maps of every MicroModel drawn as one
// atlas, the working middle of the Parameters page. Each MicroModel is a lane: its job filters as live
// switches, a heat strip of its layers (how many shown maps cover each one, stacked by kind; hover a
// layer to light the maps on it, click it to show only those), its maps as tiles grouped by what they
// are for (sized by bytes, coloured by kind, lit by state), and the cartridges they feed. Hovering a
// tile or a cartridge lights its whole lineage in the lane; clicking a tile opens it in the right rail.
// Ctrl or shift click picks several for one archive or delete. Every action is the same route as MCP
// and the API, logged in Audit.

import { useMemo, useState, type MouseEvent } from "react";
import DeleteFlow from "@/components/DeleteFlow";
import { useSelectMap } from "@/components/MapDetail";
import { BaseMark, WeightsMark, roleTitle } from "@/components/MicroModel";
import { Switch } from "@/components/ui";
import { bytes } from "@/lib/api";
import {
  ALL_STATES,
  GROUPS,
  STATES,
  STATE_SWATCH,
  cleanable,
  filterOf,
  graphOf,
  groupOf,
  isMapId,
  kindTone,
  layersOf,
  nodeKind,
  passes,
  stateOf,
  sum,
  useMaps,
  versionOf,
  walk,
  type JobFilter,
  type MapFilter,
  type MapRow,
  type RoleMaps,
} from "@/components/MapCore";

export type { MapFilter } from "@/components/MapCore";

const chip = "press mono inline-flex items-center gap-1.5 border px-2 py-1 text-[10px] uppercase tracking-[0.1em] transition-colors";
const chipOn = "border-line-strong bg-bg-2 text-text";
const chipOff = "border-line text-faint hover:text-muted";
const act = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const delCls = "press mono border-b border-line px-0.5 text-[10px] uppercase tracking-[0.1em] text-faint hover:text-error disabled:opacity-40";
const STATE_ORDER = (m: MapRow) => STATES.indexOf(stateOf(m) as (typeof STATES)[number]);

/** A tile's width: the log of its bytes, so a 300 byte ledger and a 50 MB map both read. */
const tileW = (b: number) => Math.round(Math.min(150, Math.max(64, 64 + 20 * Math.log10(Math.max(b, 1000) / 1000))));

/** A map's look by its state: what matters now is lit, what is kept for the record is dim. */
function tileLook(m: MapRow): string {
  // A failed or duplicate map is marked as such, whatever its state.
  if (m.lineage?.outcome === "failed" || m.lineage?.outcome === "duplicate") return "border-error/70 bg-bg-1 opacity-70";
  switch (m.state) {
    case "current":
      return "border-verified/60 bg-bg-2 opacity-100";
    case "stale":
      return "border-dashed border-attention/70 bg-bg-1 opacity-95";
    case "origin not recorded":
      return "border-dotted border-attention/60 bg-bg-1 opacity-90";
    case "log":
      return "border-line bg-bg-1 opacity-80";
    case "history":
      return "border-line bg-bg-1 opacity-65";
    default:
      return "border-line/60 bg-transparent opacity-50";
  }
}

// ---------------------------------------------------------------------------------------------
// The summary ribbon and the filters (above both views).

/** Every map of every MicroModel as one bar of bytes by state; a segment shows only that state. */
export function MapSummary({ filter, setFilter }: { filter: MapFilter; setFilter: (f: MapFilter) => void }) {
  const { roles } = useMaps();
  const rows = roles.filter((r) => filter.role === "all" || r.role === filter.role).flatMap((r) => r.maps);
  const total = sum(rows) || 1;
  const solo = filter.states.length === 1 ? filter.states[0] : null;
  const clean = rows.filter(cleanable);
  return (
    <div className="grid gap-1.5" data-e2e="pmap-summary">
      <div className="flex h-5 w-full min-w-0 gap-px overflow-hidden" role="group" aria-label="Maps by state, by size on disk: click a state to show only it">
        {STATES.map((s) => {
          const ms = rows.filter((m) => stateOf(m) === s);
          const b = sum(ms);
          if (!ms.length) return null;
          const on = !solo || solo === s;
          return (
            <button
              key={s}
              type="button"
              aria-pressed={solo === s}
              onClick={() => setFilter({ ...filter, states: solo === s ? ALL_STATES : [s] })}
              title={`${s}: ${ms.length} ${ms.length === 1 ? "map" : "maps"}, ${bytes(b)}${solo === s ? " (click for every state)" : " (click to show only these)"}`}
              className={`press h-full min-w-[3px] transition-opacity ${STATE_SWATCH[s]} ${on ? "opacity-90 hover:opacity-100" : "opacity-25 hover:opacity-50"}`}
              style={{ width: `${(100 * Math.max(b, total * 0.004)) / total}%` }}
            />
          );
        })}
      </div>
      <div className="mono flex flex-wrap items-baseline gap-x-4 gap-y-0.5 text-[10px] text-faint">
        {STATES.map((s) => {
          const ms = rows.filter((m) => stateOf(m) === s);
          return ms.length ? (
            <span key={s} className="flex items-baseline gap-1.5">
              <i aria-hidden className={`inline-block h-1.5 w-1.5 ${STATE_SWATCH[s]}`} />
              <span className={s === "stale" ? "text-attention" : "text-muted"}>{s}</span>
              {ms.length} · {bytes(sum(ms))}
            </span>
          ) : null;
        })}
        <span className="ml-auto" title="Experiments, history, stale maps and maps of unrecorded origin can be archived; the engine keeps what serves and what builds the serving base">
          {clean.length} cleanable · {bytes(sum(clean))}
        </span>
      </div>
    </div>
  );
}

/** The MicroModel, state, layer and word that the atlas and the list both show; and the view. */
/** The views of the middle of the Parameters page: the maps as an atlas or as a list, or the
 *  objects that hold parameters (the cartridge parameters) alone. */
export type MapView = "atlas" | "list" | "cartridges";
const VIEW_TITLE: Record<MapView, string> = { atlas: "The maps of every MicroModel as tiles, by what they are for", list: "The same maps as a grouped list", cartridges: "Every object holding parameters, per MicroModel: open one in the whole-set viewer to read, edit, mask or freeze its weights" };

export function AtlasFilters({ filter, setFilter, view, setView }: { filter: MapFilter; setFilter: (f: MapFilter) => void; view: MapView; setView: (v: MapView) => void }) {
  const { roles } = useMaps();
  const all = roles.flatMap((r) => r.maps.map((m) => ({ r, m })));
  const stateCount = (s: string) => all.filter(({ r, m }) => (filter.role === "all" || r.role === filter.role) && stateOf(m) === s).length;
  const toggleState = (s: string) => setFilter({ ...filter, states: filter.states.includes(s) ? filter.states.filter((x) => x !== s) : [...filter.states, s] });
  const shown = all.filter(({ r, m }) => (filter.role === "all" || r.role === filter.role) && passes(m, filter)).length;
  const findWords = view === "cartridges" ? "find: a cartridge, a kind, a state" : "find: a file, a kind, a cartridge, a job";
  const findCls = "mono h-7 border-b border-line-strong bg-transparent text-[11.5px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale";
  return (
    <div className="@container grid gap-2 border-t border-line pt-2.5">
      <div className="flex flex-wrap items-center gap-1.5">
        {["all", ...roles.map((r) => r.role)].map((x) => (
          <button key={x} type="button" aria-pressed={filter.role === x} onClick={() => setFilter({ ...filter, role: x })} className={`${chip} ${filter.role === x ? chipOn : chipOff}`}>
            {x === "all" ? "all" : roleTitle(x)}
            <span className="text-quiet">{x === "all" ? all.length : (roles.find((r) => r.role === x)?.maps.length ?? 0)}</span>
          </button>
        ))}
        {/* The find field: on this row when the middle is wide, on its own row when it is not. */}
        <input value={filter.text} onChange={(e) => setFilter({ ...filter, text: e.target.value })} placeholder={findWords} aria-label="Find" className={`${findCls} ml-2 hidden min-w-[10rem] flex-1 @[56rem]:block`} />
        <span className="ml-auto flex items-center gap-0" role="radiogroup" aria-label="View">
          {(["atlas", "list", "cartridges"] as const).map((v) => (
            <button key={v} type="button" role="radio" aria-checked={view === v} onClick={() => setView(v)} className={`${chip} -ml-px ${view === v ? chipOn : chipOff}`} title={VIEW_TITLE[v]} data-e2e="pmap-view" data-view={v}>
              {v}
            </button>
          ))}
        </span>
      </div>
      <input value={filter.text} onChange={(e) => setFilter({ ...filter, text: e.target.value })} placeholder={findWords} aria-label="Find" className={`${findCls} w-72 max-w-full @[56rem]:hidden`} />
      {/* The states belong to the maps; the cartridge parameters view has none to narrow by. */}
      <div className="flex flex-wrap items-center gap-1.5" data-e2e="pmap-states" hidden={view === "cartridges"}>
        {STATES.map((s) => (
          <button key={s} type="button" aria-pressed={filter.states.includes(s)} onClick={() => toggleState(s)} className={`${chip} ${filter.states.includes(s) ? chipOn : chipOff}`}>
            <i aria-hidden className={`inline-block h-1.5 w-1.5 ${STATE_SWATCH[s]}`} />
            {s}
            <span className="text-quiet">{stateCount(s)}</span>
          </button>
        ))}
        <button type="button" onClick={() => setFilter({ ...filter, states: filter.states.length === STATES.length ? ["current", "stale", "origin not recorded"] : ALL_STATES })} className="press mono ml-1 text-[10px] uppercase tracking-[0.1em] text-faint hover:text-text">
          {filter.states.length === STATES.length ? "what matters now" : "every state"}
        </button>
        {filter.layer !== null && (
          <button type="button" onClick={() => setFilter({ ...filter, layer: null })} className={`${chip} ${chipOn}`}>
            layer {filter.layer} ×
          </button>
        )}
        <span className="mono ml-auto text-[10.5px] text-faint" title="Click a tile to open it in the side panel; ctrl or shift click to pick several">
          {shown} of {all.length} maps
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
// The lanes.

type Hot = { role: string; t: "map" | "layer" | "cart"; id: string } | null;
type Lit = { files: Set<string>; carts: Set<string>; layers: Set<number> };

/** What a hovered tile, layer or cartridge lights in its lane: its lineage both ways. */
function litOf(r: RoleMaps, hot: Hot): Lit | null {
  if (!hot || hot.role !== r.role) return null;
  if (hot.t === "layer") {
    const l = Number(hot.id);
    return { files: new Set(r.maps.filter((m) => layersOf(m).includes(l)).map((m) => m.file)), carts: new Set(), layers: new Set([l]) };
  }
  const g = graphOf(r);
  const rel = new Set([...walk(g, hot.id, "up").keys(), ...walk(g, hot.id, "down").keys()]);
  const files = new Set([...rel].filter(isMapId));
  const carts = new Set([...rel].filter((x) => !isMapId(x)));
  const m = hot.t === "map" ? r.maps.find((x) => x.file === hot.id) : undefined;
  return { files, carts, layers: new Set(m ? layersOf(m) : []) };
}

/** A role's maps by state, as one bar of their bytes. */
function StateBar({ maps }: { maps: MapRow[] }) {
  const total = sum(maps) || 1;
  return (
    <div className="flex h-1.5 w-32 overflow-hidden bg-line" title={STATES.map((s) => `${s}: ${maps.filter((m) => stateOf(m) === s).length}`).join(" · ")}>
      {STATES.map((s) => {
        const b = sum(maps.filter((m) => stateOf(m) === s));
        return b > 0 ? <i key={s} className={`h-full ${STATE_SWATCH[s]}`} style={{ width: `${(100 * b) / total}%` }} /> : null;
      })}
    </div>
  );
}

/** The layers of one MicroModel: per layer, the shown maps covering it, stacked by kind. */
function LayerHeat({ maps, total, layer, lit, onLayer, onHover }: { maps: MapRow[]; total: number; layer: number | null; lit: Lit | null; onLayer: (l: number | null) => void; onHover: (l: number | null) => void }) {
  const cols = useMemo(
    () =>
      Array.from({ length: total }, (_, i) => {
        const on = maps.filter((m) => layersOf(m).includes(i));
        const kinds = new Map<string, number>();
        for (const m of on) kinds.set(m.kind, (kinds.get(m.kind) ?? 0) + 1);
        return { on, kinds: [...kinds.entries()] };
      }),
    [maps, total],
  );
  const max = Math.max(1, ...cols.map((c) => c.on.length));
  const bare = cols.filter((c) => c.on.length === 0).length;
  return (
    <div className="grid gap-0.5">
      <div className="flex h-9 w-full items-end gap-px" role="group" aria-label="Layers: hover one to light the maps on it, click it to show only those" onMouseLeave={() => onHover(null)}>
        {cols.map((c, i) => {
          const hl = lit?.layers.has(i) ?? false;
          const dim = lit !== null && lit.layers.size > 0 && !hl;
          return (
            <button
              key={i}
              type="button"
              aria-pressed={layer === i}
              onClick={() => onLayer(layer === i ? null : i)}
              onMouseEnter={() => onHover(i)}
              onFocus={() => onHover(i)}
              title={`layer ${i}: ${c.on.length} ${c.on.length === 1 ? "map" : "maps"}${c.kinds.length ? ` (${c.kinds.map(([k, n]) => `${n} ${k}`).join(", ")})` : ""}`}
              className={`press relative flex h-full min-w-0 flex-1 flex-col-reverse bg-line/30 transition-opacity hover:bg-line/70 ${layer === i ? "outline outline-1 outline-text" : hl ? "outline outline-1 outline-sense-pale/70" : ""} ${dim ? "opacity-35" : ""}`}
            >
              {c.on.length > 0 && (
                <span aria-hidden className="flex w-full flex-col-reverse" style={{ height: `${18 + 82 * (c.on.length / max)}%` }}>
                  {c.kinds.map(([k, n]) => (
                    <i key={k} className={`block w-full ${kindTone(k)} opacity-80`} style={{ flexGrow: n }} />
                  ))}
                </span>
              )}
            </button>
          );
        })}
      </div>
      <div className="mono relative h-3 text-[9px] text-quiet">
        {Array.from({ length: Math.floor((total - 1) / 4) + 1 }, (_, i) => (
          <span key={i} className="absolute -translate-x-1/2" style={{ left: `${((i * 4 + 0.5) / total) * 100}%` }}>
            {i * 4}
          </span>
        ))}
        <span className="absolute right-0">{bare ? `${bare} bare` : `${total} layers`}</span>
      </div>
    </div>
  );
}

/** One map as a tile; experiments are small squares, so a sweep reads as a swarm. */
function Tile({ m, f, picked, selected, lit, total, onPick, onHover }: { m: MapRow; f: JobFilter | undefined; picked: boolean; selected: boolean; lit: Lit | null; total: number | null; onPick: (multi: boolean) => void; onHover: (on: boolean) => void }) {
  const ls = layersOf(m);
  const on = f ? f.enabled !== false : null;
  const glow = lit?.files.has(m.file) ?? false;
  const dim = lit !== null && !glow;
  const small = m.state === "experiment";
  const cls = `press group relative min-w-0 border text-left transition-[opacity,border-color,box-shadow] duration-150 hover:opacity-100 ${tileLook(m)} ${selected ? "!border-text !opacity-100" : ""} ${picked ? "!border-sense-pale !opacity-100 ring-1 ring-sense-pale/50" : ""} ${glow ? "!opacity-100 shadow-[0_0_0_1px_var(--sense-pale)]" : ""} ${dim ? "!opacity-20" : ""}`;
  const common = {
    type: "button" as const,
    "data-e2e": "pmap-tile",
    "data-file": m.file,
    "aria-pressed": selected || picked,
    onClick: (e: MouseEvent) => onPick(e.ctrlKey || e.metaKey || e.shiftKey),
    onMouseEnter: () => onHover(true),
    onMouseLeave: () => onHover(false),
    onFocus: () => onHover(true),
    onBlur: () => onHover(false),
    title: `${m.file} · ${m.kind} · ${stateOf(m)}${m.state_why ? `: ${m.state_why}` : ""} · ${bytes(m.bytes)}`,
  };
  if (small)
    return (
      <button {...common} aria-label={m.file} className={`${cls} h-6 w-6 p-0`}>
        <i aria-hidden className={`absolute inset-[5px] ${kindTone(m.kind)} opacity-60`} />
        {picked && <span className="mono absolute -right-1 -top-1.5 text-[9px] text-sense-pale">✓</span>}
      </button>
    );
  return (
    <button {...common} style={{ width: tileW(m.bytes) }} className={`${cls} grid gap-1 p-1.5`}>
      <i aria-hidden className={`absolute inset-x-0 top-0 h-0.5 ${kindTone(m.kind)}`} />
      <span className="mono truncate text-[10px] text-text">{m.file.replace(/\.jsonl?$/, "")}</span>
      <span className="mono flex items-center justify-between gap-1 text-[9px] text-faint">
        <span className="truncate">{bytes(m.bytes)}</span>
        {on !== null && <i aria-label={on ? "plugged in" : "unplugged"} className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${on ? "bg-signal shadow-[0_0_6px_var(--signal)]" : "border border-faint"}`} />}
      </span>
      {ls.length > 0 && total !== null ? (
        <span aria-hidden className="flex h-1 gap-px">
          {Array.from({ length: total }, (_, i) => (
            <i key={i} className={`h-full flex-1 ${ls.includes(i) ? "bg-sense-pale/80" : "bg-line/60"}`} />
          ))}
        </span>
      ) : f?.layers && total ? (
        <span className="mono text-[8.5px] text-quiet">
          {f.layers} of {total} layers
        </span>
      ) : null}
      {picked && <span className="mono absolute right-1 top-1 text-[9px] text-sense-pale">✓</span>}
    </button>
  );
}

/** The cartridges a lane's maps feed, in version order; hover one to light what made it. */
function Feeds({ r, jf, lit, onHover }: { r: RoleMaps; jf: JobFilter[]; lit: Lit | null; onHover: (id: string | null) => void }) {
  const carts = useMemo(() => {
    const ids = new Set<string>();
    for (const m of r.maps) {
      for (const u of m.lineage?.used_by ?? []) if (!isMapId(u)) ids.add(u);
      const on = m.lineage?.made_on;
      if (on && !on.startsWith("open:") && !isMapId(on)) ids.add(on);
    }
    const order = { base: 0, comp: 1, cartridge: 2, "filter-cart": 3 } as Record<string, number>;
    return [...ids].map((id) => ({ id, kind: nodeKind(id, r, jf) })).sort((a, b) => (order[a.kind] ?? 9) - (order[b.kind] ?? 9) || versionOf(a.id) - versionOf(b.id) || a.id.localeCompare(b.id));
  }, [r, jf]);
  if (!carts.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-1 gap-y-1" onMouseLeave={() => onHover(null)}>
      <span className="eyebrow mr-2 text-quiet" title="The cartridges this MicroModel's maps were made on or built, from their lineage records">
        feeds
      </span>
      {carts.map((c, i) => {
        const serving = c.id === r.serving_base;
        const f = c.kind === "filter-cart" ? jf.find((x) => x.cartridge === c.id) : undefined;
        const glow = lit?.carts.has(c.id) ?? false;
        const dim = lit !== null && !glow;
        const prev = carts[i - 1];
        return (
          <span key={c.id} className="flex items-center gap-1">
            {prev && prev.kind !== c.kind && <span aria-hidden className="mono mx-1 text-[10px] text-quiet">›</span>}
            <span
              tabIndex={0}
              onMouseEnter={() => onHover(c.id)}
              onFocus={() => onHover(c.id)}
              onBlur={() => onHover(null)}
              title={`${c.id}: ${c.kind === "base" ? "base cartridge" : c.kind === "filter-cart" ? "job filter cartridge" : c.kind === "comp" ? "compensation cartridge" : "cartridge"}${serving ? ", serving now" : ""}${f ? `, ${f.enabled !== false ? "plugged in" : "unplugged"}` : ""}. Hover to light the maps it comes from and the ones made on it.`}
              className={`mono inline-flex cursor-default items-center gap-1.5 border px-1.5 py-0.5 text-[10px] transition-opacity ${serving ? "border-learned/60 text-text" : glow ? "border-sense-pale/70 text-text" : "border-line text-faint"} ${dim ? "opacity-30" : ""}`}
            >
              <i aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 ${c.kind === "base" ? "bg-learned" : c.kind === "filter-cart" ? `rotate-45 ${f?.enabled !== false ? "bg-signal" : "border border-faint"}` : "border border-learned"}`} />
              {c.id}
            </span>
          </span>
        );
      })}
    </div>
  );
}

function Lane({ r, filter, setFilter, hot, setHot, picked, onPick, addPicks }: { r: RoleMaps; filter: MapFilter; setFilter: (f: MapFilter) => void; hot: Hot; setHot: (h: Hot) => void; picked: string[]; onPick: (key: string, multi: boolean) => void; addPicks: (keys: string[]) => void }) {
  const { layerCount, filtersOf, busy, run } = useMaps();
  const { selected } = useSelectMap();
  const total = layerCount(r.model);
  const jf = filtersOf(r.role);
  const lit = litOf(r, hot);
  // What matters now first, then the biggest.
  const shown = r.maps.filter((m) => passes(m, filter)).sort((a, b) => STATE_ORDER(a) - STATE_ORDER(b) || b.bytes - a.bytes);
  const clean = shown.filter(cleanable);
  const plugged = jf.filter((x) => x.enabled !== false).length;
  return (
    <section className="grid min-w-0 gap-3 border-t border-line pt-4" data-e2e="pmap-lane" data-role={r.role}>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <span className="mono text-[13px] uppercase tracking-[0.16em] text-text">{roleTitle(r.role)}</span>
        <BaseMark base={r.serving_base} quiet />
        <WeightsMark model={r.model} />
        <span className="mono ml-auto flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-faint">
          <StateBar maps={r.maps} />
          <span>
            {shown.length} of {r.maps.length} · {bytes(sum(r.maps))}
          </span>
          {clean.length > 0 && (
            <button type="button" onClick={() => addPicks(clean.map((m) => `${r.role}/${m.file}`))} className={act} title="Picks every shown map of this MicroModel that can be archived, for one archive or delete">
              pick {clean.length} cleanable
            </button>
          )}
        </span>
      </div>

      {/* Job filters, live: what this MicroModel serves with right now. */}
      {jf.length > 0 && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1" title={`${plugged} of ${jf.length} job filters plugged in; the next call uses a change`}>
          <span className="eyebrow text-quiet">job filters</span>
          {jf.map((f) => {
            const on = f.enabled !== false;
            return (
              <span key={f.job} className="mono -my-2 inline-flex items-center gap-0.5 text-[10.5px]" onMouseEnter={() => setHot({ role: r.role, t: "map", id: `job-filter-${f.job}.json` })} onMouseLeave={() => setHot(null)}>
                <Switch on={on} busy={busy} label={`${f.job}: ${on ? "unplug" : "plug in"}`} onChange={() => void run(`${r.role}/job-filter-${f.job}.json`, "/api/surgery/job-filter", { role: r.role, job: f.job, action: on ? "disable" : "enable" }, `Job filter ${roleTitle(r.role)}/${f.job} ${on ? "unplugged" : "plugged in"}; the next call uses the change.`)} />
                <span className={on ? "text-text" : "text-faint"}>{f.job}</span>
                {typeof f.layers === "number" && <span className="text-quiet">· {f.layers}L</span>}
              </span>
            );
          })}
        </div>
      )}

      {total !== null && <LayerHeat maps={shown} total={total} layer={filter.layer} lit={lit} onLayer={(l) => setFilter({ ...filter, layer: l })} onHover={(l) => setHot(l === null ? null : { role: r.role, t: "layer", id: String(l) })} />}

      {shown.length === 0 ? (
        <p className="mono text-[10.5px] text-faint">no map of the {roleTitle(r.role)} matches what is shown</p>
      ) : (
        <div className="flex min-w-0 flex-wrap gap-x-6 gap-y-3">
          {GROUPS.map((g) => {
            const items = shown.filter((m) => groupOf(m) === g.id);
            if (!items.length) return null;
            return (
              <div key={g.id} className="grid min-w-0 max-w-full content-start gap-1.5">
                <div className="mono flex items-baseline gap-2 text-[9px] uppercase tracking-[0.14em] text-quiet" title={g.about}>
                  {g.short}
                  <span className="normal-case tracking-normal">
                    {items.length} · {bytes(sum(items))}
                  </span>
                </div>
                <div className="flex min-w-0 flex-wrap gap-1">
                  {items.map((m) => {
                    const key = `${r.role}/${m.file}`;
                    return <Tile key={key} m={m} f={filterOf(jf, m)} picked={picked.includes(key)} selected={selected === key} lit={lit} total={total} onPick={(multi) => onPick(key, multi)} onHover={(on) => setHot(on ? { role: r.role, t: "map", id: m.file } : null)} />;
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <Feeds r={r} jf={jf} lit={lit} onHover={(id) => setHot(id === null ? null : { role: r.role, t: "cart", id })} />
    </section>
  );
}

export default function MapAtlas({ filter, setFilter }: { filter: MapFilter; setFilter: (f: MapFilter) => void }) {
  const { roles, busy, run, onNote } = useMaps();
  const { selected, select } = useSelectMap();
  const [picked, setPicked] = useState<string[]>([]);
  const [armed, setArmed] = useState(false);
  const [hot, setHot] = useState<Hot>(null);
  const all = useMemo(() => roles.flatMap((r) => r.maps.map((m) => ({ r, m }))), [roles]);
  const shownRoles = roles.filter((r) => filter.role === "all" || r.role === filter.role);
  // Picks of maps that have left (archived, removed) drop out by themselves.
  const pickedRows = all.filter(({ r, m }) => picked.includes(`${r.role}/${m.file}`));
  const pickedClean = pickedRows.filter((x) => cleanable(x.m));
  const pickedRoles = [...new Set(pickedRows.map((x) => x.r.role))];
  const pickedDel = pickedRows.filter((x) => x.m.kind !== "job filter");
  const onPick = (key: string, multi: boolean) => {
    if (multi) {
      setPicked((p) => (p.includes(key) ? p.filter((x) => x !== key) : [...p, key]));
      setArmed(false);
    } else select(selected === key ? null : key);
  };
  const addPicks = (keys: string[]) => {
    setPicked((p) => [...new Set([...p, ...keys])]);
    setArmed(false);
  };
  const archivePicked = async () => {
    setArmed(false);
    for (const role of [...new Set(pickedClean.map((x) => x.r.role))]) {
      const files = pickedClean.filter((x) => x.r.role === role).map((x) => x.m.file);
      await run(`${role}:bulk`, "/api/parameter-maps/archive", { role, files }, `${pickedClean.length} ${pickedClean.length === 1 ? "map" : "maps"} moved into their MicroModels' archives. Undo in Audit, or restore them from the Archive tab.`);
    }
    setPicked([]);
  };

  return (
    <section className="grid min-w-0 gap-5" data-e2e="pmap-atlas" aria-label="Map atlas">
      {/* Several picked: one action for all of them. */}
      {pickedRows.length > 0 && (
        <div className="mono sticky top-0 z-10 flex flex-wrap items-center gap-x-4 gap-y-1 border border-sense-pale/40 bg-bg-1/95 px-3 py-2 text-[10.5px] backdrop-blur" data-e2e="pmap-picked">
          <span className="text-text">
            {pickedRows.length} picked · {bytes(sum(pickedRows.map((x) => x.m)))}
          </span>
          {armed ? (
            <>
              <span className="text-muted">
                Archive {pickedClean.length} of them? {pickedRows.length - pickedClean.length > 0 ? `${pickedRows.length - pickedClean.length} stay (they serve, build the serving base, or are kept at a fixed place). ` : ""}Restore any time; undo in Audit.
              </span>
              <button type="button" autoFocus disabled={busy} onClick={() => void archivePicked()} className={act}>
                confirm archive
              </button>
              <button type="button" onClick={() => setArmed(false)} className={act}>
                cancel
              </button>
            </>
          ) : (
            <>
              <button type="button" disabled={busy || pickedClean.length === 0} onClick={() => setArmed(true)} className={act} title={pickedClean.length ? "Moves them into their MicroModels' archives" : "None of these can be archived: they serve, build the serving base, or are kept at a fixed place"}>
                archive {pickedClean.length}
              </button>
              {pickedRoles.length === 1 ? (
                <DeleteFlow<{ audit_id?: string }>
                  path="/api/parameter-maps/remove"
                  body={{ role: pickedRoles[0], files: pickedDel.map((x) => x.m.file), confirm: true }}
                  label={`delete ${pickedDel.length}`}
                  title="Asks what would go first, then asks you to type a word; deleting cannot be undone (archiving can). Job filters are removed from their own detail."
                  startClass={delCls}
                  disabled={busy || pickedDel.length === 0}
                  done={() => `${pickedDel.length} ${roleTitle(pickedRoles[0])} maps deleted. Logged in Audit.`}
                  onDone={() => setPicked([])}
                  refresh={["/api/parameter-maps"]}
                  onNote={onNote}
                />
              ) : (
                <span className="text-faint">delete picks one MicroModel at a time</span>
              )}
              <button type="button" onClick={() => setPicked([])} className={act}>
                clear
              </button>
            </>
          )}
        </div>
      )}

      {shownRoles.map((r) => (
        <Lane key={r.role} r={r} filter={filter} setFilter={setFilter} hot={hot} setHot={setHot} picked={picked} onPick={onPick} addPicks={addPicks} />
      ))}

      {/* The key. */}
      <div className="mono flex flex-wrap gap-x-5 gap-y-1 border-t border-line pt-3 text-[9.5px] text-quiet">
        <span className="flex items-center gap-1.5">
          <i className="inline-block h-2.5 w-4 border border-verified/60" /> current
        </span>
        <span className="flex items-center gap-1.5">
          <i className="inline-block h-2.5 w-4 border border-dashed border-attention/70" /> stale
        </span>
        <span className="flex items-center gap-1.5">
          <i className="inline-block h-2.5 w-4 border border-line opacity-60" /> history
        </span>
        <span className="flex items-center gap-1.5">
          <i className="inline-block h-2.5 w-2.5 border border-line/60 opacity-60" /> experiment
        </span>
        <span className="flex items-center gap-1.5">
          <i className="inline-block h-1.5 w-1.5 rounded-full bg-signal" /> job filter plugged in
        </span>
        <span>width: size on disk · top line: kind · strip: layers covered · hover: lineage · ctrl or shift click: pick several</span>
      </div>
    </section>
  );
}
