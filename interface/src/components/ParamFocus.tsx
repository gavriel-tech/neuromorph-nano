"use client";

// One object's parameters as the working system in the middle of the Parameters page, opened from
// "view parameters" on Cartridges (`/parameters/?cartridge=<key>`) or from the cartridge
// parameters lane. The object's tensors and points lie on its MicroModel's layer strip with the norm
// of what it changes per layer; beside it, its mapping: the parameter maps it was measured on, used
// by, or that cover the same layers and points, their values drawn on the same layers. The drill
// down goes tensor, layer, then a grid of the values (paged by region, never loaded whole), where a
// drawn selection (a cell, a range, a block, a row, a channel) shows its values beside the map
// values at the same place and its place in the tensor's distribution. Any operation of the edit
// engine runs on the selection: a preview shows old against new with the norm change, and Apply
// makes a new version (on trial, plugged out, the next adapter version or a candidate unit), audited
// with undo; the source is never written. The same reads and edits are MCP nm_cartridges parameters
// and parameters_edit and GET and POST /api/cartridges/parameters.
//
// The overlay: any selection, tensor, layer or the whole object can be edited, masked or frozen in
// the object's parameter overlay (held beside it, never in its files, applied when the engine seats
// it; the default for the open weights). Masked regions are drawn dimmed and frozen ones hatched on
// the value grid, the layer strip and the tensor chips, and as read-only marks on the viewer's heatmap
// (masked dimmed, frozen hatched, edited outlined). All adjusting is in the controls block under the
// viewer: for the current selection, what the overlay does there (by entry) and its mapping (the
// related maps' values there); the overlay's entries, each a click away from selecting its region
// and showing it in the viewer, with unmask, unfreeze and undo. The same are MCP nm_cartridges overlay, overlay_edit, mask, unmask,
// freeze, unfreeze, overlay_undo and overlay_clear, and GET and POST /api/parameters/overlay.

import Link from "next/link";
import { maskPath, maskPathsIn } from "@/lib/privacy";
import { useEffect, useMemo, useRef, useState } from "react";
import { KIND_WORD, ParamDetail, ParamObjectPicker, NormStrip, fmt, type ParamLayer } from "@/components/CartridgeParams";
import ParamOverview, { type ActiveMapOverlay } from "@/components/ParamOverview";
import { layersOf, type MapRow as CatalogMapRow, type RoleMaps } from "@/components/MapCore";
import { NoteLine } from "@/components/Controls";
import { roleTitle } from "@/components/MicroModel";
import { useRail } from "@/components/Workspace";
import { Empty, Pill, Placeholder } from "@/components/ui";
import { post, refreshAll, useApi } from "@/lib/api";
import { overlaps, rectsOf, type Dims } from "@/lib/paramgeom";

type Note = { text: string; tone: "verified" | "error" };
type TensorRow = { name: string; target: string; layer: number | null; part: string; shape: number[]; dtype: string; size: number; identity: number | null; read: boolean; paged: boolean; values_editable: boolean; l2: number | null; max_abs: number | null; effect_l2: number | null; unread?: string | null };
type OvEntry = { id: string; action: "edit" | "mask" | "freeze"; op: string | null; selection: string; whole: boolean; regions: { path: string; tensor: string; values: number }[]; values: number; by: string; at_ms: number; note: string | null };
type TakesEffect = { view: string; when: string; reason: string };
type OverlaySec = { editable?: boolean; why?: string | null; entries?: OvEntry[]; counts?: { edits: number; masks: number; freezes: number }; tensors?: Record<string, { edited: number; masked: number; frozen: number }>; layers?: { layer: number; edited: boolean; masked: boolean; frozen: boolean }[]; whole_masked?: boolean; takes_effect?: TakesEffect; error?: string };
type Section = { key?: string; source?: string; kind?: string; role?: string; model?: string; state?: string; holds?: boolean; note?: string; error?: string; editable?: { why: string | null; small: number }; layers?: ParamLayer[]; tensors?: TensorRow[]; overlay?: OverlaySec };
type MapRow = { file: string; kind: string; why: string[]; made_on: string | null; used_by: string[]; targets: string[]; layers: { layer: number; value: number | null }[]; read: boolean };
type Maps = { maps?: MapRow[]; base?: string | null; error?: string };
type MapAt = { map: string; at: string; values?: (number | null)[]; value?: number | null };
type Region = { error?: string; overlay?: [number | null, boolean, boolean][] | null; tensor?: string; shape?: number[]; selected?: number; indices?: number[]; values?: (number | null)[]; identity?: number | null; summary?: { l2: number | null; max_abs: number | null; mean: number | null; min: number | null; max: number | null }; distribution?: { lo: number | null; hi: number | null; counts: number[] } | null; maps?: MapAt[] };
type DiffRow = { name: string; old_shape: number[]; shape: number[]; changed: number; old_l2: number | null; new_l2: number | null; old_effect_l2: number | null; new_effect_l2: number | null; max_change: number | null; sample: [number, number, number][] };
type Preview = { diff?: { tensors: DiffRow[]; changed: number; added: string[]; dropped: string[]; touched?: number; paged?: { name: string; touched: number; sample: [number, number, number][] | string }[] }; writes?: string; sockets?: string | null; takes_effect?: TakesEffect };
type OvReply = { changed?: boolean; audit_id?: string; added?: string[]; removed?: string[]; change?: string; takes_effect?: TakesEffect; error?: string };

/** The hatch drawn over a frozen region. */
const HATCH = "repeating-linear-gradient(45deg, var(--text) 0 1px, transparent 1px 4px)";
const OVERLAY_ROUTE = "/api/parameters/overlay";
function refreshOverlay() {
  for (const p of ["/api/cartridges/parameters", OVERLAY_ROUTE, "/api/audit"]) void refreshAll(p);
}

const GRID = 64;
const OPS = ["set", "add", "mul", "scale", "clamp", "zero", "identity", "negate", "paste", "copy", "interpolate", "drop", "rank"] as const;
type OpName = (typeof OPS)[number];
const link = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const go = "press mono min-h-8 border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40";
const field = "mono h-7 min-w-0 border-b border-line bg-transparent px-1 text-[10.5px] text-text outline-none focus-visible:border-sense-pale";
const enc = encodeURIComponent;

/** The window of a tensor shown in the grid: a path and how to map a cell to a flat index. */
function windowOf(t: TensorRow, r0: number, c0: number) {
  if (t.shape.length === 2) {
    const [R, C] = t.shape;
    const r1 = Math.min(R, r0 + GRID);
    const c1 = Math.min(C, c0 + GRID);
    return { path: `${t.name}[${r0}:${r1}, ${c0}:${c1}]`, rows: r1 - r0, cols: c1 - c0, flat: (i: number, j: number) => (r0 + i) * C + (c0 + j), two: true, R, C };
  }
  const a = r0 * GRID;
  const b = Math.min(t.size, a + GRID * GRID);
  return { path: `${t.name}[${a}:${b}]`, rows: Math.ceil((b - a) / GRID), cols: Math.min(GRID, b - a), flat: (i: number, j: number) => a + i * GRID + j, two: false, R: Math.ceil(t.size / GRID), C: GRID };
}

/** A drawn selection as a path: one cell, a range of a vector, a block of a matrix. */
function selPath(t: TensorRow, a: number, b: number): string {
  if (t.shape.length === 2) {
    const C = t.shape[1];
    const [ra, ca, rb, cb] = [Math.floor(a / C), a % C, Math.floor(b / C), b % C];
    return `${t.name}[${Math.min(ra, rb)}:${Math.max(ra, rb) + 1}, ${Math.min(ca, cb)}:${Math.max(ca, cb) + 1}]`;
  }
  return `${t.name}[${Math.min(a, b)}:${Math.max(a, b) + 1}]`;
}

/** The selected values beside the map values at the same place, and their place in the tensor's
 *  distribution. */
function Selection({ k, path }: { k: string; path: string }) {
  const q = useApi<Region>(`/api/cartridges/parameters?key=${enc(k)}&path=${enc(path)}`);
  const r = q.data;
  if (!r) return <Placeholder className="h-16" what="region" />;
  if (r.error) return <p className="mono text-[10.5px] text-error">{r.error}</p>;
  const d = r.distribution;
  const mean = r.summary?.mean ?? null;
  const bin = d && mean !== null && d.hi !== null && d.lo !== null && d.hi > d.lo ? Math.min(d.counts.length - 1, Math.max(0, Math.floor(((mean - d.lo) / (d.hi - d.lo)) * d.counts.length))) : null;
  const max = d ? Math.max(1, ...d.counts) : 1;
  const rows = (r.indices ?? []).slice(0, 16);
  const pointMaps = (r.maps ?? []).filter((m) => m.values);
  const layerMaps = (r.maps ?? []).filter((m) => m.value !== undefined);
  return (
    <div className="grid gap-2" data-e2e="cparam-selection">
      <div className="mono flex flex-wrap gap-x-4 text-[10px] text-faint">
        <span className="text-text">{r.selected} selected</span>
        <span>L2 {fmt(r.summary?.l2)}</span>
        <span>mean {fmt(mean)}</span>
        <span>
          {fmt(r.summary?.min)} to {fmt(r.summary?.max)}
        </span>
      </div>
      {d && (
        <div className="flex h-6 items-end gap-px" title="The tensor's distribution; the marked bin holds the selection's mean" data-e2e="cparam-distribution">
          {d.counts.map((c, i) => (
            <i key={i} className={`block w-full ${i === bin ? "bg-sense-pale" : "bg-muted/50"}`} style={{ height: `${Math.max(c ? 10 : 2, (100 * c) / max)}%` }} />
          ))}
        </div>
      )}
      <table className="mono w-full text-left text-[10px]">
        <thead className="text-faint">
          <tr>
            <th className="py-0.5 pr-2 font-normal">index</th>
            <th className="py-0.5 pr-2 font-normal">value</th>
            {r.overlay && <th className="py-0.5 pr-2 font-normal" title="with the object's parameter overlay">overlaid</th>}
            {pointMaps.map((m) => (
              <th key={m.map} className="max-w-[8rem] truncate py-0.5 pr-2 font-normal" title={`${m.map}: ${m.at}`}>
                {m.map.replace(/\.json$/, "")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((ix, i) => (
            <tr key={ix} className="border-t border-line/40">
              <td className="py-0.5 pr-2 text-faint">{ix}</td>
              <td className="py-0.5 pr-2 text-text">{fmt(r.values?.[i])}</td>
              {r.overlay && (
                <td className={`py-0.5 pr-2 ${r.overlay[i]?.[1] ? "text-quiet" : "text-sense-pale"}`} data-e2e="cparam-overlaid">
                  {fmt(r.overlay[i]?.[0])}
                  {r.overlay[i]?.[1] ? " masked" : ""}
                  {r.overlay[i]?.[2] ? " frozen" : ""}
                </td>
              )}
              {pointMaps.map((m) => (
                <td key={m.map} className="py-0.5 pr-2 text-muted">
                  {fmt(m.values?.[i])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {(r.indices?.length ?? 0) > rows.length && <span className="mono text-[9.5px] text-quiet">first {rows.length} of {r.selected} shown</span>}
      {layerMaps.length > 0 && (
        <div className="mono flex flex-wrap gap-x-4 text-[10px] text-faint">
          {layerMaps.map((m) => (
            <span key={m.map} title={m.at}>
              {m.map}: {fmt(m.value)} at {m.at}
            </span>
          ))}
        </div>
      )}
      {pointMaps.length === 0 && layerMaps.length === 0 && <span className="mono text-[9.5px] text-quiet">no parameter map holds values at this place</span>}
    </div>
  );
}

/** A grid of one window of a tensor's values; a click picks a cell, shift and click draws a range or
 *  a block. */
function Grid({ k, t, onPick }: { k: string; t: TensorRow; onPick: (path: string, label: string) => void }) {
  const [r0, setR0] = useState(0);
  const [c0, setC0] = useState(0);
  const [anchor, setAnchor] = useState<number | null>(null);
  const [headDim, setHeadDim] = useState(128);
  const [end, setEnd] = useState<number | null>(null);
  const w = windowOf(t, r0, c0);
  const q = useApi<Region>(t.paged ? `/api/cartridges/parameters?key=${enc(k)}&path=${enc(w.path)}` : null);
  const vals = q.data?.values ?? [];
  const id = t.identity ?? 0;
  const dev = Math.max(1e-12, ...vals.map((v) => Math.abs((v ?? id) - id)));
  const lo = anchor !== null && end !== null ? Math.min(anchor, end) : null;
  const hi = anchor !== null && end !== null ? Math.max(anchor, end) : null;
  const inSel = (ix: number) => {
    if (lo === null || hi === null) return false;
    if (!w.two) return ix >= lo && ix <= hi;
    const C = t.shape[1];
    const [ra, ca, rb, cb] = [Math.floor(lo / C), lo % C, Math.floor(hi / C), hi % C];
    const [r, c] = [Math.floor(ix / C), ix % C];
    return r >= Math.min(ra, rb) && r <= Math.max(ra, rb) && c >= Math.min(ca, cb) && c <= Math.max(ca, cb);
  };
  const click = (ix: number, shift: boolean) => {
    const a = shift && anchor !== null ? anchor : ix;
    setAnchor(a);
    setEnd(ix);
    const path = selPath(t, a, ix);
    onPick(path, path);
  };
  if (!t.paged) return <p className="mono text-[10.5px] text-faint">{t.unread ?? "its values are not held here"}</p>;
  return (
    <div className="grid gap-2" data-e2e="cparam-grid">
      <div className="mono flex flex-wrap items-center justify-between gap-2 text-[10px] text-faint">
        <span>
          {w.two ? `rows ${r0} to ${r0 + w.rows - 1} of ${w.R}, columns ${c0} to ${c0 + w.cols - 1} of ${w.C}` : `values ${r0 * GRID} to ${r0 * GRID + vals.length - 1} of ${t.size}`} · click a value, shift and click to draw a range
        </span>
        <span className="flex gap-3">
          <button type="button" className={link} disabled={r0 === 0} onClick={() => setR0(Math.max(0, r0 - (w.two ? GRID : GRID)))}>
            up
          </button>
          <button type="button" className={link} disabled={w.two ? r0 + GRID >= w.R : (r0 + GRID) * GRID >= t.size} onClick={() => setR0(r0 + GRID)}>
            down
          </button>
          {w.two && (
            <>
              <button type="button" className={link} disabled={c0 === 0} onClick={() => setC0(Math.max(0, c0 - GRID))}>
                left
              </button>
              <button type="button" className={link} disabled={c0 + GRID >= w.C} onClick={() => setC0(c0 + GRID)}>
                right
              </button>
            </>
          )}
        </span>
      </div>
      {q.data?.error ? (
        <p className="mono text-[10.5px] text-error">{q.data.error}</p>
      ) : !q.data ? (
        <Placeholder className="h-24" what="region detail" />
      ) : (
        <div className="grid w-full max-w-[640px] gap-px" style={{ gridTemplateColumns: `repeat(${w.cols}, minmax(0, 1fr))` }} role="grid" aria-label={`${t.name} values`}>
          {vals.map((v, i) => {
            const ix = (q.data?.indices ?? [])[i] ?? w.flat(Math.floor(i / w.cols), i % w.cols);
            const ov = q.data?.overlay?.[i];
            const shown = ov ? ov[0] : v;
            const d = (shown ?? id) - id;
            const pct = Math.round((100 * Math.abs(d)) / dev);
            const col = d >= 0 ? "var(--sense-pale)" : "var(--attention)";
            return (
              <button
                key={ix}
                type="button"
                onClick={(e) => click(ix, e.shiftKey)}
                title={`${ix}: ${v}${ov && ov[0] !== v ? ` (overlay ${ov[0]})` : ""}${ov?.[1] ? " · masked" : ""}${ov?.[2] ? " · frozen" : ""}`}
                className={`press aspect-square min-w-0 ${inSel(ix) ? "outline outline-1 outline-text" : ""} ${ov?.[1] ? "opacity-30" : ""}`}
                style={{ background: ov?.[2] ? `${HATCH}, color-mix(in srgb, ${col} ${pct}%, var(--line))` : `color-mix(in srgb, ${col} ${pct}%, var(--line))` }}
                data-e2e="cparam-cell"
                data-index={ix}
                data-masked={ov?.[1] ? "1" : undefined}
                data-frozen={ov?.[2] ? "1" : undefined}
              />
            );
          })}
        </div>
      )}
      {anchor !== null && w.two && (
        <div className="mono flex flex-wrap gap-3 text-[10px]">
          <button type="button" className={link} onClick={() => { const r = Math.floor(anchor / t.shape[1]); onPick(`${t.name}[${r}:${r + 1}, :]`, `row ${r}`); }}>
            its row
          </button>
          <button type="button" className={link} onClick={() => { const c = anchor % t.shape[1]; onPick(`${t.name}[:, ${c}:${c + 1}]`, `channel ${c}`); }}>
            its channel
          </button>
          {/* An attention head is a block of rows (head size rows each, the Qwen3 MicroModels use 128). */}
          <span className="flex items-center gap-1" data-e2e="cparam-head">
            <button
              type="button"
              className={link}
              disabled={!(headDim > 0) || t.shape[0] % headDim !== 0}
              title={t.shape[0] % headDim === 0 ? `rows of one head of ${headDim}` : `${t.shape[0]} rows do not split into heads of ${headDim}`}
              onClick={() => { const r = Math.floor(anchor / t.shape[1]); const hd = Math.floor(r / headDim); onPick(`${t.name}[${hd * headDim}:${(hd + 1) * headDim}, :]`, `head ${hd} (${headDim} rows)`); }}
            >
              its head
            </button>
            <input
              value={headDim}
              onChange={(e) => setHeadDim(Math.max(1, Math.floor(Number(e.target.value) || 0)))}
              aria-label="Head size in rows"
              className="mono h-6 w-12 border-b border-line-strong bg-transparent text-[10px] text-text outline-none"
              inputMode="numeric"
            />
          </span>
        </div>
      )}
    </div>
  );
}

/** The edit of the selection: an operation, a preview of old against new, Apply in the object's
 *  overlay (the default; the only way for the open weights) or as a new version. */
function EditPanel({ k, path, maps, onDone, canVersion, why }: { k: string; path: string; maps: MapRow[]; onDone: (newKey: string) => void; canVersion: boolean; why: string | null }) {
  const [target, setTarget] = useState<"overlay" | "version">("overlay");
  const [op, setOp] = useState<OpName>("scale");
  const [value, setValue] = useState("1");
  const [lo, setLo] = useState("-1");
  const [hi, setHi] = useState("1");
  const [t, setT] = useState("0.5");
  const [source, setSource] = useState("");
  const [sourceB, setSourceB] = useState("");
  const [paste, setPaste] = useState("");
  const [previewed, setPreviewed] = useState<{ sig: string; r: Preview } | null>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<Note | null>(null);
  const [made, setMade] = useState<string | null>(null);
  const mapSources = maps.filter((m) => m.file.startsWith("job-filter-") && m.targets.length > 0);
  const inOverlay = target === "overlay" || !canVersion;
  const ops = inOverlay ? OPS.filter((o) => o !== "drop" && o !== "rank") : OPS;
  const body = (dry: boolean): Record<string, unknown> => {
    const b: Record<string, unknown> = { key: k, op, path, preview: dry };
    if (inOverlay) b.action = "edit";
    if (["set", "add", "mul", "scale"].includes(op)) b.value = Number(value);
    if (op === "rank") b.rank = Number(value);
    if (op === "clamp") Object.assign(b, { lo: Number(lo), hi: Number(hi) });
    if (op === "copy" || op === "interpolate") b.source = source.startsWith("map:") ? { map: source.slice(4) } : { key: source };
    if (op === "interpolate") b.t = Number(t);
    // Between two versions when a second is given: source + t * (source_b - source).
    if (op === "interpolate" && sourceB.trim()) b.source_b = sourceB.startsWith("map:") ? { map: sourceB.slice(4) } : { key: sourceB.trim() };
    if (op === "paste") b.values = paste.split(/[\s,]+/).filter(Boolean).map(Number);
    return b;
  };
  // A preview stands for the exact request it was made for; a new selection or operation needs a
  // new preview before Apply.
  const sig = JSON.stringify(body(true));
  const preview = previewed?.sig === sig ? previewed.r : null;
  const send = async (dry: boolean) => {
    setBusy(true);
    try {
      const r = await post<Preview & OvReply & { key?: string; state?: string }>(inOverlay ? OVERLAY_ROUTE : "/api/cartridges/parameters", body(dry));
      if (dry) {
        setPreviewed({ sig, r });
        setNote(null);
      } else if (inOverlay) {
        setPreviewed(null);
        const te = r.takes_effect;
        setNote({ text: `Overlay entry ${(r.added ?? []).join(", ")}: ${te?.when ?? ""}${te?.reason ? ` (${te.reason})` : ""}. Audit ${r.audit_id ?? ""} undoes it.`, tone: "verified" });
        refreshOverlay();
      } else {
        setPreviewed(null);
        setMade(r.key ?? null);
        setNote({ text: r.changed === false ? "Nothing changed." : `New version ${r.key}: ${r.state}. Audit ${r.audit_id ?? ""} sets it aside.`, tone: "verified" });
        for (const p of ["/api/cartridges", "/api/parameter-maps", "/api/surgery", "/api/audit", "/api/units"]) void refreshAll(p);
      }
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  return (
    <div className="grid gap-3 border-t border-line pt-3" data-e2e="cparam-edit">
      <div className="mono flex flex-wrap items-center gap-2 text-[10.5px]">
        <span className="text-faint">on</span>
        <span className="text-text">{path}</span>
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <label className="grid gap-0.5">
          <span className="mono text-[9.5px] uppercase tracking-[0.12em] text-quiet">apply in</span>
          <select value={inOverlay ? "overlay" : "version"} onChange={(e) => setTarget(e.target.value as "overlay" | "version")} className={field} aria-label="Apply in" data-e2e="cparam-target" title={canVersion ? "The overlay holds the edit beside the object; a new version is a separate object" : (why ?? "a new version is not made from this object")}>
            <option value="overlay" className="bg-bg-1">
              the overlay
            </option>
            <option value="version" className="bg-bg-1" disabled={!canVersion}>
              a new version
            </option>
          </select>
        </label>
        <label className="grid gap-0.5">
          <span className="mono text-[9.5px] uppercase tracking-[0.12em] text-quiet">operation</span>
          <select value={op} onChange={(e) => setOp(e.target.value as OpName)} className={field} aria-label="Operation" data-e2e="cparam-op">
            {ops.map((o) => (
              <option key={o} value={o} className="bg-bg-1">
                {o}
              </option>
            ))}
          </select>
        </label>
        {["set", "add", "mul", "scale", "rank"].includes(op) && (
          <label className="grid gap-0.5">
            <span className="mono text-[9.5px] uppercase tracking-[0.12em] text-quiet">{op === "scale" ? "strength (0 to 4)" : op === "rank" ? "rank" : "value"}</span>
            <input value={value} onChange={(e) => setValue(e.target.value)} className={`${field} w-24`} aria-label="Value" data-e2e="cparam-value" />
          </label>
        )}
        {op === "clamp" && (
          <>
            <input value={lo} onChange={(e) => setLo(e.target.value)} className={`${field} w-20`} aria-label="Lower bound" />
            <input value={hi} onChange={(e) => setHi(e.target.value)} className={`${field} w-20`} aria-label="Upper bound" />
          </>
        )}
        {(op === "copy" || op === "interpolate") && (
          <label className="grid gap-0.5">
            <span className="mono text-[9.5px] uppercase tracking-[0.12em] text-quiet">from</span>
            <input list="cparam-sources" value={source} onChange={(e) => setSource(e.target.value)} placeholder="another version's key, or map:<role>/<file>" className={`${field} w-64`} aria-label="Source" />
            <datalist id="cparam-sources">
              {mapSources.map((m) => (
                <option key={m.file} value={`map:${m.file}`} />
              ))}
            </datalist>
          </label>
        )}
        {op === "interpolate" && <input value={t} onChange={(e) => setT(e.target.value)} className={`${field} w-16`} aria-label="Weight toward the source" />}
        {op === "interpolate" && <input list="cparam-sources" value={sourceB} onChange={(e) => setSourceB(e.target.value)} placeholder="and a second version (optional): blends the two" className={`${field} w-64`} aria-label="Second source" data-e2e="cparam-source-b" />}
        {op === "paste" && <input value={paste} onChange={(e) => setPaste(e.target.value)} placeholder="values in order, by commas" className={`${field} w-64`} aria-label="Values to paste" />}
      </div>
      <NoteLine note={note} />
      <div className="flex flex-wrap items-center gap-4">
        <button type="button" disabled={busy} onClick={() => void send(true)} className={link} data-e2e="cparam-preview-btn">
          preview
        </button>
        <button type="button" disabled={busy || !preview} onClick={() => void send(false)} className={go} data-e2e="cparam-apply-focus" title={preview ? (inOverlay ? "Writes the edit in the object's overlay; its files are never written" : "Makes a new version; the source is never written") : "Preview first"}>
          {busy ? "working" : inOverlay ? "apply in the overlay" : "apply as a new version"}
        </button>
        {made && (
          <button type="button" className={link} onClick={() => onDone(made)}>
            open {made}
          </button>
        )}
      </div>
      {preview && (
        <div className="grid gap-2" data-e2e="cparam-preview">
          <div className="mono flex flex-wrap gap-x-4 text-[10px] text-faint">
            <span className="text-text">{(() => { const nv = (preview.diff?.changed ?? 0) + (preview.diff?.paged ?? []).reduce((s, pg) => s + pg.touched, 0); return `${nv.toLocaleString()} ${nv === 1 ? "value changes" : "values change"}`; })()}</span>
            {(preview.diff?.added.length ?? 0) > 0 && <span>adds {preview.diff?.added.join(", ")}</span>}
            {(preview.diff?.dropped.length ?? 0) > 0 && <span>drops {preview.diff?.dropped.join(", ")}</span>}
            {preview.writes && <span>writes {maskPathsIn(preview.writes)}</span>}
            {preview.sockets && <span>{preview.sockets}</span>}
            {preview.takes_effect && <span data-e2e="cparam-takes-effect">{preview.takes_effect.when}</span>}
          </div>
          {(preview.diff?.paged ?? []).map((pg) => (
            <div key={pg.name} className="mono grid gap-1 border-l-2 border-sense-pale/50 pl-2 text-[10px]">
              <span className="text-muted">
                {pg.name} · {pg.touched.toLocaleString()} values touched (read by page)
              </span>
              {Array.isArray(pg.sample) && (
                <span className="flex flex-wrap gap-x-3 text-faint">
                  {pg.sample.slice(0, 12).map(([i, a, b]) => (
                    <span key={i}>
                      [{i}] {fmt(a)} to {fmt(b)}
                    </span>
                  ))}
                </span>
              )}
            </div>
          ))}
          {(preview.diff?.tensors ?? []).map((d) => (
            <div key={d.name} className="mono grid gap-1 border-l-2 border-sense-pale/50 pl-2 text-[10px]">
              <span className="text-muted">
                {d.name} · {d.changed} changed · L2 {fmt(d.old_l2)} to {fmt(d.new_l2)} · change {fmt(d.old_effect_l2)} to {fmt(d.new_effect_l2)} · largest {fmt(d.max_change)}
                {d.old_shape.join("x") !== d.shape.join("x") ? ` · shape ${d.old_shape.join("x")} to ${d.shape.join("x")}` : ""}
              </span>
              <span className="flex flex-wrap gap-x-3 text-faint">
                {d.sample.slice(0, 12).map(([i, a, b]) => (
                  <span key={i}>
                    [{i}] {fmt(a)} to {fmt(b)}
                  </span>
                ))}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** The overlay's marks on a tensor chip: edited, masked, frozen value counts. */
function ovMark(ov: OverlaySec | undefined, name: string): string {
  const m = ov?.tensors?.[name];
  if (!m) return "";
  const parts = [m.edited && `${m.edited} edited`, m.masked && `${m.masked} masked`, m.frozen && `${m.frozen} frozen`].filter(Boolean);
  return parts.length ? ` · ${parts.join(", ")}` : "";
}

/** Per layer, what the overlay does there: hatched where frozen, dimmed where masked, filled where
 *  edited. */
function OverlayStrip({ marks, total }: { marks: { layer: number; edited: boolean; masked: boolean; frozen: boolean }[]; total: number }) {
  const by = new Map(marks.map((m) => [m.layer, m]));
  return (
    <div className="flex h-2 w-full gap-px" role="group" aria-label="Layers the overlay edits, masks or freezes" data-e2e="cparam-overlay-strip">
      {Array.from({ length: total }, (_, i) => {
        const m = by.get(i);
        const what = m ? [m.edited && "edited", m.masked && "masked", m.frozen && "frozen"].filter(Boolean).join(", ") : "";
        return (
          <span
            key={i}
            title={m ? `layer ${i}: ${what}` : undefined}
            className={`block min-w-0 flex-1 ${m?.masked ? "bg-quiet/40" : m?.edited ? "bg-sense-pale/60" : "bg-transparent"}`}
            style={m?.frozen ? { backgroundImage: HATCH } : undefined}
            data-layer={i}
            data-masked={m?.masked ? "1" : undefined}
            data-frozen={m?.frozen ? "1" : undefined}
          />
        );
      })}
    </div>
  );
}

/** A tensor's two-dimensional view, as the viewer and the editor read paths on it. */
function dimsOf(t: TensorRow): Dims {
  const [r, ...rest] = t.shape;
  if (t.shape.length === 0) return { name: t.name, shape: t.shape, rows: 1, cols: 1 };
  if (t.shape.length === 1) return { name: t.name, shape: t.shape, rows: 1, cols: r };
  return { name: t.name, shape: t.shape, rows: r, cols: rest.reduce((a, b) => a * b, 1) };
}

/** The overlay entries that touch a selection: `*` all of them, `L<n>` those in layer n's tensors, a
 *  tensor or a region those whose regions share a weight with it. */
function entriesAt(path: string, entries: OvEntry[], tensors: TensorRow[], dims: Dims[]): OvEntry[] {
  if (path === "*") return entries;
  const layer = /^L(\d+)$/.exec(path.trim());
  if (layer) {
    const names = new Set(tensors.filter((t) => t.layer === Number(layer[1])).map((t) => t.name));
    return entries.filter((e) => e.whole || e.regions.some((r) => names.has(r.tensor)));
  }
  const sel = rectsOf(path, dims);
  if (!sel) return [];
  return entries.filter(
    (e) =>
      e.whole ||
      e.regions.some((r) => {
        if (r.tensor !== sel.tensor.name) return false;
        const at = rectsOf(r.path, dims);
        return !at || at.rects.some((a) => sel.rects.some((b) => overlaps(a, b)));
      }),
  );
}

/** What the current selection holds: what the overlay does there (edited and how, masked, frozen,
 *  by entry) and its mapping (the related parameter maps' values there; per layer where a map holds
 *  one value per layer). */
function SelectionState({ k, path, d, maps, onShow }: { k: string; path: string; d: Section; maps: MapRow[]; onShow: (e: OvEntry) => void }) {
  const tensors = useMemo(() => d.tensors ?? [], [d.tensors]);
  const dims = useMemo(() => tensors.map(dimsOf), [tensors]);
  const at = entriesAt(path, d.overlay?.entries ?? [], tensors, dims);
  const sel = rectsOf(path, dims);
  const size = sel ? sel.rects.reduce((s, r) => s + (r.r1 - r.r0) * (r.c1 - r.c0), 0) : 0;
  const layer = /^L(\d+)$/.exec(path.trim()) ? Number(path.trim().slice(1)) : sel ? (tensors.find((t) => t.name === sel.tensor.name)?.layer ?? null) : null;
  // The maps' values at the selection: read for a region of at most 65,536 weights.
  const q = useApi<Region>(sel && path.includes("[") && size > 0 && size <= 65_536 ? `/api/cartridges/parameters?key=${enc(k)}&path=${enc(path)}` : null);
  const point = (q.data?.maps ?? []).filter((m) => m.values && m.values.length > 0);
  const perLayer = layer === null ? [] : maps.map((m) => ({ file: m.file, v: m.layers.find((x) => x.layer === layer)?.value })).filter((m) => m.v !== undefined && m.v !== null);
  const does = (a: OvEntry["action"]) => at.filter((e) => e.action === a);
  const word = (e: OvEntry) => (e.action === "edit" ? `${e.op ?? "edit"}${e.note ? ` (${e.note})` : ""}` : e.action === "mask" ? "masked" : "frozen");
  return (
    <div className="mono grid gap-1 text-[10px]" data-e2e="cparam-selection-state">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
        <span className="text-[9.5px] uppercase tracking-[0.12em] text-quiet">here</span>
        {at.length === 0 ? (
          <span className="text-faint">nothing in the overlay touches the selection: it serves as the files hold it</span>
        ) : (
          (["edit", "mask", "freeze"] as const).map((a) =>
            does(a).length > 0 ? (
              <span key={a} className="flex flex-wrap items-baseline gap-x-1">
                <span className={a === "edit" ? "text-verified" : a === "mask" ? "text-quiet" : "text-muted"}>{a === "edit" ? "edited" : a === "mask" ? "masked" : "frozen"}</span>
                {does(a).map((e) => (
                  <button key={e.id} type="button" className={link} onClick={() => onShow(e)} title={`${word(e)} · ${e.whole ? "whole object" : e.selection} · ${e.values.toLocaleString()} values · by ${e.by}`} data-e2e="cparam-here-entry" data-id={e.id}>
                    {e.id}
                    {a === "edit" ? ` ${e.op ?? ""}` : ""}
                  </button>
                ))}
              </span>
            ) : null,
          )
        )}
      </div>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5" data-e2e="cparam-selection-maps">
        <span className="text-[9.5px] uppercase tracking-[0.12em] text-quiet">mapping</span>
        {point.map((m) => {
          const xs = (m.values ?? []).filter((x): x is number => x !== null && Number.isFinite(x));
          const mean = xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
          return (
            <span key={m.map} title={`${m.map}: ${m.at}`}>
              <span className="text-faint">{m.map.replace(/\.json$/, "")}</span> <span className="text-text">{xs.length === 1 ? fmt(mean) : `mean ${fmt(mean)} over ${xs.length}`}</span>
            </span>
          );
        })}
        {perLayer
          .filter((m) => !point.some((p) => p.map === m.file))
          .map((m) => (
            <span key={m.file} title={`${m.file}: one value per layer`}>
              <span className="text-faint">{m.file.replace(/\.json$/, "")}</span> <span className="text-text">{fmt(m.v)}</span> <span className="text-quiet">at layer {layer}</span>
            </span>
          ))}
        {point.length === 0 && perLayer.length === 0 && <span className="text-quiet">{layer === null && path === "*" ? "select a layer, tensor or region to see the maps there" : "no related parameter map holds a value here"}</span>}
        {point.length === 0 && perLayer.length > 0 && <span className="text-quiet">per layer only</span>}
      </div>
    </div>
  );
}

/** The overlay of the object: mask, freeze, unmask and unfreeze on the selection (the whole object
 *  for `*`), the entries with unmask or unfreeze each (a click on an entry selects its region and
 *  shows it in the viewer), undo of the newest change, and clear through the delete gate. */
function OverlayPanel({ k, path, ov, onShow, pulse = false }: { k: string; path: string; ov: OverlaySec | undefined; onShow: (e: OvEntry) => void; pulse?: boolean }) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<Note | null>(null);
  const [clearTok, setClearTok] = useState<string | null>(null);
  const whole = path === "*";
  const send = async (b: Record<string, unknown>) => {
    setBusy(true);
    try {
      const r = await post<OvReply & { confirm_token?: string; step?: string }>(OVERLAY_ROUTE, { key: k, ...b });
      if (r.step === "preview" && r.confirm_token) {
        setClearTok(r.confirm_token);
        setNote({ text: "Clearing removes every entry of the overlay; Audit undoes it. Confirm to clear.", tone: "verified" });
      } else {
        setClearTok(null);
        const te = r.takes_effect;
        setNote({ text: `${r.change ?? "changed"}: ${te?.when ?? ""}${te?.reason ? ` (${te.reason})` : ""}. Audit ${r.audit_id ?? ""} undoes it.`, tone: "verified" });
        refreshOverlay();
      }
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  if (!ov) return null;
  if (ov.error) return <p className="mono text-[10.5px] text-error">{ov.error}</p>;
  const entries = ov.entries ?? [];
  const sel = whole ? { whole: true } : { path };
  return (
    <div id="overlay-controls" className={`grid gap-2 border-t pt-3 transition-colors ${pulse ? "border-sense-pale bg-sense/10 px-2 pb-2" : "border-line"}`} data-e2e="cparam-overlay">
      <div className="mono flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px]">
        <span className="text-[9.5px] uppercase tracking-[0.12em] text-quiet">overlay</span>
        <span className="text-faint">on {whole ? "the whole object" : path}</span>
        <button type="button" className={link} disabled={busy || !ov.editable} onClick={() => void send({ action: "mask", ...sel })} data-e2e="cparam-mask">
          mask
        </button>
        <button type="button" className={link} disabled={busy || !ov.editable} onClick={() => void send({ action: "freeze", ...sel })} data-e2e="cparam-freeze">
          freeze
        </button>
        <button type="button" className={link} disabled={busy || !(ov.counts?.masks ?? 0)} onClick={() => void send({ action: "unmask", ...sel })} data-e2e="cparam-unmask">
          unmask
        </button>
        <button type="button" className={link} disabled={busy || !(ov.counts?.freezes ?? 0)} onClick={() => void send({ action: "unfreeze", ...sel })} data-e2e="cparam-unfreeze">
          unfreeze
        </button>
        <button type="button" className={link} disabled={busy || entries.length === 0} onClick={() => void send({ action: "undo" })} data-e2e="cparam-overlay-undo">
          undo last change
        </button>
        {clearTok ? (
          <button type="button" className={go} disabled={busy} onClick={() => void send({ action: "clear", confirm_token: clearTok })} data-e2e="cparam-overlay-clear-confirm">
            confirm clear
          </button>
        ) : (
          <button type="button" className={link} disabled={busy || entries.length === 0} onClick={() => void send({ action: "clear" })} data-e2e="cparam-overlay-clear">
            clear
          </button>
        )}
      </div>
      {ov.takes_effect && (
        <p className="mono text-[9.5px] text-faint" title={ov.takes_effect.reason} data-e2e="cparam-overlay-effect">
          a change is {ov.takes_effect.when}; {ov.takes_effect.reason}
        </p>
      )}
      <NoteLine note={note} />
      {entries.length === 0 ? (
        <span className="mono text-[9.5px] text-quiet">no entries: the object serves as its files hold it</span>
      ) : (
        <table className="mono w-full text-left text-[10px]" data-e2e="cparam-overlay-list">
          <caption className="pb-0.5 text-left text-[9.5px] text-quiet">
            {ov.counts?.edits ?? 0} edits · {ov.counts?.masks ?? 0} masks · {ov.counts?.freezes ?? 0} freezes · click one to select its region and show it in the viewer
          </caption>
          <tbody>
            {entries.map((e) => (
              <tr key={e.id} className={`border-t border-line/40 ${!e.whole && path === (e.regions[0]?.path ?? e.selection) ? "bg-bg-2" : ""}`} data-e2e="cparam-overlay-entry" data-id={e.id} data-action={e.action}>
                <td className="py-0.5 pr-2 text-faint">{e.id}</td>
                <td className={`py-0.5 pr-2 ${e.action === "edit" ? "text-verified" : e.action === "mask" ? "text-quiet" : "text-muted"}`}>{e.action === "edit" ? `edit ${e.op ?? ""}` : e.action === "mask" ? "masked" : "frozen"}</td>
                <td className="max-w-[18rem] py-0.5 pr-2" title={`${e.regions.map((r) => r.path).join(", ")}${e.note ? ` · ${e.note}` : ""} · by ${e.by}`}>
                  <button type="button" className="press block max-w-full truncate text-left text-muted underline decoration-line-strong underline-offset-2 hover:text-text" onClick={() => onShow(e)} data-e2e="cparam-overlay-select">
                    {e.whole ? "whole object" : e.selection} · {e.values.toLocaleString()} values
                  </button>
                </td>
                <td className="py-0.5 text-right">
                  {e.action !== "edit" && (
                    <button type="button" className={link} disabled={busy} onClick={() => void send({ action: e.action === "mask" ? "unmask" : "unfreeze", id: e.id })}>
                      {e.action === "mask" ? "unmask" : "unfreeze"}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/** The focused object: its tensors on the layer strip, its mapping, the drill down and the edit. */
// 2026-11-27 (parity gap B2): export the current selection's values, the same route as MCP and the
// API (POST /api/cartridges/parameters/export). It writes the values the selection names (at most
// 65,536, in the first tensor it names) to exports/parameters/ as nm-parameter-values/1; nothing of the object is
// written. The file can be pasted back with the paste operation or imported through the API.
function ExportPanel({ k, path }: { k: string; path: string }) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<Note | null>(null);
  const send = async () => {
    setBusy(true);
    try {
      const r = await post<{ file: string; tensor: string; values: number; format: string }>("/api/cartridges/parameters/export", { key: k, path });
      setNote({ text: `${r.values.toLocaleString()} values of ${r.tensor} written to ${maskPath(r.file)} (${r.format}); the object is unchanged.`, tone: "verified" });
      void refreshAll("/api/audit");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  return (
    <div className="grid gap-2 border-t border-line pt-3" data-e2e="cparam-export">
      <div className="mono flex flex-wrap items-center gap-3 text-[10.5px]">
        <span className="text-faint">values of</span>
        <span className="text-text">{path === "*" ? "the whole object" : path}</span>
        <button type="button" disabled={busy} onClick={() => void send()} className={link} data-e2e="cparam-export-btn" title="Writes the values of the first tensor the selection names (at most 65,536) to exports/parameters/ on this machine; nothing of the object is written">
          {busy ? "exporting" : "export values"}
        </button>
      </div>
      <NoteLine note={note} />
    </div>
  );
}

export default function ParamFocus({
  k,
  layerCount,
  activeMap = null,
  initialLayer = null,
  onSelectMap,
  onLeave,
  onOpen,
}: {
  k: string;
  layerCount: (model: string) => number | null;
  activeMap?: string | null;
  initialLayer?: number | null;
  onSelectMap?: (map: string | null) => void;
  onLeave: () => void;
  onOpen: (key: string) => void;
}) {
  const q = useApi<Section>(`/api/cartridges/parameters?key=${enc(k)}`);
  // The header alone (no value read), so the frame says what it is reading while the section is made.
  const head = useApi<{ tensors?: number | null }>(`/api/cartridges/parameters?key=${enc(k)}&head=1`);
  const m = useApi<Maps>(`/api/cartridges/parameters?key=${enc(k)}&maps=1`);
  const allMapsQ = useApi<{ roles?: RoleMaps[] }>("/api/parameter-maps", 15000);
  const rail = useRail();
  const [layer, setLayer] = useState<number | null>(initialLayer);
  const [tensor, setTensor] = useState<string | null>(null);
  const [path, setPath] = useState(initialLayer !== null ? `L${initialLayer}` : "*");
  const [ctlOpen, setCtlOpen] = useState(true);
  const [overlayPulse, setOverlayPulse] = useState(false);
  const [typed, setTyped] = useState("");
  const controlsRef = useRef<HTMLDivElement>(null);
  const focusOverlayControls = () => {
    setCtlOpen(true);
    setOverlayPulse(true);
    window.setTimeout(() => {
      const el = document.getElementById("overlay-controls");
      el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      const btn = el?.querySelector<HTMLButtonElement>('[data-e2e="cparam-mask"]');
      btn?.focus();
    }, 30);
    window.setTimeout(() => setOverlayPulse(false), 1200);
  };
  // A region to show in the viewer (an overlay entry picked in the controls); each pick a new `n`.
  const [target, setTarget] = useState<{ path: string; n: number } | null>(null);
  const leave = () => {
    if (rail.shown?.key?.startsWith("cparam:")) rail.clear();
    onLeave();
  };
  const showEntry = (e: OvEntry) => {
    const p = e.whole ? "*" : e.regions.length === 1 ? e.regions[0].path : e.selection;
    setPath(p);
    const first = e.regions[0]?.path;
    if (first) setTarget({ path: first, n: (target?.n ?? 0) + 1 });
  };
  const d = q.data;
  const tensors = useMemo(() => d?.tensors ?? [], [d]);
  const layers = d?.layers ?? [];
  const total = d?.model ? layerCount(d.model) : null;
  const width = total ?? Math.max(1, ...layers.map((l) => l.layer + 1));
  const shown = tensors.filter((t) => layer === null || t.layer === layer);
  const t = tensors.find((x) => x.name === tensor) ?? null;
  const maps = useMemo(() => m.data?.maps ?? [], [m.data]);
  const editable = !d?.editable?.why;
  // Only parameter maps paired/connected to THIS selected parameter object (from &maps=1), enriched with catalog metadata.
  const roleName = d?.role ?? null;
  const roleMaps = useMemo<CatalogMapRow[]>(() => {
    if (!roleName) return [];
    return allMapsQ.data?.roles?.find((r) => r.role === roleName)?.maps ?? [];
  }, [allMapsQ.data, roleName]);
  const pairedMaps = useMemo(() => {
    const byFile = new Map(roleMaps.map((rm) => [rm.file, rm]));
    return maps.map((mp) => {
      const cat = byFile.get(mp.file);
      return {
        file: mp.file,
        kind: cat?.kind ?? mp.kind,
        why: mp.why.join("; "),
      };
    });
  }, [maps, roleMaps]);
  const activeParsed = useMemo(() => {
    if (!activeMap) return null;
    const i = activeMap.indexOf("/");
    if (i < 0) return roleName ? { role: roleName, file: activeMap } : null;
    return { role: activeMap.slice(0, i), file: activeMap.slice(i + 1) };
  }, [activeMap, roleName]);
  const activeCatalog = useMemo(() => {
    if (!activeParsed) return null;
    return allMapsQ.data?.roles?.find((r) => r.role === activeParsed.role)?.maps.find((x) => x.file === activeParsed.file) ?? null;
  }, [allMapsQ.data, activeParsed]);
  // Read full JSON only for reasonably sized maps (<= 4 MiB, e.g. super-weights.json, job filters, tol-*).
  const readActivePath =
    activeParsed && (!activeCatalog || activeCatalog.bytes <= 4 * 1024 * 1024)
      ? `/api/parameter-maps/read?role=${enc(activeParsed.role)}&file=${enc(activeParsed.file)}`
      : null;
  const activeReadQ = useApi<{ mode?: string; content?: unknown; kind?: string; about?: string }>(readActivePath);
  const mapOverlay = useMemo<ActiveMapOverlay | null>(() => {
    if (!activeParsed) return null;
    const pointsByTensor: Record<string, [number, number][]> = {};
    const layerSet = new Set<number>();
    const layerValues: Record<number, number> = {};
    const targets: string[] = [];
    if (activeCatalog) {
      for (const l of layersOf(activeCatalog)) layerSet.add(l);
    }
    const related = maps.find((x) => x.file === activeParsed.file);
    if (related) {
      for (const lv of related.layers) {
        layerSet.add(lv.layer);
        if (typeof lv.value === "number") layerValues[lv.layer] = lv.value;
      }
      for (const tg of related.targets) targets.push(tg);
    }
    const content = activeReadQ.data?.mode === "full" ? activeReadQ.data.content : null;
    if (content && typeof content === "object" && !Array.isArray(content)) {
      const obj = content as Record<string, unknown>;
      // super-weights.json: { tensors: { "model.layers.0...": [[r, c], ...] } }
      if (obj.tensors && typeof obj.tensors === "object" && !Array.isArray(obj.tensors)) {
        for (const [tName, coords] of Object.entries(obj.tensors as Record<string, unknown>)) {
          if (!Array.isArray(coords)) continue;
          const pts: [number, number][] = [];
          for (const pair of coords) {
            if (Array.isArray(pair) && pair.length >= 2 && typeof pair[0] === "number" && typeof pair[1] === "number") {
              pts.push([pair[0], pair[1]]);
            }
          }
          if (pts.length > 0) {
            pointsByTensor[tName] = pts;
            const lm = /^model\.layers\.(\d+)\./.exec(tName);
            if (lm) layerSet.add(Number(lm[1]));
          }
        }
      }
      // job-filter-*.json: { parts: [{ target: "L2.swigate", gain: [...], bias: [...] }] }
      if (Array.isArray(obj.parts)) {
        for (const p of obj.parts) {
          if (!p || typeof p !== "object") continue;
          const tg = (p as { target?: unknown }).target;
          if (typeof tg === "string" && tg) {
            if (!targets.includes(tg)) targets.push(tg);
            const lm = /(?:^|[.#/:])L(\d+)(?:$|[.#/:])/.exec(tg);
            if (lm) layerSet.add(Number(lm[1]));
          }
        }
      }
    }
    return {
      id: `${activeParsed.role}/${activeParsed.file}`,
      file: activeParsed.file,
      role: activeParsed.role,
      kind: activeCatalog?.kind ?? activeReadQ.data?.kind ?? "parameter map",
      about: activeCatalog?.about ?? activeReadQ.data?.about,
      pointsByTensor,
      layers: [...layerSet].sort((a, b) => a - b),
      layerValues,
      targets,
    };
  }, [activeParsed, activeCatalog, maps, activeReadQ.data]);
  const totalOverlayPoints = useMemo(() => (mapOverlay ? Object.values(mapOverlay.pointsByTensor).reduce((a, b) => a + b.length, 0) : 0), [mapOverlay]);
  const totalOverlayTensors = useMemo(() => (mapOverlay ? Object.keys(mapOverlay.pointsByTensor).length : 0), [mapOverlay]);
  const overlayCount = (d?.overlay?.counts?.edits ?? 0) + (d?.overlay?.counts?.masks ?? 0) + (d?.overlay?.counts?.freezes ?? 0);
  // The object's summary and quick controls stay in the side panel.
  useEffect(() => {
    if (!d) return;
    rail.show({ key: `cparam:${k}`, title: <span className="truncate" title={k}>{k}</span>, body: <ParamDetail key={k} k={k} total={total} /> });
    // Shown once per object opened, when its MicroModel is known.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [k, d === null]);
  useEffect(() => {
    if (!d?.holds || typeof window === "undefined") return;
    if (window.location.hash === "#controls") {
      controlsRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [k, d?.holds]);

  if (!d && q.error) return <Empty title="The parameters did not answer">{q.error}</Empty>;
  if (!d)
    return (
      <section className="grid min-w-0 gap-5" data-e2e="cparam-focus" data-key={k} aria-busy="true">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <button
            type="button"
            className="press mono border border-line-strong bg-bg-2 px-2.5 py-1 text-[10px] uppercase tracking-[0.1em] text-text hover:border-sense-pale hover:text-sense-pale"
            onClick={leave}
            data-e2e="cparam-leave"
          >
            ← Parameters
          </button>
          <ParamObjectPicker value={k} onPick={onOpen} ariaLabel="Switch parameter object" />
          <span className="mono text-[10.5px] text-faint" data-e2e="cparam-reading">
            {head.data?.tensors != null ? `reading ${head.data.tensors} tensors` : "reading its parameters"}
          </span>
        </div>
        <Placeholder className="h-72" what="parameter map" />
      </section>
    );
  return (
    <section className="grid min-w-0 gap-5" data-e2e="cparam-focus" data-key={k}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <button
          type="button"
          className="press mono border border-line-strong bg-bg-2 px-2.5 py-1 text-[10px] uppercase tracking-[0.1em] text-text hover:border-sense-pale hover:text-sense-pale"
          onClick={leave}
          data-e2e="cparam-leave"
          title="Return to the main Parameters page (map atlas, list, and all parameter objects)"
        >
          ← Parameters
        </button>
        <ParamObjectPicker value={k} onPick={onOpen} ariaLabel="Switch parameter object" />
        {d.kind && (
          <Pill tone={d.kind === "base" ? "learned" : d.kind === "open" ? "muted" : "signal"}>
            {d.source === "job-filter" ? "job filter" : (KIND_WORD[d.kind] ?? d.kind)}
          </Pill>
        )}
        {d.role && <span className="mono text-[10.5px] uppercase tracking-[0.14em] text-muted">{roleTitle(d.role)}</span>}
        {d.state && <Pill tone={d.state === "trial" || d.state === "plugged out" ? "attention" : "verified"}>{d.state}</Pill>}
        <span className="mono ml-auto flex gap-4 text-[10px]">
          <Link href="/cartridges/" className={link} data-e2e="cparam-back">
            ← Cartridges
          </Link>
        </span>
      </div>
      {d.error ? (
        <Empty title="Not readable">{d.error}</Empty>
      ) : !d.holds ? (
        <Empty title="This cartridge holds no parameters">
          {d.editable?.why ?? "It holds operating settings rather than weight tensors."}{" "}
          <Link href="/cartridges/" className="link">
            Edit its settings on Cartridges
          </Link>
          , or pick another cartridge above.
        </Empty>
      ) : (
        <>
          {/* Parameter map overlay & parameter masking bar: lists only maps paired with this object, plus direct controls toggle & focus. */}
          <div className="mono flex flex-wrap items-center gap-x-3 gap-y-1.5 border border-line bg-bg-1/80 px-3 py-2 text-[10px]" data-e2e="param-map-overlay-bar">
            <span className="text-[9.5px] uppercase tracking-[0.12em] text-muted">map overlay</span>
            {onSelectMap && (pairedMaps.length > 0 || activeMap) ? (
              <select
                value={activeMap ?? ""}
                onChange={(e) => onSelectMap(e.target.value || null)}
                aria-label="Overlay a paired parameter map on the viewer"
                className="mono h-6 min-w-[13rem] max-w-full border border-line-strong bg-bg-2 px-1.5 text-[10px] text-text outline-none focus-visible:border-sense-pale"
              >
                <option value="" className="bg-bg-1 text-faint">
                  none ({pairedMaps.length} paired {pairedMaps.length === 1 ? "map" : "maps"})
                </option>
                {activeMap && activeParsed && !pairedMaps.some((pm) => pm.file === activeParsed.file) && (
                  <option value={activeMap} className="bg-bg-1 text-text">
                    {activeParsed.file}
                  </option>
                )}
                {pairedMaps.map((pm) => (
                  <option key={pm.file} value={`${d.role}/${pm.file}`} className="bg-bg-1 text-text">
                    {pm.file} · {pm.kind}
                  </option>
                ))}
              </select>
            ) : (
              <span className="text-quiet">{m.data ? "no paired maps for this object" : "checking paired maps..."}</span>
            )}
            {mapOverlay && (
              <>
                <span className="text-faint">
                  {totalOverlayPoints > 0
                    ? `${totalOverlayPoints.toLocaleString()} pinned weights across ${totalOverlayTensors} tensors`
                    : mapOverlay.layers.length > 0
                      ? `covers ${mapOverlay.layers.length} layers${mapOverlay.targets.length > 0 ? ` · ${mapOverlay.targets.length} sockets` : ""}`
                      : mapOverlay.kind}
                </span>
                {onSelectMap && (
                  <button type="button" className={link} onClick={() => onSelectMap(null)}>
                    clear map
                  </button>
                )}
              </>
            )}
            <span className="ml-auto flex flex-wrap items-center gap-3">
              <span className={overlayCount > 0 ? "text-attention" : "text-quiet"}>
                masking: {overlayCount > 0 ? `${d.overlay?.counts?.masks ?? 0} masked · ${d.overlay?.counts?.freezes ?? 0} frozen · ${d.overlay?.counts?.edits ?? 0} edited` : "none"}
              </span>
              <button
                type="button"
                className="press border border-sense-pale bg-sense/15 px-2 py-0.5 text-[10px] uppercase tracking-[0.08em] text-sense-pale hover:bg-sense/25 hover:text-text"
                onClick={focusOverlayControls}
                title="Open and highlight the mask, freeze, and overlay controls below"
              >
                mask / freeze
              </button>
              <button
                type="button"
                className={link}
                onClick={() => setCtlOpen((o) => !o)}
                aria-expanded={ctlOpen}
              >
                {ctlOpen ? "hide controls" : "show controls"}
              </button>
            </span>
          </div>
          {/* The whole-set viewer, the primary working view of an opened object: a region dragged
              or picked there becomes the selection the edit, mask and freeze controls act on. */}
          <ParamOverview
            key={k}
            objectKey={k}
            onSelect={(p) => setPath(p)}
            marks={d.overlay?.entries}
            maps={maps}
            mapOverlay={mapOverlay}
            layerFilter={layer}
            onSelectLayer={(l) => {
              setLayer(l);
              setTensor(null);
              setPath(l === null ? "*" : `L${l}`);
            }}
            target={target}
          />
          {/* The controls stay at the bottom of the screen while the viewer above scrolls, so a weight
              picked anywhere in the set is edited, masked or frozen without leaving it. */}
          <div
            ref={controlsRef}
            id="controls"
            className={`sticky bottom-0 z-10 grid gap-3 border-t border-line-strong bg-bg-1 pt-3 pb-2 ${ctlOpen ? "max-h-[46vh] overflow-y-auto" : "max-h-[2.5rem] overflow-hidden"}`}
            data-e2e="cparam-controls"
          >
            <div className="mono flex flex-wrap items-center gap-2 text-[10px]" data-e2e="cparam-current-selection">
              <span className="text-faint">selection</span>
              <span className="text-text">{path === "*" ? "the whole object" : path}</span>
              <button type="button" className={`${link} ml-auto`} onClick={() => setCtlOpen((o) => !o)} aria-expanded={ctlOpen} data-e2e="cparam-controls-toggle">
                {ctlOpen ? "collapse controls" : "expand controls"}
              </button>
            </div>
            <div className="mono flex flex-wrap items-center gap-2 text-[10px]">
              <span className="text-faint">select</span>
              <button type="button" className={link} onClick={() => setPath("*")}>
                whole cartridge
              </button>
              {layer !== null && (
                <button type="button" className={link} onClick={() => setPath(`L${layer}`)}>
                  layer {layer}
                </button>
              )}
              {t && (
                <button type="button" className={link} onClick={() => setPath(t.name)}>
                  whole tensor
                </button>
              )}
              <input value={typed} onChange={(e) => setTyped(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && typed.trim()) setPath(typed.trim()); }} placeholder="or a path: L3.o.gain[0:16], down[2:4, 0:8]" className={`${field} w-72`} aria-label="Selection path" data-e2e="cparam-path" />
            </div>

            <SelectionState k={k} path={path} d={d} maps={maps} onShow={showEntry} />
            <OverlayPanel k={k} path={path} ov={d.overlay} onShow={showEntry} pulse={overlayPulse} />
            {d.overlay?.editable || editable ? (
              <EditPanel k={k} path={path} maps={maps} onDone={onOpen} canVersion={editable} why={d.editable?.why ?? null} />
            ) : (
              <p className="mono text-[10.5px] text-faint" data-e2e="cparam-readonly">
                read only: {d.editable?.why}
              </p>
            )}
            <ExportPanel k={k} path={path} />
          </div>
          <div className="grid gap-1.5" data-e2e="cparam-focus-layers">
            <div className="mono flex justify-between text-[9.5px] uppercase tracking-[0.12em] text-quiet">
              <span>change per layer · click a layer to narrow</span>
              <span>{layer === null ? `${layers.length} layers` : `layer ${layer}`}</span>
            </div>
            {layers.length > 0 ? (
              <NormStrip layers={layers} total={width} picked={layer === null ? undefined : new Set([layer])} onLayer={(l) => { setLayer(layer === l ? null : l); setTensor(null); setPath(layer === l ? "*" : `L${l}`); }} />
            ) : (
              <span className="mono text-[10px] text-faint">no layers: it acts on {tensors[0]?.target ?? "one node"}</span>
            )}
            {(d.overlay?.layers?.length ?? 0) > 0 && <OverlayStrip marks={d.overlay?.layers ?? []} total={width} />}
            <div className="grid gap-1" data-e2e="cparam-maps">
              {m.data && maps.length === 0 && <span className="mono text-[9.5px] text-quiet">no parameter map of this MicroModel relates to it</span>}
              {maps.map((mp) => {
                const by = new Map(mp.layers.map((x) => [x.layer, x.value ?? 0]));
                const mx = Math.max(1e-12, ...mp.layers.map((x) => Math.abs(x.value ?? 0)));
                return (
                  <div key={mp.file} className="grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)] items-center gap-2" data-e2e="cparam-map" data-file={mp.file}>
                    <span className="mono truncate text-[9.5px] text-faint" title={`${mp.file}: ${mp.why.join("; ")}${mp.made_on ? `; measured on ${mp.made_on}` : ""}${mp.used_by.length ? `; used by ${mp.used_by.join(", ")}` : ""}`}>
                      {mp.file}
                    </span>
                    {mp.layers.length > 0 ? (
                      <span className="flex h-3 items-end gap-px">
                        {Array.from({ length: width }, (_, i) => (
                          <i key={i} className="block min-w-0 flex-1 bg-attention/70" style={{ height: by.has(i) ? `${20 + (80 * Math.abs(by.get(i) ?? 0)) / mx}%` : "0%" }} title={by.has(i) ? `layer ${i}: ${fmt(by.get(i))}` : undefined} />
                        ))}
                      </span>
                    ) : (
                      <span className="mono text-[9.5px] text-quiet">{mp.why.join("; ")}</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="flex flex-wrap gap-1.5" data-e2e="cparam-focus-tensors">
            {shown.map((x) => (
              <button key={x.name} type="button" onClick={() => { setTensor(tensor === x.name ? null : x.name); setPath(tensor === x.name ? (layer === null ? "*" : `L${layer}`) : x.name); }} aria-pressed={tensor === x.name} className={`press mono grid border px-2 py-1 text-left text-[10px] ${tensor === x.name ? "border-sense-pale text-text" : "border-line text-muted hover:text-text"}`} data-e2e="cparam-focus-tensor" data-name={x.name}>
                <span className="truncate">{x.name}</span>
                <span className="text-faint">
                  {x.shape.join("x")} · L2 {fmt(x.l2)}
                  {ovMark(d.overlay, x.name)}
                </span>
              </button>
            ))}
          </div>

          {t && (
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,20rem)]" data-e2e="cparam-drill">
              <Grid key={t.name} k={k} t={t} onPick={(p) => setPath(p)} />
              <Selection k={k} path={path} />
            </div>
          )}

        </>
      )}
    </section>
  );
}
