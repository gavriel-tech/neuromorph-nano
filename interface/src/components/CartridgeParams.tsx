"use client";

// Cartridge parameters on the Parameters page: the parameter domain covers every object that holds
// learned parameters, whether or not it is a cartridge (socket cartridges, job filters, project
// adapters, learning units, preference counts, base cartridges, open weights). The middle lists them
// per MicroModel by label alone (key, kind, state, layer range, size on disk): no tensor, value or norm
// is read until an object is opened. Picking one opens it in the right rail: its tensors with shape, norms and
// a histogram, a per-layer norm strip with a switch per layer, a strength, and direct values for
// small tensors. Apply makes a new version beside the source (on trial, plugged out, the next adapter
// version or a candidate unit); the source is never written, and Audit undoes the edit. The same
// reads and edit are MCP nm_cartridges parameters and parameters_edit, and GET and POST
// /api/cartridges/parameters. Picking a row focuses the page on it (ParamFocus), as the link
// `/parameters/?cartridge=<key>` from Cartridges does.

import { useCallback, useEffect, useMemo, useState } from "react";
import { NoteLine } from "@/components/Controls";
import { roleTitle } from "@/components/MicroModel";
import { useRail } from "@/components/Workspace";
import { Empty, Pill, Placeholder, Switch } from "@/components/ui";
import { bytes, post, refreshAll, useApi } from "@/lib/api";
import { maskPath, maskPathsIn } from "@/lib/privacy";

type Note = { text: string; tone: "verified" | "error" };
export type ParamLayer = { layer: number; tensors: number; read?: number; effect_l2: number | null; l2?: number | null };
export type ParamObject = {
  key: string;
  source: string;
  kind: string;
  model: string;
  role: string;
  name: string;
  state: string;
  editable: boolean;
  layer_range: [number, number] | null;
  bytes: number;
  read: boolean;
  overlay: { edits: number; masks: number; freezes: number } | null;
};
type ParamList = { format?: string; objects?: ParamObject[] };
type Hist = { lo: number | null; hi: number | null; counts: number[] };
type TensorRow = {
  name: string;
  target: string;
  layer: number | null;
  part: string;
  shape: number[];
  dtype: string;
  size: number;
  bytes: number;
  identity: number | null;
  scales: boolean;
  read: boolean;
  paged: boolean;
  values_editable: boolean;
  l2: number | null;
  max_abs: number | null;
  effect_l2: number | null;
  summary: { min: number | null; max: number | null; mean: number | null; histogram: Hist } | null;
  unread?: string | null;
};
type Section = {
  format?: string;
  key?: string;
  source?: string;
  kind?: string;
  model?: string;
  role?: string;
  name?: string;
  state?: string;
  path?: string;
  root?: string;
  holds?: boolean;
  note?: string;
  editable?: { scale: boolean; layers: boolean; values: boolean; why: string | null; layer_off: string; scale_max: number | null; small: number };
  totals?: { tensors: number; read: number; values: number; l2: number | null };
  layers?: ParamLayer[];
  tensors?: TensorRow[];
  held_out?: { runs_here: boolean; needs: string; how: string } | null;
  overlay?: { editable?: boolean; why?: string | null; counts?: { edits: number; masks: number; freezes: number }; takes_effect?: { when: string; reason: string } } | null;
  error?: string;
};
type Page = { values?: (number | null)[]; offset?: number; size?: number; error?: string };
type EditReply = { changed?: boolean; key?: string; audit_id?: string | null; state?: string; compat?: { fits?: string[]; not?: Record<string, string>; shapes?: string; base_compatible?: boolean | null } | null };

const ROLE_ORDER = ["reader", "writer", "embedder", "reranker"];
export const KIND_WORD: Record<string, string> = { filter: "filter cartridge", adapter: "adapter cartridge", lens: "lens adapter", readout: "read-out adapter", unit: "learning unit", preference: "preference cartridge", base: "base cartridge", open: "open weights (upstream)" };
const KIND_BG: Record<string, string> = { filter: "bg-signal", adapter: "bg-learned", lens: "bg-learned", readout: "bg-learned", unit: "bg-sense-pale", preference: "bg-attention", base: "bg-verified", open: "bg-quiet" };
const PAGE = 64;

const link = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const go = "press mono min-h-8 border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40";

/** A figure in three significant digits, or in exponent form when very large or small. */
export function fmt(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "-";
  const a = Math.abs(x);
  if (a === 0) return "0";
  return a >= 1000 || a < 0.001 ? x.toExponential(2) : String(Number(x.toPrecision(3)));
}

/** The key a link names: `?cartridge=<key>` or `#cartridge=<key>`. */
export function linkedKey(): string | null {
  if (typeof window === "undefined") return null;
  const h = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const q = new URLSearchParams(window.location.search);
  return q.get("cartridge") ?? h.get("cartridge");
}

/** Bars per layer: the L2 norm of what an object changes at each layer, against the MicroModel's
 *  layers (`total`); a base cartridge shades the layer range its label names. */
export function NormStrip({ layers, total, range, picked, onLayer }: { layers: ParamLayer[]; total: number; range?: [number, number] | null; picked?: Set<number>; onLayer?: (l: number) => void }) {
  const by = new Map(layers.map((l) => [l.layer, l]));
  const max = Math.max(1e-12, ...layers.map((l) => l.effect_l2 ?? 0));
  return (
    <div className="flex h-7 w-full items-end gap-px" role="group" aria-label="Layers: the L2 norm of what the parameters change at each">
      {Array.from({ length: total }, (_, i) => {
        const l = by.get(i);
        const inRange = range ? i >= range[0] && i <= range[1] : false;
        const h = l ? 12 + 88 * ((l.effect_l2 ?? 0) / max) : 0;
        const off = picked?.has(i) ?? false;
        const body = (
          <>
            {l && <i aria-hidden className={`block w-full ${off ? "bg-error/50" : "bg-sense-pale/80"}`} style={{ height: `${h}%` }} />}
            {!l && inRange && <i aria-hidden className="block h-1/3 w-full bg-verified/30" />}
          </>
        );
        const title = l ? `layer ${i}: ${l.tensors} tensors, change L2 ${fmt(l.effect_l2)}${off ? " (switched off in this edit)" : ""}` : inRange ? `layer ${i}: in the base cartridge's range` : `layer ${i}: none`;
        return onLayer && l ? (
          <button key={i} type="button" onClick={() => onLayer(i)} title={title} aria-pressed={off} className="press flex h-full min-w-0 flex-1 flex-col-reverse bg-line/30 hover:bg-line/70" data-e2e="cparam-layer" data-layer={i}>
            {body}
          </button>
        ) : (
          <span key={i} title={title} className="flex h-full min-w-0 flex-1 flex-col-reverse bg-line/30">
            {body}
          </span>
        );
      })}
    </div>
  );
}

/** A histogram of a tensor's values in a few bins. */
function Spark({ h }: { h: Hist | undefined }) {
  if (!h) return <span className="text-quiet">-</span>;
  const max = Math.max(1, ...h.counts);
  return (
    <span className="inline-flex h-3 w-12 items-end gap-px" title={`${fmt(h.lo)} to ${fmt(h.hi)}: ${h.counts.join(" ")}`}>
      {h.counts.map((c, i) => (
        <i key={i} className="block w-full bg-muted/70" style={{ height: `${Math.max(c ? 12 : 0, (100 * c) / max)}%` }} />
      ))}
    </span>
  );
}

/** One page of a tensor's values; small tensors' values can be set here. */
function Values({ k, t, draft, setDraft, onPage }: { k: string; t: TensorRow; draft: Record<number, string>; setDraft: (d: Record<number, string>) => void; onPage: (offset: number, values: (number | null)[]) => void }) {
  const [offset, setOffsetState] = useState(0);
  const p = useApi<Page>(t.paged ? `/api/cartridges/parameters?key=${encodeURIComponent(k)}&tensor=${encodeURIComponent(t.name)}&offset=${offset}&limit=${PAGE}` : null);
  const vals = useMemo(() => p.data?.values ?? [], [p.data]);
  useEffect(() => {
    if (p.data?.values) onPage(p.data.offset ?? 0, p.data.values);
  }, [p.data, onPage]);
  // Values set by hand are kept for the page shown; another page starts clean.
  const setOffset = (o: number) => {
    setDraft({});
    setOffsetState(o);
  };
  if (!t.paged) return <p className="mono text-[10.5px] text-faint">{t.unread ?? "its values are not held here"}</p>;
  return (
    <div className="grid gap-2" data-e2e="cparam-values">
      <div className="mono flex items-center justify-between text-[10px] text-faint">
        <span>
          {t.name} · values {offset} to {Math.min(t.size, offset + PAGE) - 1} of {t.size}
        </span>
        <span className="flex gap-3">
          <button type="button" className={link} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
            prev
          </button>
          <button type="button" className={link} disabled={offset + PAGE >= t.size} onClick={() => setOffset(offset + PAGE)}>
            next
          </button>
        </span>
      </div>
      {p.data?.error ? (
        <p className="mono text-[10.5px] text-error">{p.data.error}</p>
      ) : !p.data ? (
        <Placeholder className="h-16" what="parameters" />
      ) : (
        <div className="grid grid-cols-4 gap-x-2 gap-y-1">
          {vals.map((v, i) => {
            const at = offset + i;
            return t.values_editable ? (
              <input key={at} value={draft[at] ?? fmt(v)} onChange={(e) => setDraft({ ...draft, [at]: e.target.value })} aria-label={`${t.name} value ${at}`} title={`${at}`} className={`mono h-6 min-w-0 border-b bg-transparent px-0.5 text-[10px] outline-none focus-visible:border-sense-pale ${draft[at] !== undefined ? "border-sense-pale text-text" : "border-line text-muted"}`} />
            ) : (
              <span key={at} className="mono truncate text-[10px] text-muted" title={`${at}: ${v}`}>
                {fmt(v)}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

/** One object's parameters, opened in the rail: tensors, the per-layer norm strip with switches, a
 *  strength, direct values, and Apply (a new version). */
export function ParamDetail({ k, total }: { k: string; total: number | null }) {
  const path = `/api/cartridges/parameters?key=${encodeURIComponent(k)}`;
  const q = useApi<Section>(path);
  const [scale, setScale] = useState(1);
  const [off, setOff] = useState<Set<number>>(new Set());
  const [picked, setPicked] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, Record<number, string>>>({});
  const [pages, setPages] = useState<Record<string, { offset: number; values: (number | null)[] }>>({});
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<Note | null>(null);
  const d = q.data;
  const ed = d?.editable;
  const tensors = d?.tensors ?? [];
  const layers = d?.layers ?? [];
  const width = total ?? Math.max(1, ...layers.map((l) => l.layer + 1));
  const pickedRow = tensors.find((t) => t.name === picked) ?? null;

  const values = useMemo(() => {
    const out: Record<string, { offset: number; values: number[] }> = {};
    for (const [name, dr] of Object.entries(drafts)) {
      const idx = Object.keys(dr).map(Number).sort((a, b) => a - b);
      const pg = pages[name];
      if (!idx.length || !pg) continue;
      // One range from the first value set to the last, the values between taken from the page.
      const run: number[] = [];
      for (let i = idx[0]; i <= idx[idx.length - 1]; i++) run.push(dr[i] !== undefined ? Number(dr[i]) : Number(pg.values[i - pg.offset]));
      if (run.some((x) => !Number.isFinite(x))) continue;
      out[name] = { offset: idx[0], values: run };
    }
    return out;
  }, [drafts, pages]);
  const onPage = useCallback((name: string, offset: number, vals: (number | null)[]) => setPages((x) => (x[name]?.offset === offset && x[name]?.values === vals ? x : { ...x, [name]: { offset, values: vals } })), []);
  const changes = (scale !== 1 ? 1 : 0) + off.size + Object.keys(values).length;

  const apply = async () => {
    setBusy(true);
    try {
      const body: Record<string, unknown> = { key: k };
      if (scale !== 1) body.scale = scale;
      if (off.size) body.layers = Object.fromEntries([...off].map((l) => [String(l), "off"]));
      if (Object.keys(values).length) body.values = values;
      const r = await post<EditReply>("/api/cartridges/parameters", body);
      if (r.changed === false) {
        setNote({ text: "Nothing changed: the edit gives the same parameters.", tone: "verified" });
      } else {
        const fits = r.compat?.fits ? ` Fits ${r.compat.fits.length ? r.compat.fits.join(", ") : "no installed MicroModel"}.` : r.compat?.shapes ? ` Shapes ${r.compat.shapes}.` : "";
        setNote({ text: `New version ${r.key ?? ""}: ${r.state ?? ""}.${fits} Audit ${r.audit_id ?? ""} sets it aside.`, tone: "verified" });
        setScale(1);
        setOff(new Set());
        setDrafts({});
      }
      for (const p of ["/api/cartridges", "/api/parameter-maps", "/api/surgery", "/api/audit", "/api/units", "/api/lenses"]) void refreshAll(p);
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  if (!d && q.error) return <Empty title="The parameters did not answer">{q.error}</Empty>;
  if (!d) return <Placeholder className="h-40" what="parameters" />;
  if (d.error) return <Empty title="Not readable">{maskPathsIn(d.error, d.root)}</Empty>;
  return (
    <div className="grid gap-4" data-e2e="cparam-detail" data-key={k}>
      <div className="mono flex flex-wrap items-center gap-2 text-[10.5px] text-faint">
        <Pill tone="muted">{KIND_WORD[d.kind ?? ""] ?? d.kind}</Pill>
        <span>{roleTitle(d.role ?? "")}</span>
        <span className="truncate" title={d.model}>{d.model}</span>
        <Pill tone={d.state === "trial" || d.state === "plugged out" ? "attention" : "verified"}>{d.state}</Pill>
      </div>
      {d.note && <p className="read text-[11px] leading-relaxed text-faint">{maskPathsIn(d.note, d.root)}.</p>}
      <NoteLine note={note} />
      {!d.holds ? (
        <Empty title="This cartridge holds no parameters">{ed?.why ?? undefined}</Empty>
      ) : (
        <>
          <dl className="mono grid grid-cols-3 gap-2 text-[10.5px]">
            <div>
              <dt className="text-faint">tensors</dt>
              <dd className="text-text">
                {d.totals?.tensors}
                {d.totals && d.totals.read !== d.totals.tensors ? ` (${d.totals.read} read)` : ""}
              </dd>
            </div>
            <div>
              <dt className="text-faint">values</dt>
              <dd className="text-text">{d.totals?.values.toLocaleString()}</dd>
            </div>
            <div>
              <dt className="text-faint">L2</dt>
              <dd className="text-text">{fmt(d.totals?.l2)}</dd>
            </div>
          </dl>

          {/* What the object's parameter overlay holds (its entries, and mask, freeze and undo, are in
              the controls under the viewer). */}
          {d.overlay && (
            <div className="mono grid gap-0.5 border-l-2 border-line-strong pl-2 text-[10px] text-faint" data-e2e="cparam-overlay-summary" title="The parameter overlay is held beside the object, never in its files, and applied when the engine seats it">
              <span>
                <span className="uppercase tracking-[0.12em] text-quiet">overlay</span>{" "}
                {(d.overlay.counts?.edits ?? 0) + (d.overlay.counts?.masks ?? 0) + (d.overlay.counts?.freezes ?? 0) > 0 ? (
                  <span className="text-attention">
                    {d.overlay.counts?.edits ?? 0} edited · {d.overlay.counts?.masks ?? 0} masked · {d.overlay.counts?.freezes ?? 0} frozen
                  </span>
                ) : (
                  "no entries: it serves as its files hold it"
                )}
              </span>
              {d.overlay.takes_effect?.when && <span>{d.overlay.takes_effect.when}</span>}
              {d.overlay.editable === false && d.overlay.why && <span>{d.overlay.why}</span>}
            </div>
          )}

          {layers.length > 0 && (
            <div className="grid gap-1" data-e2e="cparam-layers">
              <div className="mono flex justify-between text-[9.5px] uppercase tracking-[0.12em] text-quiet">
                <span>change per layer{ed?.layers ? " · click a layer to switch it off" : ""}</span>
                <span>{layers.length} layers</span>
              </div>
              <NormStrip layers={layers} total={width} picked={off} onLayer={ed?.layers ? (l) => setOff((s) => { const n = new Set(s); if (n.has(l)) n.delete(l); else n.add(l); return n; }) : undefined} />
              {ed?.layers && (
                <div className="flex flex-wrap gap-x-3 gap-y-1">
                  {layers.map((l) => (
                    <span key={l.layer} className="mono inline-flex items-center gap-1 text-[10px] text-faint">
                      <Switch on={!off.has(l.layer)} label={`Layer ${l.layer} ${off.has(l.layer) ? "on" : "off"}`} onChange={(on) => setOff((s) => { const n = new Set(s); if (on) n.delete(l.layer); else n.add(l.layer); return n; })} />
                      L{l.layer}
                    </span>
                  ))}
                </div>
              )}
              {ed?.layers && <span className="mono text-[9.5px] text-quiet">off {ed.layer_off}</span>}
            </div>
          )}

          {ed?.scale && (
            <label className="grid gap-1" data-e2e="cparam-strength">
              <span className="mono flex justify-between text-[9.5px] uppercase tracking-[0.12em] text-quiet">
                <span>strength</span>
                <span className="text-text">{scale.toFixed(2)}</span>
              </span>
              <input type="range" min={0} max={ed.scale_max ?? 4} step={0.05} value={scale} onChange={(e) => setScale(Number(e.target.value))} aria-label="Strength" className="w-full accent-[var(--sense-pale)]" />
            </label>
          )}

          <div className="grid max-h-72 gap-0 overflow-auto" data-e2e="cparam-tensors">
            <table className="mono w-full text-left text-[10px]">
              <thead className="sticky top-0 bg-bg-1 text-faint">
                <tr>
                  <th className="py-1 pr-2 font-normal">tensor</th>
                  <th className="py-1 pr-2 font-normal">shape</th>
                  <th className="py-1 pr-2 font-normal">L2</th>
                  <th className="py-1 pr-2 font-normal">max</th>
                  <th className="py-1 font-normal">values</th>
                </tr>
              </thead>
              <tbody>
                {tensors.map((t) => (
                  <tr key={t.name} className={`cursor-pointer border-t border-line/40 hover:bg-bg-2 ${picked === t.name ? "bg-bg-2" : ""}`} onClick={() => setPicked(picked === t.name ? null : t.name)} data-e2e="cparam-tensor" title={`${t.target} · ${t.dtype} · ${t.size} values${t.unread ? ` · ${t.unread}` : ""}`}>
                    <td className="max-w-[9rem] truncate py-1 pr-2 text-muted">{t.name}</td>
                    <td className="py-1 pr-2 text-faint">{t.shape.join("x")}</td>
                    <td className="py-1 pr-2 text-text">{fmt(t.l2)}</td>
                    <td className="py-1 pr-2 text-faint">{fmt(t.max_abs)}</td>
                    <td className="py-1">
                      <Spark h={t.summary?.histogram} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {pickedRow && <Values key={pickedRow.name} k={k} t={pickedRow} draft={drafts[pickedRow.name] ?? {}} setDraft={(dr) => setDrafts((x) => ({ ...x, [pickedRow.name]: dr }))} onPage={(o, v) => onPage(pickedRow.name, o, v)} />}

          {d.held_out && (
            <p className="mono text-[10px] text-faint" title={d.held_out.how} data-e2e="cparam-heldout">
              held-out check: needs {d.held_out.needs}; not run here
            </p>
          )}
          {ed?.why ? (
            <p className="mono text-[10.5px] text-faint" data-e2e="cparam-readonly">
              read only: {ed.why}
            </p>
          ) : (
            <div className="flex flex-wrap items-center gap-4">
              <button type="button" disabled={busy || changes === 0} onClick={() => void apply()} className={go} data-e2e="cparam-apply">
                {busy ? "applying" : changes ? `apply ${changes} ${changes === 1 ? "change" : "changes"} as a new version` : "apply"}
              </button>
              {changes > 0 && (
                <button type="button" className={link} onClick={() => { setScale(1); setOff(new Set()); setDrafts({}); }}>
                  reset
                </button>
              )}
            </div>
          )}
          {d.path && <p className="mono break-all text-[9.5px] text-quiet" title="the object's place, relative to the Nano folder where it lies inside it">{maskPath(d.path, d.root)}</p>}
        </>
      )}
    </div>
  );
}

const PICKER_GROUPS: { id: string; label: string; match: (o: ParamObject) => boolean }[] = [
  { id: "job-filter", label: "Job filters", match: (o) => o.source === "job-filter" },
  { id: "socket", label: "Compensation & socket cartridges", match: (o) => o.source === "library" && (o.kind === "filter" || o.kind === "adapter") },
  { id: "learned", label: "Adapters, lenses & learning units", match: (o) => o.kind === "lens" || o.kind === "readout" || o.kind === "unit" || o.source === "adapter" },
  { id: "preference", label: "Preference cartridges", match: (o) => o.kind === "preference" },
  { id: "base", label: "Base cartridges", match: (o) => o.kind === "base" },
  { id: "open", label: "Open weights (optional)", match: (o) => o.kind === "open" },
];

/** Universal picker for any parameter object on this machine (cartridges, job filters, adapters,
 *  units, preference counts, base cartridges, and optional open weights). */
export function ParamObjectPicker({
  value = "",
  onPick,
  placeholder = "open any cartridge or parameter object...",
  className = "",
  ariaLabel = "Select parameter object",
}: {
  value?: string;
  onPick: (key: string) => void;
  placeholder?: string;
  className?: string;
  ariaLabel?: string;
}) {
  const q = useApi<ParamList>("/api/cartridges/parameters");
  const objects = useMemo(() => q.data?.objects ?? [], [q.data]);
  const grouped = useMemo(() => {
    const used = new Set<string>();
    const out: { id: string; label: string; items: ParamObject[] }[] = [];
    for (const g of PICKER_GROUPS) {
      const items = objects.filter((o) => !used.has(o.key) && g.match(o));
      for (const o of items) used.add(o.key);
      if (items.length > 0) out.push({ id: g.id, label: g.label, items });
    }
    const rest = objects.filter((o) => !used.has(o.key));
    if (rest.length > 0) out.push({ id: "other", label: "Other objects", items: rest });
    return out;
  }, [objects]);
  const known = value !== "" && objects.some((o) => o.key === value);
  return (
    <select
      value={value}
      onChange={(e) => {
        if (e.target.value) onPick(e.target.value);
      }}
      aria-label={ariaLabel}
      data-e2e="param-object-picker"
      className={`mono h-7 min-w-[16rem] max-w-full border border-line-strong bg-bg-2 px-2 text-[10.5px] text-text outline-none focus-visible:border-sense-pale ${className}`}
    >
      {!value && (
        <option value="" className="bg-bg-1 text-faint">
          {q.data ? `${placeholder} (${objects.length})` : "loading parameter objects..."}
        </option>
      )}
      {value && !known && (
        <option value={value} className="bg-bg-1 text-text">
          {value}
        </option>
      )}
      {grouped.map((g) => (
        <optgroup key={g.id} label={g.label} className="bg-bg-1 text-faint">
          {g.items.map((o) => {
            const tag = o.source === "job-filter" ? "job filter" : (KIND_WORD[o.kind] ?? o.kind);
            return (
              <option key={o.key} value={o.key} className="bg-bg-1 text-text">
                {o.key} [{tag}] · {roleTitle(o.role || "") || o.model} ({o.state})
              </option>
            );
          })}
        </optgroup>
      ))}
    </select>
  );
}

/** Every object holding parameters, per MicroModel, against the layer strip; one picked opens in the
 *  rail. */
export default function CartridgeParams({ layerCount, onFocus, role: onlyRole = "all", text = "" }: { layerCount: (model: string) => number | null; onFocus?: (key: string) => void; /** The page filter: one MicroModel by its role, or "all"; and the words in its find field. */ role?: string; text?: string }) {
  // Labels only, read once; refreshed on demand (after an edit, or the refresh control).
  const q = useApi<ParamList>("/api/cartridges/parameters");
  const rail = useRail();
  const every = useMemo(() => q.data?.objects ?? [], [q.data]);
  const objects = useMemo(() => {
    const words = text.trim().toLowerCase();
    return every.filter((o) => (onlyRole === "all" || o.role === onlyRole) && (!words || `${o.key} ${o.name} ${KIND_WORD[o.kind] ?? o.kind} ${o.state}`.toLowerCase().includes(words)));
  }, [every, onlyRole, text]);
  const selected = rail.shown?.key?.startsWith("cparam:") ? rail.shown.key.slice(7) : null;
  const open = useCallback(
    (key: string) => {
      if (onFocus) {
        onFocus(key);
        return;
      }
      const o = objects.find((x) => x.key === key);
      const total = o ? layerCount(o.model) : null;
      rail.show({
        key: `cparam:${key}`,
        title: (
          <span className="flex min-w-0 items-center gap-2" title={key}>
            <i aria-hidden className={`inline-block h-2.5 w-2.5 shrink-0 ${KIND_BG[o?.kind ?? ""] ?? "bg-quiet"}`} />
            <span className="truncate">{key}</span>
          </span>
        ),
        body: <ParamDetail key={key} k={key} total={total} />,
      });
    },
    [objects, layerCount, rail, onFocus],
  );

  const roles = useMemo(() => {
    const seen = [...new Set(objects.map((o) => o.role))];
    return seen.sort((a, b) => (ROLE_ORDER.indexOf(a) + 1 || 99) - (ROLE_ORDER.indexOf(b) + 1 || 99) || a.localeCompare(b));
  }, [objects]);

  return (
    <section className="grid min-w-0 gap-4 border-t border-line pt-4" data-e2e="cparam-lane" aria-label="Cartridge parameters">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="mono text-[13px] uppercase tracking-[0.16em] text-text" title="Every object that holds learned parameters, cartridge or not, per MicroModel, by label. A shaded strip is the layer range a label records. Open one to read its tensors and values, edit them in its overlay or into a new version.">
          cartridge parameters
        </span>
        <span className="mono flex items-baseline gap-3 text-[10px] text-faint">
          {q.data ? `${objects.length === every.length ? `${every.length} objects` : `${objects.length} of ${every.length} objects`} · labels only, parameters read when opened` : ""}
          <button type="button" className={link} onClick={() => void refreshAll("/api/cartridges/parameters")} data-e2e="cparam-refresh">
            refresh
          </button>
        </span>
      </div>
      {q.error && !q.data && <Empty title="The parameters did not answer">{q.error}</Empty>}
      {!q.data && !q.error && <Placeholder className="h-24" what="parameters" />}
      {q.data && every.length === 0 && <Empty title="No object holds parameters yet" />}
      {q.data && every.length > 0 && objects.length === 0 && <div className="mono py-4 text-center text-[11px] text-quiet">nothing matches the filter above</div>}
      {roles.map((role) => {
        const rows = objects.filter((o) => o.role === role);
        const chain = chainOf(rows);
        return (
          <div key={role} className="grid min-w-0 gap-2 border-t border-line/60 pt-3" data-e2e="cparam-role" data-role={role}>
            <div className="mono flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-[10px] text-faint">
              <span className="text-[12px] uppercase tracking-[0.2em] text-text">{roleTitle(role)}</span>
              <span>
                {rows.length} {rows.length === 1 ? "object" : "objects"} · {bytes(rows.reduce((a, o) => a + o.bytes, 0))}
              </span>
            </div>
            {/* The chain, bottom of the stack first: the open weights, the base cartridge cut from
                them, the filters over it, then what was learned on top. */}
            <ol className="flex min-w-0 flex-wrap items-start gap-x-2 gap-y-3">
              {chain.map((g, gi) => (
                <li key={g.id} className="flex min-w-0 items-start gap-2" data-e2e="cparam-group" data-group={g.id}>
                  {gi > 0 && (
                    <span aria-hidden className="mono mt-8 text-[11px] text-quiet">
                      ›
                    </span>
                  )}
                  <div className="grid min-w-0 content-start gap-1">
                    <span className="mono text-[9px] uppercase tracking-[0.14em] text-quiet">
                      {g.label}
                      {g.stems.length > 1 ? ` ${g.stems.length}` : ""}
                    </span>
                    <div className="flex min-w-0 flex-wrap items-start gap-1.5">
                      {g.stems.map((st) => {
                        const o = st.head;
                        const total = layerCount(o.model) ?? (o.layer_range ? o.layer_range[1] + 1 : 1);
                        const ov = o.overlay ? [o.overlay.edits && `${o.overlay.edits} edited`, o.overlay.masks && `${o.overlay.masks} masked`, o.overlay.freezes && `${o.overlay.freezes} frozen`].filter(Boolean).join(", ") : "";
                        const on = selected === o.key;
                        return (
                          <div key={o.key} className="flex min-w-0 items-start gap-1">
                            <button
                              type="button"
                              onClick={() => (onFocus ? open(o.key) : on ? rail.clear() : open(o.key))}
                              aria-pressed={on}
                              title={`${o.name || o.key} · ${KIND_WORD[o.kind] ?? o.kind} · ${o.state} · ${bytes(o.bytes)}${ov ? ` · overlay: ${ov}` : ""} · click to open in the full Parameter Viewer`}
                              className={`press grid w-[11.5rem] min-w-0 content-start gap-0.5 border border-t-2 px-2 pb-1.5 pt-1 text-left ${KIND_TOP[o.kind] ?? "border-t-quiet"} ${on ? "border-x-sense-pale border-b-sense-pale bg-bg-2" : "border-x-line border-b-line bg-bg-1/40 hover:border-x-line-3 hover:border-b-line-3 hover:bg-bg-2"}`}
                              data-e2e="cparam-row"
                              data-key={o.key}
                            >
                              <span className="mono truncate text-[11px] text-text">{o.key}</span>
                              <span className="mono truncate text-[9.5px] text-faint">
                                {KIND_WORD[o.kind] ?? o.kind} · {o.state}
                              </span>
                              <span className="mono flex items-baseline justify-between gap-2 text-[9.5px] text-faint">
                                <span>{bytes(o.bytes)}</span>
                                {ov && <span className="truncate text-attention">overlay: {ov}</span>}
                              </span>
                              {o.layer_range ? (
                                <span className="mt-1 flex h-1.5 w-full gap-px" role="img" aria-label={`layers ${o.layer_range[0]} to ${o.layer_range[1]} of ${total}`}>
                                  {Array.from({ length: total }, (_, i) => (
                                    <i key={i} className={`block min-w-0 flex-1 ${i >= o.layer_range![0] && i <= o.layer_range![1] ? "bg-sense-pale/70" : "bg-line/60"}`} />
                                  ))}
                                </span>
                              ) : (
                                <span className="mono mt-0.5 truncate text-[9px] text-quiet" data-e2e="cparam-unread">
                                  click to open in viewer
                                </span>
                              )}
                            </button>
                            {st.earlier.length > 0 && (
                              <div className="grid grid-cols-4 content-start gap-1" role="group" aria-label={`Other versions of ${st.stem}`}>
                                {st.earlier.map((e) => (
                                  <button
                                    key={e.key}
                                    type="button"
                                    onClick={() => (onFocus ? open(e.key) : selected === e.key ? rail.clear() : open(e.key))}
                                    aria-pressed={selected === e.key}
                                    title={`${e.key} · ${KIND_WORD[e.kind] ?? e.kind} · ${e.state} · ${bytes(e.bytes)} · click to open in the full Parameter Viewer`}
                                    className={`press mono grid h-7 w-8 place-items-center border text-[9.5px] ${selected === e.key ? "border-sense-pale bg-bg-2 text-text" : "border-line text-faint hover:border-line-3 hover:bg-bg-2 hover:text-text"}`}
                                    data-e2e="cparam-row"
                                    data-key={e.key}
                                  >
                                    {versionOf(e.key) !== null ? `@${versionOf(e.key)}` : "old"}
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        );
      })}
      {q.data && objects.length > 0 && (
        <p className="mono border-t border-line/60 pt-2 text-[9.5px] text-quiet">
          a tile is an object holding parameters: top line its kind, the strip the layers its label records · small squares are its other versions · click one to open it in the viewer
        </p>
      )}
    </section>
  );
}

const KIND_WHAT: Record<string, string> = {
  open: "The upstream open weights a MicroModel is built from. Never altered: an edit, a mask or a freeze is held in the overlay beside them.",
  base: "The base cartridge a MicroModel serves from, cut from its open weights.",
  filter: "A job filter or a compensation: a gain and a bias per channel at each node. An edit makes a new version beside it, plugged out.",
  adapter: "A project adapter, fitted on one project's verified places.",
  lens: "A project lens on the Embedder. Each fit is kept as a version.",
  readout: "A project read-out on the Reranker. Each fit is kept as a version.",
  unit: "A learning unit.",
  preference: "Preference counts, from edits kept or reverted.",
};

/** The key of the cartridge parameters lane, for the side panel: each kind of object that holds
 *  parameters with the colour of its tile's top line, how many there are and what it is. */
export function ParamKindsKey() {
  const q = useApi<ParamList>("/api/cartridges/parameters");
  const objects = q.data?.objects ?? [];
  const kinds = [...new Set(objects.map((o) => o.kind))].sort((a, b) => Object.keys(KIND_WORD).indexOf(a) - Object.keys(KIND_WORD).indexOf(b));
  if (!q.data) return q.error ? <p className="mono text-[10.5px] text-error">{q.error}</p> : <Placeholder className="h-20" what="parameters" />;
  if (!kinds.length) return <p className="mono text-[10.5px] text-quiet">no object holds parameters yet</p>;
  return (
    <ul className="grid gap-1" data-e2e="cparam-kinds">
      {kinds.map((kind) => {
        const os = objects.filter((o) => o.kind === kind);
        return (
          <li key={kind} className="grid gap-1 border-b border-line/40 py-2">
            <span className="flex min-w-0 items-center gap-2">
              <i aria-hidden className={`inline-block h-2 w-2 shrink-0 ${KIND_BG[kind] ?? "bg-quiet"}`} />
              <span className="mono min-w-0 truncate text-[11px] uppercase tracking-[0.12em] text-text">{KIND_WORD[kind] ?? kind}</span>
              <span className="mono ml-auto shrink-0 text-[10px] text-faint">
                {os.length} · {bytes(os.reduce((a, o) => a + o.bytes, 0))}
              </span>
            </span>
            {KIND_WHAT[kind] && <span className="read pl-4 text-[11px] leading-relaxed text-faint">{KIND_WHAT[kind]}</span>}
          </li>
        );
      })}
    </ul>
  );
}

/** The version a key names (`name@3`), or none. */
function versionOf(key: string): number | null {
  const m = /@(\d+)$/.exec(key);
  return m ? Number(m[1]) : null;
}

/** The top line of a tile, in the colour of its kind. */
const KIND_TOP: Record<string, string> = { filter: "border-t-signal", adapter: "border-t-learned", lens: "border-t-learned", readout: "border-t-learned", unit: "border-t-sense-pale", preference: "border-t-attention", base: "border-t-verified", open: "border-t-quiet" };

/** The parts of a MicroModel's parameter chain, bottom of the stack first. */
const CHAIN: { id: string; label: string; kinds: string[] }[] = [
  { id: "open", label: "open weights", kinds: ["open"] },
  { id: "base", label: "base cartridge", kinds: ["base"] },
  { id: "filter", label: "filters", kinds: ["filter"] },
  { id: "learned", label: "adapters and units", kinds: ["adapter", "lens", "readout", "unit", "preference"] },
];

type Stem = { stem: string; head: ParamObject; earlier: ParamObject[] };

/** One MicroModel's objects as its chain: per part, each object once, its other versions (the same
 *  name with another `@n`) set beside the one in use. */
function chainOf(rows: ParamObject[]): { id: string; label: string; stems: Stem[] }[] {
  const known = new Set(CHAIN.flatMap((c) => c.kinds));
  const parts = [...CHAIN, { id: "other", label: "other", kinds: [...new Set(rows.map((o) => o.kind).filter((k) => !known.has(k)))] }];
  return parts
    .map((c) => {
      const by = new Map<string, ParamObject[]>();
      for (const o of rows.filter((x) => c.kinds.includes(x.kind))) {
        const stem = versionOf(o.key) !== null ? o.key.replace(/@\d+$/, "") : o.key;
        by.set(stem, [...(by.get(stem) ?? []), o]);
      }
      const stems = [...by.entries()].map(([stem, os]) => {
        const sorted = [...os].sort((a, b) => (versionOf(a.key) ?? 0) - (versionOf(b.key) ?? 0));
        // The one in use stands for the name (the highest such version); else the newest.
        const live = sorted.filter((o) => /served|plugged in|newest|acting/.test(o.state));
        const head = (live.length ? live : sorted)[(live.length ? live : sorted).length - 1];
        return { stem, head, earlier: sorted.filter((o) => o !== head) };
      });
      return { id: c.id, label: c.label, stems };
    })
    .filter((c) => c.stems.length > 0);
}
