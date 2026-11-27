"use client";

// 2026-11-26 (audit A1): what each MicroModel is made of, read only: its tensors by name with
// element type, shape and size, and the parameter clusters the engine keeps over them
// (`GET /api/models/weights`, the same reading MCP `nm_weights` gives). Nothing here writes a
// weight file; the reading is asked for only when it is opened.
//
// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel."): the reading is a piece the
// Models page puts in its side panel when a MicroModel's "Inspect tensors" is pressed
// (`WeightsReading`, always open); `WeightsDrawer` is the same reading behind a fold, and the
// default export keeps the old list of folds for any page that wants all four at once.

import { useState } from "react";
import { Empty, Placeholder } from "@/components/ui";
import { bytes, useApi } from "@/lib/api";

type Tensor = { name: string; dtype: string; shape: number[]; bytes: number; file: string };
type Weights = { model?: string; bound?: boolean; limit?: number; tensor_list?: Tensor[]; error?: string; [k: string]: unknown };

const LIMIT = 400;

/** The scalar figures of the reading (counts, sizes, names), shown as key and value. */
function figures(w: Weights): [string, string][] {
  const out: [string, string][] = [];
  for (const [k, v] of Object.entries(w)) {
    if (k === "tensor_list" || k === "error" || k === "limit") continue;
    if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") out.push([k.replace(/_/g, " "), String(v)]);
    else if (Array.isArray(v)) out.push([k.replace(/_/g, " "), `${v.length} listed`]);
  }
  return out;
}

/** One model's tensors and clusters, read when this mounts (read only). */
export function WeightsReading({ model }: { model: string }) {
  const [filter, setFilter] = useState("");
  const q = useApi<Weights>(`/api/models/weights?model=${encodeURIComponent(model)}&limit=${LIMIT}`, 60000);
  const w = q.data;
  const list = (w?.tensor_list ?? []).filter((t) => !filter || t.name.toLowerCase().includes(filter.toLowerCase()));
  return (
    <div className="grid min-w-0 gap-3" data-e2e="weights-drawer" data-model={model}>
      {!w && q.error ? (
        <Empty title="The weights did not answer">{q.error}</Empty>
      ) : !w ? (
        <Placeholder className="h-20" what="weights" />
      ) : w.error ? (
        <Empty title="Not readable">{w.error}</Empty>
      ) : (
        <>
          <div
            className="mono text-[10px] text-quiet"
            title="Read only. The base cartridges are cut from these tensors; the files themselves are never written."
          >
            read only · {w.bound ? "as this server holds it" : "from the files on disk (not held by this server)"}
          </div>
          <dl className="mono grid grid-cols-2 gap-x-4 gap-y-0.5 text-[10.5px]">
            {figures(w).map(([k, v]) => (
              <div key={k} className="flex min-w-0 justify-between gap-2 border-b border-line/40 py-1">
                <dt className="truncate text-faint">{k}</dt>
                <dd className="truncate text-muted" title={v}>
                  {v}
                </dd>
              </div>
            ))}
          </dl>
          <label className="relative block">
            <span className="sr-only">Filter tensors</span>
            <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="filter: layers.12, attn, mlp" className="mono h-8 w-full border-b border-line-strong bg-transparent text-[11px] text-text outline-none placeholder:text-faint focus-visible:border-sense-pale" />
          </label>
          <div className="max-h-[50vh] overflow-auto">
            <table className="mono w-full table-fixed text-left text-[10.5px]">
              <colgroup>
                <col />
                <col className="w-12" />
                <col className="w-24" />
                <col className="w-16" />
              </colgroup>
              <thead className="sticky top-0 bg-bg-1 text-faint">
                <tr>
                  <th className="py-1 pr-2 font-normal">tensor</th>
                  <th className="py-1 pr-2 font-normal">type</th>
                  <th className="py-1 pr-2 font-normal">shape</th>
                  <th className="py-1 text-right font-normal">size</th>
                </tr>
              </thead>
              <tbody>
                {list.map((t) => (
                  <tr key={t.name} className="border-t border-line/40">
                    <td className="truncate py-1 pr-2 text-muted" title={`${t.name} (${t.file})`}>
                      {t.name}
                    </td>
                    <td className="truncate py-1 pr-2 text-faint">{t.dtype}</td>
                    <td className="truncate py-1 pr-2 text-faint" title={t.shape.join(" × ")}>
                      {t.shape.join("×")}
                    </td>
                    <td className="py-1 text-right text-faint">{bytes(t.bytes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <span className="mono text-[10.5px] text-quiet">
            {list.length} shown{(w.tensor_list ?? []).length >= LIMIT ? ` · the first ${LIMIT} by name` : ""}
          </span>
        </>
      )}
    </div>
  );
}

/** The reading behind a fold; nothing is asked until it is opened. */
export function WeightsDrawer({ model, label, initiallyOpen = false }: { model: string; label?: string; initiallyOpen?: boolean }) {
  const [open, setOpen] = useState(initiallyOpen);
  return (
    <div className="min-w-0 border-b border-line/60">
      <button type="button" aria-expanded={open} onClick={() => setOpen((o) => !o)} className="press mono flex min-h-9 w-full items-center justify-between gap-3 text-left text-[10.5px] uppercase tracking-[0.12em] text-muted hover:text-text">
        <span className="truncate">{label ?? `Inspect tensors & clusters · ${model}`}</span>
        <span className="text-faint">{open ? "close" : "open"}</span>
      </button>
      {open && (
        <div className="pb-4">
          <WeightsReading model={model} />
        </div>
      )}
    </div>
  );
}

export default function WeightsDrawers({ models }: { models: string[] }) {
  return (
    <section aria-label="Tensors and clusters" className="grid min-w-0">
      <h2 className="eyebrow border-b border-line pb-2 text-muted">What each MicroModel is made of</h2>
      {models.map((m) => (
        <WeightsDrawer key={m} model={m} />
      ))}
    </section>
  );
}
