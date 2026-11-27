"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel."): a map picked in the atlas or
// the list opens here, in the right rail. What it is and its state, its lineage drawn as a graph, the
// layers it covers, the held-out split gate figures of a job filter, and its controls: inspect and edit
// (MapInspector), plug in or out, rebuild on other layers, archive, remove. It reads the page context,
// so it follows the data as it polls; every action is the same route as MCP and the API, logged in Audit.

import { useCallback, useRef, useState, type ReactNode } from "react";
import DeleteFlow from "@/components/DeleteFlow";
import { MapInspector } from "@/components/MapInspector";
import MapLineage from "@/components/MapLineage";
import { roleTitle } from "@/components/MicroModel";
import { useRail } from "@/components/Workspace";
import { Pill } from "@/components/ui";
import { ago, bytes } from "@/lib/api";
import { OUTCOME_TONE, STATE_TONE, cleanable, covers, filterOf, kindTone, layersOf, nodeKind, stateOf, useMaps, type MapRow, type RoleMaps } from "@/components/MapCore";

const act = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const delCls = "press mono border-b border-line px-0.5 text-[10px] uppercase tracking-[0.1em] text-faint hover:text-error disabled:opacity-40";

const split = (k: string) => {
  const i = k.indexOf("/");
  return { role: k.slice(0, i), file: k.slice(i + 1) };
};

/** Opens a map ("role/file") in the rail, or closes it (null); `selected` is the one open there. */
export function useSelectMap() {
  const rail = useRail();
  const { roles } = useMaps();
  const n = useRef(0);
  const selected = rail.shown?.key?.startsWith("map:") ? rail.shown.key.slice(4) : null;
  const select = useCallback(
    (key: string | null, inspect = false, lineage = false) => {
      if (key === null) {
        rail.clear();
        return;
      }
      const { role, file } = split(key);
      const kind = roles.find((r) => r.role === role)?.maps.find((m) => m.file === file)?.kind ?? "";
      n.current += 1;
      rail.show({
        key: `map:${key}`,
        title: (
          <span className="flex min-w-0 items-center gap-2" title={file}>
            <i aria-hidden className={`inline-block h-2.5 w-2.5 shrink-0 ${kindTone(kind)}`} />
            <span className="truncate">{file}</span>
          </span>
        ),
        body: <MapDetail key={`${key}:${n.current}`} k={key} inspect={inspect} lineage={lineage} />,
      });
    },
    [rail, roles],
  );
  return { selected, select };
}

/** Two figures of the held-out split gate, before and after the filter, with the change. */
function Gate({ label, before, after, better }: { label: string; before: number; after: number; better: "up" | "down" }) {
  const d = after - before;
  const good = better === "up" ? d >= 0 : d <= 0;
  const digits = Math.abs(before) < 2 ? 4 : 2;
  return (
    <div className="grid grid-cols-[6.5rem_minmax(0,1fr)] items-baseline gap-2">
      <span className="mono text-[9.5px] uppercase tracking-[0.1em] text-faint">{label}</span>
      <span className="mono flex flex-wrap items-baseline gap-x-2 text-[11px]">
        <span className="text-faint">{before.toFixed(digits)}</span>
        <span className="text-quiet">to</span>
        <span className="text-text">{after.toFixed(digits)}</span>
        <span className={good ? "text-verified" : "text-attention"}>
          {d >= 0 ? "+" : ""}
          {d.toFixed(digits)}
        </span>
      </span>
    </div>
  );
}

// 2026-11-27 (parity gap B1): the lineage editor. It posts POST /api/parameter-maps/lineage, the same
// route as MCP and the API; the route has no preview, so the form states what the write does before
// it is sent. A stated outcome other than used or superseded moves the map into its archive in the
// same step (the route refuses that for a map in use). Logged in Audit; Audit undoes it.
const PURPOSES = ["serving", "base-build", "job-filter", "probe", "measurement", "experiment"] as const;
const OUTCOMES = ["used", "superseded", "failed", "duplicate", "control", "unused"] as const;
const NEEDS_WHY = new Set(["failed", "duplicate", "control"]);
const field = "mono w-full min-w-0 border-b border-line-strong bg-transparent px-0.5 py-0.5 text-[11px] text-text outline-none focus:border-sense-pale";

function FormRow({ label, title, children }: { label: string; title?: string; children: ReactNode }) {
  return (
    <label className="grid grid-cols-[6.5rem_minmax(0,1fr)] items-baseline gap-2" title={title}>
      <span className="mono text-[9.5px] uppercase tracking-[0.1em] text-faint">{label}</span>
      {children}
    </label>
  );
}

function LineageForm({ r, m, onClose }: { r: RoleMaps; m: MapRow; onClose: () => void }) {
  const { busy, run } = useMaps();
  const l = m.lineage;
  const [v, setV] = useState({ made_on: l?.made_on ?? "", made_by: l?.made_by ?? "", purpose: l?.purpose ?? "", outcome: l?.outcome ?? "", outcome_why: l?.outcome_why ?? "", note: l?.note ?? "" });
  const set = (k: keyof typeof v) => (e: { target: { value: string } }) => setV((x) => ({ ...x, [k]: e.target.value }));
  const quarantine = !!v.outcome && v.outcome !== "used" && v.outcome !== "superseded" && v.outcome !== (l?.outcome ?? "");
  const whyMissing = NEEDS_WHY.has(v.outcome) && !v.outcome_why.trim();
  const others = [...(r.serving_base ? [r.serving_base] : []), ...r.maps.filter((x) => x.file !== m.file).map((x) => x.file)];
  const listId = `lineage-on-${r.role}`;
  const save = () => {
    const body: Record<string, string> = { role: r.role, file: m.file, made_on: v.made_on, made_by: v.made_by, note: v.note, outcome_why: v.outcome_why };
    if (v.purpose) body.purpose = v.purpose;
    if (v.outcome) body.outcome = v.outcome;
    const done = quarantine ? `Lineage of ${m.file} recorded; labelled ${v.outcome}, it moved into the ${roleTitle(r.role)}'s archive. Undo in Audit.` : `Lineage of ${m.file} recorded (by you). Undo in Audit.`;
    // The form stays open, so a refused write (the reason shows in the note) keeps what was typed.
    void run(`${r.role}/${m.file}`, "/api/parameter-maps/lineage", body, done);
  };
  return (
    <form
      className="grid gap-2 border-l-2 border-sense-pale/60 pl-3"
      data-e2e="pmap-lineage-form"
      onSubmit={(e) => {
        e.preventDefault();
        if (!whyMissing) save();
      }}
    >
      <FormRow label="made on" title="What this map was measured or built on: a base cartridge, the open weights, or another map">
        <input className={field} list={listId} value={v.made_on} onChange={set("made_on")} placeholder="the cartridge or map it was measured on" />
      </FormRow>
      <datalist id={listId}>
        {others.map((o) => (
          <option key={o} value={o} />
        ))}
      </datalist>
      <FormRow label="made by">
        <input className={field} value={v.made_by} onChange={set("made_by")} placeholder="the tool or command that made it" />
      </FormRow>
      <FormRow label="purpose">
        <select className={field} value={v.purpose} onChange={set("purpose")}>
          <option value="">{l?.purpose ? `keep (${l.purpose})` : "not stated"}</option>
          {PURPOSES.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </FormRow>
      <FormRow label="outcome" title="How it turned out; only used and superseded maps are learned from">
        <select className={field} value={v.outcome} onChange={set("outcome")}>
          <option value="">{l?.outcome ? `keep (${l.outcome})` : "not stated"}</option>
          {OUTCOMES.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      </FormRow>
      <FormRow label="outcome why">
        <input className={field} value={v.outcome_why} onChange={set("outcome_why")} placeholder={NEEDS_WHY.has(v.outcome) ? "required: the gate, the dip, the measurement or the fact" : "the gate, the measurement or the fact"} />
      </FormRow>
      <FormRow label="note">
        <input className={field} value={v.note} onChange={set("note")} />
      </FormRow>
      <p className="mono text-[9.5px] leading-relaxed text-quiet">
        {quarantine ? `Labelled ${v.outcome}, the map leaves the library for the ${roleTitle(r.role)}'s archive in the same step; a map in use is refused.` : "Writes this map's lineage record as recorded by you; an emptied text field is cleared."} Logged in Audit; Audit undoes it.
      </p>
      <div className="flex flex-wrap gap-x-4">
        <button type="submit" disabled={busy || whyMissing} className={act} data-e2e="pmap-lineage-save" title={whyMissing ? `outcome ${v.outcome} needs its reason` : undefined}>
          record lineage
        </button>
        <button type="button" onClick={onClose} className={act}>
          cancel
        </button>
      </div>
    </form>
  );
}

export function MapDetail({ k, inspect: inspect0 = false, lineage: lineage0 = false }: { k: string; inspect?: boolean; lineage?: boolean }) {
  const { roles, filtersOf, layerCount, now, busy, run, onNote, buildRunning, rebuild, openObject } = useMaps();
  const rail = useRail();
  const { select } = useSelectMap();
  const [inspect, setInspect] = useState(inspect0);
  const [editLineage, setEditLineage] = useState(lineage0);
  const { role, file } = split(k);
  const r = roles.find((x) => x.role === role);
  const m = r?.maps.find((x) => x.file === file);
  if (!r || !m) {
    const archived = r?.archived?.some((a) => a.file === file);
    return (
      <div className="grid gap-2" data-e2e="pmap-detail">
        <p className="mono text-[11px] text-faint">{archived ? `${file} is in the ${roleTitle(role)}'s archive now.` : `${file} is no longer among the ${roleTitle(role)}'s maps.`}</p>
        {archived && (
          <button type="button" onClick={() => rail.open("archive")} className={`${act} justify-self-start`}>
            open the archive
          </button>
        )}
      </div>
    );
  }
  const jf = filtersOf(r.role);
  const f = filterOf(jf, m);
  const on = f ? f.enabled !== false : null;
  const total = layerCount(r.model);
  const ls = layersOf(m);
  const id = `${r.role}/${m.file}`;
  const what = covers(m);
  const st = stateOf(m);
  // Connected parameter objects: job filter itself, lineage made_on / used_by cartridges, serving base cartridge, and optional open weights.
  const connected: { key: string; label: string; tag: string; relation: string }[] = [];
  const seenConn = new Set<string>();
  const addConn = (key: string, label: string, tag: string, relation: string) => {
    if (!key || seenConn.has(key)) return;
    seenConn.add(key);
    connected.push({ key, label, tag, relation });
  };
  if (f) addConn(`filter:${r.role}/${f.job}`, `filter:${r.role}/${f.job}`, "job filter", "this filter");
  for (const u of m.lineage?.used_by ?? []) {
    const nk = nodeKind(u, r, jf);
    if (nk === "map" || nk === "archived" || nk === "missing") continue;
    const fc = nk === "filter-cart" ? jf.find((x) => x.cartridge === u) : undefined;
    const targetKey = fc ? `filter:${r.role}/${fc.job}` : u;
    const tag = nk === "base" ? "base cartridge" : nk === "comp" ? "compensation cartridge" : nk === "filter-cart" ? "job filter" : nk === "weights" ? "open weights" : "cartridge";
    addConn(targetKey, u, tag, "feeds");
  }
  if (m.lineage?.made_on) {
    const onId = m.lineage.made_on;
    const nk = nodeKind(onId, r, jf);
    if (nk !== "map" && nk !== "archived" && nk !== "missing") {
      const fc = nk === "filter-cart" ? jf.find((x) => x.cartridge === onId) : undefined;
      const targetKey = fc ? `filter:${r.role}/${fc.job}` : onId;
      const tag = nk === "base" ? "base cartridge" : nk === "comp" ? "compensation cartridge" : nk === "filter-cart" ? "job filter" : nk === "weights" ? "open weights (upstream)" : "cartridge";
      addConn(targetKey, nk === "weights" ? onId.slice(5) : onId, tag, "measured on");
    }
  }
  if (r.serving_base) addConn(r.serving_base, r.serving_base, "base cartridge", "serving now");
  addConn(`open:${r.model}`, r.model, "open weights (optional)", "upstream model");
  return (
    <div className="grid gap-5" data-e2e="pmap-detail">
      {/* What it is, now. */}
      <div className="grid gap-1.5">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="mono text-[10px] uppercase tracking-[0.1em] text-faint" title={m.about}>
            {roleTitle(r.role)} · {m.kind}
          </span>
          <Pill tone={STATE_TONE[st] ?? "attention"}>{st}</Pill>
          {on !== null && <Pill tone={on ? "signal" : "muted"}>{on ? "plugged in" : "unplugged"}</Pill>}
        </div>
        {m.state_why && <p className={`mono text-[10.5px] leading-relaxed ${st === "stale" || st === "origin not recorded" ? "text-attention" : "text-faint"}`}>{m.state_why}</p>}
        <p className="mono flex flex-wrap items-baseline gap-x-2 text-[10.5px] leading-relaxed" data-e2e="pmap-outcome">
          <Pill tone={OUTCOME_TONE[m.lineage?.outcome ?? ""] ?? "attention"} title="How this map turned out; only healthy maps (used, superseded) are learned from">
            {m.lineage?.outcome || "outcome not stated"}
          </Pill>
          {m.lineage?.outcome_why && <span className="text-faint">{m.lineage.outcome_why}</span>}
        </p>
      </div>

      {/* Connected parameter objects: open this map directly on the Parameter Viewer or jump to parameter masking. */}
      {openObject && connected.length > 0 && (
        <div className="mono grid gap-2 border border-line bg-bg-1/70 p-2.5 text-[10px]" data-e2e="pmap-connected">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-[9.5px] uppercase tracking-[0.14em] text-muted">Connected parameter objects</span>
            <span className="text-[9px] text-quiet">{connected.length} linked</span>
          </div>
          <ul className="grid gap-2">
            {connected.map((c) => (
              <li key={c.key} className="grid gap-1 border-t border-line/50 pt-1.5 first:border-t-0 first:pt-0">
                <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-0.5">
                  <span className="min-w-0 truncate text-text" title={c.key}>
                    {c.label}
                  </span>
                  <span className="text-[9px] text-learned">{c.tag}</span>
                  <span className="ml-auto text-[9px] text-quiet">{c.relation}</span>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => openObject(c.key, { map: id })}
                    className="press border border-sense-pale bg-sense/15 px-2 py-0.5 text-[9.5px] uppercase tracking-[0.08em] text-sense-pale hover:bg-sense/25 hover:text-text"
                    title={`Open ${c.label} in the Parameter Viewer with ${m.file} overlaid`}
                  >
                    view map on viewer
                  </button>
                  <button
                    type="button"
                    onClick={() => openObject(c.key, { map: id, controls: true })}
                    className="press border border-line-strong bg-bg-2 px-2 py-0.5 text-[9.5px] uppercase tracking-[0.08em] text-muted hover:border-sense-pale hover:text-text"
                    title={`Open ${c.label} with ${m.file} overlaid and jump straight to masking, freezing, and overlay controls`}
                  >
                    mask on viewer
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Controls first: this is a tool. */}
      <div className="flex flex-wrap items-start gap-x-4 gap-y-2">
        {f && openObject && (
          <button
            type="button"
            onClick={() => openObject(`filter:${r.role}/${f.job}`)}
            className="press mono border-b border-sense-pale px-0.5 text-[10px] uppercase tracking-[0.1em] text-sense-pale hover:text-text"
            title="Open this job filter in the full Parameter Viewer (heatmap, zoom, overlay, edit)"
          >
            parameter viewer
          </button>
        )}
        <button type="button" aria-expanded={inspect} onClick={() => setInspect((v) => !v)} className={f && openObject ? act : "press mono border-b border-sense-pale px-0.5 text-[10px] uppercase tracking-[0.1em] text-sense-pale hover:text-text"}>
          {inspect ? "close" : m.kind === "job filter" ? "inspect / edit JSON" : "inspect"}
        </button>
        {f && (
          <>
            <button type="button" disabled={busy} onClick={() => void run(id, "/api/surgery/job-filter", { role: r.role, job: f.job, action: on ? "disable" : "enable" }, `Job filter ${roleTitle(r.role)}/${f.job} ${on ? "unplugged" : "plugged in"}; the next call uses the change.`)} className={act}>
              {on ? "unplug" : "plug in"}
            </button>
            <button
              type="button"
              disabled={busy || buildRunning}
              onClick={() => {
                rebuild(r.role, f.job);
                rail.open("add");
              }}
              className={act}
              title="Opens the Add tab with this MicroModel and job; pick the layers there"
            >
              rebuild on other layers
            </button>
            <button
              type="button"
              disabled={busy || buildRunning}
              data-e2e="filter-check"
              onClick={() => void run(id, "/api/lab/build", { kind: "check", role: r.role, job: f.job }, `Held-out check of ${roleTitle(r.role)}/${f.job} started on the graphics card; the result shows in Live Activity and the build status.`)}
              className={act}
              title="Measures this job filter on the held-out prompts against the open weights; nothing is fitted or written"
            >
              held-out check
            </button>
          </>
        )}
        <button type="button" aria-expanded={editLineage} data-e2e="pmap-lineage-edit" onClick={() => setEditLineage((x) => !x)} className={act} title="Record or correct where this map came from, what it is for and how it turned out">
          {editLineage ? "close lineage" : "lineage"}
        </button>
        {cleanable(m) && (
          <button type="button" disabled={busy} onClick={() => void run(id, "/api/parameter-maps/archive", { role: r.role, files: [m.file] }, `${m.file} moved into the ${roleTitle(r.role)}'s archive. Undo in Audit, or restore it from the Archive tab.`)} className={act} title="Moves it into this MicroModel's archive folder; restore it any time">
            archive
          </button>
        )}
        <DeleteFlow<{ audit_id?: string }>
          path={f ? "/api/surgery/job-filter" : "/api/parameter-maps/remove"}
          body={f ? { role: r.role, job: f.job, action: "delete" } : { role: r.role, file: m.file }}
          label="remove"
          title={m.kind === "job filter" ? "Asks what would go first; a removed filter can be put back from Audit." : "Asks what would go first. Base cartridges already built from this map keep serving."}
          startClass={delCls}
          disabled={busy}
          done={() => (f ? `Job filter ${roleTitle(r.role)}/${f.job} removed (kept aside; Undo in Audit puts it back).` : `${m.file} removed from the ${roleTitle(r.role)}'s parameter maps. Logged in Audit.`)}
          refresh={["/api/parameter-maps", "/api/surgery"]}
          onNote={onNote}
        />
      </div>
      {inspect && <MapInspector role={r.role} file={m.file} onClose={() => setInspect(false)} />}
      {editLineage && <LineageForm r={r} m={m} onClose={() => setEditLineage(false)} />}

      {/* Lineage, drawn. */}
      <div className="grid gap-2">
        <div className="eyebrow" title="From the made on and used by records of each map; read top to bottom">
          Lineage
        </div>
        <MapLineage r={r} file={m.file} jf={jf} onMap={(x) => select(`${r.role}/${x}`)} onObject={openObject} />
        {m.lineage?.made_by && (
          <span className="mono truncate text-[9.5px] text-quiet" title={m.lineage.made_by}>
            made by {m.lineage.made_by}
          </span>
        )}
        {m.lineage?.recorded && <span className="mono text-[9.5px] text-quiet">recorded {m.lineage.recorded === "backfill" ? "from proof after the fact" : m.lineage.recorded === "tool" ? "by the tool that made it" : "by you"}</span>}
      </div>

      {/* Layers covered. */}
      <div className="grid gap-1.5">
        <div className="eyebrow flex items-baseline justify-between gap-3">
          <span>Layers</span>
          <span className="mono normal-case tracking-normal text-faint">{ls.length ? `${ls.length} of ${total ?? "?"}` : f?.layers ? `${f.layers} of ${total ?? "?"}, from the build record` : "no layer list"}</span>
        </div>
        {ls.length > 0 && total ? (
          <div className="grid gap-0.5">
            <div className="flex h-4 gap-px" title={`layers ${ls.join(", ")}`}>
              {Array.from({ length: total }, (_, i) => (
                <i key={i} className={`h-full flex-1 ${ls.includes(i) ? "bg-sense-pale" : "bg-line"}`} />
              ))}
            </div>
            <div className="mono flex justify-between text-[9px] text-quiet">
              <span>0</span>
              <span>{total - 1}</span>
            </div>
            {ls.some((l) => l >= total) && <span className="mono text-[9.5px] text-quiet">also names {ls.filter((l) => l >= total).join(", ")}, past the model&apos;s {total} layers (a budget or block index)</span>}
          </div>
        ) : ls.length > 0 ? (
          <span className="mono text-[10.5px] text-faint">layers {ls.join(", ")} · model layer count not reported</span>
        ) : null}
      </div>

      {/* A job filter's held-out split gate. */}
      {f && typeof f.base_cos === "number" && typeof f.filtered_cos === "number" && (
        <div className="grid gap-1.5">
          <div className="eyebrow" title="Measured on the held-out split when the filter was built: how close the output is to the upstream open weights, on the base cartridge alone and with this filter">
            Held-out split gate
          </div>
          <Gate label="cosine" before={f.base_cos} after={f.filtered_cos} better="up" />
          {typeof f.base_max_diff === "number" && typeof f.filtered_max_diff === "number" && <Gate label="max difference" before={f.base_max_diff} after={f.filtered_max_diff} better="down" />}
          <span className="mono text-[9.5px] text-quiet">
            on {f.base ?? r.serving_base ?? "the base cartridge"} · cartridge {f.cartridge}
          </span>
        </div>
      )}

      {/* The rest of the record. */}
      <div className="mono grid gap-1 border-t border-line/60 pt-3 text-[10.5px] text-faint">
        {what && <span>{what}</span>}
        <span>
          {bytes(m.bytes)}
          {now > 0 && m.modified_ms ? ` · changed ${ago(m.modified_ms, now)}` : ""}
          {m.purpose ? ` · for ${m.purpose}` : ""}
        </span>
        {m.lineage?.note && <span className="text-quiet">{m.lineage.note}</span>}
        <span className="text-quiet">{m.about}</span>
      </div>
    </div>
  );
}
