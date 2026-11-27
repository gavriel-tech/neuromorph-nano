"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... The main
// core of the pages should be on actual work flow and core function visuals"): every map, listed, as
// the second view of the Parameters page's middle (atlas | list). Each MicroModel's maps grouped by
// what they are for now (serving, what the serving base is built from, probes and logs, origin not
// recorded, history, experiments), each row with its controls; a group's old maps archive or delete in
// one go. What a kind of map is, and the row's full record, are in the right rail.

import { useState } from "react";
import DeleteFlow from "@/components/DeleteFlow";
import { useSelectMap } from "@/components/MapDetail";
import { BaseMark, WeightsMark, roleTitle } from "@/components/MicroModel";
import { useRail } from "@/components/Workspace";
import { Empty, Fold, Pill } from "@/components/ui";
import { ago, bytes } from "@/lib/api";
import { GROUPS, OUTCOME_TONE, STATE_TONE, cleanable, covers, filterOf, filtered, groupOf, kindTone, layersOf, passes, stateOf, sum, useMaps, type JobFilter, type MapFilter, type MapRow, type RoleMaps } from "@/components/MapCore";

const linkBtn = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const delBtn = "press mono border-b border-line px-0.5 text-[10px] uppercase tracking-[0.1em] text-faint hover:text-error disabled:opacity-40";

/** The layers a map covers, as a strip of the MicroModel's layers. */
function Coverage({ layers, total }: { layers: number[]; total: number | null }) {
  if (!layers.length) return null;
  if (total === null) return <div className="mono text-[10px] text-quiet">layers {layers.join(", ")}</div>;
  const set = new Set(layers);
  return (
    <div aria-label={`covers ${layers.filter((l) => l < total).length} of ${total} layers`} className="flex h-1.5 w-full max-w-[14rem] gap-px" title={`layers ${layers.join(", ")}`}>
      {Array.from({ length: total }, (_, i) => (
        <i key={i} className={`h-full flex-1 ${set.has(i) ? "bg-sense-pale" : "bg-line"}`} />
      ))}
    </div>
  );
}

function Row({ r, m, jf, total }: { r: RoleMaps; m: MapRow; jf: JobFilter[]; total: number | null }) {
  const { now, busy, run, onNote, buildRunning, rebuild, openObject } = useMaps();
  const { selected, select } = useSelectMap();
  const rail = useRail();
  const f = filterOf(jf, m);
  const on = f ? f.enabled !== false : null;
  const id = `${r.role}/${m.file}`;
  const st = stateOf(m);
  return (
    <li className={`grid min-w-0 gap-1.5 border-l-2 py-1 pl-3 ${selected === id ? "border-sense-pale" : "border-line"}`} data-e2e="pmap-row">
      <div className="flex min-w-0 items-center gap-2">
        <i aria-hidden className={`inline-block h-2 w-2 shrink-0 ${kindTone(m.kind)}`} />
        <button type="button" onClick={() => select(selected === id ? null : id)} className="press mono min-w-0 truncate text-left text-[12px] text-text hover:text-sense-pale" title={`${m.file}: ${m.about}`}>
          {m.file}
        </button>
        <span className="mono shrink-0 text-[10px] uppercase tracking-[0.1em] text-faint">{m.kind}</span>
        {on !== null && <Pill tone={on ? "verified" : "muted"}>{on ? "plugged in" : "unplugged"}</Pill>}
        {m.state && (
          <Pill tone={STATE_TONE[st] ?? "muted"} title={m.state_why}>
            {m.state}
          </Pill>
        )}
        {m.lineage?.outcome && (
          <Pill tone={OUTCOME_TONE[m.lineage.outcome] ?? "muted"} title={m.lineage.outcome_why || "how this map turned out"}>
            {m.lineage.outcome}
          </Pill>
        )}
      </div>
      <div className="mono text-[10.5px] text-faint">{[covers(m), bytes(m.bytes), now > 0 && m.modified_ms ? ago(m.modified_ms, now) : ""].filter(Boolean).join(" · ")}</div>
      <Coverage layers={layersOf(m)} total={total} />
      {(m.state_why || m.lineage?.made_by) && (
        <div className="mono grid gap-0.5 text-[10px] text-quiet" data-e2e="pmap-lineage">
          {m.state_why && <span className={st === "stale" || st === "origin not recorded" ? "text-attention" : "text-faint"}>{m.state_why}</span>}
          {m.lineage?.made_by && (
            <span className="truncate" title={m.lineage.made_by}>
              made by {m.lineage.made_by}
            </span>
          )}
        </div>
      )}
      <div className="flex flex-wrap items-start gap-x-4 gap-y-1">
        {f && openObject && (
          <button type="button" onClick={() => openObject(`filter:${r.role}/${f.job}`)} className="press mono border-b border-sense-pale px-0.5 text-[10px] uppercase tracking-[0.1em] text-sense-pale hover:text-text" title="Open this job filter in the full Parameter Viewer">
            parameter viewer
          </button>
        )}
        <button type="button" onClick={() => select(id, true)} className={f && openObject ? linkBtn : "press mono border-b border-sense-pale px-0.5 text-[10px] uppercase tracking-[0.1em] text-sense-pale hover:text-text"} title="Opens it in the side panel with its contents">
          {m.kind === "job filter" ? "inspect / edit filter" : "inspect"}
        </button>
        <button type="button" onClick={() => select(id, false, true)} className={linkBtn} data-e2e="pmap-row-lineage" title="Opens it in the side panel with its lineage editor: made on, made by, purpose, outcome and note">
          lineage
        </button>
        {f && (
          <>
            <button type="button" disabled={busy} onClick={() => void run(id, "/api/surgery/job-filter", { role: r.role, job: f.job, action: on ? "disable" : "enable" }, `Job filter ${roleTitle(r.role)}/${f.job} ${on ? "unplugged" : "plugged in"}; the next call uses the change.`)} className={linkBtn}>
              {on ? "unplug" : "plug in"}
            </button>
            <button
              type="button"
              disabled={busy || buildRunning}
              onClick={() => {
                rebuild(r.role, f.job);
                rail.open("add");
              }}
              className={linkBtn}
            >
              rebuild on other layers
            </button>
          </>
        )}
        {cleanable(m) && (
          <button type="button" disabled={busy} onClick={() => void run(id, "/api/parameter-maps/archive", { role: r.role, files: [m.file] }, `${m.file} moved into the ${roleTitle(r.role)}'s archive; nothing serves or builds from there. Undo in Audit, or restore it from the Archive tab.`)} className={linkBtn} title="Moves it into this MicroModel's archive folder; restore it any time">
            archive
          </button>
        )}
        <DeleteFlow<{ audit_id?: string }>
          path={f ? "/api/surgery/job-filter" : "/api/parameter-maps/remove"}
          body={f ? { role: r.role, job: f.job, action: "delete" } : { role: r.role, file: m.file }}
          label="remove"
          title={m.kind === "job filter" ? "Asks what would go first; a removed filter can be put back from Audit." : "Asks what would go first. Base cartridges already built from this map keep serving; building them again needs it."}
          startClass={delBtn}
          disabled={busy}
          done={() => (f ? `Job filter ${roleTitle(r.role)}/${f.job} removed (kept aside; Undo in Audit puts it back).` : `${m.file} removed from the ${roleTitle(r.role)}'s parameter maps. Logged in Audit.`)}
          refresh={["/api/parameter-maps", "/api/surgery"]}
          onNote={onNote}
        />
      </div>
    </li>
  );
}

/** A group's old maps: archive them all (asked once) or delete them all (through the delete flow). */
function Cleanup({ r, old, armed, setArmed }: { r: RoleMaps; old: MapRow[]; armed: boolean; setArmed: (v: boolean) => void }) {
  const { busy, run, onNote } = useMaps();
  const b = sum(old);
  const who = roleTitle(r.role);
  return (
    <div className="mono mb-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px]" data-e2e="pmap-cleanup">
      {armed ? (
        <>
          <span className="text-text">
            Archive {old.length} {old.length === 1 ? "map" : "maps"} ({bytes(b)}) of the {who}? Restore them any time, or undo in Audit.
          </span>
          <button
            type="button"
            autoFocus
            disabled={busy}
            onClick={() => {
              setArmed(false);
              void run(`${r.role}:cleanup`, "/api/parameter-maps/archive", { role: r.role, files: old.map((m) => m.file) }, `${old.length} ${who} maps moved into its archive. Undo in Audit, or restore them from the Archive tab.`);
            }}
            className={linkBtn}
          >
            confirm archive
          </button>
          <button type="button" onClick={() => setArmed(false)} className={linkBtn}>
            cancel
          </button>
        </>
      ) : (
        <>
          <span className="text-faint">
            clean up {old.length} {old.length === 1 ? "map" : "maps"} · {bytes(b)}:
          </span>
          <button type="button" disabled={busy} onClick={() => setArmed(true)} className={linkBtn}>
            archive all
          </button>
          <DeleteFlow<{ audit_id?: string }>
            path="/api/parameter-maps/remove"
            body={{ role: r.role, files: old.map((m) => m.file), confirm: true }}
            label="delete all"
            title="Asks what would go first, then asks you to type a word; deleting cannot be undone (archiving can)"
            startClass={delBtn}
            disabled={busy}
            done={() => `${old.length} ${who} maps deleted (${bytes(b)}). Logged in Audit.`}
            refresh={["/api/parameter-maps"]}
            onNote={onNote}
          />
        </>
      )}
    </div>
  );
}

export default function MapList({ filter }: { filter: MapFilter }) {
  const { roles, layerCount, filtersOf, openObject } = useMaps();
  const rail = useRail();
  const [armed, setArmed] = useState<string | null>(null);
  return (
    <div className="grid min-w-0 gap-8">
      {filtered(filter) && <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">filtered as the atlas</span>}
      {roles
        .filter((r) => filter.role === "all" || r.role === filter.role)
        .map((r) => {
          const total = layerCount(r.model);
          const jf = filtersOf(r.role);
          const arch = r.archived ?? [];
          return (
            <Fold
              key={r.role}
              open
              title={
                <span className="flex items-baseline gap-3">
                  <span className="text-text">{roleTitle(r.role)}</span>
                  <span className="normal-case tracking-normal">
                    {openObject && r.serving_base ? (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          openObject(r.serving_base!);
                        }}
                        className="press border-b border-transparent hover:border-sense-pale"
                        title={`Open base cartridge ${r.serving_base} in the Parameter Editor`}
                      >
                        <BaseMark base={r.serving_base} quiet labeled />
                      </button>
                    ) : (
                      <BaseMark base={r.serving_base} quiet labeled />
                    )}
                  </span>
                  <span className="normal-case tracking-normal">
                    {openObject ? (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          openObject(`open:${r.model}`);
                        }}
                        className="press border-b border-transparent hover:border-sense-pale"
                        title={`Open upstream open weights (${r.model}) in the Parameter Editor`}
                      >
                        <WeightsMark model={r.model} labeled />
                      </button>
                    ) : (
                      <WeightsMark model={r.model} labeled />
                    )}
                  </span>
                </span>
              }
              right={`${r.maps.length} maps · ${bytes(sum(r.maps))} · ${jf.filter((x) => x.enabled !== false).length} of ${jf.length} job filters plugged in`}
            >
              {r.maps.length === 0 ? (
                <Empty title="No maps yet">Build a job filter from the Add tab, or map this MicroModel with the lab tools.</Empty>
              ) : (
                <div className="grid gap-6">
                  {GROUPS.map((g) => {
                    const items = r.maps.filter((m) => groupOf(m) === g.id && passes(m, filter));
                    if (items.length === 0) return null;
                    const old = items.filter(cleanable);
                    const key = `${r.role}:${g.id}`;
                    const body = (
                      <ul className="grid gap-x-8 gap-y-4 2xl:grid-cols-2">
                        {items.map((m) => (
                          <Row key={m.file} r={r} m={m} jf={jf} total={total} />
                        ))}
                      </ul>
                    );
                    const cleanup = old.length > 0 && <Cleanup r={r} old={old} armed={armed === key} setArmed={(v) => setArmed(v ? key : null)} />;
                    return g.fold ? (
                      <Fold
                        key={g.id}
                        title={
                          <span title={g.about}>
                            {g.label}{" "}
                            <span className="normal-case tracking-normal text-faint">
                              {items.length} · {bytes(sum(items))}
                            </span>
                          </span>
                        }
                      >
                        <div className="grid gap-2">
                          {cleanup}
                          {body}
                        </div>
                      </Fold>
                    ) : (
                      <section key={g.id} className="grid gap-2" data-e2e="pmap-group" data-group={g.id}>
                        <h3 className="eyebrow flex flex-wrap items-baseline gap-x-3 border-b border-line/60 pb-1.5" title={g.about}>
                          <span>{g.label}</span>
                          <span className="normal-case tracking-normal text-faint">
                            {items.length} · {bytes(sum(items))}
                          </span>
                        </h3>
                        {cleanup}
                        {body}
                      </section>
                    );
                  })}
                </div>
              )}
              {arch.length > 0 && (
                <button type="button" onClick={() => rail.open("archive")} className={`${linkBtn} mt-4`}>
                  {arch.length} archived · {bytes(sum(arch))} · open the archive
                </button>
              )}
            </Fold>
          );
        })}
    </div>
  );
}
