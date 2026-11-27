"use client";

// 2026-11-25 (directions of Luis Jake Gabriel III: the base models are downloadable from the
// interface and kept separate from the base cartridges). Two lists, each with its own download:
// the serving base cartridges first (required to run standalone, from the cartridge repository,
// also fetching the small tokenizer/config JSONs), then the optional upstream open weights
// (Apache-2.0 Qwen3 weights at pinned Hugging Face revisions, used for fallback, output-parity
// reference checks, parameter mapping, and cartridge surgery). Every file is checked against its SHA-256 before it
// is placed; a partial file resumes. Nothing is fetched while the air gap is on.
//
// 2026-11-26 (Luis Jake Gabriel III, his words: "the right menu are libraries and or discription
// tools etc."): Downloads is a tab of the Models side panel now: one narrow column, the two lists
// one under the other, the explanations in hover titles.

import { useState } from "react";
import DeleteFlow from "@/components/DeleteFlow";
import { HFDownloadButton } from "@/components/HFDownloadManager";
import { Pill, Placeholder } from "@/components/ui";
import { bytes, post, refreshAll, useApi } from "@/lib/api";

type ModelRow = { id: string; repo: string; revision: string; bytes: number; files: number; present: number; partial: number; on_disk: number; state: string };
type CartRow = { key: string; present: boolean; on_disk: number; path: string; description?: string };
type MapFileRow = { file: string; kind: string; bytes: number; description?: string };
type MapRoleRow = { role: string; files: number; job_filters: number; super_weights: boolean; on_disk: number; items?: MapFileRow[] };
type LocalStackRow = { key: string; use: string; description: string };
type SourceFileRow = { path: string; bytes: number; kind: string; description?: string | null; installable: boolean };
type SourcesLite = { sources?: { id: string; repo: string; official: boolean; listing?: { files: SourceFileRow[] } | null }[]; publishable?: { stacks: LocalStackRow[] } };
/** `stopped`: set when the process running the download ended before it finished (then
 *  `running` is false). */
type Job = { what: string; running: boolean; done: number; total: number; current: string; current_bytes: number; failed: string[]; finished_ms: number; by?: string; updated_ms?: number; stopped?: string };
type Payload = { models?: ModelRow[]; cartridges?: CartRow[]; parameter_maps?: MapRoleRow[]; cartridge_repo?: string; token?: boolean; air_gap?: boolean; job?: Job | null };
type Note = { text: string; tone: "verified" | "error" };

const btn = "press mono border-b px-0.5 text-[10px] uppercase tracking-[0.12em] disabled:opacity-40";

/** A download waiting for its confirmation: what it fetches, in words and figures. */
type Armed = { what: string; only?: string; label: string; files: number | null; bytes: number | null };

/** 2026-11-26: the confirmation sits inline, on the ground (no browser dialog): what will be
 *  fetched, how many files, about how large, and that every file is checked against its SHA-256. */
function ConfirmRow({ armed, busy, onConfirm, onCancel }: { armed: Armed; busy: boolean; onConfirm: () => void; onCancel: () => void }) {
  return (
    <div
      role="group"
      aria-label={`Confirm download of ${armed.label}`}
      className="mono mt-2 grid gap-2 border-l border-sense-pale pl-3 text-[11px]"
      data-e2e="download-confirm"
      onKeyDown={(e) => {
        if (e.key === "Escape") onCancel();
      }}
    >
      <span className="text-text">Download {armed.label}?</span>
      <span className="text-faint">
        {armed.files !== null ? `${armed.files} ${armed.files === 1 ? "file" : "files"}` : "missing files from the official Hugging Face library"}
        {armed.bytes !== null ? ` · about ${bytes(armed.bytes)}` : " · each size is shown as the file arrives"} · every file is checked before it is kept; a partial file resumes
      </span>
      <span className="flex items-center gap-4">
        <button type="button" autoFocus disabled={busy} onClick={onConfirm} className={`${btn} border-sense-pale text-sense-pale hover:text-text`}>
          Confirm download
        </button>
        <button type="button" disabled={busy} onClick={onCancel} className={`${btn} border-line text-faint hover:text-text`}>
          Cancel
        </button>
      </span>
    </div>
  );
}

const ACTIVE_MAP_FILE = (name: string) =>
  name.startsWith("job-filter-") || name === "super-weights.json" || name.startsWith("pmap-");

function latestStacks(stacks: LocalStackRow[]): LocalStackRow[] {
  const best = new Map<string, { ver: number; row: LocalStackRow }>();
  for (const st of stacks) {
    const m = /^(.+)@(\d+)$/.exec(st.key);
    const base = (m ? m[1] : st.key).toLowerCase();
    const ver = m ? Number(m[2]) : 0;
    const prev = best.get(base);
    if (!prev || ver > prev.ver) best.set(base, { ver, row: st });
  }
  return [...best.values()].map((x) => x.row);
}

export function DownloadsPanel({ onNote }: { onNote: (n: Note) => void }) {
  const q = useApi<Payload>("/api/downloads", 2000);
  const srcQ = useApi<SourcesLite>("/api/cartridge-sources", 15000);
  const [busy, setBusy] = useState(false);
  const [armed, setArmed] = useState<Armed | null>(null);
  const [openInfo, setOpenInfo] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const toggleInfo = (id: string) => setOpenInfo((prev) => (prev === id ? null : id));
  const infoBtn = "press mono border-b border-line px-0.5 text-[9.5px] uppercase tracking-[0.1em] text-quiet hover:text-text";
  const d = q.data;
  const localStacks = latestStacks(srcQ.data?.publishable?.stacks ?? []);
  const localStackSlugs = new Set(localStacks.map((st) => `${st.key.replace("@", "-v")}.stack.json`.toLowerCase()));
  const officialSource = (srcQ.data?.sources ?? []).find((s) => s.official);
  const remoteStacks = (officialSource?.listing?.files ?? []).filter((f) => {
    if (f.kind !== "stack file" || !f.installable) return false;
    const name = (f.path.split("/").pop() ?? f.path).toLowerCase();
    return !localStackSlugs.has(name);
  });
  const job = d?.job;
  const running = !!job?.running;
  const start = async (a: Armed) => {
    setBusy(true);
    try {
      const payload = a.only ? { what: a.what, only: a.only } : { what: a.what };
      const r = await post<{ line?: string }>("/api/downloads", payload);
      for (const p of ["/api/downloads", "/api/parameter-maps", "/api/cartridge-sources"]) refreshAll(p);
      onNote({ text: `${r.line ?? "Download started"}.`, tone: "verified" });
      setArmed(null);
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  const missingCarts = (d?.cartridges ?? []).filter((c) => !c.present);
  const mapRoles = (d?.parameter_maps ?? []).map((r) => {
    const activeItems = (r.items ?? []).filter((it) => ACTIVE_MAP_FILE(it.file));
    return {
      ...r,
      items: activeItems,
      files: activeItems.length,
      job_filters: activeItems.filter((it) => it.file.startsWith("job-filter-")).length,
      on_disk: activeItems.reduce((sum, it) => sum + (it.bytes || 0), 0),
    };
  });
  const totalMapFiles = mapRoles.reduce((sum, r) => sum + r.files, 0);
  const confirmFor = (what: string, only?: string) =>
    armed && armed.what === what && armed.only === only ? <ConfirmRow armed={armed} busy={busy} onConfirm={() => void start(armed)} onCancel={() => setArmed(null)} /> : null;
  return (
    <section className="min-w-0" aria-label="Downloads">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <span className="eyebrow text-muted" title="Base cartridges are required to run (downloading them also fetches the small tokenizer and config JSONs). Upstream open weights are optional: used for fallback, output-parity reference checks, parameter mapping, and cartridge surgery. Every file is checked against its SHA-256 before it is kept; a partial file resumes.">
          Downloads
        </span>
        <span className="flex flex-wrap items-center gap-2">
          <HFDownloadButton label="open HF download window" />
          {d?.air_gap && <Pill tone="attention">air gap on: no downloads</Pill>}
        </span>
      </div>
      {!d && q.error ? (
        <div className="mono text-[11px] text-error">Downloads did not answer: {q.error}</div>
      ) : !d ? (
        <Placeholder className="h-24" what="downloads" />
      ) : (
        <div className="grid gap-6" data-e2e="downloads">
          {job && (
            <div className="mono grid gap-1.5 text-[11px]" data-e2e="download-job">
              <div className="flex flex-wrap items-center gap-3">
                <Pill tone={job.running ? "signal" : job.stopped ? "attention" : job.failed.length ? "error" : "verified"}>{job.running ? "downloading" : job.stopped ? "stopped" : job.failed.length ? "finished with faults" : "finished"}</Pill>
                <span className="text-muted">
                  {job.done} of {job.total} files
                </span>
                {job.by && <span className="text-quiet">by {job.by}</span>}
              </div>
              {job.running && job.current && (
                <span className="min-w-0 truncate text-faint" title={job.current}>
                  {job.current} · {bytes(job.current_bytes)} so far
                </span>
              )}
              <div className="h-1 w-full bg-line">
                <div className="h-full bg-sense-pale transition-[width] duration-300" style={{ width: `${job.total ? Math.round((job.done / job.total) * 100) : 0}%` }} />
              </div>
              {!job.running && job.stopped && <div className="text-attention">{job.stopped}</div>}
              {job.failed.map((f) => (
                <div key={f} className="break-all text-error">
                  {f}
                </div>
              ))}
            </div>
          )}
          <section className="min-w-0">
            <div className="mono flex flex-wrap items-baseline justify-between gap-2 text-[10px] uppercase tracking-[0.12em]">
              <span
                className="flex items-center gap-1.5 text-muted"
                title={`The Neuromorph base cartridges the MicroModels run on standalone, from ${d.cartridge_repo ?? "the public Hugging Face cartridge library"} (also fetches the small tokenizer and config JSONs).`}
              >
                <i aria-hidden className="inline-block h-1.5 w-1.5 bg-learned" />
                Base cartridges · required
              </span>
              <span className="normal-case tracking-normal text-quiet" title={`Public Hugging Face repository: https://huggingface.co/${d.cartridge_repo ?? "gavriel-tech/neuromorph-nano-cartridges"}`}>
                public library
              </span>
            </div>
            <div className="mono mb-2 mt-0.5 text-[10px] text-faint">Runs all four MicroModels standalone.</div>
            <ul className="grid">
              {(d.cartridges ?? []).map((c) => {
                const infoId = `cart:${c.key}`;
                const shown = openInfo === infoId;
                return (
                  <li key={c.key} className="grid gap-1 border-b border-line/60 py-1.5">
                    <div className="flex min-w-0 items-center justify-between gap-3">
                      <span className="mono min-w-0 truncate text-[11.5px] text-text" title={c.path}>
                        {c.key}
                      </span>
                      <span className="flex shrink-0 items-center gap-3">
                        <span className="mono text-[10.5px] text-faint">{c.present ? bytes(c.on_disk) : "not on disk"}</span>
                        <Pill tone={c.present ? "verified" : "muted"}>{c.present ? "in library" : "missing"}</Pill>
                        <button type="button" onClick={() => toggleInfo(infoId)} aria-expanded={shown} className={infoBtn}>
                          {shown ? "hide" : "info"}
                        </button>
                        {c.present && deleting !== `base:${c.key}` && (
                          <button
                            type="button"
                            onClick={() => setDeleting(`base:${c.key}`)}
                            className="press mono border-b border-line px-0.5 text-[9.5px] uppercase tracking-[0.1em] text-faint hover:border-error hover:text-error"
                            title="Preview and remove this base cartridge (serving or chosen versions are protected)"
                          >
                            delete
                          </button>
                        )}
                        {!c.present && !(armed && armed.what === "cartridges" && armed.only === c.key) && (
                          <button
                            type="button"
                            disabled={busy || running || d.air_gap}
                            onClick={() => setArmed({ what: "cartridges", only: c.key, label: c.key, files: 1, bytes: null })}
                            className={`${btn} border-sense-pale text-sense-pale hover:text-text`}
                          >
                            download
                          </button>
                        )}
                      </span>
                    </div>
                    {deleting === `base:${c.key}` && (
                      <div className="border-l border-error/60 pl-2.5">
                        <DeleteFlow
                          autoStart
                          path="/api/cartridges/bases/remove"
                          body={{ key: c.key }}
                          label="delete"
                          done={() => `Removed ${c.key}.`}
                          refresh={["/api/downloads", "/api/cartridges/bases", "/api/cartridges", "/api/status"]}
                          onNote={onNote}
                          onClose={() => setDeleting(null)}
                        />
                      </div>
                    )}
                    {shown && (
                      <div className="mono border-l border-line-strong pl-2.5 text-[10.5px] leading-relaxed text-faint">
                        <div>{c.description || "Standalone MicroModel base cartridge."}</div>
                        <div className="text-quiet">path: {c.path}</div>
                      </div>
                    )}
                    {confirmFor("cartridges", c.key)}
                  </li>
                );
              })}
            </ul>
            {missingCarts.length > 0 && !(armed && armed.what === "cartridges" && !armed.only) && (
              <button
                type="button"
                disabled={busy || running || d.air_gap}
                onClick={() => setArmed({ what: "cartridges", label: "all missing base cartridges", files: missingCarts.length, bytes: null })}
                className={`${btn} mt-3 border-sense-pale text-sense-pale hover:text-text`}
              >
                download all missing base cartridges ({missingCarts.length})
              </button>
            )}
            {confirmFor("cartridges")}
          </section>
          <section className="min-w-0">
            <div className="mono flex flex-wrap items-baseline justify-between gap-2 text-[10px] uppercase tracking-[0.12em]">
              <span
                className="flex items-center gap-1.5 text-muted"
                title={`Serving job filters, super-weights and base-build parameter maps from ${d.cartridge_repo ?? "the official Hugging Face cartridge library"}. Existing local maps are never overwritten here.`}
              >
                <i aria-hidden className="inline-block h-1.5 w-1.5 bg-sense-pale" />
                Parameter maps &amp; job filters
              </span>
              <span className="normal-case tracking-normal text-quiet">{totalMapFiles} on disk</span>
            </div>
            <div className="mono mb-2 mt-0.5 text-[10px] text-faint">Serving job filters, surgery baselines and full maps from Hugging Face.</div>
            {mapRoles.length > 0 && (
              <ul className="grid">
                {mapRoles.map((r) => {
                  const infoId = `maps:${r.role}`;
                  const shown = openInfo === infoId;
                  return (
                    <li key={r.role} className="grid gap-1 border-b border-line/60 py-1.5">
                      <div className="flex min-w-0 items-center justify-between gap-3">
                        <span className="mono min-w-0 truncate text-[11.5px] text-text">{r.role}</span>
                        <span className="flex shrink-0 items-center gap-3">
                          <span className="mono text-[10.5px] text-faint">
                            {r.files} {r.files === 1 ? "map" : "maps"} · {r.job_filters} {r.job_filters === 1 ? "filter" : "filters"} · {bytes(r.on_disk)}
                          </span>
                          <Pill tone={r.job_filters > 0 ? "verified" : "muted"}>{r.job_filters > 0 ? "installed" : "missing"}</Pill>
                          <button type="button" onClick={() => toggleInfo(infoId)} aria-expanded={shown} className={infoBtn}>
                            {shown ? "hide" : "info"}
                          </button>
                        </span>
                      </div>
                      {(r.items ?? []).length > 0 && !shown && (
                        <div className="mono flex flex-wrap gap-1 pt-0.5 text-[9.5px]">
                          {(r.items ?? []).map((it) => (
                            <span
                              key={it.file}
                              title={it.description || `${it.file} (${it.kind})`}
                              className={`border px-1.5 py-0 ${it.file.startsWith("job-filter-") ? "border-sense-pale/50 bg-sense/10 text-sense-pale" : it.file === "super-weights.json" ? "border-[#ffd452]/50 bg-[#ffd452]/10 text-[#ffd452]" : "border-line bg-bg-2/60 text-faint"}`}
                            >
                              {it.file.replace(/\.json$/, "")}
                            </span>
                          ))}
                        </div>
                      )}
                      {shown && (
                        <div className="mono grid gap-1.5 border-l border-line-strong pl-2.5 text-[10.5px] leading-relaxed text-faint">
                          {(r.items ?? []).length === 0 ? (
                            <span className="text-quiet">No local maps installed for {r.role} yet. Download serving filters or full maps below.</span>
                          ) : (
                            (r.items ?? []).map((it) => {
                              const delId = `map:${r.role}/${it.file}`;
                              const isFilter = it.file.startsWith("job-filter-") && it.file.endsWith(".json");
                              const delPath = isFilter ? "/api/surgery/job-filter" : "/api/parameter-maps/remove";
                              const delBody = isFilter
                                ? { role: r.role, job: it.file.slice("job-filter-".length, -".json".length), action: "delete" }
                                : { role: r.role, file: it.file };
                              return (
                                <div key={it.file} className="grid gap-0.5">
                                  <div className="flex items-center justify-between gap-2">
                                    <span className="min-w-0 truncate text-muted">
                                      {it.file} <span className="text-quiet">· {it.kind} · {bytes(it.bytes)}</span>
                                    </span>
                                    {deleting !== delId && (
                                      <button
                                        type="button"
                                        onClick={() => setDeleting(delId)}
                                        className="press mono shrink-0 border-b border-line px-0.5 text-[9px] uppercase tracking-[0.1em] text-faint hover:border-error hover:text-error"
                                        title="Preview and remove this file (two-step confirmation; serving files are protected)"
                                      >
                                        delete
                                      </button>
                                    )}
                                  </div>
                                  {it.description && <span className="text-faint">{it.description}</span>}
                                  {deleting === delId && (
                                    <div className="mt-1 border-l border-error/60 pl-2">
                                      <DeleteFlow
                                        autoStart
                                        path={delPath}
                                        body={delBody}
                                        label="delete"
                                        done={() => `Removed ${r.role}/${it.file}.`}
                                        refresh={["/api/downloads", "/api/parameter-maps", "/api/surgery", "/api/cartridge-sources", "/api/cartridges"]}
                                        onNote={onNote}
                                        onClose={() => setDeleting(null)}
                                      />
                                    </div>
                                  )}
                                </div>
                              );
                            })
                          )}
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
            <div className="mt-3 flex flex-wrap items-center gap-4">
              {!(armed && armed.what === "parameter-maps" && armed.only === "serving") && (
                <button
                  type="button"
                  disabled={busy || running || d.air_gap}
                  onClick={() => setArmed({ what: "parameter-maps", only: "serving", label: "serving job filters & surgery baselines", files: null, bytes: null })}
                  className={`${btn} border-sense-pale text-sense-pale hover:text-text`}
                >
                  download serving maps &amp; filters
                </button>
              )}
              {!(armed && armed.what === "parameter-maps" && !armed.only) && (
                <button
                  type="button"
                  disabled={busy || running || d.air_gap}
                  onClick={() => setArmed({ what: "parameter-maps", label: "all official parameter maps", files: null, bytes: null })}
                  className={`${btn} border-line text-muted hover:text-text`}
                >
                  download all official maps
                </button>
              )}
            </div>
            {confirmFor("parameter-maps", "serving")}
            {confirmFor("parameter-maps")}
          </section>
          <section className="min-w-0">
            <div className="mono flex flex-wrap items-baseline justify-between gap-2 text-[10px] uppercase tracking-[0.12em]">
              <span className="flex items-center gap-1.5 text-muted" title="Verified non-base hardware and tuning stacks installed locally or available from Hugging Face.">
                <i aria-hidden className="inline-block h-1.5 w-1.5 bg-verified" />
                Shared stacks
              </span>
              <span className="normal-case tracking-normal text-quiet">{localStacks.length} installed</span>
            </div>
            <div className="mono mb-2 mt-0.5 text-[10px] text-faint">Hardware and tuning stacks imported on trial before activating.</div>
            {localStacks.length === 0 && remoteStacks.length === 0 ? (
              <div className="mono text-[10.5px] text-quiet">No shared stacks installed yet. Open the HF download window to browse stacks.</div>
            ) : (
              <ul className="grid">
                {localStacks.map((st) => {
                  const infoId = `stack:${st.key}`;
                  const shown = openInfo === infoId;
                  return (
                    <li key={st.key} className="grid gap-1 border-b border-line/60 py-1.5">
                      <div className="flex min-w-0 items-center justify-between gap-3">
                        <span className="mono min-w-0 truncate text-[11.5px] text-text" title={st.use}>
                          {st.key}
                        </span>
                        <span className="flex shrink-0 items-center gap-3">
                          <span className="mono text-[10px] text-faint">{st.use}</span>
                          <Pill tone="verified">installed</Pill>
                          <button type="button" onClick={() => toggleInfo(infoId)} aria-expanded={shown} className={infoBtn}>
                            {shown ? "hide" : "info"}
                          </button>
                        </span>
                      </div>
                      {shown && (
                        <div className="mono border-l border-line-strong pl-2.5 text-[10.5px] leading-relaxed text-faint">
                          {st.description || "Installed cartridge stack."}
                        </div>
                      )}
                    </li>
                  );
                })}
                {remoteStacks.map((rf) => {
                  const name = rf.path.split("/").pop() ?? rf.path;
                  const infoId = `rstack:${rf.path}`;
                  const shown = openInfo === infoId;
                  return (
                    <li key={rf.path} className="grid gap-1 border-b border-line/60 py-1.5">
                      <div className="flex min-w-0 items-center justify-between gap-3">
                        <span className="mono min-w-0 truncate text-[11.5px] text-muted" title={rf.path}>
                          {name}
                        </span>
                        <span className="flex shrink-0 items-center gap-3">
                          <span className="mono text-[10px] text-faint">{bytes(rf.bytes)}</span>
                          <Pill tone="muted">on HF</Pill>
                          <button type="button" onClick={() => toggleInfo(infoId)} aria-expanded={shown} className={infoBtn}>
                            {shown ? "hide" : "info"}
                          </button>
                        </span>
                      </div>
                      {shown && (
                        <div className="mono border-l border-line-strong pl-2.5 text-[10.5px] leading-relaxed text-faint">
                          {rf.description || "Verified non-base stack available on Hugging Face."}
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
          <section className="min-w-0">
            <div
              className="mono flex items-center gap-1.5 text-[10px] uppercase tracking-[0.12em] text-quiet"
              title="Optional Apache-2.0 Qwen3 open weights at pinned Hugging Face revisions. Not required to run Neuromorph Nano on base cartridges. Download only for open-weight fallback, output-parity reference checks, parameter mapping, or cartridge surgery."
            >
              <i aria-hidden className="inline-block h-1.5 w-1.5 rounded-full border border-quiet" />
              Upstream open weights · optional
            </div>
            <div className="mono mb-2 mt-0.5 text-[10px] text-faint">For fallback, reference parity, parameter mapping and surgery.</div>
            <ul className="grid">
              {(d.models ?? []).map((m) => (
                <li key={m.id} className="min-w-0 border-b border-line/60 py-1.5">
                  <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3">
                    <div className="min-w-0">
                      <div className="flex min-w-0 items-center gap-2">
                        <span className="mono truncate text-[11.5px] text-muted">{m.id}</span>
                        <Pill tone={m.state === "installed" ? "verified" : m.state === "partial" ? "attention" : "muted"}>{m.state === "installed" ? "installed" : m.state === "partial" ? "partly" : m.state}</Pill>
                      </div>
                      <div className="mono mt-0.5 truncate text-[10px] text-faint" title={`${m.repo} at ${m.revision}`}>
                        {m.present} of {m.files} files · {bytes(m.on_disk)}
                        {m.bytes ? ` of about ${bytes(m.bytes)}` : ""} · {m.repo} @ {m.revision.slice(0, 8)}
                      </div>
                    </div>
                    {m.state !== "installed" && !(armed && armed.what === `models:${m.id}`) && (
                      <button
                        type="button"
                        disabled={busy || running || d.air_gap}
                        onClick={() => setArmed({ what: `models:${m.id}`, label: `the missing files of ${m.id}`, files: Math.max(0, m.files - m.present), bytes: m.bytes ? Math.max(0, m.bytes - m.on_disk) : null })}
                        className={`${btn} border-sense-pale text-sense-pale hover:text-text`}
                      >
                        {m.partial ? "resume" : "download"}
                      </button>
                    )}
                  </div>
                  {confirmFor(`models:${m.id}`)}
                </li>
              ))}
            </ul>
          </section>
        </div>
      )}
    </section>
  );
}
