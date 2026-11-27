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
import { Pill, Placeholder } from "@/components/ui";
import { bytes, post, refreshAll, useApi } from "@/lib/api";

type ModelRow = { id: string; repo: string; revision: string; bytes: number; files: number; present: number; partial: number; on_disk: number; state: string };
type CartRow = { key: string; present: boolean; on_disk: number; path: string };
/** `stopped`: set when the process running the download ended before it finished (then
 *  `running` is false). */
type Job = { what: string; running: boolean; done: number; total: number; current: string; current_bytes: number; failed: string[]; finished_ms: number; by?: string; updated_ms?: number; stopped?: string };
type Payload = { models?: ModelRow[]; cartridges?: CartRow[]; cartridge_repo?: string; token?: boolean; air_gap?: boolean; job?: Job | null };
type Note = { text: string; tone: "verified" | "error" };

const btn = "press mono border-b px-0.5 text-[10px] uppercase tracking-[0.12em] disabled:opacity-40";

/** A download waiting for its confirmation: what it fetches, in words and figures. */
type Armed = { what: string; label: string; files: number; bytes: number | null };

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
        {armed.files} {armed.files === 1 ? "file" : "files"}
        {armed.bytes !== null ? ` · about ${bytes(armed.bytes)}` : " · each size is shown as the file arrives"} · every file is checked against its SHA-256 before it is kept; a partial file resumes
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

export function DownloadsPanel({ onNote }: { onNote: (n: Note) => void }) {
  const q = useApi<Payload>("/api/downloads", 2000);
  const [busy, setBusy] = useState(false);
  const [armed, setArmed] = useState<Armed | null>(null);
  const d = q.data;
  const job = d?.job;
  const running = !!job?.running;
  const start = async (a: Armed) => {
    setBusy(true);
    try {
      const r = await post<{ line?: string }>("/api/downloads", { what: a.what });
      refreshAll("/api/downloads");
      onNote({ text: `${r.line ?? "Download started"}.`, tone: "verified" });
      setArmed(null);
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  const missingCarts = (d?.cartridges ?? []).filter((c) => !c.present);
  const confirmFor = (what: string) =>
    armed && armed.what === what ? <ConfirmRow armed={armed} busy={busy} onConfirm={() => void start(armed)} onCancel={() => setArmed(null)} /> : null;
  return (
    <section className="min-w-0" aria-label="Downloads">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <span className="eyebrow text-muted" title="Base cartridges are required to run (downloading them also fetches the small tokenizer and config JSONs). Upstream open weights are optional: used for fallback, output-parity reference checks, parameter mapping, and cartridge surgery. Every file is checked against its SHA-256 before it is kept; a partial file resumes.">
          Downloads
        </span>
        {d?.air_gap && <Pill tone="attention">air gap on: no downloads</Pill>}
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
                title={`The Neuromorph base cartridges the MicroModels run on standalone, from ${d.cartridge_repo ?? "the cartridge repository"} (also fetches the small tokenizer and config JSONs). You can also load or contribute your own cartridges and parameter maps on Cartridges and Parameters.`}
              >
                <i aria-hidden className="inline-block h-1.5 w-1.5 bg-learned" />
                Base cartridges · required
              </span>
              <span className={`normal-case tracking-normal ${d.token ? "text-verified" : "text-quiet"}`} title={d.token ? "A Hugging Face token was found on this machine" : "A private repository needs a Hugging Face token"}>
                {d.token ? "token found" : "no token"}
              </span>
            </div>
            <div className="mono mb-2 mt-0.5 text-[10px] text-faint">Runs all four MicroModels standalone.</div>
            <ul className="grid">
              {(d.cartridges ?? []).map((c) => (
                <li key={c.key} className="flex min-w-0 items-center justify-between gap-3 border-b border-line/60 py-1.5">
                  <span className="mono min-w-0 truncate text-[11.5px] text-text" title={c.path}>
                    {c.key}
                  </span>
                  <span className="flex shrink-0 items-center gap-3">
                    <span className="mono text-[10.5px] text-faint">{c.present ? bytes(c.on_disk) : "not on disk"}</span>
                    <Pill tone={c.present ? "verified" : "muted"}>{c.present ? "in the cartridge library" : "missing"}</Pill>
                  </span>
                </li>
              ))}
            </ul>
            {missingCarts.length > 0 && !(armed && armed.what === "cartridges") && (
              <button
                type="button"
                disabled={busy || running || d.air_gap}
                onClick={() => setArmed({ what: "cartridges", label: "the missing base cartridges", files: missingCarts.length, bytes: null })}
                className={`${btn} mt-3 border-sense-pale text-sense-pale hover:text-text`}
              >
                download the missing base cartridges
              </button>
            )}
            {confirmFor("cartridges")}
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
