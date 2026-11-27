"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "also it doesnt let you control the micromodels base
// open source weights completely or show them on the ui for the micromodels"): on each MicroModel's
// card, what it runs on and the controls for it, in one place: the Neuromorph base cartridge serving
// it (choose the newest, a pinned version, or open mode on the upstream weights), and the upstream
// open weights it was built from (repository, pinned revision, licence, files on disk, download or
// resume with every file checked, and its tensors). The same routes as Cartridges, Base versions and
// Models, Downloads, so a change here is a change there, logged in Audit.
//
// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden" and "the
// oipen weights used is not the micromodel. that should be clearly diferentiated but not overyly
// through text"): the explanations moved into hover titles; the two parts stay apart by their marks
// (filled square: the base cartridge it runs on; hollow ring: the upstream open weights it was built
// from). Inside the Models page, "downloads" opens the Downloads tab of the side panel in place.

import Link from "next/link";
import { useState } from "react";
import { Pill, Segmented } from "@/components/ui";
import { bytes, post, refreshAll, useApi } from "@/lib/api";
import { BaseMark, WeightsMark } from "@/components/MicroModel";
import { useRail } from "@/components/Workspace";

type Version = { key: string; version: number; bytes: number; present: boolean; refused: string | null; serving: boolean };
type BaseModel = { model: string; serving: string; choice: string; versions: Version[] };
type Bases = { models?: BaseModel[] };
type OpenRow = { id: string; repo: string; revision: string; bytes: number; files: number; present: number; partial: number; on_disk: number; state: string };
type Downloads = { models?: OpenRow[]; air_gap?: boolean; job?: { running?: boolean } | null };
type Note = { text: string; tone: "verified" | "error" };

const btn = "press mono border-b px-0.5 text-[10px] uppercase tracking-[0.12em] disabled:opacity-40";

export default function MicroModelWeights({ model, onNote }: { model: string; onNote: (n: Note) => void }) {
  const bases = useApi<Bases>("/api/cartridges/bases", 15000);
  const dl = useApi<Downloads>("/api/downloads", 5000);
  const [busy, setBusy] = useState(false);
  const [armed, setArmed] = useState(false);
  const rail = useRail();
  // Inside a Workspace the rail has a tab picked; outside one (no provider) it has none.
  const inRail = rail.active !== "";
  const b = bases.data?.models?.find((x) => x.model === model);
  const w = dl.data?.models?.find((x) => x.id === model);
  const choose = async (choice: string) => {
    setBusy(true);
    try {
      await post("/api/cartridges/bases", { model, choice });
      for (const p of ["/api/cartridges/bases", "/api/cartridges/library", "/api/audit"]) refreshAll(p);
      onNote({ text: choice === "auto" ? "The newest usable base cartridge serves from the next start. Logged in Audit, with undo." : choice === "open" ? "Open mode: this MicroModel serves the upstream open weights from the next start. Logged in Audit, with undo." : `${choice} serves from the next start. Logged in Audit, with undo.`, tone: "verified" });
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  const download = async () => {
    setBusy(true);
    try {
      const r = await post<{ line?: string }>("/api/downloads", { what: `models:${model}` });
      refreshAll("/api/downloads");
      onNote({ text: `${r.line ?? "Download started"}; every file is checked against its SHA-256 before it is kept.`, tone: "verified" });
      setArmed(false);
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  const choices = b ? ["auto", ...b.versions.filter((v) => v.present && !v.refused).map((v) => v.key), "open"] : [];
  const missing = w ? Math.max(0, w.files - w.present) : 0;
  return (
    <div className="mt-3 grid gap-2.5 border-t border-line pt-3" data-e2e="micromodel-weights">
      <div className="eyebrow" title="What serves this MicroModel: its Neuromorph base cartridge (choose the newest, a pinned version, or open mode on the upstream weights), and the upstream open weights it was built from. Changes apply at the next start and are logged in Audit, with undo.">
        Runs on
      </div>
      <div className="grid gap-1.5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <BaseMark base={b?.serving} />
          {b && (
            <Link href="/cartridges/#bases" className="link mono text-[10px]">
              versions
            </Link>
          )}
        </div>
        {b ? (
          <Segmented label={`What serves the MicroModel built from ${model}`} values={choices} value={choices.includes(b.choice) ? b.choice : "auto"} busy={busy} render={(c) => (c === "auto" ? "newest" : c === "open" ? "open mode" : c.replace(/^.*@/, "v"))} onPick={(c) => void choose(c)} />
        ) : (
          <span className="mono text-[10.5px] text-faint">{bases.error ? "base cartridges not reported" : "reading"}</span>
        )}
      </div>
      <div className="grid gap-1">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <WeightsMark model={model} />
          {w && <Pill tone={w.state === "installed" ? "verified" : w.state === "partial" ? "attention" : "muted"}>{w.state === "installed" ? "on disk" : w.state === "partial" ? "partly" : "not on disk"}</Pill>}
        </div>
        {w ? (
          <>
            <span
              className="mono truncate text-[10px] text-faint"
              title={`${w.repo} at ${w.revision}, Apache-2.0. Optional: Neuromorph Nano runs standalone on its base cartridges (plus the small tokenizer and config JSONs). Download the full upstream open weights for fallback (open mode), output-parity reference checks, parameter mapping, or cartridge surgery. Every file is checked against its SHA-256 before it is kept.`}
            >
              {w.present} of {w.files} files · {bytes(w.on_disk)}
              {w.bytes && w.on_disk < w.bytes ? ` of ${bytes(w.bytes)}` : ""} · {w.repo} @ {w.revision.slice(0, 8)}
            </span>
            <span className="mono flex flex-wrap items-center gap-x-4 text-[10.5px]">
              {w.state !== "installed" &&
                (armed ? (
                  <>
                    <span className="text-text">
                      fetch {missing} {missing === 1 ? "file" : "files"}
                      {w.bytes ? `, about ${bytes(Math.max(0, w.bytes - w.on_disk))}` : ""}?
                    </span>
                    <button type="button" autoFocus disabled={busy || dl.data?.air_gap || !!dl.data?.job?.running} onClick={() => void download()} className={`${btn} border-sense-pale text-sense-pale hover:text-text`}>
                      confirm
                    </button>
                    <button type="button" onClick={() => setArmed(false)} className={`${btn} border-line text-faint hover:text-text`}>
                      cancel
                    </button>
                  </>
                ) : (
                  <button type="button" disabled={busy || dl.data?.air_gap || !!dl.data?.job?.running} onClick={() => setArmed(true)} className={`${btn} border-sense-pale text-sense-pale hover:text-text`} title={dl.data?.air_gap ? "the air gap is on" : undefined}>
                    {w.partial ? "resume download" : "download"}
                  </button>
                ))}
              {inRail ? (
                <button type="button" onClick={() => rail.open("downloads")} className="link">
                  all downloads
                </button>
              ) : (
                <Link href="/models/#downloads" className="link">
                  all downloads
                </Link>
              )}
            </span>
          </>
        ) : (
          <span className="mono text-[10.5px] text-faint">{dl.error ? "downloads not reported" : "reading"}</span>
        )}
      </div>
    </div>
  );
}
