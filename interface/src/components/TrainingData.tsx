"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And
// that whole information database for it should be on a right side panel."): this panel now sits
// in a rail tab, one column; its explanation is one faint line, the rest in tooltips.
// 2026-11-25 (Luis Jake Gabriel III: "Allow users to prep their own air gapped data for their own
// training"). The training store to a file and back (to carry it to an air-gapped machine; the
// file's SHA-256 is checked and nothing is added twice), and the user's own reference material
// listed with its licence and hash. Nothing is copied or sent anywhere; all of it works under the
// air gap. The same actions are in MCP (nm_learning export, import, references, reference_add).

import { useState } from "react";
import { NoteLine } from "@/components/Controls";
import { Panel, Pill, Placeholder } from "@/components/ui";
import { post, refreshAll, useApi } from "@/lib/api";

type Ref = { id: string; source?: string; path?: string; kind?: string; files?: number; licence?: string; domain?: string; sha256?: string; fitted?: string[]; present?: boolean };
type Refs = { references?: Ref[]; registry?: string };
type Note = { text: string; tone: "verified" | "error" };

const btn = "press mono min-h-11 border-b px-1 text-[11px] uppercase tracking-[0.12em] disabled:opacity-40";
const field = "mono h-10 min-w-0 border-b border-line-strong bg-transparent px-1 text-xs text-text placeholder:text-quiet focus-visible:border-sense-pale";

export function TrainingDataPanel({ projectKey }: { projectKey: string | null }) {
  const refs = useApi<Refs>("/api/training/references", 30000);
  const [note, setNote] = useState<Note | null>(null);
  const [busy, setBusy] = useState("");
  const [file, setFile] = useState("");
  const [path, setPath] = useState("");
  const [licence, setLicence] = useState("");
  const [source, setSource] = useState("");

  const run = async (what: string, fn: () => Promise<string>) => {
    setBusy(what);
    try {
      setNote({ text: await fn(), tone: "verified" });
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy("");
  };
  const exportStore = () =>
    run("export", async () => {
      const r = await post<{ file: string; rows: number }>("/api/training/export", { project: projectKey });
      refreshAll("/api/audit");
      return `Exported ${r.rows} rows to ${r.file}. Carry the file to the other machine and import it there. Logged in Audit.`;
    });
  const importStore = () =>
    run("import", async () => {
      const r = await post<{ rows: number }>("/api/training/import", { project: projectKey, file: file.trim() });
      refreshAll("/api/examples");
      refreshAll("/api/learning");
      refreshAll("/api/audit");
      setFile("");
      return `Imported ${r.rows} new rows (rows already here are passed over). Logged in Audit.`;
    });
  const addRef = () =>
    run("ref", async () => {
      const r = await post<{ id: string }>("/api/training/references", { path: path.trim(), licence: licence.trim(), source: source.trim() || undefined });
      refreshAll("/api/training/references");
      refreshAll("/api/audit");
      setPath("");
      setLicence("");
      setSource("");
      return `Listed ${r.id}: hashed where it is; nothing was copied. Logged in Audit.`;
    });

  const list = refs.data?.references ?? [];
  return (
    <Panel title="Your training data" right={<Pill tone="verified">works under the air gap</Pill>}>
      <p
        className="mono mb-4 text-[10.5px] text-faint"
        title="Prepare training data on this machine and keep it here. Export the training store to a file to carry it to an air-gapped machine, import one there, and list your own reference material with its licence. Every file is checked by its hash; nothing is sent anywhere."
      >
        carry the store to and from an air-gapped machine; every file checked by its hash
      </p>
      <NoteLine note={note} />
      <div className="grid gap-6">
        <section className="min-w-0" data-e2e="training-transfer">
          <div className="eyebrow mb-2">Training store to and from a file</div>
          <div className="flex flex-wrap items-center gap-3">
            <button type="button" disabled={!!busy || !projectKey} onClick={() => void exportStore()} className={`${btn} border-line-strong text-muted hover:text-text`}>
              {busy === "export" ? "Exporting" : "Export to a file"}
            </button>
            <span className="text-xs text-faint">written to the exports folder of the Nano</span>
          </div>
          <div className="mt-3 flex min-w-0 flex-wrap items-center gap-3">
            <label className="min-w-0 flex-1">
              <span className="sr-only">Export file to import</span>
              <input value={file} onChange={(e) => setFile(e.target.value)} placeholder="path of an export file on this machine" className={`${field} w-full`} />
            </label>
            <button type="button" disabled={!!busy || !projectKey || !file.trim()} onClick={() => void importStore()} className={`${btn} border-sense-pale text-sense-pale hover:text-text`}>
              {busy === "import" ? "Importing" : "Import"}
            </button>
          </div>
        </section>
        <section className="min-w-0" data-e2e="training-references">
          <div className="eyebrow mb-2 flex flex-wrap items-baseline justify-between gap-2">
            <span>Your reference material</span>
            <span className="mono text-[11px] normal-case tracking-normal text-faint">{refs.data ? `${list.length} listed` : ""}</span>
          </div>
          {refs.error && !refs.data ? (
            <div className="text-xs text-error">{refs.error}</div>
          ) : !refs.data ? (
            <Placeholder className="h-16" what="training data" />
          ) : list.length === 0 ? (
            <div className="text-xs text-faint">Nothing listed yet.</div>
          ) : (
            <ul className="grid gap-1.5">
              {list.map((r) => (
                <li key={r.id} className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-3 border-b border-line/60 py-1.5" title={r.path}>
                  <span className="min-w-0">
                    <span className="mono block truncate text-[11.5px] text-muted">
                      {r.source || r.id} · {r.licence}
                    </span>
                    <span className="mono block truncate text-[10.5px] text-faint">
                      {r.kind} · {r.files ?? 0} {r.files === 1 ? "file" : "files"} · {(r.sha256 ?? "").slice(0, 12)}
                      {r.fitted?.length ? ` · fitted: ${r.fitted.join(", ")}` : ""}
                    </span>
                  </span>
                  <Pill tone={r.present ? "verified" : "attention"}>{r.present ? "in place" : "moved or gone"}</Pill>
                </li>
              ))}
            </ul>
          )}
          <div className="mt-3 grid gap-2">
            <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="a file or folder on this machine" aria-label="Path of the material" className={field} />
            <div className="flex min-w-0 flex-wrap items-center gap-3">
              <input value={licence} onChange={(e) => setLicence(e.target.value)} placeholder="licence (MIT, CC-BY-4.0, own)" aria-label="Licence" className={`${field} w-40`} />
              <input value={source} onChange={(e) => setSource(e.target.value)} placeholder="name (optional)" aria-label="Name of the material" className={`${field} min-w-0 flex-1`} />
              <button type="button" disabled={!!busy || !path.trim() || !licence.trim()} onClick={() => void addRef()} className={`${btn} border-sense-pale text-sense-pale hover:text-text`}>
                {busy === "ref" ? "Listing" : "List it"}
              </button>
            </div>
          </div>
        </section>
      </div>
    </Panel>
  );
}
