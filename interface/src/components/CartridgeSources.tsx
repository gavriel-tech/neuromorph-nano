"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "Users should be able to connect form our official
// hugging face cartridge libraries or 3rd parties along with the open weightsz obviously our fallback
// openweights are the Qwen open weights."): the cartridge libraries the Nano can take cartridges
// from. The official Neuromorph library is built in; add any third-party library on Hugging Face by
// its repository name. Browse lists its files (one request, only when asked); fetch downloads one
// file and installs it through the library's own checks (a stack arrives on trial; a map never
// replaces one). The Qwen3 open weights stay the fallback, in Downloads above.
//
// 2026-11-26 (Luis Jake Gabriel III, his words: "the right menu are libraries and or discription
// tools etc."): the libraries are a tab of the Models side panel: one narrow column, the long
// explanation folded into a hover title. The official public library is not live yet, and the
// official source says so instead of claiming it.

import { useState } from "react";
import DeleteFlow from "@/components/DeleteFlow";
import { NoteLine } from "@/components/Controls";
import { Empty, Pill, Placeholder } from "@/components/ui";
import { ago, bytes, post, refreshAll, useApi } from "@/lib/api";
import { useNow } from "@/lib/clock";

type Note = { text: string; tone: "verified" | "error" };
type SourceFile = { path: string; bytes: number; sha256: string | null; kind: string; role: string | null; installable: boolean; why_not: string | null };
type Listing = { at_ms: number; revision: string; files: SourceFile[]; truncated: boolean };
type Source = { id: string; repo: string; kind: string; revision: string; official: boolean; token: boolean; added_ms: number; url: string; listing: Listing | null };
type Reply = { sources: Source[]; air_gap: boolean; token_found: boolean; fallback: string; file: string };

const link = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const go = "press mono border-b border-sense-pale px-0.5 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40";
const field = "mono h-8 min-w-0 border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale";

function SourceBlock({ src, airGap, onNote }: { src: Source; airGap: boolean; onNote: (n: Note) => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [all, setAll] = useState(false);
  const [armed, setArmed] = useState<string | null>(null);
  const now = useNow();
  const browse = async () => {
    setBusy("browse");
    try {
      const r = await post<{ listing: Listing }>("/api/cartridge-sources", { action: "browse", id: src.id });
      onNote({ text: `${src.repo}: ${r.listing.files.length} files listed, ${r.listing.files.filter((f) => f.installable).length} the Nano can install.`, tone: "verified" });
      refreshAll("/api/cartridge-sources");
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  const fetchFile = async (f: SourceFile) => {
    setBusy(f.path);
    try {
      const r = await post<{ line: string }>("/api/cartridge-sources", { action: "fetch", id: src.id, path: f.path });
      onNote({ text: `${r.line}. Its progress shows in Downloads; it installs only if it passes the checks.`, tone: "verified" });
      setArmed(null);
      for (const p of ["/api/downloads", "/api/cartridges", "/api/parameter-maps"]) refreshAll(p);
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  const files = src.listing?.files ?? [];
  const shown = all ? files : files.filter((f) => f.installable);
  return (
    <li className="grid gap-2 border-b border-line/60 py-3" data-e2e="cartridge-source">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="mono truncate text-[12px] text-text" title={src.url}>{src.repo}</span>
          <Pill tone={src.official ? "verified" : "muted"}>{src.official ? "official" : "third party"}</Pill>
          <span className="mono text-[10.5px] text-faint">
            {src.kind} · {src.revision} · {src.token ? "sends your token" : "no token sent"}
          </span>
          {src.official && (
            <span className="mono basis-full text-[10px] text-quiet" title="The official public cartridge library is not live yet; until it is, the official source is the repository the base cartridges are built and kept in.">
              the official public library is not live yet
            </span>
          )}
        </span>
        <span className="flex items-center gap-4">
          <button type="button" disabled={busy !== null || airGap} onClick={() => void browse()} className={go}>
            {busy === "browse" ? "listing" : src.listing ? "browse again" : "browse"}
          </button>
          {!src.official && (
            <DeleteFlow<{ audit_id?: string }>
              path="/api/cartridge-sources"
              body={{ action: "remove", id: src.id }}
              label="remove source"
              title="Asks first; what was imported from it stays in the cartridge library on this machine, and Audit can add the source again"
              startClass="press mono border-b border-line px-0.5 text-[10px] uppercase tracking-[0.1em] text-faint hover:text-error disabled:opacity-40"
              done={() => `Source ${src.repo} removed; what was imported from it stays. Audit can add it again.`}
              refresh={["/api/cartridge-sources"]}
              onNote={onNote}
            />
          )}
        </span>
      </div>
      {src.listing && (
        <div className="grid gap-1.5">
          <div className="mono flex flex-wrap items-center gap-x-4 text-[10.5px] text-faint">
            <span>
              listed {ago(src.listing.at_ms, now)} · {files.length} files · {files.filter((f) => f.installable).length} installable{src.listing.truncated ? " · the first files only" : ""}
            </span>
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />
              show every file
            </label>
          </div>
          {shown.length === 0 ? (
            <div className="mono text-[11px] text-faint">{files.length ? "Nothing here the Nano installs; show every file to see what it holds." : "The repository holds no files."}</div>
          ) : (
            <ul className="grid max-h-[40vh] overflow-auto">
              {shown.map((f) => (
                <li key={f.path} className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3 border-t border-line/40 py-1.5">
                  <span className="min-w-0">
                    <span className="mono block truncate text-[11px] text-muted" title={f.path}>{f.path}</span>
                    <span className="mono text-[10px] text-faint">
                      {f.kind}{f.role ? ` for the ${f.role}` : ""} · {bytes(f.bytes)} · {f.sha256 ? `SHA-256 ${f.sha256.slice(0, 12)}` : "no published SHA-256 (its own checks decide)"}
                    </span>
                    {armed === f.path && (
                      <span className="mono mt-1 flex flex-wrap items-center gap-3 text-[10.5px]">
                        <span className="text-text">Fetch {f.path} from {src.repo}{f.kind === "stack file" ? "; it joins the cartridge library on this machine, on trial" : f.role ? ` as a ${f.role} map; it never replaces one` : ""}?</span>
                        <button type="button" autoFocus disabled={busy !== null} onClick={() => void fetchFile(f)} className={go}>
                          confirm fetch
                        </button>
                        <button type="button" onClick={() => setArmed(null)} className={link}>
                          cancel
                        </button>
                      </span>
                    )}
                  </span>
                  {f.installable ? (
                    armed !== f.path && (
                      <button type="button" disabled={busy !== null || airGap} onClick={() => setArmed(f.path)} className={go}>
                        fetch
                      </button>
                    )
                  ) : (
                    <span className="mono max-w-[9rem] text-right text-[10px] text-quiet">{f.why_not}</span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </li>
  );
}

export default function CartridgeSources() {
  const q = useApi<Reply>("/api/cartridge-sources", 15000);
  const [note, setNote] = useState<Note | null>(null);
  const [repo, setRepo] = useState("");
  const [kind, setKind] = useState("model");
  const [revision, setRevision] = useState("");
  const [token, setToken] = useState(false);
  const [busy, setBusy] = useState(false);
  const d = q.data;
  const add = async () => {
    setBusy(true);
    try {
      const r = await post<{ audit_id: string }>("/api/cartridge-sources", { action: "add", repo: repo.trim(), kind, revision: revision.trim(), token });
      setNote({ text: `Source ${repo.trim()} added; browse it to see what it holds. Audit ${r.audit_id} undoes it.`, tone: "verified" });
      setRepo("");
      setRevision("");
      setToken(false);
      refreshAll("/api/cartridge-sources");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  return (
    <section className="min-w-0" aria-label="Cartridge libraries">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <span
          className="eyebrow text-muted"
          title="Take cartridges from the official Neuromorph library or from any third-party library on Hugging Face. Nothing is fetched until you ask: browse lists a library's files, fetch downloads one and installs it through the same checks as ours (a stack joins on trial until its evidence lets it serve; a parameter map never replaces one of yours). A library shares stacks under stacks/ (or as name.stack.json) and maps under parameter-maps/<role>/. The fallback stays the Qwen3 open weights, in Downloads."
        >
          Cartridge libraries · official and third party
        </span>
        {d && (
          <span className="mono text-[10.5px] text-faint">
            {d.sources.length} {d.sources.length === 1 ? "source" : "sources"}
          </span>
        )}
      </div>
      <NoteLine note={note} />
      {!d && q.error ? (
        <Empty title="The sources did not answer">{q.error}</Empty>
      ) : !d ? (
        <Placeholder className="h-20" what="sources" />
      ) : (
        <>
          {d.air_gap && <p className="mono mb-2 text-[11px] text-attention">The air gap is on: no library is browsed or fetched until it is turned off.</p>}
          <ul className="grid" data-e2e="cartridge-sources">
            {d.sources.map((src) => (
              <SourceBlock key={src.id} src={src} airGap={d.air_gap} onNote={setNote} />
            ))}
          </ul>
          <form
            className="mt-4 grid gap-2"
            data-e2e="source-add"
            onSubmit={(e) => {
              e.preventDefault();
              if (repo.trim()) void add();
            }}
          >
            <span className="eyebrow">Add a third-party library</span>
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <input value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="owner/name on Hugging Face" aria-label="Repository" className={`${field} w-full basis-full`} />
              <select value={kind} onChange={(e) => setKind(e.target.value)} aria-label="Repository kind" className={field}>
                <option value="model" className="bg-bg-1">model repository</option>
                <option value="dataset" className="bg-bg-1">dataset repository</option>
              </select>
              <input value={revision} onChange={(e) => setRevision(e.target.value)} placeholder="main" aria-label="Revision" className={`${field} w-28`} />
              <button type="submit" disabled={busy || !repo.trim()} className={go}>
                {busy ? "adding" : "add"}
              </button>
            </div>
            <label className="mono flex items-start gap-2 text-[10.5px] text-faint">
              <input type="checkbox" checked={token} onChange={(e) => setToken(e.target.checked)} className="mt-0.5" />
              <span>
                send my Hugging Face token for this library (only for a private or gated one you trust; sent only to huggingface.co){d.token_found ? "" : " · no token is set on this machine"}
              </span>
            </label>
          </form>
          <p className="mono mt-3 text-[10.5px] text-quiet">Fallback: {d.fallback}.</p>
        </>
      )}
    </section>
  );
}
