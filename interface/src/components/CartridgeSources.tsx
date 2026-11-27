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
// explanation folded into a hover title. Users can browse and fetch parameter maps and stacks from
// the official public Hugging Face library or third-party repositories, and publish their own
// parameter maps or stacks back to Hugging Face (their own repo or an official Pull Request).

import { useEffect, useRef, useState } from "react";
import DeleteFlow from "@/components/DeleteFlow";
import { NoteLine } from "@/components/Controls";
import { Empty, Pill, Placeholder } from "@/components/ui";
import { ago, bytes, post, refreshAll, useApi } from "@/lib/api";
import { useNow } from "@/lib/clock";

type Note = { text: string; tone: "verified" | "error" };
type SourceFile = { path: string; bytes: number; sha256: string | null; kind: string; role: string | null; description?: string | null; installable: boolean; why_not: string | null; present?: boolean };
type Listing = { at_ms: number; revision: string; files: SourceFile[]; truncated: boolean };
type Source = { id: string; repo: string; kind: string; revision: string; official: boolean; token: boolean; added_ms: number; url: string; listing: Listing | null };
type LocalMap = { role: string; file: string; bytes: number; direct_ok: boolean; description?: string };
type LocalStack = { key: string; use: string; description: string };
type Publishable = { maps: LocalMap[]; stacks: LocalStack[] };
type Reply = { sources: Source[]; air_gap: boolean; token_found: boolean; fallback: string; file: string; publishable?: Publishable; publish_limit_bytes?: number };
type DlJob = { what: string; running: boolean; done: number; total: number; current: string; current_bytes: number; failed: string[]; finished_ms: number; by?: string; stopped?: string };
type DlReply = { job?: DlJob | null };

const link = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const go = "press mono border-b border-sense-pale px-0.5 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40";
const field = "mono h-8 min-w-0 border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale";

function SourceBlock({
  src,
  airGap,
  dlRunning,
  queuedPaths,
  onQueueFiles,
  onNote,
}: {
  src: Source;
  airGap: boolean;
  dlRunning: boolean;
  queuedPaths: Set<string>;
  onQueueFiles: (sourceId: string, repo: string, paths: string[]) => void;
  onNote: (n: Note) => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [all, setAll] = useState(false);
  const [cat, setCat] = useState<"all" | "filters" | "maps" | "stacks">("all");
  const [armed, setArmed] = useState<string | null>(null);
  const [armedAll, setArmedAll] = useState(false);
  const [armedSelected, setArmedSelected] = useState(false);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [openInfo, setOpenInfo] = useState<string | null>(null);
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
      onNote({ text: `${r.line}. Watch live progress above; it installs only if it passes the checks.`, tone: "verified" });
      setArmed(null);
      for (const p of ["/api/downloads", "/api/cartridges", "/api/parameter-maps", "/api/cartridge-sources"]) refreshAll(p);
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  const fetchAllInstallable = async () => {
    setBusy("fetch_all");
    try {
      const r = await post<{ line: string }>("/api/cartridge-sources", { action: "fetch_all", id: src.id });
      onNote({ text: `${r.line}. Watch live progress above.`, tone: "verified" });
      setArmedAll(false);
      for (const p of ["/api/downloads", "/api/cartridges", "/api/parameter-maps", "/api/cartridge-sources"]) refreshAll(p);
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  const files = (() => {
    const raw = src.listing?.files ?? [];
    const best = new Map<string, { ver: number; file: SourceFile }>();
    for (const f of raw) {
      if (f.path.split("/").some((seg) => { const s = seg.toLowerCase(); return s === "archived" || s === "archive"; })) continue;
      const dir = f.path.includes("/") ? f.path.slice(0, f.path.lastIndexOf("/")).toLowerCase() : "";
      const name = f.path.split("/").pop() ?? f.path;
      const mV = /^(.+)-v(\d+)(\.[a-z0-9.]+)?$/i.exec(name);
      const base = (mV ? `${mV[1]}${mV[3] ?? ""}` : name).toLowerCase();
      const ver = mV ? Number(mV[2]) : 0;
      const key = `${dir}/${base}`;
      const prev = best.get(key);
      if (!prev || ver > prev.ver) best.set(key, { ver, file: f });
    }
    return [...best.values()].map((x) => x.file);
  })();
  const installableFiles = files.filter((f) => f.installable);
  const installableCount = installableFiles.length;
  const isFilterFile = (f: SourceFile) => f.path.split("/").pop()?.startsWith("job-filter-") ?? false;
  const isStackFile = (f: SourceFile) => f.kind === "stack file";
  const isMapFile = (f: SourceFile) => f.kind === "parameter map" && !isFilterFile(f);
  const baseShown = all ? files : installableFiles;
  const shown = baseShown.filter((f) => {
    if (cat === "filters") return isFilterFile(f);
    if (cat === "maps") return isMapFile(f);
    if (cat === "stacks") return isStackFile(f);
    return true;
  });
  const catCounts = {
    all: baseShown.length,
    filters: baseShown.filter(isFilterFile).length,
    maps: baseShown.filter(isMapFile).length,
    stacks: baseShown.filter(isStackFile).length,
  };
  const visibleInstallable = shown.filter((f) => f.installable);
  const checkedFiles = installableFiles.filter((f) => selected[f.path]);
  const checkedBytes = checkedFiles.reduce((sum, f) => sum + (f.bytes || 0), 0);
  const toggleOne = (path: string) => setSelected((prev) => ({ ...prev, [path]: !prev[path] }));
  const selectVisibleInstallable = () => {
    setSelected((prev) => {
      const next = { ...prev };
      for (const f of visibleInstallable) next[f.path] = true;
      return next;
    });
  };
  const clearSelection = () => {
    setSelected({});
    setArmedSelected(false);
  };
  return (
    <li className="grid gap-2 border-b border-line/60 py-3" data-e2e="cartridge-source">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="mono truncate text-[12px] text-text" title={src.url}>{src.repo}</span>
          <Pill tone={src.official ? "verified" : "muted"}>{src.official ? "official" : "third party"}</Pill>
          <span className="mono text-[10.5px] text-faint">
            {src.kind} · {src.revision} · {src.token ? "sends your token" : "public"}
          </span>
        </span>
        <span className="flex flex-wrap items-center gap-3">
          <button type="button" disabled={busy !== null || airGap} onClick={() => void browse()} className={go}>
            {busy === "browse" ? "listing" : src.listing ? "browse again" : "browse"}
          </button>
          {checkedFiles.length > 0 && !armedSelected && (
            <button type="button" disabled={busy !== null || airGap || dlRunning} onClick={() => setArmedSelected(true)} className={go}>
              fetch selected ({checkedFiles.length})
            </button>
          )}
          {installableCount > 0 && !armedAll && (
            <button type="button" disabled={busy !== null || airGap || dlRunning} onClick={() => setArmedAll(true)} className={go}>
              fetch all ({installableCount})
            </button>
          )}
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
      {armedSelected && checkedFiles.length > 0 && (
        <div className="mono flex flex-wrap items-center gap-3 border-l border-sense-pale pl-3 text-[10.5px]">
          <span className="text-text">
            Import {checkedFiles.length} selected {checkedFiles.length === 1 ? "file" : "files"} ({bytes(checkedBytes)}) from {src.repo} (checked before keeping; existing local maps are never replaced)?
          </span>
          <button
            type="button"
            autoFocus
            disabled={busy !== null || dlRunning}
            onClick={() => {
              onQueueFiles(src.id, src.repo, checkedFiles.map((f) => f.path));
              clearSelection();
            }}
            className={go}
          >
            confirm fetch selected
          </button>
          <button type="button" onClick={() => setArmedSelected(false)} className={link}>
            cancel
          </button>
        </div>
      )}
      {armedAll && (
        <div className="mono flex flex-wrap items-center gap-3 border-l border-sense-pale pl-3 text-[10.5px]">
          <span className="text-text">Fetch all {installableCount} installable files from {src.repo} (existing local maps are never replaced)?</span>
          <button type="button" autoFocus disabled={busy !== null || dlRunning} onClick={() => void fetchAllInstallable()} className={go}>
            {busy === "fetch_all" ? "starting" : "confirm fetch all"}
          </button>
          <button type="button" onClick={() => setArmedAll(false)} className={link}>
            cancel
          </button>
        </div>
      )}
      {src.listing && (
        <div className="grid gap-1.5">
          <div className="mono flex flex-wrap items-center gap-x-3 gap-y-1 text-[10.5px] text-faint">
            <span>
              listed {ago(src.listing.at_ms, now)} · {files.length} files · {installableCount} installable{src.listing.truncated ? " · the first files only" : ""}
            </span>
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />
              show every file
            </label>
          </div>
          {baseShown.length > 0 && (
            <div className="mono flex flex-wrap items-center justify-between gap-2 text-[9.5px]">
              <span className="flex flex-wrap items-center gap-1">
                {(
                  [
                    ["all", `all (${catCounts.all})`],
                    ["filters", `job filters (${catCounts.filters})`],
                    ["maps", `maps (${catCounts.maps})`],
                    ["stacks", `stacks (${catCounts.stacks})`],
                  ] as const
                ).map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setCat(id)}
                    className={`press border px-1.5 py-0.5 uppercase tracking-[0.08em] ${cat === id ? "border-sense-pale bg-sense/15 text-text" : "border-line text-faint hover:text-text"}`}
                  >
                    {label}
                  </button>
                ))}
              </span>
              {visibleInstallable.length > 0 && (
                <span className="flex items-center gap-2">
                  <button type="button" onClick={selectVisibleInstallable} className={link}>
                    select {cat === "all" ? "all" : cat} ({visibleInstallable.length})
                  </button>
                  {checkedFiles.length > 0 && (
                    <button type="button" onClick={clearSelection} className={link}>
                      clear ({checkedFiles.length} · {bytes(checkedBytes)})
                    </button>
                  )}
                </span>
              )}
            </div>
          )}
          {shown.length === 0 ? (
            <div className="mono text-[11px] text-faint">{files.length ? "Nothing new to install here; tick 'show every file' to inspect installed and reference files." : "The repository holds no files."}</div>
          ) : (
            <ul className="grid max-h-[40vh] overflow-auto">
              {shown.map((f) => {
                const infoOpen = openInfo === f.path;
                const isChecked = !!selected[f.path];
                const isQueued = queuedPaths.has(`${src.id}:${f.path}`);
                return (
                  <li key={f.path} className={`grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3 border-t border-line/40 py-1.5 ${isChecked ? "bg-sense/5" : ""}`}>
                    <div className="flex min-w-0 items-start gap-2">
                      {f.installable ? (
                        <input
                          type="checkbox"
                          checked={isChecked}
                          disabled={airGap || isQueued}
                          onChange={() => toggleOne(f.path)}
                          aria-label={`Select ${f.path}`}
                          className="mt-0.5 shrink-0"
                        />
                      ) : (
                        <span className="mt-1 inline-block h-2 w-2 shrink-0 rounded-full bg-line-strong" aria-hidden />
                      )}
                      <span className="min-w-0 flex-1">
                        <span className="mono block truncate text-[11px] text-muted" title={f.path}>{f.path}</span>
                        <span className="mono text-[10px] text-faint">
                          {f.kind}{f.role ? ` for the ${f.role}` : ""} · {bytes(f.bytes)} · {f.sha256 ? `SHA-256 ${f.sha256.slice(0, 12)}` : "checked on import"}
                          {isQueued ? " · queued" : ""}
                        </span>
                        {infoOpen && f.description && (
                          <span className="mono mt-1 block border-l border-line-strong pl-2 text-[10px] leading-relaxed text-faint">
                            {f.description}
                          </span>
                        )}
                        {armed === f.path && (
                          <span className="mono mt-1 flex flex-wrap items-center gap-3 text-[10.5px]">
                            <span className="text-text">Fetch {f.path} from {src.repo}{f.kind === "stack file" ? "; it joins the cartridge library on this machine, on trial" : f.role ? ` as a ${f.role} map; it never replaces one` : ""}?</span>
                            <button type="button" autoFocus disabled={busy !== null || dlRunning} onClick={() => void fetchFile(f)} className={go}>
                              confirm fetch
                            </button>
                            <button type="button" onClick={() => setArmed(null)} className={link}>
                              cancel
                            </button>
                          </span>
                        )}
                      </span>
                    </div>
                    <span className="flex items-center gap-2.5">
                      {f.description && (
                        <button
                          type="button"
                          onClick={() => setOpenInfo(infoOpen ? null : f.path)}
                          aria-expanded={infoOpen}
                          className="press mono border-b border-line px-0.5 text-[9.5px] uppercase tracking-[0.1em] text-quiet hover:text-text"
                        >
                          {infoOpen ? "hide" : "info"}
                        </button>
                      )}
                      {f.installable ? (
                        armed !== f.path && (
                          <button type="button" disabled={busy !== null || airGap || dlRunning || isQueued} onClick={() => setArmed(f.path)} className={go}>
                            {isQueued ? "queued" : "fetch"}
                          </button>
                        )
                      ) : (
                        <span className="mono max-w-[9rem] text-right text-[10px] text-quiet">{f.why_not}</span>
                      )}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </li>
  );
}

function PublishBlock({
  publishable,
  officialRepo,
  airGap,
  tokenFound,
  onNote,
}: {
  publishable: Publishable;
  officialRepo: string;
  airGap: boolean;
  tokenFound: boolean;
  onNote: (n: Note) => void;
}) {
  const [itemKind, setItemKind] = useState<"parameter-map" | "stack">("parameter-map");
  const [selectedMap, setSelectedMap] = useState("");
  const [selectedStack, setSelectedStack] = useState("");
  const [targetRepo, setTargetRepo] = useState(officialRepo);
  const [repoKind, setRepoKind] = useState("model");
  const [revision, setRevision] = useState("main");
  const [createPr, setCreatePr] = useState(true);
  const [summary, setSummary] = useState("");
  const [armed, setArmed] = useState(false);
  const [showItemInfo, setShowItemInfo] = useState(false);
  const [busy, setBusy] = useState(false);
  const [lastUrl, setLastUrl] = useState<string | null>(null);

  const maps = publishable.maps;
  const stacks = publishable.stacks;
  const activeMap = maps.find((m) => `${m.role}/${m.file}` === selectedMap) ?? maps[0] ?? null;
  const activeStack = stacks.find((s) => s.key === selectedStack) ?? stacks[0] ?? null;
  const canPublish = !airGap && !busy && !!targetRepo.trim() && (itemKind === "parameter-map" ? !!activeMap : !!activeStack);

  const publishNow = async () => {
    setBusy(true);
    try {
      const body: Record<string, unknown> = {
        action: "publish",
        confirm: true,
        repo: targetRepo.trim(),
        kind: repoKind,
        revision: revision.trim() || "main",
        create_pr: createPr,
        item_kind: itemKind,
        summary: summary.trim(),
      };
      if (itemKind === "parameter-map" && activeMap) {
        body.role = activeMap.role;
        body.item = activeMap.file;
      } else if (itemKind === "stack" && activeStack) {
        body.item = activeStack.key;
      }
      const r = await post<{ published: { path: string; repo: string; url: string; create_pr: boolean; audit_id: string } }>("/api/cartridge-sources", body);
      setLastUrl(r.published.url);
      setArmed(false);
      setSummary("");
      onNote({
        text: `Published ${r.published.path} to ${r.published.repo}${r.published.create_pr ? " (Pull Request)" : ""}. Audit ${r.published.audit_id}.`,
        tone: "verified",
      });
      refreshAll("/api/cartridge-sources");
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
      setArmed(false);
    }
    setBusy(false);
  };

  return (
    <div className="mt-5 grid gap-2 border-t border-line/60 pt-4" data-e2e="source-publish">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span
          className="eyebrow"
          title="Upload a local parameter map or exported stack directly to your own Hugging Face repo or open a Pull Request on the official Neuromorph Nano cartridge library. Uses your Hugging Face token (HF_TOKEN or hf auth login), sent only to huggingface.co."
        >
          Publish to Hugging Face
        </span>
        <span className={`mono text-[10px] ${tokenFound ? "text-verified" : "text-quiet"}`}>
          {tokenFound ? "HF token ready" : "set HF_TOKEN or hf auth login"}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <select
          value={itemKind}
          onChange={(e) => {
            setItemKind(e.target.value === "stack" ? "stack" : "parameter-map");
            setArmed(false);
          }}
          aria-label="What to publish"
          className={field}
        >
          <option value="parameter-map" className="bg-bg-1">parameter map ({maps.length})</option>
          <option value="stack" className="bg-bg-1">stack ({stacks.length})</option>
        </select>
        {itemKind === "parameter-map" ? (
          <select
            value={activeMap ? `${activeMap.role}/${activeMap.file}` : ""}
            onChange={(e) => {
              setSelectedMap(e.target.value);
              setArmed(false);
            }}
            aria-label="Local parameter map"
            className={`${field} min-w-[12rem] flex-1`}
          >
            {maps.length === 0 ? (
              <option value="" className="bg-bg-1">no local parameter maps</option>
            ) : (
              maps.map((m) => (
                <option key={`${m.role}/${m.file}`} value={`${m.role}/${m.file}`} className="bg-bg-1">
                  {m.role}/{m.file} ({bytes(m.bytes)}{m.direct_ok ? "" : " · >10 MiB LFS"})
                </option>
              ))
            )}
          </select>
        ) : (
          <select
            value={activeStack ? activeStack.key : ""}
            onChange={(e) => {
              setSelectedStack(e.target.value);
              setArmed(false);
            }}
            aria-label="Local stack"
            className={`${field} min-w-[12rem] flex-1`}
          >
            {stacks.length === 0 ? (
              <option value="" className="bg-bg-1">no shareable stacks</option>
            ) : (
              stacks.map((st) => (
                <option key={st.key} value={st.key} className="bg-bg-1">
                  {st.key} ({st.use})
                </option>
              ))
            )}
          </select>
        )}
        <button
          type="button"
          onClick={() => setShowItemInfo((v) => !v)}
          aria-expanded={showItemInfo}
          className="press mono border-b border-line px-0.5 text-[9.5px] uppercase tracking-[0.1em] text-quiet hover:text-text"
        >
          {showItemInfo ? "hide" : "info"}
        </button>
      </div>
      {showItemInfo && (
        <p className="mono border-l border-line-strong pl-2.5 text-[10px] leading-relaxed text-faint">
          {itemKind === "parameter-map"
            ? (activeMap?.description || "Parameter map or serving job filter for this MicroModel.")
            : (activeStack?.description || "Exported cartridge stack bundling verified non-base cartridges.")}
        </p>
      )}
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <input
          value={targetRepo}
          onChange={(e) => {
            setTargetRepo(e.target.value);
            setArmed(false);
          }}
          placeholder="owner/name on Hugging Face"
          aria-label="Target Hugging Face repository"
          className={`${field} w-full basis-full`}
        />
        <select value={repoKind} onChange={(e) => setRepoKind(e.target.value)} aria-label="Target repository kind" className={field}>
          <option value="model" className="bg-bg-1">model repo</option>
          <option value="dataset" className="bg-bg-1">dataset repo</option>
        </select>
        <input value={revision} onChange={(e) => setRevision(e.target.value)} placeholder="main" aria-label="Target branch" className={`${field} w-24`} />
        <input value={summary} onChange={(e) => setSummary(e.target.value)} placeholder="commit / PR title (optional)" aria-label="Summary" className={`${field} min-w-[10rem] flex-1`} />
      </div>
      <label className="mono flex items-center gap-2 text-[10.5px] text-faint">
        <input type="checkbox" checked={createPr} onChange={(e) => setCreatePr(e.target.checked)} />
        <span>open as Pull Request (recommended when contributing to {officialRepo})</span>
      </label>
      {itemKind === "parameter-map" && activeMap && !activeMap.direct_ok && (
        <p className="mono text-[10px] text-attention">
          {activeMap.role}/{activeMap.file} is over 10 MiB. Direct browser/API upload caps at 10 MiB; click Publish for the exact `hf upload` Xet/LFS command.
        </p>
      )}
      {!armed ? (
        <div className="flex flex-wrap items-center gap-3">
          <button type="button" disabled={!canPublish} onClick={() => setArmed(true)} className={go}>
            publish to hugging face
          </button>
          {targetRepo !== officialRepo && (
            <button
              type="button"
              onClick={() => {
                setTargetRepo(officialRepo);
                setCreatePr(true);
              }}
              className={link}
            >
              use official repo (PR)
            </button>
          )}
        </div>
      ) : (
        <div className="mono flex flex-wrap items-center gap-3 border-l border-sense-pale pl-3 text-[10.5px]">
          <span className="text-text">
            Upload {itemKind === "parameter-map" && activeMap ? `${activeMap.role}/${activeMap.file}` : activeStack?.key} to {targetRepo.trim()}
            {createPr ? " as a Pull Request" : ` (${revision.trim() || "main"})`}?
          </span>
          <button type="button" autoFocus disabled={busy} onClick={() => void publishNow()} className={go}>
            {busy ? "uploading" : "confirm publish"}
          </button>
          <button type="button" disabled={busy} onClick={() => setArmed(false)} className={link}>
            cancel
          </button>
        </div>
      )}
      {lastUrl && (
        <p className="mono truncate text-[10.5px] text-verified" title={lastUrl}>
          Published: {lastUrl}
        </p>
      )}
    </div>
  );
}

type QueueItem = { id: string; repo: string; path: string };

export default function CartridgeSources() {
  const q = useApi<Reply>("/api/cartridge-sources", 15000);
  const dlQ = useApi<DlReply>("/api/downloads", 2000);
  const [note, setNote] = useState<Note | null>(null);
  const [repo, setRepo] = useState("");
  const [kind, setKind] = useState("model");
  const [revision, setRevision] = useState("");
  const [token, setToken] = useState(false);
  const [busy, setBusy] = useState(false);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const dispatchLock = useRef(false);
  const d = q.data;
  const job = dlQ.data?.job ?? null;
  const dlRunning = !!job?.running;

  useEffect(() => {
    if (queue.length === 0 || dlRunning || dispatchLock.current) return;
    const head = queue[0];
    dispatchLock.current = true;
    void post<{ line?: string }>("/api/cartridge-sources", { action: "fetch", id: head.id, path: head.path })
      .then((r) => {
        setQueue((prev) => prev.slice(1));
        setNote({ text: `${r.line ?? `Fetching ${head.path}`}.`, tone: "verified" });
        for (const p of ["/api/downloads", "/api/cartridges", "/api/parameter-maps", "/api/cartridge-sources"]) refreshAll(p);
      })
      .catch((e) => {
        setQueue((prev) => prev.slice(1));
        setNote({ text: `${head.path}: ${e instanceof Error ? e.message : String(e)}`, tone: "error" });
      })
      .finally(() => {
        dispatchLock.current = false;
      });
  }, [queue, dlRunning]);

  const queuedPaths = new Set(queue.map((it) => `${it.id}:${it.path}`));
  const queueFiles = (sourceId: string, sourceRepo: string, paths: string[]) => {
    if (paths.length === 0) return;
    setQueue((prev) => [...prev, ...paths.map((path) => ({ id: sourceId, repo: sourceRepo, path }))]);
    setNote({ text: `Queued ${paths.length} ${paths.length === 1 ? "file" : "files"} from ${sourceRepo} for verified import.`, tone: "verified" });
  };

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
          title="Take cartridges from the official Neuromorph library or from any third-party library on Hugging Face. Nothing is fetched until you ask: browse lists a library's files, fetch downloads one or selected files and installs them through the same checks as ours (a stack joins on trial until its evidence lets it serve; a parameter map never replaces one of yours). A library shares stacks under stacks/ (or as name.stack.json) and maps under parameter-maps/<role>/. The fallback stays the Qwen3 open weights, in Downloads."
        >
          Cartridge libraries · official and third party
        </span>
        {d && (
          <span className="mono text-[10.5px] text-faint">
            {d.sources.length} {d.sources.length === 1 ? "source" : "sources"}
          </span>
        )}
      </div>
      {(job || queue.length > 0) && (
        <div className="mono mb-3 grid gap-1.5 border border-line bg-bg-1 p-2.5 text-[10.5px]" data-e2e="sources-download-job">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="flex flex-wrap items-center gap-2">
              {job && (
                <Pill tone={job.running ? "signal" : job.stopped ? "attention" : job.failed.length ? "error" : "verified"}>
                  {job.running ? "downloading" : job.stopped ? "stopped" : job.failed.length ? "finished with faults" : "finished"}
                </Pill>
              )}
              {job && (
                <span className="text-muted">
                  {job.done} of {job.total} {job.total === 1 ? "file" : "files"}
                </span>
              )}
              {queue.length > 0 && <span className="text-sense-pale">· {queue.length} queued next</span>}
            </span>
            {queue.length > 0 && (
              <button type="button" onClick={() => setQueue([])} className={link}>
                clear queue
              </button>
            )}
          </div>
          {job?.running && job.current && (
            <span className="min-w-0 truncate text-faint" title={job.current}>
              {job.current} · {bytes(job.current_bytes)} so far
            </span>
          )}
          {job && (
            <div className="h-1 w-full bg-line">
              <div className="h-full bg-sense-pale transition-[width] duration-300" style={{ width: `${job.total ? Math.round((job.done / job.total) * 100) : 0}%` }} />
            </div>
          )}
          {job && !job.running && job.stopped && <div className="text-attention">{job.stopped}</div>}
          {job?.failed.map((f) => (
            <div key={f} className="break-all text-error">
              {f}
            </div>
          ))}
        </div>
      )}
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
              <SourceBlock
                key={src.id}
                src={src}
                airGap={d.air_gap}
                dlRunning={dlRunning}
                queuedPaths={queuedPaths}
                onQueueFiles={queueFiles}
                onNote={setNote}
              />
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
          {d.publishable && (
            <PublishBlock
              publishable={d.publishable}
              officialRepo={d.sources.find((s) => s.official)?.repo ?? "gavriel-tech/neuromorph-nano-cartridges"}
              airGap={d.air_gap}
              tokenFound={d.token_found}
              onNote={setNote}
            />
          )}
          <p className="mono mt-3 text-[10.5px] text-quiet">Fallback: {d.fallback}.</p>
        </>
      )}
    </section>
  );
}
