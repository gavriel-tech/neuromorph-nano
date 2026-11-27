"use client";

// Floating, movable, resizable, scrollable Hugging Face Download Manager window.
// Brings everything the official Neuromorph Nano Hugging Face repository
// (gavriel-tech/neuromorph-nano-cartridges) and any added third-party Hugging Face libraries offer
// into one unified checklist: base cartridges, serving job filters, parameter maps, shared stacks,
// and optional upstream open weights. Users can filter by category/role/search, inspect plain-English
// descriptions before downloading, select any combination of files, confirm, and watch live download
// progress in place without leaving their current page.

import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { createPortal } from "react-dom";
import DeleteFlow from "@/components/DeleteFlow";
import { Pill, Placeholder } from "@/components/ui";
import { ago, bytes, post, refreshAll, useApi } from "@/lib/api";
import { useNow } from "@/lib/clock";

type Category = "all" | "base" | "filters" | "maps" | "stacks" | "open";
type RoleFilter = "all" | "reader" | "writer" | "embedder" | "reranker" | "shared";

type ModelRow = { id: string; repo: string; revision: string; bytes: number; files: number; present: number; partial: number; on_disk: number; state: string };
type CartRow = { key: string; present: boolean; on_disk: number; path: string; description?: string };
type MapFileRow = { file: string; kind: string; bytes: number; description?: string };
type MapRoleRow = { role: string; files: number; job_filters: number; super_weights: boolean; on_disk: number; items?: MapFileRow[] };
type DlJob = { what: string; running: boolean; done: number; total: number; current: string; current_bytes: number; failed: string[]; finished_ms: number; by?: string; stopped?: string };
type DlPayload = { models?: ModelRow[]; cartridges?: CartRow[]; parameter_maps?: MapRoleRow[]; cartridge_repo?: string; air_gap?: boolean; job?: DlJob | null };

type SourceFile = { path: string; bytes: number; sha256: string | null; kind: string; role: string | null; description?: string | null; installable: boolean; why_not: string | null; present?: boolean };
type Listing = { at_ms: number; revision: string; files: SourceFile[]; truncated: boolean };
type Source = { id: string; repo: string; kind: string; revision: string; official: boolean; token: boolean; added_ms: number; url: string; listing: Listing | null };
type LocalStack = { key: string; use: string; description: string };
type SourcesReply = { sources: Source[]; air_gap: boolean; token_found: boolean; fallback: string; publishable?: { stacks: LocalStack[] } };

type CatalogAction =
  | { kind: "base"; key: string }
  | { kind: "source"; sourceId: string; path: string }
  | { kind: "open"; modelId: string };

type DeleteSpec = { path: string; body: Record<string, unknown>; refresh: string[] };

type CatalogItem = {
  id: string;
  category: Exclude<Category, "all">;
  role: Exclude<RoleFilter, "all">;
  title: string;
  subtitle: string;
  description: string;
  bytes: number | null;
  sha256: string | null;
  status: "installed" | "partial" | "available" | "blocked";
  whyBlocked: string | null;
  selectable: boolean;
  destination: string;
  action: CatalogAction;
  deleteSpec: DeleteSpec | null;
};

type QueueEntry = { id: string; title: string; action: CatalogAction };

type WinRect = { x: number; y: number; w: number; h: number };
type DragKind = "move" | "e" | "s" | "se";
type DragState = { kind: DragKind; sx: number; sy: number; start: WinRect };

const WIN_MIN_W = 400;
const WIN_MIN_H = 300;
const WIN_DEF_W = 680;
const WIN_DEF_H = 560;
const WIN_PAD = 12;

function clampWin(r: WinRect, minimized = false): WinRect {
  const vw = typeof window === "undefined" ? 1280 : window.innerWidth;
  const vh = typeof window === "undefined" ? 800 : window.innerHeight;
  const maxW = Math.max(WIN_MIN_W, Math.min(960, vw - WIN_PAD * 2));
  const maxH = Math.max(WIN_MIN_H, vh - WIN_PAD * 2);
  const w = Math.max(WIN_MIN_W, Math.min(maxW, Math.round(r.w)));
  const h = Math.max(WIN_MIN_H, Math.min(maxH, Math.round(r.h)));
  const effH = minimized ? 36 : h;
  const x = Math.max(WIN_PAD, Math.min(vw - w - WIN_PAD, Math.round(r.x)));
  const y = Math.max(WIN_PAD, Math.min(vh - effH - WIN_PAD, Math.round(r.y)));
  return { x, y, w, h };
}

function defaultWin(): WinRect {
  const vw = typeof window === "undefined" ? 1280 : window.innerWidth;
  const vh = typeof window === "undefined" ? 800 : window.innerHeight;
  const w = Math.min(WIN_DEF_W, Math.max(WIN_MIN_W, vw - 48));
  const h = Math.min(WIN_DEF_H, Math.max(WIN_MIN_H, vh - 96));
  return clampWin({
    w,
    h,
    x: Math.round((vw - w) / 2),
    y: Math.max(48, Math.round((vh - h) / 2)),
  });
}

function roleFromKey(key: string): Exclude<RoleFilter, "all"> {
  const low = key.toLowerCase();
  if (low.includes("embedder") || low.includes("embedding")) return "embedder";
  if (low.includes("reranker")) return "reranker";
  if (low.includes("writer") || low.includes("1.7b")) return "writer";
  if (low.includes("reader") || low.includes("0.6b")) return "reader";
  return "shared";
}

function normalizeRole(r: string | null | undefined, fallbackKey: string): Exclude<RoleFilter, "all"> {
  if (r === "reader" || r === "writer" || r === "embedder" || r === "reranker") return r;
  return roleFromKey(fallbackKey);
}

/** Splits a versioned name (e.g. `foo-v3.stack.json` or `foo@3`) into its family key and numeric version. */
function versionKeyOf(name: string): { base: string; ver: number } {
  const mAt = /^(.+)@(\d+)$/.exec(name);
  if (mAt) return { base: mAt[1].toLowerCase(), ver: Number(mAt[2]) };
  const mV = /^(.+)-v(\d+)(\.[a-z0-9.]+)?$/i.exec(name);
  if (mV) return { base: `${mV[1]}${mV[3] ?? ""}`.toLowerCase(), ver: Number(mV[2]) };
  return { base: name.toLowerCase(), ver: 0 };
}

function describeItemFallback(name: string, role: string, isFilter: boolean, isStack: boolean): string {
  const low = name.toLowerCase();
  if (isStack) {
    if (low.includes("gtx-1650")) return "Shared hardware tuning stack for NVIDIA GeForce GTX 1650 (4 GB VRAM): bundles measured precision, batching, and co-residency settings. Imports on trial.";
    return "Exported cartridge stack bundling verified non-base cartridges and their SHA-256 hashes. Imports on trial.";
  }
  if (isFilter) {
    const job = name.replace(/^job-filter-/, "").replace(/\.json$/, "");
    if (job === "read") return `Maps per-layer affine gain and bias sockets across ${role} layers for workspace reading and cited span extraction.`;
    if (job === "write") return `Maps per-layer affine gain and bias sockets across ${role} layers for idle L1-L4 world-model note and outline writing.`;
    if (job === "judge-code") return `Maps per-layer affine gain and bias sockets across ${role} layers for close-candidate code judging and decoy withholding.`;
    if (job === "code-slice") return `Maps per-layer affine gain and bias sockets across ${role} layers for code-slice representation and retrieval.`;
    return `Maps per-layer affine gain and bias sockets across ${role} layers for the '${job}' job.`;
  }
  if (low === "super-weights.json") return `Maps exact [row, col] high-leverage super-weight coordinates across ${role} attention and MLP projections pinned for surgery.`;
  const owl = /^pmap-owl-(\d+)\.json$/.exec(low);
  if (owl) return `Maps per-weight Wanda sensitivity and OWL layer sparsity budgets (${owl[1]}% target sparsity) across ${role} attention and MLP projections for building ${role}-base.`;
  if (low.startsWith("pmap-")) return `Maps per-weight sensitivity and layer sparsity budgets across ${role} attention and MLP projections for building ${role}-base.`;
  if (low.startsWith("act-") && low.endsWith("-bias.json")) return `Maps per-output mean-activation bias compensation offsets across ${role} layers.`;
  if (low.startsWith("act-")) return `Maps per-channel feed-forward activation magnitudes across ${role} layers.`;
  if (low.startsWith("tol")) return `Maps per-layer pruning tolerance thresholds measuring hidden-state drift across ${role} layers.`;
  if (low.startsWith("role")) return `Maps per-layer role activation energy and sensitivity separation across ${role} layers.`;
  if (low.startsWith("swigate")) return `Maps mean SwiGLU gate activations per channel across ${role} MLP layers.`;
  if (low === "surgery-probe.json") return `Maps layer-by-layer hidden-state drift comparing ${role}-base against upstream open weights.`;
  const stem = name.replace(/\.json$/, "");
  return `Maps parameter measurements (${stem}) across ${role} layers.`;
}

function isArchivedPath(path: string): boolean {
  return path.split("/").some((seg) => {
    const s = seg.toLowerCase();
    return s === "archived" || s === "archive";
  });
}

/** Keeps only the highest version per logical file family (prevents doubles and decommissioned versions). */
function latestOnly(files: SourceFile[]): SourceFile[] {
  const best = new Map<string, { ver: number; file: SourceFile }>();
  for (const f of files) {
    if (isArchivedPath(f.path)) continue;
    const dir = f.path.includes("/") ? f.path.slice(0, f.path.lastIndexOf("/")).toLowerCase() : "";
    const name = f.path.split("/").pop() ?? f.path;
    const { base, ver } = versionKeyOf(name);
    const key = `${dir}/${base}`;
    const prev = best.get(key);
    if (!prev || ver > prev.ver) best.set(key, { ver, file: f });
  }
  return [...best.values()].map((x) => x.file);
}

const CAT_BADGE: Record<Exclude<Category, "all">, { label: string; dot: string }> = {
  base: { label: "base cartridge", dot: "bg-learned" },
  filters: { label: "job filter", dot: "bg-sense-pale" },
  maps: { label: "parameter map", dot: "bg-sense" },
  stacks: { label: "stack", dot: "bg-verified" },
  open: { label: "open weights", dot: "border border-quiet" },
};

const pillBtn = "press mono border border-line-strong bg-bg-2/70 px-1.5 py-0.5 text-[9.5px] uppercase tracking-[0.08em] text-muted hover:border-sense-pale hover:text-text disabled:opacity-40";
const linkBtn = "press mono border-b border-line-strong px-0.5 text-[9.5px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";

export function HFDownloadManager({
  open,
  onClose,
  defaultCategory = "all",
}: {
  open: boolean;
  onClose: () => void;
  defaultCategory?: Category;
}) {
  const dlQ = useApi<DlPayload>(open ? "/api/downloads" : null, 2000);
  const srcQ = useApi<SourcesReply>(open ? "/api/cartridge-sources" : null, 15000);
  const now = useNow();

  const [win, setWin] = useState<WinRect>(defaultWin);
  const [minimized, setMinimized] = useState(false);
  const [dragging, setDragging] = useState<DragKind | null>(null);
  const dragRef = useRef<DragState | null>(null);

  const [sourceId, setSourceId] = useState<string>("official");
  const [category, setCategory] = useState<Category>(defaultCategory);
  const [role, setRole] = useState<RoleFilter>("all");
  const [query, setQuery] = useState("");
  const [hideInstalled, setHideInstalled] = useState(false);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [openInfo, setOpenInfo] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [armedBatch, setArmedBatch] = useState(false);
  const [browsing, setBrowsing] = useState(false);
  const [note, setNote] = useState<{ text: string; tone: "ok" | "err" } | null>(null);
  const [queue, setQueue] = useState<QueueEntry[]>([]);

  const lastAutoRefreshRef = useRef<string | null>(null);
  const dispatchLockRef = useRef(false);

  const sources = useMemo(() => srcQ.data?.sources ?? [], [srcQ.data]);
  const activeSource = useMemo(
    () => sources.find((s) => s.id === sourceId) ?? sources.find((s) => s.official) ?? sources[0] ?? null,
    [sources, sourceId],
  );
  const airGap = Boolean(dlQ.data?.air_gap || srcQ.data?.air_gap);
  const job = dlQ.data?.job ?? null;
  const dlRunning = Boolean(job?.running);

  // Always refresh the selected source live from Hugging Face when opening or switching source,
  // so stale cached listings from older repository layouts never show decommissioned files.
  useEffect(() => {
    if (!open) {
      lastAutoRefreshRef.current = null;
      return;
    }
    if (airGap || !activeSource) return;
    if (lastAutoRefreshRef.current === activeSource.id) return;
    lastAutoRefreshRef.current = activeSource.id;
    void post<{ listing: Listing }>("/api/cartridge-sources", { action: "browse", id: activeSource.id })
      .then(() => {
        refreshAll("/api/cartridge-sources");
      })
      .catch((e) => {
        setNote({ text: e instanceof Error ? e.message : String(e), tone: "err" });
      });
  }, [open, airGap, activeSource]);

  // Sequential queue dispatcher across base cartridges, HF source files, and open weights.
  useEffect(() => {
    if (queue.length === 0 || dlRunning || dispatchLockRef.current) return;
    const head = queue[0];
    dispatchLockRef.current = true;
    const req =
      head.action.kind === "base"
        ? post<{ line?: string }>("/api/downloads", { what: "cartridges", only: head.action.key })
        : head.action.kind === "open"
          ? post<{ line?: string }>("/api/downloads", { what: `models:${head.action.modelId}` })
          : post<{ line?: string }>("/api/cartridge-sources", { action: "fetch", id: head.action.sourceId, path: head.action.path });
    void req
      .then((r) => {
        setQueue((prev) => prev.slice(1));
        setNote({ text: r.line ?? `Downloading ${head.title}`, tone: "ok" });
        for (const p of ["/api/downloads", "/api/cartridges", "/api/parameter-maps", "/api/cartridge-sources", "/api/status"]) {
          refreshAll(p);
        }
      })
      .catch((e) => {
        setQueue((prev) => prev.slice(1));
        setNote({ text: `${head.title}: ${e instanceof Error ? e.message : String(e)}`, tone: "err" });
      })
      .finally(() => {
        dispatchLockRef.current = false;
      });
  }, [queue, dlRunning]);

  // Keep popup clamped inside viewport and close on Escape.
  useEffect(() => {
    if (!open) return;
    const onResize = () => setWin((w) => clampWin(w, minimized));
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      onClose();
    };
    window.addEventListener("resize", onResize);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("resize", onResize);
      window.removeEventListener("keydown", onKey);
    };
  }, [open, minimized, onClose]);

  const beginDrag = (kind: DragKind, e: ReactPointerEvent<HTMLElement>) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = { kind, sx: e.clientX, sy: e.clientY, start: win };
    setDragging(kind);
  };
  const moveDrag = (e: ReactPointerEvent<HTMLElement>) => {
    const d = dragRef.current;
    if (!d) return;
    const dx = e.clientX - d.sx;
    const dy = e.clientY - d.sy;
    if (d.kind === "move") {
      setWin(clampWin({ ...d.start, x: d.start.x + dx, y: d.start.y + dy }, minimized));
    } else if (d.kind === "e") {
      setWin(clampWin({ ...d.start, w: d.start.w + dx }, minimized));
    } else if (d.kind === "s") {
      setWin(clampWin({ ...d.start, h: d.start.h + dy }, minimized));
    } else {
      setWin(clampWin({ ...d.start, w: d.start.w + dx, h: d.start.h + dy }, minimized));
    }
  };
  const endDrag = () => {
    dragRef.current = null;
    setDragging(null);
  };

  const refreshCatalog = async () => {
    if (!activeSource || airGap) return;
    setBrowsing(true);
    setNote(null);
    try {
      const r = await post<{ listing: Listing }>("/api/cartridge-sources", { action: "browse", id: activeSource.id });
      refreshAll("/api/cartridge-sources");
      refreshAll("/api/downloads");
      setNote({
        text: `Refreshed ${activeSource.repo}: ${r.listing.files.length} files (${r.listing.files.filter((f) => f.installable).length} installable).`,
        tone: "ok",
      });
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "err" });
    }
    setBrowsing(false);
  };

  // Build the unified catalog across base cartridges, HF job filters, parameter maps, stacks, and optional open weights.
  const allItems = useMemo<CatalogItem[]>(() => {
    const out: CatalogItem[] = [];
    const officialSelected = !activeSource || activeSource.official;
    const repoLabel = activeSource?.repo ?? dlQ.data?.cartridge_repo ?? "gavriel-tech/neuromorph-nano-cartridges";

    const deleteForMap = (r: string, file: string): DeleteSpec => {
      if (file.startsWith("job-filter-") && file.endsWith(".json")) {
        const jobName = file.slice("job-filter-".length, -".json".length);
        return {
          path: "/api/surgery/job-filter",
          body: { role: r, job: jobName, action: "delete" },
          refresh: ["/api/downloads", "/api/parameter-maps", "/api/surgery", "/api/cartridge-sources", "/api/cartridges"],
        };
      }
      return {
        path: "/api/parameter-maps/remove",
        body: { role: r, file },
        refresh: ["/api/downloads", "/api/parameter-maps", "/api/cartridge-sources", "/api/cartridges"],
      };
    };

    // 1. Base cartridges (official library only).
    if (officialSelected) {
      for (const c of dlQ.data?.cartridges ?? []) {
        const r = roleFromKey(c.key);
        const matchingHf = activeSource?.listing?.files.find(
          (f) => f.kind === "base cartridge" && f.path.includes(`/${r}/`),
        );
        out.push({
          id: `base:${c.key}`,
          category: "base",
          role: r,
          title: c.key,
          subtitle: `${repoLabel} · base/${r}/${c.key.replace(/@\d+$/, "")}.cartridge`,
          description: c.description || `Standalone Neuromorph base cartridge for the ${r} MicroModel.`,
          bytes: c.present ? c.on_disk : (matchingHf?.bytes ?? null),
          sha256: matchingHf?.sha256 ?? null,
          status: c.present ? "installed" : "available",
          whyBlocked: null,
          selectable: !c.present,
          destination: c.path,
          action: { kind: "base", key: c.key },
          deleteSpec: c.present
            ? {
                path: "/api/cartridges/bases/remove",
                body: { key: c.key },
                refresh: ["/api/downloads", "/api/cartridges/bases", "/api/cartridges", "/api/status"],
              }
            : null,
        });
      }
    }

    // Local lookup sets so Hugging Face items accurately show installed vs available without duplicating.
    const localMapSet = new Set<string>();
    for (const grp of dlQ.data?.parameter_maps ?? []) {
      const r = normalizeRole(grp.role, grp.role);
      for (const it of grp.items ?? []) localMapSet.add(`${r}/${it.file}`.toLowerCase());
    }
    const localStackSlugs = new Set<string>();
    for (const st of srcQ.data?.publishable?.stacks ?? []) {
      localStackSlugs.add(`${st.key.replace("@", "-v")}.stack.json`.toLowerCase());
    }

    // 2. Listed files from the live Hugging Face source (latest version only per family, no junk/duplicates).
    if (activeSource?.listing) {
      const deduped = latestOnly(activeSource.listing.files);
      for (const f of deduped) {
        if (f.kind === "base cartridge") continue;
        const name = f.path.split("/").pop() ?? f.path;
        const isFilter = f.kind === "parameter map" && name.startsWith("job-filter-");
        const isStack = f.kind === "stack file";
        const isMap = f.kind === "parameter map" && !isFilter;
        if (!isFilter && !isStack && !isMap) continue;
        const cat: Exclude<Category, "all"> = isFilter ? "filters" : isStack ? "stacks" : "maps";
        const r = isStack ? "shared" : normalizeRole(f.role, f.path);
        const installed = isStack
          ? localStackSlugs.has(name.toLowerCase())
          : Boolean(f.present) || localMapSet.has(`${r}/${name}`.toLowerCase());
        out.push({
          id: `src:${activeSource.id}:${f.path}`,
          category: cat,
          role: r,
          title: name,
          subtitle: `${activeSource.repo} · ${f.path}`,
          description: f.description || describeItemFallback(name, r, isFilter, isStack),
          bytes: f.bytes > 0 ? f.bytes : null,
          sha256: f.sha256,
          status: installed ? "installed" : f.installable ? "available" : "blocked",
          whyBlocked: installed ? null : f.why_not,
          selectable: f.installable && !installed,
          destination: isStack ? "library/cartridges/ (imports on trial)" : `library/parameter-maps/${r}/${name}`,
          action: { kind: "source", sourceId: activeSource.id, path: f.path },
          deleteSpec: installed && !isStack ? deleteForMap(r, name) : null,
        });
      }
    }

    // 3. Optional upstream open weights (official catalog view).
    if (officialSelected) {
      for (const m of dlQ.data?.models ?? []) {
        const r = roleFromKey(m.id);
        const installed = m.state === "installed";
        const partial = m.state === "partial";
        const remBytes = m.bytes ? Math.max(0, m.bytes - m.on_disk) : null;
        out.push({
          id: `open:${m.id}`,
          category: "open",
          role: r,
          title: m.id,
          subtitle: `${m.repo} @ ${m.revision.slice(0, 8)} · ${m.present}/${m.files} files`,
          description: `Optional upstream Apache-2.0 open weights for ${r}. Not required to run on base cartridges; used for open-mode fallback, parity reference checks, parameter mapping, and cartridge surgery.`,
          bytes: installed ? m.on_disk : (remBytes ?? (m.bytes || null)),
          sha256: null,
          status: installed ? "installed" : partial ? "partial" : "available",
          whyBlocked: null,
          selectable: !installed,
          destination: `weights/${m.id}/files/`,
          action: { kind: "open", modelId: m.id },
          deleteSpec: null,
        });
      }
    }

    return out;
  }, [activeSource, dlQ.data, srcQ.data?.publishable?.stacks]);

  const catCounts = useMemo(() => {
    const c = { all: 0, base: 0, filters: 0, maps: 0, stacks: 0, open: 0 };
    for (const it of allItems) {
      if (hideInstalled && it.status === "installed") continue;
      c.all += 1;
      c[it.category] += 1;
    }
    return c;
  }, [allItems, hideInstalled]);

  const visibleItems = useMemo(() => {
    const q = query.trim().toLowerCase();
    return allItems.filter((it) => {
      if (hideInstalled && it.status === "installed") return false;
      if (category !== "all" && it.category !== category) return false;
      if (role !== "all" && it.role !== role) return false;
      if (q) {
        const hay = `${it.title} ${it.subtitle} ${it.description} ${it.role} ${CAT_BADGE[it.category].label}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [allItems, hideInstalled, category, role, query]);

  const queuedIds = useMemo(() => new Set(queue.map((q) => q.id)), [queue]);
  const selectedItems = useMemo(
    () => allItems.filter((it) => it.selectable && selected[it.id] && !queuedIds.has(it.id)),
    [allItems, selected, queuedIds],
  );
  const selectedKnownBytes = selectedItems.reduce((sum, it) => sum + (it.bytes ?? 0), 0);
  const visibleSelectable = visibleItems.filter((it) => it.selectable && !queuedIds.has(it.id));

  const toggleItem = (id: string) => {
    setSelected((prev) => ({ ...prev, [id]: !prev[id] }));
    setArmedBatch(false);
  };
  const selectVisible = () => {
    setSelected((prev) => {
      const next = { ...prev };
      for (const it of visibleSelectable) next[it.id] = true;
      return next;
    });
    setArmedBatch(false);
  };
  const clearSelected = () => {
    setSelected({});
    setArmedBatch(false);
  };
  const enqueueItems = (items: CatalogItem[]) => {
    if (items.length === 0) return;
    setQueue((prev) => {
      const seen = new Set(prev.map((x) => x.id));
      const additions: QueueEntry[] = [];
      for (const it of items) {
        if (!seen.has(it.id)) additions.push({ id: it.id, title: it.title, action: it.action });
      }
      return [...prev, ...additions];
    });
    setSelected((prev) => {
      const next = { ...prev };
      for (const it of items) delete next[it.id];
      return next;
    });
    setArmedBatch(false);
    setNote({
      text: `Started download queue for ${items.length} ${items.length === 1 ? "item" : "items"}. Every file is verified before keeping.`,
      tone: "ok",
    });
  };

  if (!open || typeof document === "undefined") return null;

  const availCount = allItems.filter((it) => it.selectable).length;
  const installedCount = allItems.filter((it) => it.status === "installed").length;
  const headerBadge = airGap
    ? "air gap"
    : dlRunning && job
      ? `${job.done}/${job.total} downloading`
      : queue.length > 0
        ? `${queue.length} queued`
        : `${availCount} available · ${installedCount} installed`;

  const panel = (
    <aside
      role="dialog"
      aria-label="Hugging Face Download Manager"
      data-e2e="hf-download-manager"
      className="mono fixed z-50 flex flex-col overflow-hidden border border-line-3 bg-bg-1/97 text-[10.5px] text-text shadow-[0_20px_56px_rgba(2,3,10,0.9)] backdrop-blur-md"
      style={{
        left: win.x,
        top: win.y,
        width: win.w,
        height: minimized ? "auto" : win.h,
        maxWidth: "calc(100vw - 24px)",
        maxHeight: "calc(100vh - 24px)",
      }}
    >
      {/* Draggable Title Bar */}
      <div
        onPointerDown={(e) => beginDrag("move", e)}
        onPointerMove={moveDrag}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onDoubleClick={() => setMinimized((m) => !m)}
        className={`flex h-9 shrink-0 select-none items-center justify-between gap-2 border-b border-line-strong bg-bg-2 px-3 ${dragging === "move" ? "cursor-grabbing" : "cursor-grab"}`}
        title="Drag to move · Double-click to collapse"
      >
        <div className="flex min-w-0 items-center gap-2">
          <svg width="8" height="12" viewBox="0 0 8 12" aria-hidden className="shrink-0 text-quiet">
            <circle cx="2" cy="2.5" r="1" fill="currentColor" />
            <circle cx="6" cy="2.5" r="1" fill="currentColor" />
            <circle cx="2" cy="6" r="1" fill="currentColor" />
            <circle cx="6" cy="6" r="1" fill="currentColor" />
            <circle cx="2" cy="9.5" r="1" fill="currentColor" />
            <circle cx="6" cy="9.5" r="1" fill="currentColor" />
          </svg>
          <span className="truncate text-[10px] font-medium uppercase tracking-[0.14em] text-text">Hugging Face Downloads</span>
          <span className="hidden truncate border border-line-strong bg-bg-1 px-1.5 py-0.5 text-[9px] text-sense-pale sm:inline">{headerBadge}</span>
        </div>
        <div className="flex shrink-0 items-center gap-1" onPointerDown={(e) => e.stopPropagation()}>
          <button
            type="button"
            onClick={() => {
              setWin(defaultWin());
              setMinimized(false);
            }}
            title="Reset window position and size"
            className="press h-6 border border-line px-1.5 text-[9px] uppercase tracking-[0.08em] text-faint hover:border-line-3 hover:text-text"
          >
            fit
          </button>
          <button
            type="button"
            onClick={() => setMinimized((m) => !m)}
            title={minimized ? "Expand download manager" : "Collapse download manager"}
            aria-label={minimized ? "Expand download manager" : "Collapse download manager"}
            className="press flex h-6 w-6 items-center justify-center border border-line text-[12px] leading-none text-faint hover:border-line-3 hover:text-text"
          >
            {minimized ? "+" : "-"}
          </button>
          <button
            type="button"
            onClick={onClose}
            title="Close download manager (Esc)"
            aria-label="Close download manager"
            className="press flex h-6 w-6 items-center justify-center border border-line text-[12px] leading-none text-faint hover:border-error/70 hover:text-error"
          >
            x
          </button>
        </div>
      </div>

      {!minimized && (
        <>
          {/* Source, Filters & Search Controls (fixed top strip) */}
          <div className="shrink-0 space-y-2 border-b border-line bg-bg-1 px-3 py-2.5">
            <div className="flex flex-wrap items-center gap-2">
              <label className="flex min-w-0 flex-1 items-center gap-1.5">
                <span className="shrink-0 text-[9px] uppercase tracking-[0.12em] text-quiet">source</span>
                <select
                  value={activeSource?.id ?? "official"}
                  onChange={(e) => {
                    setSourceId(e.target.value);
                    setSelected({});
                    setArmedBatch(false);
                  }}
                  aria-label="Hugging Face library source"
                  className="mono h-7 min-w-0 flex-1 border border-line-strong bg-bg-2 px-1.5 text-[10.5px] text-text outline-none focus-visible:border-sense-pale"
                >
                  {sources.length === 0 ? (
                    <option value="official" className="bg-bg-1">gavriel-tech/neuromorph-nano-cartridges (official)</option>
                  ) : (
                    sources.map((s) => (
                      <option key={s.id} value={s.id} className="bg-bg-1">
                        {s.repo} {s.official ? "(official)" : "(third party)"}
                      </option>
                    ))
                  )}
                </select>
              </label>
              <button
                type="button"
                disabled={browsing || airGap || !activeSource}
                onClick={() => void refreshCatalog()}
                className={pillBtn}
                title="Query Hugging Face for the latest file list and sizes"
              >
                {browsing ? "listing..." : "refresh catalog"}
              </button>
              {activeSource?.listing && (
                <span className="text-[9.5px] text-quiet">listed {ago(activeSource.listing.at_ms, now)}</span>
              )}
            </div>

            {/* Category Tabs */}
            <div className="flex flex-wrap items-center gap-1">
              {(
                [
                  ["all", `all (${catCounts.all})`],
                  ["base", `base cartridges (${catCounts.base})`],
                  ["filters", `job filters (${catCounts.filters})`],
                  ["maps", `maps (${catCounts.maps})`],
                  ["stacks", `stacks (${catCounts.stacks})`],
                  ["open", `open weights (${catCounts.open})`],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setCategory(id)}
                  className={`press border px-2 py-0.5 text-[9.5px] uppercase tracking-[0.08em] ${category === id ? "border-sense-pale bg-sense/20 text-text" : "border-line text-faint hover:border-line-3 hover:text-text"}`}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Role Filter + Search + Hide Installed */}
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex flex-wrap items-center gap-1">
                {(["all", "reader", "writer", "embedder", "reranker", "shared"] as const).map((r) => (
                  <button
                    key={r}
                    type="button"
                    onClick={() => setRole(r)}
                    className={`press border px-1.5 py-0.5 text-[9px] uppercase tracking-[0.08em] ${role === r ? "border-line-3 bg-bg-2 text-text" : "border-line/70 text-quiet hover:text-text"}`}
                  >
                    {r === "all" ? "all roles" : r}
                  </button>
                ))}
              </div>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="search name or description..."
                aria-label="Filter Hugging Face files"
                className="mono h-6 min-w-[9rem] flex-1 border border-line-strong bg-bg px-2 text-[10px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
              />
              <label className="flex shrink-0 items-center gap-1 text-[9.5px] text-faint">
                <input type="checkbox" checked={hideInstalled} onChange={(e) => setHideInstalled(e.target.checked)} />
                <span>hide installed</span>
              </label>
            </div>

            {airGap && (
              <div className="border border-attention/50 bg-attention/10 px-2.5 py-1.5 text-[10px] text-attention">
                Air-Gap Mode is on: browsing and downloading from Hugging Face are disabled.
              </div>
            )}

            {(job || queue.length > 0) && (
              <div className="grid gap-1 border border-line-strong bg-bg-2/70 px-2.5 py-2 text-[10px]" data-e2e="hf-manager-job">
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
                    <button type="button" onClick={() => setQueue([])} className={linkBtn}>
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

            {note && (
              <p className={`text-[10px] leading-snug ${note.tone === "err" ? "text-error" : "text-verified"}`}>
                {note.text}
              </p>
            )}
          </div>

          {/* Scrollable Unified Catalog Body */}
          <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden p-3">
            {!dlQ.data && !srcQ.data ? (
              <Placeholder className="h-32" what="Hugging Face catalog" />
            ) : visibleItems.length === 0 ? (
              <div className="space-y-2 border border-line bg-bg-2/40 p-4 text-[10.5px] text-faint">
                <p className="text-text">No files match the current filter.</p>
                {activeSource && !activeSource.listing && !airGap && (
                  <p>
                    Click <strong className="text-text">refresh catalog</strong> above to fetch the file list from {activeSource.repo}.
                  </p>
                )}
              </div>
            ) : (
              <ul className="grid gap-1.5" data-e2e="hf-manager-list">
                {visibleItems.map((it) => {
                  const isChecked = Boolean(selected[it.id]);
                  const isQueued = queuedIds.has(it.id);
                  const infoShown = openInfo === it.id;
                  const meta = CAT_BADGE[it.category];
                  return (
                    <li
                      key={it.id}
                      className={`grid gap-1.5 border px-2.5 py-2 transition-colors ${isChecked ? "border-sense-pale bg-sense/10" : "border-line bg-bg-2/35 hover:border-line-strong"}`}
                    >
                      <div className="flex min-w-0 items-start gap-2.5">
                        <div className="pt-0.5">
                          {it.selectable ? (
                            <input
                              type="checkbox"
                              checked={isChecked}
                              disabled={airGap || isQueued}
                              onChange={() => toggleItem(it.id)}
                              aria-label={`Select ${it.title}`}
                            />
                          ) : (
                            <span
                              aria-hidden
                              className={`mt-1 inline-block h-2.5 w-2.5 rounded-full ${it.status === "installed" ? "bg-verified" : "bg-line-strong"}`}
                              title={it.status === "installed" ? "Already installed" : (it.whyBlocked ?? "Unavailable")}
                            />
                          )}
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                            <span className="flex items-center gap-1.5">
                              <i aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 ${meta.dot}`} />
                              <span className="break-all text-[11.5px] text-text">{it.title}</span>
                            </span>
                            <span className="border border-line bg-bg-1 px-1 py-0 text-[8.5px] uppercase tracking-[0.08em] text-faint">
                              {meta.label}
                            </span>
                            <span className="text-[9px] uppercase tracking-[0.08em] text-quiet">{it.role}</span>
                            {it.bytes !== null && <span className="text-[9.5px] text-faint">{bytes(it.bytes)}</span>}
                          </div>
                          <p className="mt-0.5 line-clamp-2 text-[10px] leading-snug text-faint">{it.description}</p>
                        </div>
                        <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
                          <Pill
                            tone={
                              isQueued
                                ? "signal"
                                : it.status === "installed"
                                  ? "verified"
                                  : it.status === "partial"
                                    ? "attention"
                                    : it.status === "available"
                                      ? "muted"
                                      : "muted"
                            }
                          >
                            {isQueued ? "queued" : it.status === "installed" ? "installed" : it.status === "partial" ? "partial" : it.status === "available" ? "available" : "info only"}
                          </Pill>
                          <button
                            type="button"
                            onClick={() => setOpenInfo(infoShown ? null : it.id)}
                            aria-expanded={infoShown}
                            className={linkBtn}
                          >
                            {infoShown ? "hide" : "info"}
                          </button>
                          {it.selectable && !isQueued && (
                            <button
                              type="button"
                              disabled={airGap}
                              onClick={() => {
                                setSelected((prev) => ({ ...prev, [it.id]: true }));
                                setArmedBatch(true);
                              }}
                              className="press mono border border-sense-pale/70 bg-sense/15 px-1.5 py-0.5 text-[9px] uppercase tracking-[0.08em] text-sense-pale hover:bg-sense/25 hover:text-text disabled:opacity-40"
                            >
                              get
                            </button>
                          )}
                          {it.deleteSpec && deletingId !== it.id && (
                            <button
                              type="button"
                              onClick={() => setDeletingId(it.id)}
                              className="press mono border-b border-line px-0.5 text-[9.5px] uppercase tracking-[0.1em] text-faint hover:border-error hover:text-error"
                              title="Preview and remove this downloaded file (two-step confirmation; serving files are protected)"
                            >
                              delete
                            </button>
                          )}
                        </div>
                      </div>
                      {it.deleteSpec && deletingId === it.id && (
                        <div className="ml-6 border-l border-error/60 pl-2.5">
                          <DeleteFlow
                            autoStart
                            path={it.deleteSpec.path}
                            body={it.deleteSpec.body}
                            label="delete"
                            done={() => `Removed ${it.title}.`}
                            refresh={it.deleteSpec.refresh}
                            onNote={(n) => setNote({ text: n.text, tone: n.tone === "error" ? "err" : "ok" })}
                            onClose={() => setDeletingId(null)}
                          />
                        </div>
                      )}
                      {infoShown && (
                        <div className="ml-6 grid gap-0.5 border-l border-line-strong pl-2.5 text-[9.5px] leading-relaxed text-faint">
                          <div className="text-muted">{it.description}</div>
                          <div className="break-all text-quiet">source: {it.subtitle}</div>
                          <div className="break-all text-quiet">installs to: {it.destination}</div>
                          {it.sha256 && <div className="text-quiet">SHA-256: {it.sha256}</div>}
                          {it.whyBlocked && <div className="text-attention">{it.whyBlocked}</div>}
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          {/* Sticky Selection & Confirm Download Footer */}
          <div className="shrink-0 space-y-2 border-t border-line-strong bg-bg-2/90 px-3 py-2">
            {armedBatch && selectedItems.length > 0 && (
              <div className="flex flex-wrap items-center justify-between gap-2 border border-sense-pale/60 bg-sense/15 px-2.5 py-1.5 text-[10px]">
                <span className="text-text">
                  Download {selectedItems.length} selected {selectedItems.length === 1 ? "item" : "items"}
                  {selectedKnownBytes > 0 ? ` (~${bytes(selectedKnownBytes)})` : ""}? Every file is checked against SHA-256 / library checks before keeping.
                </span>
                <span className="flex shrink-0 items-center gap-2">
                  <button
                    type="button"
                    autoFocus
                    disabled={airGap}
                    onClick={() => enqueueItems(selectedItems)}
                    className="press mono border border-sense-pale bg-sense/30 px-2.5 py-1 text-[9.5px] uppercase tracking-[0.1em] text-text hover:bg-sense/45 disabled:opacity-40"
                  >
                    confirm download
                  </button>
                  <button type="button" onClick={() => setArmedBatch(false)} className={pillBtn}>
                    cancel
                  </button>
                </span>
              </div>
            )}
            <div className="flex flex-wrap items-center justify-between gap-2 text-[10px]">
              <div className="flex flex-wrap items-center gap-2.5">
                <button type="button" disabled={visibleSelectable.length === 0 || airGap} onClick={selectVisible} className={linkBtn}>
                  select visible ({visibleSelectable.length})
                </button>
                <button type="button" disabled={selectedItems.length === 0} onClick={clearSelected} className={linkBtn}>
                  clear
                </button>
                <span className="text-faint">
                  {selectedItems.length > 0
                    ? `${selectedItems.length} selected${selectedKnownBytes > 0 ? ` · ~${bytes(selectedKnownBytes)}` : ""}`
                    : "Check any files above to choose what to download"}
                </span>
              </div>
              <button
                type="button"
                disabled={selectedItems.length === 0 || airGap}
                onClick={() => setArmedBatch(true)}
                className="press mono border border-sense-pale bg-sense/20 px-3 py-1 text-[10px] uppercase tracking-[0.1em] text-sense-pale hover:bg-sense/35 hover:text-text disabled:opacity-40"
                data-e2e="hf-manager-download-btn"
              >
                download selected ({selectedItems.length})
              </button>
            </div>
          </div>

          {/* Slim Resize Footer Bar */}
          <div className="flex h-5 shrink-0 select-none items-center justify-between border-t border-line bg-bg-2 pl-2.5 pr-0.5 text-[8.5px] text-quiet">
            <span className="truncate">drag title to move · drag edge or corner to resize · Esc to close</span>
            <div
              role="separator"
              aria-label="Resize Hugging Face download manager"
              onPointerDown={(e) => beginDrag("se", e)}
              onPointerMove={moveDrag}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
              className="flex h-5 w-6 shrink-0 cursor-nwse-resize items-center justify-center text-faint hover:text-sense-pale"
              title="Drag to resize"
            >
              <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
                <path d="M9 1L1 9M9 5L5 9M9 8.5L8.5 9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="square" />
              </svg>
            </div>
          </div>

          {/* Edge resize strips */}
          <div
            aria-hidden
            onPointerDown={(e) => beginDrag("e", e)}
            onPointerMove={moveDrag}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            className="absolute top-9 right-0 bottom-5 w-1.5 cursor-ew-resize"
          />
          <div
            aria-hidden
            onPointerDown={(e) => beginDrag("s", e)}
            onPointerMove={moveDrag}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            className="absolute right-6 bottom-0 left-0 h-1.5 cursor-ns-resize"
          />
        </>
      )}
    </aside>
  );

  return createPortal(panel, document.body);
}

/** One-click button that opens the floating Hugging Face Download Manager window. */
export function HFDownloadButton({
  label = "HF downloads",
  defaultCategory = "all",
  className,
}: {
  label?: string;
  defaultCategory?: Category;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={
          className ??
          "press mono border border-sense-pale/70 bg-sense/15 px-2 py-0.5 text-[9.5px] uppercase tracking-[0.1em] text-sense-pale hover:bg-sense/25 hover:text-text"
        }
        title="Open the floating Hugging Face Download Manager to browse and select base cartridges, job filters, parameter maps, stacks, and optional open weights"
        data-e2e="hf-download-open"
      >
        {label}
      </button>
      <HFDownloadManager open={open} onClose={() => setOpen(false)} defaultCategory={defaultCategory} />
    </>
  );
}
