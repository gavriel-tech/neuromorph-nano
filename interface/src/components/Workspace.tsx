"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden" ... "that
// whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals" ... "think of it as if it's an interactive
// system for the main parts the right menu are libraries and or discription tools etc. left menu are
// the main interface nav like it is now."): every page is three columns. The left is the page nav
// (Shell). The middle is the working system: its live visuals and its controls, no paragraphs. The
// right is the rail: the libraries, the references, the descriptions and the detail of whatever is
// picked in the middle, in tabs. The rail folds away (remembered per page); below the xl breakpoint
// it slides over from the right.
//
// Use:
//   <Workspace title="Cartridges" status={...} rail={[{ id: "library", label: "Library", body: <Library/> }]} about={<p>...</p>}>
//     ...the working system...
//   </Workspace>
// Anything in the middle can open a tab, or show a detail, in the rail with useRail():
//   const rail = useRail(); rail.show({ title: "pmap-owl-45.json", body: <MapDetail .../> }); rail.open("library");

import { createContext, useCallback, useContext, useEffect, useMemo, useState, useSyncExternalStore, type ReactNode } from "react";
import { Reticle } from "@/components/ui";
import { readPref, writePref } from "@/lib/prefs";

export type RailTab = { id: string; label: string; body: ReactNode; count?: ReactNode };
type Shown = { title: ReactNode; body: ReactNode; key?: string };
type RailApi = { open: (id: string) => void; show: (d: Shown) => void; clear: () => void; shown: Shown | null; active: string; isOpen: boolean };

const RailCtx = createContext<RailApi>({ open: () => {}, show: () => {}, clear: () => {}, shown: null, active: "", isOpen: false });

/** Open a rail tab or show a detail in the rail, from anywhere in the page. */
export function useRail(): RailApi {
  return useContext(RailCtx);
}

const store = {
  get(k: string): string | null {
    try {
      return readPref(k);
    } catch {
      return null;
    }
  },
  set(k: string, v: string) {
    try {
      writePref(k, v);
      window.dispatchEvent(new StorageEvent("storage", { key: k }));
    } catch {
      // storage unavailable: kept for this visit
    }
  },
};

const onStorage = (cb: () => void) => {
  window.addEventListener("storage", cb);
  return () => window.removeEventListener("storage", cb);
};
const WIDE = "(min-width: 1280px)";
const onWide = (cb: () => void) => {
  const mq = window.matchMedia(WIDE);
  mq.addEventListener("change", cb);
  return () => mq.removeEventListener("change", cb);
};

const tabCls = (on: boolean) => `press mono shrink-0 border-b px-1 pb-1.5 pt-1 text-[10px] uppercase tracking-[0.12em] transition-colors ${on ? "border-sense-pale text-text" : "border-transparent text-faint hover:text-muted"}`;

/** The right rail: tabs, then the picked tab's content; a detail shown from the page comes first. */
function Rail({ tabs, about, active, setActive, shown, clear, onClose }: { tabs: RailTab[]; about?: ReactNode; active: string; setActive: (id: string) => void; shown: Shown | null; clear: () => void; onClose: () => void }) {
  const all: RailTab[] = [...(shown ? [{ id: "detail", label: "Detail", body: shown.body }] : []), ...tabs, ...(about ? [{ id: "about", label: "About", body: about }] : [])];
  const cur = all.find((t) => t.id === active) ?? all[0];
  return (
    <div className="flex h-full min-h-0 flex-col" data-e2e="rail">
      {/* 2026-11-27: the tabs wrap onto a second line when they do not fit, so none is cut off at the panel's edge. */}
      <div className="flex shrink-0 flex-wrap items-end gap-x-3 gap-y-0.5 border-b border-line px-4 pt-3">
        {all.map((t) => (
          <button key={t.id} type="button" aria-pressed={cur?.id === t.id} onClick={() => setActive(t.id)} className={tabCls(cur?.id === t.id)} data-e2e="rail-tab" data-tab={t.id}>
            {t.label}
            {t.count !== undefined && <span className="ml-1.5 text-quiet">{t.count}</span>}
          </button>
        ))}
        <button type="button" onClick={onClose} aria-label="Fold the side panel" title="Fold the side panel" className="press mono mb-1 ml-auto shrink-0 text-[12px] text-faint hover:text-text">
          ⟩
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4" data-e2e="rail-body" data-tab={cur?.id}>
        {cur?.id === "detail" && shown && (
          <div className="mb-3 flex items-start justify-between gap-3">
            <div className="mono min-w-0 text-[12px] text-text">{shown.title}</div>
            <button type="button" onClick={clear} aria-label="Close the detail" className="press mono shrink-0 text-[11px] text-faint hover:text-text">
              ×
            </button>
          </div>
        )}
        {/* Every tab stays mounted (the others hidden), so a half-typed form survives a tab change. */}
        {all.map((t) => (
          <div key={t.id} hidden={t.id !== cur?.id} className="min-w-0 max-w-full" data-e2e="rail-pane" data-tab={t.id}>
            {t.body}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Workspace({
  title,
  status,
  rail = [],
  about,
  children,
  railKey,
  defaultTab,
}: {
  /** The page's name, one line. */
  title: string;
  /** Live figures on the title line (short; no sentences). */
  status?: ReactNode;
  /** The rail's tabs: libraries, references, tools. */
  rail?: RailTab[];
  /** What the page is and how it works: shown only in the rail's About tab. */
  about?: ReactNode;
  children: ReactNode;
  /** Remembers the rail open or folded, and its tab, under this key (defaults to the title). */
  railKey?: string;
  defaultTab?: string;
}) {
  const key = `nm-rail:${railKey ?? title}`;
  const wide = useSyncExternalStore(onWide, () => window.matchMedia(WIDE).matches, () => true);
  const narrow = !wide;
  const saved = useSyncExternalStore(onStorage, () => store.get(key), () => null);
  const savedTab = useSyncExternalStore(onStorage, () => store.get(`${key}:tab`), () => null);
  // Wide: open unless folded before (remembered). Narrow: a drawer, closed until asked for.
  const [drawer, setDrawer] = useState(false);
  const isOpen = narrow ? drawer : saved !== "closed";
  const [picked, setPicked] = useState<string | null>(null);
  const active = picked ?? savedTab ?? defaultTab ?? rail[0]?.id ?? "about";
  const [shown, setShown] = useState<Shown | null>(null);
  const openRail = useCallback(
    (v: boolean) => {
      if (narrow) setDrawer(v);
      else store.set(key, v ? "open" : "closed");
    },
    [key, narrow],
  );
  const pick = useCallback(
    (id: string) => {
      setPicked(id);
      if (id !== "detail") store.set(`${key}:tab`, id);
    },
    [key],
  );
  // open, show and clear keep their identity across tab changes, so a page effect that lists
  // them does not pull the viewer back to a tab.
  // Opened by the page (an address, a link): shown, not remembered; only a tab click is.
  const open = useCallback(
    (id: string) => {
      setPicked(id);
      openRail(true);
    },
    [openRail],
  );
  const show = useCallback(
    (d: Shown) => {
      setShown(d);
      setPicked("detail");
      openRail(true);
    },
    [openRail],
  );
  const clear = useCallback(() => {
    setShown(null);
    setPicked((a) => (a === "detail" ? null : a));
  }, []);
  const api = useMemo<RailApi>(() => ({ open, show, clear, shown, active, isOpen }), [open, show, clear, shown, active, isOpen]);
  // 2026-11-26: an address naming a side panel tab (`/parameters/#add`, `/cartridges/#bases`) opens
  // that tab, on arrival and when the address changes; any other address is the page's own.
  const tabIds = rail.map((r) => r.id).join(",");
  const hasAbout = !!about;
  useEffect(() => {
    const ids = new Set(tabIds.split(",").filter(Boolean));
    if (hasAbout) ids.add("about");
    const go = () => {
      const h = window.location.hash.replace(/^#/, "");
      if (h && ids.has(h)) open(h);
    };
    go();
    window.addEventListener("hashchange", go);
    return () => window.removeEventListener("hashchange", go);
  }, [tabIds, hasAbout, open]);
  const hasRail = rail.length > 0 || !!about;
  return (
    <RailCtx.Provider value={api}>
      <div className={`grid min-w-0 flex-1 lg:h-full lg:min-h-0 ${hasRail && isOpen && !narrow ? "xl:grid-cols-[minmax(0,1fr)_minmax(20rem,25rem)]" : ""}`}>
        <div className="flex min-w-0 flex-col lg:h-full lg:min-h-0">
          <header className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-2 border-b border-line px-4 pt-4 pb-3 md:gap-x-6 md:px-8 md:pt-6 2xl:px-10" data-e2e="work-head">
            <h1 className="mono flex items-center gap-2.5 text-[13px] font-normal uppercase leading-none tracking-[0.26em] text-text sm:gap-3 sm:text-[14px] sm:tracking-[0.3em]">
              <Reticle className="h-3.5 w-3.5 text-sense-pale" />
              {title}
            </h1>
            {status && <div className="mono flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-[10px] uppercase tracking-[0.1em] text-muted sm:gap-x-4 sm:text-[10.5px]">{status}</div>}
            {hasRail && (
              <span className="ml-auto flex shrink-0 items-center gap-3">
                {about && (
                  <button type="button" onClick={() => api.open("about")} className="press mono py-1 text-[10px] uppercase tracking-[0.12em] text-faint hover:text-text" title="What this page is and how it works">
                    about
                  </button>
                )}
                {!isOpen && (
                  <button type="button" onClick={() => openRail(true)} className="press mono border-b border-line-strong py-1 text-[10px] uppercase tracking-[0.12em] text-muted hover:text-text" data-e2e="rail-open">
                    ⟨ {rail.map((t) => t.label.toLowerCase()).slice(0, 2).join(", ") || "about"}
                  </button>
                )}
              </span>
            )}
          </header>
          <div className="min-w-0 flex-1 px-4 pt-5 pb-6 md:px-8 md:pb-7 2xl:px-10 lg:min-h-0 lg:overflow-y-auto" data-e2e="work-scroll">
            <div className="min-w-0" data-e2e="work-main">
              {children}
            </div>
          </div>
        </div>
        {/* The rail stays mounted while folded or closed (hidden), so its forms keep what was
            typed and anything in it can be found and opened from the page. */}
        {hasRail && !narrow && (
          <aside hidden={!isOpen} className="h-full min-h-0 min-w-0 border-l border-line bg-bg-1/60" aria-label="Side panel">
            <Rail tabs={rail} about={about} active={active} setActive={pick} shown={shown} clear={api.clear} onClose={() => openRail(false)} />
          </aside>
        )}
        {hasRail && narrow && (
          <div hidden={!isOpen} className="fixed inset-0 z-50 flex justify-end">
            <button aria-label="Close the side panel" className="absolute inset-0 bg-bg/70" onClick={() => openRail(false)} />
            <aside className="relative h-full w-[min(26rem,92vw)] border-l border-line bg-bg-1" aria-label="Side panel">
              <Rail tabs={rail} about={about} active={active} setActive={pick} shown={shown} clear={api.clear} onClose={() => openRail(false)} />
            </aside>
          </div>
        )}
      </div>
    </RailCtx.Provider>
  );
}

/** A short line of plain reference text for the rail (no headings needed). */
export function RailSection({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <section className="grid gap-2 border-b border-line/60 pb-4 [&:not(:first-child)]:pt-4">
      <h3 className="eyebrow text-muted">{title}</h3>
      <div className="min-w-0">{children}</div>
    </section>
  );
}
