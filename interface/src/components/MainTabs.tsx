"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "loops can be in memory but have its own tab in
// memory maybe" ... "not in the right side bar" ... "the world model inside the memory section
// should have its own tab too. right now its just awkwardly through in there."): tabs for the
// middle of a page, the working system, apart from the rail's tabs. The address names the tab
// (`/memory/#world`); an older address that named a part of a tab opens that tab. Every tab stays
// mounted (the others hidden), so a half-typed form survives a tab change.
// 2026-11-27 (his words: "The main center screen for every page should be easy to navigate"): the
// bar carries the page's own controls at its right end (`right`), the last tab picked is kept in
// this browser when a `storageKey` is given (an address that names a tab still wins), and the
// arrow keys move between tabs.

import { useSyncExternalStore, type KeyboardEvent, type ReactNode } from "react";
import { setHash, useHash } from "@/lib/hash";
import { readPref, writePref } from "@/lib/prefs";

export type MainTab = { id: string; label: string; count?: ReactNode; body: ReactNode; aliases?: string[]; title?: string };

const tabCls = (on: boolean) =>
  `press mono relative inline-flex min-h-10 shrink-0 items-center gap-1.5 border-b-2 px-3 text-[11px] uppercase tracking-[0.12em] transition-colors ${on ? "border-sense-pale bg-bg-2/40 text-text" : "border-transparent text-faint hover:bg-bg-2/30 hover:text-text"}`;

const onStorage = (cb: () => void) => {
  window.addEventListener("storage", cb);
  return () => window.removeEventListener("storage", cb);
};

function readStore(key: string | undefined): string | null {
  if (!key) return null;
  try {
    return readPref(key);
  } catch {
    return null;
  }
}

export default function MainTabs({ tabs, e2e = "main-tabs", right, storageKey, label = "Views" }: { tabs: MainTab[]; e2e?: string; right?: ReactNode; storageKey?: string; label?: string }) {
  const hash = useHash().replace(/^#/, "");
  const stored = useSyncExternalStore(
    onStorage,
    () => readStore(storageKey),
    () => null,
  );
  const named = tabs.find((x) => x.id === hash || x.aliases?.includes(hash));
  const cur = named ?? tabs.find((x) => x.id === stored) ?? tabs[0];
  const pick = (id: string) => {
    if (storageKey) {
      try {
        writePref(storageKey, id);
        window.dispatchEvent(new StorageEvent("storage", { key: storageKey }));
      } catch {
        // storage unavailable: the choice holds through the address
      }
    }
    setHash(`#${id}`, false);
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft" && e.key !== "Home" && e.key !== "End") return;
    const i = Math.max(0, tabs.findIndex((t) => t.id === cur?.id));
    const next = e.key === "Home" ? tabs[0] : e.key === "End" ? tabs[tabs.length - 1] : tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    if (!next) return;
    e.preventDefault();
    pick(next.id);
    e.currentTarget.querySelector<HTMLButtonElement>(`[data-tab="${next.id}"]`)?.focus();
  };
  return (
    <div className="grid min-w-0 gap-5" data-e2e={e2e}>
      <div className="flex min-w-0 flex-wrap items-end justify-between gap-x-6 gap-y-2 border-b border-line">
        <div role="tablist" aria-label={label} className="flex min-w-0 flex-wrap items-end gap-x-1 gap-y-1" onKeyDown={onKey}>
          {tabs.map((t) => {
            const on = cur?.id === t.id;
            return (
              <button key={t.id} type="button" role="tab" aria-selected={on} tabIndex={on ? 0 : -1} title={t.title} onClick={() => pick(t.id)} className={tabCls(on)} data-e2e="main-tab" data-tab={t.id}>
                {t.label}
                {t.count !== undefined && t.count !== null && <span className={on ? "text-muted" : "text-quiet"}>{t.count}</span>}
              </button>
            );
          })}
        </div>
        {right && <div className="flex min-w-0 flex-wrap items-center gap-x-5 gap-y-1 pb-1">{right}</div>}
      </div>
      {tabs.map((t) => (
        <div key={t.id} role="tabpanel" aria-label={t.label} hidden={t.id !== cur?.id} className="grid min-w-0 gap-8" data-e2e="main-pane" data-tab={t.id}>
          {t.body}
        </div>
      ))}
    </div>
  );
}
