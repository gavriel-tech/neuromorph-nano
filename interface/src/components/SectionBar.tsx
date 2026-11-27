"use client";

// 2026-11-26: the sticky section bar of the long pages (first made for Settings on 2026-11-25:
// "flow and visual hierarchy", directions of Luis Jake Gabriel III). One view per section instead
// of one long scroll; the bar stays in reach while scrolling; the section is in the address
// (`/models/#downloads`), so a link opens it, and the last one chosen is kept in this browser.

import { useState, useSyncExternalStore } from "react";
import { setHash, useHash } from "@/lib/hash";
import { readPref, writePref } from "@/lib/prefs";

export type Section = { id: string; label: string; about?: string };

/** The section shown: the one the address names, else the one picked on this page, else the one
 *  kept in this browser, else the first. `pick` changes it, writes the address and keeps it. */
export function useSection<T extends Section>(sections: readonly T[], storageKey: string, notRemembered: readonly string[] = []): { id: T["id"]; about: string; pick: (id: T["id"]) => void; inAddress: string | null } {
  const hash = useHash();
  const inAddress = hash.replace(/^#/, "") || null;
  const stored = useSyncExternalStore(
    (cb) => {
      window.addEventListener("storage", cb);
      return () => window.removeEventListener("storage", cb);
    },
    () => {
      try {
        return readPref(storageKey);
      } catch {
        return null;
      }
    },
    () => null,
  );
  const [picked, setPicked] = useState<string | null>(null);
  const fromAddress = sections.find((s) => s.id === inAddress)?.id;
  const id = (fromAddress ?? picked ?? sections.find((s) => s.id === stored && !notRemembered.includes(s.id))?.id ?? sections[0]?.id ?? "") as T["id"];
  const pick = (next: T["id"]) => {
    setPicked(next);
    setHash(`#${next}`, false);
    try {
      // A section the page does not reopen by itself (an explainer) is not kept as the last choice.
      writePref(storageKey, notRemembered.includes(next) ? null : next);
    } catch {
      // storage unavailable: the choice holds for this visit
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  return { id, about: sections.find((s) => s.id === id)?.about ?? "", pick, inAddress };
}

/** The bar itself: one button per section, the current one underlined in the sensory colour.
 *  Arrow keys move between sections; Enter or Space picks. */
export default function SectionBar<T extends Section>({ sections, value, onPick, label }: { sections: readonly T[]; value: T["id"]; onPick: (id: T["id"]) => void; label: string }) {
  return (
    <nav aria-label={label} className="sticky top-14 z-30 -mx-4 mb-8 border-b border-line bg-bg/95 px-4 backdrop-blur lg:top-0" data-e2e="section-bar">
      <ul
        className="flex gap-1 overflow-x-auto [scrollbar-width:none]"
        onKeyDown={(e) => {
          if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
          const i = sections.findIndex((s) => s.id === value);
          const next = sections[(i + (e.key === "ArrowRight" ? 1 : sections.length - 1)) % sections.length];
          if (next) {
            e.preventDefault();
            onPick(next.id);
            const el = (e.currentTarget as HTMLElement).querySelector<HTMLButtonElement>(`[data-section="${next.id}"]`);
            el?.focus();
          }
        }}
      >
        {sections.map((v) => {
          const on = v.id === value;
          return (
            <li key={v.id} className="shrink-0">
              <a
                href={`#${v.id}`}
                data-section={v.id}
                aria-current={on ? "page" : undefined}
                onClick={(e) => {
                  e.preventDefault();
                  onPick(v.id);
                }}
                className={`press mono relative inline-flex min-h-11 items-center px-3 text-[11px] uppercase tracking-[0.1em] ${on ? "text-text" : "text-faint hover:text-text"}`}
              >
                {v.label}
                {on && <span aria-hidden className="absolute right-3 bottom-[-1px] left-3 h-[2px] bg-sense-pale" />}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

/** The line under the bar saying what the section holds. */
export function SectionAbout({ text }: { text: string }) {
  return text ? <p className="mono -mt-4 mb-8 text-[11px] text-faint">{text}</p> : null;
}
