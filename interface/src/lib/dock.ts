"use client";

import { useCallback, useEffect, useLayoutEffect, useState, type RefObject } from "react";
import type { Cartridge } from "./api";

const reduced = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** One cartridge seating: the source (its card, or the tray) lifts 2 px, travels and changes
 *  shape into the bar that spans its layers (480 ms on the dock spring, the one bounce in the
 *  product), then the covered layers take their ticks from the middle outward. FLIP: the bar is
 *  drawn in place, and animated from the source's box back to its own. Transform and opacity only. */
function seat(bar: Element, src: Element) {
  const el = bar as SVGGElement;
  const a = src.getBoundingClientRect();
  const b = el.getBoundingClientRect();
  if (!b.width || !b.height) return;
  if (reduced()) {
    el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160, easing: "linear" });
    return;
  }
  const sx = a.width / b.width;
  const sy = a.height / b.height;
  const dx = a.left - b.left;
  const dy = a.top - b.top;
  el.style.transformBox = "fill-box";
  el.style.transformOrigin = "0 0";
  el.animate(
    [
      { transform: `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})`, opacity: 0.6, easing: cssVar("--ease", "ease-out") },
      { transform: `translate(${dx}px, ${dy - 2}px) scale(${sx}, ${sy})`, opacity: 1, offset: 0.2, easing: cssVar("--ease-dock", "cubic-bezier(0.34, 1.3, 0.64, 1)") },
      { transform: "none", opacity: 1 },
    ],
    { duration: 600, easing: "linear" },
  );
  const ticks = [...el.querySelectorAll(".dock-tick")];
  const mid = (ticks.length - 1) / 2;
  ticks.forEach((t, i) => {
    (t as SVGElement).animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160, delay: 600 + Math.abs(i - mid) * 10, fill: "backwards", easing: "linear" });
  });
}

/** Watches the cartridges for one that becomes loaded (it docks) or unloaded (it leaves, 260 ms
 *  ease-in). Nothing plays on the first reply: the view starts still. Returns the cartridges that
 *  are leaving, so they stay drawn until they have gone. */
export function useDocking(root: RefObject<Element | null>, carts: Cartridge[] | null): Cartridge[] {
  const key = carts ? carts.filter((c) => c.loaded).map((c) => c.id).sort().join("|") : null;
  const [st, setSt] = useState<{ key: string | null; loaded: Map<string, Cartridge> | null; arriving: string[]; leaving: Cartridge[] }>({ key: null, loaded: null, arriving: [], leaving: [] });
  if (carts && key !== st.key) {
    const now = new Map(carts.filter((c) => c.loaded).map((c) => [c.id, c]));
    if (st.loaded === null) setSt({ key, loaded: now, arriving: [], leaving: [] });
    else {
      const arriving = [...now.keys()].filter((id) => !st.loaded!.has(id));
      const left = [...st.loaded.values()].filter((c) => !now.has(c.id));
      setSt({ key, loaded: now, arriving, leaving: [...st.leaving.filter((c) => !now.has(c.id)), ...left] });
    }
  }
  useLayoutEffect(() => {
    const r = root.current;
    if (!r || !st.arriving.length) return;
    for (const id of st.arriving) {
      const bars = r.querySelectorAll(`[data-cart="${CSS.escape(id)}"]`);
      const src = r.querySelector(`[data-cart-card="${CSS.escape(id)}"]`) ?? r.querySelector("[data-dock-src]");
      if (src) bars.forEach((b) => seat(b, src));
    }
  }, [st.arriving, root]);
  useEffect(() => {
    if (!st.leaving.length) return;
    const t = window.setTimeout(() => setSt((s) => ({ ...s, leaving: [] })), 300);
    return () => clearTimeout(t);
  }, [st.leaving]);
  return st.leaving;
}

/** The same watch for a canvas view: cartridges that became loaded since the view started are
 *  `arriving` until the view calls `seated` for them (after drawing them travel to their dock).
 *  Nothing arrives on the first reply: the view starts still. */
export function useDockQueue(carts: Cartridge[] | null): { arriving: string[]; seated: (id: string) => void } {
  const key = carts ? carts.filter((c) => c.loaded).map((c) => c.id).sort().join("|") : null;
  const [st, setSt] = useState<{ key: string | null; loaded: Set<string> | null; arriving: string[] }>({ key: null, loaded: null, arriving: [] });
  if (carts && key !== st.key) {
    const now = new Set(carts.filter((c) => c.loaded).map((c) => c.id));
    if (st.loaded === null) setSt({ key, loaded: now, arriving: [] });
    else setSt({ key, loaded: now, arriving: [...st.arriving.filter((id) => now.has(id)), ...[...now].filter((id) => !st.loaded!.has(id) && !st.arriving.includes(id))] });
  }
  const seated = useCallback((id: string) => setSt((s) => ({ ...s, arriving: s.arriving.filter((x) => x !== id) })), [setSt]);
  return { arriving: st.arriving, seated };
}
