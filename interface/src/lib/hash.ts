"use client";

import { useSyncExternalStore } from "react";

/** The URL hash as React state. Changes made here push a history entry, so the browser's back
 *  button steps back through the topology too. */
const subs = new Set<() => void>();

function subscribe(cb: () => void) {
  subs.add(cb);
  window.addEventListener("hashchange", cb);
  window.addEventListener("popstate", cb);
  return () => {
    subs.delete(cb);
    window.removeEventListener("hashchange", cb);
    window.removeEventListener("popstate", cb);
  };
}

export function useHash(): string {
  return useSyncExternalStore(
    subscribe,
    () => window.location.hash,
    () => "",
  );
}

export function setHash(hash: string, push = true) {
  if (window.location.hash === hash) return;
  const url = window.location.pathname + window.location.search + hash;
  if (push) window.history.pushState(window.history.state, "", url);
  else window.history.replaceState(window.history.state, "", url);
  subs.forEach((f) => f());
}
