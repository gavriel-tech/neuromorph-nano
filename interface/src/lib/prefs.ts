/** The viewer's remembered choices (a tab, a section), read from localStorage once and then from
 *  memory. Views read them on every render through useSyncExternalStore, and a synchronous
 *  localStorage read per render was a measured cost on pages with many tabs (2026-11-27). A
 *  "storage" event (another tab, or this tab's own notice after a write) drops the cached value. */
const cache = new Map<string, string | null>();
let listening = false;

function listen() {
  if (listening || typeof window === "undefined") return;
  listening = true;
  window.addEventListener("storage", (e) => {
    if (e.key === null) cache.clear();
    else cache.delete(e.key);
  });
}

export function readPref(key: string): string | null {
  listen();
  if (cache.has(key)) return cache.get(key) ?? null;
  let v: string | null = null;
  try {
    v = window.localStorage.getItem(key);
  } catch {
    v = null;
  }
  cache.set(key, v);
  return v;
}

/** Writes (or with null removes) a remembered choice; throws when storage is unavailable, as
 *  localStorage does, after keeping the value for this visit. */
export function writePref(key: string, value: string | null) {
  listen();
  cache.set(key, value);
  if (value === null) window.localStorage.removeItem(key);
  else window.localStorage.setItem(key, value);
}
