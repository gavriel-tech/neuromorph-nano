"use client";

/* One live feed for every reader: the snapshot `GET /api/live`, kept fresh by the event stream
 * `GET /api/live/stream` (Server-Sent Events of timeline items). When the stream cannot be opened
 * or fails, the snapshot is polled every second instead, and its timeline is diffed for new
 * events, so readers see the same events either way. History present when the page opens is
 * never replayed as fresh activity. */
import { useEffect, useMemo, useRef, useSyncExternalStore } from "react";
import { apiBase, readCache, refresh, useApi } from "./api";
import { eventKey, mergeTimeline, parseLive, parseLiveEvent, type LiveEvent, type LiveReply } from "./livefeed";

export const LIVE_API = "/api/live";
export const LIVE_STREAM = "/api/live/stream";

/** How the feed reaches the engine: the stream, polling, or not yet known. */
export type FeedMode = "connecting" | "stream" | "polling";

type State = { mode: FeedMode; events: LiveEvent[]; primed: boolean; version: number };
let state: State = { mode: "connecting", events: [], primed: false, version: 0 };
const subs = new Set<() => void>();
const freshSubs = new Set<(e: LiveEvent) => void>();
const seen = new Set<string>();
let source: EventSource | null = null;
let users = 0;
let retry: ReturnType<typeof setTimeout> | null = null;
let snapTimer: ReturnType<typeof setTimeout> | null = null;

function set(next: Partial<State>) {
  state = { ...state, ...next, version: state.version + 1 };
  subs.forEach((f) => f());
}

/** Takes events in. `fresh` events not seen before are handed to the fresh-event readers once the
 *  feed is primed; the first batch only primes it (history). */
export function ingest(list: LiveEvent[], fresh: boolean): void {
  if (!list.length) {
    if (!state.primed) set({ primed: true });
    return;
  }
  const novel: LiveEvent[] = [];
  for (const e of list) {
    const k = eventKey(e);
    if (seen.has(k)) continue;
    seen.add(k);
    novel.push(e);
  }
  if (seen.size > 4000) {
    const keep = state.events.slice(-400).map(eventKey);
    seen.clear();
    keep.forEach((k) => seen.add(k));
    novel.forEach((e) => seen.add(eventKey(e)));
  }
  const primed = state.primed;
  if (novel.length || !primed) set({ events: mergeTimeline(state.events, novel), primed: true });
  if (primed && fresh) for (const e of novel) freshSubs.forEach((f) => f(e));
}

function snapSoon() {
  // A stream event changes the snapshot (active stages, writer queue): read it again, at most
  // a few times a second.
  if (snapTimer) return;
  snapTimer = setTimeout(() => {
    snapTimer = null;
    void refresh(LIVE_API);
  }, 250);
}

function open() {
  if (source || typeof window === "undefined" || typeof EventSource === "undefined") {
    if (typeof EventSource === "undefined" && state.mode !== "polling") set({ mode: "polling" });
    return;
  }
  let opened = false;
  try {
    source = new EventSource(apiBase() + LIVE_STREAM);
  } catch {
    source = null;
    set({ mode: "polling" });
    return;
  }
  source.onopen = () => {
    opened = true;
    if (state.mode !== "stream") set({ mode: "stream" });
  };
  // The engine names its events: `state` (the whole live state, once), `live` (one timeline
  // event) and `now` (the state, active stages, executing MicroModels and writer changed). An
  // unnamed message is read as timeline events too.
  const onEvents = (m: MessageEvent) => {
    let data: unknown = null;
    try {
      data = JSON.parse(m.data);
    } catch {
      return;
    }
    const list = (Array.isArray(data) ? data : [data]).map(parseLiveEvent).filter((e): e is LiveEvent => e !== null);
    ingest(list, true);
    snapSoon();
  };
  source.onmessage = onEvents;
  source.addEventListener("live", onEvents as EventListener);
  source.addEventListener("now", () => snapSoon());
  source.addEventListener("state", () => snapSoon());
  source.onerror = () => {
    // A stream that never opened (no route) or broke: poll instead, and try the stream again later.
    source?.close();
    source = null;
    if (state.mode !== "polling") set({ mode: "polling" });
    if (retry) clearTimeout(retry);
    if (users > 0)
      retry = setTimeout(
        () => {
          retry = null;
          if (users > 0) open();
        },
        opened ? 5_000 : 30_000,
      );
  };
}

function subscribe(cb: () => void) {
  subs.add(cb);
  users++;
  if (users === 1) open();
  return () => {
    subs.delete(cb);
    users--;
    if (users === 0) {
      source?.close();
      source = null;
      if (retry) clearTimeout(retry);
      retry = null;
      set({ mode: "connecting" });
    }
  };
}

export type LiveFeed = {
  reply: LiveReply | null;
  error: string | null;
  loading: boolean;
  /** When the snapshot was last read (ms since 1970), for live elapsed times. */
  at: number | null;
  mode: FeedMode;
  /** Every event seen, oldest first (the snapshot's timeline and the stream's events). */
  events: LiveEvent[];
};

const noSubscribe = () => () => {};

/** How often the snapshot is read: every second while the stream is down, every few seconds while
 *  the stream carries the events, every half minute while the engine answers that it has no such
 *  route (an older build), so a missing route is not asked for every second. */
export function pollEvery(mode: FeedMode, lastError: string | null): number {
  if (lastError && /^404\b/.test(lastError)) return 30_000;
  return mode === "stream" ? 3000 : 1000;
}

/** The live feed. `enabled` false reads nothing (a drawing without live parts). */
export function useLive(enabled = true): LiveFeed {
  const st = useSyncExternalStore(
    enabled ? subscribe : noSubscribe,
    () => state,
    () => state,
  );
  const q = useApi<unknown>(enabled ? LIVE_API : null, pollEvery(st.mode, readCache<unknown>(LIVE_API).error));
  const reply = useMemo(() => (q.data ? parseLive(q.data) : null), [q.data]);
  const tl = reply?.timeline;
  useEffect(() => {
    if (tl) ingest(tl, true);
  }, [tl]);
  return { reply, error: q.error, loading: q.loading, at: q.at, mode: st.mode, events: st.events.length ? st.events : (tl ?? []) };
}

/** Calls `onEvent` for each fresh event (seen after the page opened), from the stream or from the
 *  polled timeline. */
export function useLiveEvents(onEvent: (e: LiveEvent) => void): void {
  const ref = useRef(onEvent);
  useEffect(() => {
    ref.current = onEvent;
  }, [onEvent]);
  useEffect(() => {
    const f = (e: LiveEvent) => ref.current(e);
    freshSubs.add(f);
    return () => {
      freshSubs.delete(f);
    };
  }, []);
}

/** Clears the feed for unit tests. */
export function _resetLiveForTest(): void {
  source?.close();
  source = null;
  users = 0;
  seen.clear();
  subs.clear();
  freshSubs.clear();
  state = { mode: "connecting", events: [], primed: false, version: 0 };
}

/** Adds a fresh-event reader outside React (tests). */
export function onFresh(f: (e: LiveEvent) => void): () => void {
  freshSubs.add(f);
  return () => freshSubs.delete(f);
}
