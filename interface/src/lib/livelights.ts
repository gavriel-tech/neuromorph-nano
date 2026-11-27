"use client";

/* Live lights for any drawing with a live layer (the topology, the anatomy floorplan): each real
 * event of `/api/live` lights the parts it names (ganglion level, tentacle, MicroModel slot,
 * writer and memory hierarchy, homeostasis on a preemption), and the stages the snapshot says are
 * running stay lit while they run. Nothing lights without an event from the engine. */
import { useCallback, useEffect, useRef } from "react";
import { engineServing, holdFor, partIdsFor, type LiveEvent, type LiveReply } from "./livefeed";
import { useLive, useLiveEvents } from "./livehook";
import type { LiveLayer } from "./livelayer";
import { resolvePart, type Tree } from "./topology";

/** The drawn keys an event lights: for each anatomy id it names, the deepest drawn part that
 *  holds it. `has` says whether a key is drawn. */
export function liveKeys(tree: Tree | null, has: (k: string) => boolean, e: Pick<LiveEvent, "stage" | "kind" | "model" | "summary"> & { label?: string | null; tentacles?: number[] | null }): Set<string> {
  const out = new Set<string>();
  if (!tree) return out;
  for (const id of partIdsFor(e))
    for (const n of resolvePart(tree, id))
      for (let i = n.path.length; i >= 1; i--) {
        const k = n.path.slice(0, i).join("/");
        if (has(k)) {
          out.add(k);
          break;
        }
      }
  return out;
}

/** What the snapshot says is running now, as drawn keys with the reading behind each: the stages
 *  in `active` (`stage:<name>`), each MicroModel the engine reports executing a forward
 *  (`exec:<role>`), and the Writer while its idle session is open (`writer:session`). Empty when
 *  nothing runs; with no engine serving only the interface's own stages (a lab build, a backup). */
export function snapshotLights(reply: LiveReply | null, keysOf: (e: Pick<LiveEvent, "stage" | "kind" | "model" | "summary">) => Set<string>): Map<string, string> {
  const want = new Map<string, string>();
  if (!reply) return want;
  for (const a of reply.active) for (const k of keysOf({ stage: a.stage, kind: "", model: a.model, summary: "" })) want.set(k, `stage:${a.stage}`);
  if (engineServing(reply) === false) return want;
  for (const m of reply.models) if (m.executing) for (const k of keysOf({ stage: "", kind: "exec", model: m.role, summary: "" })) if (!want.has(k)) want.set(k, `exec:${m.role}`);
  if (reply.writer?.in_session) for (const k of keysOf({ stage: "", kind: "exec", model: "writer", summary: "" })) if (!want.has(k)) want.set(k, "writer:session");
  return want;
}

/** Lights what the snapshot says runs now and settles what it no longer reports. `held` is the
 *  set lit by the snapshot before; the new set is returned. The hold reaches a little past the
 *  next snapshot, so a part that keeps running stays lit without a gap. */
export function applySnapshot(lv: LiveLayer, want: Map<string, string>, held: Map<string, string>, streaming: boolean): Map<string, string> {
  for (const [k, why] of want) lv.light(k, streaming ? 3500 : 1500, why);
  for (const k of held.keys()) if (!want.has(k)) lv.release(k);
  return want;
}

/** Returns the `onLive` callback for a SceneView; the layer it receives is lit by live events. */
export function useLiveLights(tree: Tree | null): (l: LiveLayer | null) => void {
  const layer = useRef<LiveLayer | null>(null);
  const onLive = useCallback((l: LiveLayer | null) => {
    layer.current = l;
  }, []);
  const { reply, mode } = useLive(tree !== null);
  const onEvent = useCallback(
    (e: LiveEvent) => {
      const lv = layer.current;
      if (!lv) return;
      const hold = holdFor(e);
      for (const k of liveKeys(tree, (x) => lv.has(x), e)) lv.light(k, hold, `stage:${e.stage}`);
    },
    [tree],
  );
  useLiveEvents(onEvent);
  const streaming = mode === "stream";
  const held = useRef(new Map<string, string>());
  useEffect(() => {
    const lv = layer.current;
    if (!lv) return;
    held.current = applySnapshot(lv, snapshotLights(reply, (e) => liveKeys(tree, (x) => lv.has(x), e)), held.current, streaming);
  }, [reply, tree, streaming]);
  return onLive;
}
