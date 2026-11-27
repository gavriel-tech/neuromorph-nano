"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The main core of the pages should be on actual work
// flow and core function visuals ... the right menu are libraries and or discription tools etc."):
// with `panel="rail"` the organism fills the middle and the picked part's details open in the page's
// side panel (Workspace), not beside the map. Zoom and pan stay ("the topology and live activity
// visual space of the organizim has no move or zooming functions").

import Link from "next/link";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import DotText from "@/components/DotText";
import { LastCall, SystemsNow } from "@/components/LiveState";
import PartPanel from "@/components/PartPanel";
import SceneView from "@/components/SceneView";
import { FigureLoading, Placeholder } from "@/components/ui";
import { useRoleMap, type RoleMap } from "@/lib/roles";
import { useRail } from "@/components/Workspace";
import { msText, useActivity, useFade } from "@/lib/activity";
import { ago, useApi, type Audit, type AuditRow, type Cartridge, type Part, type Project, type Projects, type Status } from "@/lib/api";
import { bodyScene, LONGEST_JOB } from "@/lib/body";
import { useNow } from "@/lib/clock";
import { useDockQueue } from "@/lib/dock";
import { setHash, useHash } from "@/lib/hash";
import { isCall, lastBrainCall, LIVE_PATH, mark, sortRows, stepJob } from "@/lib/live";
import { SPEED, type LiveLayer } from "@/lib/livelayer";
import { engineServing, eventGroup, GROUP_NAMES, holdFor, liveLine, type LiveEvent } from "@/lib/livefeed";
import { applySnapshot, liveKeys, snapshotLights } from "@/lib/livelights";
import { useLive, useLiveEvents } from "@/lib/livehook";
import { droneSets, type PaintOpts } from "@/lib/paint";
import { callLabel } from "@/lib/privacy";
import { plateScene } from "@/lib/plates";
import { keyOf, polyLen, type Box, type Route, type Scene } from "@/lib/scene";
import { buildTree, figure, flowRanks, nodeRank, parseHash, pathHash, resolvePart, within, type Anatomy, type TNode, type Tree } from "@/lib/topology";

export { LIVE_PATH };
const ZOOM_MS = 480;
/** Opening a part: the layers in front of it thin out first, then the camera moves in. */
const THIN_MS = 240;

const reduced = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const related = (a: string, b: string) => a === b || a.startsWith(b + "/") || b.startsWith(a + "/");

function ease(): string {
  const v = typeof window === "undefined" ? "" : getComputedStyle(document.documentElement).getPropertyValue("--ease").trim();
  return v || "cubic-bezier(0.23, 1, 0.32, 1)";
}

/** The scene for a part: the organism in plan at the top, a plate for anything inside. */
export function buildScene(tree: Tree, focus: TNode, W: number, H: number, carts: Cartridge[], status: Status | null, drones?: Map<string, Set<string>>, projects?: Project[] | null): Scene {
  return focus.layout === "organism" ? bodyScene(tree, focus, W, H, carts, status, projects ?? null) : plateScene(tree, focus, W, H, carts, status, drones);
}

/** Camera keyframes between two framings: scale on a log scale so each doubling takes the same
 *  time, position following scale. */
function camFrames(k0: number, t0: { x: number; y: number }, k1: number, t1: { x: number; y: number }, op: (u: number) => number): Keyframe[] {
  const out: Keyframe[] = [];
  const N = 12;
  for (let i = 0; i <= N; i++) {
    const s = i / N;
    const k = Math.exp(Math.log(k0) + (Math.log(k1) - Math.log(k0)) * s);
    const u = k0 === k1 ? s : (k - k0) / (k1 - k0);
    out.push({ offset: s, transform: `translate(${t0.x + (t1.x - t0.x) * u}px, ${t0.y + (t1.y - t0.y) * u}px) scale(${k})`, opacity: op(s) });
  }
  return out;
}

/** The routes a signal takes from one lit part to the next: a direct route, else the shortest
 *  chain of routes (at most six) whose ends meet. */
function chain(routes: Route[], a: string, b: string): Route[] | null {
  const direct = routes.filter((r) => related(r.from, a) && related(r.to, b));
  if (direct.length) return [direct.sort((p, q) => Number(q.from === a) + Number(q.to === b) - Number(p.from === a) - Number(p.to === b))[0]];
  const starts = routes.filter((r) => related(r.from, a)).sort((p, q) => Number(q.from === a) - Number(p.from === a));
  const queue: Route[][] = starts.map((r) => [r]);
  const seen = new Set(starts.map((r) => r.to));
  while (queue.length) {
    const path = queue.shift()!;
    if (path.length > 6) break;
    const last = path[path.length - 1];
    if (related(last.to, b)) return path;
    for (const r of routes) {
      if (r.from !== last.to || seen.has(r.to)) continue;
      seen.add(r.to);
      queue.push([...path, r]);
    }
  }
  return null;
}

/** The deepest region of a scene whose part holds `target`, for the camera. */
function holderIn(scene: Scene, target: TNode): Box | null {
  let best: { key: string; box: Box } | null = null;
  for (const r of scene.regions) if (!r.key.includes("/.") && within(target, r.node) && (!best || r.key.length > best.key.length)) best = r;
  return best?.box ?? null;
}

type Readout = { id: string; tool: string; mode?: string; ms?: number; time: number; label?: string; whole: boolean; elsewhere: string[]; running: boolean; replay: boolean };

/* One tree for every reader: the topology, its side panel tabs and the pages' plates all read
 * the same replies, so the tree is built once per change of those replies, not once per reader. */
let treeMemo: { a: unknown; p: unknown; s: unknown; c: unknown; r: unknown; t: Tree } | null = null;
function sharedTree(anatomy: Anatomy, parts: Record<string, Part>, status: Status | null, carts: Cartridge[] | null, roles: RoleMap): Tree {
  const m = treeMemo;
  if (m && m.a === anatomy && m.p === parts && m.s === status && m.c === carts && m.r === roles) return m.t;
  // Kinds only from the engine's own role reply, never from the defaults used while it loads.
  const t = buildTree({ anatomy, parts, status, cartridges: carts, roleOf: roles.reported ? roles.roleOf : undefined });
  treeMemo = { a: anatomy, p: parts, s: status, c: carts, r: roles, t };
  return t;
}

/** A hovered part's title: a MicroModel names its kind and the open weights it runs on
 *  ("Embedder · qwen3-embedding-0.6b"); a Core Brain slot names the weights its role runs on now. */
export function tipTitle(n: TNode, roles: RoleMap): string {
  if (n.role === "model") return n.sub && n.sub !== "MicroModel" ? `${n.sub} · ${n.label}` : `MicroModel · ${n.label}`;
  const slot = n.ids.map((id) => /^nmbrain\.slot#(\w+)$/.exec(id)?.[1]).find(Boolean);
  if (slot && roles.reported) {
    const m = roles.modelOf(slot);
    if (m) return `${n.label} · ${m}`;
  }
  return n.label;
}

export function useTopology() {
  const settings = useApi<Record<string, unknown>>("/api/settings");
  const parts = useApi<Record<string, Part>>("/api/parts");
  const status = useApi<Status>("/api/status", 2000);
  const carts = useApi<Cartridge[]>("/api/cartridges", 2000);
  const live = useApi<Audit>(LIVE_PATH, 1000);
  const projects = useApi<Projects>("/api/projects", 10000);
  const roles = useRoleMap();
  const anatomy = (settings.data?.anatomy as Anatomy | undefined) ?? null;
  // The first tree waits for the first reply (or failure) of the status and the cartridges as
  // well, so the figure is built and painted once on arrival, not once per reply as they come in.
  const ready = !!anatomy && !!parts.data && !status.loading && !carts.loading && roles.answered;
  const tree = useMemo(() => (ready && anatomy && parts.data ? sharedTree(anatomy, parts.data, status.data, carts.data, roles) : null), [ready, anatomy, parts.data, status.data, carts.data, roles]);
  const error = settings.error ?? parts.error ?? (status.error && !status.data ? status.error : null);
  return { tree, roles, status: status.data, statusError: status.error, carts: carts.data, live: live.data, liveError: live.error, settings: settings.data, projects: projects.data?.projects ?? null, projectsRead: !projects.loading, error };
}

const NO_CARTS: Cartridge[] = [];

/** The live audit rows of the topology, oldest first. */
function useRows(live: Audit | null | undefined): AuditRow[] {
  return useMemo(() => sortRows((live?.audit ?? []) as AuditRow[]), [live]);
}

const goTo = (path: string[]) => setHash(pathHash(path));

/** A part's details for the page's side panel, read live (the panel keeps it mounted). */
export function PartRail({ path, onGo = goTo, topologyLink = false }: { path: string; onGo?: (path: string[]) => void; topologyLink?: boolean }) {
  const { tree, status, carts, live, settings, projects, error } = useTopology();
  const rows = useRows(live);
  const act = useActivity(tree);
  if (!tree) return error ? <div className="mono text-[11px] text-error">The anatomy did not answer ({error}).</div> : <Placeholder className="h-40" what="part" />;
  const node = tree.byPath.get(path);
  if (!node) return <div className="mono text-[11px] text-faint">This part is not in the anatomy the engine reports.</div>;
  return <PartPanel embedded topologyLink={topologyLink} node={node} tree={tree} status={status} carts={carts ?? null} rows={rows} settings={settings ?? null} projects={projects} onGo={onGo} act={act} />;
}

/** The systems of the organism and their state now, for a side panel tab. */
export function SystemsRail({ onGo = goTo }: { onGo?: (path: string[]) => void }) {
  const { tree, status, carts, live, projects, error } = useTopology();
  const rows = useRows(live);
  if (!tree) return error ? <div className="mono text-[11px] text-error">The anatomy did not answer ({error}).</div> : <Placeholder className="h-40" what="systems" />;
  return <SystemsNow flush tree={tree} rows={rows} status={status} carts={carts ?? null} projects={projects} onGo={onGo} />;
}

/** The running or last call as the path it took through the systems, for a side panel tab. */
export function LastCallRail() {
  const { tree, live, error } = useTopology();
  const rows = useRows(live);
  if (!tree) return error ? <div className="mono text-[11px] text-error">The anatomy did not answer ({error}).</div> : <Placeholder className="h-24" what="last call" />;
  return <LastCall flush tree={tree} rows={rows} />;
}

export default function Topology({ variant, panel = "side" }: { variant: "hero" | "full"; panel?: "side" | "rail" }) {
  const { tree, roles, status, statusError, carts, live, liveError, settings, projects, projectsRead, error } = useTopology();
  const cartList = carts ?? NO_CARTS;
  const hash = useHash();
  const now = useNow();
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ W: number; H: number } | null>(null);
  const sizeRef = useRef<{ W: number; H: number } | null>(null);
  useEffect(() => {
    sizeRef.current = size;
  }, [size]);
  const { arriving, seated } = useDockQueue(carts);

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    // While the window is being resized the kept picture is scaled; it is drawn again, sharp,
    // once the size has settled.
    let t = 0;
    const ro = new ResizeObserver(([e]) => {
      const W = Math.round(e.contentRect.width);
      const H = Math.round(e.contentRect.height);
      const apply = () => setSize((s) => (s && s.W === W && s.H === H ? s : W > 0 && H > 0 ? { W, H } : s));
      clearTimeout(t);
      if (!sizeRef.current) apply();
      else t = window.setTimeout(apply, 140);
    });
    ro.observe(el);
    return () => {
      clearTimeout(t);
      ro.disconnect();
    };
  }, []);

  /* Selection from the hash: the selected node, and the level in view (the node itself when it
   * has parts inside, else its parent). */
  const { sel, focus } = useMemo(() => {
    if (!tree) return { sel: null as TNode | null, focus: null as TNode | null };
    let n = tree.root;
    for (const k of parseHash(hash)) {
      const c = n.children.find((x) => x.key === k);
      if (!c) break;
      n = c;
    }
    return { sel: n, focus: n.children.length ? n : (n.parent ?? tree.root) };
  }, [tree, hash]);
  const focusKey = focus?.path.join("/") ?? null;

  const [anim, setAnim] = useState<{ key: string; prev: string | null; seq: number } | null>(null);
  if (focusKey !== null && anim === null) setAnim({ key: focusKey, prev: null, seq: 0 });
  else if (focusKey !== null && anim && anim.key !== focusKey) setAnim({ key: focusKey, prev: anim.key, seq: anim.seq + 1 });

  const W = size?.W ?? 0;
  const H = size?.H ?? 0;
  const drones = useMemo(() => droneSets(settings), [settings]);
  const scene = useMemo(() => (tree && focus && W && projectsRead ? buildScene(tree, focus, W, H, cartList, status, drones, projects) : null), [tree, focus, W, H, cartList, status, drones, projects, projectsRead]);
  const prevNode = anim?.prev != null ? (tree?.byPath.get(anim.prev) ?? null) : null;
  const prevScene = useMemo(() => (tree && prevNode && W ? buildScene(tree, prevNode, W, H, cartList, status, drones, projects) : null), [tree, prevNode, W, H, cartList, status, drones, projects]);
  const hide = useMemo(() => new Set(arriving), [arriving]);
  const act = useActivity(tree);
  const fade = useFade(act);
  const paint: PaintOpts = useMemo(() => ({ status, carts: cartList, drones, hide, act, fade }), [status, cartList, drones, hide, act, fade]);

  const curLayer = useRef<HTMLDivElement>(null);
  const oldLayer = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<Scene | null>(null);
  const liveRef = useRef<LiveLayer | null>(null);
  const keyboardNav = useRef<string | null>(null);
  useEffect(() => {
    sceneRef.current = scene;
  }, [scene]);
  const onLive = useCallback((l: LiveLayer | null) => {
    liveRef.current = l;
  }, []);

  /* The cartridge moment: each newly loaded cartridge travels from the rack to its dock. */
  const started = useRef(new Set<string>());
  useEffect(() => {
    for (const id of arriving) {
      if (started.current.has(id)) continue;
      started.current.add(id);
      const l = liveRef.current;
      if (l) l.dock(id, () => seated(id));
      else seated(id);
    }
  }, [arriving, seated, scene]);

  /* The drill-down: one continuous camera move. Opening a part, the new view grows out of the
   * part's place while the old view is flown into; going back is the same move reversed. */
  useLayoutEffect(() => {
    if (!anim || anim.prev === null || !tree || !W) return;
    const cur = curLayer.current;
    const old = oldLayer.current;
    const curNode = tree.byPath.get(anim.key);
    const oldNode = tree.byPath.get(anim.prev);
    if (!cur || !curNode || !oldNode) return;
    const done = () => setAnim((a) => (a && a.seq === anim.seq ? { ...a, prev: null } : a));
    if (reduced()) {
      cur.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160, easing: "linear" });
      old?.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 160, easing: "linear", fill: "forwards" }).finished.then(done, done);
      return;
    }
    let r: Box | null = null;
    let dir: "in" | "out" | "cross" = "cross";
    if (within(curNode, oldNode)) {
      r = holderIn(buildScene(tree, oldNode, W, H, cartList, status, drones, projects), curNode);
      dir = r ? "in" : "cross";
    } else if (within(oldNode, curNode)) {
      r = holderIn(buildScene(tree, curNode, W, H, cartList, status, drones, projects), oldNode);
      dir = r ? "out" : "cross";
    }
    const opts: KeyframeAnimationOptions = { duration: ZOOM_MS, easing: ease() };
    let fin: Animation | undefined;
    if (r && dir !== "cross") {
      const cx = r.x + r.w / 2;
      const cy = r.y + r.h / 2;
      const k = clamp(Math.max(r.w / W, r.h / H), 0.04, 0.7);
      const smallT = { x: cx - (W / 2) * k, y: cy - (H / 2) * k };
      const largeT = { x: W / 2 - cx / k, y: H / 2 - cy / k };
      const zero = { x: 0, y: 0 };
      const fadeIn = (s: number) => clamp((s - 0.25) / 0.55, 0, 1);
      const fadeOut = (s: number) => 1 - clamp(s / 0.6, 0, 1);
      if (dir === "in") {
        // Through the skin: what lies in front of the part thins out and its anatomy comes
        // forward, then the camera moves in and the part opens.
        const thinCanvas = old?.querySelector<HTMLCanvasElement>("canvas[data-thin]");
        thinCanvas?.animate([{ opacity: 0 }, { opacity: 1 }], { duration: THIN_MS, easing: "linear", fill: "forwards" });
        const late = { ...opts, delay: thinCanvas ? THIN_MS * 0.8 : 0 };
        cur.animate(camFrames(k, smallT, 1, zero, fadeIn), { ...late, fill: "backwards" });
        fin = old?.animate(camFrames(1, zero, 1 / k, largeT, fadeOut), { ...late, fill: "forwards" });
      } else {
        cur.animate(camFrames(1 / k, largeT, 1, zero, fadeIn), opts);
        fin = old?.animate(camFrames(1, zero, k, smallT, fadeOut), { ...opts, fill: "forwards" });
      }
    } else {
      cur.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 200, easing: ease() });
      fin = old?.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 160, easing: ease(), fill: "forwards" });
    }
    if (fin) fin.finished.then(done, done);
    else done();
    const safety = window.setTimeout(done, ZOOM_MS + THIN_MS + 160);
    const from = keyboardNav.current;
    keyboardNav.current = null;
    if (from !== null) {
      requestAnimationFrame(() => {
        const back = dir === "out" ? keyOf(oldNode) : null;
        const target = (back ? cur.querySelector<SVGGElement>(`[data-key="${CSS.escape(back)}"][tabindex="0"]`) : null) ?? cur.querySelector<SVGGElement>('.t-node[tabindex="0"]');
        target?.focus({ preventScroll: true });
      });
    }
    return () => clearTimeout(safety);
  }, [anim?.seq]); // eslint-disable-line react-hooks/exhaustive-deps

  /* Navigation */
  const go = useCallback((path: string[], viaKeyboard = false) => {
    if (viaKeyboard) keyboardNav.current = path.join("/");
    setHash(pathHash(path));
  }, []);
  const up = useCallback(
    (viaKeyboard = false) => {
      if (!sel || !focus) return;
      if (sel !== focus) go(focus.path, viaKeyboard);
      else if (focus.parent) go(focus.parent.path, viaKeyboard);
    },
    [sel, focus, go],
  );
  /* With the side panel: the picked part's details open there; picking it again brings them back
   * after the panel's detail was closed. */
  const rail = useRail();
  const railRef = useRef(rail);
  useEffect(() => {
    railRef.current = rail;
  });
  const selKey = sel ? sel.path.join("/") : null;
  const selLabel = sel?.label ?? "";
  const [reshow, setReshow] = useState(0);
  useEffect(() => {
    if (panel !== "rail" || selKey === null) return;
    const r = railRef.current;
    if (selKey === "") {
      if (r.shown?.key?.startsWith("part:")) r.clear();
      return;
    }
    r.show({ key: `part:${selKey}`, title: selLabel, body: <PartRail path={selKey} /> });
  }, [panel, selKey, selLabel, reshow]);
  const activate = useCallback(
    (key: string, viaKeyboard: boolean) => {
      const n = tree?.byPath.get(key);
      if (!n) return;
      if (panel === "rail" && key === selKey) setReshow((x) => x + 1);
      go(n.path, viaKeyboard);
    },
    [tree, go, panel, selKey],
  );
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      const a = document.activeElement;
      if (a && /^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName)) return;
      const inside = !!box.current?.contains(a) || a === document.body || variant === "full";
      if (!inside) return;
      e.preventDefault();
      up(!!box.current?.contains(a));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [up, variant]);

  /* Pinch and drag: a pinch outward zooms the kept picture and, past a point, opens the part
   * under the fingers; a pinch inward past the start goes back; a drag pans while zoomed. A
   * trackpad pinch (wheel with the control key) does the same. */
  const [view, setView] = useState({ k: 1, x: 0, y: 0 });
  const [viewFor, setViewFor] = useState<string | null>(null);
  if (viewFor !== focusKey) {
    setViewFor(focusKey);
    setView({ k: 1, x: 0, y: 0 });
  }
  type Gest = { pts: Map<number, { x: number; y: number }>; start: { d: number; cx: number; cy: number; k: number; x: number; y: number } | null; pan: { x: number; y: number; vx: number; vy: number } | null; moved: boolean; wheelT: number };
  const g = useRef<Gest>({ pts: new Map(), start: null, pan: null, moved: false, wheelT: 0 });
  const [gActive, setGActive] = useState(false);
  const viewRef = useRef(view);
  useEffect(() => {
    viewRef.current = view;
  }, [view]);
  const settle = useCallback(
    (v: { k: number; x: number; y: number }, cx: number, cy: number, opens = true) => {
      const sc = sceneRef.current;
      // 2026-11-26 (Luis Jake Gabriel III: "the topology and live activity visual space of the
      // organizim has no move or zooming functions"): the wheel and the zoom buttons only zoom and
      // pan; a pinch keeps opening the part under the fingers.
      if (!opens) {
        const next = v.k <= 1.02 ? { k: 1, x: 0, y: 0 } : { k: v.k, x: clamp(v.x, W - W * v.k, 0), y: clamp(v.y, H - H * v.k, 0) };
        // Quick repeated clicks build on each other, before the next render.
        viewRef.current = next;
        setView(next);
        return;
      }
      if (v.k > 2.4 && sc) {
        // The part under the pinch opens.
        const px = (cx - v.x) / v.k;
        const py = (cy - v.y) / v.k;
        let best: { key: string; area: number } | null = null;
        for (const r of sc.regions) {
          if (r.key.includes("/.") || !r.node.children.length) continue;
          const b = r.box;
          if (px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h && (!best || b.w * b.h < best.area)) best = { key: r.key, area: b.w * b.h };
        }
        setView({ k: 1, x: 0, y: 0 });
        if (best) activate(best.key, false);
        return;
      }
      if (v.k < 0.8) {
        setView({ k: 1, x: 0, y: 0 });
        up(false);
        return;
      }
      if (v.k <= 1.02) setView({ k: 1, x: 0, y: 0 });
      else setView({ k: v.k, x: clamp(v.x, W - W * v.k, 0), y: clamp(v.y, H - H * v.k, 0) });
    },
    [activate, up, W, H],
  );
  const localXY = (e: { clientX: number; clientY: number }) => {
    const r = box.current?.getBoundingClientRect();
    return { x: e.clientX - (r?.left ?? 0), y: e.clientY - (r?.top ?? 0) };
  };
  const gesture = {
    down: (e: React.PointerEvent<HTMLDivElement>) => {
      const s = g.current;
      s.pts.set(e.pointerId, localXY(e));
      s.moved = false;
      if (s.pts.size === 2) {
        const [a, b] = [...s.pts.values()];
        const v = viewRef.current;
        s.start = { d: Math.hypot(a.x - b.x, a.y - b.y) || 1, cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2, k: v.k, x: v.x, y: v.y };
        s.pan = null;
        setGActive(true);
      } else if (s.pts.size === 1 && viewRef.current.k > 1.01) {
        const p = localXY(e);
        s.pan = { x: p.x, y: p.y, vx: viewRef.current.x, vy: viewRef.current.y };
      }
    },
    move: (e: React.PointerEvent<HTMLDivElement>) => {
      const s = g.current;
      if (!s.pts.has(e.pointerId)) return;
      s.pts.set(e.pointerId, localXY(e));
      if (s.start && s.pts.size >= 2) {
        const [a, b] = [...s.pts.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y) || 1;
        const k = clamp((s.start.k * d) / s.start.d, 0.6, 4);
        const cx = (a.x + b.x) / 2;
        const cy = (a.y + b.y) / 2;
        // The point under the fingers stays under them.
        const px = (s.start.cx - s.start.x) / s.start.k;
        const py = (s.start.cy - s.start.y) / s.start.k;
        s.moved = true;
        setView({ k, x: cx - px * k, y: cy - py * k });
      } else if (s.pan) {
        const pan = s.pan;
        const p = localXY(e);
        const dx = p.x - pan.x;
        const dy = p.y - pan.y;
        if (!s.moved && Math.hypot(dx, dy) < 6) return;
        s.moved = true;
        setGActive(true);
        setView((v) => ({ ...v, x: pan.vx + dx, y: pan.vy + dy }));
      }
    },
    up: (e: React.PointerEvent<HTMLDivElement>) => {
      const s = g.current;
      const wasPinch = !!s.start && s.pts.size >= 2;
      const pts = [...s.pts.values()];
      s.pts.delete(e.pointerId);
      if (wasPinch) {
        const cx = pts.reduce((a, p) => a + p.x, 0) / pts.length;
        const cy = pts.reduce((a, p) => a + p.y, 0) / pts.length;
        s.start = null;
        setGActive(false);
        settle(viewRef.current, cx, cy);
      } else if (s.pan && s.pts.size === 0) {
        s.pan = null;
        setGActive(false);
        settle(viewRef.current, W / 2, H / 2, false);
      }
    },
    wheel: (e: { deltaY: number; clientX: number; clientY: number }) => {
      const s = g.current;
      const p = localXY(e);
      const v = viewRef.current;
      const k = clamp(v.k * Math.exp(-e.deltaY * 0.0025), 1, 6);
      const px = (p.x - v.x) / v.k;
      const py = (p.y - v.y) / v.k;
      setGActive(true);
      setView({ k, x: p.x - px * k, y: p.y - py * k });
      clearTimeout(s.wheelT);
      s.wheelT = window.setTimeout(() => {
        setGActive(false);
        settle(viewRef.current, p.x, p.y, false);
      }, 260);
    },
    click: (e: React.MouseEvent<HTMLDivElement>) => {
      // A drag or a pinch is not a click.
      if (g.current.moved) {
        e.stopPropagation();
        e.preventDefault();
        g.current.moved = false;
      }
    },
  };

  const zoomBy = useCallback(
    (f: number) => {
      const v = viewRef.current;
      const k = clamp(v.k * f, 1, 6);
      const px = (W / 2 - v.x) / v.k;
      const py = (H / 2 - v.y) / v.k;
      settle({ k, x: W / 2 - px * k, y: H / 2 - py * k }, W / 2, H / 2, false);
    },
    [settle, W, H],
  );
  // The wheel is taken by the picture (not the page) only while the pointer is over it.
  const wheelRef = useRef(gesture.wheel);
  useEffect(() => {
    wheelRef.current = gesture.wheel;
  });
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      wheelRef.current(e);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [scene]);

  /* Hover: the plain name, one number, and the engine id, quiet. */
  const [hover, setHover] = useState<{ key: string; x: number; y: number } | null>(null);
  const onHover = useCallback((h: { key: string; x: number; y: number } | null) => setHover((p) => (h && p && p.key === h.key ? p : h)), []);

  /* ---------------- Live: each call lights the parts it used, in order ---------------- */
  const rows = useRows(live);
  const ranks = useMemo(() => (tree ? flowRanks(tree) : new Map<string, number>()), [tree]);
  const [readout, setReadout] = useState<Readout | null>(null);
  /* The job each tentacle carried in the call being played, when the engine reports it. */
  const [jobs, setJobs] = useState<{ id: string; map: Map<string, string> } | null>(null);
  const timers = useRef<number[]>([]);

  const play = useCallback(
    (row: AuditRow, replay = false) => {
      const sc = sceneRef.current;
      const lv = liveRef.current;
      if (!tree || !sc || !lv) return;
      timers.current.forEach(clearTimeout);
      timers.current = [];
      lv.releaseHeld();
      const running = row.state === "running";
      const named = Array.isArray(row.steps) && row.steps.length ? row.steps.map((s) => s.part) : Array.isArray(row.parts) ? row.parts : [];
      // The deepest drawn part for a node.
      const nearest = (n: TNode) => {
        for (let i = n.path.length; i >= 1; i--) {
          const k = n.path.slice(0, i).join("/");
          if (lv.has(k)) return k;
        }
        return null;
      };
      const elsewhere = new Set<string>();
      const steps = new Map<number, Set<string>>();
      // The part id the call named for each drawn key, so each light says what lit it.
      const whyOf = new Map<string, string>();
      named.forEach((p, idx) => {
        for (const n of resolvePart(tree, p)) {
          const k = nearest(n);
          if (!k) {
            const main = tree.byPath.get(n.path[0] ?? "");
            if (main && !within(sc.focus, main)) elsewhere.add(main.label);
            continue;
          }
          // With steps from the engine, their run order is the order; else the anatomy's flow.
          const r = row.steps?.length ? (row.steps[idx].start_ms ?? idx) : nodeRank(n, ranks);
          if (!steps.has(r)) steps.set(r, new Set());
          steps.get(r)!.add(k);
          if (!whyOf.has(k)) whyOf.set(k, p);
        }
      });
      const ordered = [...steps.entries()].sort((a, b) => a[0] - b[0]).map(([, s]) => [...s]);
      const groups: string[][] = [];
      for (const g of ordered) {
        const last = groups[groups.length - 1];
        if (last && last.length === g.length && last.every((k) => g.includes(k))) continue;
        groups.push(g);
      }
      let t = 0;
      const startAt: number[] = [];
      for (let i = 0; i < groups.length; i++) {
        startAt.push(t);
        if (i === groups.length - 1) break;
        let travel = 0;
        const doneSeg = new Set<string>();
        let pairs = 0;
        for (const a of groups[i])
          for (const b of groups[i + 1]) {
            if (a === b || pairs > 24) continue;
            pairs++;
            const path = chain(sc.routes, a, b);
            if (!path) continue;
            let off = 0;
            for (const seg of path) {
              const dur = Math.max(160, (polyLen(seg.pts) / SPEED) * 1000);
              const id = `${seg.from}>${seg.to}@${Math.round(off)}`;
              if (!doneSeg.has(id) && !reduced()) {
                doneSeg.add(id);
                lv.pulse(seg, t + off + 40);
              }
              off += dur;
            }
            travel = Math.max(travel, off);
          }
        t += reduced() ? 200 : Math.max(220, travel + 40);
      }
      const end = t + 900;
      groups.forEach((g, i) => {
        timers.current.push(
          window.setTimeout(() => {
            for (const k of g) lv.light(k, running ? null : Math.max(500, (startAt[i + 1] ?? end) - startAt[i] + 500), `${replay ? "replay" : "call"}:${row.id}:${whyOf.get(k) ?? ""}`);
          }, startAt[i]),
        );
      });
      // A call that names no parts lights nothing: the readout says so. Only a part the engine
      // named is ever lit.
      setReadout({ id: row.id, tool: row.tool ?? "tool", mode: row.mode, ms: row.ms, time: row.time_ms, label: row.label, whole: named.length === 0, elsewhere: [...elsewhere], running, replay });
      const jm = new Map<string, string>();
      for (const st of row.steps ?? []) {
        const j = stepJob(st);
        const tn = j ? tree.tentacles.find((x) => x.key === `t${j.tentacle}`) : undefined;
        if (j && tn && !jm.has(keyOf(tn))) jm.set(keyOf(tn), sc.detail === 2 ? `T${j.tentacle + 1} · ${j.job.slice(0, LONGEST_JOB.length)}` : `T${j.tentacle + 1}`);
      }
      setJobs(jm.size ? { id: row.id, map: jm } : null);
      if (!running)
        timers.current.push(
          window.setTimeout(() => {
            setReadout((r) => (r && r.id === row.id ? null : r));
            setJobs((x) => (x && x.id === row.id ? null : x));
          }, end + 5000),
        );
    },
    [tree, ranks],
  );

  /* ---------------- Live: the engine's own stage events light their parts ----------------
   * Each event of `/api/live` (stream or polled timeline) lights the ganglion level, the tentacle
   * a read names, the MicroModel slot that ran it, the writer and memory hierarchy for an idle
   * writer pass (L1 to L4), the homeostasis for a preemption (a short pulse). The stages the
   * snapshot says are running stay lit while they run. Nothing lights without an event. */
  const liveFeed = useLive();
  const [liveTag, setLiveTag] = useState<{ text: string; key: string } | null>(null);
  const tagTimer = useRef(0);
  const keysFor = useCallback(
    (e: Pick<LiveEvent, "stage" | "kind" | "model" | "summary"> & { label?: string | null; tentacles?: number[] | null }) => {
      const lv = liveRef.current;
      return lv ? liveKeys(tree, (k) => lv.has(k), e) : new Set<string>();
    },
    [tree],
  );
  const onLiveEvent = useCallback(
    (e: LiveEvent) => {
      const lv = liveRef.current;
      if (!lv) return;
      const hold = holdFor(e);
      for (const k of keysFor(e)) lv.light(k, hold, `stage:${e.stage}`);
      const g = eventGroup(e);
      setLiveTag({ text: `${GROUP_NAMES[g]} · ${e.stage.replace(/_/g, " ")}${e.model ? ` · ${e.model}` : ""}${typeof e.ms === "number" ? ` · ${msText(e.ms)}` : ""}`, key: `${e.at_ms}|${e.stage}` });
      window.clearTimeout(tagTimer.current);
      tagTimer.current = window.setTimeout(() => setLiveTag(null), Math.max(2500, hold + 1500));
    },
    [keysFor],
  );
  useLiveEvents(onLiveEvent);
  const activeReply = liveFeed.reply;
  const streaming = liveFeed.mode === "stream";
  const snapHeld = useRef(new Map<string, string>());
  useEffect(() => {
    const lv = liveRef.current;
    if (!lv) return;
    // The stages running, each MicroModel executing a forward and the Writer's open idle session
    // stay lit while the snapshot reports them, and settle when it no longer does.
    snapHeld.current = applySnapshot(lv, snapshotLights(activeReply, keysFor), snapHeld.current, streaming);
  }, [activeReply, keysFor, streaming]);
  useEffect(() => () => window.clearTimeout(tagTimer.current), []);

  const playRef = useRef(play);
  useEffect(() => {
    playRef.current = play;
  }, [play]);

  const seen = useRef<{ mark: string | null; running: string | null }>({ mark: null, running: null });
  useEffect(() => {
    if (!rows.length || !tree) return;
    const newest = rows[rows.length - 1];
    const s = seen.current;
    if (s.mark === null) {
      // History is not replayed on arrival: the view starts still.
      s.mark = mark(newest);
      return;
    }
    // A call that was running has finished: its parts settle into the afterglow.
    if (s.running) {
      const same = rows.find((r) => r.id === s.running);
      if (!same || same.state !== "running") {
        s.running = null;
        if (same && mark(same) <= s.mark) {
          window.setTimeout(() => liveRef.current?.releaseHeld(), 0);
          return;
        }
      }
    }
    const fresh = rows.filter((r) => mark(r) > s.mark! && isCall(r));
    if (!fresh.length) return;
    s.mark = mark(newest);
    const row = fresh[fresh.length - 1];
    if (row.state === "running") s.running = row.id;
    window.setTimeout(() => playRef.current(row), 0);
  }, [rows, tree]);

  useEffect(() => {
    const t = timers.current;
    return () => t.forEach(clearTimeout);
  }, []);

  // The header's age is of the newest named call; replay plays the last call through the Core Brain.
  const lastRow = useMemo(() => [...rows].reverse().find(isCall) ?? null, [rows]);
  const replayRow = useMemo(() => (tree ? lastBrainCall(rows, (p) => resolvePart(tree, p).some((n) => n.path[0] === "core-brain")) : null), [rows, tree]);
  const online = !liveError && !statusError && !!live;
  const connecting = !live && !liveError && !statusError;
  // The interface answering is not an engine serving: only a serving process makes the view live.
  const serving = engineServing(liveFeed.reply);
  const engineOff = online && serving === false;

  /* Breadcrumb */
  const crumbs: TNode[] = [];
  for (let x: TNode | null | undefined = sel; x; x = x.parent) crumbs.unshift(x);

  const header = (
    <div className={`flex min-h-10 flex-wrap items-center justify-between gap-x-4 gap-y-2 ${variant === "hero" ? "px-4 pt-3 md:px-5" : ""}`}>
      <nav aria-label="Topology path" className="mono flex min-w-0 flex-wrap items-center gap-0.5 text-[12px]">
        {crumbs.map((c, i) => {
          const last = i === crumbs.length - 1;
          return (
            <span key={c.path.join("/") || "root"} className="flex min-w-0 items-center gap-0.5">
              {i > 0 && <span className="px-1 text-quiet">/</span>}
              <button onClick={() => go(c.path)} aria-current={last ? "page" : undefined} className={`press inline-flex h-8 min-w-0 items-center truncate px-1 uppercase tracking-[0.08em] ${last ? "text-text" : "text-faint hover:text-text"}`}>
                {i === 0 ? "Organism" : c.label}
              </button>
            </span>
          );
        })}
        <span className="ml-3 hidden truncate text-[11px] text-quiet xl:inline" title="Address of this view">
          {pathHash(sel?.path ?? [])}
        </span>
      </nav>
      <div className="mono flex items-center gap-3 text-[11px] uppercase tracking-[0.08em]">
        <span className="flex h-8 items-center gap-2 text-faint" aria-live="polite">
          <span aria-hidden className={`inline-block h-1.5 w-1.5 ${engineOff ? "border border-quiet" : online ? "bg-verified" : connecting ? "bg-quiet" : "bg-error"}`} />
          <span className={online && !engineOff ? "text-muted" : undefined} data-e2e="topology-engine" data-serving={serving === null ? "unknown" : String(serving)}>
            {engineOff ? "No engine serving" : online ? "Live" : connecting ? "Connecting" : "Not connected"}
          </span>
          {online && !engineOff && liveFeed.reply && (
            <span className={`hidden max-w-[260px] truncate normal-case tracking-normal md:inline ${liveFeed.reply.state === "busy" ? "text-signal" : "text-faint"}`} data-e2e="topology-live-state">
              {liveLine(liveFeed.reply)}
            </span>
          )}
          {lastRow && now > 0 && <span className="hidden normal-case tracking-normal sm:inline">last call {ago(lastRow.time_ms, now)}</span>}
        </span>
        <button onClick={() => replayRow && play(replayRow, true)} disabled={!replayRow} title="Plays the recorded parts of the last call again; marked as a replay while it plays" className="press inline-flex h-8 items-center gap-1.5 border-b border-line-strong text-muted hover:border-text hover:text-text disabled:opacity-40">
          Replay last call
        </button>
        {variant === "hero" && (
          <Link href={`/topology/${pathHash(sel?.path ?? [])}`} className="press inline-flex h-8 items-center border-b border-line-strong text-muted hover:border-text hover:text-text">
            Full view
          </Link>
        )}
      </div>
    </div>
  );

  const tipNode = hover ? tree?.byPath.get(hover.key) : null;
  const opening = anim?.prev && tree && prevNode ? tree.byPath.get(anim.key) : undefined;
  const thinKey = opening && prevNode && within(opening, prevNode) && !reduced() ? anim?.key : undefined;
  const map = (
    <div
      ref={box}
      className="topo night relative min-h-0 flex-1 overflow-hidden"
      style={{ touchAction: view.k > 1.01 ? "none" : "pan-y" }}
      onPointerDown={gesture.down}
      onPointerMove={gesture.move}
      onPointerUp={gesture.up}
      onPointerCancel={gesture.up}
      onClickCapture={gesture.click}
      onKeyDown={(e) => {
        if (e.key === "+" || e.key === "=") zoomBy(1.4);
        else if (e.key === "-") zoomBy(1 / 1.4);
        else if (e.key === "0") setView({ k: 1, x: 0, y: 0 });
      }}
    >
      {scene && (
        <div className="mono absolute right-2 bottom-2 z-10 flex items-center gap-px border border-line bg-bg/80 text-[11px] backdrop-blur" data-e2e="topo-zoom" onPointerDown={(e) => e.stopPropagation()}>
          <button type="button" aria-label="Zoom in" title="Zoom in (or scroll; + key)" onClick={() => zoomBy(1.4)} className="press h-7 w-7 text-muted hover:text-text">
            +
          </button>
          <button type="button" aria-label="Zoom out" title="Zoom out (or scroll; - key)" onClick={() => zoomBy(1 / 1.4)} disabled={view.k <= 1.01} className="press h-7 w-7 text-muted hover:text-text disabled:opacity-30">
            -
          </button>
          <button type="button" aria-label="Reset the view" title="Back to the whole picture (0 key)" onClick={() => setView({ k: 1, x: 0, y: 0 })} disabled={view.k <= 1.01} className="press h-7 px-2 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-30">
            {view.k > 1.01 ? `${Math.round(view.k * 100)}%` : "fit"}
          </button>
        </div>
      )}
      {!scene && !error && <FigureLoading what={tree ? "topology figure" : "anatomy"} />}
      {error && !tree && <div className="mono absolute inset-0 flex items-center justify-center p-6 text-center text-[12px] text-muted">The engine is not answering, so the topology cannot be drawn. Start it with neuromorph-nano ui.</div>}
      {prevScene && anim?.prev && (
        <div ref={oldLayer} key={`old-${anim.seq}`} className="absolute inset-0" style={{ transformOrigin: "0 0", pointerEvents: "none" }} aria-hidden>
          <SceneView scene={prevScene} paint={paint} status={status} interactive={false} selected={null} thin={thinKey} />
        </div>
      )}
      {scene && (
        <div ref={curLayer} key={`cur-${anim?.seq ?? 0}`} className="absolute inset-0" style={{ transformOrigin: "0 0" }}>
          <div className="absolute inset-0" style={{ transformOrigin: "0 0", transform: view.k !== 1 || view.x || view.y ? `translate(${view.x}px, ${view.y}px) scale(${view.k})` : undefined, transition: gActive ? "none" : "transform var(--d-3) var(--ease)" }}>
            <SceneView scene={scene} paint={paint} status={status} interactive selected={sel && sel !== focus ? keyOf(sel) : null} onLive={onLive} onActivate={activate} onUp={up} onHover={onHover} relabel={jobs?.map} />
          </div>
        </div>
      )}
      {scene && focus && focus.children.length === 0 && (
        <div className="mono pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 text-center text-[12px] text-faint">Nothing inside this component yet.</div>
      )}
      {scene && focus?.key === "cartridges" && focus.children.length === 0 && (
        <div className="mono pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-3 text-center text-[12px] text-faint">
          <span aria-hidden className="dither-slot" />
          No cartridges loaded. A learned cartridge docks here when it loads.
        </div>
      )}
      {tipNode && hover && (
        <div role="tooltip" className="tip pointer-events-none absolute z-10 max-w-[280px] -translate-x-1/2 -translate-y-full px-2.5 py-1.5" style={{ left: clamp(hover.x, 140, Math.max(140, W - 140)), top: Math.max(44, hover.y - 10) }}>
          <div className="truncate text-[12px] text-text" data-e2e="topo-tip-title">{tipTitle(tipNode, roles)}</div>
          {jobs?.map.get(keyOf(tipNode)) && <div className="mono truncate text-[11px] text-signal">{jobs.map.get(keyOf(tipNode))}</div>}
          {figure(tipNode, status) && <div className="mono truncate text-[11px] text-faint">{figure(tipNode, status)}</div>}
          {tipNode.ids[0] && <div className="mono truncate text-[10.5px] text-quiet">{tipNode.ids[0]}</div>}
        </div>
      )}
      {readout && (
        <div key={readout.id + readout.time} role="status" className="fade-in pointer-events-none absolute top-6 left-12 z-10 max-w-[min(460px,calc(100%-60px))] md:top-8 md:left-16">
          <div className="flex items-end gap-3">
            <span className={readout.running ? "text-signal" : "text-sense-pale"}>
              <DotText text={readout.running ? "RUN" : typeof readout.ms === "number" ? (readout.ms < 1 ? "<1MS" : `${readout.ms}MS`) : "NOT TIMED"} size={22} label={readout.running ? "running" : msText(readout.ms)} />
            </span>
            <span className="mono pb-0.5 text-[11px] text-muted">{readout.tool}</span>
            <span className="mono pb-0.5 text-[11px] text-faint">{new Date(readout.time).toLocaleTimeString()}</span>
            {readout.replay ? (
              <span className="mono border border-line-3 px-1 pb-px text-[10px] uppercase tracking-[0.12em] text-attention" data-e2e="topology-replay">
                replay
              </span>
            ) : !readout.running ? (
              // The call has ended: what plays now is its recorded path, at the drawing's pace.
              <span className="mono border border-line-3 px-1 pb-px text-[10px] uppercase tracking-[0.12em] text-faint" data-e2e="topology-trace" title="The call has ended; its recorded path is drawn in the order its parts ran">
                recorded path
              </span>
            ) : null}
          </div>
          {readout.label && <div className="mono mt-1 truncate text-[11.5px] text-muted">{callLabel(readout.label, readout.tool).short}</div>}
          {(readout.whole || readout.elsewhere.length > 0) && <div className="mono mt-0.5 text-[11px] text-faint">{readout.whole ? "This call named no components, so no part is lit." : `Also ran in ${readout.elsewhere.join(", ")}.`}</div>}
        </div>
      )}
      {engineOff && !liveTag && scene && scene.kind === "body" && (
        <div className="mono pointer-events-none absolute bottom-4 left-5 z-10 max-w-[calc(100%-170px)] truncate text-[11px] text-faint" data-e2e="topology-at-rest" title="No engine is serving, so nothing here is live; the organism is drawn at rest">
          No engine serving: drawn at rest. Start one: <span className="text-muted">neuromorph-nano serve</span>
        </div>
      )}
      {liveTag && (
        <div key={liveTag.key} role="status" className="fade-in mono pointer-events-none absolute bottom-4 left-5 z-10 max-w-[min(420px,calc(100%-40px))] truncate text-[11px] text-sense-pale" data-e2e="topology-live-event">
          {liveTag.text}
        </div>
      )}
      {scene && scene.detail === 2 && scene.kind === "body" && (
        // Above the zoom controls, so the two never sit on each other.
        <div aria-hidden className="mono pointer-events-none absolute right-3 bottom-12 hidden flex-col items-end gap-1 text-[10.5px] text-faint md:flex">
          <span className="flex items-center gap-2">
            computing now <i className="sw" style={{ background: "var(--signal)" }} />
          </span>
          <span className="flex items-center gap-2">
            sensory, inward <i className="sw" style={{ background: "var(--sense)" }} />
          </span>
          <span className="flex items-center gap-2">
            behaviour, outward <i className="sw" style={{ background: "var(--behav-hot)" }} />
          </span>
          <span className="flex items-center gap-2">
            cartridge <i className="sw spectral-sw" />
          </span>
        </div>
      )}
    </div>
  );

  if (variant === "hero") {
    return (
      <div className="flex h-full min-h-0 flex-col">
        {header}
        {map}
        {panel !== "rail" && sel && sel.path.length > 0 && (
          <div key={sel.path.join("/")} className="fade-in flex flex-wrap items-baseline gap-x-3 gap-y-1 border-t border-line px-4 py-3 text-xs md:px-5">
            <span className="mono text-[12px] uppercase tracking-[0.08em] text-text">{sel.label}</span>
            <span className="min-w-0 flex-1 basis-64 truncate text-faint">{sel.about}</span>
            <Link href={`/topology/${pathHash(sel.path)}`} className="link">
              Details
            </Link>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      {header}
      <div className="relative flex min-h-0 flex-1 gap-0 lg:portrait:flex-col">
        <div className="ticks flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden border border-line">{map}</div>
        {panel !== "rail" && tree && sel && <PartPanel node={sel} tree={tree} status={status} carts={carts} rows={rows} settings={settings} projects={projects} onGo={go} act={act} />}
      </div>
    </div>
  );
}

export type { Tree };
