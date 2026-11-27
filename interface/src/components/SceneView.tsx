"use client";

/** One topology view: the still picture on a canvas (painted once per change), the live layer on
 *  a second canvas, and the annotation layer over both (square markers, label chips and leader
 *  lines) as SVG. The annotation layer is what is clicked and what the keyboard reaches; a click
 *  on the drawing itself is resolved to the part under it. */
import { memo, useEffect, useLayoutEffect, useMemo, useRef, type KeyboardEvent, type MouseEvent, type PointerEvent } from "react";
import type { Status } from "@/lib/api";
import { counters, LiveLayer } from "@/lib/livelayer";
import { paintBase, type PaintOpts } from "@/lib/paint";
import type { Annot, Scene } from "@/lib/scene";
import { figure, type TNode } from "@/lib/topology";

type Props = {
  scene: Scene;
  paint: PaintOpts;
  status: Status | null;
  interactive: boolean;
  selected: string | null;
  onLive?: (l: LiveLayer | null) => void;
  onActivate?: (key: string, viaKeyboard: boolean) => void;
  onUp?: (viaKeyboard: boolean) => void;
  onHover?: (h: { key: string; x: number; y: number } | null) => void;
  /** During a drill-down into this part: a second picture where everything in front of it is
   *  thinned out and its own anatomy comes forward, faded in by the camera move. */
  thin?: string;
  /** Labels that stand for a live state, by key: shown in the working colour instead of the
   *  scene's own label (a tentacle's job during a call). */
  relabel?: Map<string, string>;
};

const CELL = 14;

/** The part under a point of the drawing: the deepest region holding the nearest point. */
function useHit(scene: Scene) {
  return useMemo(() => {
    const grid = new Map<string, number[]>();
    scene.pts.forEach((p, i) => {
      if (!p.k) return;
      const id = `${Math.floor(p.x / CELL)},${Math.floor(p.y / CELL)}`;
      let l = grid.get(id);
      if (!l) grid.set(id, (l = []));
      l.push(i);
    });
    const regions = scene.regions.filter((r) => !r.key.endsWith("/.x"));
    return (x: number, y: number): { key: string; node: TNode } | null => {
      const gx = Math.floor(x / CELL);
      const gy = Math.floor(y / CELL);
      let best = -1;
      let bd = 11 * 11;
      for (let a = -1; a <= 1; a++)
        for (let b = -1; b <= 1; b++)
          for (const i of grid.get(`${gx + a},${gy + b}`) ?? []) {
            const p = scene.pts[i];
            const d = (p.x - x) ** 2 + (p.y - y) ** 2;
            if (d < bd) {
              bd = d;
              best = i;
            }
          }
      if (best < 0) return null;
      const k = scene.pts[best].k;
      let hit: { key: string; node: TNode } | null = null;
      for (const r of regions) if ((k === r.key || k.startsWith(r.key + "/")) && (!hit || r.key.length > hit.key.length)) hit = { key: r.key, node: r.node };
      // The material region covers the whole frame; only its own points open it.
      if (hit && scene.kind === "body" && hit.key === "substrate" && !k.startsWith("substrate")) return null;
      return hit;
    };
  }, [scene]);
}

function aria(n: TNode | undefined, label: string, status: Status | null): string {
  if (!n) return label;
  const fg = figure(n, status);
  return `${n.label}${n.sub ? `, ${n.sub}` : ""}${fg ? `. ${fg}` : ""}${n.children.length ? ". Press Enter to open." : ""}`;
}

const MONO_W = 6.75; // width of one 11 px monospace glyph

function Chip({ a, selected, node, status, interactive }: { a: Annot; selected: boolean; node?: TNode; status: Status | null; interactive: boolean }) {
  const tw = a.label.length * MONO_W;
  const w = tw + 26;
  const left = a.side === "l";
  const x = left ? a.x : a.x - w;
  const end = left ? x + w : x;
  return (
    <g data-key={a.key} className={`t-node${selected ? " is-selected" : ""}`} tabIndex={interactive ? 0 : -1} role="button" aria-label={aria(node, a.label, status)} aria-current={selected ? "true" : undefined}>
      <rect x={x - 4} y={a.y - 22} width={w + 8} height={44} fill="transparent" />
      <path d={`M ${end} ${a.y} L ${a.mx} ${a.my}`} className="leader" />
      <rect x={a.mx - 3} y={a.my - 3} width={6} height={6} className="marker" />
      <rect x={x} y={a.y - 10} width={w} height={20} className="chip" />
      <rect x={x} y={a.y - 10} width={w} height={20} className="hi" />
      <rect x={x + 7} y={a.y - 3} width={6} height={6} className="chip-mark" />
      <text x={x + 19} y={a.y + 4} className="chip-text">
        {a.label}
      </text>
      {a.sub && (
        <text x={left ? x : x + w} y={a.y + 24} textAnchor={left ? "start" : "end"} className="chip-sub">
          {a.sub}
        </text>
      )}
      <rect className="ring" x={x - 3} y={a.y - 13} width={w + 6} height={26} />
    </g>
  );
}

function Minor({ a, selected, node, status, interactive, live }: { a: Annot; selected: boolean; node?: TNode; status: Status | null; interactive: boolean; live?: string }) {
  const anchor = a.side === "l" ? "start" : a.side === "r" ? "end" : "middle";
  const label = live ?? a.label;
  const tw = label.length * 6.2;
  const x0 = anchor === "start" ? a.x : anchor === "end" ? a.x - tw : a.x - tw / 2;
  return (
    <g data-key={a.key} className={`t-node${selected ? " is-selected" : ""}`} tabIndex={interactive ? 0 : -1} role="button" aria-label={aria(node, label, status)}>
      <rect x={x0 - 8} y={a.y - 25} width={tw + 16} height={44} fill="transparent" />
      <text x={a.x} y={a.y} textAnchor={anchor} className={live ? "minor-text working" : "minor-text"}>
        {label}
      </text>
      <rect className="ring" x={x0 - 4} y={a.y - 12} width={tw + 8} height={18} />
    </g>
  );
}

function Block({ a, selected, node, status, interactive }: { a: Annot; selected: boolean; node?: TNode; status: Status | null; interactive: boolean }) {
  const w = a.w ?? 100;
  const h = a.h ?? 60;
  const label = a.label.length * MONO_W > w - 40 ? a.label.slice(0, Math.max(3, Math.floor((w - 46) / MONO_W))) + "…" : a.label;
  const cw = label.length * MONO_W + 18 + (a.count ? String(a.count).length * MONO_W + 10 : 0);
  return (
    <g data-key={a.key} className={`t-node block${selected ? " is-selected" : ""}`} tabIndex={interactive ? 0 : -1} role="button" aria-label={aria(node, a.label, status)} aria-current={selected ? "true" : undefined}>
      <rect x={a.x} y={a.y} width={w} height={h} fill="transparent" />
      <rect x={a.x} y={a.y} width={w} height={h} className="hi-edge" />
      <rect x={a.x + 4} y={a.y + 3} width={cw} height={16} className="chip" />
      <text x={a.x + 12} y={a.y + 14.5} className="chip-text">
        {label}
      </text>
      {a.count !== undefined && (
        <text x={a.x + 4 + cw - 8} y={a.y + 14.5} textAnchor="end" className="chip-count">
          {a.count}
        </text>
      )}
      {a.sub && (
        <text x={a.x + 4} y={a.y + 32} className="chip-sub">
          {a.sub}
        </text>
      )}
      <rect className="ring" x={a.x - 3} y={a.y - 3} width={w + 6} height={h + 6} />
    </g>
  );
}

function Callout({ a, selected, node, status, interactive }: { a: Annot; selected: boolean; node?: TNode; status: Status | null; interactive: boolean }) {
  const left = a.side === "l";
  const tw = a.label.length * 6.4;
  const tx = left ? a.x + 6 : a.x - 6;
  return (
    <g data-key={a.key} className={`t-node${selected ? " is-selected" : ""}`} tabIndex={interactive ? 0 : -1} role="button" aria-label={aria(node, a.label, status)}>
      <path d={`M ${a.mx} ${a.my} L ${a.x} ${a.y}`} className="leader" />
      <rect x={a.mx - 2.5} y={a.my - 2.5} width={5} height={5} className="marker" />
      <rect x={left ? tx - 4 : tx - tw - 4} y={a.y - 20} width={tw + 8} height={40} fill="transparent" />
      <text x={tx} y={a.y + 3.5} textAnchor={left ? "start" : "end"} className="callout-text">
        {a.label}
      </text>
      <rect className="ring" x={left ? tx - 4 : tx - tw - 4} y={a.y - 10} width={tw + 8} height={17} />
    </g>
  );
}

/** Numbered ticks along the top and left edges of the scene frame, so positions can be read. */
function FrameTicks({ W, H }: { W: number; H: number }) {
  const xs = [];
  for (let x = 50; x < W - 44; x += 50) xs.push(x);
  const ys = [];
  for (let y = 50; y < H - 30; y += 50) ys.push(y);
  return (
    <g aria-hidden className="ticks-g">
      {xs.map((x) => (
        <g key={`x${x}`}>
          <line x1={x} x2={x} y1={0} y2={x % 100 === 0 ? 7 : 4} />
          {x % 100 === 0 && (
            <text x={x + 3} y={12} className="tick-num">
              {String(x).padStart(4, "0")}
            </text>
          )}
        </g>
      ))}
      {ys.map((y) => (
        <g key={`y${y}`}>
          <line x1={0} x2={y % 100 === 0 ? 7 : 4} y1={y} y2={y} />
          {y % 100 === 0 && (
            <text x={10} y={y + 3} className="tick-num">
              {String(y).padStart(4, "0")}
            </text>
          )}
        </g>
      ))}
    </g>
  );
}

/** How long a same-size base repaint waits for further changes (see the base paint below). */
const BASE_SETTLE_MS = 120;

function SceneViewImpl({ scene, paint, status, interactive, selected, onLive, onActivate, onUp, onHover, thin, relabel }: Props) {
  const base = useRef<HTMLCanvasElement>(null);
  const thinned = useRef<HTMLCanvasElement>(null);
  const live = useRef<HTMLCanvasElement>(null);
  const hit = useHit(scene);
  const { W, H } = scene;
  const byKey = useMemo(() => {
    const m = new Map<string, TNode>();
    for (const r of scene.regions) m.set(r.key, r.node);
    return m;
  }, [scene]);

  // The base is painted at once when the canvas is new or resized (a resize clears it). A repaint
  // at the same size waits BASE_SETTLE_MS for the rest of the replies that change it: a page load
  // brings the status, cartridges, roles and activity a few hundred ms apart, and each one used to
  // cost a full base paint (measured 2026-11-27).
  const basePending = useRef(0);
  useLayoutEffect(() => {
    const c = base.current;
    if (!c) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cw = Math.round(W * dpr);
    const ch = Math.round(H * dpr);
    const draw = () => {
      basePending.current = 0;
      const ctx = c.getContext("2d");
      if (!ctx) return;
      const t0 = performance.now();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      paintBase(ctx, scene, paint);
      const k = counters();
      k.base++;
      k.lastBaseMs = Math.round(performance.now() - t0);
    };
    if (c.width !== cw || c.height !== ch || !c.dataset.painted) {
      c.width = cw;
      c.height = ch;
      c.dataset.painted = "1";
      draw();
      return;
    }
    basePending.current = window.setTimeout(draw, BASE_SETTLE_MS);
    return () => {
      clearTimeout(basePending.current);
      basePending.current = 0;
    };
  }, [scene, paint, W, H]);

  useLayoutEffect(() => {
    const c = thinned.current;
    if (!c || !thin) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = Math.round(W * dpr);
    c.height = Math.round(H * dpr);
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    paintBase(ctx, scene, { ...paint, emph: thin });
    counters().base++;
  }, [scene, paint, W, H, thin]);

  useEffect(() => {
    const c = live.current;
    if (!c) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = Math.round(W * dpr);
    c.height = Math.round(H * dpr);
    const l = new LiveLayer(c, scene, dpr);
    onLive?.(l);
    return () => {
      l.destroy();
      onLive?.(null);
    };
  }, [scene, W, H, onLive]);

  const local = (e: { clientX: number; clientY: number; currentTarget: Element }) => {
    const r = e.currentTarget.getBoundingClientRect();
    // The view may be scaled during a camera move: map back to scene units.
    return { x: ((e.clientX - r.left) / r.width) * W, y: ((e.clientY - r.top) / r.height) * H };
  };
  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    if (!interactive || !onHover) return;
    const t = e.target as Element;
    const g = t.closest<SVGGElement>("[data-key]");
    if (g) {
      const key = g.dataset.key ?? "";
      onHover({ key, ...local(e) });
      return;
    }
    const p = local(e);
    const h = hit(p.x, p.y);
    (e.currentTarget as HTMLDivElement).style.cursor = h ? "pointer" : "default";
    onHover(h ? { key: h.key, ...p } : null);
  };
  // Touch: the first tap on a part selects it and shows its label; a second tap opens it.
  const pointer = useRef<{ type: string; tapped: string | null }>({ type: "mouse", tapped: null });
  const onDown = (e: PointerEvent<HTMLDivElement>) => {
    pointer.current.type = e.pointerType;
  };
  const onClick = (e: MouseEvent<HTMLDivElement>) => {
    if (!interactive) return;
    const g = (e.target as Element).closest<SVGGElement>("[data-key]");
    const p = local(e);
    const key = g ? (g.dataset.key ?? "") : (hit(p.x, p.y)?.key ?? null);
    if (key === null) {
      pointer.current.tapped = null;
      return onUp?.(false);
    }
    if (pointer.current.type === "touch" && pointer.current.tapped !== key) {
      pointer.current.tapped = key;
      onHover?.({ key, ...p });
      return;
    }
    pointer.current.tapped = null;
    onActivate?.(key, false);
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const g = (e.target as Element).closest<SVGGElement>("[data-key]");
    if (g && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      onActivate?.(g.dataset.key ?? "", true);
    } else if (e.key === "Backspace") {
      e.preventDefault();
      onUp?.(true);
    }
  };
  const onFocus = (e: React.FocusEvent<HTMLDivElement>) => {
    const g = (e.target as Element).closest<SVGGElement>("[data-key]");
    if (!g || !onHover) return;
    const r = g.getBoundingClientRect();
    const o = e.currentTarget.getBoundingClientRect();
    onHover({ key: g.dataset.key ?? "", x: ((r.left - o.left + r.width / 2) / o.width) * W, y: ((r.top - o.top) / o.height) * H });
  };

  const selAnnot = selected ? scene.annots.find((a) => a.key === selected && a.kind !== "pad") : null;
  return (
    <div className="absolute inset-0" onPointerDown={onDown} onPointerMove={onMove} onPointerLeave={(e) => e.pointerType !== "touch" && onHover?.(null)} onClick={onClick} onKeyDown={onKey} onFocus={onFocus} onBlur={() => onHover?.(null)}>
      <canvas ref={base} className="absolute inset-0 block h-full w-full" aria-hidden />
      {thin && <canvas ref={thinned} data-thin className="absolute inset-0 block h-full w-full" style={{ opacity: 0 }} aria-hidden />}
      <canvas ref={live} className="absolute inset-0 block h-full w-full" aria-hidden data-e2e={interactive ? "live-layer" : undefined} data-live-lit="" data-live-glow="" data-live-why="" data-live-signals="0" />
      <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" className="annot absolute inset-0 block" role="group" aria-label={`Topology of the Neuromorph Nano: ${scene.focus.label}`}>
        {interactive && <FrameTicks W={W} H={H} />}
        <g aria-hidden>
          {scene.texts.map((t, i) => (
            <text key={i} x={t.x} y={t.y} textAnchor={t.anchor} className={`plate-${t.kind}${t.quiet ? " quiet" : ""}`} style={t.size ? { fontSize: t.size } : undefined}>
              {t.text}
            </text>
          ))}
          {scene.annots
            .filter((a) => a.kind === "pad")
            .map((a, i) => (
              <text key={`p${i}`} x={a.x} y={a.y + 3.5} textAnchor={a.side === "l" ? "start" : a.side === "r" ? "end" : "middle"} className="pad-text">
                {a.label}
              </text>
            ))}
        </g>
        {scene.annots.map((a, i) => {
          const props = { a, selected: selected === a.key, node: byKey.get(a.key) ?? findNode(scene.focus, a.key), status, interactive };
          if (a.kind === "chip") return <Chip key={i} {...props} />;
          if (a.kind === "minor") return <Minor key={i} {...props} live={relabel?.get(a.key)} />;
          if (a.kind === "block") return <Block key={i} {...props} />;
          if (a.kind === "callout") return <Callout key={i} {...props} />;
          return null;
        })}
        {selAnnot && (
          <g aria-hidden className="reticle">
            <circle cx={selAnnot.kind === "block" ? selAnnot.x + (selAnnot.w ?? 0) / 2 : selAnnot.mx} cy={selAnnot.kind === "block" ? selAnnot.y + (selAnnot.h ?? 0) / 2 : selAnnot.my} r={10} />
            {[0, 1, 2, 3].map((q) => {
              const cx = selAnnot.kind === "block" ? selAnnot.x + (selAnnot.w ?? 0) / 2 : selAnnot.mx;
              const cy = selAnnot.kind === "block" ? selAnnot.y + (selAnnot.h ?? 0) / 2 : selAnnot.my;
              const a = (q * Math.PI) / 2;
              return <line key={q} x1={cx + Math.cos(a) * 13} y1={cy + Math.sin(a) * 13} x2={cx + Math.cos(a) * 18} y2={cy + Math.sin(a) * 18} />;
            })}
          </g>
        )}
      </svg>
    </div>
  );
}

function findNode(root: TNode, key: string): TNode | undefined {
  let n: TNode | undefined = root;
  const rel = root.path.length ? key.slice(root.path.join("/").length + 1) : key;
  for (const k of rel.split("/")) {
    n = n?.children.find((c) => c.key === k);
    if (!n) return undefined;
  }
  return n;
}

const SceneView = memo(SceneViewImpl);
export default SceneView;
