"use client";

/** A page's centrepiece drawn as a plate: a scene built for the frame's size, painted once by the
 *  topology's painter (depth, bloom, mist, grain) and kept; painted again only when the data or
 *  the size changes. Parts on it can be opened in the topology when `onOpen` is given. */
import { useLayoutEffect, useMemo, useRef, useState } from "react";
import SceneView from "@/components/SceneView";
import { FigureLoading } from "@/components/ui";
import type { Status } from "@/lib/api";
import type { PaintOpts } from "@/lib/paint";
import type { Scene } from "@/lib/scene";
import { useLiveLights } from "@/lib/livelights";
import type { Tree } from "@/lib/topology";

/** With `tree`, the plate's parts light from the engine's live events (its live layer), as the
 *  topology's do. Until the scene can be built the frame shows "loading <what>" at its final
 *  size; with `failed`, what did not answer. */
export default function PlateCanvas({ build, paint, status, className = "", label, onOpen, tree, what = "figure", failed = null }: { build: (W: number, H: number) => Scene | null; paint: PaintOpts; status: Status | null; className?: string; label: string; onOpen?: (key: string) => void; tree?: Tree | null; what?: string; failed?: string | null }) {
  const onLive = useLiveLights(tree ?? null);
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ W: number; H: number } | null>(null);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    let t = 0;
    let first = true;
    const ro = new ResizeObserver(([e]) => {
      const W = Math.round(e.contentRect.width);
      const H = Math.round(e.contentRect.height);
      const apply = () => setSize((s) => (s && s.W === W && s.H === H ? s : W > 0 && H > 0 ? { W, H } : s));
      clearTimeout(t);
      if (first) {
        first = false;
        apply();
      } else t = window.setTimeout(apply, 140);
    });
    ro.observe(el);
    return () => {
      clearTimeout(t);
      ro.disconnect();
    };
  }, []);
  const scene = useMemo(() => (size ? build(size.W, size.H) : null), [build, size]);
  return (
    <div ref={box} className={`topo night relative overflow-hidden ${className}`} aria-label={label} role="group">
      {!scene && <FigureLoading what={what} failed={failed} />}
      {scene && <SceneView scene={scene} paint={paint} status={status} interactive={!!onOpen} selected={null} onActivate={onOpen ? (k) => onOpen(k) : undefined} onLive={tree ? onLive : undefined} />}
    </div>
  );
}
