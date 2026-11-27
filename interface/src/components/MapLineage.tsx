"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "aslo the parameters main page ui should be way better.
// more visual and interactive with real fucntionality."): the lineage of one map, drawn. Every node on
// one column, read top to bottom: the upstream open weights and whatever it was made on, then this
// map, then everything it feeds (base cartridges, the maps made on those, the job filters and their
// cartridges). Arcs on the left are the made on / used by records of nmapi::lineage, nothing guessed.
// Hovering a row lights its arcs; a map row opens that map.

import { useMemo, useState } from "react";
import { NODE_WORD, graphOf, kindVar, nodeKind, stateOf, versionOf, walk, type JobFilter, type NodeKind, type RoleMaps } from "@/components/MapCore";

const ROW = 22;
const GX = 46;
const DOT = GX - 7;
const MAX_NODES = 22;
const ORDER: Record<NodeKind, number> = { weights: 0, base: 1, comp: 2, cartridge: 3, map: 4, archived: 5, missing: 6, "filter-cart": 7 };

type Row = { t: "band"; label: string } | { t: "node"; id: string; kind: NodeKind; level: number };

function Glyph({ kind, color, y, me }: { kind: NodeKind; color: string; y: number; me: boolean }) {
  if (me)
    return (
      <g>
        <circle cx={DOT} cy={y} r={6.5} fill="none" stroke="var(--sense-pale)" strokeOpacity={0.5} />
        <circle cx={DOT} cy={y} r={4} fill="var(--text-1)" />
      </g>
    );
  switch (kind) {
    case "weights":
      return <circle cx={DOT} cy={y} r={3.5} fill="var(--bg-0)" stroke="var(--text-4)" />;
    case "base":
      return <rect x={DOT - 3.5} y={y - 3.5} width={7} height={7} fill="var(--learned)" />;
    case "comp":
    case "cartridge":
      return <rect x={DOT - 3.5} y={y - 3.5} width={7} height={7} fill="var(--bg-0)" stroke="var(--learned)" />;
    case "filter-cart":
      return <rect x={DOT - 3.5} y={y - 3.5} width={7} height={7} fill="var(--signal)" transform={`rotate(45 ${DOT} ${y})`} />;
    case "map":
      return <rect x={DOT - 3.5} y={y - 3.5} width={7} height={7} fill={color} />;
    default:
      return <rect x={DOT - 3.5} y={y - 3.5} width={7} height={7} fill="var(--bg-0)" stroke="var(--text-4)" strokeDasharray="2 1.5" />;
  }
}

export default function MapLineage({ r, file, jf, onMap }: { r: RoleMaps; file: string; jf: JobFilter[]; onMap: (file: string) => void }) {
  const [hover, setHover] = useState<string | null>(null);
  const { rows, edges, more } = useMemo(() => {
    const g = graphOf(r);
    const up = walk(g, file, "up", 4);
    const down = walk(g, file, "down", 5);
    // Ancestors above (negative levels), descendants below; a longest-path pass so a cartridge built
    // from a map that was itself made on an earlier cartridge sits below both.
    const lvl = new Map<string, number>([[file, 0]]);
    for (const n of up.keys()) if (n !== file) lvl.set(n, -1);
    for (const n of down.keys()) if (n !== file && !up.has(n)) lvl.set(n, 1);
    for (let it = 0; it < 16; it++) {
      let changed = false;
      for (const [a, bs] of g.down)
        for (const b of bs) {
          const la = lvl.get(a);
          const lb = lvl.get(b);
          if (la === undefined || lb === undefined) continue;
          if (lb > 0 && la >= 0 && lb < la + 1 && la < 9) {
            lvl.set(b, la + 1);
            changed = true;
          }
          if (la < 0 && lb <= 0 && la > lb - 1 && lb > -9) {
            lvl.set(a, lb - 1);
            changed = true;
          }
        }
      if (!changed) break;
    }
    let nodes = [...lvl.entries()].map(([id, level]) => ({ id, level, kind: nodeKind(id, r, jf) }));
    const total = nodes.length;
    if (nodes.length > MAX_NODES) nodes = nodes.sort((a, b) => Math.abs(a.level) - Math.abs(b.level)).slice(0, MAX_NODES);
    nodes.sort((a, b) => a.level - b.level || ORDER[a.kind] - ORDER[b.kind] || versionOf(a.id) - versionOf(b.id) || a.id.localeCompare(b.id));
    const out: Row[] = [];
    const anc = nodes.filter((n) => n.level < 0);
    const desc = nodes.filter((n) => n.level > 0);
    out.push({ t: "band", label: anc.length ? "made from" : "made on: not recorded" });
    for (const n of anc) out.push({ t: "node", ...n });
    out.push({ t: "node", id: file, kind: "map", level: 0 });
    out.push({ t: "band", label: desc.length ? "feeds" : "feeds nothing now" });
    for (const n of desc) out.push({ t: "node", ...n });
    const at = new Map<string, number>();
    out.forEach((x, i) => x.t === "node" && at.set(x.id, i));
    const es: { a: string; b: string; i: number; j: number }[] = [];
    for (const [a, bs] of g.down)
      for (const b of bs) {
        const i = at.get(a);
        const j = at.get(b);
        if (i !== undefined && j !== undefined && i < j) es.push({ a, b, i, j });
      }
    return { rows: out, edges: es, more: total - nodes.length };
  }, [r, file, jf]);

  const h = rows.length * ROW;
  const y = (i: number) => i * ROW + ROW / 2;
  return (
    <div className="grid gap-1" data-e2e="pmap-chain">
      <div className="relative" onMouseLeave={() => setHover(null)}>
        <svg aria-hidden width={GX} height={h} className="absolute left-0 top-0 overflow-visible">
          {edges.map((e) => {
            const k = Math.min(DOT - 4, 7 + 5 * (e.j - e.i));
            const lit = hover !== null && (e.a === hover || e.b === hover);
            const mine = e.a === file || e.b === file;
            return (
              <path
                key={`${e.a}>${e.b}`}
                d={`M ${DOT} ${y(e.i)} C ${DOT - k} ${y(e.i)}, ${DOT - k} ${y(e.j)}, ${DOT} ${y(e.j)}`}
                fill="none"
                stroke={lit ? "var(--sense-pale)" : mine ? "var(--text-4)" : "var(--line-3)"}
                strokeWidth={lit ? 1.5 : 1}
                style={{ transition: "stroke var(--d-1) var(--ease)" }}
              />
            );
          })}
          {rows.map((x, i) => {
            if (x.t !== "node") return null;
            const m = x.kind === "map" ? r.maps.find((mm) => mm.file === x.id) : undefined;
            return <Glyph key={x.id} kind={x.kind} color={m ? kindVar(m.kind) : "var(--text-4)"} y={y(i)} me={x.id === file} />;
          })}
        </svg>
        <ol className="grid" style={{ paddingLeft: GX }}>
          {rows.map((x, i) => {
            if (x.t === "band")
              return (
                <li key={`band-${i}`} className="mono flex items-center text-[9px] uppercase tracking-[0.14em] text-quiet" style={{ height: ROW }}>
                  {x.label}
                </li>
              );
            const me = x.id === file;
            const m = x.kind === "map" ? r.maps.find((mm) => mm.file === x.id) : undefined;
            const f = x.kind === "filter-cart" ? jf.find((ff) => ff.cartridge === x.id) : undefined;
            const tag = me
              ? m?.kind
              : m
                ? `${m.kind} · ${stateOf(m)}`
                : x.kind === "weights"
                  ? NODE_WORD.weights
                  : x.kind === "base" && r.serving_base === x.id
                    ? "base cartridge · serving"
                    : f
                      ? `${NODE_WORD["filter-cart"]} · ${f.enabled !== false ? "plugged in" : "unplugged"}`
                      : NODE_WORD[x.kind];
            const label = x.kind === "weights" ? x.id.slice(5) : x.id;
            return (
              <li key={x.id} className={`flex min-w-0 items-center gap-2 ${hover === x.id ? "bg-bg-2/60" : ""}`} style={{ height: ROW }} onMouseEnter={() => setHover(x.id)}>
                {me ? (
                  <span className="mono shrink-0 text-[11px] text-text">this map</span>
                ) : m ? (
                  <button type="button" onClick={() => onMap(x.id)} onFocus={() => setHover(x.id)} className="press mono min-w-0 truncate border-b border-line-strong text-left text-[10.5px] text-muted hover:text-text" title={`Open ${x.id}`}>
                    {label}
                  </button>
                ) : (
                  <span className={`mono min-w-0 truncate text-[10.5px] ${x.kind === "weights" ? "text-quiet" : x.kind === "base" || x.kind === "filter-cart" ? "text-muted" : "text-faint"}`} title={`${label}: ${NODE_WORD[x.kind]}`}>
                    {label}
                  </span>
                )}
                <span className={`mono shrink-0 truncate text-[9px] ${tag?.endsWith("serving") || tag?.endsWith("plugged in") ? "text-verified/80" : "text-quiet"}`}>{tag}</span>
              </li>
            );
          })}
        </ol>
      </div>
      {more > 0 && <span className="mono text-[9.5px] text-quiet">and {more} more further along</span>}
    </div>
  );
}
