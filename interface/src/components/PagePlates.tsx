"use client";

/** The centrepieces of the pages, each drawn from that page's own real data. They are static
 *  pictures (drawn once per change of data), in the same language as the topology: points and
 *  cells on the night ground, fine lines, monospace labels on leaders, the spectral bar of a
 *  cartridge. Where the engine reports nothing yet, the marks are dim and still. */
import { useId, type ReactNode } from "react";
import { maskPathsIn } from "@/lib/privacy";
import DotText from "@/components/DotText";
import { bytes, type AuditRow, type Cartridge } from "@/lib/api";
import type { LiveModel } from "@/lib/livefeed";
import { HEADS_FALLBACK, LAYERS_FALLBACK } from "@/lib/scene";

const HUES = [330, 20, 48, 140, 190, 220, 262, 290];

/** The spectral banding of a cartridge, as an SVG pattern. */
function Spectral({ id }: { id: string }) {
  return (
    <pattern id={id} width="4" height={HUES.length * 1.3} patternUnits="userSpaceOnUse">
      {HUES.map((h, i) => (
        <rect key={h} x={0} y={i * 1.3} width={4} height={0.9} fill={`hsl(${h} 85% 68%)`} />
      ))}
    </pattern>
  );
}

/** One MicroModel as its stack of layers seen from above at an angle: 28 thin planes (layer 0 at
 *  the bottom), each a row of 16 head cells. Drone reader heads are brighter; a layer whose
 *  weights the engine reports on the graphics card carries a tick at its right end. Docked
 *  cartridges sit as spectral bars beside the layers they cover. With `levels` (real calls per
 *  layer over the last day, 0 to 1) each layer's cells are as bright as its work; without them
 *  every cell is drawn alike apart from those marks. With `live` (the MicroModel's state from
 *  `/api/live`): while it executes, the stack carries a cold-white rim; the job-path filters
 *  (`nmengn-sockets/1`) and preference cartridges docked on it stand as spectral bars after the
 *  learned cartridges (a preference bar capped in violet, an inactive one dashed and dim), its
 *  project lens is a blue rule along the stack's left edge, and its precision is set under it.
 *  `layers` and `heads` are the model's own counts as the engine reports them (`/api/status`
 *  `models[]`); only when it reports none is the stated Qwen3 shape drawn, and the label says the
 *  shape is not reported. */
export function Matrix({ id, cell = 9, drones, carts, scale = true, onCard, fluid = false, levels, live, layers, heads }: { id: string; cell?: number; drones?: Set<string>; carts: Cartridge[]; scale?: boolean; onCard?: number[]; fluid?: boolean; levels?: number[] | null; live?: LiveModel | null; layers?: number; heads?: number }) {
  const uid = useId().replace(/:/g, "");
  const layersKnown = typeof layers === "number" && layers > 0;
  const headsKnown = typeof heads === "number" && heads > 0;
  const L = layersKnown ? Math.round(layers) : LAYERS_FALLBACK;
  const Hd = headsKnown ? Math.round(heads) : HEADS_FALLBACK;
  const ox = scale ? 26 : 2;
  const pitch = cell * 0.72;
  const dep = cell * 0.3;
  const sk = cell * 0.5;
  const top = dep + 2;
  const own = carts.filter((c) => c.loaded && (c.model ?? c.base) === id);
  const ownIds = new Set(own.map((c) => c.id));
  const sockets = (live?.filters ?? []).filter((f) => f.kind !== "lens" && !ownIds.has(f.id));
  const lens = live?.lens ?? live?.filters.find((f) => f.kind === "lens")?.id ?? null;
  const executing = live?.executing === true;
  const W = ox + Hd * cell + sk + 26 + Math.max(0, own.length + sockets.length - 3) * 5;
  const H = top + L * pitch + (scale ? 20 : 4) + (scale && live?.precision ? 11 : 0);
  const card = new Set(onCard ?? []);
  const yOf = (l: number) => top + (L - 1 - l) * pitch + pitch;
  const tile = (x: number, y: number, w: number) => `M${x.toFixed(1)} ${y.toFixed(1)}h${w.toFixed(1)}l${sk.toFixed(1)} ${(-dep).toFixed(1)}h${(-w).toFixed(1)}z`;
  return (
    <svg width={fluid ? "100%" : W} height={fluid ? undefined : H} style={fluid ? { maxWidth: W * 1.5, height: "auto" } : undefined} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${id}: ${layersKnown ? `${L} layers` : "layer count not reported"} by ${headsKnown ? `${Hd} heads` : "head count not reported"}${executing ? ", executing now" : live?.resident ? ", resident" : ""}${live?.precision ? `, ${live.precision}` : ""}${drones?.size ? `, ${drones.size} drone reader heads` : ""}${card.size ? `, ${card.size} layers on the graphics card` : ""}${own.length ? `, ${own.length} cartridges docked` : ", no cartridges docked"}${sockets.length ? `, ${sockets.length} sockets docked` : ""}${lens ? `, lens ${lens}` : ""}`} className="block" data-executing={executing ? "yes" : "no"}>
      <defs>
        <Spectral id={`sp${uid}`} />
      </defs>
      {Array.from({ length: L }, (_, l) => {
        const y = yOf(l);
        let plain = "";
        let lit = "";
        for (let h = 0; h < Hd; h++) {
          const d = tile(ox + h * cell + cell * 0.08, y, cell * 0.8);
          if (drones?.has(`${l}:${h}`)) lit += d;
          else plain += d;
        }
        return (
          <g key={l}>
            <path d={plain} fill={levels && levels[l] > 0 ? "#9fb4e6" : "#5b6c96"} opacity={levels ? 0.26 + 0.7 * Math.min(1, levels[l]) : 0.4} />
            {lit && <path d={lit} fill="#dfe7f7" opacity={0.95} />}
            <line x1={ox} x2={ox + Hd * cell} y1={y + 0.5} y2={y + 0.5} stroke="#9fb0dd" strokeOpacity={0.35} strokeWidth={0.6} />
            {card.has(l) && <line x1={ox + Hd * cell + sk * 0.5 + 2} x2={ox + Hd * cell + sk * 0.5 + 7} y1={y - dep / 2} y2={y - dep / 2} stroke="var(--sense-pale)" strokeWidth={1.2} />}
          </g>
        );
      })}
      {scale &&
        Array.from({ length: Math.ceil(L / 4) }, (_, i) => (
          <g key={i} className="mono" fill="var(--text-4)" fontSize={8.5}>
            <line x1={ox - 6} x2={ox - 2} y1={yOf(i * 4)} y2={yOf(i * 4)} stroke="var(--line-3)" />
            <text x={ox - 8} y={yOf(i * 4) + 3} textAnchor="end">
              {String(i * 4).padStart(2, "0")}
            </text>
          </g>
        ))}
      <line x1={ox + Hd * cell + sk + 5} x2={ox + Hd * cell + sk + 5} y1={yOf(L - 1) - dep} y2={yOf(0)} stroke="var(--line-3)" strokeDasharray="1 2" />
      {own.map((c, k) => {
        const [a, b] = c.layers ?? [0, L - 1];
        return <rect key={c.id} x={ox + Hd * cell + sk + 3 + k * 5} y={yOf(b) - dep} width={3.5} height={yOf(a) - yOf(b) + dep} fill={`url(#sp${uid})`} />;
      })}
      {sockets.map((f, k) => {
        const x = ox + Hd * cell + sk + 3 + (own.length + k) * 5;
        const y0 = yOf(L - 1) - dep;
        const h = yOf(0) - yOf(L - 1) + dep;
        return (
          <g key={`s-${f.id}`} opacity={f.on ? 1 : 0.4}>
            <title>{`${f.kind === "preference" ? "preference cartridge" : "job-path filter"} ${f.id}${f.on ? "" : " (off)"}`}</title>
            <rect x={x} y={y0} width={3.5} height={h} fill={`url(#sp${uid})`} strokeDasharray={f.on ? undefined : "1 1.5"} stroke={f.on ? "none" : "var(--text-3)"} strokeWidth={0.5} />
            {f.kind === "preference" && <rect x={x} y={y0 - 2.5} width={3.5} height={2} fill="var(--behav)" />}
          </g>
        );
      })}
      {lens && (
        <g>
          <title>{`project lens ${lens}`}</title>
          <line x1={ox - 1.5} x2={ox - 1.5} y1={yOf(L - 1) - dep * 0.5} y2={yOf(0)} stroke="var(--sense)" strokeWidth={1.4} strokeOpacity={0.9} />
        </g>
      )}
      {executing && <path d={`M${ox - 3} ${yOf(0) + 2}h${Hd * cell + 4}l${sk + 2} ${-(dep + 2)}v${-(L - 1) * pitch - 2}h${-(Hd * cell + 4)}l${-(sk + 2)} ${dep + 2}z`} fill="none" stroke="var(--signal)" strokeWidth={1} strokeOpacity={0.85} />}
      {scale && live?.precision && (
        <text x={ox + Hd * cell + sk + 2} y={yOf(0) + 25} textAnchor="end" className="mono" fill={executing ? "var(--signal)" : "var(--text-3)"} fontSize={8.5}>
          {executing ? "executing · " : live.resident ? "resident · " : ""}
          {live.precision}
        </text>
      )}
      {scale &&
        Array.from({ length: Math.ceil(Hd / 5) }, (_, i) => i * 5).map((h) => (
          <text key={h} x={ox + (h + 0.5) * cell} y={yOf(0) + 13} textAnchor="middle" className="mono" fill="var(--text-4)" fontSize={8.5}>
            {h}
          </text>
        ))}
    </svg>
  );
}

/** A socket on the rack besides the cartridges: a job-path filter (`nmengn-sockets/1`), a project
 *  embedder lens or a preference cartridge the engine reports, with its on/off state and size on
 *  disk when known. */
export type RackSocket = { id: string; kind: "filter" | "lens" | "preference"; on: boolean; bytes?: number | null; role?: string | null; layers?: [number, number] | null };
/** The job-filter build with its held-out split gate (`/api/surgery` `job_filter_build`). */
export type RackBuild = { status: string; role?: string; job?: string; message?: string; started_ms?: number; finished_ms?: number } | null;

/** The kind word under a cartridge slab: base (read only), preference, or learned. */
export function cartKind(c: Pick<Cartridge, "family" | "class" | "kind">): "base" | "preference" | "learned" {
  if (c.class === "preference" || /pref/i.test(c.family ?? "")) return "preference";
  if (/fundament|base/i.test(`${c.family ?? ""} ${c.kind ?? ""}`)) return "base";
  return "learned";
}

/** The cartridge rack seen from above at an angle, in the same drawing as the matrices: a floor
 *  with rails, and on it one slab per cartridge of every type (read-only base cartridges, learned
 *  and preference cartridges, then the job-path filters and preference cartridges docked on the
 *  MicroModels' sockets), standing over the model layers it touches (its height and lift on the
 *  28-layer scale at the left, its width by its size on disk), its face in spectral bands. A
 *  project lens lies flat on the floor. A loaded or switched-on piece is lit and marked docked or
 *  on; a switched-off one is dim and dashed; a verified one carries a tick. Under each slab: its
 *  state, its size on disk and its kind. A job-filter build in progress stands as a dashed ghost
 *  slab with its split-gate state. With nothing on it the floor stays empty and says so. */
export function Rack({ carts, layers = LAYERS_FALLBACK, sockets = [], build = null }: { carts: Cartridge[]; layers?: number; sockets?: RackSocket[]; build?: RackBuild }) {
  const uid = useId().replace(/:/g, "");
  const building = !!build && build.status === "running";
  const ended = !!build && (build.status === "done" || build.status === "error" || build.status === "stopped");
  const n = carts.length + sockets.length + (building ? 1 : 0);
  const pitch = 6;
  const dep = 44;
  const sk = 66;
  const ox = 34;
  const floorW = Math.max(300, n * 34 + 60);
  const W = ox + floorW + sk + 18;
  const top = 18;
  const baseY = top + layers * pitch + dep + 8;
  const H = baseY + 46 + (ended || building ? 12 : 0);
  const maxB = Math.max(1, ...carts.map((c) => c.bytes ?? 0), ...sockets.map((s) => s.bytes ?? 0));
  // The back wall: one rung per model layer, the scale every slab is read against.
  const wallX = ox + sk;
  const wallY = (l: number) => baseY - dep - (l + 1) * pitch;
  const floor = `M${ox} ${baseY}h${floorW}l${sk} ${-dep}h${-floorW}z`;
  const on = carts.filter((c) => c.loaded).length + sockets.filter((s) => s.on).length;
  const width = (b: number | null | undefined) => 6 + 14 * Math.sqrt((b ?? 0) / maxB);
  const size = (b: number | null | undefined) => (typeof b === "number" && b > 0 ? bytes(b) : "size -");
  const words = (x: number, w: number, lit: boolean, a: string, b: string, c: string, tone?: string) => (
    <>
      <text x={x + w / 2} y={baseY + 16} textAnchor="middle" className="mono" fontSize={8.5} fill={tone ?? (lit ? "var(--text-2)" : "var(--text-4)")}>
        {a}
      </text>
      <text x={x + w / 2} y={baseY + 27} textAnchor="middle" className="mono" fontSize={8} fill="var(--text-4)">
        {b}
      </text>
      <text x={x + w / 2} y={baseY + 38} textAnchor="middle" className="mono" fontSize={8} fill="var(--text-4)">
        {c}
      </text>
    </>
  );
  const slab = (key: string, i: number, range: [number, number], w: number, lit: boolean, dashed: boolean, ok: boolean, a: string, b: string, c: string, cap?: string) => {
    const [l0, l1] = range;
    // Each slab stands against the wall, so its height reads directly on the wall's scale.
    const x = wallX + 18 + i * 34;
    const y0 = wallY(l1);
    const h = wallY(l0) + pitch - y0;
    const dash = dashed ? "2 2" : undefined;
    return (
      <g key={key}>
        {/* The slab's shadow line on the floor, under the layers it covers. */}
        <line x1={x} x2={x + w} y1={baseY - dep} y2={baseY - dep} stroke="#9fb0dd" strokeOpacity={0.5} />
        <line x1={x + w / 2} x2={x + w / 2} y1={y0 + h} y2={baseY - dep} stroke="var(--line-3)" strokeDasharray="1 2" />
        <path d={`M${x} ${y0}l${-sk * 0.3} ${dep * 0.3}v${h}l${sk * 0.3} ${-dep * 0.3}z`} fill={`url(#r${uid})`} opacity={lit ? 0.55 : 0.25} />
        <path d={`M${x} ${y0}l${-sk * 0.3} ${dep * 0.3}h${w}l${sk * 0.3} ${-dep * 0.3}z`} fill="#dfe7f7" opacity={lit ? 0.5 : 0.18} />
        <rect x={x} y={y0} width={w} height={h} fill={`url(#r${uid})`} opacity={lit ? 0.95 : 0.4} />
        <rect x={x - 0.5} y={y0 - 0.5} width={w + 1} height={h + 1} fill="none" stroke="var(--text-3)" strokeWidth={0.5} strokeDasharray={dash} />
        {cap && <rect x={x} y={y0 - 3} width={w} height={2} fill={cap} />}
        {ok && <path d={`M${x + 1} ${y0 - 9}l2.5 2.5l5 -5`} fill="none" stroke="var(--verified)" strokeWidth={1} />}
        {words(x, w, lit, a, b, c)}
      </g>
    );
  };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" style={{ maxWidth: W * 1.6 }} role="img" aria-label={n ? `The rack holds ${carts.length} cartridges and ${sockets.length} sockets, ${on} docked or on${building ? ", a job filter building" : ""}` : "The rack is empty: no cartridge learned yet"}>
      <defs>
        <Spectral id={`r${uid}`} />
        <linearGradient id={`f${uid}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#9fb0dd" stopOpacity={0.05} />
          <stop offset="1" stopColor="#9fb0dd" stopOpacity={0.16} />
        </linearGradient>
      </defs>
      {/* The layer scale the slabs stand against. */}
      {Array.from({ length: layers }, (_, l) => (
        <line key={`w${l}`} x1={wallX} x2={wallX + floorW} y1={wallY(l) + pitch / 2} y2={wallY(l) + pitch / 2} stroke="#9fb0dd" strokeOpacity={l % 4 === 0 ? 0.24 : 0.1} strokeWidth={0.6} strokeDasharray={l % 4 === 0 ? undefined : "1 3"} />
      ))}
      <line x1={wallX} x2={wallX} y1={wallY(layers - 1)} y2={baseY - dep} stroke="#9fb0dd" strokeOpacity={0.35} strokeWidth={0.6} />
      <line x1={wallX + floorW} x2={wallX + floorW} y1={wallY(layers - 1)} y2={baseY - dep} stroke="#9fb0dd" strokeOpacity={0.35} strokeWidth={0.6} />
      {Array.from({ length: layers / 4 }, (_, i) => {
        const l = i * 4;
        return (
          <g key={i} className="mono" fill="var(--text-4)" fontSize={8.5}>
            <line x1={wallX - 6} x2={wallX - 1} y1={wallY(l) + pitch / 2} y2={wallY(l) + pitch / 2} stroke="var(--line-3)" />
            <text x={wallX - 8} y={wallY(l) + pitch / 2 + 3} textAnchor="end">
              {String(l).padStart(2, "0")}
            </text>
          </g>
        );
      })}
      {/* Floor and rails. */}
      <path d={floor} fill={`url(#f${uid})`} stroke="#9fb0dd" strokeOpacity={0.45} strokeWidth={0.7} />
      {[0.2, 0.4, 0.6, 0.8].map((v) => (
        <line key={v} x1={ox + sk * v} x2={ox + floorW + sk * v} y1={baseY - dep * v} y2={baseY - dep * v} stroke="#9fb0dd" strokeOpacity={0.22} strokeWidth={0.6} strokeDasharray="1 3" />
      ))}
      {Array.from({ length: Math.floor(floorW / 12) + 1 }, (_, i) => (
        <line key={i} x1={ox + i * 12} x2={ox + i * 12} y1={baseY} y2={baseY + (i % 4 === 0 ? 5 : 3)} stroke="var(--line-3)" />
      ))}
      {n === 0 && (
        <text x={ox + floorW / 2 + sk / 2} y={baseY - dep / 2 + 4} textAnchor="middle" className="mono" fontSize={10} fill="var(--text-4)" letterSpacing="0.12em">
          NO CARTRIDGE LEARNED
        </text>
      )}
      {carts.map((c, i) => {
        const kind = cartKind(c);
        const ok = c.verified ?? /^(pass|passed|ok|verified|accept)$/i.test(c.check);
        return slab(c.id, i, c.layers ?? [0, layers - 1], width(c.bytes), c.loaded, false, ok, c.loaded ? "docked" : "in rack", size(c.bytes), kind === "base" ? "base, read only" : kind, kind === "preference" ? "var(--behav)" : undefined);
      })}
      {sockets.map((s, k) => {
        const i = carts.length + k;
        if (s.kind === "lens") {
          // A lens lies flat on the floor: it shapes the embedder's reading, not a layer range.
          const x = wallX + 18 + i * 34;
          const fy = baseY - dep * 0.35;
          const w = width(s.bytes) + 8;
          return (
            <g key={`lens-${s.id}`} opacity={s.on ? 1 : 0.45}>
              <path d={`M${x} ${fy}h${w}l${sk * 0.25} ${-dep * 0.25}h${-w}z`} fill="var(--sense)" fillOpacity={s.on ? 0.55 : 0.2} stroke="var(--sense-pale)" strokeWidth={0.6} strokeDasharray={s.on ? undefined : "2 2"} />
              {words(x, w, s.on, s.on ? "on" : "off", size(s.bytes), "lens")}
            </g>
          );
        }
        return slab(`${s.kind}-${s.id}`, i, s.layers ?? [0, layers - 1], Math.max(5, width(s.bytes) * 0.6), s.on, !s.on, false, s.on ? "on" : "off", size(s.bytes), s.kind === "filter" ? "job filter" : "preference", s.kind === "preference" ? "var(--behav)" : "var(--verified)");
      })}
      {building && (
        <g key="building">
          <rect x={wallX + 18 + (carts.length + sockets.length) * 34} y={wallY(layers - 1)} width={8} height={wallY(0) + pitch - wallY(layers - 1)} fill="none" stroke="var(--signal)" strokeWidth={0.7} strokeDasharray="2 2" />
          {words(wallX + 18 + (carts.length + sockets.length) * 34, 8, true, "building", build?.role ?? "", build?.job ?? "", "var(--signal)")}
        </g>
      )}
      {(building || ended) && (
        <text x={ox} y={H - 4} className="mono" fontSize={8.5} fill={build?.status === "error" ? "var(--error)" : build?.status === "stopped" ? "var(--attention)" : building ? "var(--signal)" : "var(--verified)"}>
          {`job filter ${build?.role ?? ""}/${build?.job ?? ""} · ${building ? "building, split gate pending" : build?.status === "done" ? "split gate done" : build?.status === "stopped" ? "build stopped: its process ended" : "split gate held it back"}${build?.message ? ` · ${maskPathsIn(build.message).length > 60 ? `${maskPathsIn(build.message).slice(0, 59)}...` : maskPathsIn(build.message)}` : ""}`}
        </text>
      )}
    </svg>
  );
}

/** A cartridge as it stands in a bay of the library rack. */
export type Slab = { key: string; family: "fundamentals" | "learning" | "preference"; layers: [number, number] | null; settings: boolean; lit: boolean; trial: boolean; verified: boolean; bytes: number };
/** One bay: the stack plugged in for one use, its cartridges in the order they apply. */
export type Bay = { use: string; label: string; stack: string | null; trial: boolean; inUse: boolean; slabs: Slab[] };

/** The cartridge library on the rack: one bay per use, each holding its stack's cartridges in the
 *  order they apply (numbered), standing over the MicroModel layers they filter on the wall's
 *  scale. A cartridge that filters settings values rather than layers lies flat on the floor. A
 *  stack in use is lit; one on trial is drawn dashed and dim and never changes what is served. */
export function LibraryRack({ bays, layers = LAYERS_FALLBACK }: { bays: Bay[]; layers?: number }) {
  const uid = useId().replace(/:/g, "");
  const pitch = 6;
  const dep = 44;
  const sk = 66;
  const ox = 34;
  const slabStep = 30;
  const widths = bays.map((b) => Math.max(b.slabs.length * slabStep + 16, b.label.length * 5.4 + 18, 70));
  const floorW = Math.max(300, widths.reduce((a, w) => a + w, 0) + 40);
  const W = ox + floorW + sk + 18;
  const top = 18;
  const baseY = top + layers * pitch + dep + 8;
  const H = baseY + 44;
  const wallX = ox + sk;
  const wallY = (l: number) => baseY - dep - (l + 1) * pitch;
  const floor = `M${ox} ${baseY}h${floorW}l${sk} ${-dep}h${-floorW}z`;
  const total = bays.reduce((a, b) => a + b.slabs.length, 0);
  let cursor = wallX + 20;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" style={{ maxWidth: W * 1.6 }} role="img" aria-label={`The stacks in the cartridge library: ${bays.length} ${bays.length === 1 ? "stack" : "stacks"}, ${total} cartridges, in the order they apply`}>
      <defs>
        <Spectral id={`l${uid}`} />
        <linearGradient id={`g${uid}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#9fb0dd" stopOpacity={0.05} />
          <stop offset="1" stopColor="#9fb0dd" stopOpacity={0.16} />
        </linearGradient>
      </defs>
      {Array.from({ length: layers }, (_, l) => (
        <line key={`w${l}`} x1={wallX} x2={wallX + floorW} y1={wallY(l) + pitch / 2} y2={wallY(l) + pitch / 2} stroke="#9fb0dd" strokeOpacity={l % 4 === 0 ? 0.24 : 0.1} strokeWidth={0.6} strokeDasharray={l % 4 === 0 ? undefined : "1 3"} />
      ))}
      <line x1={wallX} x2={wallX} y1={wallY(layers - 1)} y2={baseY - dep} stroke="#9fb0dd" strokeOpacity={0.35} strokeWidth={0.6} />
      {Array.from({ length: layers / 4 }, (_, i) => (
        <g key={i} className="mono" fill="var(--text-4)" fontSize={8.5}>
          <line x1={wallX - 6} x2={wallX - 1} y1={wallY(i * 4) + pitch / 2} y2={wallY(i * 4) + pitch / 2} stroke="var(--line-3)" />
          <text x={wallX - 8} y={wallY(i * 4) + pitch / 2 + 3} textAnchor="end">
            {String(i * 4).padStart(2, "0")}
          </text>
        </g>
      ))}
      <path d={floor} fill={`url(#g${uid})`} stroke="#9fb0dd" strokeOpacity={0.45} strokeWidth={0.7} />
      {[0.25, 0.5, 0.75].map((v) => (
        <line key={v} x1={ox + sk * v} x2={ox + floorW + sk * v} y1={baseY - dep * v} y2={baseY - dep * v} stroke="#9fb0dd" strokeOpacity={0.22} strokeWidth={0.6} strokeDasharray="1 3" />
      ))}
      {bays.length === 0 && (
        <text x={ox + floorW / 2 + sk / 2} y={baseY - dep / 2 + 4} textAnchor="middle" className="mono" fontSize={10} fill="var(--text-4)" letterSpacing="0.12em">
          NO STACK IN THE LIBRARY
        </text>
      )}
      {bays.map((b, bi) => {
        const x0 = cursor;
        const bw = widths[bi];
        cursor += bw;
        return (
          <g key={b.use + (b.stack ?? "")}>
            {/* The bay's divider on the floor and its use, set under it. */}
            <line x1={x0 - 8} x2={x0 - 8 - sk * 0.6} y1={baseY - dep} y2={baseY - dep * 0.4} stroke="var(--line-3)" strokeDasharray="2 2" />
            <line x1={x0 - 6} x2={x0 + bw - 20} y1={baseY + 6} y2={baseY + 6} stroke={b.inUse ? "var(--sense-pale)" : "var(--line-3)"} strokeOpacity={b.inUse ? 0.8 : 1} />
            <text x={x0 - 6} y={baseY + 19} className="mono" fontSize={8.5} fill={b.inUse ? "var(--text-2)" : "var(--text-4)"} letterSpacing="0.06em">
              {b.label.toUpperCase()}
            </text>
            <text x={x0 - 6} y={baseY + 31} className="mono" fontSize={8} fill="var(--text-4)">
              {b.trial ? "on trial" : b.inUse ? "in use" : "not in use"}
            </text>
            {b.slabs.map((s, i) => {
              const x = x0 + i * slabStep;
              const w = 6 + 10 * Math.min(1, Math.sqrt(s.bytes / 65536));
              const dim = s.trial || !s.lit;
              const dash = s.trial ? "2 2" : undefined;
              if (s.settings || !s.layers) {
                // A settings filter: a flat plate on the floor, not standing on the layer scale.
                const fy = baseY - dep * 0.35;
                return (
                  <g key={s.key}>
                    <path d={`M${x} ${fy}h${w + 8}l${sk * 0.25} ${-dep * 0.25}h${-(w + 8)}z`} fill={`url(#l${uid})`} opacity={dim ? 0.35 : 0.95} stroke="var(--text-3)" strokeWidth={0.5} strokeDasharray={dash} />
                    <text x={x + (w + 8) / 2 + sk * 0.12} y={fy - dep * 0.25 - 5} textAnchor="middle" className="mono" fontSize={8.5} fill={dim ? "var(--text-4)" : "var(--text-1)"}>
                      {i + 1}
                    </text>
                    {s.verified && <path d={`M${x - 9} ${fy - 6}l2.5 2.5l5 -5`} fill="none" stroke="var(--verified)" strokeWidth={1} />}
                  </g>
                );
              }
              const [a, l2] = s.layers;
              const y0 = wallY(l2);
              const h = wallY(a) + pitch - y0;
              return (
                <g key={s.key}>
                  <line x1={x} x2={x + w} y1={baseY - dep} y2={baseY - dep} stroke="#9fb0dd" strokeOpacity={0.5} />
                  <line x1={x + w / 2} x2={x + w / 2} y1={y0 + h} y2={baseY - dep} stroke="var(--line-3)" strokeDasharray="1 2" />
                  <path d={`M${x} ${y0}l${-sk * 0.3} ${dep * 0.3}v${h}l${sk * 0.3} ${-dep * 0.3}z`} fill={`url(#l${uid})`} opacity={dim ? 0.22 : 0.55} />
                  <rect x={x} y={y0} width={w} height={h} fill={`url(#l${uid})`} opacity={dim ? 0.38 : 0.95} />
                  <rect x={x - 0.5} y={y0 - 0.5} width={w + 1} height={h + 1} fill="none" stroke="var(--text-3)" strokeWidth={0.5} strokeDasharray={dash} />
                  <text x={x + w / 2} y={y0 - 5} textAnchor="middle" className="mono" fontSize={8.5} fill={dim ? "var(--text-4)" : "var(--text-1)"}>
                    {i + 1}
                  </text>
                  {s.verified && <path d={`M${x + w + 2} ${y0 + 2}l2.5 2.5l5 -5`} fill="none" stroke="var(--verified)" strokeWidth={1} />}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

/** The audit tape: one mark per call along time, its height by how long it ran, its colour by
 *  its state; the parts it named are ticks under it. With `onPick` each mark opens its event and
 *  the open one (`picked`, the row's id) stands out; the legend names the colours. */
export function Tape({ rows, now, onPick, picked = null }: { rows: AuditRow[]; now: number; onPick?: (r: AuditRow) => void; picked?: string | null }) {
  const W = 1200;
  const H = 168;
  const base = 112;
  const list = rows.filter((r) => typeof r.time_ms === "number");
  if (!list.length) return null;
  const t0 = Math.min(...list.map((r) => r.time_ms));
  const t1 = Math.max(now || 0, ...list.map((r) => r.time_ms));
  const span = Math.max(1, t1 - t0);
  const maxMs = Math.max(1, ...list.map((r) => r.ms ?? 0));
  const x = (t: number) => 20 + ((t - t0) / span) * (W - 40);
  const colourOf = (r: AuditRow) => (r.ok === false ? "var(--error)" : r.kind === "setting" ? "var(--behav)" : r.kind === "outcome" ? "var(--verified)" : "#c9d4ea");
  const legend: [string, string][] = [
    ["call or change", "#c9d4ea"],
    ["setting", "var(--behav)"],
    ["outcome", "var(--verified)"],
    ["failed", "var(--error)"],
  ];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role={onPick ? "group" : "img"} aria-label={`Tape of ${list.length} recorded events`}>
      <text x={20} y={14} className="mono" fontSize={10} letterSpacing="0.16em" fill="var(--text-4)">
        {list.length} EVENTS · HEIGHT BY HOW LONG EACH RAN
      </text>
      {legend.map(([word, c], i) => (
        <g key={word} transform={`translate(${W - 20 - legend.slice(i).reduce((a, [w2]) => a + w2.length * 6.1 + 30, 0) + 18} 0)`}>
          <circle cx={4} cy={10.5} r={2.2} fill={c} />
          <text x={12} y={14} className="mono" fontSize={10} fill="var(--text-4)">
            {word}
          </text>
        </g>
      ))}
      <line x1={20} x2={W - 20} y1={base} y2={base} stroke="var(--line-3)" />
      {Array.from({ length: 13 }, (_, i) => (
        <line key={i} x1={20 + (i * (W - 40)) / 12} x2={20 + (i * (W - 40)) / 12} y1={base} y2={i % 3 === 0 ? base + 8 : base + 4} stroke="var(--line-3)" />
      ))}
      {list.map((r) => {
        const h = 8 + 80 * Math.sqrt((r.ms ?? 0) / maxMs);
        const c = colourOf(r);
        const parts = Array.isArray(r.parts) ? r.parts.length : 0;
        const on = picked === r.id;
        const cx = x(r.time_ms);
        const mark = (
          <>
            <line x1={cx} x2={cx} y1={base} y2={base - h} stroke={on ? "var(--sense-pale)" : c} strokeOpacity={on ? 1 : 0.7} strokeWidth={on ? 1.6 : 1} />
            <circle cx={cx} cy={base - h} r={on ? 3 : 1.8} fill={on ? "var(--sense-pale)" : c} />
            {on && <path d={`M${cx - 4} ${base + 12}L${cx} ${base + 6}L${cx + 4} ${base + 12}z`} fill="var(--sense-pale)" />}
            {Array.from({ length: Math.min(8, parts) }, (_, k) => (
              <circle key={k} cx={cx} cy={base + 14 + k * 3.2} r={0.9} fill="#c9d4ea" opacity={0.6} />
            ))}
          </>
        );
        if (!onPick) return <g key={r.id}>{mark}</g>;
        const say = `${new Date(r.time_ms).toLocaleTimeString()} · ${r.kind}${typeof r.ms === "number" ? ` · ${Math.round(r.ms)} ms` : ""}${r.ok === false ? " · failed" : ""}. Open this event.`;
        return (
          <g
            key={r.id}
            className="fig-node"
            role="button"
            tabIndex={-1}
            aria-label={say}
            aria-pressed={on}
            data-e2e="audit-tape-mark"
            onClick={() => onPick(r)}
          >
            <title>{say}</title>
            <rect className="fig-box" x={cx - 4} y={base - 92} width={8} height={104} fill="transparent" stroke="transparent" />
            {mark}
          </g>
        );
      })}
      <text x={20} y={H - 4} className="mono" fontSize={10} fill="var(--text-4)">
        {new Date(t0).toLocaleString()}
      </text>
      <text x={W - 20} y={H - 4} textAnchor="end" className="mono" fontSize={10} fill="var(--text-4)">
        now
      </text>
    </svg>
  );
}

/** A big figure: tiny label, dot-matrix number, caption. */
export function Big({ label, value, caption, tone = "text-text" }: { label: string; value: string; caption?: ReactNode; tone?: string }) {
  return (
    <div className="min-w-0">
      <div className="eyebrow">{label}</div>
      <div className={`mt-3 ${tone}`}>
        <DotText text={value} size={42} />
      </div>
      {caption && <div className="mono mt-2 text-[11px] text-faint">{caption}</div>}
    </div>
  );
}
