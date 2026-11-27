"use client";

/** Drawing of one topology level. Every part is drawn from layout geometry; every lit-able shape
 *  carries `data-live` with the path of the node it stands for, and its state layers (`on`,
 *  `glow`, `trace`, `wake`) are switched by opacity alone. Clickable parts carry `data-key`. */
import { memo, type ReactNode } from "react";
import type { Cartridge, Status } from "@/lib/api";
import { fit, keyOf, roleOf, wrap, type Box, type Cell, type Column, type Edge, type Item, type Label, type Level } from "@/lib/layout";
import { figure, portsOf, type TNode } from "@/lib/topology";

type Ctx = { detail: 0 | 1 | 2; selected: string | null; status: Status | null; carts: Cartridge[]; leaving: Cartridge[]; interactive: boolean };

function aria(n: TNode, status: Status | null): string {
  const fg = figure(n, status);
  return `${n.label}${n.sub ? `, ${n.sub}` : ""}${fg ? `. ${fg}` : ""}${n.children.length ? ". Press Enter to open." : ""}`;
}

/** A clickable part. */
function Node({ n, k, ctx, children, focusable = true }: { n: TNode; k: string; ctx: Ctx; children: ReactNode; focusable?: boolean }) {
  return (
    <g
      data-key={k}
      className={`t-node${ctx.selected === k ? " is-selected" : ""}`}
      tabIndex={ctx.interactive && focusable ? 0 : -1}
      role="button"
      aria-label={aria(n, ctx.status)}
      aria-current={ctx.selected === k ? "true" : undefined}
      opacity={n.present ? 1 : 0.5}
    >
      {children}
    </g>
  );
}

/** The state layers of one shape: drawn once per layer, each switched by opacity. */
function Layers({ shape, glow = true, wake = true }: { shape: (p: { className: string; stroke?: string; fill?: string; strokeWidth?: number; opacity?: number }) => ReactNode; glow?: boolean; wake?: boolean }) {
  return (
    <>
      {glow && shape({ className: "glow", stroke: "var(--signal)", fill: "none", strokeWidth: 5, opacity: 0.1 })}
      {shape({ className: "trace", stroke: "var(--signal-dim)", fill: "none", strokeWidth: 1 })}
      {shape({ className: "on", stroke: "var(--signal)", fill: "none", strokeWidth: 1.25 })}
      {wake && shape({ className: "wake", stroke: "var(--signal)", fill: "none", strokeWidth: 1 })}
    </>
  );
}

function LabelView({ l, ctx, big = false }: { l: Label; ctx: Ctx; big?: boolean }) {
  const size = ctx.detail === 0 ? 11.5 : big ? 14 : 13;
  const y = l.sub ? l.y - 2 : l.y + 4;
  return (
    <g className="lbl">
      {l.leader && <path d={l.leader} fill="none" stroke="var(--line-1)" strokeWidth={1} />}
      {l.leader && l.dot && <circle cx={l.dot.x} cy={l.dot.y} r={1.6} fill="var(--line-3)" />}
      <text x={l.x} y={y} textAnchor={l.anchor} fontSize={size} fontWeight={500} fill="var(--text-1)">
        {l.title}
      </text>
      {l.sub && (
        <text x={l.x} y={y + 15} textAnchor={l.anchor} fontSize={11} fill="var(--text-3)">
          {l.sub}
        </text>
      )}
    </g>
  );
}

/** A small mark that opens when its node works. */
function CellMark({ c, tip }: { c: Cell; tip?: string }) {
  const square = c.kind === "g0";
  return (
    <g data-live={c.key} className="cell" data-tip={tip ?? c.key}>
      {square ? (
        <>
          <rect x={c.x - c.r} y={c.y - c.r} width={c.r * 2} height={c.r * 2} rx={0.8} fill="var(--bg-0)" stroke="var(--line-3)" />
          <rect className="trace" x={c.x - c.r} y={c.y - c.r} width={c.r * 2} height={c.r * 2} rx={0.8} fill="var(--signal-dim)" />
          <rect className="on" x={c.x - c.r} y={c.y - c.r} width={c.r * 2} height={c.r * 2} rx={0.8} fill="var(--signal)" />
        </>
      ) : (
        <>
          <circle cx={c.x} cy={c.y} r={c.r} fill="var(--bg-0)" stroke={c.kind === "tip" ? "var(--text-3)" : "var(--line-3)"} />
          <circle className="trace" cx={c.x} cy={c.y} r={c.r} fill="var(--signal-dim)" />
          <circle className="on" cx={c.x} cy={c.y} r={c.r} fill="var(--signal)" />
        </>
      )}
    </g>
  );
}

function strokeFor(e: Edge): { stroke: string; dash?: string } {
  if (e.kind === "leader") return { stroke: "var(--line-1)" };
  const dash = e.kind === "down" || (e.mode && e.mode !== "direct") ? "1.5 3" : undefined;
  return { stroke: "var(--line-2)", dash };
}

function EdgeView({ e }: { e: Edge }) {
  const s = strokeFor(e);
  return (
    <g>
      <path d={e.d} fill="none" stroke={s.stroke} strokeWidth={1} strokeDasharray={s.dash} strokeLinecap="round" />
      {e.tick && (
        <path d="M -4 -3.5 L 0 0 L -4 3.5" transform={`translate(${e.tick.x} ${e.tick.y}) rotate(${e.tick.a})`} fill="none" stroke="var(--line-3)" strokeWidth={1} strokeLinecap="round" strokeLinejoin="round" />
      )}
    </g>
  );
}

/* ---------- Level 0 parts ---------- */

function Mantle({ it, ctx }: { it: Extract<Item, { kind: "mantle" }>; ctx: Ctx }) {
  const n = it.node;
  return (
    <Node n={n} k={it.key} ctx={ctx}>
      <path d={it.d} fill="transparent" />
      <g data-live={it.key}>
        <path d={it.d} fill="var(--bg-1)" stroke="var(--line-2)" />
        <path className="hi" d={it.d} fill="var(--bg-2)" stroke="var(--line-3)" />
        <path className="sel" d={it.d} fill="var(--bg-3)" stroke="var(--text-2)" />
        <Layers shape={(p) => <path key={p.className} d={it.d} {...p} />} />
      </g>
      {it.links.map((d, i) => (
        <path key={i} d={d} fill="none" stroke="var(--line-2)" strokeWidth={0.75} aria-hidden />
      ))}
      {it.cells.map((c) => (
        <g key={c.key} data-key={c.key} className="t-node">
          <circle cx={c.x} cy={c.y} r={Math.max(6, c.r + 4)} fill="transparent" />
          <CellMark c={c} />
        </g>
      ))}
      <LabelView l={it.label} ctx={ctx} big />
    </Node>
  );
}

function Spine({ it, ctx }: { it: Extract<Item, { kind: "spine" }>; ctx: Ctx }) {
  return (
    <Node n={it.node} k={it.key} ctx={ctx}>
      <rect x={it.box.x} y={it.box.y} width={it.box.w} height={it.box.h} fill="transparent" />
      <g data-live={it.key}>
        <line x1={it.x} x2={it.x} y1={it.top} y2={it.bottom} stroke="var(--line-2)" />
        <line className="hi" x1={it.x} x2={it.x} y1={it.top} y2={it.bottom} stroke="var(--line-3)" strokeWidth={1.5} />
        <Layers glow={false} wake={false} shape={(p) => <line key={p.className} x1={it.x} x2={it.x} y1={it.top} y2={it.bottom} {...p} />} />
      </g>
      {it.vertebrae.map((v, i) => {
        const w = v.w ?? 20;
        const h = v.h ?? 5;
        const r = { x: v.x - w / 2, y: v.y - h / 2, width: w, height: h, rx: h / 2 };
        return (
          <g key={v.key} data-key={v.key} className="t-node" data-tip={v.key}>
            <rect x={v.x - w / 2 - 6} y={v.y - 9} width={w + 12} height={18} fill="transparent" />
            <g data-live={v.key}>
              <rect {...r} fill="var(--bg-2)" stroke="var(--line-3)" />
              <Layers glow={false} shape={(p) => <rect key={p.className} {...r} {...p} />} />
              <rect className="on" {...r} fill="var(--signal)" stroke="none" />
            </g>
            <LabelView l={it.vlabels[i]} ctx={ctx} />
          </g>
        );
      })}
      <LabelView l={it.label} ctx={ctx} />
    </Node>
  );
}

function Ring({ it, ctx, clickable }: { it: Extract<Item, { kind: "ring" }>; ctx: Ctx; clickable: boolean }) {
  const body = (
    <>
      <ellipse cx={it.cx} cy={it.cy} rx={it.rx + 10} ry={it.ry + 8} fill="transparent" />
      <g data-live={clickable ? it.key : keyOf(it.node)}>
        <ellipse cx={it.cx} cy={it.cy} rx={it.rx} ry={it.ry} fill="var(--bg-1)" stroke="var(--line-2)" />
        <ellipse className="hi" cx={it.cx} cy={it.cy} rx={it.rx} ry={it.ry} fill="var(--bg-2)" stroke="var(--line-3)" />
        <ellipse className="sel" cx={it.cx} cy={it.cy} rx={it.rx} ry={it.ry} fill="var(--bg-3)" stroke="var(--text-2)" />
        <Layers shape={(p) => <ellipse key={p.className} cx={it.cx} cy={it.cy} rx={it.rx} ry={it.ry} {...p} />} />
      </g>
      <ellipse cx={it.cx} cy={it.cy} rx={it.rx * 0.72} ry={it.ry * 0.62} fill="none" stroke="var(--line-1)" />
      {it.joints.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={1.5} fill="var(--line-3)" />
      ))}
      {it.label && <LabelView l={it.label} ctx={ctx} />}
    </>
  );
  if (!clickable) return <g aria-hidden>{body}</g>;
  return (
    <Node n={it.node} k={it.key} ctx={ctx}>
      {body}
    </Node>
  );
}

function Arm({ it, ctx, clickable }: { it: Extract<Item, { kind: "arm" }>; ctx: Ctx; clickable: boolean }) {
  const a = it.arm;
  const liveKey = keyOf(it.node);
  const body = (
    <>
      <path d={a.d} fill="none" stroke="transparent" strokeWidth={18} style={{ pointerEvents: "stroke" }} />
      <circle cx={a.tip.x} cy={a.tip.y} r={14} fill="transparent" />
      <g data-live={liveKey}>
        <path d={a.body} fill="var(--line-2)" />
        <path className="hi" d={a.body} fill="var(--line-3)" />
        <path className="sel" d={a.body} fill="var(--text-2)" />
        <path className="glow" d={a.d} fill="none" stroke="var(--signal)" strokeWidth={6} opacity={0.08} strokeLinecap="round" />
        <path className="trace" d={a.d} fill="none" stroke="var(--signal-dim)" strokeWidth={1.25} strokeLinecap="round" />
        <path className="on" d={a.d} fill="none" stroke="var(--signal)" strokeWidth={1.5} strokeLinecap="round" />
      </g>
      {a.window && (
        <g data-live={liveKey}>
          <line x1={a.window.x} x2={a.window.x + (a.window.w ?? 0)} y1={a.window.y} y2={a.window.y} stroke="var(--line-3)" strokeWidth={2} strokeLinecap="round" />
          <line className="trace" x1={a.window.x} x2={a.window.x + (a.window.w ?? 0)} y1={a.window.y} y2={a.window.y} stroke="var(--signal-dim)" strokeWidth={2} strokeLinecap="round" />
          <line className="on" x1={a.window.x} x2={a.window.x + (a.window.w ?? 0)} y1={a.window.y} y2={a.window.y} stroke="var(--signal)" strokeWidth={2} strokeLinecap="round" />
          <line x1={a.tip.x} x2={a.tip.x} y1={a.tip.y + 6} y2={a.window.y - 3} stroke="var(--line-1)" strokeDasharray="1 3" />
        </g>
      )}
      {a.dendrites.map((c, i) => (
        <CellMark key={"s" + i} c={{ ...c, key: a.marks.find((m) => m.kind === "tip")?.key ?? c.key }} tip={c.key} />
      ))}
      {a.marks.map((m) => (
        <g key={m.key} data-key={clickable ? undefined : m.key} className={clickable ? undefined : "t-node"} data-tip={m.key}>
          {!clickable && <circle cx={m.x} cy={m.y} r={10} fill="transparent" />}
          <CellMark c={m} />
        </g>
      ))}
      {it.label && (
        <g>
          <text x={it.label.x} y={it.label.y} textAnchor="middle" fontSize={ctx.detail === 0 ? 10.5 : 12} fontWeight={500} fill="var(--text-2)">
            {it.label.title}
          </text>
          {it.label.sub && (
            <text x={it.label.x} y={it.label.y + 14} textAnchor="middle" fontSize={10.5} fill="var(--text-3)">
              {it.label.sub}
            </text>
          )}
        </g>
      )}
    </>
  );
  if (!clickable) return <g>{body}</g>;
  return (
    <Node n={it.node} k={it.key} ctx={ctx}>
      {body}
    </Node>
  );
}

/** A model's layer column with its cartridge dock beside it. */
export function ColumnView({ c, carts, leaving, big = false }: { c: Column; carts: Cartridge[]; leaving: Cartridge[]; big?: boolean }) {
  const pitch = c.h / c.layers;
  const bars = pitch >= 2.4;
  const own = [...carts.filter((x) => x.loaded && (x.model ?? x.base) === c.model), ...leaving.filter((x) => (x.model ?? x.base) === c.model)];
  const dx = c.x + c.w + (big ? 6 : 3);
  const dw = big ? 4 : 2.5;
  return (
    <g data-dock={c.model}>
      {bars ? (
        Array.from({ length: c.layers }, (_, i) => <rect key={i} x={c.x} y={c.y + i * pitch} width={c.w} height={Math.max(1, pitch - (big ? 1.6 : 1))} rx={0.5} fill="var(--line-2)" />)
      ) : (
        <>
          <rect x={c.x} y={c.y} width={c.w} height={c.h} rx={1} fill="var(--line-2)" />
          {Array.from({ length: Math.floor(c.layers / 4) - 1 }, (_, i) => (
            <line key={i} x1={c.x} x2={c.x + c.w} y1={c.y + (i + 1) * 4 * pitch} y2={c.y + (i + 1) * 4 * pitch} stroke="var(--bg-1)" strokeWidth={1} />
          ))}
        </>
      )}
      <line x1={dx + dw / 2} x2={dx + dw / 2} y1={c.y} y2={c.y + c.h} stroke="var(--line-2)" strokeDasharray="1.5 2.5" />
      {own.map((x, k) => {
        const [a, b] = x.layers ?? [0, c.layers - 1];
        const y = c.y + a * pitch;
        const h = (b - a + 1) * pitch;
        const off = dx + k * (dw + 2);
        const gone = leaving.includes(x);
        return (
          <g key={x.id} data-cart={x.id} className={gone ? "dock-leave" : undefined}>
            <rect x={off} y={y} width={dw} height={h} rx={dw / 2} fill="var(--learned)" />
            {big &&
              bars &&
              Array.from({ length: b - a + 1 }, (_, i) => <line key={i} className="dock-tick" x1={c.x + c.w + 1} x2={c.x + c.w + 3.5} y1={y + (i + 0.5) * pitch} y2={y + (i + 0.5) * pitch} stroke="var(--learned)" strokeWidth={1} />)}
          </g>
        );
      })}
    </g>
  );
}

/** A small cartridge glyph: a card whose hatch density shows its bit depth. */
export function Hatch({ x, y, w, h, bits, loaded, id }: { x: number; y: number; w: number; h: number; bits?: string; loaded: boolean; id?: string }) {
  const b = Number(/\d+/.exec(bits ?? "")?.[0] ?? 8);
  const step = b <= 4 ? 3.2 : b <= 8 ? 2.2 : 1.6;
  const lines = Math.floor(h / step);
  return (
    <g data-cart-card={id}>
      <rect x={x} y={y} width={w} height={h} rx={1.5} fill={loaded ? "oklch(0.8 0.075 295 / 0.16)" : "var(--bg-1)"} stroke="var(--learned)" strokeOpacity={loaded ? 0.9 : 0.5} />
      {Array.from({ length: lines - 1 }, (_, i) => (
        <line key={i} x1={x + 1.5} x2={x + w - 1.5} y1={y + (i + 1) * step} y2={y + (i + 1) * step} stroke="var(--learned)" strokeOpacity={0.45} strokeWidth={0.75} />
      ))}
    </g>
  );
}

function Organ({ it, ctx }: { it: Extract<Item, { kind: "organ" }>; ctx: Ctx }) {
  const b = it.box;
  let body: ReactNode = null;
  if (it.organ === "runtime") {
    const cx = b.x + b.w / 2;
    const cy = b.y + b.h / 2;
    body = (
      <g data-live={it.key}>
        <circle cx={cx} cy={cy} r={4.5} fill="var(--bg-0)" stroke="var(--text-3)" />
        <circle className="hi" cx={cx} cy={cy} r={4.5} fill="var(--bg-2)" stroke="var(--text-2)" />
        <circle cx={cx} cy={cy} r={1.5} fill="var(--text-3)" />
        <Layers shape={(p) => <circle key={p.className} cx={cx} cy={cy} r={4.5} {...p} />} />
      </g>
    );
  } else if (it.organ === "memory") {
    // Tiers: core, active and archive, and the dashed temporary notes.
    const rows = [0.46, 0.72, 1, 0.72];
    const h = Math.max(2.6, b.h / 7);
    const shapes = (p: { className?: string; stroke?: string; fill?: string }) =>
      rows.map((r, i) => <rect key={i} className={p.className} x={b.x + b.w - b.w * r} y={b.y + i * (h + 2.6)} width={b.w * r} height={h} rx={h / 2} fill={p.fill ?? "var(--bg-2)"} stroke={p.stroke ?? "var(--line-3)"} strokeDasharray={i === 3 ? "1.5 2" : undefined} />);
    body = (
      <g data-live={it.key}>
        <rect x={b.x - 6} y={b.y - 6} width={b.w + 12} height={b.h + 10} fill="transparent" />
        {shapes({})}
        <g className="hi">{shapes({ fill: "var(--bg-3)", stroke: "var(--text-3)" })}</g>
        <g className="trace">{shapes({ fill: "none", stroke: "var(--signal-dim)" })}</g>
        <g className="on">{shapes({ fill: "none", stroke: "var(--signal)" })}</g>
      </g>
    );
  } else if (it.organ === "engine") {
    const cols = it.extra.columns ?? [];
    body = (
      <g data-live={it.key}>
        <rect x={b.x} y={b.y} width={b.w} height={b.h} fill="transparent" />
        {cols.map((c) => (
          <g key={c.model} data-live={`${it.key}/micromodels/${c.model}`.replace(/\./g, "-")}>
            <ColumnView c={c} carts={ctx.carts} leaving={ctx.leaving} />
          </g>
        ))}
        <g className="hi">
          {cols.map((c) => (
            <rect key={c.model} x={c.x - 1} y={c.y - 1} width={c.w + 2} height={c.h + 2} rx={1.5} fill="none" stroke="var(--line-3)" />
          ))}
        </g>
        <g className="trace">
          {cols.map((c) => (
            <rect key={c.model} x={c.x - 1} y={c.y - 1} width={c.w + 2} height={c.h + 2} rx={1.5} fill="none" stroke="var(--signal-dim)" />
          ))}
        </g>
        <g className="on">
          {cols.map((c) => (
            <rect key={c.model} x={c.x - 1} y={c.y - 1} width={c.w + 2} height={c.h + 2} rx={1.5} fill="none" stroke="var(--signal)" />
          ))}
        </g>
      </g>
    );
  } else if (it.organ === "learning") {
    const t = it.extra.tray!;
    const carts = ctx.carts.slice(0, 6);
    body = (
      <g data-live={it.key}>
        <rect x={b.x} y={b.y} width={b.w} height={b.h} fill="transparent" />
        <rect data-dock-src x={t.x} y={t.y} width={t.w} height={t.h} rx={3} fill="var(--bg-1)" stroke="var(--line-2)" strokeDasharray={carts.length ? undefined : "2 2"} />
        <rect className="hi" x={t.x} y={t.y} width={t.w} height={t.h} rx={3} fill="var(--bg-2)" stroke="var(--line-3)" />
        {carts.map((c, i) => (
          <Hatch key={c.id} id={c.id} x={t.x + 4 + i * 9} y={t.y + 2.5} w={6} h={t.h - 5} bits={c.quantization} loaded={c.loaded} />
        ))}
        <Layers glow={false} shape={(p) => <rect key={p.className} x={t.x} y={t.y} width={t.w} height={t.h} rx={3} {...p} />} />
      </g>
    );
  } else if (it.organ === "substrate") {
    const band = it.extra.band!;
    body = (
      <g data-live={it.key}>
        <rect x={b.x} y={b.y} width={b.w} height={b.h} fill="transparent" />
        <line x1={band.x0} x2={band.x1} y1={band.y} y2={band.y} stroke="var(--line-2)" />
        <line className="hi" x1={band.x0} x2={band.x1} y1={band.y} y2={band.y} stroke="var(--line-3)" strokeWidth={1.5} />
        <Layers glow={false} wake={false} shape={(p) => <line key={p.className} x1={band.x0} x2={band.x1} y1={band.y} y2={band.y} {...p} />} />
      </g>
    );
  }
  return (
    <Node n={it.node} k={it.key} ctx={ctx}>
      {body}
      <LabelView l={it.label} ctx={ctx} />
    </Node>
  );
}

/* ---------- Drill-down parts ---------- */

function rectLayers(b: Box, rx: number) {
  return <Layers shape={(p) => <rect key={p.className} x={b.x} y={b.y} width={b.w} height={b.h} rx={rx} {...p} />} />;
}

function Ports({ n, b }: { n: TNode; b: Box }) {
  const { ins, outs } = portsOf(n);
  const tick = (x: number, k: number, total: number, dir: 1 | -1) => {
    const y = b.y + b.h / 2 + (k - (total - 1) / 2) * 7;
    return <line key={`${dir}${k}`} x1={x} x2={x + dir * 4} y1={y} y2={y} stroke="var(--line-3)" />;
  };
  return (
    <g aria-hidden>
      {ins.slice(0, 4).map((_, k) => tick(b.x, k, Math.min(4, ins.length), -1))}
      {outs.slice(0, 4).map((_, k) => tick(b.x + b.w, k, Math.min(4, outs.length), 1))}
    </g>
  );
}

function Card({ it, ctx }: { it: Extract<Item, { kind: "card" }>; ctx: Ctx }) {
  const n = it.node;
  const b = it.box;
  const pad = b.w < 170 ? 10 : 14;
  const tw = b.w - pad * 2 - (n.cartridge ? 16 : 0);
  const small = b.h < 70;
  const role = roleOf(n);
  const roleLines = small ? [] : wrap(role, tw, 11.5, b.h >= 100 ? 3 : b.h >= 84 ? 2 : 1);
  const kids = n.children;
  const dotsY = b.y + b.h - 13;
  const maxDots = Math.max(0, Math.floor((tw - 56) / 9));
  const showId = ctx.detail === 2 && b.h >= 96 && !kids.length && roleLines.length < 3 && n.ids[0];
  const absent = !n.present;
  return (
    <Node n={n} k={it.key} ctx={ctx}>
      <g data-live={it.key}>
        <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={6} fill={absent ? "none" : "var(--bg-2)"} stroke={absent ? "var(--line-1)" : "var(--line-2)"} strokeDasharray={absent ? "3 3" : undefined} />
        <rect className="hi" x={b.x} y={b.y} width={b.w} height={b.h} rx={6} fill="var(--bg-3)" stroke="var(--line-3)" />
        <rect className="sel" x={b.x} y={b.y} width={b.w} height={b.h} rx={6} fill="var(--bg-3)" stroke="var(--text-2)" />
        {rectLayers(b, 6)}
        <rect className="load" x={b.x + 1} y={b.y + b.h - 3} width={b.w - 2} height={2} rx={1} fill="var(--signal)" style={{ transform: "scaleX(0)", transformBox: "fill-box", transformOrigin: "left" }} />
      </g>
      <rect className="ring" x={b.x - 4} y={b.y - 4} width={b.w + 8} height={b.h + 8} rx={9} fill="none" stroke="var(--text-2)" strokeWidth={2} />
      <Ports n={n} b={b} />
      <text x={b.x + pad} y={b.y + (small ? b.h / 2 + 4.5 : 22)} fontSize={13} fontWeight={500} fill={absent ? "var(--text-4)" : "var(--text-1)"}>
        {fit(n.label, tw - (kids.length ? 34 : 0), 13)}
      </text>
      {kids.length > 0 && !small && (
        <text x={b.x + b.w - pad} y={b.y + 22} textAnchor="end" fontSize={10.5} className="mono" fill="var(--text-3)">
          {kids.length}
        </text>
      )}
      {roleLines.map((l, i) => (
        <text key={i} x={b.x + pad} y={b.y + 40 + i * 15} fontSize={11.5} fill="var(--text-3)">
          {l}
        </text>
      ))}
      {n.cartridge && <Hatch x={b.x + b.w - pad - 9} y={b.y + 11} w={9} h={13} bits={n.cartridge.quantization} loaded={n.cartridge.loaded} />}
      {kids.length > 0 && b.h >= 84 && (
        <g>
          {kids.slice(0, maxDots).map((c, i) => (
            <CellMark key={c.key} c={{ key: keyOf(c), x: b.x + pad + 3 + i * 9, y: dotsY, r: 2.4, kind: "cell" }} tip={keyOf(c)} />
          ))}
          {kids.length > maxDots && (
            <text x={b.x + pad + maxDots * 9 + 2} y={dotsY + 3.5} fontSize={10} className="mono" fill="var(--text-4)">
              +{kids.length - maxDots}
            </text>
          )}
        </g>
      )}
      {showId && (
        <text x={b.x + pad} y={b.y + b.h - 11} fontSize={10} className="mono" fill="var(--text-4)">
          {fit(n.ids[0], tw, 10, true)}
        </text>
      )}
    </Node>
  );
}

function Plate({ it, ctx }: { it: Extract<Item, { kind: "plate" }>; ctx: Ctx }) {
  const n = it.node;
  const b = it.box;
  const narrow = b.w < 420;
  const textW = narrow ? b.w - 28 : Math.min(300, b.w * 0.36);
  const name = n.sub ? n.sub.charAt(0).toUpperCase() + n.sub.slice(1) : "";
  const about = wrap(n.about, textW, 11.5, narrow ? 1 : b.h > 110 ? 3 : 2);
  return (
    <Node n={n} k={it.key} ctx={ctx}>
      <g data-live={it.key}>
        <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={8} fill="var(--bg-1)" stroke="var(--line-2)" />
        <rect className="hi" x={b.x} y={b.y} width={b.w} height={b.h} rx={8} fill="var(--bg-2)" stroke="var(--line-3)" />
        <rect className="sel" x={b.x} y={b.y} width={b.w} height={b.h} rx={8} fill="var(--bg-3)" stroke="var(--text-2)" />
        {rectLayers(b, 8)}
      </g>
      <rect className="ring" x={b.x - 4} y={b.y - 4} width={b.w + 8} height={b.h + 8} rx={11} fill="none" stroke="var(--text-2)" strokeWidth={2} />
      <text x={b.x + 16} y={b.y + 27} fontSize={17} fontWeight={500} className="mono" fill="var(--text-1)">
        {n.label}
      </text>
      <text x={b.x + 16 + n.label.length * 10.5 + 10} y={b.y + 26} fontSize={13} fontWeight={500} fill="var(--text-1)">
        {fit(name, textW - n.label.length * 10.5 - 10, 13)}
      </text>
      {about.map((l, i) => (
        <text key={i} x={b.x + 16} y={b.y + 46 + i * 15} fontSize={11.5} fill="var(--text-3)">
          {l}
        </text>
      ))}
      {it.chips.map((c) =>
        c.kind === "g0" ? (
          <g key={c.key} data-key={c.key} className="t-node" data-tip={c.key}>
            <circle cx={c.x} cy={c.y} r={c.r + 6} fill="transparent" />
            <CellMark c={c} />
            <text x={c.x} y={c.y + c.r + 13} textAnchor="middle" fontSize={10} className="mono" fill="var(--text-3)">
              {c.label}
            </text>
          </g>
        ) : (
          <g key={c.key} data-key={c.key} className="t-node" data-tip={c.key}>
            <g data-live={c.key}>
              <rect x={c.x} y={c.y} width={c.w} height={c.h} rx={4} fill="var(--bg-2)" stroke="var(--line-2)" />
              <rect className="hi" x={c.x} y={c.y} width={c.w} height={c.h} rx={4} fill="var(--bg-3)" stroke="var(--line-3)" />
              <Layers glow={false} wake={false} shape={(p) => <rect key={p.className} x={c.x} y={c.y} width={c.w} height={c.h} rx={4} {...p} />} />
            </g>
            <text x={c.x + 10} y={(c.y ?? 0) + (c.h ?? 0) / 2 + 4} fontSize={11.5} fill="var(--text-2)">
              {c.label}
            </text>
          </g>
        ),
      )}
    </Node>
  );
}

function ModelCard({ it, ctx }: { it: Extract<Item, { kind: "model" }>; ctx: Ctx }) {
  const n = it.node;
  const b = it.box;
  const m = n.model;
  const tx = it.column.x + it.column.w + 34;
  const tw = b.x + b.w - tx - 14;
  const own = ctx.carts.filter((c) => (c.model ?? c.base) === m?.id);
  const loaded = own.filter((c) => c.loaded).length;
  const gb = m ? (m.bytes / (1 << 30)).toFixed(2) : "";
  return (
    <Node n={n} k={it.key} ctx={ctx}>
      <g data-live={it.key}>
        <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={8} fill="var(--bg-1)" stroke="var(--line-2)" />
        <rect className="hi" x={b.x} y={b.y} width={b.w} height={b.h} rx={8} fill="var(--bg-2)" stroke="var(--line-3)" />
        <rect className="sel" x={b.x} y={b.y} width={b.w} height={b.h} rx={8} fill="var(--bg-3)" stroke="var(--text-2)" />
        {rectLayers(b, 8)}
      </g>
      <rect className="ring" x={b.x - 4} y={b.y - 4} width={b.w + 8} height={b.h + 8} rx={11} fill="none" stroke="var(--text-2)" strokeWidth={2} />
      <ColumnView c={it.column} carts={ctx.carts} leaving={ctx.leaving} big />
      <text x={tx} y={b.y + 30} fontSize={14} fontWeight={500} className="mono" fill="var(--text-1)">
        {fit(n.label, tw, 14, true)}
      </text>
      <text x={tx} y={b.y + 48} fontSize={11.5} fill="var(--text-3)">
        {fit("base cartridges and upstream weights, read only", tw, 11.5)}
      </text>
      <text x={tx} y={b.y + 74} fontSize={11.5} className="mono" fill="var(--text-2)">
        {m ? `${m.precision === "default" ? "default precision" : `${m.precision}-bit`} · ${gb} GB · ${it.column.layers} layers` : ""}
      </text>
      <text x={tx} y={b.y + 92} fontSize={11.5} fill={loaded ? "var(--learned)" : "var(--text-3)"}>
        {own.length ? `${loaded} of ${own.length} cartridges docked` : "no cartridges docked"}
      </text>
    </Node>
  );
}

/* ---------- One level ---------- */

type LevelProps = { level: Level; selected: string | null; status: Status | null; carts: Cartridge[]; leaving: Cartridge[]; interactive: boolean };

const LevelView = memo(function LevelView({ level, selected, status, carts, leaving, interactive }: LevelProps) {
  const ctx: Ctx = { detail: level.detail, selected, status, carts, leaving, interactive };
  const L0 = level.focus.layout === "organism";
  return (
    <>
      {level.frame && <path d={level.frame} fill="none" stroke="var(--line-1)" aria-hidden />}
      {level.ground && (
        <g aria-hidden>
          <line x1={level.ground.x0} x2={level.ground.x1} y1={level.ground.y} y2={level.ground.y} stroke="var(--line-2)" />
          {level.ground.label && (
            <text x={(level.ground.x0 + level.ground.x1) / 2} y={level.ground.y + 20} textAnchor="middle" fontSize={11} fill="var(--text-3)">
              {level.ground.label}
            </text>
          )}
        </g>
      )}
      {level.notes?.map((n, i) => (
        <text key={i} x={n.x} y={n.y} textAnchor={n.anchor} fontSize={11} className="mono" fill="var(--text-4)" aria-hidden>
          {n.text}
        </text>
      ))}
      {level.band && (
        <g aria-hidden>
          <line x1={level.W < 560 ? 12 : 32} x2={level.W - (level.W < 560 ? 12 : 32)} y1={level.band.y} y2={level.band.y} stroke="var(--line-1)" />
          <text x={level.W < 560 ? 12 : 32} y={level.band.y + 18} fontSize={11} letterSpacing="0.08em" className="mono" fill="var(--text-3)">
            {level.band.label.toUpperCase()}
          </text>
        </g>
      )}
      <g aria-hidden>
        {level.edges.map((e, i) => (
          <EdgeView key={i} e={e} />
        ))}
        {level.detail === 2 &&
          level.stubs.map((s) => (
            <g key={s.key + s.dir}>
              <path d={s.d} stroke="var(--line-2)" strokeDasharray="1.5 3" fill="none" />
              <text x={s.x} y={s.y - 6} textAnchor={s.dir === "in" ? "end" : "start"} fontSize={10.5} fill="var(--text-3)">
                {fit(`${s.dir === "in" ? "from" : "to"} ${s.label}`, 160, 10.5)}
              </text>
            </g>
          ))}
      </g>
      <g className="routes" aria-hidden />
      {level.items.map((it) => {
        switch (it.kind) {
          case "mantle":
            return <Mantle key={it.key} it={it} ctx={ctx} />;
          case "spine":
            return <Spine key={it.key} it={it} ctx={ctx} />;
          case "ring":
            return <Ring key={it.key} it={it} ctx={ctx} clickable={L0} />;
          case "arm":
            return <Arm key={it.key} it={it} ctx={ctx} clickable={!it.key.endsWith("/.arm")} />;
          case "organ":
            return <Organ key={it.key} it={it} ctx={ctx} />;
          case "plate":
            return <Plate key={it.key} it={it} ctx={ctx} />;
          case "model":
            return <ModelCard key={it.key} it={it} ctx={ctx} />;
          default:
            return <Card key={it.key} it={it} ctx={ctx} />;
        }
      })}
      <g className="pulses" aria-hidden />
    </>
  );
});

export default LevelView;
