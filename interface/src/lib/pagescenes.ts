/** Scenes for the centrepieces of the data pages, drawn in the same language as the topology and
 *  painted by the same painter (depth, bloom, mist, grain). Every point stands for a real record. */
import { bytes, type AuditRow, type Cartridge } from "./api";
import { LEVELS } from "./comprehension";
import { gateLine, learnedCounts, learnedLine, type Learned } from "./learned";
import type { LiveWriter } from "./livefeed";
import { maskName } from "./privacy";
import { clamp, emptyScene, hash01, keyOf, type Scene, type XY } from "./scene";
import { drawStrata, type Folder, type Tier } from "./strata";
import type { TNode } from "./topology";

/** Learning: the calls the engine has recorded flow as points along a funnel toward the cartridge
 *  slot, one cluster per call (one point per part it named), older calls deeper and softer,
 *  verified outcomes brighter. Nothing enters the slot until the engine reports a learned piece;
 *  the slot then holds one spectral bar per cartridge. The caption counts the learned pieces the
 *  engine reports (job filters, lenses, preference cartridges, G3 muscle updates) and the split
 *  gate's verdicts; with only the cartridges known, it counts what they say. */
export function learningScene(focus: TNode, rows: AuditRow[], carts: Cartridge[], W: number, H: number, learned?: Learned | null): Scene {
  const got = learned ?? learnedCounts({ carts });
  const S = emptyScene(focus, W, H, "plate", W < 560 ? 0 : W < 980 ? 1 : 2);
  const narrow = W < 640;
  const slot = { x: W * (narrow ? 0.84 : 0.8), y: H * 0.5, h: clamp(H * 0.46, 60, 300) };
  const x0 = narrow ? 16 : W * 0.06;
  const list = [...rows].filter((r) => typeof r.time_ms === "number").sort((a, b) => a.time_ms - b.time_ms);
  const n = list.length;
  // The funnel's guide lines: the paths evidence takes into the slot.
  const lanes = 9;
  for (let j = 0; j < lanes; j++) {
    const v = j / (lanes - 1) - 0.5;
    const pts = Array.from({ length: 40 }, (_, q) => {
      const t = q / 39;
      const spread = Math.pow(1 - t, 1.6);
      return { x: x0 + (slot.x - 18 - x0) * t, y: slot.y + v * (H * 0.82 * spread + slot.h * 0.7 * (1 - spread)) };
    });
    S.lines.push({ pts, a: 0.16 + (1 - Math.abs(v) * 2) * 0.18, w: 0.6, dash: j % 2 ? [1, 4] : undefined });
  }
  list.forEach((r, i) => {
    const t = n > 1 ? i / (n - 1) : 1;
    // Older calls sit far out in the field; the newest nearest the slot.
    const tx = 0.05 + t * 0.78;
    const v = hash01(i, 3) - 0.5;
    const spread = Math.pow(1 - tx, 1.6);
    const cx = x0 + (slot.x - 18 - x0) * tx;
    const cy = slot.y + v * (H * 0.82 * spread + slot.h * 0.7 * (1 - spread));
    const verified = r.kind === "outcome" && r.ok !== false;
    const parts = Array.isArray(r.parts) ? r.parts.length : 0;
    const m = Math.max(1, Math.min(14, parts || 1));
    const z = 0.62 - t * 0.58;
    for (let q = 0; q < m; q++) {
      const rr = 2.2 * Math.sqrt(q);
      const a = q * 2.39996;
      S.pts.push({ x: cx + Math.cos(a) * rr, y: cy + Math.sin(a) * rr, r: verified ? 1.3 : 0.85, a: verified ? 1 : 0.45 + t * 0.4, k: "", t: "b", z: clamp(z, 0.05, 0.95) });
    }
  });
  // The slot: one spectral bar per cartridge, or an empty slot of faint points.
  const shown = carts.slice(0, 10);
  if (shown.length) {
    shown.forEach((c, j) => {
      S.docks.push({ cart: c.id, x: slot.x + j * 9, y: slot.y - slot.h / 2, w: 4, h: slot.h });
    });
  } else {
    for (let y = -slot.h / 2; y <= slot.h / 2; y += 3)
      for (let x = 0; x < 9; x += 3) S.pts.push({ x: slot.x + x, y: slot.y + y, r: 0.55, a: 0.32, k: "", t: "p", z: 0.1 });
  }
  S.lines.push({ pts: [{ x: slot.x - 6, y: slot.y - slot.h / 2 - 8 }, { x: slot.x - 6, y: slot.y + slot.h / 2 + 8 }], a: 0.4, w: 0.7 });
  // Beside the slot, one bead per learned piece the engine reports (job filters, lenses, judge
  // read-outs, preference cartridges, G3 muscle updates), bottom up, at most what fits.
  const pieces = got.total ?? 0;
  if (pieces > 0) {
    const bx = slot.x + Math.max(1, shown.length) * 9 + 10;
    const fit = Math.max(1, Math.floor(slot.h / 5));
    const beads = Math.min(pieces, fit);
    for (let q = 0; q < beads; q++) S.pts.push({ x: bx, y: slot.y + slot.h / 2 - q * 5, r: 1.2, a: 0.95, k: "", t: "p", z: 0.04 });
    if (!narrow) S.texts.push({ x: bx + 8, y: slot.y + slot.h / 2 + 4, text: `${pieces} learned${pieces > beads ? ` · ${beads} shown` : ""}`, anchor: "start", kind: "mono", size: 9.5, quiet: true });
  }
  if (!narrow) {
    S.lines.push({ pts: [{ x: slot.x + 14, y: slot.y - slot.h / 2 }, { x: slot.x + 50, y: slot.y - slot.h / 2 - 30 }, { x: slot.x + 64, y: slot.y - slot.h / 2 - 30 }], a: 0.45, w: 0.6 });
    S.texts.push({ x: slot.x + 70, y: slot.y - slot.h / 2 - 26, text: shown.length ? `${shown.length} ${shown.length === 1 ? "cartridge" : "cartridges"}` : "cartridge slot, empty", anchor: "start", kind: "mono", size: 11 });
    S.texts.push({ x: x0, y: 26, text: "evidence from verified work", anchor: "start", kind: "caps" });
    S.texts.push({ x: x0, y: 44, text: `${n} calls recorded · ${list.filter((r) => r.kind === "outcome" && r.ok !== false).length} verified outcomes · ${learnedLine(got)}`, anchor: "start", kind: "mono", size: 10.5, quiet: true });
    S.texts.push({ x: x0, y: 60, text: gateLine(got), anchor: "start", kind: "mono", size: 10, quiet: true });
    S.texts.push({ x: x0, y: H - 14, text: "older", anchor: "start", kind: "mono", size: 9.5, quiet: true });
    S.texts.push({ x: slot.x - 24, y: H - 14, text: "newer", anchor: "end", kind: "mono", size: 9.5, quiet: true });
  }
  S.mist.push({ x: slot.x, y: slot.y, r: slot.h * 1.2, a: 0.4 });
  S.mist.push({ x: W * 0.4, y: H * 0.5, r: W * 0.4, a: 0.3 });
  return S;
}

/** Memory: the four tiers as strata seen at an angle, each project's folder a column through
 *  them, labelled on leaders. Drawn from `/api/projects` (and per-tier counts when the engine
 *  reports them). */
export function memoryScene(focus: TNode, folders: Folder[], W: number, H: number, ownIndex: number | null): Scene {
  const S = emptyScene(focus, W, H, "plate", W < 560 ? 0 : W < 980 ? 1 : 2);
  const narrow = W < 640;
  const labelW = narrow ? 0 : clamp(W * 0.2, 150, 330);
  const u = clamp(Math.min(W, H) / 520, 0.8, 1.6);
  const w = Math.min(W - labelW * 2 - 40, H * 2.6);
  const avail = H - (narrow ? 70 : 96);
  // The stack: four planes of depth d, each `gap` below the last.
  const d = clamp(avail * 0.26, 44, 220);
  const gap = (avail - d * 0.95) / 3;
  const top = narrow ? 30 : 46;
  const stores = focus.children.find((c) => c.key === "stores");
  const tiersNode = stores?.children.find((c) => c.ids.includes("nmmem.tiers"));
  const out = drawStrata(S, { cx: W / 2, top, w, d, gap, sp: clamp(w / 110, 5, 10), u, tierKey: () => keyOf(tiersNode ?? stores ?? focus), folderKey: keyOf(stores ?? focus), folders });
  const says: Record<Tier, string> = { core: "identity, objectives, most used", active: "topics in use", archive: "older topics, never deleted", temporary: "notes: 3 uses save one" };
  out.planes.forEach((p, i) => {
    const sum = folders.reduce((a, f) => a + (f.tiers?.[p.tier] ?? 0), 0);
    const known = folders.some((f) => f.tiers?.[p.tier] !== undefined);
    const count = p.tier === "core" && ownIndex !== null ? `${ownIndex} in the Nano's own index` : known ? `${sum} records` : "count not reported";
    if (narrow) {
      S.texts.push({ x: p.left.x + 4, y: p.poly[0].y - 6, text: p.tier, anchor: "start", kind: "caps", size: 9.5 });
      return;
    }
    const lx = p.left.x - 28;
    S.lines.push({ pts: [{ x: p.left.x - 4, y: p.left.y }, { x: lx + 6, y: p.left.y }], a: 0.4, w: 0.6 });
    S.texts.push({ x: lx, y: p.left.y - 3, text: `${String(i + 1).padStart(2, "0")} · ${p.tier}`, anchor: "end", kind: "caps" });
    // The fuller line only where it fits beside the plane; else the count alone.
    const full = `${says[p.tier]} · ${count}`;
    S.texts.push({ x: lx, y: p.left.y + 12, text: S.detail === 2 && full.length * 6.4 < lx - 12 ? full : count, anchor: "end", kind: "mono", size: 10, quiet: true });
  });
  // Folder labels to the right of the stack, on leaders from each column's top.
  // 2026-11-27 (layout gap D2): a label is cut to the room between its start and the figure's right
  // edge (an ellipsis, the full name stays in the side panel's project list), and the second line
  // drops its later parts before it is cut, so nothing clips in a figure narrower than about 760 px.
  const room = (x: number, anchor: "start" | "end") => (anchor === "start" ? W - x - 10 : x - 10);
  const cut = (t: string, px: number, size: number) => {
    const n = Math.max(4, Math.floor(px / (size * 0.56)));
    return t.length <= n ? t : `${t.slice(0, n - 3)}...`;
  };
  out.columns.forEach((c, j) => {
    const ty = top + 10 + j * 34;
    const tx = narrow ? W - 12 : out.planes[0].right.x + 30;
    const anchor = narrow ? "end" : "start";
    const px = room(tx, anchor);
    S.lines.push({ pts: [{ x: c.top.x, y: c.foot[0][1].y }, { x: c.top.x, y: ty - 4 }, { x: tx + (narrow ? 6 : -6), y: ty - 4 }], a: 0.38, w: 0.6 });
    const tot = c.folder.tiers ? Object.values(c.folder.tiers).reduce((a, b) => a + (b ?? 0), 0) : null;
    S.texts.push({ x: tx, y: ty, text: cut(c.folder.own ? c.folder.label : `project ${maskName(c.folder.label)}`, px, 11.5), anchor, kind: "mono", size: 11.5 });
    const parts = c.folder.own ? [`${tot} records`] : [`${c.folder.episodes} ${c.folder.episodes === 1 ? "episode" : "episodes"}`, bytes(c.folder.bytes), ...(tot !== null ? [`${tot} records`] : [])];
    const maxN = Math.max(4, Math.floor(px / (10 * 0.56)));
    let k = parts.length;
    while (k > 1 && parts.slice(0, k).join(" · ").length > maxN) k -= 1;
    S.texts.push({ x: tx, y: ty + 13, text: cut(parts.slice(0, k).join(" · "), px, 10), anchor, kind: "mono", size: 10, quiet: true });
  });
  if (!folders.length) S.texts.push({ x: W / 2, y: top + d * 0.55, text: "no project folders yet", anchor: "middle", kind: "mono", size: 11, quiet: true });
  S.regions.push({ key: keyOf(focus), node: focus, box: out.box });
  S.mist.push({ x: W / 2, y: top + avail * 0.45, r: Math.max(w, avail) * 0.7, a: 0.4 });
  return S;
}

/** One world-model level as the figure needs it: its count of notes and, from the idle writer,
 *  its progress (done of total) when the engine reports it. */
export type WorldLevel = { level: 1 | 2 | 3 | 4; count: number | null; done?: number | null; total?: number | null };

/** The plain names the world-model figure gives the four levels. */
export const WORLD_NAMES: Record<1 | 2 | 3 | 4, string> = { 1: "L1 piece", 2: "L2 file", 3: "L3 module", 4: "L4 architecture" };

/** Memory's world model: the four levels of codebase comprehension (L1 piece, L2 file, L3 module,
 *  L4 architecture) as stepped strata, the narrowest (architecture) on top, drawn like the tiers.
 *  Each level's notes fill its plane as points (one point per note, or per group of notes when
 *  there are more than fit, said on its leader). A level the idle writer is working on carries its
 *  progress as a bright run along its front edge. The writer's queue stands at the side as a column
 *  of beads, one per queued item. With no counts the planes are drawn empty and say so. */
export function worldModelScene(focus: TNode, levels: WorldLevel[], writer: LiveWriter | null, W: number, H: number): Scene {
  const S = emptyScene(focus, W, H, "plate", W < 560 ? 0 : W < 980 ? 1 : 2);
  const narrow = W < 640;
  const labelW = narrow ? 0 : clamp(W * 0.22, 150, 320);
  const u = clamp(Math.min(W, H) / 520, 0.8, 1.6);
  const wMax = Math.max(120, Math.min(W - labelW * 2 - 40, H * 2.4));
  const top = narrow ? 30 : 40;
  const avail = H - top - (narrow ? 34 : 46);
  const d = clamp(avail * 0.2, 26, 120);
  const gap = (avail - d) / 3;
  const cx = narrow ? W / 2 : labelW + 20 + wMax / 2;
  const stores = focus.children.find((c) => c.key === "stores");
  const hier = stores?.children.find((c) => c.ids.includes("nmmem.hierarchy")) ?? stores ?? focus;
  const key = keyOf(hier);
  const sp = clamp(wMax / 70, 5, 11);
  const byLevel = new Map(levels.map((l) => [l.level, l]));
  const maxCount = Math.max(1, ...levels.map((l) => l.count ?? 0));
  // Planes top to bottom: L4, L3, L2, L1, each wider than the one above.
  const order: (1 | 2 | 3 | 4)[] = [4, 3, 2, 1];
  const planes: { level: 1 | 2 | 3 | 4; left: XY; right: XY; front: [XY, XY]; per: number }[] = [];
  order.forEach((lv, i) => {
    const f = 0.4 + 0.2 * i;
    const w = wMax * f;
    const z = clamp(0.1 + i * 0.16, 0.05, 0.8);
    const at = (uu: number, v: number): XY => ({ x: cx + uu * w * (0.72 + 0.28 * v), y: top + i * gap + v * d });
    // The plane's lattice, faint, brighter toward the front.
    const nu = Math.max(4, Math.round(w / sp));
    const nv = Math.max(3, Math.round(d / (sp * 0.8)));
    for (let a = 0; a <= nu; a++)
      for (let b = 0; b <= nv; b++) {
        const v = b / nv;
        const p = at(a / nu - 0.5, v);
        S.pts.push({ x: p.x, y: p.y, r: (0.36 + v * 0.2) * u, a: 0.16 + v * 0.18, k: key, t: "w", z: Math.min(z + 0.1, 0.7) });
      }
    // The rim, dotted, the front edge brightest.
    const edge = (u0: number, v0: number, u1: number, v1: number, a: number) => {
      const p0 = at(u0, v0);
      const p1 = at(u1, v1);
      const m = Math.max(2, Math.round(Math.hypot(p1.x - p0.x, p1.y - p0.y) / (sp * 0.42)));
      for (let q = 0; q <= m; q++) {
        const t = q / m;
        S.pts.push({ x: p0.x + (p1.x - p0.x) * t, y: p0.y + (p1.y - p0.y) * t, r: 0.5 * u, a, k: key, t: "b", z: clamp(z - 0.06, 0.02, 0.9) });
      }
    };
    edge(-0.5, 1, 0.5, 1, 0.8);
    edge(-0.5, 0, -0.5, 1, 0.4);
    edge(0.5, 0, 0.5, 1, 0.4);
    edge(-0.5, 0, 0.5, 0, 0.2);
    // The notes: one point each, or one per group when more than fit the plane.
    const L = byLevel.get(lv);
    const count = Math.max(0, Math.round(L?.count ?? 0));
    const cap = Math.max(1, Math.round(nu * nv * 1.6));
    const per = count > cap ? Math.ceil(count / cap) : 1;
    const shown = Math.ceil(count / per);
    const cols = Math.max(1, Math.ceil(Math.sqrt(shown * ((w * 0.86) / Math.max(1, d * 0.7)))));
    const rowsN = Math.max(1, Math.ceil(shown / cols));
    for (let q = 0; q < shown; q++) {
      const a = q % cols;
      const b = Math.floor(q / cols);
      const p = at(-0.43 + ((a + 0.5) / cols) * 0.86, 0.15 + ((b + 0.5) / rowsN) * 0.7);
      S.pts.push({ x: p.x, y: p.y, r: 0.75 * u, a: 0.55 + 0.4 * Math.sqrt(count / maxCount), k: key, t: "b", z: clamp(z - 0.04, 0.02, 0.9) });
    }
    // Progress of the idle writer on this level: a bright run along the front edge.
    if (L && typeof L.done === "number" && typeof L.total === "number" && L.total > 0) {
      const share = clamp(L.done / L.total, 0, 1);
      const p0 = at(-0.5, 1);
      const p1 = at(-0.5 + share, 1);
      S.lines.push({ pts: [{ x: p0.x, y: p0.y + 3 * u }, { x: p1.x, y: p1.y + 3 * u }], a: 0.95, w: 1.6, hue: "behav", k: key });
    }
    planes.push({ level: lv, left: at(-0.5, 0.55), right: at(0.5, 0.55), front: [at(-0.5, 1), at(0.5, 1)], per });
  });
  // The column joining the planes' front corners: the levels roll up into each other.
  for (const side of [0, 1] as const) S.lines.push({ pts: planes.map((p) => p.front[side]), a: 0.22, w: 0.6, dash: [1, 3], k: key });
  // Leaders and labels.
  planes.forEach((p) => {
    const L = byLevel.get(p.level);
    const def = LEVELS.find((x) => x.level === p.level);
    const count = L?.count;
    const prog = L && typeof L.done === "number" && typeof L.total === "number" ? ` · writer ${L.done} of ${L.total}` : "";
    const line = count === null || count === undefined ? "count not reported" : `${count} ${count === 1 ? "note" : "notes"}${p.per > 1 ? ` · 1 point = ${p.per}` : ""}${prog}`;
    if (narrow) {
      S.texts.push({ x: p.left.x + 4, y: p.left.y - 6, text: `${WORLD_NAMES[p.level]} · ${count ?? "-"}`, anchor: "start", kind: "caps", size: 9.5 });
      return;
    }
    const lx = p.left.x - 26;
    S.lines.push({ pts: [{ x: p.left.x - 4, y: p.left.y }, { x: lx + 6, y: p.left.y }], a: 0.4, w: 0.6 });
    S.texts.push({ x: lx, y: p.left.y - 3, text: WORLD_NAMES[p.level], anchor: "end", kind: "caps" });
    const full = def ? `${def.about} · ${line}` : line;
    S.texts.push({ x: lx, y: p.left.y + 12, text: S.detail === 2 && full.length * 6.2 < lx - 12 ? full : line, anchor: "end", kind: "mono", size: 10, quiet: true });
  });
  // The idle writer's queue: a column of beads beside the stack, one per queued item.
  if (!narrow) {
    const qx = planes[planes.length - 1].right.x + 46;
    const qTop = top + 6;
    const qBot = top + 3 * gap + d;
    S.lines.push({ pts: [{ x: qx, y: qTop + 40 }, { x: qx, y: qBot }], a: 0.3, w: 0.6 });
    const queue = writer?.queue ?? null;
    const beads = Math.min(Math.max(0, queue ?? 0), 80);
    const step = Math.min(6 * u, (qBot - qTop - 48) / Math.max(1, beads));
    for (let q = 0; q < beads; q++) S.pts.push({ x: qx, y: qBot - 4 - q * step, r: 1.3 * u, a: 0.95, k: key, t: "p", z: 0.03 });
    S.texts.push({ x: qx - 4, y: qTop + 4, text: "idle writer queue", anchor: "start", kind: "caps", size: 9.5 });
    const held = writer?.held === true ? "writer held" : writer?.held === false ? "writer idle" : null;
    const qText = queue === null ? "queue not reported" : `${queue} queued${held ? ` · ${held}` : ""}${writer?.preempted ? ` · ${writer.preempted} preempted` : ""}`;
    S.texts.push({ x: qx - 4, y: qTop + 18, text: qText, anchor: "start", kind: "mono", size: 10, quiet: true });
    if (writer?.last_event) S.texts.push({ x: qx - 4, y: qTop + 32, text: writer.last_event.length > 48 ? `${writer.last_event.slice(0, 47)}...` : writer.last_event, anchor: "start", kind: "mono", size: 9.5, quiet: true });
  }
  if (levels.every((l) => !l.count)) S.texts.push({ x: cx, y: top + 3 * gap + d + 22, text: "no world-model notes yet: the idle writer fills these levels", anchor: "middle", kind: "mono", size: 10.5, quiet: true });
  S.regions.push({ key: keyOf(focus), node: focus, box: { x: cx - wMax / 2, y: top, w: wMax, h: 3 * gap + d } });
  S.mist.push({ x: cx, y: top + avail * 0.5, r: Math.max(wMax, avail) * 0.65, a: 0.38 });
  return S;
}
