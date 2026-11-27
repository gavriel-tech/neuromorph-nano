/** The organism in plan, built from the real anatomy and drawn as a translucent body of points.
 *
 *  Back to front: the material field the tips reach into, a deep haze under the body, the eight
 *  tentacles (each a tapering band of points with its core strand inside, g0 where it leaves the
 *  ring, the drone tip with its readers), the four GCX rings, the Core Brain (its MicroModels as
 *  matrices of cells whose rows follow the body's curve, the engine hugging them, the other
 *  components as organs around them), the faint skin of the mantle over all of it, the memory
 *  pool below, the MCP port above and the cartridge rack at the side. Every anatomy edge is one
 *  fine curved line; lines that share a way gather into bundles.
 *
 *  Depth is carried by each point's `z`: near structures sharp and bright, deep ones soft. */
// 2026-11-26 (Luis Jake Gabriel III, his words: "the position that the conintual learning points to
// on the live activity uns unneccesarly angled and far from the actual label for no reason"): the
// learning label sits beside the rack on a short level leader (see the annotation layer below).
import type { Cartridge, Project, Status } from "./api";
import { clamp, emptyScene, hash01, HEADS_FALLBACK, keyOf, LAYERS_FALLBACK, type Annot, type Pt, type Scene, type XY } from "./scene";
import { drawStrata, foldersOf } from "./strata";
import { flowRanks, nodeRank, type TNode, type Tree } from "./topology";

/** Spreads chips in one column so none overlaps the next or a fixed band; keeps their order. */
function spreadAround(items: { y: number }[], gap: number, top: number, bottom: number, fixed: number[], abovePad = gap * 0.8, belowPad = gap * 0.8) {
  items.sort((a, b) => a.y - b.y);
  const clearDown = (y: number) => {
    let cur = y;
    for (let pass = 0; pass <= fixed.length; pass++) {
      let moved = false;
      for (const f of fixed) {
        if (cur > f - abovePad && cur < f + belowPad) {
          cur = f + belowPad;
          moved = true;
        }
      }
      if (!moved) break;
    }
    return cur;
  };
  const clearUp = (y: number) => {
    let cur = y;
    for (let pass = 0; pass <= fixed.length; pass++) {
      let moved = false;
      for (const f of fixed) {
        if (cur > f - abovePad && cur < f + belowPad) {
          cur = f - abovePad;
          moved = true;
        }
      }
      if (!moved) break;
    }
    return cur;
  };
  let prev = top - gap;
  for (const it of items) {
    it.y = clearDown(Math.max(it.y, prev + gap));
    prev = it.y;
  }
  const over = items.length ? items[items.length - 1].y - bottom : 0;
  if (over > 0) {
    for (let i = items.length - 1; i >= 0; i--) {
      const next = i === items.length - 1 ? bottom : items[i + 1].y - gap;
      items[i].y = clearUp(Math.min(items[i].y, next));
    }
    if (items[0].y < top) {
      let p = top - gap;
      for (const it of items) {
        it.y = Math.min(bottom, clearDown(Math.max(it.y, p + gap)));
        p = it.y;
      }
    }
  }
}

/** The longest name among the jobs the plan can give a tentacle, for label room. */
export const LONGEST_JOB = "reconciliation";

export function bodyScene(tree: Tree, focus: TNode, W: number, H: number, carts: Cartridge[], status: Status | null = null, projects: Project[] | null = null): Scene {
  const detail: Scene["detail"] = W < 560 || H < 380 ? 0 : W < 980 || H < 560 ? 1 : 2;
  const S = emptyScene(focus, W, H, "body", detail);
  const narrow = W < 640;
  // Labels in bands above and below are kept for very tall, narrow frames only; otherwise the
  // side columns hold them with level leaders.
  const tall = !narrow && H > W * 1.9;
  const kids = new Map(focus.children.map((c) => [c.key, c]));
  const hasMem = kids.has("memory");
  // Wide frames keep a label column on each side; tall frames put the labels in bands above and
  // below; narrow frames name the main parts on the figure.
  const side = narrow || tall ? 0 : detail === 2 ? clamp(W * 0.16, 188, 232) : clamp(W * 0.145, 156, 228);
  const band = tall ? clamp(H * 0.1, 64, 140) : 0;
  // Memory has its own band under the body in both normal and tall frames, clear of tentacles.
  const memBand = hasMem ? clamp(H * 0.17, 56, 180) : 0;
  const yTop = tall ? band + 12 : narrow ? 20 : 24;
  const bandBot = tall ? band : 8;
  const yBot = H - bandBot - memBand - (memBand ? 8 : 0);
  const cx = W / 2;
  const cy = (yTop + yBot) / 2;
  const padX = side ? 8 : 10;
  const padY = 10;
  const availX = Math.max(20, (W - 2 * side) / 2 - padX);
  const availY = Math.max(20, (yBot - yTop) / 2 - padY);
  const AX = availX;
  const AY = availY;
  const R_avail = Math.min(availX, availY);
  const u = clamp(R_avail / 440, 0.25, 1.6);
  const rB = Math.min(320, R_avail * 0.32);
  const rG = [rB * 1.08, rB * 1.15, rB * 1.22, rB * 1.33]; // G3, G2, G1, g0
  const skin = rB * 1.42;
  const rBase = rG[3] + 2 * u;
  const fanOut = 26 * u;
  const tipLabelPadY = Math.max(4, 20 * u + 10 - fanOut);
  const tipLabelPadX = side ? 8 : detail === 2 ? 32 : 28;
  const outerPad = fanOut + Math.max(tipLabelPadY, side ? 6 : tipLabelPadX);
  const R_tip = Math.max(rBase + 4 * u, R_avail - outerPad);
  const pts = S.pts;
  const add = (p: Pt) => pts.push(p);
  /** Where each anatomy id sits in the drawing, for its wires. */
  const anchor = new Map<string, XY>();
  const polar = (a: number, r: number): XY => ({ x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r });

  const brain = kids.get("core-brain");
  const engine = kids.get("engine");
  const gcx = kids.get("gcx");
  const pod = kids.get("neuropod");
  const subs = kids.get("substrate");
  const mem = kids.get("memory");
  const runtime = kids.get("runtime");
  const learn = kids.get("learning");
  const tentacles = pod?.children.filter((c) => c.role === "tentacle") ?? [];
  const nT = Math.max(1, tentacles.length);
  const angOf = (i: number) => -Math.PI / 2 + Math.PI / nT + (i * Math.PI * 2) / nT;
  const levels = gcx?.children ?? []; // G3, G2, G1, g0
  const lvl = (k: string) => levels.find((l) => l.key === k);

  /* ---------------- Tentacle geometry first: proportional equal-length arms inside R_tip ---------------- */
  type Arm = { t: TNode; a: number; P: XY[]; Tn: XY[]; L: number; w0: number };
  const N = 160;
  const phi = (s: number) => -0.22 * Math.sin(Math.PI * s) + 0.95 * Math.pow(Math.max(0, s - 0.58) / 0.42, 2);
  const unit: XY[] = [{ x: 0, y: 0 }];
  for (let q = 1; q <= N; q++) {
    const h = phi((q - 0.5) / N);
    const p = unit[q - 1];
    unit.push({ x: p.x + Math.cos(h) / N, y: p.y + Math.sin(h) / N });
  }
  const maxR = (len: number) => {
    let m = 0;
    for (const p of unit) m = Math.max(m, Math.hypot(rBase + len * p.x, len * p.y));
    return m;
  };
  let lo = 0;
  let hi = Math.max(1, R_tip * 1.5);
  while (maxR(hi) < R_tip) hi *= 1.5;
  for (let k = 0; k < 28; k++) {
    const mid = (lo + hi) / 2;
    if (maxR(mid) <= R_tip) lo = mid;
    else hi = mid;
  }
  const L_arm = lo;
  const w0_arm = Math.min(rB * 0.42, ((2 * Math.PI * rG[3]) / nT) * 0.44, L_arm * 0.35);

  const arms: Arm[] = tentacles.map((t, i) => {
    const a = angOf(i);
    const ca = Math.cos(a);
    const sa = Math.sin(a);
    const vertical = Math.abs(sa) > Math.abs(ca) + 1e-6;
    const sx = Math.abs(ca) < 1e-6 ? 1 : Math.sign(ca);
    const sy = Math.abs(sa) < 1e-6 ? 1 : Math.sign(sa);
    // Each arm bows gently one way, then curls away from the nearest axis at its tip, mirrored
    // so the figure stays symmetric.
    const sgn = vertical ? -sx * sy : sx * sy;
    const heading = (s: number) => a + sgn * phi(s);
    const B = polar(a, rBase);
    const P: XY[] = [B];
    for (let q = 1; q <= N; q++) {
      const h = heading((q - 0.5) / N);
      const p = P[q - 1];
      P.push({ x: p.x + Math.cos(h) * (L_arm / N), y: p.y + Math.sin(h) * (L_arm / N) });
    }
    const Tn = P.map((_, q) => {
      const h = heading(q / N);
      return { x: -Math.sin(h), y: Math.cos(h) };
    });
    return { t, a, P, Tn, L: L_arm, w0: w0_arm };
  });
  const armAt = (arm: Arm, s: number): XY => arm.P[Math.round(clamp(s, 0, 1) * (arm.P.length - 1))];
  const armN = (arm: Arm, s: number): XY => arm.Tn[Math.round(clamp(s, 0, 1) * (arm.Tn.length - 1))];
  // Tapering width; the flare where an arm joins the ring eases out over its first sixth.
  const widthAt = (arm: Arm, s: number) => arm.w0 * (0.86 * Math.pow(1 - s, 0.5) + 0.26 * Math.exp(-Math.pow(s / 0.12, 1.4))) + 2.2 * u;
  const tipXY = arms.map((arm) => armAt(arm, 1));

  /* ---------------- Material: the outer field, rows bending around the centre ---------------- */
  const nFold = projects?.length ?? 0;
  const memSub = `4 tiers${projects ? ` · ${nFold} project ${nFold === 1 ? "folder" : "folders"}` : ""}`;
  const memChipW = Math.max("Memory".length * 6.75 + 26, detail === 2 ? memSub.length * 6.5 : 0);
  const maxPoolRx = !narrow ? Math.max(40, cx - 8 - 26 - memChipW) : 300;
  const pool = memBand
    ? {
        x: cx,
        y: H - bandBot - memBand / 2 - 2,
        rx: Math.min(clamp(Math.min(availX * 0.45, memBand * 1.6), 40, 300), maxPoolRx),
        ry: Math.min(memBand * 0.36, Math.max(10, (memBand - 12) * 0.42)),
      }
    : { x: cx, y: H - bandBot - 24, rx: 40, ry: 12 };
  if (subs) {
    let i = 0;
    const de = 7.5 * u;
    for (let e = 0.42; e < 1.45; e += de / Math.min(AX, AY)) {
      const per = Math.PI * 2 * Math.sqrt(((AX * e) ** 2 + (AY * e) ** 2) / 2);
      const n = Math.floor(per / (5.2 * u));
      for (let q = 0; q < n; q++) {
        i++;
        const a = (q / n) * Math.PI * 2 + e * 3.1;
        const jx = cx + Math.cos(a) * AX * e * 1.04 + (hash01(i, 1) - 0.5) * 3 * u;
        const jy = cy + Math.sin(a) * AY * e * 1.04 + (hash01(i, 2) - 0.5) * 3 * u;
        if (jx < 2 || jy < 2 || jx > W - 2 || jy > H - 2) continue;
        if (tall && (jy < band || jy > H - band)) continue;
        if (Math.hypot(jx - cx, jy - cy) < skin * 1.05) continue;
        if (((jx - pool.x) / (pool.rx * 1.15 + 18)) ** 2 + ((jy - pool.y) / (pool.ry * (memBand ? 1.3 : 1.6) + 14)) ** 2 < 1) continue;
        const f = Math.exp(-((e - 0.95) ** 2) / 0.05);
        if (hash01(i, 3) > 0.26 + f * 0.74) continue;
        let k = keyOf(subs);
        let alpha = 0.12 + f * 0.42;
        tipXY.forEach((T, ti) => {
          if (Math.hypot(jx - T.x, jy - T.y) < 46 * u) {
            k = keyOf(tentacles[ti]);
            alpha += 0.12;
          }
        });
        const deep = hash01(i, 5) < 0.3;
        add({ x: jx, y: jy, r: 0.62 * u, a: alpha, k, t: "g", z: deep ? 0.85 : 0.5, cell: true });
      }
    }
    S.regions.push({ key: keyOf(subs), node: subs, box: { x: 0, y: 0, w: W, h: H } });
    tentacles.forEach((t, ti) => {
      const tip = t.children.find((c) => c.key === "tip");
      const T = tipXY[ti];
      S.routes.push({ from: keyOf(subs), to: tip ? keyOf(tip) : keyOf(t), pts: [{ x: T.x + (T.x - cx) * 0.08, y: T.y + (T.y - cy) * 0.08 }, T] });
    });
    if (subs.children[0]?.anatomy) anchor.set(subs.children[0].anatomy.id, { x: cx - AX * 0.98, y: cy + AY * 0.05 });
  }

  /* ---------------- Deep haze under the body: soft, so the anatomy above reads at its own depth ---------------- */
  if (brain) {
    const n = Math.round(520 * clamp(rB / 180, 0.5, 1.6));
    for (let q = 0; q < n; q++) {
      const rr = skin * 0.98 * Math.sqrt(hash01(q, 61));
      const a = hash01(q, 62) * Math.PI * 2;
      add({ x: cx + Math.cos(a) * rr + 2 * u, y: cy + Math.sin(a) * rr + 3 * u, r: 1.4 * u, a: 0.1 + 0.12 * (rr / skin), k: keyOf(brain), t: "b", z: 0.95 });
    }
  }

  /* ---------------- Neuropod: eight tentacles ---------------- */
  arms.forEach((arm, i) => {
    const t = arm.t;
    const kid = (k: string) => t.children.find((c) => c.key === k);
    const g0 = kid("g0");
    const core = kid("core");
    const tip = kid("tip");
    const tK = keyOf(t);
    const coreK = core ? keyOf(core) : tK;
    const g0K = g0 ? keyOf(g0) : tK;
    const tipK = tip ? keyOf(tip) : tK;
    const step = 2.3 * u;
    let row = 0;
    for (let d = 0; d < arm.L * 0.97; d += step, row++) {
      const s = d / arm.L;
      const p = armAt(arm, s);
      const nv = armN(arm, s);
      const w = widthAt(arm, s);
      const n = Math.max(3, Math.round(w / (1.9 * u)));
      for (let j = 0; j < n; j++) {
        const v = j / (n - 1) - 0.5;
        const rim = Math.pow(Math.abs(v * 2), 6);
        const id = i * 100003 + row * 131 + j;
        if (rim < 0.4 && hash01(id, 3) > 0.78 + 0.15 * (1 - s)) continue;
        const o = v * w + Math.sin(row * 0.35 + j) * 0.25 * u;
        // Translucent: the rim reads brighter (more body along the line of sight), lit a little
        // more on the side that faces the light.
        const lit = 0.06 * v * (arm.a > Math.PI / 2 || arm.a < -Math.PI / 2 ? -1 : 1);
        const a = (0.2 + rim * 0.52 + lit + 0.16 * Math.cos(v * Math.PI)) * (1 - s * 0.2);
        const key = s < 0.07 ? g0K : s > 0.93 ? tipK : tK;
        add({ x: p.x + nv.x * o, y: p.y + nv.y * o, r: (0.4 + rim * 0.38) * u, a: s < 0.07 ? a + 0.3 : a, k: key, t: "b", z: rim > 0.4 ? 0.12 : 0.4 });
        // The underside, deeper and soft: gives the arm its thickness.
        if (rim < 0.4 && hash01(id, 7) < 0.16) add({ x: p.x + nv.x * o * 0.8 + 2.2 * u, y: p.y + nv.y * o * 0.8 + 3.2 * u, r: 0.9 * u, a: 0.2, k: tK, t: "b", z: 0.9 });
      }
      // The tentacle core runs down the middle: two fine strands, brighter, sharp.
      if (s > 0.07 && s < 0.9) {
        const f = 1 - s * 0.4;
        for (const side of [-1, 1]) {
          const off = side * (0.9 + 0.5 * Math.sin(row * 0.22)) * u;
          add({ x: p.x + nv.x * off, y: p.y + nv.y * off, r: (0.55 + f * 0.35) * u, a: 0.55 + f * 0.42, k: coreK, t: "b", z: 0.08, cell: true });
        }
        if (row % 3 === 0) add({ x: p.x, y: p.y, r: 1.4 * u, a: 0.35, k: coreK, t: "b", z: 0.75 });
      }
    }
    // The drone tip: where the arm meets the material it fans into dendrites, five stems that
    // each fork once, no longer than the fan they replace.
    const end = armAt(arm, 1);
    const pre = armAt(arm, 0.985);
    const dl = Math.hypot(end.x - pre.x, end.y - pre.y) || 1;
    const dir = { x: (end.x - pre.x) / dl, y: (end.y - pre.y) / dl };
    const turn = (v: XY, da: number): XY => ({ x: v.x * Math.cos(da) - v.y * Math.sin(da), y: v.x * Math.sin(da) + v.y * Math.cos(da) });
    const strand = (o: XY, v: XY, len: number, f0: number, f1: number) => {
      for (let s = 0; s <= len; s += 1.8 * u) {
        const f = f0 + (f1 - f0) * (s / len);
        add({ x: o.x + v.x * s, y: o.y + v.y * s, r: (0.38 + f * 0.62) * u, a: 0.3 + f * 0.6, k: tipK, t: "b", z: 0.15 });
      }
    };
    for (let q = 0; q < 5; q++) {
      const da = (q / 4 - 0.5) * 1.15;
      const v = turn(dir, da);
      const len = (17 + 9 * (1 - Math.abs(da))) * u;
      const stem = len * 0.55;
      strand(end, v, stem, 1, 0.5);
      const fork = { x: end.x + v.x * stem, y: end.y + v.y * stem };
      for (const side of [-1, 1]) strand(fork, turn(v, side * 0.42), len - stem, 0.5, 0);
    }
    for (let q = 0; q < 24; q++) {
      const rr = 4.5 * u * Math.sqrt((q + 0.5) / 24);
      const aa = q * 2.39996;
      add({ x: end.x + Math.cos(aa) * rr, y: end.y + Math.sin(aa) * rr, r: 0.75 * u, a: 0.9, k: tipK, t: "b", z: 0.1, cell: true });
    }
    // Readers: one dendrite each along the last part of the arm, alternating sides.
    const readers = tip?.children ?? [];
    readers.forEach((rd, j) => {
      const s = 0.7 + j * (0.2 / Math.max(1, readers.length));
      const p = armAt(arm, s);
      const nv = armN(arm, s);
      const sd = j % 2 ? 1 : -1;
      const off = (widthAt(arm, s) / 2 + 3.6 * u) * sd;
      const c = { x: p.x + nv.x * off, y: p.y + nv.y * off };
      for (let q = 0; q < 12; q++) {
        const aa = (q / 12) * Math.PI * 2;
        add({ x: c.x + Math.cos(aa) * 2.8 * u, y: c.y + Math.sin(aa) * 2.8 * u, r: 0.55 * u, a: 0.8, k: keyOf(rd), t: "b", z: 0.1, cell: true });
      }
      add({ x: c.x, y: c.y, r: 0.9 * u, a: 0.95, k: keyOf(rd), t: "b", z: 0.1, cell: true });
    });
    for (const id of t.anatomyIds) {
      if (id.startsWith("nmpod.unit#t")) anchor.set(id, armAt(arm, 0.3));
      else if (id.startsWith("nmpod.unit#d")) anchor.set(id, armAt(arm, 0.92));
      else if (id.startsWith("nmapi.tentacle_core")) anchor.set(id, armAt(arm, 0.55));
    }
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const p of arm.P) {
      x0 = Math.min(x0, p.x);
      y0 = Math.min(y0, p.y);
      x1 = Math.max(x1, p.x);
      y1 = Math.max(y1, p.y);
    }
    S.regions.push({ key: tK, node: t, box: { x: x0 - 14, y: y0 - 14, w: x1 - x0 + 28, h: y1 - y0 + 28 } });
    const seg = (s0: number, s1: number, m = 18): XY[] => Array.from({ length: m + 1 }, (_, q) => armAt(arm, s0 + ((s1 - s0) * q) / m));
    if (tip && core) S.routes.push({ from: keyOf(tip), to: keyOf(core), pts: seg(0.98, 0.55) });
    const g0lv = lvl("g0");
    if (core && g0) S.routes.push({ from: keyOf(core), to: keyOf(g0), pts: seg(0.55, 0.03) });
    if (g0 && g0lv) S.routes.push({ from: keyOf(g0), to: keyOf(g0lv), pts: [armAt(arm, 0.03), polar(arm.a, rG[3])] });
    if (tip) S.routes.push({ from: tK, to: keyOf(tip), pts: seg(0, 0.98, 24), down: true });
    if (brain && tip) {
      const plan = brain.children.find((c) => c.key === "stack-plan") ?? brain;
      S.routes.push({ from: keyOf(plan), to: keyOf(tip), pts: [{ x: cx, y: cy }, ...seg(0, 0.98, 24)], down: true });
    }
  });
  if (pod) {
    S.regions.push({ key: keyOf(pod), node: pod, box: { x: cx - R_avail, y: cy - R_avail, w: R_avail * 2, h: R_avail * 2 } });
    if (pod.anatomy) anchor.set(pod.anatomy.id, polar(-Math.PI / 2, rG[3]));
  }

  /* ---------------- GCX: four rings ---------------- */
  levels.forEach((lv, li) => {
    const R = rG[Math.min(3, li)];
    const sp = clamp(R * 0.011, 1.3, 3.6);
    if (lv.key === "g0") {
      // A faint full ring, and a bright arc where each tentacle joins.
      const total = Math.floor((Math.PI * 2 * R) / (sp * 1.6));
      for (let q = 0; q < total; q++) {
        const p = polar((q / total) * Math.PI * 2, R);
        add({ x: p.x, y: p.y, r: 0.45 * u, a: 0.3, k: keyOf(lv), t: "b", z: 0.3 });
      }
      tentacles.forEach((t, i) => {
        const a0 = angOf(i);
        const span = ((Math.PI * 2) / nT) * 0.3;
        // Each arc is the tentacle's own g0_tx, which lives inside the tentacle.
        const kid = t.children.find((c) => c.key === "g0") ?? lv;
        for (const row of [-1, 0, 1])
          for (let a = a0 - span; a <= a0 + span; a += sp / R) {
            const edge = 1 - Math.abs(a - a0) / span;
            const p = polar(a, R + row * 2 * u);
            add({ x: p.x, y: p.y, r: (0.45 + edge * 0.7) * u, a: 0.4 + edge * 0.55, k: keyOf(kid), t: "b", z: 0.2, cell: true });
          }
        for (const id of kid.anatomyIds) anchor.set(id, polar(a0, R));
      });
    } else {
      const parts = lv.children.length ? lv.children : [lv];
      const n = parts.length;
      const total = Math.floor((Math.PI * 2 * R) / sp);
      for (let i = 0; i < total; i++) {
        const a = -Math.PI / 2 + (i / total) * Math.PI * 2;
        const pos = (i / total) * n;
        const pi = Math.min(n - 1, Math.floor(pos));
        const within = pos - pi;
        if (n > 1 && (within < 0.02 || within > 0.98)) continue;
        const big = lv.key === "g1" ? (i % 3 === 0 ? 1 : 0.5) : lv.key === "g2" ? 0.8 : 0.65;
        for (const dr of lv.key === "g2" ? [-0.9, 0.9] : [0]) {
          const p = polar(a, R + dr * u);
          add({ x: p.x, y: p.y, r: (0.38 + big * 0.55) * u, a: 0.38 + big * 0.42, k: keyOf(parts[pi]), t: "b", z: 0.25, cell: true });
        }
      }
    }
    if (lv.key !== "g0") for (const id of lv.anatomyIds) anchor.set(id, polar(-Math.PI / 2, R));
    S.regions.push({ key: keyOf(lv), node: lv, box: { x: cx - R, y: cy - R, w: R * 2, h: R * 2 } });
  });
  if (gcx) {
    S.regions.push({ key: keyOf(gcx), node: gcx, box: { x: cx - rG[3], y: cy - rG[3], w: rG[3] * 2, h: rG[3] * 2 } });
    const up = -Math.PI / 2;
    const chainK = ["g0", "g1", "g2", "g3"];
    for (let i = 0; i < 3; i++) {
      const A = lvl(chainK[i]);
      const B = lvl(chainK[i + 1]);
      if (A && B) S.routes.push({ from: keyOf(A), to: keyOf(B), pts: [polar(up, rG[3 - i]), polar(up, rG[2 - i])] });
    }
    const g1 = lvl("g1");
    if (g1 && brain) S.routes.push({ from: keyOf(g1), to: keyOf(brain), pts: [polar(up, rG[2]), { x: cx, y: cy }] });
    const g3 = lvl("g3");
    if (g3 && brain) S.routes.push({ from: keyOf(brain), to: keyOf(g3), pts: [{ x: cx, y: cy }, polar(Math.PI / 2, rG[0])], down: true });
  }

  /* ---------------- Core Brain ---------------- */
  const slotsNode = brain?.children.find((c) => c.key === "micromodels");
  const slots = slotsNode?.children ?? [];
  const nM = Math.max(1, slots.length);
  const rIn = rB * 0.22;
  const rOut = rB * 0.66;
  const dr = (rOut - rIn) / LAYERS_FALLBACK;
  const sectorSpan = ((Math.PI * 2) / Math.max(nM, 3)) * 0.84;
  const sectorAt = (i: number) => -Math.PI / 2 + ((Math.PI * 2) / nM) * (i + 0.5);
  // The matrices lie in a ring round the nucleus, each an arc of it. A layer's row is an arc
  // round the centre: layer 0 at the outer edge where the rings feed in, the last layer by the
  // nucleus; the heads run along the arc, so each cell is a small curved tile.
  const cellA = (i: number, h: number) => sectorAt(i) + ((h + 0.5) / HEADS_FALLBACK - 0.5) * sectorSpan;
  const cellXY = (i: number, l: number, h: number): XY => polar(cellA(i, h), rOut - (l + 0.5) * dr);
  slots.forEach((slot, i) => {
    const card = typeof slot.anatomy?.config.card === "string" ? (slot.anatomy.config.card as string) : undefined;
    for (let l = 0; l < LAYERS_FALLBACK; l++)
      for (let h = 0; h < HEADS_FALLBACK; h++) {
        const p = cellXY(i, l, h);
        const r = rOut - (l + 0.5) * dr;
        add({ x: p.x, y: p.y, r: dr * 0.3, ex: ((r * sectorSpan) / HEADS_FALLBACK) * 0.44, rot: cellA(i, h) + Math.PI / 2, a: 0.4, k: keyOf(slot), t: "m", model: card, layer: l, head: h, z: 0.15 });
      }
    // Corner ticks at the four corners of the matrix, like a plate's registration marks.
    for (const [l, h] of [
      [-1.2, -1],
      [-1.2, HEADS_FALLBACK],
      [LAYERS_FALLBACK + 0.2, -1],
      [LAYERS_FALLBACK + 0.2, HEADS_FALLBACK],
    ]) {
      const p = cellXY(i, l, h);
      add({ x: p.x, y: p.y, r: 0.7 * u, a: 0.7, k: keyOf(slot), t: "p", z: 0.1 });
    }
    const mid = cellXY(i, 0, HEADS_FALLBACK / 2 - 0.5);
    // A beacon on the matrix only when the engine reports this model held on the graphics card.
    const held = status?.loaded?.find((m) => m.id === card && /cuda|gpu|graphics/i.test(m.device ?? ""));
    if (held) {
      const b = polar(sectorAt(i), rOut + dr * 4.5);
      add({ x: b.x, y: b.y, r: 1.9 * u, a: 1, k: keyOf(slot), t: "p", z: 0.02 });
      for (let q = 0; q < 10; q++) {
        const aa = (q / 10) * Math.PI * 2;
        add({ x: b.x + Math.cos(aa) * 4.5 * u, y: b.y + Math.sin(aa) * 4.5 * u, r: 0.6 * u, a: 0.8, k: keyOf(slot), t: "p", z: 0.05 });
      }
    }
    for (const id of slot.anatomy ? [slot.anatomy.id] : []) anchor.set(id, mid);
    let k = 0;
    for (const c of carts) {
      if (!c.loaded || (c.model ?? c.base) !== card) continue;
      const [a, b] = c.layers ?? [0, LAYERS_FALLBACK - 1];
      const pa = cellXY(i, a, HEADS_FALLBACK + 1 + k * 1.4);
      const pb = cellXY(i, b, HEADS_FALLBACK + 1 + k * 1.4);
      S.docks.push({ cart: c.id, x: (pa.x + pb.x) / 2 - u, y: Math.min(pa.y, pb.y), w: 2 * u, h: Math.max(3, Math.abs(pb.y - pa.y)) });
      k++;
    }
  });
  if (engine) {
    // The engine hugs the matrices: an arc of points under each, one run per engine part.
    const parts = engine.children.filter((c) => c.key !== "micromodels");
    const runs = parts.length ? parts : [engine];
    const R = rOut + dr * 2.2;
    let q = 0;
    for (let i = 0; i < nM; i++) {
      const c = sectorAt(i);
      for (let a = c - sectorSpan / 2; a <= c + sectorSpan / 2; a += (2 * u) / R, q++) {
        const part = runs[q % runs.length];
        const p = polar(a, R);
        add({ x: p.x, y: p.y, r: 0.6 * u, a: 0.7, k: keyOf(part), t: "b", z: 0.2, cell: true });
      }
    }
  }
  let comps: TNode[] = [];
  if (brain) {
    const ranks = flowRanks(tree);
    comps = brain.children
      .filter((c) => c.key !== "micromodels")
      .map((c, i) => ({ c, i, r: nodeRank(c, ranks) }))
      .sort((a, b) => a.r - b.r || a.i - b.i)
      .map((x) => x.c);
    // The nucleus: where the brain's work meets; fine and even, so the matrices round it stay legible.
    for (let q = 0; q < 110; q++) {
      const rr = rB * 0.13 * Math.sqrt((q + 0.5) / 110);
      const a = q * 2.39996;
      add({ x: cx + Math.cos(a) * rr, y: cy + Math.sin(a) * rr, r: 0.42 * u, a: 0.34, k: keyOf(brain), t: "b", z: 0.3 });
    }
    // Organs: each component a soft cluster around the matrices, in flow order round the brain.
    const organR = rB * 0.87;
    const seedOf = (i: number): XY => polar(-Math.PI / 2 + ((i + 0.5) / Math.max(1, comps.length)) * Math.PI * 2 + Math.PI / Math.max(4, comps.length) / 2, organR);
    const arc = (Math.PI * 2 * organR) / Math.max(1, comps.length);
    const cluster = (o: XY, rc: number, n: number, k: string, z: number, bright: number) => {
      for (let q = 0; q < n; q++) {
        const rr = rc * Math.sqrt((q + 0.5) / n);
        const a = q * 2.39996;
        const f = 1 - rr / rc;
        add({ x: o.x + Math.cos(a) * rr, y: o.y + Math.sin(a) * rr, r: (0.42 + f * 0.6) * u, a: (0.4 + f * 0.45) * bright, k, t: "b", z, cell: true });
      }
    };
    comps.forEach((c, i) => {
      const o = seedOf(i);
      const kidsN = Math.min(10, c.children.length);
      if (kidsN) {
        // A component that holds sub-components: a dotted membrane with each sub-component a
        // small cluster inside it, so what sits inside what reads at a glance.
        const rc = Math.min(rB * 0.16, arc * 0.44);
        const ring = Math.max(18, Math.round((Math.PI * 2 * rc) / (2.1 * u)));
        for (let q = 0; q < ring; q++) {
          const a = (q / ring) * Math.PI * 2;
          add({ x: o.x + Math.cos(a) * rc, y: o.y + Math.sin(a) * rc, r: 0.5 * u, a: 0.7, k: keyOf(c), t: "b", z: 0.12 });
        }
        const sub = c.children.slice(0, kidsN);
        const rs = Math.min(rc * 0.34, (rc * 1.6) / Math.sqrt(sub.length) / 2);
        sub.forEach((sc, j) => {
          const rr = sub.length === 1 ? 0 : (rc - rs - 2 * u) * Math.sqrt((j + 0.5) / sub.length);
          const a = j * 2.39996;
          const so = { x: o.x + Math.cos(a) * rr, y: o.y + Math.sin(a) * rr };
          cluster(so, rs, Math.round(22 * clamp(rB / 180, 0.6, 1.4)), keyOf(sc), 0.22, 1);
          for (const id of sc.anatomyIds) anchor.set(id, so);
        });
      } else {
        const rc = Math.min(rB * 0.075, arc * 0.38, rB * 0.11);
        cluster(o, rc, Math.round(50 * clamp(rB / 180, 0.6, 1.4)), keyOf(c), 0.22, 1);
      }
      for (const id of c.anatomyIds) if (!anchor.has(id)) anchor.set(id, o);
      const rBox = kidsN ? Math.min(rB * 0.16, arc * 0.44) : Math.min(rB * 0.075, arc * 0.38, rB * 0.11);
      S.regions.push({ key: keyOf(c), node: c, box: { x: o.x - rBox, y: o.y - rBox, w: rBox * 2, h: rBox * 2 } });
      S.routes.push({ from: keyOf(c), to: keyOf(brain), pts: [o, { x: cx, y: cy }] });
    });
    // The skin of the mantle: a faint film of points in rows that bend around the centre, its rim
    // brighter, swelling where each tentacle joins. It is see-through: everything inside shows.
    const sp = clamp(rB * 0.026, 2, 5.2);
    let idx = 0;
    const rim = (a: number) => skin * (1 + 0.05 * Math.cos(nT * (a - angOf(0))));
    for (let r = sp; r < skin * 1.06; r += sp * 0.86) {
      const n = Math.max(6, Math.floor((Math.PI * 2 * r) / sp));
      for (let q = 0; q < n; q++) {
        const a = (q / n) * Math.PI * 2 + r * 0.019;
        const R = rim(a);
        if (r > R) continue;
        idx++;
        const depth = r / R;
        const edge = Math.pow(depth, 10);
        if (edge < 0.2 && hash01(idx, 9) > 0.55) continue;
        add({ x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r, r: (0.32 + edge * 0.5) * u, a: 0.07 + edge * 0.6, k: keyOf(brain), t: "b", z: edge > 0.3 ? 0.05 : 0.1 });
      }
    }
    S.mist.push({ x: cx, y: cy, r: skin * 2.2, a: 0.6 });
    S.mist.push({ x: cx, y: cy, r: rB * 1.1, a: 0.16 });
    S.regions.push({ key: keyOf(brain), node: brain, box: { x: cx - rB, y: cy - rB, w: rB * 2, h: rB * 2 } });
    if (engine) S.routes.push({ from: keyOf(engine), to: keyOf(brain), pts: [polar(Math.PI / 2, rOut + dr * 2), { x: cx, y: cy }] });
    slots.forEach((slot, i) => {
      const p = polar(sectorAt(i), (rIn + rOut) / 2);
      S.routes.push({ from: keyOf(slot), to: keyOf(brain), pts: [p, { x: cx, y: cy }] });
      S.routes.push({ from: keyOf(brain), to: keyOf(slot), pts: [{ x: cx, y: cy }, p], down: true });
    });
  }
  const bandBox = { x: cx - rOut, y: cy - rOut, w: rOut * 2, h: rOut * 2 };

  /* ---------------- Memory: its four tiers as strata below the body, project folders through them ---------------- */
  if (mem) {
    const stores = mem.children.find((c) => c.key === "stores");
    const tiersNode = stores?.children.find((c) => c.ids.includes("nmmem.tiers"));
    const tierKey = keyOf(tiersNode ?? stores ?? mem);
    const out = drawStrata(S, {
      cx: pool.x,
      top: pool.y - pool.ry,
      w: pool.rx * 2,
      d: pool.ry * 0.62,
      gap: (pool.ry * 2 - pool.ry * 0.62) / 3,
      sp: clamp(pool.rx / 26, 2.6, 6),
      u,
      tierKey: () => tierKey,
      folderKey: keyOf(stores ?? mem),
      folders: foldersOf(projects).slice(0, 6),
    });
    const joint = { x: pool.x, y: out.box.y - 4 };
    if (mem.anatomy) anchor.set(mem.anatomy.id, joint);
    for (const g of mem.children) for (const id of g.anatomyIds) anchor.set(id, joint);
    // The one link from the body down to its memory: a fine dotted line.
    if (memBand) for (let y = cy + skin + 6 * u; y < joint.y - 2; y += 4 * u) add({ x: cx, y, r: 0.55 * u, a: 0.5, k: keyOf(mem), t: "p", z: 0.2 });
    S.regions.push({ key: keyOf(mem), node: mem, box: { x: out.box.x - 6, y: out.box.y - 6, w: out.box.w + 12, h: out.box.h + 12 } });
    S.mist.push({ x: pool.x, y: pool.y, r: pool.rx * 1.3, a: 0.22 });
  }

  /* ---------------- MCP and runtime: the port above ---------------- */
  const port = { x: cx, y: Math.max(yTop + 10 * u, cy - R_tip - 14 * u) };
  if (runtime) {
    for (let q = 0; q < 18; q++) {
      const a = (q / 18) * Math.PI * 2;
      add({ x: port.x + Math.cos(a) * 7 * u, y: port.y + Math.sin(a) * 7 * u, r: 0.75 * u, a: 0.8, k: keyOf(runtime), t: "p", z: 0.1 });
    }
    add({ x: port.x, y: port.y, r: 1.6 * u, a: 1, k: keyOf(runtime), t: "p", z: 0.05 });
    for (let y = port.y + 12 * u; y < cy - skin - 2 * u; y += 4 * u) add({ x: cx, y, r: 0.6 * u, a: 0.55, k: keyOf(runtime), t: "p", z: 0.2 });
    for (let y = port.y - 12 * u, a = 0.4; y > yTop + 3 && a > 0.05; y -= 6 * u, a *= 0.88) add({ x: cx, y, r: 0.55 * u, a, k: keyOf(runtime), t: "p", z: 0.3 });
    if (focus.anatomy) anchor.set(focus.anatomy.id, port);
    S.regions.push({ key: keyOf(runtime), node: runtime, box: { x: port.x - 14, y: port.y - 14, w: 28, h: 28 } });
    if (brain) {
      S.routes.push({ from: keyOf(runtime), to: keyOf(brain), pts: [port, { x: cx, y: cy }] });
      S.routes.push({ from: keyOf(brain), to: keyOf(runtime), pts: [{ x: cx, y: cy }, port], down: true });
    }
  }

  /* ---------------- Continuous learning: the cartridge rack ---------------- */
  const cartsNode = learn?.children.find((c) => c.key === "cartridges");
  // 2026-11-26: the base and hardware cartridges are their own group under learning (they are not
  // learned); the rack still shows every cartridge plugged into the MicroModels, learned or not.
  const baseNode = learn?.children.find((c) => c.key === "base-cartridges");
  const rackList = [...(cartsNode?.children ?? []), ...(baseNode?.children ?? [])];
  const rackCount = Math.max(1, Math.min(8, rackList.length));
  const rackHW = ((rackCount - 1) / 2) * 7 * u + 4;
  const rackHH = 17 * u + 4;
  const rackBoxHW = Math.max(22, rackHW + 2);
  const rackBoxHH = rackHH + 2;
  const maxTipAnnotSpan = (detail === 2 ? `T${nT} · ${LONGEST_JOB}`.length : `T${nT}`.length) * 6.8;
  const rackMinX = R_tip + fanOut + 22 * u + maxTipAnnotSpan + rackBoxHW + 8;
  const rack =
    availX >= rackMinX + rackBoxHW + 4
      ? { x: cx + Math.min(availX - rackBoxHW - 4, rackMinX), y: cy }
      : { x: cx + availX - rackBoxHW - 2, y: cy + availY - rackBoxHH - 2 };
  if (learn) {
    const h = 34 * u;
    for (let i = 0; i < rackCount; i++) {
      const x = rack.x + (i - (rackCount - 1) / 2) * 7 * u;
      const c = rackList[i];
      for (let y = -h / 2; y <= h / 2; y += 2.2 * u) add({ x, y: rack.y + y, r: 0.55 * u, a: c ? 0.6 : 0.3, k: c ? keyOf(c) : keyOf(cartsNode ?? learn), t: "c", cart: c?.cartridge?.id, z: 0.2 });
    }
    S.rack = rack;
    S.regions.push({ key: keyOf(learn), node: learn, box: { x: rack.x - rackBoxHW, y: rack.y - rackBoxHH, w: rackBoxHW * 2, h: rackBoxHH * 2 } });
    if (engine) S.routes.push({ from: keyOf(learn), to: keyOf(engine), pts: [rack, polar(0, rOut + dr * 2)], down: true });
  }
  if (engine) S.regions.push({ key: keyOf(engine), node: engine, box: bandBox });
  if (slotsNode) S.regions.push({ key: keyOf(slotsNode), node: slotsNode, box: bandBox });

  /* ---------------- Wires: one fine curved line per anatomy edge ---------------- */
  const keyFor = (id: string): string | null => {
    const list = tree.index.get(id);
    if (!list?.length) return null;
    return keyOf(list.reduce((a, b) => (b.path.length > a.path.length ? b : a)));
  };
  const g1ids = new Set(lvl("g1")?.anatomyIds ?? []);
  const onG1 = (other: XY | undefined): XY | undefined => (other ? polar(Math.atan2(other.y - cy, other.x - cx), rG[2]) : undefined);
  tree.edges.forEach((e, q) => {
    const A0 = anchor.get(e.from);
    const B0 = anchor.get(e.to);
    const A = g1ids.has(e.from) ? onG1(B0) : A0;
    const B = g1ids.has(e.to) ? onG1(A0) : B0;
    const kf = keyFor(e.from);
    const kt = keyFor(e.to);
    if (!A || !B || !kf || !kt) return;
    // Pull the middle toward the centre so lines that share a way gather, then let them wander.
    const mx = (A.x + B.x) / 2;
    const my = (A.y + B.y) / 2;
    const pull = Math.hypot(A.x - B.x, A.y - B.y) > rB ? 0.32 : 0.15;
    const len = Math.hypot(B.x - A.x, B.y - A.y) || 1;
    const nx = -(B.y - A.y) / len;
    const ny = (B.x - A.x) / len;
    const wob = (hash01(q, 31) - 0.5) * Math.min(26 * u, len * 0.18);
    const C = { x: mx + (cx - mx) * pull + nx * wob, y: my + (cy - my) * pull + ny * wob };
    const N = 24;
    const out: XY[] = [];
    for (let s = 0; s <= N; s++) {
      const v = s / N;
      const m = 1 - v;
      const wander = Math.sin(v * Math.PI * 3 + q) * 1.6 * u * Math.sin(v * Math.PI);
      out.push({ x: m * m * A.x + 2 * m * v * C.x + v * v * B.x + nx * wander, y: m * m * A.y + 2 * m * v * C.y + v * v * B.y + ny * wander });
    }
    const fromBrain = brain ? kf.startsWith(keyOf(brain) + "/") && !kt.startsWith(keyOf(brain) + "/") : false;
    S.wires.push({ from: kf, to: kt, pts: out, mode: e.mode.kind, down: fromBrain });
    S.routes.push({ from: kf, to: kt, pts: out, down: fromBrain });
  });

  /* ---------------- Stippled mist around the body ---------------- */
  const mist = Math.round(((W * H) / (90 * u * u)) * 0.16);
  for (let q = 0; q < mist; q++) {
    const a = hash01(q, 41) * Math.PI * 2;
    const rr = Math.pow(hash01(q, 42), 0.7) * 1.3;
    const x = cx + Math.cos(a) * AX * rr;
    const y = cy + Math.sin(a) * AY * rr;
    if (x < 0 || y < 0 || x > W || y > H) continue;
    const f = Math.exp(-((rr - 0.6) ** 2) / 0.2);
    if (hash01(q, 43) > f) continue;
    add({ x, y, r: 0.42 * u, a: 0.06 + f * 0.1, k: "", t: "g", z: hash01(q, 44) < 0.25 ? 0.95 : 0.6 });
  }
  S.mist.push({ x: cx, y: cy, r: Math.max(AX, AY) * 1.25, a: 0.42 });

  /* ---------------- Annotation layer ---------------- */
  const L: Annot[] = [];
  const R: Annot[] = [];
  const xl = 46;
  const xr = W - 22;
  // Where a chip's leader meets the figure once the chip's row is known: on a ring or edge at
  // the chip's own height where the part allows it, so leaders run level and never cross.
  const aims = new Map<string, (y: number) => XY>();
  const onCircle = (R: number, sideSign: number) => (y: number): XY => {
    const dy = clamp(y - cy, -R * 0.92, R * 0.92);
    return { x: cx + sideSign * Math.sqrt(R * R - dy * dy), y: cy + dy };
  };
  const chip = (list: Annot[], n: TNode | undefined, label: string, m: XY, sub?: string, aim?: (y: number) => XY) => {
    if (!n) return;
    list.push({ key: keyOf(n), label, sub: detail === 2 ? sub : undefined, mx: m.x, my: m.y, x: list === L ? xl : xr, y: m.y, side: list === L ? "l" : "r", kind: "chip" });
    if (aim) aims.set(keyOf(n), aim);
  };
  chip(L, brain, "Core Brain", polar(Math.PI + 0.62, rB * 0.82), brain?.sub, onCircle(skin * 0.97, -1));
  chip(L, slotsNode, "MicroModels", cellXY(nM > 2 ? 2 : 0, 6, 2), `${slots.length} regions, 28 layers each`, onCircle((rIn + rOut) / 2, -1));
  chip(L, engine, "Engine", polar(Math.PI * 0.75 + sectorSpan * 0.3, rOut + dr * 2.2), "runs the MicroModels", onCircle(rOut + dr * 2.2, -1));
  chip(L, subs, "Material", { x: cx - AX * 0.97, y: cy + AY * 0.08 }, "the context substrate, in RAM", (y) => ({ x: cx - AX * 0.95, y }));
  // In its own band, Memory's label sits with it, beside the stack on a short leader.
  if (mem && memBand && !narrow) {
    const sub = detail === 2 ? memSub : undefined;
    const wide = Math.max("Memory".length * 6.75 + 26, (sub?.length ?? 0) * 6.5);
    S.annots.push({ key: keyOf(mem), label: "Memory", sub, mx: pool.x - pool.rx * 0.84, my: pool.y, x: Math.max(8, pool.x - pool.rx - 26 - wide), y: pool.y - 6, side: "l", kind: "chip" });
  }
  else chip(L, mem, "Memory", { x: pool.x - pool.rx * 0.85, y: pool.y }, memSub, (y) => ({ x: pool.x - pool.rx * 0.88, y: clamp(y, pool.y - pool.ry * 0.9, pool.y + pool.ry * 0.9) }));
  chip(R, runtime, narrow ? "MCP" : "MCP and runtime", { x: port.x + 7 * u, y: port.y }, runtime?.sub);
  levels.forEach((lv, li) => chip(R, lv, lv.label, polar([-0.16, -0.08, 0.08, 0.16][li] ?? 0, rG[Math.min(3, li)]), lv.sub, onCircle(rG[Math.min(3, li)], 1)));
  const onArm = (arm: Arm | undefined) => (y: number): XY => {
    if (!arm) return polar(0, rG[3]);
    let best = arm.P[0];
    for (let q = Math.floor(arm.P.length * 0.15); q < arm.P.length * 0.85; q++) if (Math.abs(arm.P[q].y - y) < Math.abs(best.y - y)) best = arm.P[q];
    return best;
  };
  const podArm = arms[1] ?? arms[0];
  chip(R, pod, "Neuropod", podArm ? armAt(podArm, 0.45) : polar(0, rG[3]), pod?.sub, onArm(podArm));
  chip(R, learn, narrow ? "Learning" : "Continuous learning", { x: rack.x + 8 * u, y: rack.y + 14 * u }, carts.length ? `${carts.filter((c) => c.loaded).length} of ${carts.length} cartridges loaded` : "no cartridges loaded", (y) => ({ x: rack.x + 6 * u, y: clamp(y, rack.y - 16 * u, rack.y + 16 * u) }));
  // 2026-11-26 (Luis Jake Gabriel III: on Live Activity continual learning sat between g0 and G1;
  // then: its pointer was "unneccesarly angled and far from the actual label"): the learning chip keeps
  // its own row beside the rack, and the ganglion level chips rank above it instead. Their leaders
  // land on their rings at the chip's own height, so they stay level wherever the chips sit.
  const learnChip = learn ? R.find((a) => a.key === keyOf(learn)) : undefined;
  if (learnChip) {
    const levelChips = levels.map((lv) => R.find((a) => a.key === keyOf(lv))).filter((a): a is Annot => !!a);
    levelChips.forEach((a, i) => {
      a.y = Math.min(a.y, learnChip.y - (levelChips.length - i) * 0.5);
    });
  }

  // Tentacle names sit just past their tips.
  const tipAnnots: Annot[] = arms.map((arm, i) => {
    const T = tipXY[i];
    const dx = T.x - cx;
    const dy = T.y - cy;
    const dl = Math.hypot(dx, dy) || 1;
    const side: Annot["side"] = dx > 6 ? "l" : dx < -6 ? "r" : "c";
    // The number names the tentacle. At rest it claims no job; while a call's step on it is lit
    // and in its afterglow, the figure adds the job or lens the step reported (Topology relabel).
    const label = `T${i + 1}`;
    // Room for the longest job a call can give it, so a live job label never leaves the frame.
    const tw = (detail === 2 ? Math.max(label.length, `T${i + 1} · ${LONGEST_JOB}`.length) : label.length) * 6.8;
    let x = T.x + (dx / dl) * 22 * u;
    const y = clamp(T.y + (dy / dl) * 20 * u + 4, yTop + 12, yBot - 6);
    if (y - 11 <= port.y + 15 && y + 4 >= port.y - 15) {
      if (side === "l" && x < cx + 18) x = cx + 18;
      else if (side === "r" && x > cx - 18) x = cx - 18;
    }
    // A name that would run off the frame sits above its tip instead.
    if ((side === "l" && x + tw > W - 8) || (side === "r" && x - tw < 8)) {
      return { key: keyOf(arm.t), label, mx: T.x, my: T.y, x: clamp(T.x, tw / 2 + 8, W - tw / 2 - 8), y: clamp(T.y - 12 * u - 6, yTop + 12, yBot - 6), side: "c", kind: "minor" };
    }
    return { key: keyOf(arm.t), label, mx: T.x, my: T.y, x: side === "c" ? clamp(x, tw / 2 + 8, W - tw / 2 - 8) : clamp(x, 24, W - 24), y, side, kind: "minor" };
  });

  if (tall) {
    // Tall frames: the labels in two bands, above and below the figure, in columns.
    const place = (list: Annot[], y0: number, y1: number) => {
      const cols = Math.max(1, Math.min(list.length, Math.floor((W - 40) / 200)));
      const rows = Math.ceil(list.length / cols);
      const cw = (W - 40) / cols;
      list.sort((a, b) => a.mx - b.mx);
      list.forEach((a, i) => {
        const c = i % cols;
        const r = Math.floor(i / cols);
        a.side = "l";
        a.x = 20 + c * cw + 10;
        a.y = y0 + ((r + 0.5) / rows) * (y1 - y0);
      });
    };
    place(L, 26, band - 4);
    place(R, H - band + 8, H - 26);
    S.annots.push(...L, ...R);
  } else if (!narrow) {
    const gap = detail === 2 ? 42 : 30;
    // Tentacle names that reach into a label column are fixed bands the chips keep clear of.
    const span = (a: Annot) => {
      const tw = (detail === 2 ? Math.max(a.label.length, `T0 · ${LONGEST_JOB}`.length) : a.label.length) * 6.8;
      return a.side === "l" ? [a.x, a.x + tw] : a.side === "r" ? [a.x - tw, a.x] : [a.x - tw / 2, a.x + tw / 2];
    };
    const chipW = (a: Annot) => Math.max(a.label.length * 6.75 + 26, a.sub ? a.sub.length * 6.5 : 0);
    const lMaxX = L.length ? xl + Math.max(...L.map(chipW)) : xl + 120;
    const rMinX = R.length ? xr - Math.max(...R.map(chipW)) : xr - 150;
    const fixedL = tipAnnots.filter((a) => span(a)[0] < lMaxX + 12).map((a) => a.y);
    const fixedR = tipAnnots.filter((a) => span(a)[1] > rMinX - 12).map((a) => a.y);
    const abovePad = detail === 2 ? 40 : 20;
    const belowPad = 18;
    const spreadBot = detail === 2 ? yBot - 28 : yBot - 12;
    spreadAround(L, gap, 34, spreadBot, fixedL, abovePad, belowPad);
    spreadAround(R, gap, 34, spreadBot, fixedR, abovePad, belowPad);
    for (const a of [...L, ...R]) {
      const aim = aims.get(a.key);
      if (!aim) continue;
      const m = aim(a.y);
      a.mx = m.x;
      a.my = m.y;
    }
    // 2026-11-26 (Luis Jake Gabriel III: the continual learning pointer was "unneccesarly angled and
    // far from the actual label for no reason"): the learning chip leaves the far column and sits
    // just right of the rack, its leader a short level line from the rack's edge to the chip, when
    // that place is clear of every other label; else it keeps its column place.
    if (learnChip) {
      const sw = (a: Annot) => Math.max(a.label.length * 6.75 + 26, a.sub ? a.sub.length * 6.5 : 0);
      const tipW = (a: Annot) => (detail === 2 && a.key.startsWith("neuropod/t") ? Math.max(a.label.length, `T0 · ${LONGEST_JOB}`.length) : a.label.length) * 6.8;
      const boxOf = (a: Annot) => {
        if (a.kind === "chip") {
          const w = sw(a);
          const x = a.side === "l" ? a.x : a.side === "r" ? a.x - w : a.x - w / 2;
          return { x, y: a.y - 12, w, h: (a.sub ? 30 : 12) + 12 };
        }
        const w = tipW(a);
        const x = a.side === "l" ? a.x : a.side === "r" ? a.x - w : a.x - w / 2;
        return { x, y: a.y - 13, w, h: 19 };
      };
      const hit = (p: { x: number; y: number; w: number; h: number }, q: { x: number; y: number; w: number; h: number }) => p.x < q.x + q.w && p.x + p.w > q.x && p.y < q.y + q.h && p.y + p.h > q.y;
      const rackBox = { x: rack.x - rackBoxHW, y: rack.y - rackBoxHH, w: rackBoxHW * 2, h: rackBoxHH * 2 };
      const others = [...L, ...R, ...tipAnnots].filter((o) => o !== learnChip).map(boxOf);
      const w = sw(learnChip);
      const cw = learnChip.label.length * 6.75 + 26;
      const fits = (a: Annot) => {
        const b = boxOf(a);
        return b.x >= 4 && b.x + b.w <= W - 4 && b.y >= 4 && b.y + b.h <= H - 4 && !hit(b, rackBox) && !others.some((o) => hit(b, o));
      };
      // Candidates, nearest first: beside the rack at a height inside it (a level leader from its
      // right edge), then under it, then over it (a short upright leader from its corner).
      const cands: { x: number; y: number; mx: number; my: number }[] = [];
      const y0 = clamp(learnChip.y, rack.y - rackHH + 6, rack.y + rackHH - 6);
      const xBeside = Math.min(xr, rackBox.x + rackBox.w + 14 + w);
      for (let d = 0; d <= rackHH * 2; d += 4) for (const y of d ? [y0 - d, y0 + d] : [y0]) if (y >= rack.y - rackHH + 6 && y <= rack.y + rackHH - 6) cands.push({ x: xBeside, y, mx: rack.x + rackHW, my: y });
      const left = rackBox.x + 2;
      const xUnder = left + cw;
      cands.push({ x: xUnder, y: rackBox.y + rackBox.h + 18, mx: left, my: rackBox.y + rackBox.h });
      cands.push({ x: xUnder, y: rackBox.y - (detail === 2 && learnChip.sub ? 46 : 18), mx: left, my: rackBox.y });
      const pick = cands.find((c) => fits({ ...learnChip, x: c.x, y: c.y }));
      if (pick) Object.assign(learnChip, pick);
    }
    S.annots.push(...L, ...R);
  } else {
    const keep = [brain, gcx, pod, mem].filter(Boolean) as TNode[];
    for (const n of keep) {
      const r = S.regions.find((x) => x.key === keyOf(n));
      if (r) S.annots.push({ key: keyOf(n), label: n.short ?? n.label, mx: r.box.x + r.box.w / 2, my: r.box.y, x: r.box.x + r.box.w / 2, y: Math.max(12, r.box.y - 6), side: "c", kind: "minor" });
    }
  }
  S.annots.push(...tipAnnots);
  return S;
}
