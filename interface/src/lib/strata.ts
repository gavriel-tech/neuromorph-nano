/** Memory drawn as strata: its four tiers (core, active, archive, temporary) as planes one above
 *  another, seen from above at an angle. Each plane is an even lattice of points with a dotted
 *  rim, its front edge brighter than its back; deeper planes sit lower, a little smaller, softer
 *  and dimmer. Each project's folder is a column through every plane (a project folder holds
 *  every tier): a nested footprint on each plane, sized by the folder's bytes, joined by fine
 *  lines, with one bright bead per verified episode on the column's axis. Where the engine
 *  reports how many records a tier of a folder holds, that many points fill its footprint on
 *  that plane; otherwise the footprint is drawn empty. Nothing here varies without data. */
import type { Project } from "./api";
import { clamp, type Scene, type XY } from "./scene";

export const TIERS = ["core", "active", "archive", "temporary"] as const;
export type Tier = (typeof TIERS)[number];

/** A project folder as the memory figure needs it. `tiers` is optional: records per tier. */
export type Folder = { key: string; label: string; episodes: number; bytes: number; tiers?: Partial<Record<Tier, number>>; own?: boolean; overlap?: number };

export function foldersOf(projects: Project[] | null | undefined): Folder[] {
  return (projects ?? []).map((p) => {
    const q = p as Project & { tiers?: Partial<Record<Tier, number>>; tiers_overlap?: number };
    const t = q.tiers;
    return { key: p.key, label: p.key, episodes: p.episodes, bytes: p.bytes, tiers: t && typeof t === "object" ? t : undefined, overlap: typeof q.tiers_overlap === "number" ? q.tiers_overlap : undefined };
  });
}

export type StrataOpts = {
  /** Centre of the stack in x, and the top of the first plane's back edge. */
  cx: number;
  top: number;
  /** Width of the nearest plane, its depth on screen, and the drop from one plane to the next. */
  w: number;
  d: number;
  gap: number;
  /** Point spacing of the plane lattice. */
  sp: number;
  /** Size unit for points. */
  u: number;
  /** Key for each plane's points and for the folders' points. */
  tierKey: (t: Tier) => string;
  folderKey: string;
  folders: Folder[];
};

export type StrataOut = {
  /** Each plane's outline (back left, back right, front right, front left) and its label anchor. */
  planes: { tier: Tier; poly: XY[]; left: XY; right: XY }[];
  /** Each folder's column: its axis top (on the first plane) and bottom (on the last). */
  columns: { folder: Folder; top: XY; bottom: XY; foot: XY[][] }[];
  box: { x: number; y: number; w: number; h: number };
};

export function drawStrata(S: Scene, o: StrataOpts): StrataOut {
  const n = TIERS.length;
  const shrink = 0.07;
  /** Screen position of (u, v) on plane i: u across (-0.5 to 0.5), v from back (0) to front (1). */
  const at = (i: number, u: number, v: number): XY => {
    // Planes keep one width so a folder's column stands upright; deeper planes are foreshortened a little.
    const s = 1 - i * shrink;
    return { x: o.cx + u * o.w * (0.7 + 0.3 * v), y: o.top + i * o.gap + v * o.d * s };
  };
  const planes: StrataOut["planes"] = [];
  TIERS.forEach((tier, i) => {
    const k = o.tierKey(tier);
    const s = 1 - i * shrink;
    const z = clamp(0.12 + i * 0.2, 0.05, 0.9);
    const dim = 1 - i * 0.12;
    // Surface: an even lattice, a touch brighter toward the front edge so the plane reads as lying flat.
    const nu = Math.max(6, Math.round(o.w / o.sp));
    const nv = Math.max(3, Math.round((o.d * s) / (o.sp * 0.8)));
    for (let a = 0; a <= nu; a++)
      for (let b = 0; b <= nv; b++) {
        const v = b / nv;
        const p = at(i, a / nu - 0.5, v);
        S.pts.push({ x: p.x, y: p.y, r: (0.4 + v * 0.25) * o.u, a: (0.26 + v * 0.26) * dim, k, t: "w", z: Math.min(z, 0.6), cell: true, act: `tier:${tier}` });
      }
    // Rim: dotted, the front edge brightest, the back edge faint.
    const edge = (u0: number, v0: number, u1: number, v1: number, a: number) => {
      const p0 = at(i, u0, v0);
      const p1 = at(i, u1, v1);
      const len = Math.hypot(p1.x - p0.x, p1.y - p0.y);
      const m = Math.max(2, Math.round(len / (o.sp * 0.42)));
      for (let q = 0; q <= m; q++) {
        const t = q / m;
        S.pts.push({ x: p0.x + (p1.x - p0.x) * t, y: p0.y + (p1.y - p0.y) * t, r: 0.5 * o.u, a: a * dim, k, t: "b", z: clamp(z - 0.08, 0.02, 0.9) });
      }
    };
    edge(-0.5, 1, 0.5, 1, 0.8);
    edge(-0.5, 0, -0.5, 1, 0.42);
    edge(0.5, 0, 0.5, 1, 0.42);
    edge(-0.5, 0, 0.5, 0, 0.22);
    planes.push({ tier, poly: [at(i, -0.5, 0), at(i, 0.5, 0), at(i, 0.5, 1), at(i, -0.5, 1)], left: at(i, -0.5, 0.55), right: at(i, 0.5, 0.55) });
  });

  // Folders: columns through every plane.
  const F = o.folders;
  const maxB = Math.max(1, ...F.map((f) => f.bytes));
  const columns: StrataOut["columns"] = [];
  F.forEach((f, j) => {
    const uc = F.length === 1 ? 0 : -0.34 + (0.68 * j) / (F.length - 1);
    const hu = clamp(0.035 + 0.11 * Math.sqrt(f.bytes / maxB), 0.03, 0.46 / Math.max(1, F.length));
    const hv = clamp(0.1 + 0.16 * Math.sqrt(f.bytes / maxB), 0.08, 0.28);
    const vc = 0.52;
    const foot: XY[][] = [];
    TIERS.forEach((tier, i) => {
      const z = clamp(0.08 + i * 0.2, 0.02, 0.9);
      const dim = 1 - i * 0.12;
      const corners = [at(i, uc - hu, vc - hv), at(i, uc + hu, vc - hv), at(i, uc + hu, vc + hv), at(i, uc - hu, vc + hv)];
      foot.push(corners);
      // The footprint's rim, nested inside the plane.
      for (let e = 0; e < 4; e++) {
        const p0 = corners[e];
        const p1 = corners[(e + 1) % 4];
        const len = Math.hypot(p1.x - p0.x, p1.y - p0.y);
        const m = Math.max(2, Math.round(len / (o.sp * 0.3)));
        for (let q = 0; q < m; q++) {
          const t = q / m;
          S.pts.push({ x: p0.x + (p1.x - p0.x) * t, y: p0.y + (p1.y - p0.y) * t, r: 0.42 * o.u, a: (e === 2 ? 0.7 : 0.4) * dim, k: o.folderKey, t: "b", z });
        }
      }
      // Records the engine reports for this tier of this folder: that many points inside.
      const count = Math.max(0, Math.round(f.tiers?.[tier] ?? 0));
      const shown = Math.min(count, 600);
      const cols = Math.max(1, Math.ceil(Math.sqrt(shown * ((hu * o.w) / Math.max(1e-6, hv * o.d)))));
      for (let q = 0; q < shown; q++) {
        const a = q % cols;
        const b = Math.floor(q / cols);
        const rowsN = Math.ceil(shown / cols);
        const p = at(i, uc - hu + ((a + 0.5) / cols) * hu * 2, vc - hv + ((b + 0.5) / rowsN) * hv * 2);
        S.pts.push({ x: p.x, y: p.y, r: 0.7 * o.u, a: 0.8 * dim, k: o.folderKey, t: "b", z });
      }
    });
    // The column: fine lines joining the footprints' front corners from plane to plane.
    for (const c of [3, 2, 0, 1]) S.lines.push({ pts: foot.map((ft) => ft[c]), a: c >= 2 ? 0.32 : 0.14, w: 0.6, dash: c >= 2 ? undefined : [1, 3], k: o.folderKey });
    // Episodes: one bead each on the column's axis, from the top plane down.
    const top = at(0, uc, vc);
    const bottom = at(n - 1, uc, vc);
    S.lines.push({ pts: [top, bottom], a: 0.22, w: 0.6, k: o.folderKey });
    const beads = Math.min(f.episodes, 64);
    for (let q = 0; q < beads; q++) {
      const t = beads === 1 ? 0.5 : q / (beads - 1);
      const y = top.y + 6 * o.u + (bottom.y - top.y - 12 * o.u) * t;
      S.pts.push({ x: top.x + (bottom.x - top.x) * t, y, r: 1.5 * o.u, a: 1, k: o.folderKey, t: "p", z: 0.03 });
    }
    columns.push({ folder: f, top, bottom, foot });
  });
  const p0 = planes[0].poly;
  const pl = planes[n - 1].poly;
  const box = { x: p0[3].x, y: p0[0].y, w: p0[2].x - p0[3].x, h: pl[2].y - p0[0].y };
  return { planes, columns, box };
}
