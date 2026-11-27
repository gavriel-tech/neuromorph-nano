/* The graphics card meter per MicroModel: each of the models the Nano runs, what the serving
 * processes hold of it on the card now (`/api/status` `loaded[].gpu_bytes`), and its share of the
 * ceiling (`budget.gpu_mb_max`). Pure, shared by the settings page and the tests. */

import type { Status } from "@/lib/api";

/** `device` is where the model is held (`/api/status` `loaded[].device`), null when not reported;
 *  `precision` is its weight precision when the live state reports one. */
export type VramSegment = { id: string; role?: string; held: boolean; gpuBytes: number; device: string | null; precision?: string | null; share: number };

const MB = 1024 * 1024;

/** One segment per model the engine lists (every model, held or not, in its order), then any
 *  model held that is not listed. Bytes of the same model held by more than one process add up.
 *  `share` is the part of the ceiling, 0..1 (all shares together may pass 1 when over it). */
export function vramSegments(status: Pick<Status, "models" | "loaded"> | null | undefined, ceilingMb: number): VramSegment[] {
  const ceiling = Math.max(1, ceilingMb) * MB;
  const held = new Map<string, { bytes: number; device: string | null }>();
  for (const m of status?.loaded ?? []) {
    const cur = held.get(m.id) ?? { bytes: 0, device: null };
    cur.bytes += Math.max(0, m.gpu_bytes ?? 0);
    cur.device = cur.device ?? m.device ?? null;
    held.set(m.id, cur);
  }
  const ids = [...(status?.models ?? []).map((m) => m.id)];
  for (const id of held.keys()) if (!ids.includes(id)) ids.push(id);
  return ids.map((id) => {
    const h = held.get(id);
    const gpuBytes = h?.bytes ?? 0;
    return { id, held: Boolean(h), gpuBytes, device: h?.device ?? null, share: gpuBytes / ceiling };
  });
}

/** One segment per MicroModel of the live state (`/api/live` `models[]`: embedder, reranker,
 *  reader, writer, in the engine's order): its card id (else its role), whether it is resident,
 *  what it holds on the card (`vram_mb`) and its precision. The live state names no device per
 *  model, so `device` stays null. This is what the serving engine holds;
 *  `/api/status` `loaded[]` is used when the live state lists no model. */
export function liveVramSegments(models: { role: string; card: string | null; resident: boolean; vram_mb: number | null; precision: string | null }[] | null | undefined, ceilingMb: number): VramSegment[] {
  const ceiling = Math.max(1, ceilingMb) * MB;
  return (models ?? []).map((m) => {
    const gpuBytes = Math.max(0, m.vram_mb ?? 0) * MB;
    return { id: m.card ?? m.role, role: m.role, held: m.resident && gpuBytes > 0, gpuBytes, device: null, precision: m.precision, share: gpuBytes / ceiling };
  });
}

/** Megabytes, rounded. */
export function mb(bytes: number): number {
  return Math.round(bytes / MB);
}

/** One part of the graphics card's memory in the live breakdown (`/api/live` `vram`). */
export type VramPart = { id: string; label: string; mb: number; share: number; kind: "model" | "kv" | "other" | "reserve" | "free" };

const VRAM_ORDER = ["embedder", "reranker", "reader", "writer"];

/** The card's memory as the four MicroModels, KV caches and windows, other use, the free reserve
 *  kept for live calls, and the rest free. Shares are of the card's total (0..1). Null when the
 *  engine reports no total. Unknown model keys keep their own name, after the four. */
export function vramBreakdown(v: { total_mb: number | null; used_mb: number | null; free_mb: number | null; reserve_mb: number | null; by_model: Record<string, number> } | null | undefined): VramPart[] | null {
  if (!v || v.total_mb === null || v.total_mb <= 0) return null;
  const total = v.total_mb;
  const parts: VramPart[] = [];
  const keys = Object.keys(v.by_model);
  const rank = (k: string) => {
    const i = VRAM_ORDER.findIndex((o) => k.toLowerCase().includes(o));
    return i < 0 ? (/kv|window|cache/i.test(k) ? 10 : 5) : i;
  };
  keys.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  let models = 0;
  for (const k of keys) {
    const mbv = Math.max(0, v.by_model[k]);
    models += mbv;
    const kv = /kv|window|cache/i.test(k);
    // Card memory no MicroModel accounts for, as the engine names it (2026-11-27).
    if (k === "unattributed") {
      if (mbv > 0.5) parts.push({ id: k, label: "not attributed to a model", mb: mbv, share: mbv / total, kind: "other" });
      continue;
    }
    parts.push({ id: k, label: kv ? "KV and windows" : k === "filters" ? "job filters" : k, mb: mbv, share: mbv / total, kind: kv ? "kv" : "model" });
  }
  const used = v.used_mb ?? (v.free_mb !== null ? total - v.free_mb : models);
  const other = Math.max(0, used - models);
  if (other > 0.5) parts.push({ id: "other", label: "other use", mb: other, share: other / total, kind: "other" });
  const free = Math.max(0, v.free_mb ?? total - used);
  const reserve = Math.min(free, Math.max(0, v.reserve_mb ?? 0));
  if (reserve > 0) parts.push({ id: "reserve", label: "free reserve", mb: reserve, share: reserve / total, kind: "reserve" });
  if (free - reserve > 0) parts.push({ id: "free", label: "free", mb: free - reserve, share: (free - reserve) / total, kind: "free" });
  return parts;
}

/** The graphics card ceiling `budget.gpu_mb_max` resolves to. `auto` (the default, and a missing
 *  value) follows this card: the engine's measured ceiling (`capacity.gpu_ceiling_mb`, the card
 *  total less the live calls' measured room). A pinned number (1,024 to 81,920 MiB) is used as it
 *  is. `mb` is null when auto and the engine reports no ceiling: then nothing is guessed, and the
 *  label says the ceiling is not reported. */
export type GpuCeiling = { mb: number | null; auto: boolean; label: string };
export const AUTO_CARD = "auto (follows this card)";
export function gpuCeiling(raw: string | null | undefined, cap?: { gpu_ceiling_mb: number | null; card: { total_mb: number | null; reserve_mb: number | null } | null } | null): GpuCeiling {
  const v = (raw ?? "auto").trim().toLowerCase();
  if (/^\d+$/.test(v) && Number(v) > 0) {
    const n = Number(v);
    return { mb: n, auto: false, label: `${n} MiB` };
  }
  const mbv = typeof cap?.gpu_ceiling_mb === "number" && cap.gpu_ceiling_mb > 0 ? cap.gpu_ceiling_mb : null;
  return { mb: mbv, auto: true, label: mbv !== null ? `${AUTO_CARD} · ${Math.round(mbv)} MiB` : `${AUTO_CARD} · ceiling not reported` };
}
