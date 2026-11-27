/* The user's hardware limiters (`GET /api/limits`, `POST /api/limits`,
 * `POST /api/limits/apply-recommended`) and the three controls behind them: `limits.vram_mb`
 * (`0` runs on the processor only from the next start), `limits.ram_mb` and `limits.cpu_threads`.
 * Each is `auto` (follows this machine) or a whole number. Pure parsers and words, shared by the
 * settings page, the controls and the tests.
 *
 * Every field is read defensively: an engine build without the route gives null, a field in
 * another shape gives null for that field, and nothing is guessed. */

import { limitValue, parseCapacity, parseLiveLimits, type LiveCapacity, type LiveLimits, type LiveWriter } from "@/lib/livefeed";

export { limitValue };

export type LimitField = "vram_mb" | "ram_mb" | "cpu_threads";
export const LIMIT_FIELDS: readonly LimitField[] = ["vram_mb", "ram_mb", "cpu_threads"];
/** The control key of each limiter in `/api/controls`. */
export const LIMIT_KEYS: Record<LimitField, string> = { vram_mb: "limits.vram_mb", ram_mb: "limits.ram_mb", cpu_threads: "limits.cpu_threads" };
export const LIMIT_KEY_SET = new Set(Object.values(LIMIT_KEYS));
export const LIMIT_LABELS: Record<LimitField, string> = { vram_mb: "Graphics card memory (VRAM)", ram_mb: "System memory (RAM)", cpu_threads: "Processor threads" };
export const LIMIT_SHORT: Record<LimitField, string> = { vram_mb: "VRAM", ram_mb: "RAM", cpu_threads: "CPU threads" };
export const LIMIT_UNITS: Record<LimitField, string> = { vram_mb: "MiB", ram_mb: "MiB", cpu_threads: "threads" };

/** A limiter value as the engine gives it: `auto`, a whole number as a string, or null when not
 *  reported. */
export type LimitValue = string | null;
export type LimitTriple = Record<LimitField, LimitValue>;
export type LimitsRecommendation = {
  /** The tier id (`cpu-only`, ...) and its plain label (`Processor only (no usable graphics card)`). */
  tier: string | null;
  tier_label: string | null;
  values: LimitTriple;
  reasons: Record<LimitField, string | null>;
  summary: string | null;
  /** Machine-wide notes with the recommendation (low disk space, for one). */
  notes: string[];
};
export type LimitsReply = {
  /** The saved settings (`current`, keyed `limits.vram_mb` and so on). */
  current: LimitTriple;
  /** The values in force in the process that answered (`effective`). */
  effective: LimitTriple;
  /** The engine's detail of what is in force (`effective.vram`, `.ram`, `.cpu`, `.mode`). */
  effectiveDetail: LiveLimits | null;
  recommended: LimitsRecommendation | null;
  /** This machine's detected hardware (`profile`), read whether or not an engine is serving. */
  profile: LiveCapacity | null;
  /** When a change applies, in words (`applies`). */
  applies: string | null;
  /** Which process the values in force come from, in words (`effective_from`); null when not
   *  reported. */
  effectiveFrom?: string | null;
};

const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);

/** One field of a block, by its short name (`vram_mb`) or its control key (`limits.vram_mb`). */
const fieldOf = (o: Record<string, unknown> | null, f: LimitField): unknown => (o ? (o[f] !== undefined ? o[f] : o[LIMIT_KEYS[f]]) : undefined);

/** The three values of a block (`current`, `effective`, `capacity.limits`), keyed by the short
 *  names or the full control keys. A field holding `{value}` (a recommendation entry) gives its
 *  value. */
export function parseTriple(v: unknown): LimitTriple {
  const o = obj(v);
  const one = (f: LimitField) => {
    const x = fieldOf(o, f);
    return limitValue(obj(x) ? obj(x)?.value : x);
  };
  return { vram_mb: one("vram_mb"), ram_mb: one("ram_mb"), cpu_threads: one("cpu_threads") };
}

/** The recommendation (`recommendation`, or `recommended` in an earlier draft of the contract):
 *  each limiter as `{key, value, reason}`, the tier and its label, and the notes. */
export function parseRecommendation(v: unknown): LimitsRecommendation | null {
  const r = obj(v);
  if (!r) return null;
  const reasons = obj(r.reasons);
  const reason = (f: LimitField) => str(obj(fieldOf(r, f))?.reason) ?? str(reasons?.[f]);
  const notes = (Array.isArray(r.notes) ? r.notes : typeof r.notes === "string" ? [r.notes] : []).map(str).filter((x): x is string => x !== null);
  return {
    tier: str(r.tier),
    tier_label: str(r.tier_label),
    values: parseTriple(r),
    reasons: { vram_mb: reason("vram_mb"), ram_mb: reason("ram_mb"), cpu_threads: reason("cpu_threads") },
    summary: str(r.summary),
    notes,
  };
}

/** The whole `/api/limits` reply. Null when the reply is not an object (an engine build without
 *  the route). */
export function parseLimits(raw: unknown): LimitsReply | null {
  const o = obj(raw);
  if (!o) return null;
  const detail = parseLiveLimits(o.effective);
  return {
    current: parseTriple(o.current),
    effective: detail ? { vram_mb: detail.vram_mb, ram_mb: detail.ram_mb, cpu_threads: detail.cpu_threads } : parseTriple(o.effective),
    effectiveDetail: detail,
    recommended: parseRecommendation(o.recommendation ?? o.recommended),
    profile: parseCapacity(o.profile),
    applies: str(o.applies),
    effectiveFrom: str(o.effective_from),
  };
}

/** The machine to show: the live capacity while an engine is serving (what that engine measured),
 *  else this machine's detected hardware from `/api/limits` `profile`, else whatever the live
 *  reply carries. `source` says which. */
export function machineCapacity(live: LiveCapacity | null | undefined, serving: boolean | null, profile: LiveCapacity | null | undefined): { cap: LiveCapacity | null; source: "live" | "profile" | null } {
  if (serving !== false && live) return { cap: live, source: "live" };
  if (profile) return { cap: profile, source: "profile" };
  if (live) return { cap: live, source: "live" };
  return { cap: null, source: null };
}

/** What `auto` (or a cap) resolves to now, from the engine's detail: the card memory the Nano may
 *  use, the host memory room it plans in, the threads it runs on. Null when not reported. */
export function limitResolved(field: LimitField, d: LiveLimits | null | undefined): number | null {
  if (!d) return null;
  if (field === "vram_mb") return d.vram?.effective_free_mb ?? d.vram?.cap_mb ?? null;
  if (field === "ram_mb") return d.ram?.room_mb ?? d.ram?.cap_mb ?? null;
  return d.cpu?.effective_threads ?? null;
}

/** What the Nano itself uses now under one limiter (its card memory, its own host memory), from the
 *  engine's detail. Null when not reported (and for threads). */
export function limitUsed(field: LimitField, d: LiveLimits | null | undefined): number | null {
  if (!d) return null;
  if (field === "vram_mb") return d.vram?.used_mb ?? null;
  if (field === "ram_mb") return d.ram?.used_mb ?? null;
  return null;
}

/** The value in force with what it resolves to: `auto · 698 MiB free`, `auto · 12 threads`,
 *  `CPU only`, `4096 MiB`. */
export function inForceShort(field: LimitField, v: LimitValue, d: LiveLimits | null | undefined): string {
  const base = limitShort(field, v);
  if (v !== "auto") return base;
  const n = limitResolved(field, d);
  if (n === null) return base;
  if (field === "vram_mb") return `${base} · ${Math.round(n)} MiB free`;
  if (field === "ram_mb") return `${base} · ${Math.round(n)} MiB room`;
  return `${base} · ${n === 1 ? "1 thread" : `${n} threads`}`;
}

/** The number a value stands for, or null for `auto` and unreported. */
export function limitNumber(v: LimitValue): number | null {
  return v !== null && /^\d+$/.test(v) ? Number(v) : null;
}

/** A limiter value in words: `auto (follows this machine)`, `0 MiB (processor only from the
 *  next start)`, `8192 MiB`, `6 threads`, or `not reported`. */
export function limitWords(field: LimitField, v: LimitValue): string {
  if (v === null) return "not reported";
  if (v === "auto") return "auto (follows this machine)";
  if (field === "vram_mb" && v === "0") return "0 MiB (processor only from the next start)";
  if (field === "cpu_threads") return v === "1" ? "1 thread" : `${v} threads`;
  return `${v} ${LIMIT_UNITS[field]}`;
}

/** Short form for meters and tables: `auto`, `CPU only`, `8192 MiB`, `6`. */
export function limitShort(field: LimitField, v: LimitValue): string {
  if (v === null) return "-";
  if (v === "auto") return "auto";
  if (field === "vram_mb" && v === "0") return "CPU only";
  return field === "cpu_threads" ? v : `${v} MiB`;
}

/** What a user typed for one limiter, checked before it is sent. `auto` or a whole number:
 *  graphics card memory 0 (processor only) or 256 MiB and up, system memory 512 MiB and up,
 *  processor threads 1 and up. When the hardware is known, a number above it is refused. */
export type LimitCheck = { ok: true; value: string } | { ok: false; error: string };
export const LIMIT_MIN: Record<LimitField, number> = { vram_mb: 256, ram_mb: 512, cpu_threads: 1 };
export function validateLimit(field: LimitField, raw: string, detected?: number | null): LimitCheck {
  const s = raw.trim();
  if (!s) return { ok: false, error: "Enter auto or a whole number" };
  if (s.toLowerCase() === "auto") return { ok: true, value: "auto" };
  if (!/^\d+$/.test(s)) return { ok: false, error: "Must be auto or a whole number" };
  const n = Number(s);
  if (!Number.isSafeInteger(n)) return { ok: false, error: "Number is out of range" };
  if (field === "vram_mb" && n === 0) return { ok: true, value: "0" };
  const u = field === "cpu_threads" ? (LIMIT_MIN[field] === 1 ? " thread" : " threads") : " MiB";
  if (n < LIMIT_MIN[field]) return { ok: false, error: field === "vram_mb" ? `Must be 0 (processor only) or at least ${LIMIT_MIN[field]} MiB` : `Must be at least ${LIMIT_MIN[field]}${u}` };
  if (typeof detected === "number" && detected > 0 && n > detected) {
    return { ok: false, error: `Must be at most ${Math.round(detected)}${field === "cpu_threads" ? " threads" : " MiB"} (this machine)` };
  }
  return { ok: true, value: String(n) };
}

/** The hardware the engine detected for one limiter (from `profile`, else the live capacity):
 *  the card's total memory, the host's total memory, the processor threads. Null when not known;
 *  on a machine without a graphics card, VRAM is 0. */
export function detectedFor(field: LimitField, profile: LiveCapacity | null | undefined): number | null {
  if (!profile) return null;
  if (field === "vram_mb") return profile.card ? profile.card.total_mb : 0;
  if (field === "ram_mb") return profile.host_total_mb;
  return profile.threads;
}

export function detectedWords(field: LimitField, profile: LiveCapacity | null | undefined): string {
  if (!profile) return "not reported";
  if (field === "vram_mb") {
    if (!profile.card) return "no graphics card (processor only)";
    const t = profile.card.total_mb;
    return `${profile.card.name ?? "graphics card"}${t !== null ? ` · ${Math.round(t)} MiB` : ""}`;
  }
  if (field === "ram_mb") {
    const t = profile.host_total_mb;
    const a = profile.host_available_mb;
    return t !== null ? `${Math.round(t)} MiB${a !== null ? ` · ${Math.round(a)} free` : ""}` : "not reported";
  }
  return profile.threads !== null ? `${profile.threads} threads` : "not reported";
}

/** One change Apply Recommended Defaults would make. */
export type LimitChange = { field: LimitField; key: string; label: string; from: LimitValue; to: LimitValue };

/** The changes applying the recommendation would make: each limiter whose current setting differs
 *  from the recommended one. Empty when nothing is recommended or all already match. */
export function recommendedChanges(reply: LimitsReply | null | undefined): LimitChange[] {
  const rec = reply?.recommended;
  if (!reply || !rec) return [];
  const out: LimitChange[] = [];
  for (const f of LIMIT_FIELDS) {
    const to = rec.values[f];
    if (to === null) continue;
    const from = reply.current[f];
    if (from !== to) out.push({ field: f, key: LIMIT_KEYS[f], label: LIMIT_LABELS[f], from, to });
  }
  return out;
}

/** The body of `POST /api/limits`: one limiter by its control key. */
export function setLimitBody(field: LimitField, value: string): { key: string; value: string } {
  return { key: LIMIT_KEYS[field], value };
}

/** The body of `POST /api/limits/apply-recommended`: only ever sent after the confirmation. */
export const APPLY_RECOMMENDED_BODY = Object.freeze({ confirm: true as const });

/** A tier name in words, for the spectral band of tiers. Unknown names are kept as given. */
export const TIER_BANDS: { id: string; label: string; match: RegExp; hue: string }[] = [
  { id: "cpu", label: "processor only", match: /cpu|processor|none|no.?gpu/i, hue: "#ff6b9a" },
  { id: "4gb", label: "4 GB card", match: /(^|\D)[2-6](\s*-\s*\d+)?\s*(gb|gib|g)\b|small|entry/i, hue: "#ffd36b" },
  { id: "8-12gb", label: "8 to 12 GB card", match: /(^|\D)(8|10|11|12)(\s*-\s*1[0-2])?\s*(gb|gib|g)\b|mid/i, hue: "#7dffa0" },
  { id: "16gb", label: "16 to 24+ GB card", match: /(^|\D)(1[3-9]|2\d|[3-9]\d)(\s*-\s*\d+)?\+?\s*(gb|gib|g)\b|large|high/i, hue: "#6bc8ff" },
];

/** Which band of `TIER_BANDS` a tier names, or -1. */
export function tierBand(tier: string | null | undefined): number {
  if (!tier) return -1;
  return TIER_BANDS.findIndex((b) => b.match.test(tier));
}

/** Where the VRAM cap falls on the card's bar, 0..1, or null when there is no numeric cap (auto,
 *  processor only or unreported) or no card total. */
export function vramCapShare(cap: LimitValue, totalMb: number | null | undefined): number | null {
  const n = limitNumber(cap);
  if (n === null || n <= 0 || typeof totalMb !== "number" || totalMb <= 0) return null;
  return Math.min(1, n / totalMb);
}

/** How the MicroModels sit on the card now: from the engine's residency mode
 *  (`capacity.limits.mode`: `four seated`, `evict and restore (...)`, `none seated`, `processor`)
 *  when given, else from the idle writer's `kept_resident`: all four seated together, or the
 *  writer swapped in and out (evict and restore). Null when neither is reported. Pass `mode` only
 *  from a serving engine: with none serving, the interface process's own mode says nothing about
 *  the MicroModels. */
export type Residency = { mode: "seated" | "swapping" | "none" | "processor"; words: string } | null;
export function residency(writer: Pick<LiveWriter, "kept_resident"> | null | undefined, mode?: string | null): Residency {
  const m = (mode ?? "").trim().toLowerCase();
  if (/four seated|all seated/.test(m)) return { mode: "seated", words: "All four MicroModels seated together on the card" };
  if (/evict|restore|swap/.test(m)) return { mode: "swapping", words: "Writer swaps in and out (evict and restore) to stay inside the limit" };
  if (/none seated/.test(m)) return { mode: "none", words: "No MicroModel seated on the card yet" };
  if (/processor|cpu/.test(m)) return { mode: "processor", words: "Every MicroModel runs on the processor" };
  const k = writer?.kept_resident;
  if (k === true) return { mode: "seated", words: "All four MicroModels seated together on the card" };
  if (k === false) return { mode: "swapping", words: "Writer swaps in and out (evict and restore) to stay inside the limit" };
  return null;
}
