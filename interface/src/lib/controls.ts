import type { Control } from "./api";

export const BUDGET_GROUP = "Budgets and allowances";
export const LIMITS_GROUP = "Hardware limits";
export const AGENTS_GROUP = "Coding agents";

/** The user's hardware limiters: each takes `auto` (follows this machine) or a whole number.
 *  `limits.vram_mb` 0 runs on the processor only from the next start. */
export const LIMIT_CONTROL_KEYS = new Set(["limits.vram_mb", "limits.ram_mb", "limits.cpu_threads"]);
export function isLimitControl(c: { key?: string }): boolean {
  return typeof c.key === "string" && LIMIT_CONTROL_KEYS.has(c.key);
}

export const CONTROL_GROUP_ORDER = [
  "Precision",
  LIMITS_GROUP,
  BUDGET_GROUP,
  "Memory and privacy",
  // 2026-11-26: these controls act for every coding agent that connects over MCP (Claude Code,
  // Cursor, VS Code, Antigravity, Windsurf and any other client), so the group says so.
  AGENTS_GROUP,
  "Learning",
  "Routing",
  "Cartridges",
  "Storage",
  "Network",
  "Other",
] as const;

export const MEMORY_KEYS = new Set(["memory", "dense_cache", "recall_cache", "prompts", "prompts.days", "audit.keep_days"]);
/** Priming is read under its API 7 key, or the earlier `nudges` a build before it gives. */
export const CLAUDE_KEYS = new Set(["priming", "nudges", "mode", "mcp.tools", "mcp.tool_count", "mcp.tool_control", "mcp.enabled_tools", "mcp.dynamic_tools", "host.claude_lean"]);
export const LEARNING_KEYS = new Set(["muscle", "learning", "learning.cycle", "examples"]);
export const NETWORK_KEYS = new Set(["network", "air_gap"]);

export function isBudgetControl(c: Control): boolean {
  return (
    c.key.startsWith("budget.") ||
    c.key.startsWith("quota.") ||
    c.key === "rerank.early_drop" ||
    c.key === "writer.idle" ||
    c.group === BUDGET_GROUP
  );
}

export function isNumericControl(c: Control): boolean {
  return (
    isLimitControl(c) ||
    c.kind === "integer" ||
    ((c.key.startsWith("budget.") || c.key.startsWith("quota.") || c.key === "mcp.tool_count" || c.key === "prompts.days") &&
      typeof c.min === "number" &&
      typeof c.max === "number")
  );
}

export function controlGroupName(c: Control): string {
  if (c.key.startsWith("precision.")) return "Precision";
  if (isLimitControl(c)) return LIMITS_GROUP;
  if (MEMORY_KEYS.has(c.key)) return "Memory and privacy";
  if (CLAUDE_KEYS.has(c.key)) return AGENTS_GROUP;
  if (LEARNING_KEYS.has(c.key)) return "Learning";
  if (isBudgetControl(c)) return BUDGET_GROUP;
  if (NETWORK_KEYS.has(c.key) || c.group === "Network") return "Network";
  if (c.group === "Claude Code") return AGENTS_GROUP;
  if (c.group && c.group.trim()) return c.group;
  return "Other";
}

/** True for a numeric control that also takes `auto` (it follows the machine): its default, its
 *  value or one of its presets is `auto`. `budget.gpu_mb_max` is one (default `auto`, or a number
 *  up to 81,920 MiB). */
export function acceptsAuto(c: Pick<Control, "values" | "default" | "value"> & { key?: string }): boolean {
  return isLimitControl(c) || c.default === "auto" || c.value === "auto" || (c.values ?? []).includes("auto");
}

/** `value` is null for `auto`. */
export type IntValidation =
  | { ok: true; value: number | null; normalized: string }
  | { ok: false; error: string };

export function validateIntegerControl(c: Control, raw: string): IntValidation {
  const s = raw.trim();
  if (acceptsAuto(c) && s.toLowerCase() === "auto") return { ok: true, value: null, normalized: "auto" };
  if (!s) return { ok: false, error: acceptsAuto(c) ? "Enter a whole number or auto" : "Enter a whole number" };
  if (!/^-?\d+$/.test(s)) return { ok: false, error: acceptsAuto(c) ? "Must be a whole number or auto" : "Must be a whole number" };
  const n = Number(s);
  if (!Number.isSafeInteger(n)) return { ok: false, error: "Number is out of range" };
  // Graphics card memory 0 is the processor-only setting, whatever the card's lower bound.
  if (c.key === "limits.vram_mb" && n === 0) return { ok: true, value: 0, normalized: "0" };
  if (isLimitControl(c) && n < 0) return { ok: false, error: "Must be auto or a whole number of 0 or more" };
  const u = c.unit ? ` ${c.unit}` : "";
  if (typeof c.min === "number" && n < c.min) return { ok: false, error: `Must be at least ${c.min}${u}` };
  if (typeof c.max === "number" && n > c.max) return { ok: false, error: `Must be at most ${c.max}${u}` };
  return { ok: true, value: n, normalized: String(n) };
}

/** One step up or down. From `auto`, stepping starts at `autoValue` (what auto resolves to on
 *  this machine) when known. */
export function stepIntegerControl(c: Control, raw: string, delta: -1 | 1, autoValue?: number | null): string {
  const s = raw.trim();
  let cur: number;
  if (/^-?\d+$/.test(s) && Number.isSafeInteger(Number(s))) {
    cur = Number(s);
  } else if (s.toLowerCase() === "auto" && typeof autoValue === "number" && Number.isFinite(autoValue)) {
    cur = Math.round(autoValue);
  } else if (/^-?\d+$/.test(c.value) && Number.isSafeInteger(Number(c.value))) {
    cur = Number(c.value);
  } else if (/^-?\d+$/.test(c.default) && Number.isSafeInteger(Number(c.default))) {
    cur = Number(c.default);
  } else {
    cur = typeof c.min === "number" ? c.min : 0;
  }
  const step = typeof c.step === "number" && c.step > 0 ? Math.trunc(c.step) : 1;
  let next = cur + delta * step;
  if (typeof c.min === "number") next = Math.max(c.min, next);
  if (typeof c.max === "number") next = Math.min(c.max, next);
  return String(next);
}

/** Formats a control value for status notices and discrete scale labels. */
export function formatControlValue(v: string, key = "", unit?: string): string {
  if (v === "default") return "auto";
  if (v === "auto" && key === "budget.gpu_mb_max") return "auto (follows this card)";
  if (v === "auto" && LIMIT_CONTROL_KEYS.has(key)) return "auto (follows this machine)";
  if (v === "0" && key === "limits.vram_mb") return "0 (processor only from the next start)";
  if (/^\d+$/.test(v) && key === "limits.cpu_threads") return v === "1" ? "1 thread" : `${v} threads`;
  if (v === "0" && key.startsWith("quota.")) return "unlimited";
  if (/^\d+$/.test(v) && key.startsWith("precision.")) return `${v}-bit`;
  if (/^\d+$/.test(v) && key.endsWith(".days")) return v === "1" ? "1 day" : `${v} days`;
  if (/^-?\d+$/.test(v) && unit) return `${v} ${unit}`;
  return v;
}
