"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals ... We need the interface to be
// amazing."): the profiles, the two master switches, the tool switches and the budgets stay as
// controls; their descriptions move into tooltips, and LeanModeGuide (the Settings rail) lists
// every profile and every tool with what it does. Columns follow the panel's width, not the window.

import { useMemo, useState } from "react";
import { NoteLine, setControl } from "@/components/Controls";
import { RailSection } from "@/components/Workspace";
import { Panel, Pill, Placeholder } from "@/components/ui";
import { useApi, type Control } from "@/lib/api";
import { validateIntegerControl } from "@/lib/controls";

type Note = { text: string; tone: "verified" | "error" };

type ToolGroup = "Core retrieval" | "Build & verification" | "Memory & workspace" | "Anatomy & improvement" | "Control & parity" | "Other tools";

type ToolMeta = {
  name: string;
  group: ToolGroup;
  desc?: string;
  /** A typed-in guess at the tool's schema size in tokens, not a measurement. */
  estTokens?: number;
};

/** What this page knows about some of the engine's MCP tools. The list itself, and its order,
 *  come from the engine (`all_tools` on the `mcp.enabled_tools` control); a tool not described
 *  here is shown by its name only. */
const KNOWN_TOOLS: Record<string, Omit<ToolMeta, "name">> = {
  nm_context_pack: { group: "Core retrieval", desc: "Primary codebase search returning cited ranges", estTokens: 210 },
  nm_symbol_find: { group: "Core retrieval", desc: "Exact definition and symbol lookup across files", estTokens: 130 },
  nm_view: { group: "Core retrieval", desc: "Read one specific substrate lens or file view", estTokens: 135 },
  nm_deps: { group: "Core retrieval", desc: "Inspect crate, package, and module dependencies", estTokens: 125 },
  nm_feedback: { group: "Core retrieval", desc: "Send outcome or correction signal to learning", estTokens: 145 },
  nm_repo_map: { group: "Core retrieval", desc: "High-level repository structure and file map", estTokens: 130 },
  nm_build_run: { group: "Build & verification", desc: "Run check, build, or tests on a workspace package", estTokens: 155 },
  nm_build_explain: { group: "Build & verification", desc: "Explain compiler or test failure diagnostics", estTokens: 140 },
  nm_stepcheck: { group: "Build & verification", desc: "Run step-level verification checks", estTokens: 135 },
  nm_verify: { group: "Build & verification", desc: "Verify invariants and integrity after edits", estTokens: 140 },
  nm_status: { group: "Memory & workspace", desc: "Engine status, models, memory, and device health", estTokens: 120 },
  nm_sync: { group: "Memory & workspace", desc: "Incrementally rescan modified workspace files", estTokens: 115 },
  nm_memory: { group: "Memory & workspace", desc: "Query or inspect persistent cross-session memory", estTokens: 150 },
  nm_goal: { group: "Memory & workspace", desc: "Read or update active session goals and objectives", estTokens: 135 },
  nm_anatomy: { group: "Anatomy & improvement", desc: "Inspect organism anatomy graph and assemblies", estTokens: 125 },
  nm_weights: { group: "Anatomy & improvement", desc: "Inspect resident MicroModel weights and cartridges", estTokens: 130 },
  nm_improve_scan: { group: "Anatomy & improvement", desc: "Scan codebase for candidate improvements", estTokens: 145 },
  nm_patch_propose: { group: "Anatomy & improvement", desc: "Propose a bounded, verified patch", estTokens: 165 },
  nm_unit: { group: "Anatomy & improvement", desc: "Learning units: list, register, load and unload low-rank adapters", estTokens: 130 },
  // 2026-11-26: the six interface tools, one for one with the pages (nmapi::parity).
  nm_live: { group: "Control & parity", desc: "Live activity: what each part does now, the last calls and their stage times", estTokens: 180 },
  nm_storage: { group: "Control & parity", desc: "Disk use by category, prune and delete (two steps), forget a project or a memory, audit retention", estTokens: 330 },
  nm_cartridges: { group: "Control & parity", desc: "Cartridges, stacks, base versions, parameter maps and job filters: list, switch, build, remove", estTokens: 380 },
  nm_learning: { group: "Control & parity", desc: "Learning engine: cycles, rollback, training store, ledger, learned units plugged in or out", estTokens: 360 },
  nm_comprehension: { group: "Control & parity", desc: "A project's world model: L1 to L4 notes, routing, lenses and memory tiers", estTokens: 150 },
  nm_settings: { group: "Control & parity", desc: "Every interface control, status, audit and undo, devices, roles, downloads, hardware limits", estTokens: 300 },
};

const TOOL_GROUPS: ToolGroup[] = [
  "Core retrieval",
  "Build & verification",
  "Memory & workspace",
  "Anatomy & improvement",
  "Control & parity",
  "Other tools",
];

/** The engine's MCP tools in its coding priority order, from `all_tools` on the
 *  `mcp.enabled_tools` control; empty until /api/controls answers or when the engine does not
 *  report the list. */
export function mcpToolList(byKey: Map<string, Control>): string[] {
  const all = byKey.get("mcp.enabled_tools")?.all_tools;
  return Array.isArray(all) ? all.filter((n): n is string => typeof n === "string" && n.length > 0) : [];
}

function toolCatalog(names: string[]): ToolMeta[] {
  return names.map((name) => ({ name, ...(KNOWN_TOOLS[name] ?? { group: "Other tools" as const }) }));
}

type Preset = {
  id: string;
  label: string;
  summary: string;
  changes: Array<[string, string]>;
};

/** The advertised count that means "every tool": the engine's tool count, held to the count
 *  control's own maximum when it reports one. */
function fullToolCount(byKey: Map<string, Control>, total: number): number {
  const max = byKey.get("mcp.tool_count")?.max;
  return typeof max === "number" && max > 0 ? Math.min(total, max) : total;
}

function buildPresets(full: number, total: number, standard: number): Preset[] {
  return [
    {
      id: "default",
      label: "Default (Off)",
      summary: `Tool Control OFF, Dynamic OFF: default model settings (${Math.min(standard, total)} of ${total} tools advertised; the interface tools stay callable by name)`,
      changes: [
        ["mcp.tool_control", "off"],
        ["mcp.dynamic_tools", "off"],
        ["mcp.tools", "full"],
        ["mcp.tool_count", String(Math.min(standard, full))],
        ["host.claude_lean", "off"],
        ["budget.adaptive_pack", "on"],
        ["budget.focused_tokens", "1800"],
        ["budget.pack_tokens", "4000"],
      ],
    },
    {
      id: "all",
      label: `All ${full} (with Interface Controls)`,
      summary: `Tool Control OFF, Dynamic OFF: every tool advertised, including the six that control the interface's pages (${full} of ${total})`,
      changes: [
        ["mcp.tool_control", "off"],
        ["mcp.dynamic_tools", "off"],
        ["mcp.tools", "full"],
        ["mcp.tool_count", String(full)],
        ["budget.adaptive_pack", "on"],
      ],
    },
    {
      id: "dynamic",
      label: "Dynamic Lean",
      summary: "Tool Control OFF, Dynamic ON: starts at Core 5, activates extra tools on demand",
      changes: [
        ["mcp.tool_control", "off"],
        ["mcp.dynamic_tools", "on"],
        ["mcp.tools", "lean"],
        ["mcp.tool_count", "5"],
        ["host.claude_lean", "on"],
        ["budget.adaptive_pack", "on"],
        ["budget.focused_tokens", "1800"],
        ["budget.pack_tokens", "4000"],
      ],
    },
    {
      id: "custom_strict",
      label: "Custom Core 5 (Strict)",
      summary: "Tool Control ON, Dynamic OFF: strictly your 5 selected tools, others disabled",
      changes: [
        ["mcp.tool_control", "on"],
        ["mcp.dynamic_tools", "off"],
        ["mcp.enabled_tools", "nm_context_pack,nm_symbol_find,nm_view,nm_deps,nm_feedback"],
        ["host.claude_lean", "on"],
        ["budget.adaptive_pack", "on"],
        ["budget.focused_tokens", "1800"],
        ["budget.pack_tokens", "4000"],
      ],
    },
    {
      id: "custom_dynamic",
      label: "Custom + Dynamic",
      summary: "Tool Control ON, Dynamic ON: starts with your selected tools + activates on demand",
      changes: [
        ["mcp.tool_control", "on"],
        ["mcp.dynamic_tools", "on"],
        ["mcp.enabled_tools", "nm_context_pack"],
        ["host.claude_lean", "on"],
        ["budget.adaptive_pack", "on"],
        ["budget.focused_tokens", "1500"],
        ["budget.pack_tokens", "3000"],
      ],
    },
  ];
}

/** Which tools the engine's controls turn on, read only from the controls' real values. */
function resolveActiveTools(byKey: Map<string, Control>, tools: string[]): {
  toolControlOn: boolean;
  dynamicOn: boolean;
  enabled: Set<string>;
  /** Names in the saved selection that the engine's tool list does not carry; kept on save. */
  extras: string[];
} {
  const toolControlOn = byKey.get("mcp.tool_control")?.value === "on";
  const dynamicOn = byKey.get("mcp.dynamic_tools")?.value === "on";
  if (toolControlOn) {
    const customRaw = byKey.get("mcp.enabled_tools")?.value?.trim() ?? "";
    if (customRaw && customRaw.toLowerCase() !== "auto") {
      const names = customRaw
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const picked = names.filter((name) => tools.includes(name));
      const extras = names.filter((name) => !tools.includes(name));
      if (picked.length > 0) {
        return { toolControlOn, dynamicOn, enabled: new Set(picked), extras };
      }
    }
  }
  const profile = byKey.get("mcp.tools")?.value;
  const countRaw = Number(byKey.get("mcp.tool_count")?.value);
  const count = Number.isFinite(countRaw) && countRaw > 0 ? countRaw : tools.length;
  const eff = profile === "lean" ? Math.min(5, count) : Math.max(1, Math.min(tools.length, count));
  return { toolControlOn, dynamicOn, enabled: new Set(tools.slice(0, eff)), extras: [] };
}

function BudgetQuickRow({
  control,
  busy,
  onSave,
}: {
  control: Control;
  busy: boolean;
  onSave: (key: string, val: string) => Promise<void>;
}) {
  const [draft, setDraft] = useState(control.value);
  const [prev, setPrev] = useState(control.value);
  const [err, setErr] = useState<string | null>(null);

  if (prev !== control.value) {
    setPrev(control.value);
    setDraft(control.value);
    setErr(null);
  }

  const commit = async (val: string) => {
    const check = validateIntegerControl(control, val);
    if (!check.ok) {
      setErr(check.error);
      return;
    }
    setErr(null);
    await onSave(control.key, check.normalized);
  };

  return (
    <div className="border-b border-line/60 py-2.5 last:border-b-0">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="mono text-[11px] uppercase tracking-[0.1em] text-muted">
          {control.label}
          {control.unit ? <span className="text-quiet"> · {control.unit}</span> : null}
        </span>
        <span className="mono text-xs text-text">
          {control.value === "0" && control.key.startsWith("quota.") ? "unlimited" : control.value}
        </span>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        {control.values.map((v) => {
          const active = v === control.value;
          return (
            <button
              key={v}
              type="button"
              disabled={busy}
              onClick={() => void commit(v)}
              className={`press mono min-h-7 border px-2 text-[10px] uppercase tracking-[0.06em] ${
                active ? "border-sense-pale text-text" : "border-line text-faint hover:border-line-strong hover:text-text"
              }`}
            >
              {v === "0" && control.key.startsWith("quota.") ? "unlimited" : v}
            </button>
          );
        })}
        <form
          className="ml-auto flex items-center gap-1"
          onSubmit={(e) => {
            e.preventDefault();
            void commit(draft);
          }}
        >
          <input
            aria-label={control.label}
            inputMode="numeric"
            value={draft}
            disabled={busy}
            onChange={(e) => setDraft(e.target.value)}
            className="mono h-7 w-24 border border-line bg-transparent px-2 text-right text-xs text-text outline-none focus-visible:border-sense-pale"
          />
          <button
            type="submit"
            disabled={busy || draft.trim() === control.value}
            className="press mono h-7 border border-line-strong px-2 text-[10px] uppercase tracking-[0.08em] text-muted hover:text-text disabled:opacity-40"
          >
            Set
          </button>
        </form>
      </div>
      {err && <div className="mono mt-1 text-[11px] text-error">{err}</div>}
    </div>
  );
}

export default function LeanModePanel() {
  const { data: rawControls, error: controlsError } = useApi<Control[] | { controls?: Control[] }>("/api/controls", 5000);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<Note | null>(null);

  const byKey = useMemo(() => {
    const map = new Map<string, Control>();
    const list = Array.isArray(rawControls) ? rawControls : rawControls?.controls ?? [];
    for (const c of list) map.set(c.key, c);
    return map;
  }, [rawControls]);

  const tools = useMemo(() => mcpToolList(byKey), [byKey]);
  const catalog = useMemo(() => toolCatalog(tools), [tools]);
  const total = tools.length;
  const fullCount = fullToolCount(byKey, total);
  // The count control's own default (19: every tool before the six interface tools).
  const standardCount = Number(byKey.get("mcp.tool_count")?.default) || fullCount;
  const presets = useMemo(() => buildPresets(fullCount, total, standardCount), [fullCount, total, standardCount]);

  const { toolControlOn, dynamicOn, enabled, extras } = useMemo(() => resolveActiveTools(byKey, tools), [byKey, tools]);
  // A typed-in estimate of the schema tokens the enabled tools add to each turn, over the tools
  // this page has a guess for; nothing here is measured.
  const estimate = useMemo(() => {
    const on = catalog.filter((t) => enabled.has(t.name));
    const guessed = on.filter((t) => typeof t.estTokens === "number");
    return { tokens: guessed.reduce((s, t) => s + (t.estTokens ?? 0), 0), covered: guessed.length, of: on.length };
  }, [catalog, enabled]);

  const activeModeLabel = toolControlOn
    ? (dynamicOn ? "custom + dynamic" : "custom")
    : (dynamicOn && enabled.size < total ? "dynamic" : "default");

  const activeModeBanner = toolControlOn
    ? (dynamicOn
        ? `Tool Control ON + Dynamic ON: starts with your ${enabled.size} of ${total} selected tools and activates extra tools on demand.`
        : `Tool Control ON (strict): only your ${enabled.size} of ${total} selected tools are active; all other tools are turned off and blocked.`)
    : (dynamicOn && enabled.size < total
        ? `Tool Control OFF + Dynamic ON: starts with ${enabled.size} of ${total} tools advertised and activates the others when a call needs them.`
        : `Tool Control OFF: using default model settings (${enabled.size} of ${total} tools advertised).`);

  const saveOne = async (key: string, value: string) => {
    setBusy(true);
    const res = await setControl(key, value);
    setBusy(false);
    setNote(res);
  };

  const saveMany = async (changes: Array<[string, string]>, okMessage: string) => {
    setBusy(true);
    let last: Note = { text: okMessage, tone: "verified" };
    // Keys this engine's /api/controls does not carry cannot be set; they are named, not hidden.
    const skipped: string[] = [];
    for (const [k, v] of changes) {
      if (!byKey.has(k)) {
        skipped.push(k);
        continue;
      }
      if (byKey.get(k)?.value !== v) {
        last = await setControl(k, v);
        if (last.tone === "error") break;
      }
    }
    setBusy(false);
    if (last.tone === "error") setNote(last);
    else if (skipped.length > 0)
      setNote({
        text: `Not fully saved: this engine has no ${skipped.length === 1 ? "control" : "controls"} ${skipped.join(", ")}, so ${skipped.length === 1 ? "it was" : "they were"} skipped. The other changes were saved and logged in Audit.`,
        tone: "error",
      });
    else setNote({ text: okMessage, tone: "verified" });
  };

  const applyPreset = async (preset: Preset) => {
    await saveMany(
      preset.changes,
      `${preset.label} profile saved (${preset.summary}). Every change is logged in Audit with undo.`
    );
  };

  const toggleToolControl = async () => {
    if (toolControlOn) {
      // Turning Tool Control OFF returns to default model settings (unless Dynamic is ON, in
      // which case Dynamic starts from lean Core 5). Saved custom selections stay intact.
      const changes: Array<[string, string]> = [["mcp.tool_control", "off"]];
      if (!dynamicOn) {
        changes.push(["mcp.tools", "full"], ["mcp.tool_count", String(fullCount)]);
      }
      await saveMany(
        changes,
        dynamicOn
          ? "Tool Control turned OFF (Dynamic stays ON)."
          : `Tool Control turned OFF: restored default model settings (${fullCount} of ${total} tools advertised).`
      );
    } else {
      // Turning Tool Control ON activates individual selection; seed from Core 5 if still "auto".
      const currentCustom = byKey.get("mcp.enabled_tools")?.value?.trim() ?? "auto";
      const changes: Array<[string, string]> = [["mcp.tool_control", "on"]];
      if (!currentCustom || currentCustom.toLowerCase() === "auto") {
        const seed = tools.filter((n) => enabled.has(n));
        const list = seed.length > 0 && seed.length < total
          ? seed
          : tools.slice(0, 5);
        changes.push(["mcp.enabled_tools", list.join(",")]);
      }
      await saveMany(changes, "Tool Control turned ON: select the exact tools you want below.");
    }
  };

  const toggleDynamic = async () => {
    if (dynamicOn) {
      const changes: Array<[string, string]> = [["mcp.dynamic_tools", "off"]];
      if (!toolControlOn) {
        changes.push(["mcp.tools", "full"], ["mcp.tool_count", String(fullCount)]);
      }
      await saveMany(
        changes,
        toolControlOn
          ? "Dynamic tool activation turned OFF: strictly locked to your selected tools."
          : `Dynamic tool activation turned OFF: restored default model settings (${fullCount} of ${total} tools advertised).`
      );
    } else {
      const changes: Array<[string, string]> = [["mcp.dynamic_tools", "on"]];
      if (!toolControlOn && enabled.size >= total) {
        changes.push(["mcp.tools", "lean"], ["mcp.tool_count", "5"]);
      }
      await saveMany(
        changes,
        toolControlOn
          ? "Dynamic tool activation turned ON: starts with your selected tools and activates extra tools on demand."
          : "Dynamic tool activation turned ON: starts lean (Core 5) and activates other tools on demand."
      );
    }
  };

  const toggleTool = async (toolName: string) => {
    if (!toolControlOn) return;
    const next = new Set(enabled);
    if (next.has(toolName)) {
      if (next.size <= 1) {
        setNote({ text: "Keep at least 1 MCP tool enabled while Tool Control is ON (nm_context_pack recommended).", tone: "error" });
        return;
      }
      next.delete(toolName);
    } else {
      next.add(toolName);
    }
    // The engine's order, then any saved names its list does not carry, so none is dropped.
    const ordered = [...tools.filter((n) => next.has(n)), ...extras];
    await saveOne("mcp.enabled_tools", ordered.join(","));
  };

  const setQuickTools = async (names: string[]) => {
    await saveMany(
      [
        ["mcp.tool_control", "on"],
        ["mcp.enabled_tools", names.join(",")],
      ],
      `Selected ${names.length} of ${total} tools.`
    );
  };

  const toggleKeys = [
    { key: "host.claude_lean", title: "Claude Code lean host", sub: "Disable built-in Artifact tool description in .claude/settings.local.json" },
    { key: "budget.adaptive_pack", title: "Adaptive compact packs", sub: "Shrink focused high-confidence packs down to Focused tokens" },
  ];

  const budgetKeys = [
    "budget.pack_tokens",
    "budget.focused_tokens",
    "budget.tentacle_tokens",
    "quota.session_tokens",
    "quota.daily_tokens",
    "quota.session_packs",
  ];

  const earlyDrop = byKey.get("rerank.early_drop");
  const quotaAction = byKey.get("quota.action");

  return (
    <Panel
      title="Lean Mode, Tool Control & Token Budgeting"
      right={
        <div className="flex flex-wrap items-center gap-2">
          {total > 0 && (
            <Pill tone={toolControlOn || (dynamicOn && enabled.size < total) ? "verified" : "signal"}>
              {enabled.size} of {total} tools ({activeModeLabel})
            </Pill>
          )}
          {estimate.covered > 0 && (
            <Pill
              tone="muted"
              title="Typed-in guesses of each tool's schema size, not measured by the engine."
            >
              about {estimate.tokens.toLocaleString()} tokens a turn, estimated
              {estimate.covered < estimate.of ? ` for ${estimate.covered} of ${estimate.of} tools` : ""}
            </Pill>
          )}
        </div>
      }
    >
      <NoteLine note={note} />
      {byKey.size === 0 && controlsError ? (
        <div className="mono text-xs text-error">Controls not reported: {controlsError}</div>
      ) : byKey.size === 0 ? (
        <Placeholder className="h-48" what="lean mode" />
      ) : (
        <div className="@container grid gap-6">
          {/* 1. Presets */}
          <div className="grid gap-2">
            <div className="eyebrow">Quick Lean Mode profiles</div>
            <div className="grid grid-cols-2 gap-1.5 @3xl:grid-cols-5">
              {presets.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  disabled={busy}
                  onClick={() => void applyPreset(p)}
                  title={p.summary}
                  className="press mono min-h-10 border border-line px-2.5 py-1.5 text-left text-[10.5px] uppercase leading-snug tracking-[0.08em] text-text transition-colors hover:border-sense-pale disabled:opacity-50"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* 2. Tool Control + Dynamic Master Switches + Per-Tool Selection */}
          <div className="border-t border-line pt-5">
            <div className="mb-3 grid gap-2 @xl:grid-cols-2">
              <button
                type="button"
                role="switch"
                aria-checked={toolControlOn}
                disabled={busy}
                onClick={() => void toggleToolControl()}
                title={`OFF uses default model settings. ON lets you select which of the ${total > 0 ? `${total} ` : ""}tools to include or turn off.`}
                className={`press flex items-center justify-between gap-3 border px-3 py-2 text-left transition-colors ${
                  toolControlOn ? "border-sense-pale bg-sense-pale/[0.06]" : "border-line hover:border-line-strong"
                }`}
              >
                <div className="min-w-0">
                  <div className="mono text-xs uppercase tracking-[0.08em] text-text">Tool Control</div>
                </div>
                <span
                  className={`mono shrink-0 border px-2.5 py-1 text-[10px] uppercase tracking-[0.1em] ${
                    toolControlOn ? "border-verified text-verified" : "border-line text-faint"
                  }`}
                >
                  {toolControlOn ? "ON" : "OFF"}
                </span>
              </button>

              <button
                type="button"
                role="switch"
                aria-checked={dynamicOn}
                disabled={busy}
                onClick={() => void toggleDynamic()}
                title="Works alone or alongside Tool Control: activates extra tools into tools/list on demand when needed."
                className={`press flex items-center justify-between gap-3 border px-3 py-2 text-left transition-colors ${
                  dynamicOn ? "border-sense-pale bg-sense-pale/[0.06]" : "border-line hover:border-line-strong"
                }`}
              >
                <div className="min-w-0">
                  <div className="mono text-xs uppercase tracking-[0.08em] text-text">Dynamic Tool Activation</div>
                </div>
                <span
                  className={`mono shrink-0 border px-2.5 py-1 text-[10px] uppercase tracking-[0.1em] ${
                    dynamicOn ? "border-verified text-verified" : "border-line text-faint"
                  }`}
                >
                  {dynamicOn ? "ON" : "OFF"}
                </span>
              </button>
            </div>

            <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border border-line/70 px-3 py-2">
              <div className="mono min-w-0 text-[11px] text-muted">{activeModeBanner}</div>
              {toolControlOn && (
                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void setQuickTools(tools.slice(0, 5))}
                    className="press mono min-h-7 border border-line px-2.5 text-[10px] uppercase tracking-[0.08em] text-muted hover:border-line-strong hover:text-text"
                  >
                    Core 5
                  </button>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void setQuickTools(tools)}
                    className="press mono min-h-7 border border-line px-2.5 text-[10px] uppercase tracking-[0.08em] text-muted hover:border-line-strong hover:text-text"
                  >
                    All {total} ON
                  </button>
                </div>
              )}
            </div>

            <div className={`grid gap-3 @3xl:grid-cols-2 transition-opacity ${toolControlOn ? "" : "opacity-55"}`}>
              {total === 0 && (
                <div className="mono text-xs text-faint">The engine did not report its MCP tool list.</div>
              )}
              {TOOL_GROUPS.map((grp) => {
                const items = catalog.filter((t) => t.group === grp);
                if (items.length === 0) return null;
                const onCount = items.filter((t) => enabled.has(t.name)).length;
                return (
                  <div key={grp} className="border border-line/80 p-2.5">
                    <div className="eyebrow mb-2 flex items-center justify-between">
                      <span>{grp}</span>
                      <span className="mono text-[11px] text-faint">
                        {onCount} / {items.length} ON
                      </span>
                    </div>
                    <div className="grid gap-1">
                      {items.map((tool) => {
                        const isOn = enabled.has(tool.name);
                        return (
                          <button
                            key={tool.name}
                            type="button"
                            role="switch"
                            aria-checked={isOn}
                            disabled={busy || !toolControlOn}
                            onClick={() => void toggleTool(tool.name)}
                            title={tool.desc ? `${tool.name}: ${tool.desc}` : tool.name}
                            className={`press flex items-center justify-between gap-3 border px-2.5 py-1.5 text-left transition-colors disabled:cursor-not-allowed ${
                              isOn
                                ? "border-sense-pale/70 bg-sense-pale/[0.04]"
                                : "border-line/60 opacity-75 hover:border-line-strong hover:opacity-100"
                            }`}
                          >
                            <div className="min-w-0">
                              <div className="flex flex-wrap items-center gap-2">
                                <span className={`mono text-xs ${isOn ? "text-text" : "text-muted"}`}>{tool.name}</span>
                                {tool.estTokens !== undefined && (
                                  <span className="mono text-[10px] text-quiet" title="Estimated schema size, not measured">
                                    about {tool.estTokens} tok, est.
                                  </span>
                                )}
                              </div>
                            </div>
                            <span
                              className={`mono shrink-0 border px-2 py-0.5 text-[10px] uppercase tracking-[0.1em] ${
                                isOn ? "border-verified text-verified" : "border-line text-faint"
                              }`}
                            >
                              {isOn ? "ON" : "OFF"}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 3. Granular Lean Toggles & Budgeting */}
          <div className="grid gap-6 border-t border-line pt-5 @3xl:grid-cols-2">
            <div>
              <div className="eyebrow mb-2">Lean behavior switches</div>
              <div className="grid gap-2">
                {toggleKeys.map((item) => {
                  const c = byKey.get(item.key);
                  if (!c) return null;
                  const isOn = c.value === "on";
                  return (
                    <button
                      key={item.key}
                      type="button"
                      role="switch"
                      aria-checked={isOn}
                      disabled={busy}
                      onClick={() => void saveOne(item.key, isOn ? "off" : "on")}
                      title={item.sub}
                      className={`press flex items-center justify-between gap-3 border px-3 py-2 text-left ${
                        isOn ? "border-sense-pale/70" : "border-line hover:border-line-strong"
                      }`}
                    >
                      <div className="min-w-0">
                        <div className="mono text-xs text-text">{item.title}</div>
                      </div>
                      <span
                        className={`mono shrink-0 border px-2 py-0.5 text-[10px] uppercase tracking-[0.1em] ${
                          isOn ? "border-verified text-verified" : "border-line text-faint"
                        }`}
                      >
                        {isOn ? "ON" : "OFF"}
                      </span>
                    </button>
                  );
                })}
                {earlyDrop && (
                  <div className="border border-line px-3 py-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="mono text-xs text-text">Reranker early drop</span>
                      <span className="mono text-[11px] text-muted">{earlyDrop.value}</span>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {earlyDrop.values.map((v) => (
                        <button
                          key={v}
                          type="button"
                          disabled={busy}
                          onClick={() => void saveOne(earlyDrop.key, v)}
                          className={`press mono min-h-7 border px-2.5 text-[10px] uppercase tracking-[0.08em] ${
                            v === earlyDrop.value
                              ? "border-sense-pale text-text"
                              : "border-line text-faint hover:border-line-strong hover:text-text"
                          }`}
                        >
                          {v}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                {quotaAction && (
                  <div className="border border-line px-3 py-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="mono text-xs text-text">Quota limit action</span>
                      <span className="mono text-[11px] text-muted">{quotaAction.value}</span>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {quotaAction.values.map((v) => (
                        <button
                          key={v}
                          type="button"
                          disabled={busy}
                          onClick={() => void saveOne(quotaAction.key, v)}
                          className={`press mono min-h-7 border px-2.5 text-[10px] uppercase tracking-[0.08em] ${
                            v === quotaAction.value
                              ? "border-sense-pale text-text"
                              : "border-line text-faint hover:border-line-strong hover:text-text"
                          }`}
                        >
                          {v}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div>
              <div className="eyebrow mb-2">Token budgeting & quotas</div>
              <div className="border border-line/80 px-3">
                {budgetKeys.map((k) => {
                  const c = byKey.get(k);
                  return c ? <BudgetQuickRow key={k} control={c} busy={busy} onSave={saveOne} /> : null;
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}

/** The reference for the rail: what each profile sets, and what each of the engine's MCP tools
 *  does (from the list the engine reports; a tool this page has no words for is named only). */
export function LeanModeGuide() {
  const { data: rawControls } = useApi<Control[] | { controls?: Control[] }>("/api/controls", 15000);
  const byKey = useMemo(() => {
    const map = new Map<string, Control>();
    const list = Array.isArray(rawControls) ? rawControls : rawControls?.controls ?? [];
    for (const c of list) map.set(c.key, c);
    return map;
  }, [rawControls]);
  const tools = mcpToolList(byKey);
  const total = tools.length;
  const full = fullToolCount(byKey, total);
  const standard = Number(byKey.get("mcp.tool_count")?.default) || full;
  const presets = buildPresets(full, total, standard);
  const catalog = toolCatalog(tools);
  return (
    <div className="grid gap-0">
      <RailSection title="Profiles">
        <dl className="grid gap-2 text-[11px]">
          {presets.map((p) => (
            <div key={p.id} className="grid gap-0.5">
              <dt className="mono uppercase tracking-[0.08em] text-muted">{p.label}</dt>
              <dd className="read leading-relaxed text-faint">{p.summary}</dd>
            </div>
          ))}
        </dl>
      </RailSection>
      <RailSection title="Switches">
        <ul className="read grid gap-1.5 text-[11px] text-faint">
          <li>Tool Control: OFF uses default model settings; ON lets you pick exactly which tools are included.</li>
          <li>Dynamic Tool Activation: works alone or with Tool Control; activates extra tools into tools/list when a call needs them.</li>
          <li>Token estimates are typed-in guesses of each tool&apos;s schema size, not measured by the engine.</li>
        </ul>
      </RailSection>
      <RailSection title={`Tools${total ? ` · ${total}` : ""}`}>
        {total === 0 ? (
          <div className="mono text-[11px] text-faint">The engine did not report its MCP tool list.</div>
        ) : (
          <div className="grid gap-3">
            {TOOL_GROUPS.map((g) => {
              const items = catalog.filter((t) => t.group === g);
              if (!items.length) return null;
              return (
                <div key={g} className="grid gap-1">
                  <div className="eyebrow text-quiet">{g}</div>
                  <dl className="grid gap-1 text-[11px]">
                    {items.map((t) => (
                      <div key={t.name} className="grid gap-0">
                        <dt className="mono text-muted">{t.name}</dt>
                        {t.desc && <dd className="read text-faint">{t.desc}</dd>}
                      </div>
                    ))}
                  </dl>
                </div>
              );
            })}
          </div>
        )}
      </RailSection>
    </div>
  );
}
