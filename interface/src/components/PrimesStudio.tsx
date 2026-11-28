"use client";

import Link from "next/link";
import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { NoteLine } from "@/components/Controls";
import DeleteFlow from "@/components/DeleteFlow";
import ProjectMenu from "@/components/ProjectMenu";
import { Empty, Pill, Placeholder, Segmented, Switch } from "@/components/ui";
import { RailSection } from "@/components/Workspace";
import { ago, post, projectLabelByKey, refreshAll, useApi, type BuiltinPrime, type CustomPrime, type Ledger, type PrimePreset, type PrimesReply } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { primeCounts, useProjectKey } from "@/lib/evidence";

type Note = { text: string; tone: "verified" | "error" };

const DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"] as const;

const TRIGGER_LABELS: Record<string, string> = {
  "session-start": "Session start",
  prompt: "Every user prompt",
  compact: "After compaction",
  stop: "Before stop",
  edit: "Before file edit (PreToolUse)",
  failure: "On build/test failure",
  tools: "After tool call (PostToolUse)",
  "error-repeats": "On repeating error",
  interval: "Timed interval (minutes)",
  schedule: "Scheduled clock window (UTC)",
};

type FormState = {
  id: string | null;
  name: string;
  text: string;
  trigger: string;
  every: string;
  intervalMin: string;
  days: string[];
  after: string;
  before: string;
  repeat: string;
  max: string;
  glob: string;
  error: string;
  errorCount: string;
  loopId: string;
  enabled: boolean;
};

const EMPTY_FORM: FormState = {
  id: null,
  name: "",
  text: "",
  trigger: "interval",
  every: "1",
  intervalMin: "30",
  days: [],
  after: "",
  before: "",
  repeat: "always",
  max: "3",
  glob: "",
  error: "",
  errorCount: "2",
  loopId: "",
  enabled: false,
};

function formFromPrime(c: CustomPrime): FormState {
  return {
    id: c.id,
    name: c.name,
    text: c.text,
    trigger: c.trigger,
    every: String(c.every ?? 1),
    intervalMin: String(c.interval_min ?? 0),
    days: c.schedule?.days ?? [],
    after: c.schedule?.after ?? "",
    before: c.schedule?.before ?? "",
    repeat: c.repeat ?? "always",
    max: String(c.max ?? 3),
    glob: c.glob ?? "",
    error: c.error ?? "",
    errorCount: String(c.error_count ?? 2),
    loopId: c.loop_id ?? "",
    enabled: c.enabled,
  };
}

const BUILTIN_TEMPLATES: Record<
  string,
  { trigger: string; intervalMin: string; repeat: string; max: string; glob: string; text: string }
> = {
  "prompt-recall": {
    trigger: "prompt",
    intervalMin: "0",
    repeat: "once",
    max: "1",
    glob: "",
    text: "Before searching the codebase broadly, call nm_context_pack to check verified past answers for this topic and confirm cited line ranges are still fresh on disk.",
  },
  "failure-fix": {
    trigger: "failure",
    intervalMin: "2",
    repeat: "always",
    max: "5",
    glob: "",
    text: "A build or test check just failed. Inspect the exact error location, check prior verified fixes for this error signature, and resolve the root cause rather than bypassing the check.",
  },
  "edit-impact": {
    trigger: "edit",
    intervalMin: "0",
    repeat: "always",
    max: "5",
    glob: "",
    text: "Before editing this file, check downstream callers and imports with nm_deps or nm_symbol_find, and verify whether these lines hold a previously verified fix or contract.",
  },
  resume: {
    trigger: "compact",
    intervalMin: "0",
    repeat: "always",
    max: "5",
    glob: "",
    text: "Context was just compacted. Re-anchor on the active goal, review modified files and failing checks from this session, and continue without repeating completed work.",
  },
  core: {
    trigger: "session-start",
    intervalMin: "0",
    repeat: "once",
    max: "1",
    glob: "",
    text: "Session started. Load this project's 1-core architectural invariants, build commands, and safety boundaries before planning edits.",
  },
  "open-loop": {
    trigger: "stop",
    intervalMin: "0",
    repeat: "always",
    max: "5",
    glob: "",
    text: "Before finishing this turn, verify that all modified files compile cleanly, relevant tests pass, and no temporary debug code or unfinished work remains.",
  },
};

function formFromBuiltin(b: BuiltinPrime): FormState {
  const tpl = BUILTIN_TEMPLATES[b.id];
  return {
    ...EMPTY_FORM,
    id: null,
    name: b.label,
    trigger: tpl?.trigger ?? "prompt",
    intervalMin: tpl?.intervalMin ?? "0",
    repeat: tpl?.repeat ?? "always",
    max: tpl?.max ?? "3",
    glob: tpl?.glob ?? "",
    text: tpl?.text ?? b.summary,
    enabled: false,
  };
}

function formFromPreset(pr: PrimePreset): FormState {
  const first = pr.custom[0];
  if (first) {
    return {
      ...formFromPrime(first),
      id: null,
      // Preloaded presets always default to OFF until user explicitly toggles them on.
      enabled: false,
    };
  }
  return {
    ...EMPTY_FORM,
    name: pr.name,
    text: pr.description,
    enabled: false,
  };
}

function scheduleSummary(c: CustomPrime): string | null {
  const sc = c.schedule;
  if (!sc) return null;
  const parts: string[] = [];
  if (sc.days && sc.days.length > 0) parts.push(sc.days.map((d) => d.toUpperCase()).join(" "));
  if (sc.after || sc.before) parts.push(`${sc.after || "00:00"}-${sc.before || "23:59"} UTC`);
  return parts.length > 0 ? parts.join(" · ") : null;
}

/** Bold headline cadence/trigger string for an alarm-style prime row. */
function headlineOf(c: CustomPrime): string {
  const sched = scheduleSummary(c);
  if (c.trigger === "interval") {
    return `Every ${c.interval_min || 15}m`;
  }
  if (c.trigger === "schedule") {
    return sched ?? "Scheduled UTC";
  }
  if (c.trigger === "error-repeats") {
    return `Error ×${c.error_count || 2}`;
  }
  const base = TRIGGER_LABELS[c.trigger] ?? c.trigger;
  if (c.every > 1) return `${base} · every ${c.every}`;
  if (c.interval_min > 0) return `${base} · ${c.interval_min}m cooldown`;
  return base;
}

function refreshPrimesAll() {
  refreshAll("/api/primes");
  refreshAll("/api/controls");
  refreshAll("/api/audit");
  refreshAll("/api/ledger");
}

const pinnedSubs = new Set<() => void>();
function readPinnedRaw(projectKey: string): string {
  if (typeof window === "undefined") return "[]";
  try {
    return window.localStorage.getItem(`nano.primes.pinned.${projectKey}`) ?? "[]";
  } catch {
    return "[]";
  }
}
function writePinnedRaw(projectKey: string, ids: string[]) {
  const uniq = [...new Set(ids)];
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(`nano.primes.pinned.${projectKey}`, JSON.stringify(uniq));
    } catch {
      // ignore storage errors
    }
  }
  pinnedSubs.forEach((f) => f());
}
function usePinnedBuiltins(projectKey: string): [string[], (ids: string[]) => void] {
  const raw = useSyncExternalStore(
    (cb) => {
      pinnedSubs.add(cb);
      const onStorage = (e: StorageEvent) => {
        if (e.key === `nano.primes.pinned.${projectKey}`) cb();
      };
      if (typeof window !== "undefined") window.addEventListener("storage", onStorage);
      return () => {
        pinnedSubs.delete(cb);
        if (typeof window !== "undefined") window.removeEventListener("storage", onStorage);
      };
    },
    () => readPinnedRaw(projectKey),
    () => "[]",
  );
  const list = useMemo(() => {
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === "string") : [];
    } catch {
      return [];
    }
  }, [raw]);
  return [list, (ids: string[]) => writePinnedRaw(projectKey, ids)];
}

export default function PrimesStudio({ onBack }: { onBack?: () => void }) {
  const { key: globalKey, projects } = useProjectKey();
  const projList = useMemo(
    () =>
      [...(projects?.projects ?? [])].sort((a, b) => {
        if (Boolean(a.serving) !== Boolean(b.serving)) return a.serving ? -1 : 1;
        return (b.last_ms ?? 0) - (a.last_ms ?? 0);
      }),
    [projects],
  );
  const projectKey = globalKey ?? projList[0]?.key ?? "nano";
  const q = useApi<PrimesReply>(`/api/primes?project=${encodeURIComponent(projectKey)}`, 5000);
  const d = q.data;
  const now = useNow();

  const [note, setNote] = useState<Note | null>(null);
  const [busy, setBusy] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [presetPickerOpen, setPresetPickerOpen] = useState(false);
  const [presetTab, setPresetTab] = useState<"verified" | "schedule">("verified");
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [saveAsPreset, setSaveAsPreset] = useState(false);
  const [presetName, setPresetName] = useState("");
  const [presetDesc, setPresetDesc] = useState("");
  const [pinnedList, updatePinned] = usePinnedBuiltins(projectKey);

  useEffect(() => {
    if (!presetPickerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPresetPickerOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [presetPickerOpen]);

  const openAddSheet = () => {
    setForm(EMPTY_FORM);
    setSelectedPreset(null);
    setPresetPickerOpen(false);
    setSaveAsPreset(false);
    setSheetOpen(true);
    setNote(null);
  };

  const openEditSheet = (c: CustomPrime) => {
    setForm(formFromPrime(c));
    setSelectedPreset(null);
    setPresetPickerOpen(false);
    setSaveAsPreset(false);
    setSheetOpen(true);
    setNote(null);
  };

  const setMode = async (mode: string) => {
    setBusy(true);
    try {
      await post("/api/controls", { key: "priming", value: mode });
      setNote({ text: `Master priming set to "${mode}". Logged in Audit with Undo.`, tone: "verified" });
      refreshPrimesAll();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const turnAllOff = async () => {
    setBusy(true);
    try {
      await post("/api/primes", { action: "all_off", project: projectKey });
      setNote({ text: "Turned priming off and switched every prime off. Undoable in Audit.", tone: "verified" });
      refreshPrimesAll();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const toggleBuiltin = async (id: string, enabled: boolean) => {
    // Keep this verified prime visible in the user's list even when switched OFF (only Remove removes it).
    if (!pinnedList.includes(id)) updatePinned([...pinnedList, id]);
    setBusy(true);
    try {
      await post("/api/primes", { action: "toggle_builtin", project: projectKey, id, enabled });
      setNote({ text: `Verified prime "${id}" switched ${enabled ? "ON" : "OFF"}. Undoable in Audit.`, tone: "verified" });
      refreshPrimesAll();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const removeBuiltinFromList = async (b: BuiltinPrime) => {
    updatePinned(pinnedList.filter((x) => x !== b.id));
    if (b.enabled) {
      setBusy(true);
      try {
        await post("/api/primes", { action: "toggle_builtin", project: projectKey, id: b.id, enabled: false });
        setNote({ text: `Removed "${b.label}" from your primes list. Undoable in Audit.`, tone: "verified" });
        refreshPrimesAll();
      } catch (e) {
        setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
      }
      setBusy(false);
    } else {
      setNote({ text: `Removed "${b.label}" from your primes list.`, tone: "verified" });
    }
  };

  const toggleCustom = async (id: string, enabled: boolean) => {
    setBusy(true);
    try {
      await post("/api/primes", { action: "toggle", project: projectKey, id, enabled });
      setNote({ text: `Prime switched ${enabled ? "ON" : "OFF"}. Undoable in Audit.`, tone: "verified" });
      refreshPrimesAll();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const submitForm = async () => {
    const name = form.name.trim();
    const text = form.text.trim();
    if (!name || !text) return;
    setBusy(true);
    try {
      const payload: Record<string, unknown> = {
        action: form.id ? "edit" : "add",
        project: projectKey,
        name,
        text,
        trigger: form.trigger,
        every: Math.max(1, Number.parseInt(form.every || "1", 10) || 1),
        interval_min: Math.max(0, Number.parseInt(form.intervalMin || "0", 10) || 0),
        repeat: form.repeat,
        max: Math.max(1, Number.parseInt(form.max || "3", 10) || 3),
        glob: form.glob.trim(),
        error: form.error.trim(),
        error_count: Math.min(5, Math.max(2, Number.parseInt(form.errorCount || "2", 10) || 2)),
        loop_id: form.loopId.trim(),
        enabled: form.enabled,
      };
      if (form.id) payload.id = form.id;
      if (form.days.length > 0 || form.after.trim() || form.before.trim()) {
        payload.schedule = {
          days: form.days,
          after: form.after.trim(),
          before: form.before.trim(),
        };
      } else {
        payload.schedule = {};
      }
      await post("/api/primes", payload);
      if (saveAsPreset && presetName.trim()) {
        await post("/api/primes", {
          action: "preset_save",
          project: projectKey,
          name: presetName.trim(),
          description: presetDesc.trim() || `Saved preset with "${name}"`,
        });
      }
      setSheetOpen(false);
      setForm(EMPTY_FORM);
      setSelectedPreset(null);
      setSaveAsPreset(false);
      setPresetName("");
      setPresetDesc("");
      setNote({
        text: form.id
          ? `Saved changes to "${name}". Undoable in Audit.`
          : `Added "${name}" (${form.enabled ? "ON" : "OFF"}). Undoable in Audit.`,
        tone: "verified",
      });
      refreshPrimesAll();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const applySavedPreset = async (pr: PrimePreset) => {
    setBusy(true);
    try {
      await post("/api/primes", { action: "preset_apply", project: projectKey, name: pr.name });
      setSheetOpen(false);
      setNote({
        text: pr.builtin
          ? `Added preloaded preset "${pr.name}" (switched OFF until you toggle it on).`
          : `Applied saved preset "${pr.name}". Undoable in Audit.`,
        tone: "verified",
      });
      refreshPrimesAll();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const builtins = d?.builtins ?? [];
  const custom = d?.custom ?? [];
  const presets = d?.presets ?? [];
  const loops = d?.loops ?? [];

  const activeBuiltins = builtins.filter((b) => b.enabled).length;
  const activeCustom = custom.filter((c) => c.enabled).length;
  const anyArmed = d?.mode !== "off" || activeBuiltins > 0 || activeCustom > 0;
  const pinnedIds = new Set(pinnedList);
  const listedBuiltins = builtins.filter((b) => b.enabled || pinnedIds.has(b.id));

  return (
    <div className="grid min-w-0 gap-5" data-e2e="primes-studio">
      {/* Top Action Bar: Back + Project Dropdown (with built-in Add/Edit/Delete) + Master Mode + "+ Add Prime" */}
      <div className="flex flex-col gap-3 border-b border-line pb-4 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
        <div className="flex min-w-0 flex-wrap items-center gap-3">
          {onBack && (
            <button
              type="button"
              onClick={onBack}
              data-e2e="primes-back-to-log"
              className="press mono h-9 shrink-0 border border-line-strong px-2.5 text-[10.5px] uppercase tracking-[0.12em] text-muted hover:border-sense-pale hover:text-text"
            >
              &larr; Audit Log
            </button>
          )}
          <ProjectMenu placement="down" label="project" />
        </div>

        <div className="flex min-w-0 flex-wrap items-center gap-2.5 sm:gap-3">
          <div className="w-full sm:w-44">
            <Segmented
              values={["off", "quiet", "on"]}
              value={d?.mode ?? "off"}
              label="Master priming mode"
              onPick={(v) => void setMode(v)}
            />
          </div>
          {anyArmed && (
            <button
              type="button"
              disabled={busy}
              onClick={() => void turnAllOff()}
              data-e2e="primes-all-off"
              className="press mono h-9 shrink-0 border border-line-strong px-2.5 text-[10px] uppercase tracking-[0.12em] text-faint hover:border-error hover:text-error disabled:opacity-40"
            >
              All off
            </button>
          )}
          <button
            type="button"
            onClick={() => (sheetOpen ? setSheetOpen(false) : openAddSheet())}
            data-e2e="primes-new-custom"
            className="press mono flex h-9 flex-1 items-center justify-center gap-1.5 border border-sense-pale bg-sense/20 px-3.5 text-[11px] font-medium uppercase tracking-[0.12em] text-text hover:bg-sense/35 sm:flex-none"
          >
            <span className="text-[14px] leading-none text-sense-pale">{sheetOpen ? "×" : "+"}</span>
            <span>{sheetOpen ? "Close" : "Add Prime"}</span>
          </button>
        </div>
      </div>

      <NoteLine note={note} />

      {/* Add / Edit Prime Sheet (Apple Alarms "+ Add" workflow: choose/customize preset or create custom) */}
      {sheetOpen && d && (
        <div
          className="grid min-w-0 gap-4 border border-sense-pale/70 bg-bg-1/90 p-3.5 shadow-[0_0_24px_rgb(15_103_245/0.12)] sm:p-5"
          data-e2e="prime-custom-form"
        >
          {/* Sheet Header: Cancel (left) · Title (center) · Confirm Save/Add (right) */}
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line pb-3">
            <button
              type="button"
              onClick={() => {
                setSheetOpen(false);
                setForm(EMPTY_FORM);
                setSelectedPreset(null);
              }}
              className="press mono h-8 shrink-0 border border-line-strong px-2 text-[10px] uppercase tracking-[0.1em] text-muted hover:border-text hover:text-text sm:px-2.5 sm:text-[10.5px]"
            >
              × Cancel
            </button>
            <div className="mono text-[11px] font-medium uppercase tracking-[0.16em] text-text sm:text-[12px] sm:tracking-[0.18em]">
              {form.id ? "Edit Prime" : "Add Prime"}
            </div>
            <button
              type="button"
              disabled={busy || !form.name.trim() || !form.text.trim()}
              onClick={() => void submitForm()}
              data-e2e="prime-form-submit"
              className="press mono h-8 shrink-0 border border-sense-pale bg-sense/25 px-2.5 text-[10px] font-medium uppercase tracking-[0.1em] text-text hover:bg-sense/40 disabled:opacity-40 sm:px-3 sm:text-[10.5px]"
            >
              {busy ? "Saving..." : form.id ? "✓ Save" : "✓ Add Prime"}
            </button>
          </div>

          {/* Compact Preset Selector Bar inside "+ Add Prime" (opens organized pop-up menu) */}
          {!form.id && (
            <div className="flex flex-wrap items-center justify-between gap-2.5 border-b border-line pb-3.5">
              <div className="min-w-0">
                <div className="mono text-[10px] uppercase tracking-[0.14em] text-faint">Preset</div>
                <div className="mono mt-0.5 truncate text-[12px] text-text">
                  {selectedPreset
                    ? (builtins.find((b) => b.id === selectedPreset)?.label ?? selectedPreset)
                    : "Custom (blank)"}
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-2.5">
                {selectedPreset && (
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedPreset(null);
                      setForm(EMPTY_FORM);
                    }}
                    className="press mono text-[10px] uppercase tracking-[0.12em] text-faint hover:text-text"
                  >
                    Clear
                  </button>
                )}
                <button
                  type="button"
                  data-e2e="preset-picker-open"
                  onClick={() => setPresetPickerOpen(true)}
                  className="press mono inline-flex h-8 items-center gap-1.5 border border-sense-pale/70 bg-bg-2/60 px-3 text-[10.5px] uppercase tracking-[0.12em] text-text hover:border-sense-pale hover:bg-sense/20"
                >
                  <span>Browse presets</span>
                  <span className="text-sense-pale">({builtins.length + presets.length})</span>
                  <svg aria-hidden viewBox="0 0 16 16" className="h-3 w-3 shrink-0 text-faint">
                    <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.2" />
                  </svg>
                </button>
              </div>
            </div>
          )}

          {/* Organized Presets Pop-Up Menu (tabs for Verified presets vs Prompt & schedule presets) */}
          {!form.id && presetPickerOpen && (
            <div
              role="dialog"
              aria-modal="true"
              aria-label="Choose a prime preset"
              data-e2e="preset-picker-modal"
              className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6"
            >
              <button
                type="button"
                aria-label="Close presets menu"
                onClick={() => setPresetPickerOpen(false)}
                className="fixed inset-0 bg-bg/75 backdrop-blur-xs"
              />
              <div className="relative z-10 flex max-h-[84dvh] w-full max-w-2xl flex-col border border-sense-pale/70 bg-bg-1 shadow-[0_18px_48px_rgb(0_0_0/0.75)]">
                {/* Modal Header */}
                <div className="flex shrink-0 items-center justify-between gap-3 border-b border-line px-4 py-3 sm:px-5">
                  <span className="mono text-[11.5px] font-medium uppercase tracking-[0.16em] text-text">
                    Choose a preset
                  </span>
                  <button
                    type="button"
                    onClick={() => setPresetPickerOpen(false)}
                    className="press mono border border-line-strong px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-muted hover:border-text hover:text-text"
                  >
                    × Close
                  </button>
                </div>

                {/* Modal Tabs */}
                <div className="flex shrink-0 gap-1 border-b border-line bg-bg-0/40 px-4 sm:px-5" role="tablist">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={presetTab === "verified"}
                    data-e2e="preset-tab-verified"
                    onClick={() => setPresetTab("verified")}
                    className={`press mono relative inline-flex min-h-10 items-center gap-1.5 px-2.5 text-[10.5px] uppercase tracking-[0.12em] ${
                      presetTab === "verified" ? "text-text" : "text-faint hover:text-text"
                    }`}
                  >
                    <span>Verified presets</span>
                    <span className="text-quiet">{builtins.length}</span>
                    {presetTab === "verified" && (
                      <span aria-hidden className="absolute right-2 bottom-0 left-2 h-[2px] bg-sense-pale" />
                    )}
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={presetTab === "schedule"}
                    data-e2e="preset-tab-schedule"
                    onClick={() => setPresetTab("schedule")}
                    className={`press mono relative inline-flex min-h-10 items-center gap-1.5 px-2.5 text-[10.5px] uppercase tracking-[0.12em] ${
                      presetTab === "schedule" ? "text-text" : "text-faint hover:text-text"
                    }`}
                  >
                    <span>Prompt &amp; schedule</span>
                    <span className="text-quiet">{presets.length}</span>
                    {presetTab === "schedule" && (
                      <span aria-hidden className="absolute right-2 bottom-0 left-2 h-[2px] bg-sense-pale" />
                    )}
                  </button>
                </div>

                {/* Modal Body */}
                <div className="min-h-0 flex-1 overflow-y-auto p-4 sm:p-5">
                  {presetTab === "verified" ? (
                    <ul className="grid gap-2">
                      {builtins.map((b) => {
                        const picked = selectedPreset === b.id;
                        return (
                          <li
                            key={b.id}
                            className={`flex min-w-0 items-center justify-between gap-3 border p-3 ${
                              picked
                                ? "border-sense-pale bg-bg-2/75"
                                : "border-line bg-bg-0/50 hover:border-sense-pale/70"
                            }`}
                            data-e2e="builtin-prime"
                            data-rule={b.id}
                          >
                            <button
                              type="button"
                              onClick={() => {
                                setSelectedPreset(b.id);
                                setForm(formFromBuiltin(b));
                                setPresetPickerOpen(false);
                              }}
                              className="press min-w-0 flex-1 text-left"
                            >
                              <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5">
                                <span className="mono text-[12px] font-medium text-text">{b.label}</span>
                                <span className="mono text-[10px] text-sense-pale">{b.trigger}</span>
                              </div>
                              <p className="read mt-0.5 text-[11px] leading-relaxed text-faint">{b.summary}</p>
                            </button>
                            <div className="flex shrink-0 items-center gap-2">
                              <Pill tone="verified">verified</Pill>
                              <button
                                type="button"
                                onClick={() => {
                                  setSelectedPreset(b.id);
                                  setForm(formFromBuiltin(b));
                                  setPresetPickerOpen(false);
                                }}
                                className="press mono border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.1em] text-sense-pale hover:text-text"
                              >
                                Use
                              </button>
                            </div>
                          </li>
                        );
                      })}
                    </ul>
                  ) : (
                    <ul className="grid gap-2">
                      {presets.map((pr) => {
                        const picked = selectedPreset === pr.name;
                        return (
                          <li
                            key={pr.name}
                            className={`flex min-w-0 flex-wrap items-start justify-between gap-2.5 border p-3 sm:flex-nowrap sm:items-center ${
                              picked ? "border-sense-pale bg-bg-2/75" : "border-line bg-bg-0/50 hover:border-line-strong"
                            }`}
                            data-e2e="prime-preset"
                            data-preset={pr.name}
                          >
                            <button
                              type="button"
                              onClick={() => {
                                setSelectedPreset(pr.name);
                                setForm(formFromPreset(pr));
                                setPresetPickerOpen(false);
                              }}
                              className="press min-w-0 flex-1 text-left"
                            >
                              <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5">
                                <span className="mono text-[12px] font-medium text-text">{pr.name}</span>
                                {pr.custom[0] && (
                                  <span className="mono text-[10px] text-sense-pale">{headlineOf(pr.custom[0])}</span>
                                )}
                              </div>
                              <p className="read mt-0.5 text-[11px] leading-relaxed text-faint">{pr.description}</p>
                            </button>
                            <div className="flex shrink-0 items-center gap-2">
                              <Pill tone={pr.builtin ? "muted" : "signal"}>{pr.builtin ? "preset" : "saved"}</Pill>
                              <button
                                type="button"
                                onClick={() => {
                                  setSelectedPreset(pr.name);
                                  setForm(formFromPreset(pr));
                                  setPresetPickerOpen(false);
                                }}
                                className="press mono border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.1em] text-sense-pale hover:text-text"
                              >
                                Use
                              </button>
                              {!pr.builtin && (
                                <button
                                  type="button"
                                  disabled={busy}
                                  onClick={() => {
                                    setPresetPickerOpen(false);
                                    void applySavedPreset(pr);
                                  }}
                                  className="press mono border-b border-line-strong px-1 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text"
                                >
                                  Apply
                                </button>
                              )}
                              {!pr.builtin && (
                                <DeleteFlow
                                  path="/api/primes"
                                  body={{ action: "preset_delete", name: pr.name }}
                                  label="×"
                                  done={() => `Deleted saved preset "${pr.name}".`}
                                  refresh={["/api/primes", "/api/audit"]}
                                  onNote={setNote}
                                  startClass="press mono px-1 text-[11px] text-faint hover:text-error"
                                />
                              )}
                            </div>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Structured Alarm-style Configuration Rows */}
          <form
            className="grid gap-3.5"
            onSubmit={(e) => {
              e.preventDefault();
              void submitForm();
            }}
          >
            <div className="grid gap-3 md:grid-cols-2">
              <label className="grid gap-1">
                <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">Label / Name</span>
                <input
                  value={form.name}
                  onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                  placeholder="e.g. 30m verification check-in"
                  maxLength={80}
                  required
                  className="mono h-9 w-full border-b border-line-strong bg-transparent px-1 text-[12px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
                />
              </label>

              <label className="relative grid gap-1">
                <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">Trigger</span>
                <select
                  value={form.trigger}
                  onChange={(e) => setForm((f) => ({ ...f, trigger: e.target.value }))}
                  className="press mono h-9 w-full appearance-none border-b border-line-strong bg-transparent pr-7 pl-1 text-[12px] text-text hover:border-text focus-visible:border-sense-pale"
                >
                  {(d.triggers ?? Object.keys(TRIGGER_LABELS)).map((t) => (
                    <option key={t} value={t} className="bg-bg-1 text-text">
                      {TRIGGER_LABELS[t] ?? t}
                    </option>
                  ))}
                </select>
                <svg aria-hidden viewBox="0 0 16 16" className="pointer-events-none absolute right-2 bottom-3 h-3 w-3 text-faint">
                  <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.2" />
                </svg>
              </label>
            </div>

            <label className="grid gap-1">
              <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">Prompt / Nudge Text</span>
              <textarea
                rows={3}
                value={form.text}
                onChange={(e) => setForm((f) => ({ ...f, text: e.target.value }))}
                placeholder="Write or edit the exact reminder injected into your coding agent when this prime fires..."
                maxLength={2000}
                required
                className="mono w-full border border-line bg-bg-0/70 p-2.5 text-[11.5px] leading-relaxed text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
              />
            </label>

            {/* Timing, Interval, Repeat */}
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <label className="grid gap-1">
                <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">
                  {form.trigger === "interval" ? "Interval (minutes)" : "Cooldown (minutes)"}
                </span>
                <input
                  type="number"
                  min={0}
                  max={1440}
                  value={form.intervalMin}
                  onChange={(e) => setForm((f) => ({ ...f, intervalMin: e.target.value }))}
                  className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11.5px] text-text outline-none focus-visible:border-sense-pale"
                />
              </label>
              <label className="grid gap-1">
                <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">Every N events</span>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={form.every}
                  onChange={(e) => setForm((f) => ({ ...f, every: e.target.value }))}
                  className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11.5px] text-text outline-none focus-visible:border-sense-pale"
                />
              </label>
              <label className="relative grid gap-1">
                <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">Repeat</span>
                <select
                  value={form.repeat}
                  onChange={(e) => setForm((f) => ({ ...f, repeat: e.target.value }))}
                  className="press mono h-8 w-full appearance-none border-b border-line-strong bg-transparent pr-6 pl-1 text-[11.5px] text-text focus-visible:border-sense-pale"
                >
                  <option value="always" className="bg-bg-1 text-text">Always</option>
                  <option value="once" className="bg-bg-1 text-text">Once per session</option>
                  <option value="max" className="bg-bg-1 text-text">Max per session</option>
                </select>
                <svg aria-hidden viewBox="0 0 16 16" className="pointer-events-none absolute right-2 bottom-2.5 h-3 w-3 text-faint">
                  <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.2" />
                </svg>
              </label>
              {form.repeat === "max" ? (
                <label className="grid gap-1">
                  <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">Max count</span>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={form.max}
                    onChange={(e) => setForm((f) => ({ ...f, max: e.target.value }))}
                    className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11.5px] text-text outline-none focus-visible:border-sense-pale"
                  />
                </label>
              ) : (
                <div className="flex items-end justify-between border-b border-line-strong pb-1.5">
                  <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">
                    {form.enabled ? "On" : "Off"}
                  </span>
                  <Switch
                    on={form.enabled}
                    label="Enabled state"
                    onChange={(next) => setForm((f) => ({ ...f, enabled: next }))}
                  />
                </div>
              )}
            </div>

            {/* Schedule Window (UTC) */}
            <div className="grid gap-2 border-t border-line/60 pt-3">
              <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">Schedule days &amp; clock window (optional, UTC 24h)</span>
              <div className="flex flex-wrap items-center gap-2.5 sm:gap-3">
                <div className="flex flex-wrap gap-1">
                  {DAYS.map((day) => {
                    const on = form.days.includes(day);
                    return (
                      <button
                        key={day}
                        type="button"
                        onClick={() =>
                          setForm((f) => ({
                            ...f,
                            days: on ? f.days.filter((x) => x !== day) : [...f.days, day],
                          }))
                        }
                        className={`press mono h-8 border px-2 text-[10px] uppercase tracking-[0.1em] sm:h-7 ${
                          on ? "border-sense-pale bg-bg-2 text-sense-pale" : "border-line text-faint hover:text-text"
                        }`}
                      >
                        {day}
                      </button>
                    );
                  })}
                </div>
                <div className="flex flex-wrap items-center gap-3">
                  <label className="flex items-center gap-1.5">
                    <span className="mono text-[10px] text-faint">after</span>
                    <input
                      value={form.after}
                      onChange={(e) => setForm((f) => ({ ...f, after: e.target.value }))}
                      placeholder="09:00"
                      className="mono h-8 w-16 border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale sm:h-7"
                    />
                  </label>
                  <label className="flex items-center gap-1.5">
                    <span className="mono text-[10px] text-faint">before</span>
                    <input
                      value={form.before}
                      onChange={(e) => setForm((f) => ({ ...f, before: e.target.value }))}
                      placeholder="18:00"
                      className="mono h-8 w-16 border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale sm:h-7"
                    />
                  </label>
                </div>
              </div>
            </div>

            {/* Scope & Loop */}
            <div className="grid gap-3 border-t border-line/60 pt-3 md:grid-cols-3">
              <label className="grid gap-1">
                <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">File glob (optional)</span>
                <input
                  value={form.glob}
                  onChange={(e) => setForm((f) => ({ ...f, glob: e.target.value }))}
                  placeholder="e.g. crates/**/*.rs"
                  className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
                />
              </label>
              <label className="grid gap-1">
                <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">Error signature</span>
                <input
                  value={form.error}
                  onChange={(e) => setForm((f) => ({ ...f, error: e.target.value }))}
                  placeholder="e.g. E0277 or TS2322"
                  className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
                />
              </label>
              <label className="relative grid gap-1">
                <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">Linked guidance loop</span>
                <select
                  value={form.loopId}
                  onChange={(e) => setForm((f) => ({ ...f, loopId: e.target.value }))}
                  className="press mono h-8 w-full appearance-none border-b border-line-strong bg-transparent pr-6 pl-1 text-[11px] text-text focus-visible:border-sense-pale"
                >
                  <option value="" className="bg-bg-1 text-text">None</option>
                  {loops.map((l) => (
                    <option key={l.id ?? l.name} value={l.id ?? ""} className="bg-bg-1 text-text">
                      {l.name || l.id}
                    </option>
                  ))}
                </select>
                <svg aria-hidden viewBox="0 0 16 16" className="pointer-events-none absolute right-2 bottom-2.5 h-3 w-3 text-faint">
                  <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.2" />
                </svg>
              </label>
            </div>

            {/* Optional: Also Save as Reusable Preset */}
            <div className="grid gap-2 border-t border-line/60 pt-3">
              <div className="flex items-center justify-between gap-3">
                <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">Also save project primes as a reusable preset</span>
                <Switch on={saveAsPreset} label="Save as preset" onChange={setSaveAsPreset} />
              </div>
              {saveAsPreset && (
                <div className="grid gap-2.5 md:grid-cols-2">
                  <input
                    value={presetName}
                    onChange={(e) => setPresetName(e.target.value)}
                    placeholder="Preset name (e.g. my-rust-checks)"
                    maxLength={60}
                    className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
                  />
                  <input
                    value={presetDesc}
                    onChange={(e) => setPresetDesc(e.target.value)}
                    placeholder="Short description"
                    maxLength={200}
                    className="mono h-8 w-full border-b border-line-strong bg-transparent px-1 text-[11px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
                  />
                </div>
              )}
            </div>
          </form>
        </div>
      )}

      {q.error && !d ? (
        <Empty title="Primes studio did not answer">{q.error}</Empty>
      ) : d?.error ? (
        <Empty title="Primes unavailable">{d.error}</Empty>
      ) : !d ? (
        <Placeholder className="h-64" what="primes studio" />
      ) : (
        <div className="grid gap-6">
          {/* Unified Primes List (clean alarm-style rows) */}
          <section className="grid gap-2">
            <div className="flex items-baseline justify-between border-b border-line-strong pb-2">
              <h2 className="mono text-[11px] uppercase tracking-[0.16em] text-text">
                Primes · {activeBuiltins + activeCustom} active
              </h2>
              <span className="mono text-[10px] text-faint">
                {d.mode === "off" ? "Master off" : `Mode: ${d.mode}`}
              </span>
            </div>

            {custom.length === 0 && listedBuiltins.length === 0 ? (
              <div className="py-4">
                <Empty title="No primes added">
                  <button
                    type="button"
                    onClick={openAddSheet}
                    className="press mono mt-2 inline-flex h-8 items-center border-b border-sense-pale text-[10.5px] uppercase tracking-[0.12em] text-sense-pale hover:text-text"
                  >
                    + Add Prime
                  </button>
                </Empty>
              </div>
            ) : (
              <ul className="grid divide-y divide-line border-b border-line">
                {/* Verified presets added to this project: stay in the list when toggled OFF; removed only via Remove */}
                {listedBuiltins.map((b) => (
                    <li
                      key={`builtin:${b.id}`}
                      className={`grid min-w-0 gap-2 py-3.5 transition-opacity sm:py-4 ${b.enabled ? "opacity-100" : "opacity-65 hover:opacity-95"}`}
                      data-e2e="active-builtin-prime"
                      data-rule={b.id}
                    >
                      <div className="flex min-w-0 items-center justify-between gap-2.5 sm:gap-4">
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1 sm:gap-x-3">
                            <span className={`mono text-[14px] sm:text-[16px] ${b.enabled ? "text-text" : "text-muted"}`}>{b.label}</span>
                            <span className="mono text-[10.5px] text-sense-pale sm:text-[11px]">{b.trigger}</span>
                            <Pill tone="verified">verified preset</Pill>
                          </div>
                          <p className="read mt-1 text-[11.5px] leading-relaxed text-faint">{b.summary}</p>
                        </div>
                        <div className="flex shrink-0 items-center gap-2 sm:gap-3">
                          <Switch
                            on={b.enabled}
                            disabled={busy}
                            label={`Toggle ${b.label}`}
                            onChange={(next) => void toggleBuiltin(b.id, next)}
                          />
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <span className="mono text-[10px] text-quiet">
                          {b.enabled ? "on" : "off"} · built-in verified rule
                        </span>
                        <div className="flex items-center gap-3">
                          <button
                            type="button"
                            onClick={() => {
                              setForm(formFromBuiltin(b));
                              setSelectedPreset(b.id);
                              setSheetOpen(true);
                            }}
                            className="press mono border-b border-line-strong px-1 text-[10px] uppercase tracking-[0.12em] text-muted hover:border-sense-pale hover:text-text"
                          >
                            Edit
                          </button>
                          <button
                            type="button"
                            disabled={busy}
                            data-e2e="remove-builtin-prime"
                            onClick={() => void removeBuiltinFromList(b)}
                            className="press mono border-b border-line-strong px-1 text-[10px] uppercase tracking-[0.12em] text-faint hover:border-error hover:text-error disabled:opacity-40"
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    </li>
                  ))}

                {/* Custom & preset-added primes */}
                {custom.map((c) => {
                  const sched = scheduleSummary(c);
                  return (
                    <li
                      key={c.id}
                      className={`grid min-w-0 gap-2 py-3.5 transition-opacity sm:py-4 ${c.enabled ? "opacity-100" : "opacity-65 hover:opacity-95"}`}
                      data-e2e="custom-prime"
                      data-id={c.id}
                    >
                      <div className="flex min-w-0 items-center justify-between gap-2.5 sm:gap-4">
                        <button
                          type="button"
                          onClick={() => openEditSheet(c)}
                          className="press group min-w-0 flex-1 text-left"
                        >
                          <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1 sm:gap-x-3">
                            <span className={`mono text-[15px] font-normal tracking-tight sm:text-[18px] ${c.enabled ? "text-text" : "text-muted"}`}>
                              {headlineOf(c)}
                            </span>
                            <span className="mono text-[11.5px] text-sense-pale group-hover:text-text sm:text-[12px]">{c.name}</span>
                            {sched && c.trigger !== "schedule" && <Pill tone="muted">{sched}</Pill>}
                            {c.glob && <Pill tone="muted">{c.glob}</Pill>}
                            {c.loop_id && <Pill tone="verified">loop: {c.loop_id}</Pill>}
                          </div>
                          <p className="read mt-1 line-clamp-2 text-[11.5px] leading-relaxed text-faint">
                            {c.text}
                          </p>
                        </button>

                        <div className="flex shrink-0 items-center gap-2 sm:gap-3">
                          <Switch
                            on={c.enabled}
                            disabled={busy}
                            label={`Toggle ${c.name}`}
                            onChange={(next) => void toggleCustom(c.id, next)}
                          />
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <span className="mono text-[10px] text-quiet">
                          {c.repeat === "once" ? "once / session" : c.repeat === "max" ? `max ${c.max} / session` : "always"}
                          {" · "}fired {c.fired ?? 0} {c.fired === 1 ? "time" : "times"}
                          {c.last_ms && c.last_ms > 0 && now > 0 ? ` · last ${ago(c.last_ms, now)}` : ""}
                        </span>
                        <div className="flex items-center gap-3">
                          <button
                            type="button"
                            onClick={() => openEditSheet(c)}
                            className="press mono border-b border-line-strong px-1 text-[10px] uppercase tracking-[0.12em] text-muted hover:border-text hover:text-text"
                          >
                            Edit
                          </button>
                          <DeleteFlow
                            path="/api/primes"
                            body={{ action: "remove", project: projectKey, id: c.id }}
                            label="Delete"
                            done={() => `Deleted "${c.name}". Undoable in Audit.`}
                            refresh={["/api/primes", "/api/audit"]}
                            onNote={setNote}
                            startClass="press mono border-b border-line-strong px-1 text-[10px] uppercase tracking-[0.12em] text-faint hover:border-error hover:text-error disabled:opacity-40"
                          />
                        </div>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>

          {/* Linked Guidance Loops footer */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
            <span className="mono text-[10.5px] text-faint">
              Guidance loops for this project: <strong className="text-text">{loops.length}</strong>
            </span>
            <Link href="/memory/" className="mono text-[10.5px] uppercase tracking-[0.12em] text-sense-pale hover:text-text">
              Manage loops &amp; steps in Memory &rarr;
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}

/** Quick prime control right-rail for Audit and Learning: view primes, toggle on/off, and quick-edit prompt & prime type inline. */
export function PrimesRail({
  inStudio = false,
  onOpenStudio,
  onOpenLog,
}: {
  inStudio?: boolean;
  onOpenStudio?: () => void;
  onOpenLog?: () => void;
}) {
  const { key: globalKey, projects } = useProjectKey();
  const projectKey = globalKey ?? projects?.projects?.[0]?.key ?? "nano";
  const q = useApi<PrimesReply>(`/api/primes?project=${encodeURIComponent(projectKey)}`, 5000);
  const ledger = useApi<Ledger>(`/api/ledger?project=${encodeURIComponent(projectKey)}`, 10000);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<Note | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingIsNew, setEditingIsNew] = useState(false);
  const [editName, setEditName] = useState("");
  const [editTrigger, setEditTrigger] = useState("prompt");
  const [editInterval, setEditInterval] = useState("0");
  const [editText, setEditText] = useState("");

  const d = q.data;
  const builtins = d?.builtins ?? [];
  const custom = d?.custom ?? [];
  const pc = primeCounts(ledger.data);

  const activeBuiltins = builtins.filter((b) => b.enabled).length;
  const activeCustom = custom.filter((c) => c.enabled).length;

  const startQuickEdit = (c: CustomPrime) => {
    setEditingId(c.id);
    setEditingIsNew(false);
    setEditName(c.name);
    setEditTrigger(c.trigger);
    setEditInterval(String(c.interval_min ?? 0));
    setEditText(c.text);
    setNote(null);
  };

  const startBuiltinQuickEdit = (b: BuiltinPrime) => {
    const tpl = BUILTIN_TEMPLATES[b.id];
    setEditingId(`builtin:${b.id}`);
    setEditingIsNew(true);
    setEditName(b.label);
    setEditTrigger(tpl?.trigger ?? "prompt");
    setEditInterval(tpl?.intervalMin ?? "0");
    setEditText(tpl?.text ?? b.summary);
    setNote(null);
  };

  const runAction = async (body: Record<string, unknown>, okText: string) => {
    setBusy(true);
    try {
      await post("/api/primes", { project: projectKey, ...body });
      setNote({ text: okText, tone: "verified" });
      refreshPrimesAll();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const saveQuickEdit = async (id: string) => {
    if (!editName.trim() || !editText.trim()) return;
    setBusy(true);
    try {
      const payload: Record<string, unknown> = {
        project: projectKey,
        action: editingIsNew ? "add" : "edit",
        name: editName.trim(),
        trigger: editTrigger,
        interval_min: Math.max(0, Number.parseInt(editInterval || "0", 10) || 0),
        text: editText.trim(),
      };
      if (editingIsNew) {
        payload.enabled = false;
      } else {
        payload.id = id;
      }
      await post("/api/primes", payload);
      setEditingId(null);
      setEditingIsNew(false);
      setNote({ text: editingIsNew ? `Added customized "${editName.trim()}" (off).` : `Saved "${editName.trim()}".`, tone: "verified" });
      refreshPrimesAll();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  const setMode = async (mode: string) => {
    setBusy(true);
    try {
      await post("/api/controls", { key: "priming", value: mode });
      setNote({ text: `Priming set to "${mode}".`, tone: "verified" });
      refreshPrimesAll();
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };

  return (
    <div className="grid gap-0" data-e2e="primes-rail">
      <RailSection title="Quick prime control">
        <div className="grid gap-2.5">
          <div className="flex items-center justify-between gap-2">
            <span className="mono truncate text-[12px] text-text">{projectLabelByKey(projects, projectKey)}</span>
            {inStudio && onOpenLog ? (
              <button
                type="button"
                onClick={onOpenLog}
                className="press mono shrink-0 border-b border-sense-pale text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text"
              >
                Back to Log
              </button>
            ) : onOpenStudio ? (
              <button
                type="button"
                onClick={onOpenStudio}
                className="press mono shrink-0 border-b border-sense-pale text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text"
              >
                Open Studio
              </button>
            ) : (
              <Link
                href="/audit/#primes"
                className="press mono shrink-0 border-b border-sense-pale text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text"
              >
                Open Studio
              </Link>
            )}
          </div>
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0 flex-1">
              <Segmented
                values={["off", "quiet", "on"]}
                value={d?.mode ?? "off"}
                label="Master priming"
                onPick={(v) => void setMode(v)}
              />
            </div>
            {(d?.mode !== "off" || activeBuiltins > 0 || activeCustom > 0) && (
              <button
                type="button"
                disabled={busy}
                onClick={() => void runAction({ action: "all_off" }, "All primes switched off.")}
                className="press mono shrink-0 border-b border-line-strong text-[10px] uppercase tracking-[0.12em] text-faint hover:border-error hover:text-error disabled:opacity-40"
              >
                All off
              </button>
            )}
          </div>
          <NoteLine note={note} />
        </div>
      </RailSection>

      <RailSection title={`Your primes · ${activeCustom}/${custom.length} on`}>
        {!d ? (
          <p className="mono text-[11px] text-faint">Loading...</p>
        ) : custom.length === 0 ? (
          <p className="read text-[11px] leading-relaxed text-faint">
            No custom primes yet. Use{" "}
            {onOpenStudio ? (
              <button type="button" onClick={onOpenStudio} className="link">
                Primes Studio
              </button>
            ) : (
              <Link href="/audit/#primes" className="link">
                Primes Studio
              </Link>
            )}{" "}
            to add one from a preset or custom trigger.
          </p>
        ) : (
          <ul className="grid gap-2.5">
            {custom.map((c) => {
              const isEdit = editingId === c.id;
              return (
                <li key={c.id} className="grid gap-1.5 border-b border-line/60 pb-2.5 last:border-b-0">
                  <div className="flex items-center justify-between gap-2">
                    <div className="min-w-0">
                      <div className="mono truncate text-[11.5px] text-text">{c.name}</div>
                      <div className="mono truncate text-[9.5px] text-sense-pale">{headlineOf(c)}</div>
                    </div>
                    <Switch
                      on={c.enabled}
                      disabled={busy}
                      label={`Toggle ${c.name}`}
                      onChange={(next) => void runAction({ action: "toggle", id: c.id, enabled: next }, `${c.name} ${next ? "ON" : "OFF"}.`)}
                    />
                  </div>
                  {!isEdit ? (
                    <>
                      <p className="read line-clamp-2 text-[10.5px] leading-relaxed text-faint">{c.text}</p>
                      <button
                        type="button"
                        onClick={() => startQuickEdit(c)}
                        className="press mono w-fit border-b border-line-strong text-[9.5px] uppercase tracking-[0.12em] text-muted hover:border-sense-pale hover:text-text"
                      >
                        Edit prompt &amp; type
                      </button>
                    </>
                  ) : (
                    <form
                      className="grid gap-2 border-l-2 border-sense-pale bg-bg-0/60 py-2 pr-1.5 pl-2.5"
                      onSubmit={(e) => {
                        e.preventDefault();
                        void saveQuickEdit(c.id);
                      }}
                    >
                      <input
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        placeholder="Prime name"
                        maxLength={80}
                        required
                        className="mono h-7 w-full border-b border-line-strong bg-transparent text-[11px] text-text outline-none focus-visible:border-sense-pale"
                      />
                      <div className="grid grid-cols-[minmax(0,1fr)_4.5rem] gap-2">
                        <select
                          aria-label="Prime type"
                          value={editTrigger}
                          onChange={(e) => setEditTrigger(e.target.value)}
                          className="press mono h-7 w-full appearance-none border-b border-line-strong bg-transparent text-[10.5px] text-text focus-visible:border-sense-pale"
                        >
                          {(d.triggers ?? Object.keys(TRIGGER_LABELS)).map((t) => (
                            <option key={t} value={t} className="bg-bg-1 text-text">
                              {TRIGGER_LABELS[t] ?? t}
                            </option>
                          ))}
                        </select>
                        <input
                          type="number"
                          min={0}
                          max={1440}
                          value={editInterval}
                          onChange={(e) => setEditInterval(e.target.value)}
                          title="Interval or cooldown (minutes)"
                          placeholder="min"
                          className="mono h-7 w-full border-b border-line-strong bg-transparent text-right text-[10.5px] text-text outline-none focus-visible:border-sense-pale"
                        />
                      </div>
                      <textarea
                        rows={3}
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        placeholder="Prime prompt text"
                        maxLength={2000}
                        required
                        className="mono w-full border border-line bg-bg-1/70 p-1.5 text-[10.5px] leading-relaxed text-text outline-none focus-visible:border-sense-pale"
                      />
                      <div className="flex items-center gap-3">
                        <button
                          type="submit"
                          disabled={busy || !editName.trim() || !editText.trim()}
                          className="press mono border-b border-sense-pale text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40"
                        >
                          Save
                        </button>
                        <button
                          type="button"
                          onClick={() => setEditingId(null)}
                          className="press mono text-[10px] uppercase tracking-[0.12em] text-faint hover:text-text"
                        >
                          Cancel
                        </button>
                      </div>
                    </form>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </RailSection>

      <RailSection title={`Verified presets · ${activeBuiltins}/${builtins.length} on`}>
        <ul className="grid gap-1.5">
          {builtins.map((b) => {
            const bid = `builtin:${b.id}`;
            const isEdit = editingId === bid;
            return (
              <li key={b.id} className="grid gap-1.5 border-b border-line/40 pb-1.5 last:border-b-0">
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <div className="mono truncate text-[11px] text-muted">{b.label}</div>
                    <div className="mono truncate text-[9.5px] text-quiet">{b.trigger}</div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <button
                      type="button"
                      onClick={() => (isEdit ? setEditingId(null) : startBuiltinQuickEdit(b))}
                      className="press mono border-b border-line-strong px-0.5 text-[9.5px] uppercase tracking-[0.1em] text-muted hover:border-sense-pale hover:text-text"
                    >
                      {isEdit ? "Close" : "Edit"}
                    </button>
                    <Switch
                      on={b.enabled}
                      disabled={busy}
                      label={`Toggle ${b.label}`}
                      onChange={(next) => void runAction({ action: "toggle_builtin", id: b.id, enabled: next }, `${b.label} ${next ? "ON" : "OFF"}.`)}
                    />
                  </div>
                </div>
                {isEdit && d && (
                  <form
                    className="grid gap-2 border-l-2 border-sense-pale bg-bg-0/60 py-2 pr-1.5 pl-2.5"
                    onSubmit={(e) => {
                      e.preventDefault();
                      void saveQuickEdit(bid);
                    }}
                  >
                    <input
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      placeholder="Prime name"
                      maxLength={80}
                      required
                      className="mono h-7 w-full border-b border-line-strong bg-transparent text-[11px] text-text outline-none focus-visible:border-sense-pale"
                    />
                    <div className="grid grid-cols-[minmax(0,1fr)_4.5rem] gap-2">
                      <select
                        aria-label="Prime type"
                        value={editTrigger}
                        onChange={(e) => setEditTrigger(e.target.value)}
                        className="press mono h-7 w-full appearance-none border-b border-line-strong bg-transparent text-[10.5px] text-text focus-visible:border-sense-pale"
                      >
                        {(d.triggers ?? Object.keys(TRIGGER_LABELS)).map((t) => (
                          <option key={t} value={t} className="bg-bg-1 text-text">
                            {TRIGGER_LABELS[t] ?? t}
                          </option>
                        ))}
                      </select>
                      <input
                        type="number"
                        min={0}
                        max={1440}
                        value={editInterval}
                        onChange={(e) => setEditInterval(e.target.value)}
                        title="Interval or cooldown (minutes)"
                        placeholder="min"
                        className="mono h-7 w-full border-b border-line-strong bg-transparent text-right text-[10.5px] text-text outline-none focus-visible:border-sense-pale"
                      />
                    </div>
                    <textarea
                      rows={3}
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      placeholder="Prime prompt text"
                      maxLength={2000}
                      required
                      className="mono w-full border border-line bg-bg-1/70 p-1.5 text-[10.5px] leading-relaxed text-text outline-none focus-visible:border-sense-pale"
                    />
                    <div className="flex items-center gap-3">
                      <button
                        type="submit"
                        disabled={busy || !editName.trim() || !editText.trim()}
                        className="press mono border-b border-sense-pale text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40"
                      >
                        Save custom
                      </button>
                      <button
                        type="button"
                        onClick={() => setEditingId(null)}
                        className="press mono text-[10px] uppercase tracking-[0.12em] text-faint hover:text-text"
                      >
                        Cancel
                      </button>
                    </div>
                  </form>
                )}
              </li>
            );
          })}
        </ul>
      </RailSection>

      <RailSection title="Delivered">
        <div className="mono flex flex-wrap gap-x-3 gap-y-1 text-[10.5px] text-faint">
          <span><strong className="text-text">{pc?.given ?? 0}</strong> given</span>
          <span className="text-verified"><strong>{pc?.followed ?? 0}</strong> followed</span>
          <span>{pc?.passed_over ?? 0} passed</span>
          <span>{pc?.window_open ?? 0} open</span>
        </div>
      </RailSection>
    </div>
  );
}
