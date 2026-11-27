"use client";

import { useState } from "react";
import { Empty, Masked, Panel, Pill, Placeholder } from "@/components/ui";
import { post, readCache, refresh, refreshAll, useApi, useApiLevel, type Control, type Library } from "@/lib/api";
import {
  acceptsAuto,
  AGENTS_GROUP,
  CLAUDE_KEYS,
  isLimitControl,
  CONTROL_GROUP_ORDER,
  LEARNING_KEYS,
  MEMORY_KEYS,
  controlGroupName,
  formatControlValue,
  isNumericControl,
  stepIntegerControl,
  validateIntegerControl,
} from "@/lib/controls";
import { cartridgeWords } from "@/lib/evidence";
import { parseLive, type LiveCapacity } from "@/lib/livefeed";
import { limitResolved } from "@/lib/limits";
import { gpuCeiling } from "@/lib/vram";
import { maskPath } from "@/lib/privacy";

type Reply = { audit_id: string | null; controls: Control[] };
type Note = { text: string; tone: "verified" | "error" };

/** Sets one control through the engine (with the local token) and refreshes what depends on it. */
export async function setControl(key: string, value: string): Promise<Note> {
  try {
    const r = await post<Reply>("/api/controls", { key, value });
    await refresh("/api/controls");
    refreshAll("/api/audit");
    const c = r.controls.find((x) => x.key === key);
    return { text: `${c?.label ?? key} set to ${formatControlValue(value, key, c?.unit)}. Applies ${c?.applies ?? "next start"}. Logged in Audit, where it can be undone.`, tone: "verified" };
  } catch (e) {
    return { text: e instanceof Error ? e.message : String(e), tone: "error" };
  }
}

/** Switches the stack of cartridges for one use (or the use off, with null) through the engine;
 *  audited as a library row, undone from Audit, applied at the next start of a server. */
export async function setStack(use: string, stack: string | null): Promise<Note> {
  try {
    await post<unknown>("/api/cartridges", { library: use, stack });
    refreshAll("/api/cartridges");
    refreshAll("/api/audit");
    return { text: `Stack for ${use} set to ${stack ?? "off"}. Applies at the next start of a server. Logged in Audit, where it can be undone.`, tone: "verified" };
  } catch (e) {
    return { text: e instanceof Error ? e.message : String(e), tone: "error" };
  }
}

/** A graduated scale, like the dial of an instrument laid flat: one tick per position on a
 *  hairline, the set position lit with a marker above it, the position names under the ticks.
 *  Each position is a button (44 px tall) reached by keyboard as a radio group. */
function Scale({ values, value, busy, label, onPick, short = false, ckey = "", render }: { values: string[]; value: string; busy: boolean; label: string; onPick: (v: string) => void; short?: boolean; ckey?: string; render?: (v: string) => string }) {
  const i = Math.max(0, values.indexOf(value));
  const n = values.length;
  return (
    <div role="radiogroup" aria-label={label} className={`relative mt-2 grid ${busy ? "opacity-60" : ""}`} style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
      <span aria-hidden className="absolute top-[13px] h-px bg-line-strong" style={{ left: `${50 / n}%`, right: `${50 / n}%` }} />
      <span
        aria-hidden
        className="absolute top-[3px] h-0 w-0 border-x-[4px] border-t-[5px] border-x-transparent border-t-sense-pale"
        style={{ left: `calc(${((i + 0.5) / n) * 100}% - 4px)`, transition: "left var(--d-2) var(--ease)" }}
      />
      {values.map((v, j) => {
        const on = j === i;
        return (
          <button key={v} role="radio" aria-checked={on} disabled={busy} onClick={() => !on && onPick(v)} className="press group relative flex min-h-11 min-w-0 flex-col items-center pt-[9px]">
            <span aria-hidden className="flex h-[11px] items-start">
              <span className={`block w-px ${on ? "h-[11px] bg-sense-pale shadow-[0_0_6px_rgb(127_176_255/0.8)]" : "h-[8px] bg-line-3 group-hover:bg-faint"}`} />
            </span>
            <span className={`mono mt-1.5 max-w-full truncate text-[10.5px] uppercase tracking-[0.06em] ${on ? "text-text" : "text-faint group-hover:text-text"}`}>{render ? render(v) : short && /^\d+$/.test(v) ? v : formatControlValue(v, ckey)}</span>
          </button>
        );
      })}
    </div>
  );
}

/** What a hardware limiter set to `auto` resolves to now: the engine's effective value
 *  (`capacity.limits`) when it is a number, else the measured ceiling or hardware. Null when the
 *  engine reports neither. */
function limitAutoValue(key: string, cap: LiveCapacity | null | undefined): number | null {
  if (!cap) return null;
  const field = key === "limits.vram_mb" ? "vram_mb" : key === "limits.ram_mb" ? "ram_mb" : "cpu_threads";
  const eff = cap.limits?.[field];
  if (eff && /^\d+$/.test(eff)) return Number(eff);
  // Auto: what the engine reports it resolves to now (card memory free to the Nano, host memory
  // room, threads in use).
  const resolved = limitResolved(field, cap.limits);
  if (resolved !== null && resolved > 0) return resolved;
  if (field === "vram_mb") return cap.gpu_ceiling_mb;
  if (field === "ram_mb") return cap.ram_ceiling_mb;
  return cap.threads;
}

function NumericControl({ c, compact = false, foldHelp = false, onNote }: { c: Control; compact?: boolean; foldHelp?: boolean; onNote?: (n: Note) => void }) {
  const [draft, setDraft] = useState(c.value);
  const [prevValue, setPrevValue] = useState(c.value);
  const [busy, setBusy] = useState(false);
  const [inlineError, setInlineError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  if (prevValue !== c.value) {
    setPrevValue(c.value);
    setDraft(c.value);
    setInlineError(null);
  }

  const validation = validateIntegerControl(c, draft);
  const auto = acceptsAuto(c);
  // What auto resolves to on this machine (graphics card ceiling), read from the live feed's
  // cache when a page already polls it; stepping from auto starts there.
  const liveCap = auto ? parseLive(readCache<unknown>("/api/live").data)?.capacity : null;
  const autoMb = auto && c.key === "budget.gpu_mb_max" ? gpuCeiling("auto", liveCap).mb : auto && isLimitControl(c) ? limitAutoValue(c.key, liveCap) : null;
  const commit = async (val: string) => {
    const check = validateIntegerControl(c, val);
    if (!check.ok) {
      setInlineError(check.error);
      setSaved(false);
      return;
    }
    setBusy(true);
    setInlineError(null);
    setSaved(false);
    const n = await setControl(c.key, check.normalized);
    setBusy(false);
    if (n.tone === "verified") {
      setDraft(check.normalized);
      setSaved(true);
    } else {
      setInlineError(n.text);
    }
    onNote?.(n);
  };

  const helpText = c.help || c.desc || "";
  const numDraft = Number(draft.trim());
  const atMin = typeof c.min === "number" && Number.isFinite(numDraft) && numDraft <= c.min;
  const atMax = typeof c.max === "number" && Number.isFinite(numDraft) && numDraft >= c.max;
  const unchanged = validation.ok && validation.normalized === c.value;

  return (
    <div className="min-w-0 py-3">
      <div className="flex min-w-0 items-baseline justify-between gap-2">
        <span className="mono min-w-0 truncate text-[10.5px] uppercase tracking-[0.12em] text-muted" title={c.label}>
          {c.label}
          {c.unit && <span className="text-quiet"> · {c.unit}</span>}
        </span>
        <span className="flex shrink-0 items-center gap-2">
          {!c.wired && <Pill tone="attention">not active yet</Pill>}
          {c.source !== "default" && <Pill>changed</Pill>}
        </span>
      </div>
      {c.values.length > 0 && (
        <div role="group" aria-label={`${c.label} presets`} className={`mt-2 flex flex-wrap gap-1.5 ${busy ? "opacity-60" : ""}`}>
          {c.values.map((v) => {
            const on = v === c.value;
            return (
              <button
                key={v}
                type="button"
                aria-pressed={on}
                disabled={busy}
                onClick={() => {
                  setDraft(v);
                  void commit(v);
                }}
                className={`press mono min-h-8 border px-2 text-[10.5px] uppercase tracking-[0.06em] ${on ? "border-sense-pale text-text" : "border-line text-faint hover:border-line-strong hover:text-text"}`}
              >
                {v === "0" && c.key.startsWith("quota.") ? "unlimited (0)" : v === "auto" ? formatControlValue(v, c.key) : v}
              </button>
            );
          })}
        </div>
      )}
      {auto && !c.values.includes("auto") && (
        <div className={`mt-2 flex flex-wrap items-center gap-1.5 ${busy ? "opacity-60" : ""}`}>
          <button
            type="button"
            aria-pressed={c.value === "auto"}
            disabled={busy}
            onClick={() => {
              setDraft("auto");
              void commit("auto");
            }}
            className={`press mono min-h-8 border px-2 text-[10.5px] uppercase tracking-[0.06em] ${c.value === "auto" ? "border-sense-pale text-text" : "border-line text-faint hover:border-line-strong hover:text-text"}`}
          >
            {formatControlValue("auto", c.key)}
          </button>
          {c.value === "auto" && autoMb !== null && (
            <span className="mono text-[10.5px] text-faint">
              now {Math.round(autoMb)} {c.unit ?? (c.key === "limits.cpu_threads" ? "threads" : "MiB")} on this {isLimitControl(c) ? "machine" : "card"}
            </span>
          )}
        </div>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void commit(draft);
        }}
        className="mt-2 flex flex-wrap items-center gap-1.5"
      >
        <button
          type="button"
          aria-label={`Decrease ${c.label}`}
          disabled={busy || atMin}
          onClick={() => {
            setSaved(false);
            setInlineError(null);
            setDraft(stepIntegerControl(c, draft, -1, autoMb));
          }}
          className="press mono flex h-9 w-9 shrink-0 items-center justify-center border border-line text-xs text-muted hover:border-line-strong hover:text-text disabled:opacity-40"
        >
          -
        </button>
        <input
          type="text"
          inputMode="numeric"
          role="spinbutton"
          aria-label={c.label}
          aria-valuenow={validation.ok && validation.value !== null ? validation.value : undefined}
          aria-valuetext={validation.ok && validation.value === null ? formatControlValue("auto", c.key) : undefined}
          aria-valuemin={c.min}
          aria-valuemax={c.max}
          aria-invalid={!validation.ok || !!inlineError}
          disabled={busy}
          value={draft}
          onChange={(e) => {
            setSaved(false);
            setInlineError(null);
            setDraft(e.target.value);
          }}
          onKeyDown={(e) => {
            if (e.key === "ArrowUp") {
              e.preventDefault();
              setSaved(false);
              setInlineError(null);
              setDraft(stepIntegerControl(c, draft, 1, autoMb));
            } else if (e.key === "ArrowDown") {
              e.preventDefault();
              setSaved(false);
              setInlineError(null);
              setDraft(stepIntegerControl(c, draft, -1, autoMb));
            }
          }}
          className="mono tnum h-9 w-24 min-w-0 border-b border-line-strong bg-transparent px-1.5 text-xs text-text outline-none focus-visible:border-sense-pale"
        />
        <button
          type="button"
          aria-label={`Increase ${c.label}`}
          disabled={busy || atMax}
          onClick={() => {
            setSaved(false);
            setInlineError(null);
            setDraft(stepIntegerControl(c, draft, 1, autoMb));
          }}
          className="press mono flex h-9 w-9 shrink-0 items-center justify-center border border-line text-xs text-muted hover:border-line-strong hover:text-text disabled:opacity-40"
        >
          +
        </button>
        {c.unit && <span className="mono shrink-0 text-[10.5px] text-faint">{c.unit}</span>}
        <button
          type="submit"
          disabled={busy || !validation.ok || unchanged}
          className="press mono ml-auto min-h-9 border-b border-line-strong px-2 text-[10.5px] uppercase tracking-[0.12em] text-muted hover:border-text hover:text-text disabled:opacity-40"
        >
          {busy ? "Saving" : "Apply"}
        </button>
      </form>
      {!validation.ok ? (
        <div role="alert" className="mono mt-1.5 text-[10.5px] text-error">
          {validation.error}
        </div>
      ) : inlineError ? (
        <div role="alert" className="mono mt-1.5 text-[10.5px] text-error">
          {inlineError}
        </div>
      ) : saved ? (
        <div role="status" className="mono mt-1.5 text-[10.5px] text-verified">
          Saved ({formatControlValue(c.value, c.key, c.unit)})
        </div>
      ) : null}
      {!compact && (
        <div className="read mt-1.5 text-xs leading-relaxed text-faint">
          {/* 2026-11-26 (balanced panels): with foldHelp the explanation opens on demand; the
              range, default and when it applies stay in view. */}
          {foldHelp && helpText ? (
            <details className="group mb-1">
              <summary className="press mono inline-flex min-h-8 cursor-pointer list-none items-center gap-1.5 text-[10px] uppercase tracking-[0.12em] text-faint hover:text-text [&::-webkit-details-marker]:hidden">
                what this does
                <svg aria-hidden viewBox="0 0 10 10" className="h-2 w-2 transition-transform group-open:rotate-180" fill="none" stroke="currentColor" strokeWidth={1.2}>
                  <path d="M1.5 3.5 5 7l3.5-3.5" />
                </svg>
              </summary>
              <span className="block pt-1">{helpText}</span>
            </details>
          ) : helpText ? (
            `${helpText} `
          ) : (
            ""
          )}
          <span className="mono text-[10.5px] text-quiet">
            range {c.min ?? ".."}..{c.max ?? ".."}
            {c.unit ? ` ${c.unit}` : ""}
            {auto ? " or auto" : ""} · default {formatControlValue(c.default, c.key, c.unit)} · applies {c.applies}
          </span>
        </div>
      )}
    </div>
  );
}

/** One control as a channel of the instrument: its name in small capitals, its state, the
 *  scale that sets it, and in full form what it does. */
/** `foldHelp`: the help text sits behind a "what this does" toggle (detail on demand), the
 *  scale and its state always in view. */
export function ControlRow({ c, compact = false, foldHelp = false, onNote }: { c: Control; compact?: boolean; foldHelp?: boolean; onNote?: (n: Note) => void }) {
  const [busy, setBusy] = useState(false);
  if (isNumericControl(c)) {
    return <NumericControl c={c} compact={compact} foldHelp={foldHelp} onNote={onNote} />;
  }
  const pick = async (v: string) => {
    setBusy(true);
    const n = await setControl(c.key, v);
    setBusy(false);
    onNote?.(n);
  };
  const onOff = c.values.length === 2 && c.values.includes("on") && c.values.includes("off");
  const values = onOff ? ["off", "on"] : c.values;
  const bits = c.key.startsWith("precision.");
  const helpText = c.help || c.desc || "";
  return (
    <div className="min-w-0 py-3">
      <div className="flex min-w-0 items-baseline justify-between gap-2">
        <span className="mono min-w-0 truncate text-[10.5px] uppercase tracking-[0.12em] text-muted" title={c.label}>
          {c.label}
          {compact && bits && <span className="text-quiet"> · bits</span>}
        </span>
        <span className="flex shrink-0 items-center gap-2">
          {!c.wired && <Pill tone="attention">not active yet</Pill>}
          {c.source !== "default" && <Pill>changed</Pill>}
        </span>
      </div>
      <Scale values={values} value={c.value} busy={busy} label={c.label} onPick={pick} short={compact && bits} ckey={c.key} />
      {!compact && !foldHelp && (
        <div className="read mt-1 text-xs leading-relaxed text-faint">
          {helpText} Applies {c.applies}.
        </div>
      )}
      {!compact && foldHelp && helpText && (
        <details className="group mt-1">
          <summary className="press mono inline-flex min-h-8 cursor-pointer list-none items-center gap-1.5 text-[10px] uppercase tracking-[0.12em] text-faint hover:text-text [&::-webkit-details-marker]:hidden">
            what this does
            <svg aria-hidden viewBox="0 0 10 10" className="h-2 w-2 transition-transform group-open:rotate-180" fill="none" stroke="currentColor" strokeWidth={1.2}>
              <path d="M1.5 3.5 5 7l3.5-3.5" />
            </svg>
          </summary>
          <div className="read mt-1 text-xs leading-relaxed text-faint">
            {helpText} Applies {c.applies}.
          </div>
        </details>
      )}
    </div>
  );
}

/** Channels laid side by side on one surface, hairlines between them. */
export function ControlSurface({ controls, compact = false, onNote, cols = "grid-cols-2" }: { controls: Control[]; compact?: boolean; onNote?: (n: Note) => void; cols?: string }) {
  return (
    <div className={`grid ${cols} border-t border-l border-line`}>
      {controls.map((c) => (
        <div key={c.key} className="min-w-0 border-r border-b border-line px-3">
          {/* 2026-11-26 (balanced panels): each channel's explanation opens on demand, so the
              channels of a row carry similar heights. */}
          <ControlRow c={c} compact={compact} foldHelp onNote={onNote} />
        </div>
      ))}
    </div>
  );
}

/** A use in plain words: `hardware:gpu` reads "hardware · graphics card". */
export function wordForUse(use: string): string {
  const [a, b] = use.split(":");
  const piece: Record<string, string> = { gpu: "graphics card", cpu: "processor", ram: "RAM", disk: "disk" };
  return b ? `${a} · ${piece[b] ?? b}` : a;
}

/** One use's stack as a channel of the instrument: the stacks the library holds for that use and
 *  off, the one in use lit. A pick goes to the engine as a switch of the library. */
export function StackChannel({ use, stacks, current, why, trial, onNote }: { use: string; stacks: string[]; current: string | null; why?: string; trial?: boolean; onNote?: (n: Note) => void }) {
  const [busy, setBusy] = useState(false);
  const values = [...stacks, "off"];
  const pick = async (v: string) => {
    setBusy(true);
    const n = await setStack(use, v === "off" ? null : v);
    setBusy(false);
    onNote?.(n);
  };
  return (
    <div className="min-w-0 py-3">
      <div className="flex min-w-0 items-baseline justify-between gap-2">
        <span className="mono min-w-0 truncate text-[10.5px] uppercase tracking-[0.12em] text-muted" title={use}>
          Stack · {wordForUse(use)}
        </span>
        {trial && <Pill tone="attention">on trial</Pill>}
      </div>
      <Scale values={values} value={current ?? "off"} busy={busy} label={`Stack for ${use}`} onPick={pick} render={(v) => v.replace(/@\d+$/, "")} />
      <div className="read mt-1 text-xs leading-relaxed text-faint">{why ? `In use: ${why}.` : "No stack in use for this use."} Applies at the next start of a server.</div>
    </div>
  );
}

/** Every use the library has a stack for, as channels side by side. */
export function StackSurface({ lib, onNote, cols = "grid-cols-1 sm:grid-cols-2" }: { lib: Library; onNote?: (n: Note) => void; cols?: string }) {
  const stacks = lib.catalogue?.stacks ?? [];
  const uses = [...new Set([...(lib.in_use?.uses ?? []).map((u) => u.use), ...stacks.map((s) => s.use)])];
  if (!uses.length) return <div className="text-xs text-faint">No stack in the cartridge library yet.</div>;
  return (
    <div className={`grid ${cols} border-t border-l border-line`}>
      {uses.map((u) => {
        const inUse = lib.in_use?.uses?.find((x) => x.use === u);
        const ids = stacks.filter((s) => s.use === u).map((s) => (s.version ? `${s.id}@${s.version}` : s.id));
        const cur = inUse?.stack ?? null;
        return (
          <div key={u} className="min-w-0 border-r border-b border-line px-3">
            <StackChannel use={u} stacks={cur && !ids.includes(cur) ? [cur, ...ids] : ids} current={cur} why={inUse?.why ? cartridgeWords(inUse.why) : undefined} trial={inUse?.trial} onNote={onNote} />
          </div>
        );
      })}
    </div>
  );
}

export function NoteLine({ note }: { note: Note | null }) {
  if (!note) return null;
  return (
    <div role="status" className={`fade-in mono mb-3 border-l-2 pl-3 text-xs leading-relaxed ${note.tone === "verified" ? "border-verified text-verified" : "border-error text-error"}`}>
      {note.text}
    </div>
  );
}

type SavedChoice = { value: string; ok: boolean; why: string | null };
type SavedEntry = { name: string; role: string | null; value: string; default: string; saved: boolean; in_effect: string; choices: SavedChoice[]; locked: string | null; set_key: string };
type SavedKey = {
  key: string;
  label: string;
  help: string;
  entry: string;
  applies: string;
  undo: boolean;
  read_only: string | null;
  in_file: boolean;
  value: unknown;
  set: { route: string; key: string; mcp: string } | null;
  also: { route: string; mcp: string; interface: string } | null;
  entries: SavedEntry[];
};
export type SavedSettings = { file: string; present: boolean; controls: number; controls_saved: number; keys: SavedKey[]; unknown: string[] };

const SAVED_API = "/api/controls/saved";

/** One entry of a map the settings file holds (a MicroModel's base cartridge, a role's model, a
 *  base cartridge's compensation) as a channel: the values it may take, the saved one lit. A pick
 *  goes to the engine through the controls route under the entry's own key. */
function SavedEntryChannel({ k, e, onNote }: { k: SavedKey; e: SavedEntry; onNote?: (n: Note) => void }) {
  const [busy, setBusy] = useState(false);
  const values = e.choices.filter((c) => c.ok || c.value === e.value).map((c) => c.value);
  const refused = e.choices.filter((c) => !c.ok);
  const role = e.role ? e.role.charAt(0).toUpperCase() + e.role.slice(1) : null;
  const title = role && role.toLowerCase() !== e.name ? `${role} · ${e.name}` : (role ?? e.name);
  const pick = async (v: string) => {
    setBusy(true);
    let n: Note;
    try {
      await post<unknown>("/api/controls", { key: e.set_key, value: v });
      await refresh(SAVED_API);
      refreshAll("/api/cartridges/bases");
      refreshAll("/api/models/roles");
      refreshAll("/api/audit");
      n = { text: `${k.label}: ${title} set to ${v}. Applies ${k.applies}. Logged in Audit${k.undo ? ", where it can be undone" : ""}.`, tone: "verified" };
    } catch (err) {
      n = { text: err instanceof Error ? err.message : String(err), tone: "error" };
    }
    setBusy(false);
    onNote?.(n);
  };
  return (
    <div className="min-w-0 py-3" data-e2e="saved-entry" data-key={e.set_key} data-value={e.value}>
      <div className="flex min-w-0 items-baseline justify-between gap-2">
        <span className="mono min-w-0 truncate text-[10.5px] uppercase tracking-[0.12em] text-muted" title={e.set_key}>
          {title}
        </span>
        <span className="flex shrink-0 items-center gap-2">
          {e.locked && <Pill tone="attention">locked</Pill>}
          {e.saved && e.value !== e.default && <Pill>changed</Pill>}
        </span>
      </div>
      <Scale values={values.length ? values : [e.value]} value={e.value} busy={busy} label={`${k.label}, ${title}`} onPick={pick} render={(v) => v} />
      <div className="mono mt-1 text-[10px] leading-relaxed text-quiet">
        in effect: <span className="text-muted">{e.in_effect}</span>
      </div>
      {(e.locked || refused.length > 0) && (
        <details className="group mt-1">
          <summary className="press mono inline-flex min-h-8 cursor-pointer list-none items-center gap-1.5 text-[10px] uppercase tracking-[0.12em] text-faint hover:text-text [&::-webkit-details-marker]:hidden">
            {e.locked ? "why it is locked" : `${refused.length} not offered`}
            <svg aria-hidden viewBox="0 0 10 10" className="h-2 w-2 transition-transform group-open:rotate-180" fill="none" stroke="currentColor" strokeWidth={1.2}>
              <path d="M1.5 3.5 5 7l3.5-3.5" />
            </svg>
          </summary>
          <div className="read mt-1 grid gap-1 text-xs leading-relaxed text-faint">
            {e.locked && <div>{e.locked}</div>}
            {refused.map((c) => (
              <div key={c.value}>
                <span className="mono text-muted">{c.value}</span>: {c.why ?? "not usable here"}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

/** The settings file key by key, beside the controls above: every other key it holds, each entry
 *  set here, with the page, route and MCP op that also set it; a key nothing sets is shown read
 *  only with the reason. A key no part of this build reads is named, never hidden. */
export function SavedSettingsFile({ onNote }: { onNote?: (n: Note) => void }) {
  const { data, error } = useApi<SavedSettings>(SAVED_API, 15000);
  if (error && !data) {
    return (
      <div className="mono flex items-center gap-2 text-[10.5px] text-quiet" data-e2e="saved-settings" data-state="absent">
        <span aria-hidden className="inline-block h-1.5 w-1.5 border border-faint" />
        <span>the settings file view is not reported by this engine build ({error})</span>
      </div>
    );
  }
  if (!data) return <Placeholder className="h-20" what="settings file" />;
  return (
    <div className="grid min-w-0 gap-6" data-e2e="saved-settings" data-state="ready" data-keys={data.keys.length} data-unknown={data.unknown.length}>
      <div className="mono flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[10.5px] text-quiet">
        <Masked id="settings:file" shown={maskPath(data.file)} full={data.file} />
        <span>{data.present ? "present" : "not written yet (every value at its default)"}</span>
        <span>
          {data.controls} controls, {data.controls_saved} changed from default
        </span>
        <span>{data.keys.length} other keys</span>
      </div>
      {data.keys.map((k) => (
        <div key={k.key} className="min-w-0" data-e2e="saved-key" data-key={k.key} data-read-only={k.read_only ? "true" : "false"}>
          <div className="mb-2 flex min-w-0 flex-wrap items-baseline justify-between gap-2">
            <span className="mono min-w-0 text-[10.5px] uppercase tracking-[0.12em] text-muted">{k.label}</span>
            <span className="mono text-[10px] text-quiet">
              {k.key}
              {k.read_only ? " · read only" : ` · ${k.entries.length} ${k.entries.length === 1 ? "entry" : "entries"}`}
            </span>
          </div>
          {k.read_only ? (
            <div className="border border-line px-3 py-2">
              <div className="mono break-words text-[11px] text-muted">{typeof k.value === "string" ? k.value : k.in_file ? JSON.stringify(k.value) : "not in the file"}</div>
              <div className="read mt-1 text-xs leading-relaxed text-faint">Read only: {k.read_only}.</div>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 border-t border-l border-line sm:grid-cols-2 xl:grid-cols-4">
                {k.entries.map((e) => (
                  <div key={e.set_key} className="min-w-0 border-r border-b border-line px-3">
                    <SavedEntryChannel k={k} e={e} onNote={onNote} />
                  </div>
                ))}
              </div>
              <div className="read mt-1.5 text-xs leading-relaxed text-faint">
                {k.help} Applies {k.applies}.
                {k.also && k.also.interface !== "Settings, Controls" && <> Also set on {k.also.interface}.</>}
                {k.set && k.also && (
                  <span className="mono text-[10px] text-quiet">
                    {" "}
                    API: {k.also.route}
                    {k.also.route !== k.set.route ? `, or ${k.set.route}` : ""} with key {k.set.key}. MCP: {k.also.mcp}
                    {k.also.mcp !== k.set.mcp ? `, or ${k.set.mcp}` : ""}.
                  </span>
                )}
              </div>
            </>
          )}
        </div>
      ))}
      {data.unknown.length > 0 && (
        <div className="mono text-[10.5px] leading-relaxed text-attention" data-e2e="saved-unknown">
          Keys in the file that no part of this build reads (kept as they are): {data.unknown.join(", ")}
        </div>
      )}
    </div>
  );
}

/** Controls the documented API names, by the version that brings them; one this engine build does
 *  not report is named as absent, never drawn as a channel it cannot set. */
const EXPECTED: [string, string, number][] = [
  ["priming", "Priming", 7],
  ["examples", "Keep training examples", 7],
];

export default function Controls() {
  const { data: controls, error } = useApi<Control[]>("/api/controls");
  const level = useApiLevel();
  const lib = useApi<Library>(level !== null && level >= 5 ? "/api/cartridges/library" : null, 15000);
  const [note, setNote] = useState<Note | null>(null);
  const absent = controls ? EXPECTED.filter(([k]) => !controls.some((c) => c.key === k || (k === "priming" && c.key === "nudges"))) : [];
  const groupedControls = new Map<string, Control[]>();
  for (const c of controls ?? []) {
    if (c.key === "mcp.enabled_tools") continue;
    const g = controlGroupName(c);
    const list = groupedControls.get(g) ?? [];
    list.push(c);
    groupedControls.set(g, list);
  }
  const extraGroups = [...groupedControls.keys()].filter((g) => !(CONTROL_GROUP_ORDER as readonly string[]).includes(g));
  const groupTitles = [...CONTROL_GROUP_ORDER, ...extraGroups];

  return (
    <Panel title="Control surface" right={note ? <Pill tone={note.tone}>{note.tone === "verified" ? "saved" : "refused"}</Pill> : undefined}>
      <NoteLine note={note} />
      {error && !controls && <Empty title="Controls did not answer">{error}</Empty>}
      {!controls && !error && <Placeholder className="h-40" what="controls" />}
      {controls && (
        <div className="grid gap-8">
          {error && <div className="mono text-xs text-error">Refresh failed ({error}); showing last values.</div>}
          {groupTitles.map((title) => {
            const list = groupedControls.get(title) ?? [];
            const missing = absent.filter(([k]) => (title === AGENTS_GROUP ? CLAUDE_KEYS.has(k) : title === "Learning" ? LEARNING_KEYS.has(k) : title === "Memory and privacy" ? MEMORY_KEYS.has(k) : false));
            if (!list.length && !missing.length) return null;
            return (
              <div key={title} className="min-w-0">
                <div className="eyebrow mb-2 flex items-center justify-between">
                  <span>{title}</span>
                  <span className="text-quiet">{list.length} channels</span>
                </div>
                {list.length > 0 && <ControlSurface controls={list} onNote={setNote} cols="grid-cols-1 sm:grid-cols-2 xl:grid-cols-4" />}
                {missing.map(([k, label, v]) => (
                  <div key={k} className="mono mt-2 flex items-center gap-2 text-[10.5px] text-quiet">
                    <span aria-hidden className="inline-block h-1.5 w-1.5 border border-faint" />
                    <span className="uppercase tracking-[0.12em]">{label}</span>
                    <span>· not reported by this engine build (API {level ?? "?"}); it comes with API {v}</span>
                  </div>
                ))}
              </div>
            );
          })}
          <div className="min-w-0">
            <div className="eyebrow mb-2 flex items-center justify-between">
              <span>Cartridge stacks</span>
              <span className="text-quiet">{lib.data ? `${new Set([...(lib.data.in_use?.uses ?? []).map((u) => u.use), ...(lib.data.catalogue?.stacks ?? []).map((s) => s.use)]).size} uses` : ""}</span>
            </div>
            {level !== null && level < 5 ? (
              <div className="mono flex items-center gap-2 text-[10.5px] text-quiet">
                <span aria-hidden className="inline-block h-1.5 w-1.5 border border-faint" />
                <span>the cartridge library is not reported by this engine build (API {level}); its stack switch comes with API 5</span>
              </div>
            ) : lib.error && !lib.data ? (
              <div className="mono text-xs text-error">The cartridge library did not answer ({lib.error})</div>
            ) : lib.data ? (
              <StackSurface lib={lib.data} onNote={setNote} cols="grid-cols-1 sm:grid-cols-2 xl:grid-cols-4" />
            ) : (
              <Placeholder className="h-20" what="cartridge stacks" />
            )}
          </div>
          <div className="min-w-0">
            <div className="eyebrow mb-2 flex items-center justify-between">
              <span>Settings file</span>
              <span className="text-quiet">settings.json, key by key</span>
            </div>
            <SavedSettingsFile onNote={setNote} />
          </div>
        </div>
      )}
    </Panel>
  );
}
