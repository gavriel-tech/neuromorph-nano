"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals ... We need the interface to be
// amazing."): the limiters and their meters stay; the recommendation's summary and reasons move
// into tooltips and into LimitsGuide, which the Settings rail shows.

/** The user's hardware limiters on the settings page: one row each for graphics card memory,
 *  system memory and processor threads (the setting, what is in force now, what this machine
 *  has), the hardware-recommended defaults with their tier and reasons, and the live view of the
 *  MicroModels tiling inside the limits. Every change goes through a confirmation and is sent to
 *  `POST /api/limits` (one limiter) or `POST /api/limits/apply-recommended` (all of them). An
 *  engine build without the route says so plainly; nothing is guessed. */
import { useEffect, useRef, useState } from "react";
import { NoteLine } from "@/components/Controls";
import { LiveBadge, NoEngine, VramBar } from "@/components/LivePanels";
import { RailSection } from "@/components/Workspace";
import { Empty, Panel, Pill, Placeholder } from "@/components/ui";
import { post, refresh, refreshAll, useApi } from "@/lib/api";
import {
  APPLY_RECOMMENDED_BODY,
  detectedFor,
  detectedWords,
  inForceShort,
  LIMIT_FIELDS,
  LIMIT_LABELS,
  LIMIT_SHORT,
  limitNumber,
  limitShort,
  limitUsed,
  limitWords,
  machineCapacity,
  parseLimits,
  recommendedChanges,
  residency,
  setLimitBody,
  TIER_BANDS,
  tierBand,
  validateLimit,
  type LimitField,
  type LimitsReply,
} from "@/lib/limits";
import { engineServing, type LiveCapacity, type LiveLimits } from "@/lib/livefeed";
import { useLive } from "@/lib/livehook";
import { AREA_COST, AREA_HOLDS, AREA_KEYS, AREA_LABELS, AREA_NAMES, AREA_SHARE, areaLine, parseMemoryAreas, validateArea, type AreaName, type AreaRow } from "@/lib/memoryareas";

type Note = { text: string; tone: "verified" | "error" };

/** One spectral hue per limiter, from the cartridge bands. */
const HUE: Record<LimitField, string> = { vram_mb: "#6bc8ff", ram_mb: "#7dffa0", cpu_threads: "#ffd36b" };

export const LIMITS_API = "/api/limits";

/** After a limiter changes: the limits, the controls, the live state and Audit are read again. */
async function afterWrite() {
  await refresh(LIMITS_API);
  refreshAll("/api/controls");
  refreshAll("/api/live");
  refreshAll("/api/audit");
}

/** Sets one limiter through the engine. */
export async function setLimit(field: LimitField, value: string): Promise<Note> {
  try {
    await post<unknown>(LIMITS_API, setLimitBody(field, value));
    await afterWrite();
    return { text: `${LIMIT_LABELS[field]} limit set to ${limitWords(field, value)}. Logged in Audit, where it can be undone.`, tone: "verified" };
  } catch (e) {
    return { text: e instanceof Error ? e.message : String(e), tone: "error" };
  }
}

/** Applies the recommended defaults through the engine (sent only after the confirmation). */
export async function applyRecommended(): Promise<Note> {
  try {
    await post<unknown>(`${LIMITS_API}/apply-recommended`, APPLY_RECOMMENDED_BODY);
    await afterWrite();
    return { text: "Recommended defaults applied. Logged in Audit, where it can be undone.", tone: "verified" };
  } catch (e) {
    return { text: e instanceof Error ? e.message : String(e), tone: "error" };
  }
}

/** A monospace leader: the name, a dotted rule, the value. */
function Leader({ k, v, tone = "text-text", title }: { k: string; v: string; tone?: string; title?: string }) {
  return (
    <div className="mono flex min-w-0 items-baseline gap-2 text-[11px]" title={title ?? v}>
      <span className="shrink-0 text-faint">{k}</span>
      <span aria-hidden className="mb-[3px] min-w-4 flex-1 border-b border-dotted border-line-3" />
      <span className={`min-w-0 truncate text-right ${tone}`}>{v}</span>
    </div>
  );
}

/** Live use of one limiter now, from the live feed: VRAM held on the card, host memory in use,
 *  or null when not reported. */
function usedNow(field: LimitField, cap: LiveCapacity | null, vramUsed: number | null, detail: LiveLimits | null): number | null {
  // The Nano's own use under the limit when the engine reports it, else the card's or host's use.
  const own = limitUsed(field, detail);
  if (own !== null) return own;
  if (field === "vram_mb") return vramUsed;
  if (field === "ram_mb") return cap && cap.host_total_mb !== null && cap.host_available_mb !== null ? Math.max(0, cap.host_total_mb - cap.host_available_mb) : null;
  return null;
}

function LimitRow({ field, reply, editable, live, serving, vramUsed, onNote }: { field: LimitField; reply: LimitsReply; editable: boolean; live: LiveCapacity | null; serving: boolean | null; vramUsed: number | null; onNote: (n: Note) => void }) {
  const current = reply.current[field];
  // In force: what the serving engine reports (`capacity.limits`), else what `/api/limits` says
  // is in force in the process that answered.
  const engineLimits = serving !== false ? (live?.limits ?? null) : null;
  const detail = engineLimits ?? reply.effectiveDetail;
  const effective = engineLimits?.[field] ?? reply.effective[field];
  const profile = machineCapacity(live, serving, reply.profile).cap;
  const detected = detectedFor(field, profile);
  const [draft, setDraft] = useState(current ?? "");
  const [prev, setPrev] = useState(current);
  const [armed, setArmed] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (prev !== current) {
    setPrev(current);
    setDraft(current ?? "");
    setArmed(null);
  }
  const effN = limitNumber(effective);
  const used = serving === false ? null : usedNow(field, live, vramUsed, detail);
  // The meter: what is in force as a share of the hardware, and what is in use now inside it.
  const span = detected && detected > 0 ? detected : null;
  const effShare = span && effN !== null ? Math.min(1, effN / span) : effective === "auto" && span ? 1 : null;
  const usedShare = span && used !== null ? Math.min(1, used / span) : null;

  const arm = () => {
    const check = validateLimit(field, draft, detected);
    if (!check.ok) {
      setError(check.error);
      return;
    }
    setError(null);
    if (check.value === current) return;
    setArmed(check.value);
  };
  const confirm = async () => {
    if (armed === null) return;
    setBusy(true);
    const n = await setLimit(field, armed);
    setBusy(false);
    setArmed(null);
    if (n.tone === "error") setError(n.text);
    onNote(n);
  };

  return (
    <li className="grid min-w-0 gap-2 border-b border-line/60 py-3 last:border-b-0" data-limit={field}>
      <div className="flex min-w-0 items-baseline justify-between gap-3">
        <span className="mono flex min-w-0 items-center gap-2 text-[10.5px] uppercase tracking-[0.12em] text-muted">
          <i aria-hidden className="inline-block h-1.5 w-1.5 shrink-0" style={{ background: HUE[field] }} />
          <span className="truncate">{LIMIT_LABELS[field]}</span>
        </span>
        {current !== null && current !== "auto" && <Pill>capped</Pill>}
      </div>
      <div className="grid gap-x-6 gap-y-0.5 sm:grid-cols-3">
        <Leader k="setting" v={limitShort(field, current)} title={limitWords(field, current)} />
        <Leader k="in force now" v={inForceShort(field, effective, serving === false ? null : detail)} tone={effective === "0" && field === "vram_mb" ? "text-attention" : "text-text"} title={limitWords(field, effective)} />
        <Leader k="this machine" v={detectedWords(field, profile)} tone="text-muted" />
      </div>
      {span !== null && (
        <div className="relative h-1.5 w-full overflow-hidden bg-line/50" role="img" aria-label={`${LIMIT_SHORT[field]}: ${limitShort(field, effective)} in force of ${Math.round(span)}${used !== null ? `, ${Math.round(used)} in use` : ""}`}>
          {effShare !== null && <i className="absolute inset-y-0 left-0 block opacity-35" style={{ width: `${effShare * 100}%`, background: HUE[field] }} />}
          {usedShare !== null && <i className="absolute inset-y-0 left-0 block" style={{ width: `${usedShare * 100}%`, background: HUE[field] }} />}
          {effShare !== null && effShare < 1 && <i className="absolute inset-y-0 block w-px bg-signal" style={{ left: `${effShare * 100}%` }} />}
        </div>
      )}
      {used !== null && effN !== null && effN > 0 && (
        <div className={`mono text-[10.5px] ${used > effN ? "text-attention" : "text-faint"}`}>
          {Math.round(used)} / {effN} MiB used of the limit
        </div>
      )}
      {used !== null && (effN === null || effN <= 0) && field !== "cpu_threads" && limitUsed(field, detail) !== null && <div className="mono text-[10.5px] text-faint">the Nano uses {Math.round(used)} MiB now</div>}
      {field === "vram_mb" && serving !== false && detail?.vram?.cpu_only_next_start && <div className="mono text-[10.5px] text-attention">processor only from the next start</div>}
      {editable && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            arm();
          }}
          className="flex flex-wrap items-center gap-1.5"
        >
          <button
            type="button"
            aria-pressed={draft.trim().toLowerCase() === "auto"}
            disabled={busy}
            onClick={() => {
              setDraft("auto");
              setError(null);
              setArmed(null);
            }}
            className={`press mono min-h-8 border px-2 text-[10.5px] uppercase tracking-[0.06em] ${draft.trim().toLowerCase() === "auto" ? "border-sense-pale text-text" : "border-line text-faint hover:border-line-strong hover:text-text"}`}
          >
            auto
          </button>
          {field === "vram_mb" && (
            <button
              type="button"
              aria-pressed={draft.trim() === "0"}
              disabled={busy}
              onClick={() => {
                setDraft("0");
                setError(null);
                setArmed(null);
              }}
              className={`press mono min-h-8 border px-2 text-[10.5px] uppercase tracking-[0.06em] ${draft.trim() === "0" ? "border-sense-pale text-text" : "border-line text-faint hover:border-line-strong hover:text-text"}`}
            >
              CPU only (0)
            </button>
          )}
          <input
            type="text"
            inputMode="numeric"
            aria-label={`${LIMIT_LABELS[field]} limit`}
            aria-invalid={!!error}
            disabled={busy}
            value={draft}
            placeholder="auto or a number"
            onChange={(e) => {
              setDraft(e.target.value);
              setError(null);
              setArmed(null);
            }}
            className="mono tnum h-9 w-28 min-w-0 border-b border-line-strong bg-transparent px-1.5 text-xs text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
          />
          <span className="mono text-[10.5px] text-faint">{field === "cpu_threads" ? "threads" : "MiB"}</span>
          <button type="submit" disabled={busy || draft.trim() === (current ?? "")} className="press mono ml-auto min-h-9 border-b border-line-strong px-2 text-[10.5px] uppercase tracking-[0.12em] text-muted hover:border-text hover:text-text disabled:opacity-40">
            Set limit
          </button>
        </form>
      )}
      {error && (
        <div role="alert" className="mono text-[10.5px] text-error">
          {error}
        </div>
      )}
      {armed !== null && (
        <div role="alertdialog" aria-label={`Confirm ${LIMIT_SHORT[field]} limit`} className="fade-in grid gap-2 border-l-2 border-sense-pale pl-3">
          <div className="text-xs leading-relaxed text-muted">
            Change the {LIMIT_LABELS[field].toLowerCase()} limit from <span className="mono text-text">{limitWords(field, current)}</span> to <span className="mono text-text">{limitWords(field, armed)}</span>?
            {field === "vram_mb" && armed === "0" ? " The engine runs on the processor only from its next start." : " The engine adapts to it live; logged in Audit."}
          </div>
          <div className="flex items-center gap-3">
            <button type="button" disabled={busy} onClick={confirm} className="press mono min-h-11 border-b border-sense-pale px-1 text-[11px] uppercase tracking-[0.12em] text-text">
              {busy ? "Saving" : "Confirm change"}
            </button>
            <button type="button" disabled={busy} onClick={() => setArmed(null)} className="press mono min-h-11 px-1 text-[11px] uppercase tracking-[0.12em] text-faint hover:text-text">
              Keep
            </button>
          </div>
        </div>
      )}
    </li>
  );
}

/** One memory area beside the system memory limit (2026-11-27): what it holds, what lowering it
 *  costs, its use against its budget, and its setting (auto or MiB), changed after a
 *  confirmation through `POST /api/limits`. The engine refuses a size that would take the set
 *  areas past the process budget, and the refusal is shown as it says it. */
function AreaRowView({ area, row, editable, onNote }: { area: AreaName; row: AreaRow; editable: boolean; onNote: (n: Note) => void }) {
  const current = row.setting;
  const [draft, setDraft] = useState(current ?? "");
  const [prev, setPrev] = useState(current);
  const [armed, setArmed] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (prev !== current) {
    setPrev(current);
    setDraft(current ?? "");
    setArmed(null);
  }
  const words = (v: string | null) => (v === null ? "not reported" : v === "auto" ? `auto (${AREA_SHARE[area]}% of the process budget)` : `${v} MiB`);
  const arm = () => {
    const check = validateArea(draft);
    if (!check.ok) {
      setError(check.error);
      return;
    }
    setError(null);
    if (check.value === current) return;
    setArmed(check.value);
  };
  const confirm = async () => {
    if (armed === null) return;
    setBusy(true);
    let n: Note;
    try {
      await post<unknown>(LIMITS_API, { key: AREA_KEYS[area], value: armed });
      await afterWrite();
      n = { text: `${AREA_LABELS[area]} memory set to ${words(armed)}. Logged in Audit, where it can be undone.`, tone: "verified" };
    } catch (e) {
      n = { text: e instanceof Error ? e.message : String(e), tone: "error" };
    }
    setBusy(false);
    setArmed(null);
    if (n.tone === "error") setError(n.text);
    onNote(n);
  };
  return (
    <li className="grid min-w-0 gap-1.5 border-b border-line/60 py-2.5 last:border-b-0" data-area={area}>
      <div className="flex min-w-0 items-baseline justify-between gap-3">
        <span className="mono truncate text-[10.5px] uppercase tracking-[0.12em] text-muted" title={`${AREA_HOLDS[area]} ${AREA_COST[area]}`}>
          {AREA_LABELS[area]}
        </span>
        <span className={`mono tnum shrink-0 text-[10.5px] ${row.full ? "text-attention" : "text-faint"}`} title={row.state ?? undefined}>
          {areaLine(row)}
        </span>
      </div>
      <div className="read text-[11px] leading-relaxed text-faint">
        {AREA_HOLDS[area]} {AREA_COST[area]}
      </div>
      {editable && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            arm();
          }}
          className="flex flex-wrap items-center gap-1.5"
        >
          <button
            type="button"
            aria-pressed={draft.trim().toLowerCase() === "auto"}
            disabled={busy}
            onClick={() => {
              setDraft("auto");
              setError(null);
              setArmed(null);
            }}
            className={`press mono min-h-8 border px-2 text-[10.5px] uppercase tracking-[0.06em] ${draft.trim().toLowerCase() === "auto" ? "border-sense-pale text-text" : "border-line text-faint hover:border-line-strong hover:text-text"}`}
          >
            auto
          </button>
          <input
            type="text"
            inputMode="numeric"
            aria-label={`${AREA_LABELS[area]} memory`}
            aria-invalid={!!error}
            disabled={busy}
            value={draft}
            placeholder="auto or a number"
            onChange={(e) => {
              setDraft(e.target.value);
              setError(null);
              setArmed(null);
            }}
            className="mono tnum h-9 w-28 min-w-0 border-b border-line-strong bg-transparent px-1.5 text-xs text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale"
          />
          <span className="mono text-[10.5px] text-faint">MiB</span>
          <button type="submit" disabled={busy || draft.trim() === (current ?? "")} className="press mono ml-auto min-h-9 border-b border-line-strong px-2 text-[10.5px] uppercase tracking-[0.12em] text-muted hover:border-text hover:text-text disabled:opacity-40">
            Set area
          </button>
        </form>
      )}
      {error && (
        <div role="alert" className="mono text-[10.5px] text-error">
          {error}
        </div>
      )}
      {armed !== null && (
        <div role="alertdialog" aria-label={`Confirm ${AREA_LABELS[area]} memory`} className="fade-in grid gap-2 border-l-2 border-sense-pale pl-3">
          <div className="text-xs leading-relaxed text-muted">
            Change the {AREA_LABELS[area].toLowerCase()} memory from <span className="mono text-text">{words(current)}</span> to <span className="mono text-text">{words(armed)}</span>? The engine adapts to it live; logged in Audit.
          </div>
          <div className="flex items-center gap-3">
            <button type="button" disabled={busy} onClick={confirm} className="press mono min-h-11 border-b border-sense-pale px-1 text-[11px] uppercase tracking-[0.12em] text-text">
              {busy ? "Saving" : "Confirm change"}
            </button>
            <button type="button" disabled={busy} onClick={() => setArmed(null)} className="press mono min-h-11 px-1 text-[11px] uppercase tracking-[0.12em] text-faint hover:text-text">
              Keep
            </button>
          </div>
        </div>
      )}
    </li>
  );
}

/** The three memory areas under the system memory limit (`memory_areas` of `/api/limits`). */
function MemoryAreasView({ data, editable, onNote }: { data: unknown; editable: boolean; onNote: (n: Note) => void }) {
  const m = parseMemoryAreas(data);
  if (!m) return <div className="mono text-[10.5px] text-faint">This engine build does not report memory areas.</div>;
  return (
    <div className="grid gap-1" data-e2e="memory-areas">
      <div className="eyebrow">System memory by area</div>
      <ul aria-label="Memory areas">
        {AREA_NAMES.map((a) => (
          <AreaRowView key={a} area={a} row={m.areas[a]} editable={editable} onNote={onNote} />
        ))}
      </ul>
      {m.intake && <div className={`mono text-[10.5px] ${m.intake.full ? "text-attention" : "text-faint"}`}>intake: {m.intake.note ?? "not reported"}{m.intake.waiting ? ` (${m.intake.waiting} pieces waiting)` : ""}</div>}
    </div>
  );
}

/** The four hardware tiers as spectral bands, the recommended one lit. */
function TierBand({ tier }: { tier: string | null }) {
  const at = tierBand(tier);
  return (
    <div role="list" aria-label="Hardware tiers" className="grid grid-cols-4 gap-px" data-e2e="limits-tiers">
      {TIER_BANDS.map((b, i) => {
        const on = i === at;
        return (
          <div key={b.id} role="listitem" aria-current={on ? "true" : undefined} className="grid min-w-0 gap-1">
            <i aria-hidden className="block h-1.5" style={{ background: b.hue, opacity: on ? 1 : 0.22, boxShadow: on ? `0 0 8px ${b.hue}` : undefined }} />
            <span className={`mono truncate text-[10px] uppercase tracking-[0.06em] ${on ? "text-text" : "text-quiet"}`}>{b.label}</span>
          </div>
        );
      })}
    </div>
  );
}

function Recommendation({ reply, editable, onNote }: { reply: LimitsReply; editable: boolean; onNote: (n: Note) => void }) {
  const rec = reply.recommended;
  const changes = recommendedChanges(reply);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const confirmRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (open) confirmRef.current?.focus();
  }, [open]);
  if (!rec) return <div className="mono text-[11px] text-faint">The limits reply carries no recommendation.</div>;
  const go = async () => {
    setBusy(true);
    const n = await applyRecommended();
    setBusy(false);
    setOpen(false);
    onNote(n);
  };
  return (
    <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] content-start gap-3" data-e2e="limits-recommended">
      <div className="eyebrow flex items-center justify-between gap-3">
        <span>Recommended</span>
        {(rec.tier_label ?? rec.tier) && (
          <span className="mono min-w-0 truncate text-text normal-case" title={rec.summary ?? rec.tier ?? undefined}>
            {rec.tier_label ?? rec.tier}
          </span>
        )}
      </div>
      <TierBand tier={rec.tier} />
      <ul className="grid gap-1">
        {LIMIT_FIELDS.map((f) => (
          <li key={f}>
            <Leader k={LIMIT_SHORT[f]} v={limitShort(f, rec.values[f])} title={rec.reasons[f] ? `${limitWords(f, rec.values[f])}: ${rec.reasons[f]}` : limitWords(f, rec.values[f])} />
          </li>
        ))}
      </ul>
      {rec.notes.length > 0 && (
        <ul className="grid gap-1 border-l-2 border-attention pl-3" data-e2e="limits-notes">
          {rec.notes.map((n) => (
            <li key={n} className="read text-[11px] leading-relaxed text-attention">
              {n}
            </li>
          ))}
        </ul>
      )}
      {editable && (
        <div className="grid gap-2 border-t border-line/60 pt-3">
          <button
            type="button"
            disabled={busy || changes.length === 0}
            onClick={() => setOpen(true)}
            aria-haspopup="dialog"
            aria-expanded={open}
            className="press mono min-h-11 justify-self-start border-b border-line-strong px-1 text-[11px] uppercase tracking-[0.12em] text-muted hover:text-text disabled:opacity-50"
          >
            {changes.length === 0 ? "At recommended defaults" : "Apply Recommended Defaults"}
          </button>
          {open && (
            <div role="alertdialog" aria-label="Confirm Apply Recommended Defaults" className="fade-in grid gap-2 border-l-2 border-sense-pale pl-3" data-e2e="limits-apply-confirm">
              <div className="text-xs text-muted">This changes {changes.length === 1 ? "one limit" : `${changes.length} limits`}:</div>
              <ul className="grid gap-1">
                {changes.map((c) => (
                  <li key={c.key} className="mono grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 text-[11px]">
                    <span className="text-faint">{LIMIT_SHORT[c.field]}</span>
                    <span className="min-w-0 truncate text-right">
                      <span className="text-muted">{limitShort(c.field, c.from)}</span>
                      <span className="text-quiet"> to </span>
                      <span className="text-text">{limitShort(c.field, c.to)}</span>
                    </span>
                  </li>
                ))}
              </ul>
              <div className="flex items-center gap-3">
                <button ref={confirmRef} type="button" disabled={busy} onClick={go} className="press mono min-h-11 border-b border-sense-pale px-1 text-[11px] uppercase tracking-[0.12em] text-text">
                  {busy ? "Applying" : "Confirm apply"}
                </button>
                <button type="button" disabled={busy} onClick={() => setOpen(false)} className="press mono min-h-11 px-1 text-[11px] uppercase tracking-[0.12em] text-faint hover:text-text">
                  Keep current
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** The live tiling inside the limits: the card by MicroModel with the cap marked, and whether the
 *  four MicroModels are seated together or the writer swaps in and out. */
export function TilingNow({ capMb }: { capMb?: number | null }) {
  const live = useLive();
  const r = live.reply;
  const serving = engineServing(r);
  const res = serving === false ? null : residency(r?.writer, r?.capacity?.limits?.mode);
  const cap = capMb ?? limitNumber(r?.capacity?.limits?.vram_mb ?? null);
  return (
    <div className="grid gap-2" data-e2e="limits-tiling">
      <VramBar vram={r?.vram ?? null} backend={r?.backend ?? null} capMb={cap && cap > 0 ? cap : null} serving={serving} />
      <div className="mono flex items-center gap-2 text-[10.5px]" data-e2e="limits-residency" data-mode={res?.mode ?? "unknown"}>
        <i aria-hidden className="inline-block h-1.5 w-1.5 shrink-0" style={{ background: res?.mode === "seated" ? "var(--verified)" : res?.mode === "swapping" ? "var(--learned)" : "var(--text-4)" }} />
        <span className={res ? "text-muted" : "text-faint"}>{res ? res.words : serving === false ? "No engine serving: no MicroModel is seated" : r ? "The live state does not say how the MicroModels sit" : "Reading the live state"}</span>
        {r?.writer?.headroom_mb !== null && r?.writer?.headroom_mb !== undefined && <span className="ml-auto shrink-0 text-faint">headroom {Math.round(r.writer.headroom_mb)} MiB</span>}
      </div>
    </div>
  );
}

export default function HardwareLimitsPanel() {
  const limits = useApi<unknown>(LIMITS_API, 5000);
  const live = useLive();
  const [note, setNote] = useState<Note | null>(null);
  const reply = parseLimits(limits.data);
  const cap = live.reply?.capacity ?? null;
  const serving = engineServing(live.reply);
  const missing = !reply && !!limits.error && /^404\b/.test(limits.error);
  // Without the route, the rows still show what the live feed reports (in force and detected),
  // read only.
  const capLimits = cap?.limits ?? null;
  const none = { vram_mb: null, ram_mb: null, cpu_threads: null };
  const shown: LimitsReply | null =
    reply ??
    (missing || limits.error
      ? { current: none, effective: capLimits ? { vram_mb: capLimits.vram_mb, ram_mb: capLimits.ram_mb, cpu_threads: capLimits.cpu_threads } : none, effectiveDetail: capLimits, recommended: null, profile: cap, applies: null }
      : null);
  const editable = !!reply;
  const vramCap = limitNumber((serving !== false ? cap?.limits?.vram_mb : null) ?? reply?.effective.vram_mb ?? null);
  const rec = reply?.recommended ?? null;

  return (
    <Panel
      title="Hardware limits"
      right={
        <div className="flex flex-wrap items-center gap-2">
          {rec?.tier && <Pill tone="signal">tier {rec.tier}</Pill>}
          {note && <Pill tone={note.tone}>{note.tone === "verified" ? "saved" : "refused"}</Pill>}
          <LiveBadge />
        </div>
      }
    >
      <NoteLine note={note} />
      {!shown ? (
        <Placeholder className="h-40" what="hardware limits" />
      ) : (
        <div className="@container">
        <div className="grid gap-6 @4xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]" data-e2e="hardware-limits">
          <div className="grid min-w-0 content-start gap-4 @4xl:border-r @4xl:border-line/60 @4xl:pr-6">
            {!reply && (
              <div className="mono flex items-center gap-2 text-[10.5px] text-quiet">
                <span aria-hidden className="inline-block h-1.5 w-1.5 border border-faint" />
                <span>{missing ? "This engine build does not serve /api/limits yet: the limits below are read only, from the live feed." : `Limits did not answer (${limits.error}); showing what the live feed reports.`}</span>
              </div>
            )}
            {reply && limits.error && <div className="mono text-xs text-error">Refresh failed ({limits.error}); showing last values.</div>}
            {serving === false && (
              <div className="grid gap-0.5" title="Settings are saved and apply when an engine starts; this machine is shown as detected.">
                <NoEngine />
              </div>
            )}
            {reply?.effectiveFrom && (
              <div className="mono truncate text-[10.5px] text-quiet" data-e2e="limits-effective-from" title={`in force now: ${reply.effectiveFrom}${reply.applies ? `; applies ${reply.applies}` : ""}`}>
                in force now: {reply.effectiveFrom}
              </div>
            )}
            <ul aria-label="Hardware limiters">
              {LIMIT_FIELDS.map((f) => (
                <LimitRow key={f} field={f} reply={shown} editable={editable} live={cap} serving={serving} vramUsed={live.reply?.vram?.used_mb ?? null} onNote={setNote} />
              ))}
            </ul>
            <MemoryAreasView data={limits.data} editable={editable} onNote={setNote} />
          </div>
          {/* 2026-11-26 (Luis Jake Gabriel III: balanced panels): the tiling now sits under the
              recommendation, so the two columns carry similar weight. */}
          <div className="grid min-w-0 content-start gap-6">
            {reply ? (
              <Recommendation reply={reply} editable={editable} onNote={setNote} />
            ) : (
              <Empty title="Recommended defaults not reported" slot />
            )}
            <div className="grid gap-2 border-t border-line/60 pt-3">
              <div className="eyebrow">Tiling inside the limits, now</div>
              <TilingNow capMb={vramCap} />
            </div>
          </div>
        </div>
        </div>
      )}
    </Panel>
  );
}

/** The reference for the rail: when a change applies, what is in force now, and the recommended
 *  value of each limiter with the engine's reason for it. Only what `/api/limits` reports. */
export function LimitsGuide() {
  const limits = useApi<unknown>(LIMITS_API, 15000);
  const reply = parseLimits(limits.data);
  const rec = reply?.recommended ?? null;
  if (!reply) return <div className="mono text-[11px] text-faint">{limits.error ? `Limits did not answer (${limits.error}).` : "Reading the limits."}</div>;
  return (
    <div className="grid gap-0">
      {(reply.applies || reply.effectiveFrom) && (
        <RailSection title="When a change applies">
          {reply.applies && <p className="read text-[11px] leading-relaxed text-muted">{reply.applies}</p>}
          {reply.effectiveFrom && <p className="read mt-1 text-[11px] leading-relaxed text-faint">In force now: {reply.effectiveFrom}</p>}
        </RailSection>
      )}
      {rec && (
        <RailSection title={`Recommended${rec.tier_label ?? rec.tier ? ` · ${rec.tier_label ?? rec.tier}` : ""}`}>
          {rec.summary && <p className="read mb-2 text-[11px] leading-relaxed text-muted">{rec.summary}</p>}
          <dl className="grid gap-2">
            {LIMIT_FIELDS.map((f) => (
              <div key={f} className="grid gap-0.5">
                <dt className="mono flex items-baseline justify-between gap-3 text-[11px]">
                  <span className="text-faint">{LIMIT_LABELS[f]}</span>
                  <span className="text-text">{limitShort(f, rec.values[f])}</span>
                </dt>
                {rec.reasons[f] && <dd className="read text-[11px] leading-relaxed text-faint">{rec.reasons[f]}</dd>}
              </div>
            ))}
          </dl>
        </RailSection>
      )}
      <RailSection title="Reading the rows">
        <ul className="read grid gap-1 text-[11px] text-faint">
          <li>Setting: what you chose. In force now: what the serving engine holds to. This machine: what was detected.</li>
          <li>The meter: the pale band is the limit in force as a share of the hardware, the bright band what is in use now.</li>
          <li>Auto follows the hardware. A card limit of 0 runs on the processor only, from the next start.</li>
        </ul>
      </RailSection>
    </div>
  );
}
