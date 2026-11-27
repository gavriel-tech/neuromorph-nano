"use client";

// Stated preference cartridges: the user says what they want and what to avoid in plain words, sees
// each rule the engine parsed from them (what it does; enforced, checked, biased or guidance), and
// each want's strength (editable, a new version), confirms, and the engine
// writes a versioned preference cartridge through the cartridge library. Below: every stated
// cartridge with its rules, scope and plug state, edits (each one a new version), plug and unplug
// (a library use switch, as on the decks), the gated removal, and a draft checked against the rules
// in force. Every action is the engine's `/api/preferences` routes, the same ones MCP
// (`nm_learning op preference_*`) and the `preferences` command run. Descriptions live in the side
// panel (`PreferenceNotes`); the middle holds only the working controls.
//
// Mounted on the Learning page and, at merge, in the Cartridges page's Build tab.

import { useState } from "react";
import DeleteFlow from "@/components/DeleteFlow";
import { NoteLine } from "@/components/Controls";
import { RailSection } from "@/components/Workspace";
import { Pill, type Tone } from "@/components/ui";
import { post, refresh, refreshAll, useApi } from "@/lib/api";
import { STRENGTH_MAX, STRENGTH_MIN, kindLabel, parseStrength, ruleStrength, strengthText } from "@/lib/preferences";

export type PrefRule = {
  id: string;
  kind: string;
  said: string;
  does: string;
  enforcement: "enforced" | "checked" | "biased" | "guidance" | string;
  label?: string;
  strength?: number | string;
  scope_text: string;
  scope_words: string;
  cartridge?: string;
};
export type PrefCartridge = {
  id: string;
  key: string;
  version: number;
  name: string;
  scope: string;
  scope_words: string;
  made: string;
  trial: boolean;
  plugged: boolean;
  use: string | null;
  stack: string | null;
  learned_values: boolean;
  learned_keys: number;
  rules: PrefRule[];
  versions: { key: string; version: number; made: string }[];
  error?: string;
};
export type PrefList = { cartridges?: PrefCartridge[]; in_force?: number; enforcement?: Record<string, string>; forms?: string[]; applies?: string; error?: string };
type Preview = { step: string; written: boolean; cartridge?: { id: string; name: string; scope: string; scope_words: string; version: number; adds_to: string | null }; rules?: PrefRule[]; kept?: number; key?: string; audit_id?: string; all_rules?: PrefRule[]; dropped?: string[] };
type Applied = {
  text: string;
  changed: boolean;
  mode: string;
  changes: { rule: string; count: number; cartridge: string }[];
  violations: { rule: string; line: number; what: string; cartridge: string }[];
  wants_missed?: { rule: string; term: string; strength: number | string; what: string; cartridge: string }[];
  guidance: PrefRule[];
  in_scope: number;
};
type Note = { text: string; tone: "verified" | "error" };

export const PREFS_PATH = "/api/preferences";
const BTN = "press mono min-h-9 border border-line-strong px-3 text-[11px] uppercase tracking-[0.12em] text-muted hover:text-text disabled:opacity-40";
const BTN_GO = "press mono min-h-9 border border-sense-pale px-3 text-[11px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40";
const FIELD = "mono min-h-9 w-full border border-line bg-transparent px-2 text-[12px] text-text placeholder:text-faint focus:border-line-strong focus:outline-none";

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function enforcementTone(e: string): Tone {
  return e === "enforced" ? "verified" : e === "checked" ? "signal" : e === "biased" ? "learned" : "muted";
}

/** A want's strength with its edit: the value typed is checked against the engine's range. */
function StrengthControl({ r, onStrength, busy }: { r: PrefRule; onStrength?: (x: number) => void; busy?: boolean }) {
  const now = ruleStrength(r);
  const [typed, setTyped] = useState("");
  if (now === null) return null;
  const next = parseStrength(typed);
  return (
    <span className="mono flex items-center gap-1 text-[10.5px] text-muted" data-e2e="pref-strength" data-strength={strengthText(now)}>
      strength {strengthText(now)}
      {onStrength && (
        <>
          <input
            className="mono h-6 w-14 border border-line bg-transparent px-1 text-[10.5px] text-text placeholder:text-faint focus:border-line-strong focus:outline-none"
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            placeholder={strengthText(now)}
            inputMode="decimal"
            aria-label={`Strength of want ${r.id}, ${STRENGTH_MIN} to ${STRENGTH_MAX}`}
            data-e2e="pref-strength-input"
          />
          <button
            type="button"
            className="press mono text-[10.5px] uppercase tracking-[0.12em] text-faint hover:text-text disabled:opacity-40"
            disabled={busy || next === null || next === now}
            onClick={() => {
              if (next !== null) onStrength(next);
              setTyped("");
            }}
            data-e2e="pref-strength-set"
            title={`${STRENGTH_MIN} to ${STRENGTH_MAX}; writes a new version`}
          >
            set
          </button>
        </>
      )}
    </span>
  );
}

/** One rule as a row: its id, the words it came from, what it does, its kind (want, avoid, ...),
 * how it is held, its scope, and a want's strength. */
function RuleRow({ r, onDrop, onStrength, busy }: { r: PrefRule; onDrop?: () => void; onStrength?: (x: number) => void; busy?: boolean }) {
  return (
    <li className="grid grid-cols-[2.5rem_minmax(0,1fr)_auto] items-start gap-2 border-b border-line py-1.5" data-e2e="pref-rule" data-kind={r.kind} data-enforcement={r.enforcement}>
      <span className="mono text-[10.5px] text-faint">{r.id}</span>
      <span className="min-w-0">
        <span className="block truncate text-[12px] text-text" title={r.said}>
          {r.said}
        </span>
        <span className="mono block truncate text-[10.5px] text-muted" title={r.does}>
          {r.does}
          {r.scope_text !== "global" ? ` · ${r.scope_words}` : ""}
        </span>
        <StrengthControl r={r} onStrength={onStrength} busy={busy} />
      </span>
      <span className="flex items-center gap-2">
        <span className="mono text-[10.5px] uppercase tracking-[0.12em] text-faint" data-e2e="pref-kind">
          {kindLabel(r.kind, r.label)}
        </span>
        <Pill tone={enforcementTone(r.enforcement)}>{r.enforcement}</Pill>
        {onDrop && (
          <button type="button" className="press mono text-[10.5px] uppercase tracking-[0.12em] text-faint hover:text-text disabled:opacity-40" onClick={onDrop} disabled={busy} data-e2e="pref-drop" aria-label={`Drop rule ${r.id}`}>
            drop
          </button>
        )}
      </span>
    </li>
  );
}

/** The chat box: words in, the parsed rules previewed, then written on confirmation. */
function Say({ onNote, cartridges }: { onNote: (n: Note) => void; cartridges: PrefCartridge[] }) {
  const [text, setText] = useState("");
  const [name, setName] = useState("");
  const [scope, setScope] = useState("");
  const [target, setTarget] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [busy, setBusy] = useState(false);
  const body = () => {
    const b: Record<string, unknown> = { text };
    if (target) b.cartridge = target;
    else if (name.trim()) b.name = name.trim();
    if (scope.trim()) b.scope = scope.trim();
    return b;
  };
  const ask = async () => {
    setBusy(true);
    try {
      setPreview(await post<Preview>("/api/preferences/say", body()));
    } catch (e) {
      onNote({ text: errText(e), tone: "error" });
    } finally {
      setBusy(false);
    }
  };
  const write = async () => {
    setBusy(true);
    try {
      const r = await post<Preview>("/api/preferences/say", { ...body(), confirm: true });
      onNote({ text: `Written ${r.key ?? ""}: ${r.rules?.length ?? 0} rule(s), plugged in. Logged in Audit, where it can be undone.`, tone: "verified" });
      setPreview(null);
      setText("");
      await Promise.all([refresh(PREFS_PATH), refreshAll("/api/cartridges"), refreshAll("/api/audit")]);
    } catch (e) {
      onNote({ text: errText(e), tone: "error" });
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="grid gap-2" data-e2e="pref-say">
      <textarea
        className={`${FIELD} min-h-20 py-2`}
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setPreview(null);
        }}
        placeholder="Tell the Nano what you want and what to avoid, e.g. no em dashes; prefer 'colour'; strongly prefer concise replies; for .kicad_pcb files, keep power traces at least 0.5 mm wide"
        aria-label="Your preference in your own words"
        data-e2e="pref-input"
      />
      <div className="grid gap-2 sm:grid-cols-3">
        <select className={FIELD} value={target} onChange={(e) => { setTarget(e.target.value); setPreview(null); }} aria-label="Cartridge to add to" data-e2e="pref-target">
          <option value="">new cartridge</option>
          {cartridges.map((c) => (
            <option key={c.id} value={c.id}>
              add to {c.name || c.id}
            </option>
          ))}
        </select>
        <input className={FIELD} value={name} disabled={!!target} onChange={(e) => { setName(e.target.value); setPreview(null); }} placeholder="name (new cartridge)" aria-label="Cartridge name" data-e2e="pref-name" />
        <input className={FIELD} value={scope} onChange={(e) => { setScope(e.target.value); setPreview(null); }} placeholder="scope: global" aria-label="Cartridge scope" data-e2e="pref-scope" />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className={BTN} disabled={busy || !text.trim()} onClick={ask} data-e2e="pref-preview-btn">
          {busy && !preview ? "reading" : "preview rules"}
        </button>
        {preview && !preview.written && (
          <button type="button" className={BTN_GO} disabled={busy} onClick={write} data-e2e="pref-confirm">
            {busy ? "writing" : `write ${preview.cartridge?.id ?? ""} v${preview.cartridge?.version ?? ""}`}
          </button>
        )}
      </div>
      {preview && !preview.written && (
        <div className="border border-line px-3 py-2" data-e2e="pref-preview">
          <div className="mono mb-1 flex flex-wrap gap-x-3 text-[10.5px] uppercase tracking-[0.12em] text-faint">
            <span>{preview.cartridge?.adds_to ? `adds to ${preview.cartridge.adds_to}` : "new cartridge"}</span>
            <span>{preview.cartridge?.scope_words}</span>
            <span>{preview.kept ? `${preview.kept} rule(s) kept` : ""}</span>
          </div>
          <ul>
            {(preview.rules ?? []).map((r) => (
              <RuleRow key={r.id} r={r} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/** One stated cartridge: its rules, scope, plug state, an added rule, dropped rules, removal. */
function CartridgeRow({ c, onNote }: { c: PrefCartridge; onNote: (n: Note) => void }) {
  const [add, setAdd] = useState("");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const after = () => Promise.all([refresh(PREFS_PATH), refreshAll("/api/cartridges"), refreshAll("/api/audit")]);
  const edit = async (b: Record<string, unknown>, what: string) => {
    setBusy(true);
    try {
      const r = await post<Preview>("/api/preferences/edit", { id: c.id, ...b, confirm: true });
      onNote({ text: r.written ? `${what}: ${r.key}. Logged in Audit, where it can be undone.` : `${c.id}: nothing changed`, tone: "verified" });
      setAdd("");
      await after();
    } catch (e) {
      onNote({ text: errText(e), tone: "error" });
    } finally {
      setBusy(false);
    }
  };
  const plug = async (on: boolean) => {
    setBusy(true);
    try {
      await post("/api/preferences/plug", { id: c.id, on });
      onNote({ text: `${c.id} ${on ? "plugged in" : "unplugged"}. Logged in Audit, where it can be undone.`, tone: "verified" });
      await after();
    } catch (e) {
      onNote({ text: errText(e), tone: "error" });
    } finally {
      setBusy(false);
    }
  };
  return (
    <li className="border border-line" data-e2e="pref-cartridge" data-id={c.id} data-plugged={c.plugged ? "true" : "false"}>
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
        <button type="button" className="press min-w-0 text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
          <span className="block truncate text-[12.5px] text-text">{c.name || c.id}</span>
          <span className="mono block truncate text-[10.5px] text-faint">
            {c.key} · {c.rules.length} rule(s) · {c.scope_words}
            {c.versions.length > 1 ? ` · ${c.versions.length} versions` : ""}
          </span>
        </button>
        <span className="flex items-center gap-2">
          {c.trial && <Pill tone="attention">trial</Pill>}
          <Pill tone={c.plugged ? "verified" : "muted"}>{c.plugged ? "plugged in" : "unplugged"}</Pill>
          <button type="button" className={BTN} disabled={busy || !c.stack} onClick={() => plug(!c.plugged)} data-e2e="pref-plug">
            {c.plugged ? "unplug" : "plug in"}
          </button>
        </span>
      </div>
      {c.learned_values && (
        <div className="mono border-t border-line px-3 py-1 text-[10.5px] text-learned" data-e2e="pref-learned">
          also holds learned preference values ({c.learned_keys})
        </div>
      )}
      {c.error && <div className="mono border-t border-line px-3 py-1 text-[11px] text-error">{c.error}</div>}
      {open && (
        <div className="grid gap-2 border-t border-line px-3 py-2">
          <ul>
            {c.rules.map((r) => (
              <RuleRow
                key={r.id}
                r={r}
                busy={busy}
                onDrop={c.rules.length > 1 ? () => edit({ drop: [r.id] }, `Rule ${r.id} dropped`) : undefined}
                onStrength={(x) => edit({ rule: r.id, strength: x }, `Want ${r.id} strength set to ${strengthText(x)}`)}
              />
            ))}
          </ul>
          <div className="flex gap-2">
            <input className={FIELD} value={add} onChange={(e) => setAdd(e.target.value)} placeholder="add a rule in your words" aria-label={`Add a rule to ${c.id}`} data-e2e="pref-add-input" />
            <button type="button" className={BTN} disabled={busy || !add.trim()} onClick={() => edit({ add }, "Rule added")} data-e2e="pref-add-btn">
              add
            </button>
          </div>
          <DeleteFlow path="/api/preferences/remove" body={{ id: c.id }} label="remove cartridge" e2e="pref-remove" refresh={[PREFS_PATH, "/api/cartridges"]} onNote={onNote} done={() => `${c.id} removed (its files stay; Audit undoes it)`} />
        </div>
      )}
    </li>
  );
}

/** A draft checked against the rules in force. */
function Check({ onNote }: { onNote: (n: Note) => void }) {
  const [text, setText] = useState("");
  const [path, setPath] = useState("");
  const [out, setOut] = useState<Applied | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    try {
      setOut(await post<Applied>("/api/preferences/apply", path.trim() ? { text, path: path.trim() } : { text }));
    } catch (e) {
      onNote({ text: errText(e), tone: "error" });
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="grid gap-2" data-e2e="pref-check">
      <textarea className={`${FIELD} min-h-16 py-2`} value={text} onChange={(e) => setText(e.target.value)} placeholder="a draft to check" aria-label="Draft to check" data-e2e="pref-check-input" />
      <div className="flex gap-2">
        <input className={FIELD} value={path} onChange={(e) => setPath(e.target.value)} placeholder="file path (optional; code is checked, never rewritten)" aria-label="Draft file path" />
        <button type="button" className={BTN} disabled={busy || !text} onClick={run} data-e2e="pref-check-btn">
          check
        </button>
      </div>
      {out && (
        <div className="grid gap-1 border border-line px-3 py-2" data-e2e="pref-check-out" data-changed={out.changed ? "true" : "false"}>
          <span className="mono text-[10.5px] uppercase tracking-[0.12em] text-faint">
            {out.mode} · {out.in_scope} rule(s) in scope · {out.changes.reduce((a, c) => a + c.count, 0)} change(s) · {out.violations.length} violation(s) · {(out.wants_missed ?? []).length} want(s) missed
          </span>
          <pre className="mono whitespace-pre-wrap text-[12px] text-text" data-e2e="pref-check-text">
            {out.text}
          </pre>
          {out.violations.map((v, i) => (
            <span key={i} className="mono text-[11px] text-attention">
              line {v.line}: {v.what} ({v.rule})
            </span>
          ))}
          {(out.wants_missed ?? []).map((m, i) => (
            <span key={`w${i}`} className="mono text-[11px] text-learned" data-e2e="pref-want-missed">
              want {m.rule}: no &quot;{m.term}&quot; in the draft (strength {strengthText(Number(m.strength))}; information, not a violation)
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/** The builder: say, the cartridges, check a draft. */
export default function PreferenceBuilder() {
  const q = useApi<PrefList>(PREFS_PATH, 5000);
  const [note, setNote] = useState<Note | null>(null);
  const carts = q.data?.cartridges ?? [];
  return (
    <div className="grid min-w-0 gap-4" data-e2e="pref-builder">
      <NoteLine note={note} />
      <Say onNote={setNote} cartridges={carts} />
      <div className="mono flex flex-wrap gap-x-3 text-[10.5px] uppercase tracking-[0.12em] text-faint">
        <span data-e2e="pref-in-force">{q.data ? `${q.data.in_force ?? 0} rule(s) in force` : q.error ? `did not answer (${q.error})` : "reading"}</span>
        <span>{carts.length} cartridge(s)</span>
      </div>
      <ul className="grid gap-2" data-e2e="pref-cartridges">
        {carts.map((c) => (
          <CartridgeRow key={c.id} c={c} onNote={setNote} />
        ))}
      </ul>
      <Check onNote={setNote} />
    </div>
  );
}

/** The side panel's notes: how each kind of rule is held and the forms the engine reads. */
export function PreferenceNotes() {
  const q = useApi<PrefList>(PREFS_PATH, 15000);
  return (
    <div className="grid gap-0" data-e2e="pref-notes">
      <RailSection title="Stated preferences">
        <p className="text-[12px] leading-relaxed text-muted">
          Say what you want and what to avoid in your own words. A want (prefer X, always use X, I like X, more X) holds a strength from 0.1 to 2.0, 0.5 by default, 1.0 for strongly and 0.25 for slightly; a wanted term is a bounded push in the Writer decoding, and the mask of an avoided word always wins over it. Each statement becomes a rule in a preference cartridge: a library cartridge of rules (no weights), versioned, plugged in or out like any cartridge, and applied after every other stack. Your agent can do the same with nm_learning op preference_say.
        </p>
      </RailSection>
      <RailSection title="How rules are held">
        <ul className="grid gap-2">
          {Object.entries(q.data?.enforcement ?? {}).map(([k, d]) => (
            <li key={k} className="grid gap-0.5">
              <Pill tone={enforcementTone(k)}>{k}</Pill>
              <span className="text-[11.5px] leading-relaxed text-muted">{d}</span>
            </li>
          ))}
        </ul>
      </RailSection>
      <RailSection title="What the engine reads">
        <ul className="grid gap-1">
          {(q.data?.forms ?? []).map((f) => (
            <li key={f} className="mono text-[11px] leading-relaxed text-muted">
              {f}
            </li>
          ))}
        </ul>
        {q.data?.applies && <p className="mono mt-2 text-[10.5px] text-faint">{q.data.applies}</p>}
      </RailSection>
    </div>
  );
}
