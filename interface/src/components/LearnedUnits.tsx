"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel."): the explanation moved into
// the heading's tooltip; this panel's home is the Cartridges side panel (Learned), where it sits
// beside the decks the units dock on.
// 2026-11-26 (Luis Jake Gabriel III, his words: "We should be able to control and compartmentalized
// and plug in and unplug the learned units"): every learned unit, grouped by the compartment it was
// fitted in and the only one it loads in, each with its own plug, and each compartment with a
// switch for all of its units. Reads and writes `/api/learned-units`; every change is an audit row
// that Audit can undo.

import { useState } from "react";
import { Empty, Fold, Panel, Pill, Placeholder, Switch } from "@/components/ui";
import { ago, bytes, post, refreshAll, useApi } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { maskName, maskPath } from "@/lib/privacy";

export type LearnedUnit = {
  key: string;
  id: string;
  kind: "adapter" | "piece" | "unit";
  compartment: string;
  plugged: boolean;
  acts: boolean;
  why_not: string | null;
  switch_key: string;
  name?: string;
  does?: string;
  role?: string;
  model?: string;
  acts_on?: string;
  made_from?: string;
  measured?: string;
  made?: string;
  version?: number;
  versions?: number;
  bytes?: number;
  path?: string;
  modified_ms?: number;
  loads?: string;
};

export type Compartment = { key: string; label: string; root: string; holds: string; on: boolean; switch_key: string; units: LearnedUnit[] };
export type LearnedUnitsReply = { file: string; total: number; acting: number; applies: string; compartments: Compartment[] };

const KIND: Record<LearnedUnit["kind"], string> = { adapter: "project adapter", piece: "learned piece", unit: "learning unit" };

/** The compartment's name as it may be shown (project names shortened). */
export function compartmentName(c: Pick<Compartment, "key" | "label">): string {
  if (c.key === "nano" || c.key === "shared") return c.label;
  return `project ${maskName(c.label)}`;
}

function UnitRow({ u, busy, onPlug, now }: { u: LearnedUnit; busy: boolean; onPlug: (on: boolean) => void; now: number }) {
  return (
    <li className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-4 gap-y-1 border-b border-line/60 py-3" data-e2e="learned-unit" data-key={u.key}>
      <div className="min-w-0">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="mono truncate text-[12px] text-text" title={u.id}>
            {u.name ? `${u.name} · ` : ""}
            {u.id}
          </span>
          <span className="mono text-[10.5px] uppercase tracking-[0.12em] text-faint">{KIND[u.kind]}</span>
          {u.acts ? <Pill tone="verified">acting</Pill> : <Pill tone="muted" title={u.why_not ?? undefined}>{u.why_not ?? "not acting"}</Pill>}
        </div>
        {u.does && <p className="read mt-1 max-w-3xl text-[11.5px] leading-relaxed text-muted">{u.does}</p>}
        <div className="mono mt-1.5 grid gap-0.5 text-[10.5px] text-faint">
          <span>
            {u.role ? `${u.role} · ` : ""}
            {u.model ?? ""}
            {u.acts_on ? ` · acts on ${u.acts_on}` : ""}
          </span>
          {u.measured && <span title={u.made_from}>measured: {u.measured}</span>}
          <span>
            {typeof u.version === "number" ? `v${u.version}${u.versions && u.versions > 1 ? ` of ${u.versions} kept` : ""} · ` : ""}
            {bytes(u.bytes ?? 0)}
            {u.modified_ms ? ` · ${ago(u.modified_ms, now)}` : u.made ? ` · ${u.made}` : ""}
          </span>
          {u.path && (
            <span className="truncate" title={u.path}>
              {maskPath(u.path)}
            </span>
          )}
          {u.loads && <span className="text-quiet">loads {u.loads}</span>}
        </div>
      </div>
      <div className="flex items-center gap-2 pt-0.5">
        <span className="mono hidden text-[10.5px] uppercase tracking-[0.12em] text-faint sm:inline">{u.plugged ? "plugged in" : "unplugged"}</span>
        <Switch on={u.plugged} busy={busy} label={`${u.plugged ? "Unplug" : "Plug in"} ${u.id}`} onChange={onPlug} />
      </div>
    </li>
  );
}

function CompartmentBlock({ c, current, busyKey, act, now }: { c: Compartment; current: boolean; busyKey: string | null; act: (body: Record<string, unknown>, key: string) => void; now: number }) {
  const acting = c.units.filter((u) => u.acts).length;
  const head = (
    <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
      <span>{compartmentName(c)}</span>
      <span className="mono text-[10.5px] normal-case tracking-normal text-faint">
        {c.units.length === 0 ? "no units yet" : `${acting} of ${c.units.length} acting`}
        {c.on ? "" : " · switched off"}
      </span>
    </span>
  );
  const body = (
    <div className="pt-3" data-e2e="compartment" data-key={c.key}>
      <div className="flex items-start justify-between gap-4 border-b border-line/60 pb-3">
        <div className="min-w-0">
          <p className="read text-[11.5px] leading-relaxed text-muted">{c.holds}.</p>
          {c.root && (
            <p className="mono mt-0.5 truncate text-[10.5px] text-faint" title={c.root}>
              {maskPath(c.root)}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="mono hidden text-[10.5px] uppercase tracking-[0.12em] text-faint sm:inline">{c.on ? "compartment on" : "compartment off"}</span>
          <Switch on={c.on} busy={busyKey === c.switch_key} label={`Switch ${c.on ? "off" : "on"} every learned unit of ${compartmentName(c)}`} onChange={(on) => act({ compartment: c.key, on }, c.switch_key)} />
        </div>
      </div>
      {c.units.length === 0 ? (
        <p className="mono py-3 text-[11px] text-faint">Nothing learned in this compartment yet. Units appear here once one passes its own gate.</p>
      ) : (
        <ul className="grid">
          {c.units.map((u) => (
            <UnitRow key={u.key} u={u} now={now} busy={busyKey === u.switch_key} onPlug={(on) => act({ unit: u.key, on }, u.switch_key)} />
          ))}
        </ul>
      )}
    </div>
  );
  return (
    <Fold title={head} open={current || c.units.length > 0} id={`compartment-${c.key}`}>
      {body}
    </Fold>
  );
}

/** The Learned units panel: every compartment, the current project's first. */
export default function LearnedUnits({ projectKey }: { projectKey: string | null }) {
  const reply = useApi<LearnedUnitsReply>("/api/learned-units", 10000);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [note, setNote] = useState<{ text: string; tone: "verified" | "error" } | null>(null);
  const [fresh, setFresh] = useState<LearnedUnitsReply | null>(null);
  const now = useNow();
  const data = fresh ?? reply.data;

  const act = async (body: Record<string, unknown>, key: string) => {
    setBusyKey(key);
    setNote(null);
    try {
      const r = await post<{ audit_id: string | null; learned_units: LearnedUnitsReply }>("/api/learned-units", body);
      setFresh(r.learned_units);
      const what = typeof body.unit === "string" ? (body.on ? "Plugged in" : "Unplugged") : body.on ? "Compartment switched on" : "Compartment switched off";
      setNote({ text: r.audit_id ? `${what}; applies at the next call of the serving process. Audit ${r.audit_id} undoes it.` : "Already so; nothing changed.", tone: "verified" });
      // The shared poller catches up, then its reply takes over again.
      void refreshAll("/api/learned-units").then(() => setFresh(null));
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    } finally {
      setBusyKey(null);
    }
  };

  const comps = [...(data?.compartments ?? [])].sort((a, b) => {
    const rank = (c: Compartment) => (c.key === projectKey ? 0 : c.units.length > 0 ? 1 : c.key === "nano" || c.key === "shared" ? 2 : 3);
    return rank(a) - rank(b) || a.key.localeCompare(b.key);
  });
  const empty = comps.filter((c) => c.units.length === 0 && c.key !== projectKey);
  const shown = comps.filter((c) => c.units.length > 0 || c.key === projectKey);

  return (
    <Panel
      title={<span title="Everything the Nano has learned, kept beside the frozen models and never merged into them. Each unit sits in the compartment it was fitted in, the only one it loads in. Unplug a unit, or switch a whole compartment off, and the outputs are the base's again, bit for bit.">Learned units</span>}
      right={data ? <span className="mono text-[11px] text-faint" data-e2e="learned-units-count">{data.acting} of {data.total} acting</span> : undefined}
    >
      {reply.error && !data ? (
        <Empty title="The learned units did not answer">{reply.error}</Empty>
      ) : !data ? (
        <div className="grid gap-2 py-2">
          <Placeholder className="h-3 w-2/3" />
          <Placeholder className="h-3 w-1/2" />
        </div>
      ) : (
        <div className="grid gap-4" data-e2e="learned-units">
          {shown.map((c) => (
            <CompartmentBlock key={c.key} c={c} current={c.key === projectKey} busyKey={busyKey} act={act} now={now} />
          ))}
          {empty.length > 0 && (
            <Fold title={<span>Compartments with nothing learned yet <span className="mono text-[10.5px] normal-case tracking-normal text-faint">{empty.length}</span></span>}>
              <div className="grid gap-4 pt-3">
                {empty.map((c) => (
                  <CompartmentBlock key={c.key} c={c} current={false} busyKey={busyKey} act={act} now={now} />
                ))}
              </div>
            </Fold>
          )}
          <p className="mono text-[10.5px] text-faint">
            Applies {data.applies}. State: <span title={data.file}>{maskPath(data.file)}</span>
          </p>
        </div>
      )}
      {note && <p className={`mono mt-3 text-[11px] ${note.tone === "error" ? "text-error" : "text-verified"}`} role="status">{note.text}</p>}
    </Panel>
  );
}
