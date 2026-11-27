"use client";

// 2026-11-25 (directions of Luis Jake Gabriel III: the MicroModels are choosable, changeable and
// swappable per role, and their routing is customizable, in the code and the interface). The roles
// panel shows the model each role runs on and lets the user put another installed model of the
// right kind there (from the next start); the routing panel sets who decides each kind of close
// call, when the Core Brain asks for more context and how the tentacles judge. Both go through the
// engine's API, so MCP (nm_settings roles, role_set, set) and the command line show the same.
//
// 2026-11-26 (Luis Jake Gabriel III, his words: "the right menu are libraries and or discription
// tools etc. left menu are the main interface nav like it is now."): roles and routing are a tab of
// the Models side panel now, so both lists run in one narrow column and each routing explanation
// sits under a fold. ModuleSwitches (on the Cartridges page) is unchanged.

import { BaseMark, WeightsMark, useServingBases } from "@/components/MicroModel";
import { useState } from "react";
import { Panel, Pill, Placeholder, Segmented } from "@/components/ui";
import { post, refreshAll, useApi, type Control } from "@/lib/api";

type Choice = { id: string; ok: boolean; why: string | null };
type RoleRow = { role: string; model: string; default: string; needs: string; locked: string | null; choices: Choice[] };
type RolesPayload = { applies?: string; roles?: RoleRow[]; models?: { id: string; kind: string; role: string | null }[]; error?: string };
type Note = { text: string; tone: "verified" | "error" };

export function RolesPanel({ onNote }: { onNote: (n: Note) => void }) {
  const q = useApi<RolesPayload>("/api/models/roles", 10000);
  const servingBases = useServingBases();
  const [busy, setBusy] = useState<string | null>(null);
  const pick = async (role: string, model: string) => {
    setBusy(role);
    try {
      const r = await post<{ line?: string }>("/api/models/roles", { role, model });
      for (const p of ["/api/models/roles", "/api/cartridges/compat", "/api/audit"]) refreshAll(p);
      onNote({ text: `${r.line ?? `the ${role} runs on ${model} from the next start`}. Logged in Audit.`, tone: "verified" });
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  return (
    <Panel title="Roles" right={<span className="mono text-[10.5px] text-faint">which MicroModel runs each job · next start</span>}>
      {!q.data && q.error ? (
        <div className="mono text-[11px] text-error">The roles did not answer: {q.error}</div>
      ) : !q.data ? (
        <Placeholder className="h-24" what="roles" />
      ) : q.data.error ? (
        <div className="mono text-[11px] text-error">{q.data.error}</div>
      ) : (
        <ul className="grid gap-y-5" data-e2e="model-roles">
          {(q.data.roles ?? []).map((r) => {
            const swapped = r.model !== r.default;
            return (
              <li key={r.role} className="min-w-0">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="mono text-[13px] uppercase tracking-[0.2em] text-text">{r.role}</span>
                  {r.locked ? <Pill>locked</Pill> : swapped ? <Pill tone="learned">chosen</Pill> : <Pill tone="verified">default</Pill>}
                </div>
                <div className="mt-1 flex flex-wrap items-baseline gap-x-2.5">
                  <BaseMark base={servingBases.get(r.model)} />
                  <WeightsMark model={r.model} />
                  {(q.data?.roles ?? []).some((o) => o.role !== r.role && o.model === r.model) ? <span className="mono text-[10.5px] text-faint">shared with the {(q.data?.roles ?? []).filter((o) => o.role !== r.role && o.model === r.model).map((o) => o.role).join(", ")}</span> : null}
                </div>
                <div className="mono mt-0.5 text-[10.5px] text-faint">needs {/^[aeiou]/.test(r.needs) ? "an" : "a"} {r.needs} model</div>
                {r.locked ? (
                  <p className="read mt-2 text-[11px] leading-relaxed text-faint">{r.locked}</p>
                ) : (
                  <label className="mt-2 block">
                    <span className="sr-only">Model for the {r.role}</span>
                    <select
                      value={r.model}
                      disabled={busy !== null}
                      onChange={(e) => void pick(r.role, e.target.value === r.default ? "default" : e.target.value)}
                      className="mono h-9 w-full border-b border-line-strong bg-transparent text-[12px] text-text outline-none focus-visible:border-sense-pale disabled:opacity-50"
                    >
                      {r.choices.map((c) => (
                        <option key={c.id} value={c.id} disabled={!c.ok && c.id !== r.model} className="bg-bg-1">
                          {c.id}
                          {c.id === r.default ? " (default)" : ""}
                          {!c.ok && c.id !== r.model && c.why ? ` · ${c.why}` : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                {!r.locked && r.choices.filter((c) => c.ok && c.id !== r.model).length === 0 && (
                  <p className="mono mt-1.5 text-[10.5px] text-quiet">no other installed {r.needs} model; add one to the models folder to swap</p>
                )}
              </li>
            );
          })}
        </ul>
      )}
      <p className="mono mt-4 text-[10.5px] text-quiet">a swapped role runs with its new model&apos;s cartridges; every cartridge stays plugged into the model it was made for, ready when the role goes back; the measured routing record holds only for the models it was measured on</p>
    </Panel>
  );
}

const ROUTING: { key: string; render?: (v: string) => string }[] = [
  { key: "routing.code" },
  { key: "routing.prose" },
  { key: "routing.ask" },
  { key: "routing.record" },
  { key: "routing.tentacle_judge" },
];

export function RoutingPanel({ onNote }: { onNote: (n: Note) => void }) {
  const q = useApi<Control[]>("/api/controls");
  const [busy, setBusy] = useState<string | null>(null);
  const set = async (key: string, value: string) => {
    setBusy(key);
    try {
      await post("/api/controls", { key, value });
      refreshAll("/api/controls");
      refreshAll("/api/audit");
      onNote({ text: `${key} set to ${value}. Logged in Audit, where it can be undone.`, tone: "verified" });
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  const rows = ROUTING.map((r) => q.data?.find((c) => c.key === r.key)).filter((c): c is Control => !!c);
  return (
    <Panel title="Routing" right={<span className="mono text-[10.5px] text-faint">who decides what · undo in Audit</span>}>
      {!q.data && q.error ? (
        <div className="mono text-[11px] text-error">The routing controls did not answer: {q.error}</div>
      ) : !q.data ? (
        <Placeholder className="h-24" what="routing" />
      ) : rows.length === 0 ? (
        <div className="mono text-[11px] text-faint">The routing controls come with a newer engine build.</div>
      ) : (
        <ul className="grid gap-y-5" data-e2e="model-routing">
          {rows.map((c) => (
            <li key={c.key} className="min-w-0">
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <span className="text-[13px] text-text">{c.label}</span>
                <span className="mono text-[10.5px] text-faint">{c.value === c.default ? "default" : "set"} · {c.applies}</span>
              </div>
              <div className="mt-2 overflow-x-auto">
                <Segmented label={c.label} values={c.values} value={c.value} busy={busy === c.key} onPick={(v) => void set(c.key, v)} />
              </div>
              {c.help && (
                <details className="group mt-1">
                  <summary className="press mono inline-flex min-h-7 cursor-pointer list-none items-center text-[10px] uppercase tracking-[0.12em] text-faint hover:text-text [&::-webkit-details-marker]:hidden">what this does</summary>
                  <p className="read mt-1 text-[11px] leading-relaxed text-faint">{c.help}</p>
                </details>
              )}
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

/** The precision control of each role (`precision.base` is the reader's). */
const PRECISION_KEY: Record<string, string> = { reader: "precision.base", writer: "precision.writer", embedder: "precision.embedder", reranker: "precision.reranker" };

/** One MicroModel's own switches, carried on its deck (2026-11-25, directions of Luis Jake Gabriel
 *  III: the switches move with the MicroModels; a modular system): the model the role runs on and
 *  its precision, the same controls as the Models page and MCP. */
export function ModuleSwitches({ role, onNote }: { role: string; onNote: (n: Note) => void }) {
  const roles = useApi<RolesPayload>("/api/models/roles", 10000);
  const controls = useApi<Control[]>("/api/controls");
  const [busy, setBusy] = useState(false);
  const r = roles.data?.roles?.find((x) => x.role === role);
  const prec = controls.data?.find((c) => c.key === PRECISION_KEY[role]);
  const run = async (path: string, body: unknown, done: (reply: { line?: string }) => string, refresh: string[]) => {
    setBusy(true);
    try {
      const reply = await post<{ line?: string }>(path, body);
      for (const p of [...refresh, "/api/audit"]) refreshAll(p);
      onNote({ text: done(reply), tone: "verified" });
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  if (!r) return null;
  return (
    <div className="grid gap-2 border border-line/70 bg-bg-1/60 px-2.5 py-2" data-e2e={`module-${role}`}>
      <label className="mono flex min-w-0 items-center justify-between gap-2 text-[10px] uppercase tracking-[0.12em] text-faint">
        <span className="shrink-0">model</span>
        {r.locked ? (
          <span className="truncate normal-case tracking-normal text-muted" title={r.locked}>
            {r.model} · locked
          </span>
        ) : (
          <select
            value={r.model}
            disabled={busy}
            aria-label={`Model for the ${role}`}
            onChange={(e) =>
              void run("/api/models/roles", { role, model: e.target.value === r.default ? "default" : e.target.value }, (x) => `${x.line ?? `the ${role} runs on ${e.target.value} from the next start`}. Logged in Audit.`, ["/api/models/roles", "/api/cartridges/compat"])
            }
            className="mono min-w-0 flex-1 truncate border-b border-line-strong bg-transparent text-right text-[11px] normal-case tracking-normal text-text outline-none focus-visible:border-sense-pale disabled:opacity-50"
          >
            {r.choices.map((c) => (
              <option key={c.id} value={c.id} disabled={!c.ok && c.id !== r.model} className="bg-bg-1">
                {c.id}
                {c.id === r.default ? " (default)" : ""}
              </option>
            ))}
          </select>
        )}
      </label>
      {prec && (
        <div className="mono flex min-w-0 items-center justify-between gap-2 text-[10px] uppercase tracking-[0.12em] text-faint">
          <span className="shrink-0">precision</span>
          <div className="min-w-0 overflow-x-auto">
            <Segmented
              label={`Precision of the ${role}`}
              values={prec.values}
              value={prec.value}
              busy={busy}
              render={(v) => (v === "default" ? "auto" : `${v} bit`)}
              onPick={(v) => void run("/api/controls", { key: prec.key, value: v }, () => `The ${role}'s precision set to ${v === "default" ? "auto" : `${v} bits`} (${prec.applies}). Logged in Audit, where it can be undone.`, ["/api/controls"])}
            />
          </div>
        </div>
      )}
    </div>
  );
}
