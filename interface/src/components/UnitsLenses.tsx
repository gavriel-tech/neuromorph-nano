"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel."): the explanation moved into
// the heading's tooltip; this panel's home is the Cartridges side panel (Learned), where it sits
// beside the decks the units dock on.
// 2026-11-26 (audit A5 and A6; Luis Jake Gabriel III: "plug in and unplug the learned units"):
// the registered learning units (`/api/units`: low-rank adapters beside one node of a MicroModel)
// and the project lenses (`/api/lenses`: the embedder's per-project adapter), each with its own
// controls. Plugging a lens in or out is the same learned-unit plug the Learned units panel uses;
// a unit loads through its registry with a hash check. Removals go through the two-step delete flow.

import { useState } from "react";
import DeleteFlow from "@/components/DeleteFlow";
import { NoteLine } from "@/components/Controls";
import { Empty, Panel, Pill, Placeholder, Switch } from "@/components/ui";
import { ago, bytes, post, refreshAll, useApi } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { maskName, maskPath } from "@/lib/privacy";

type Note = { text: string; tone: "verified" | "error" };

type UnitRow = {
  id: string;
  base: string;
  base_sha256: string;
  model_sha256: string | null;
  base_compatible: boolean | null;
  slot: string;
  target: string;
  rank: number;
  scale: string | number;
  status: string;
  lineage: string[];
  loaded: boolean;
  files: number;
  files_verified: boolean;
  bytes: number;
  path: string;
  registered_at: number;
  unregister: string;
};
type UnitsReply = { units: UnitRow[]; registry: string; shared_on: boolean; register: string; applies: string };

type LensVersion = { version: number; path: string; bytes: number };
type LensRow = {
  key: string;
  id: string;
  compartment: string;
  plugged: boolean;
  acts: boolean;
  why_not: string | null;
  does?: string;
  model?: string;
  measured?: string;
  made?: string;
  version?: number;
  versions?: number;
  bytes?: number;
  version_files?: LensVersion[];
};
type LensesReply = { project: string | null; lenses: LensRow[]; serving_switch: string; lens_on: boolean; applies: string };

const delBtn = "press mono border-b border-line px-0.5 text-[10px] uppercase tracking-[0.12em] text-faint hover:text-error disabled:opacity-40";

/** The registered learning units: what each is, where it attaches, whether it fits its MicroModel,
 *  and load, unload or unregister. */
export function UnitsPanel() {
  const q = useApi<UnitsReply>("/api/units", 10000);
  const [note, setNote] = useState<Note | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const now = useNow();
  const act = async (id: string, action: "load" | "unload") => {
    setBusy(id);
    try {
      const r = await post<{ audit_id?: string | null }>("/api/units", { id, action });
      setNote({ text: r.audit_id ? `Unit ${id} ${action === "load" ? "plugged in" : "unplugged"}; a server picks it up at its next start. Audit ${r.audit_id} undoes it.` : `Unit ${id} was already ${action === "load" ? "plugged in" : "unplugged"}.`, tone: "verified" });
      for (const p of ["/api/units", "/api/learned-units", "/api/cartridges", "/api/audit"]) refreshAll(p);
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  const d = q.data;
  return (
    <Panel title={<span title="A small low-rank adapter fitted from checked outcomes and kept beside one node of a MicroModel: its correction is added to that node's output and never merged into it, so unplugging gives the base back bit for bit. A unit plugs in only when its base fits by hash.">Learning units</span>} right={d ? <span className="mono text-[11px] text-faint">{d.units.filter((u) => u.loaded).length} of {d.units.length} plugged in{d.shared_on ? "" : " · compartment off"}</span> : undefined}>
      <NoteLine note={note} />
      {!d && q.error ? (
        <Empty title="The learning units did not answer">{q.error}</Empty>
      ) : !d ? (
        <Placeholder className="h-16" what="learning units" />
      ) : d.units.length === 0 ? (
        <Empty title="No learning unit registered yet" slot>
          {d.register}. Registered units appear here with their target, rank and checks.
        </Empty>
      ) : (
        <ul className="grid" data-e2e="units">
          {d.units.map((u) => (
            <li key={u.id} className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-4 border-b border-line/60 py-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                  <span className="mono text-[12px] text-text">{u.id}</span>
                  <Pill tone={u.status === "fitted" ? "verified" : u.status === "rejected" || u.status === "rolled-back" ? "error" : "muted"}>{u.status}</Pill>
                  <Pill tone={u.base_compatible === true ? "verified" : u.base_compatible === false ? "attention" : "muted"}>{u.base_compatible === true ? "fits its base" : u.base_compatible === false ? "base changed" : "base not checked"}</Pill>
                  <Pill tone={u.files_verified ? "verified" : u.files ? "error" : "muted"}>{u.files ? (u.files_verified ? "files match" : "a file fails its hash") : "unfitted"}</Pill>
                </div>
                <div className="mono mt-1.5 grid gap-0.5 text-[10.5px] text-faint">
                  <span>{u.slot} · {u.base} · attaches beside {u.target}</span>
                  <span>rank {u.rank} · scale {u.scale} · {u.files} {u.files === 1 ? "file" : "files"} · {bytes(u.bytes)} · registered {u.registered_at ? ago(u.registered_at, now) : "at an unknown time"}</span>
                  <span className="truncate" title={u.path}>{maskPath(u.path)}</span>
                </div>
              </div>
              <div className="flex items-center gap-3 pt-0.5">
                <Switch on={u.loaded} busy={busy === u.id} disabled={!u.files} label={`${u.loaded ? "Unplug" : "Plug in"} learning unit ${u.id}`} onChange={(on) => void act(u.id, on ? "load" : "unload")} />
                {u.files === 0 && (
                  <DeleteFlow<{ audit_id?: string }>
                    path="/api/units"
                    body={{ action: "unregister", id: u.id }}
                    label="unregister"
                    title="Takes an unfitted unit out of the registry; Audit can register it again."
                    startClass={delBtn}
                    done={() => `Unit ${u.id} unregistered; its card is kept in Audit, where it can be registered again.`}
                    refresh={["/api/units", "/api/learned-units"]}
                    onNote={setNote}
                  />
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
      {d && <p className="mono mt-3 text-[10.5px] text-faint">{d.applies}. Registry: <span title={d.registry}>{maskPath(d.registry)}</span></p>}
    </Panel>
  );
}

/** The project lenses: each plugged in or out (the learned-unit plug), and removed, one version or
 *  all, through the delete flow. */
export function LensesPanel({ project = null, compact = false }: { project?: string | null; compact?: boolean }) {
  const q = useApi<LensesReply>(project ? `/api/lenses?project=${encodeURIComponent(project)}` : "/api/lenses", 10000);
  const [note, setNote] = useState<Note | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const act = async (l: LensRow, on: boolean) => {
    setBusy(l.id);
    try {
      const r = await post<{ audit_id?: string | null }>("/api/lenses", { id: l.id, action: on ? "enable" : "disable" });
      setNote({ text: r.audit_id ? `Lens ${l.id} ${on ? "plugged in" : "unplugged"}; applies at the next call. Audit ${r.audit_id} undoes it.` : "Already so; nothing changed.", tone: "verified" });
      for (const p of ["/api/lenses", "/api/learned-units", "/api/audit"]) refreshAll(p);
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  const d = q.data;
  return (
    <Panel title={<span title={compact ? undefined : "Fitted at idle from one project's own material (commit messages to the functions they changed, doc comments hidden from their definitions). It reshapes only the question's vector toward that project's items, loads only while its own project is served, and is kept only if it beat a shuffled-label control on held-out files."}>Project lenses</span>} right={d ? <span className="mono text-[11px] text-faint">{d.lenses.filter((l) => l.acts).length} of {d.lenses.length} acting{d.lens_on ? "" : ` · ${d.serving_switch}=off`}</span> : undefined}>
      <NoteLine note={note} />
      {!d && q.error ? (
        <Empty title="The lenses did not answer">{q.error}</Empty>
      ) : !d ? (
        <Placeholder className="h-16" what="lenses" />
      ) : d.lenses.length === 0 ? (
        <Empty title="No lens fitted yet" slot>
          A lens appears once the idle fit makes one for a project and it passes its own control.
        </Empty>
      ) : (
        <ul className="grid" data-e2e="lenses">
          {d.lenses.map((l) => (
            <li key={l.key} className="grid gap-2 border-b border-line/60 py-3">
              <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-4">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                    <span className="mono truncate text-[12px] text-text" title={l.id}>{l.id}</span>
                    <span className="mono text-[10.5px] text-faint">project {maskName(l.compartment)}</span>
                    {l.acts ? <Pill tone="verified">acting</Pill> : <Pill tone="muted" title={l.why_not ?? undefined}>{l.why_not ?? "not acting"}</Pill>}
                  </div>
                  {l.measured && <div className="mono mt-1 text-[10.5px] text-faint">measured: {l.measured}</div>}
                </div>
                <Switch on={l.plugged} busy={busy === l.id} label={`${l.plugged ? "Unplug" : "Plug in"} lens ${l.id}`} onChange={(on) => void act(l, on)} />
              </div>
              <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
                {(l.version_files ?? []).map((v) => (
                  <span key={v.version} className="mono flex items-center gap-2 text-[10.5px] text-faint">
                    <span className={v.version === l.version ? "text-text" : ""} title={v.path}>
                      v{v.version} · {bytes(v.bytes)}
                    </span>
                    {(l.version_files ?? []).length > 1 && (
                      <DeleteFlow<{ audit_id?: string }>
                        path="/api/lenses"
                        body={{ action: "remove", id: l.id, version: v.version }}
                        label="remove"
                        title={`Asks what would go first; removes v${v.version} only.`}
                        startClass={delBtn}
                        done={() => `Lens ${l.id} v${v.version} removed. Logged in Audit; the idle fit can make a lens again from the project's material.`}
                        refresh={["/api/lenses", "/api/learned-units"]}
                        onNote={setNote}
                      />
                    )}
                  </span>
                ))}
                <DeleteFlow<{ audit_id?: string }>
                  path="/api/lenses"
                  body={{ action: "remove", id: l.id }}
                  label="remove every version"
                  title="Asks what would go first; removes every version of this lens."
                  startClass={delBtn}
                  done={() => `Lens ${l.id} removed with every version. Logged in Audit; the idle fit can make one again from the project's material.`}
                  refresh={["/api/lenses", "/api/learned-units"]}
                  onNote={setNote}
                />
                {l.made && <span className="mono text-[10.5px] text-quiet">made {l.made}</span>}
              </div>
            </li>
          ))}
        </ul>
      )}
      {d && <p className="mono mt-3 text-[10.5px] text-faint">Applies {d.applies}.</p>}
    </Panel>
  );
}

