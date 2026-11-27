"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And
// that whole information database for it should be on a right side panel."): this panel now sits
// in a rail tab; its explanation is one faint line, the rest in tooltips.
// 2026-11-26 (audit B2): the session goals of a project, the objectives MCP `nm_goal` reads and
// sets. Your coding agents see the open ones; add one here, mark one done or dropped, or open it
// again. Every change is an audit row the Audit page can undo.

import { useState } from "react";
import { NoteLine } from "@/components/Controls";
import { Empty, Panel, Pill, Placeholder } from "@/components/ui";
import { post, refreshAll, useApi } from "@/lib/api";

type Objective = { id: number | string; objective: string; status: string };
type Goals = { project?: string; objectives?: Objective[]; open?: number; total?: number; served?: boolean; error?: string };
type Note = { text: string; tone: "verified" | "error" };

export default function GoalsPanel({ projectKey }: { projectKey: string | null }) {
  const url = projectKey ? `/api/goals?project=${encodeURIComponent(projectKey)}` : null;
  const q = useApi<Goals>(url, 10000);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<Note | null>(null);
  const send = async (body: Record<string, unknown>, key: string, done: string) => {
    if (!projectKey) return;
    setBusy(key);
    try {
      const r = await post<{ note?: string; audit_id?: string }>("/api/goals", { project: projectKey, ...body });
      setNote({ text: `${done}${r.note ? ` ${r.note}` : ""} Audit undoes it.`, tone: "verified" });
      if (key === "add") setText("");
      for (const p of ["/api/goals", "/api/audit"]) refreshAll(p);
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  const g = q.data;
  const list = g?.objectives ?? [];
  return (
    <Panel title="Session goals" right={g && !g.error ? <span className="mono text-[11px] text-faint">{g.open ?? 0} open of {g.total ?? list.length}</span> : undefined}>
      <p className="mono mb-3 text-[10.5px] text-faint" title="What this project is working toward right now. Your coding agents read the open goals with each context pack (nm_goal) and can add or close them too.">
        read by your coding agents with each context pack
      </p>
      <NoteLine note={note} />
      {!projectKey ? (
        <Empty title="No project picked" />
      ) : !g && q.error ? (
        <Empty title="The goals did not answer">{q.error}</Empty>
      ) : !g ? (
        <Placeholder className="h-12" what="goals" />
      ) : g.error ? (
        <Empty title="Goals not available">{g.error}</Empty>
      ) : (
        <div className="grid gap-3" data-e2e="goals">
          <form
            className="flex min-w-0 items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (text.trim()) void send({ objective: text.trim() }, "add", "Goal added.");
            }}
          >
            <label className="min-w-0 flex-1">
              <span className="sr-only">New goal</span>
              <input value={text} onChange={(e) => setText(e.target.value)} maxLength={400} placeholder="a goal for this project, for example: finish the release checks" className="mono h-9 w-full border-b border-line-strong bg-transparent px-1 text-xs text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale" />
            </label>
            <button type="submit" disabled={busy !== null || !text.trim()} className="press mono min-h-9 border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40">
              add goal
            </button>
          </form>
          {list.length === 0 ? (
            <div className="mono text-[11px] text-faint">No goal set for this project yet.</div>
          ) : (
            <ul className="grid">
              {list.map((o) => {
                const id = Number(o.id);
                const open = o.status === "open";
                return (
                  <li key={String(o.id)} className="grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-3 border-b border-line/60 py-2">
                    <span className="min-w-0">
                      <span className={`block break-words text-xs ${open ? "text-muted" : "text-faint line-through"}`}>{o.objective}</span>
                      <span className="mono text-[10px] text-quiet">goal {String(o.id)}</span>
                    </span>
                    <span className="flex items-center gap-3">
                      <Pill tone={o.status === "done" ? "verified" : "muted"}>{o.status}</Pill>
                      {open ? (
                        <>
                          <button type="button" disabled={busy !== null} onClick={() => void send({ id, status: "done" }, `d${id}`, `Goal ${id} marked done.`)} className="press mono min-h-8 text-[10px] uppercase tracking-[0.1em] text-faint hover:text-text disabled:opacity-40">
                            done
                          </button>
                          <button type="button" disabled={busy !== null} onClick={() => void send({ id, status: "dropped" }, `x${id}`, `Goal ${id} dropped.`)} className="press mono min-h-8 text-[10px] uppercase tracking-[0.1em] text-faint hover:text-text disabled:opacity-40">
                            drop
                          </button>
                        </>
                      ) : (
                        <button type="button" disabled={busy !== null} onClick={() => void send({ id, status: "open" }, `o${id}`, `Goal ${id} opened again.`)} className="press mono min-h-8 text-[10px] uppercase tracking-[0.1em] text-faint hover:text-text disabled:opacity-40">
                          open again
                        </button>
                      )}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </Panel>
  );
}
