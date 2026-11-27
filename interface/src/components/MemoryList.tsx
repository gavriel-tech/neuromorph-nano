"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And
// that whole information database for it should be on a right side panel."): this panel now sits
// in a rail tab; its explanation is one faint line, the rest in tooltips.
// 2026-11-26 (audit B1): add a memory (kept, temporary, or a working note) through the same write
// gate and never-save screen MCP `nm_memory` uses, pin or unpin one, and sweep the temporary notes.
// 2026-11-26 (Luis Jake Gabriel III: delete a single memory, and every delete verified in more
// than one step). The project's memories as the engine's tier views hold them, searchable, each
// with a delete through the delete flow. A deleted memory leaves recall and every view; by design
// the memory journal keeps a record that it existed, and forgetting the project erases everything.

import { useState } from "react";
import DeleteFlow, { type DeleteNote } from "@/components/DeleteFlow";
import { NoteLine } from "@/components/Controls";
import { Empty, Panel, Pill, Placeholder } from "@/components/ui";
import { ago, post, refreshAll, useApi } from "@/lib/api";
import { useNow } from "@/lib/clock";

type Item = { id: string; tier: string; topic: string; text: string; state: string; class: string; recorded: number | null };
type Reply = { project?: string; items?: Item[]; total?: number; error?: string };

const SHOW = 50;

type WriteReply = { applied?: string; note?: string; audit_id?: string; result?: { admitted?: boolean; reason?: string; why?: string; promoted?: unknown[]; cleared?: number } };

const KINDS: { id: "store" | "temp" | "note"; label: string; about: string }[] = [
  { id: "store", label: "keep", about: "kept as a memory once it passes the relevance gate; credentials, keys, tokens and e-mail addresses are refused" },
  { id: "temp", label: "temporary", about: "a temporary note: saved when used on three run-days, cleared when unused for seven" },
  { id: "note", label: "working note", about: "a note for the serving session only" },
];

/** The add row: the kind, the text, and the engine's answer (kept, queued or refused). */
function AddMemory({ projectKey, onNote }: { projectKey: string; onNote: (n: DeleteNote) => void }) {
  const [kind, setKind] = useState<"store" | "temp" | "note">("store");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const about = KINDS.find((k) => k.id === kind)?.about ?? "";
  const add = async () => {
    setBusy(true);
    try {
      const r = await post<WriteReply>("/api/memory/add", { project: projectKey, kind, text: text.trim() });
      const refused = r.result?.admitted === false;
      onNote(
        refused
          ? { text: `Not kept: the gate refused it${r.result?.reason || r.result?.why ? ` (${r.result?.reason ?? r.result?.why})` : ""}. Logged in Audit, without the text.`, tone: "error" }
          : { text: `${r.note ?? "Added"}. Logged in Audit (the kind and id, never the text).`, tone: "verified" },
      );
      if (!refused) setText("");
      for (const pth of ["/api/memory", "/api/audit"]) refreshAll(pth);
    } catch (e) {
      onNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(false);
  };
  return (
    <div className="mb-4 grid gap-2 border-b border-line/60 pb-4" data-e2e="memory-add">
      <div className="flex flex-wrap items-center gap-3">
        <span className="eyebrow">Add a memory</span>
        <div role="radiogroup" aria-label="Kind of memory" className="flex gap-1">
          {KINDS.map((k) => (
            <button key={k.id} type="button" role="radio" aria-checked={kind === k.id} onClick={() => setKind(k.id)} className={`press mono min-h-8 border-b px-1.5 text-[10px] uppercase tracking-[0.1em] ${kind === k.id ? "border-sense-pale text-text" : "border-line text-faint hover:text-text"}`}>
              {k.label}
            </button>
          ))}
        </div>
      </div>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={2} maxLength={4000} placeholder="a decision, a convention, a fact about this project to remember" className="mono w-full resize-y border-b border-line-strong bg-transparent px-1 py-1 text-xs text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale" />
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="read text-[11px] text-faint">{about}</span>
        <button type="button" disabled={busy || !text.trim()} onClick={() => void add()} className="press mono min-h-8 border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40">
          {busy ? "adding" : "add"}
        </button>
      </div>
    </div>
  );
}

export function MemoryList({ projectKey }: { projectKey: string | null }) {
  const [q, setQ] = useState("");
  const [more, setMore] = useState(false);
  const [note, setNote] = useState<DeleteNote | null>(null);
  const [deleting, setDeleting] = useState<Item | null>(null);
  const now = useNow();
  const url = projectKey ? `/api/memory/items?project=${encodeURIComponent(projectKey)}&limit=2000${q.trim() ? `&q=${encodeURIComponent(q.trim())}` : ""}` : null;
  const r = useApi<Reply>(url, 30000);
  const [pinBusy, setPinBusy] = useState<string | null>(null);
  const [sweepBusy, setSweepBusy] = useState(false);
  const pin = async (m: Item, on: boolean) => {
    if (!projectKey) return;
    setPinBusy(m.id);
    try {
      const x = await post<WriteReply>("/api/memory/pin", { project: projectKey, id: m.id, pin: on });
      setNote({ text: `${on ? "Pinned" : "Unpinned"}: ${x.note ?? "done"}. Audit undoes it.`, tone: "verified" });
      for (const pth of ["/api/memory", "/api/audit"]) refreshAll(pth);
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setPinBusy(null);
  };
  const sweep = async () => {
    if (!projectKey) return;
    setSweepBusy(true);
    try {
      const x = await post<WriteReply>("/api/memory/sweep", { project: projectKey });
      const promoted = Array.isArray(x.result?.promoted) ? x.result.promoted.length : 0;
      setNote({ text: `Temporary notes swept: ${promoted} saved, ${x.result?.cleared ?? 0} cleared. ${x.note ?? ""} Logged in Audit.`.trim(), tone: "verified" });
      for (const pth of ["/api/memory", "/api/audit"]) refreshAll(pth);
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setSweepBusy(false);
  };
  const items = r.data?.items ?? [];
  const shown = more ? items : items.slice(0, SHOW);
  return (
    <Panel
      title="What the Nano remembers"
      right={
        <span className="flex items-center gap-4">
          {r.data && <span className="mono text-[11px] text-faint">{r.data.total ?? items.length} memories</span>}
          {projectKey && (
            <button type="button" disabled={sweepBusy} onClick={() => void sweep()} title="Saves temporary notes used on three run-days and clears those unused for seven" className="press mono min-h-8 border-b border-line-strong px-1 text-[10px] uppercase tracking-[0.12em] text-muted hover:text-text disabled:opacity-40">
              {sweepBusy ? "sweeping" : "sweep temporary notes"}
            </button>
          )}
        </span>
      }
    >
      <p className="mono mb-3 text-[10.5px] text-faint" title="Deleting a memory removes it from what the Nano recalls and shows. The memory journal keeps a record that it existed; to erase a project's memory completely, forget the project (Storage).">
        a delete leaves recall; the journal keeps that it existed
      </p>
      <NoteLine note={note} />
      {!projectKey ? (
        <Empty title="No project picked" />
      ) : r.error && !r.data ? (
        <Empty title={r.error.startsWith("404") ? "Not served by this engine build" : "The memories did not answer"}>{r.error}</Empty>
      ) : r.data?.error ? (
        <Empty title="The memories did not answer">{r.data.error}</Empty>
      ) : !r.data ? (
        <Placeholder className="h-24" what="memory items" />
      ) : (
        <>
          <AddMemory projectKey={projectKey} onNote={setNote} />
          <label className="mb-3 block">
            <span className="sr-only">Search the memories</span>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="search the memories" className="mono h-10 w-full border-b border-line-strong bg-transparent px-1 text-xs text-text placeholder:text-quiet focus-visible:border-sense-pale" />
          </label>
          {deleting && (
            <div className="mb-3">
              <DeleteFlow<{ applied?: string; note?: string }>
                key={deleting.id}
                path="/api/memory/forget"
                body={{ project: projectKey, id: deleting.id }}
                label="Delete"
                autoStart
                done={(x) => x?.note ?? "Memory deleted. Logged in Audit."}
                refresh={["/api/memory", "/api/projects"]}
                onNote={setNote}
                onClose={() => setDeleting(null)}
              />
            </div>
          )}
          {items.length === 0 ? (
            <Empty title={q.trim() ? "No memory holds that" : "Nothing remembered yet"} />
          ) : (
            <ul className="grid gap-1" data-e2e="memory-items">
              {shown.map((m) => (
                <li key={m.id} className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3 border-b border-line/60 py-1.5">
                  <span className="min-w-0">
                    <span className="block break-words text-xs text-muted">{m.text}</span>
                    <span className="mono mt-0.5 flex flex-wrap items-center gap-2 text-[10px] text-faint">
                      <Pill>{m.tier}</Pill>
                      {m.topic}
                      {m.class === "pinned" ? " · pinned" : ""}
                      {m.recorded && now > 0 ? ` · ${ago(m.recorded, now)}` : ""}
                    </span>
                  </span>
                  <span className="flex items-center gap-3">
                  <button
                    type="button"
                    disabled={pinBusy !== null}
                    onClick={() => void pin(m, m.class !== "pinned")}
                    title={m.class === "pinned" ? "Unpin: it may move between tiers by use again" : "Pin: it never drops, merges or leaves its tier by itself"}
                    className={`press mono min-h-9 text-[10.5px] uppercase tracking-[0.1em] disabled:opacity-30 ${m.class === "pinned" ? "text-sense-pale hover:text-text" : "text-faint hover:text-text"}`}
                  >
                    {pinBusy === m.id ? "..." : m.class === "pinned" ? "Unpin" : "Pin"}
                  </button>
                  <button
                    type="button"
                    disabled={deleting !== null}
                    onClick={() => setDeleting(m)}
                    title="Delete this memory: the engine shows it, then asks you to type delete"
                    className="press mono min-h-9 text-[10.5px] uppercase tracking-[0.1em] text-faint hover:text-error disabled:opacity-30"
                  >
                    Delete
                  </button>
                  </span>
                </li>
              ))}
            </ul>
          )}
          {items.length > SHOW && (
            <button type="button" onClick={() => setMore((x) => !x)} className="press mono mt-2 text-[10.5px] uppercase tracking-[0.12em] text-faint hover:text-text">
              {more ? `Show the newest ${SHOW}` : `Show all ${items.length}`}
            </button>
          )}
        </>
      )}
    </Panel>
  );
}
