"use client";

// A live teacher for distillation (POST /api/training/distil/teach; MCP nm_learning teach_*): a
// teaching set holds the prompts the teacher answers and the instruction it follows. The user's
// agent (Claude Code, Antigravity, any MCP client, with a cloud or a local model) answers them over
// MCP, or an installed open-weights model answers them here, and the answers join the teacher data
// with the teacher named. Building from them goes through the same held-out gate.

import { useState } from "react";
import { NoteLine } from "@/components/Controls";
import { Pill } from "@/components/ui";
import { post, refreshAll, useApi, type Status } from "@/lib/api";

type Note = { text: string; tone: "verified" | "error" };
type Sets = { sets: { set: string; prompts: number; open: number; instruction: string; licence: string }[] };
const PATH = "/api/training/distil/teach";
const link = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const field = "mono min-w-0 border-b border-line-strong bg-transparent text-[12px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale";

export default function TeachPanel() {
  const q = useApi<Sets>(PATH, 10000);
  const status = useApi<Status>("/api/status", 30000);
  const [note, setNote] = useState<Note | null>(null);
  const [set, setSet] = useState("");
  const [prompts, setPrompts] = useState("");
  const [instruction, setInstruction] = useState("");
  const [licence, setLicence] = useState("");
  const [teacher, setTeacher] = useState("qwen3-1.7b");
  const [busy, setBusy] = useState<string | null>(null);
  const models = (status.data?.models ?? []).map((m) => m.id);
  const run = async (what: string, path: string, body: Record<string, unknown>, done: string) => {
    setBusy(what);
    try {
      await post(path, body);
      setNote({ text: done, tone: "verified" });
      refreshAll(PATH);
      refreshAll("/api/training/distil");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
  };
  const list = prompts.split("\n").map((p) => p.trim()).filter(Boolean);
  return (
    <div className="grid gap-4" data-e2e="teach-panel">
      <NoteLine note={note} />
      <ul className="grid" data-e2e="teach-sets">
        {(q.data?.sets ?? []).length === 0 && <li className="mono text-[10.5px] text-quiet">no teaching set yet</li>}
        {(q.data?.sets ?? []).map((st) => (
          <li key={st.set} className="grid gap-1 border-b border-line/60 py-2" data-e2e="teach-set">
            <div className="flex flex-wrap items-baseline justify-between gap-x-3">
              <span className="mono text-[12px] text-text">{st.set}</span>
              <span className="mono text-[10px] text-faint">
                {st.open} open of {st.prompts}
              </span>
            </div>
            {st.instruction && <span className="mono truncate text-[10.5px] text-muted" title={st.instruction}>{st.instruction}</span>}
            <div className="mono flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px]">
              <Pill tone="muted" title="In Claude Code, Antigravity or any MCP client: ask your agent to teach this set. It calls nm_learning teach_next, answers each prompt as the teacher, and sends them with teach_answer.">
                your agent: nm_learning teach_next, teach_answer
              </Pill>
              <select value={teacher} onChange={(e) => setTeacher(e.target.value)} aria-label="Local teacher model" className={`${field} h-7 text-[10.5px]`}>
                {(models.length ? models : ["qwen3-1.7b"]).map((m) => (
                  <option key={m} value={m} className="bg-bg-1">
                    {m}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className={link}
                disabled={!!busy || st.open === 0}
                data-e2e="teach-local"
                title="An installed open-weights model answers the open prompts, on the graphics card; its answers join the teacher data when it ends"
                onClick={() => void run(`local:${st.set}`, "/api/lab/build", { kind: "teach", role: "writer", set: st.set, teacher }, `${teacher} is answering the ${st.open} open prompts of ${st.set}; follow it in Live Activity.`)}
              >
                {busy === `local:${st.set}` ? "starting" : "teach with a local model"}
              </button>
            </div>
          </li>
        ))}
      </ul>
      <div className="grid gap-2 border-t border-line/60 pt-3" data-e2e="teach-start">
        <span className="eyebrow">open a teaching set</span>
        <input value={set} onChange={(e) => setSet(e.target.value)} placeholder="set name, for example notes-style" aria-label="Set name" className={`${field} h-9`} spellCheck={false} />
        <textarea value={prompts} onChange={(e) => setPrompts(e.target.value)} placeholder="the prompts the teacher answers, one per line" aria-label="Prompts" rows={4} className={`${field} py-1`} />
        <input value={instruction} onChange={(e) => setInstruction(e.target.value)} placeholder="what the teacher does in every answer (shown to the teacher only)" aria-label="Instruction" className={`${field} h-9`} />
        <input value={licence} onChange={(e) => setLicence(e.target.value)} placeholder="licence of the teacher's answers, for example own" aria-label="Licence" className={`${field} h-9`} />
        <div className="flex items-center gap-3">
          <button type="button" className={link} disabled={!!busy || !set.trim() || list.length === 0} onClick={() => void run("start", PATH, { action: "start", set: set.trim(), prompts: list, instruction, licence }, `Teaching set ${set.trim()} opened with ${list.length} prompts. Ask your agent to teach it, or teach it with a local model.`)} data-e2e="teach-start-button">
            {busy === "start" ? "opening" : "open the set"}
          </button>
          <span className="mono text-[10px] text-quiet">{list.length} prompts</span>
        </div>
      </div>
    </div>
  );
}
