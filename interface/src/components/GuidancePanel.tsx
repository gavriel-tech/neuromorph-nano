"use client";

// The agent's guidance for the picked project (`/api/guidance`, the same route `nm_goal` and
// `neuromorph-nano guidance` use): reminders the hooks deliver to the coding agent at their
// triggers, loops that continue the agent at a stop until their condition holds (capped), the step
// graph with ready, blocked and done steps, and what was sent to the agent and when, each with the
// evidence ledger's verdict (a step nudge followed, a loop's condition met before its cap, a
// reminder's edits under its glob keeping a passing check). Working
// controls only; what each part means is in the rail's Guidance tab. Every change is an audit row
// the Audit page undoes; removing goes through the delete flow.
// 2026-11-27 (Luis Jake Gabriel III, his words: "The main center screen for every page should be
// easy to navigate and beautifully visual and interactive ... We should be able to visualize the
// system in a high level professional and technical way"): the tab opens on the guidance circuit
// (hook events, the three kinds of guidance, what was sent and its verdicts), then loops, the step
// graph, reminders and deliveries as its detail. Each add form opens from its part's head, so the
// view is the system and not a stack of forms; a hook event in the figure opens the reminder form
// at that event.

import { useMemo, useState, type ReactNode } from "react";
import DeleteFlow from "@/components/DeleteFlow";
import GuidanceFigure, { type GuidanceTarget } from "@/components/GuidanceFigure";
import { NoteLine } from "@/components/Controls";
import { Empty, Pill, Placeholder, Switch } from "@/components/ui";
import { RailSection } from "@/components/Workspace";
import { ago, post, refreshAll, useApi } from "@/lib/api";
import { useNow } from "@/lib/clock";
import {
  CONDITIONS,
  ERROR_COUNT_MAX,
  ERROR_COUNT_MIN,
  LOOP_CAP,
  LOOP_DEFAULT,
  NODE_H,
  NODE_W,
  PARTS,
  REPEATS,
  STALL_DEFAULT,
  STALL_MAX,
  TRIGGERS,
  budgetWords,
  channelWords,
  edgePath,
  endTone,
  endWords,
  fit,
  layout,
  listOf,
  loopTone,
  msWords,
  partWords,
  repeatWords,
  triggerWords,
  untilWords,
  verdictTally,
  verdictTone,
  verdictWords,
  whoWords,
  type Delivery,
  type GuidanceReply,
  type Loop,
  type Outcome,
  type Reminder,
  type Step,
} from "@/lib/guidance";

type Note = { text: string; tone: "verified" | "error" };
type Goals = { objectives?: { id: number | string; objective: string; status: string }[] };
type WriteReply = { note?: string; audit_id?: string };
type Send = (body: Record<string, unknown>, key: string) => Promise<boolean>;

const INPUT = "mono h-9 w-full min-w-0 border-b border-line-strong bg-transparent px-1 text-xs text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale";
const SELECT = "press mono h-9 min-w-0 appearance-none border-b border-line-strong bg-transparent px-1 text-[11px] text-text hover:border-text focus-visible:border-sense-pale";
const NUM = "mono h-9 w-16 border-b border-line-strong bg-transparent px-1 text-xs text-text outline-none focus-visible:border-sense-pale";
const ACT = "press mono min-h-8 text-[10px] uppercase tracking-[0.1em] text-faint hover:text-text disabled:opacity-40";
const ADD = "press mono min-h-9 shrink-0 border-b border-sense-pale px-1 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40";
const LABEL = "mono text-[10px] uppercase tracking-[0.12em] text-faint";

const STATUS_STROKE: Record<string, string> = { done: "var(--verified)", ready: "var(--sense-pale)", blocked: "var(--text-4)" };

type AddForm = { label: string; open: boolean; onToggle: () => void; form: ReactNode };

/** One part of the guidance: its name and figures on a hairline, the button that opens its add
 *  form, the form (kept mounted while closed, so what was typed stays), then its rows. */
function Part({ title, right, e2e, children, add }: { title: string; right?: ReactNode; e2e: string; children: ReactNode; add?: AddForm }) {
  return (
    <section className="grid min-w-0 scroll-mt-6 gap-2.5" data-e2e={e2e}>
      <div className="flex min-h-8 flex-wrap items-center justify-between gap-x-4 gap-y-1 border-b border-line pb-1">
        <h3 className="eyebrow flex items-center gap-2 text-muted">
          <span aria-hidden className="inline-block h-1 w-1 bg-sense-pale" />
          {title}
        </h3>
        <span className="flex flex-wrap items-center gap-x-4 gap-y-1">
          {right}
          {add && (
            <button type="button" onClick={add.onToggle} aria-expanded={add.open} className={`press mono min-h-8 border-b px-1 text-[10px] uppercase tracking-[0.12em] ${add.open ? "border-line-strong text-muted hover:text-text" : "border-sense-pale text-sense-pale hover:text-text"}`} data-e2e={`${e2e}-toggle`}>
              {add.open ? "close" : `+ ${add.label}`}
            </button>
          )}
        </span>
      </div>
      {add && (
        <div hidden={!add.open} className="border border-line bg-bg-1/50 px-3 pt-2 pb-3">
          {add.form}
        </div>
      )}
      {children}
    </section>
  );
}

const EMPTY = "mono border-l border-line pl-3 text-[11px] leading-relaxed text-faint";

function Field({ label, children, grow = false }: { label: string; children: ReactNode; grow?: boolean }) {
  return (
    <label className={`grid min-w-0 gap-0.5 ${grow ? "flex-1 basis-56" : ""}`}>
      <span className={LABEL}>{label}</span>
      {children}
    </label>
  );
}

export default function GuidancePanel({ projectKey }: { projectKey: string | null }) {
  const url = projectKey ? `/api/guidance?project=${encodeURIComponent(projectKey)}` : null;
  const q = useApi<GuidanceReply>(url, 5000);
  const goals = useApi<Goals>(projectKey ? `/api/goals?project=${encodeURIComponent(projectKey)}` : null, 15000);
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<Note | null>(null);
  const [removing, setRemoving] = useState<{ kind: string; id: string } | null>(null);
  const now = useNow();
  // Which add forms are open, and the hook event the reminder form starts at.
  const [open, setOpen] = useState<{ reminder: boolean; loop: boolean; step: boolean }>({ reminder: false, loop: false, step: false });
  const [trigger, setTrigger] = useState<string>("prompt");
  const toggle = (k: "reminder" | "loop" | "step") => setOpen((o) => ({ ...o, [k]: !o[k] }));
  const goTo = (t: GuidanceTarget) => document.querySelector(`[data-e2e="guidance-${t}"]`)?.scrollIntoView({ block: "start", behavior: "smooth" });

  const send: Send = async (body, key) => {
    if (!projectKey) return false;
    setBusy(key);
    let ok = false;
    try {
      const r = await post<WriteReply>("/api/guidance", { project: projectKey, ...body });
      setNote({ text: r?.note ?? "Done. Audit undoes it.", tone: "verified" });
      ok = true;
      for (const p of ["/api/guidance", "/api/audit"]) void refreshAll(p);
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy(null);
    return ok;
  };

  const g = q.data;
  const openGoals = (goals.data?.objectives ?? []).filter((o) => o.status === "open");
  const remove = (kind: string, id: string) => setRemoving({ kind, id });

  if (!projectKey) return <Empty title="No project picked" />;
  if (!g && q.error) return <Empty title="Guidance did not answer">{q.error}</Empty>;
  if (!g) return <Placeholder className="h-24" what="guidance" />;
  if (g.error) return <Empty title="Guidance not available">{g.error}</Empty>;

  return (
    <div className="grid min-w-0 gap-6" data-e2e="guidance">
      <div className="flex flex-wrap items-center gap-2">
        <Pill tone={g.settings?.reminders === "off" ? "muted" : "verified"} title="The reminders setting (Settings)">
          reminders {g.settings?.reminders ?? "on"}
        </Pill>
        <Pill tone={g.settings?.loops === "off" ? "muted" : "verified"} title="The loops setting (Settings)">
          loops {g.settings?.loops ?? "on"}
        </Pill>
        {g.settings?.memory === "off" && <Pill tone="attention">memory off: nothing is delivered</Pill>}
        <span className="mono ml-auto text-[10.5px] text-faint" title={g.file}>
          version {g.version ?? 0} · {g.delivered_total ?? 0} delivered
        </span>
      </div>
      <NoteLine note={note} />
      {removing && (
        <DeleteFlow<WriteReply>
          key={`${removing.kind}:${removing.id}`}
          path="/api/guidance"
          body={{ project: projectKey, action: "remove", kind: removing.kind, id: removing.id }}
          label="Remove"
          autoStart
          done={(r) => r?.note ?? `${removing.kind} ${removing.id} removed. Audit puts it back.`}
          refresh={["/api/guidance"]}
          onNote={setNote}
          onClose={() => setRemoving(null)}
          e2e="guidance-remove"
        />
      )}

      <GuidanceFigure
        g={g}
        onOpen={goTo}
        onTrigger={(t) => {
          setTrigger(t);
          setOpen((o) => ({ ...o, reminder: true }));
          goTo("reminders");
        }}
      />
      <div className="mono -mt-4 flex flex-wrap justify-between gap-x-4 gap-y-0.5 text-[10px] uppercase tracking-[0.16em] text-faint">
        <span>Fig. 3 · hook events, guidance, what was sent · a bright line is in use</span>
        <span className="normal-case tracking-normal">an event opens the reminder form at that event</span>
      </div>

      <Part
        title="Loops"
        e2e="guidance-loops"
        right={
          <span className="mono text-[10.5px] text-faint">
            {(g.loops ?? []).filter((l) => l.state === "active").length} active · {(g.loops ?? []).length} set
          </span>
        }
        add={{ label: "add loop", open: open.loop, onToggle: () => toggle("loop"), form: <LoopAdd send={send} busy={busy} goals={openGoals} cap={g.limits?.loop_cap ?? LOOP_CAP} stallMax={g.limits?.stall_max ?? STALL_MAX} /> }}
      >
        {(g.loops ?? []).length === 0 ? (
          <div className={EMPTY}>No loop set. A loop continues the agent at a stop until its condition holds, its maximum is reached, it stalls or its budget is spent.</div>
        ) : (
          <ul className="grid">
            {(g.loops ?? []).map((l) => (
              <LoopRow key={l.id} l={l} send={send} busy={busy} onRemove={() => remove("loop", l.id)} />
            ))}
          </ul>
        )}
      </Part>

      <Part title="Ended loops" e2e="guidance-outcomes" right={<span className="mono text-[10.5px] text-faint">{g.outcomes_total ?? 0} ended · newest {(g.outcomes ?? []).length}</span>}>
        <Outcomes list={g.outcomes ?? []} now={now} />
      </Part>

      <Part
        title="Step graph"
        e2e="guidance-steps"
        right={
          <span className="mono text-[10.5px] text-faint">
            {g.counts?.done ?? 0} done · {g.counts?.ready ?? 0} ready · {g.counts?.blocked ?? 0} blocked
          </span>
        }
        add={{ label: "add step", open: open.step, onToggle: () => toggle("step"), form: <StepAdd send={send} busy={busy} steps={g.steps ?? []} goals={openGoals} /> }}
      >
        <StepGraph steps={g.steps ?? []} next={g.next_step ?? null} send={send} busy={busy} onRemove={(id) => remove("step", id)} />
      </Part>

      <Part
        title="Reminders"
        e2e="guidance-reminders"
        right={
          <span className="mono text-[10.5px] text-faint">
            {(g.reminders ?? []).filter((r) => r.enabled).length} on · {(g.reminders ?? []).length} set
          </span>
        }
        add={{ label: "add reminder", open: open.reminder, onToggle: () => toggle("reminder"), form: <ReminderAdd send={send} busy={busy} trigger={trigger} setTrigger={setTrigger} /> }}
      >
        {(g.reminders ?? []).length === 0 ? (
          <div className={EMPTY}>No reminder set. A reminder is a note delivered to the agent at a hook event, the repeated-error event included.</div>
        ) : (
          <ul className="grid">
            {(g.reminders ?? []).map((r) => (
              <ReminderRow key={r.id} r={r} send={send} busy={busy} onRemove={() => remove("reminder", r.id)} />
            ))}
          </ul>
        )}
      </Part>

      <Part
        title="Sent to the agent"
        e2e="guidance-deliveries"
        right={
          <span className="mono text-[10.5px] text-faint" data-e2e="guidance-verdicts" title="Verdicts of the evidence ledger, all judged deliveries">
            {(() => {
              const t = verdictTally(g.verdicts);
              return `${t.held} held · ${t.missed} missed · ${t.open} open · ${t.unjudged} not judged · newest ${(g.deliveries ?? []).length}`;
            })()}
          </span>
        }
      >
        <Deliveries list={g.deliveries ?? []} now={now} />
      </Part>
    </div>
  );
}

function ReminderAdd({ send, busy, trigger, setTrigger }: { send: Send; busy: string | null; trigger: string; setTrigger: (t: string) => void }) {
  const [text, setText] = useState("");
  const [every, setEvery] = useState(1);
  const [glob, setGlob] = useState("");
  const [repeat, setRepeat] = useState<string>("always");
  const [max, setMax] = useState(1);
  const [files, setFiles] = useState("");
  const [error, setError] = useState("");
  const [count, setCount] = useState(ERROR_COUNT_MIN);
  const counted = trigger === "prompt" || trigger === "tools";
  const onError = trigger === "error-repeats";
  return (
    <form
      className="flex min-w-0 flex-wrap items-end gap-x-3 gap-y-2"
      data-e2e="guidance-reminder-add"
      onSubmit={(e) => {
        e.preventDefault();
        if (!text.trim() || (onError && !error.trim())) return;
        const body: Record<string, unknown> = { action: "add", kind: "reminder", text: text.trim(), trigger, repeat };
        if (counted) body.every = every;
        if (onError) {
          body.error = error.trim();
          body.count = count;
        }
        if (trigger === "edit" && glob.trim()) body.glob = glob.trim();
        if (repeat === "max") body.max = max;
        if (files.trim()) body.files = files.trim();
        void send(body, "reminder-add").then((ok) => ok && setText(""));
      }}
    >
      <Field label="reminder" grow>
        <input value={text} onChange={(e) => setText(e.target.value)} maxLength={2000} placeholder="run the full tests before you finish" className={INPUT} data-e2e="guidance-reminder-text" />
      </Field>
      <Field label="when">
        <select value={trigger} onChange={(e) => setTrigger(e.target.value)} className={SELECT} data-e2e="guidance-reminder-trigger">
          {TRIGGERS.map((t) => (
            <option key={t} value={t}>
              {triggerWords({ trigger: t, every: 1, glob: null })}
            </option>
          ))}
        </select>
      </Field>
      {counted && (
        <Field label="every nth">
          <input type="number" min={1} max={1000} value={every} onChange={(e) => setEvery(Math.max(1, Number(e.target.value) || 1))} className={NUM} />
        </Field>
      )}
      {trigger === "edit" && (
        <Field label="paths">
          <input value={glob} onChange={(e) => setGlob(e.target.value)} maxLength={200} placeholder="src/**" className={`${INPUT} w-32`} />
        </Field>
      )}
      {onError && (
        <>
          <Field label="error code or message">
            <input value={error} onChange={(e) => setError(e.target.value)} maxLength={80} placeholder="E0308" className={`${INPUT} w-32`} data-e2e="guidance-reminder-error" />
          </Field>
          <Field label={`repeats (${ERROR_COUNT_MIN} to ${ERROR_COUNT_MAX})`}>
            <input type="number" min={ERROR_COUNT_MIN} max={ERROR_COUNT_MAX} value={count} onChange={(e) => setCount(Math.min(ERROR_COUNT_MAX, Math.max(ERROR_COUNT_MIN, Number(e.target.value) || ERROR_COUNT_MIN)))} className={NUM} />
          </Field>
        </>
      )}
      <Field label="how often">
        <select value={repeat} onChange={(e) => setRepeat(e.target.value)} className={SELECT}>
          {REPEATS.map((r) => (
            <option key={r} value={r}>
              {r === "max" ? "at most n a session" : repeatWords({ repeat: r, max: 1 })}
            </option>
          ))}
        </select>
      </Field>
      {repeat === "max" && (
        <Field label="n">
          <input type="number" min={1} max={1000} value={max} onChange={(e) => setMax(Math.max(1, Number(e.target.value) || 1))} className={NUM} />
        </Field>
      )}
      <Field label="files (optional)">
        <input value={files} onChange={(e) => setFiles(e.target.value)} maxLength={200} placeholder="docs/**" className={`${INPUT} w-28`} />
      </Field>
      <button type="submit" disabled={busy !== null || !text.trim() || (onError && !error.trim())} className={ADD} data-e2e="guidance-reminder-submit">
        add reminder
      </button>
    </form>
  );
}

function ReminderRow({ r, send, busy, onRemove }: { r: Reminder; send: Send; busy: string | null; onRemove: () => void }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(r.text);
  return (
    <li className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-x-3 border-b border-line/60 py-2" data-e2e="guidance-reminder" data-id={r.id} data-enabled={String(r.enabled)}>
      <Switch on={r.enabled} label={`Reminder ${r.id} ${r.enabled ? "on" : "off"}`} busy={busy === `rs${r.id}`} onChange={(on) => void send({ action: on ? "enable" : "disable", kind: "reminder", id: r.id }, `rs${r.id}`)} />
      <span className="min-w-0">
        {editing ? (
          <form
            className="flex items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (text.trim()) void send({ action: "edit", kind: "reminder", id: r.id, text: text.trim() }, `re${r.id}`).then((ok) => ok && setEditing(false));
            }}
          >
            <input value={text} onChange={(e) => setText(e.target.value)} maxLength={2000} className={INPUT} aria-label={`Reminder ${r.id} text`} />
            <button type="submit" disabled={busy !== null || !text.trim()} className={ACT}>
              save
            </button>
          </form>
        ) : (
          <span className={`block break-words text-xs ${r.enabled ? "text-muted" : "text-faint"}`}>{r.text}</span>
        )}
        <span className="mono text-[10px] text-quiet">
          {r.id} · {triggerWords(r)} · {repeatWords(r)}
          {r.files ? ` · files ${r.files}` : ""} · {whoWords(r.by)} · delivered {r.delivered ?? 0}
        </span>
      </span>
      <span className="flex items-center gap-3">
        <button type="button" disabled={busy !== null} onClick={() => {
            if (!editing) setText(r.text);
            setEditing(!editing);
          }} className={ACT}>
          {editing ? "cancel" : "edit"}
        </button>
        <button type="button" disabled={busy !== null} onClick={() => void send({ action: "reset", kind: "reminder", id: r.id }, `rr${r.id}`)} className={ACT} title="Count this session's deliveries again from now">
          reset
        </button>
        <button type="button" disabled={busy !== null} onClick={onRemove} className={ACT}>
          remove
        </button>
      </span>
    </li>
  );
}

function LoopAdd({ send, busy, goals, cap, stallMax }: { send: Send; busy: string | null; goals: { id: number | string; objective: string }[]; cap: number; stallMax: number }) {
  const [prompt, setPrompt] = useState("");
  const [until, setUntil] = useState<string>("check-passes");
  const [parts, setParts] = useState<string[]>(["check-passes", "steps-done"]);
  const [goal, setGoal] = useState("");
  const [max, setMax] = useState(LOOP_DEFAULT);
  const [stall, setStall] = useState(STALL_DEFAULT);
  const [budgetS, setBudgetS] = useState("");
  const [budgetTools, setBudgetTools] = useState("");
  const combined = until === "all-of" || until === "any-of";
  const uses = (c: string) => until === c || (combined && parts.includes(c));
  const needsGoal = uses("goal-done");
  const takesGoal = needsGoal || uses("steps-done");
  const partsOk = !combined || (parts.length >= 2 && parts.length <= 4);
  return (
    <form
      className="flex min-w-0 flex-wrap items-end gap-x-3 gap-y-2"
      data-e2e="guidance-loop-add"
      onSubmit={(e) => {
        e.preventDefault();
        if (!prompt.trim() || (needsGoal && !goal) || !partsOk) return;
        const body: Record<string, unknown> = { action: "add", kind: "loop", prompt: prompt.trim(), until, max, stall };
        if (combined) body.parts = parts;
        if (takesGoal && goal) body.goal = Number(goal);
        if (Number(budgetS) > 0) body.budget_ms = Math.round(Number(budgetS) * 1000);
        if (Number(budgetTools) > 0) body.budget_tools = Math.round(Number(budgetTools));
        void send(body, "loop-add").then((ok) => ok && setPrompt(""));
      }}
    >
      <Field label="loop prompt" grow>
        <input value={prompt} onChange={(e) => setPrompt(e.target.value)} maxLength={2000} placeholder="fix what fails and run the tests again" className={INPUT} data-e2e="guidance-loop-prompt" />
      </Field>
      <Field label="until">
        <select value={until} onChange={(e) => setUntil(e.target.value)} className={SELECT} data-e2e="guidance-loop-until">
          {CONDITIONS.map((c) => (
            <option key={c} value={c}>
              {c === "goal-done" ? "a goal is closed" : c === "steps-done" ? "every step is done" : c === "check-passes" ? "build or test passes after the last edit" : c === "all-of" ? "all of the parts" : c === "any-of" ? "any of the parts" : "ended by hand"}
            </option>
          ))}
        </select>
      </Field>
      {combined && (
        <fieldset className="grid min-w-0 gap-0.5" data-e2e="guidance-loop-parts">
          <legend className={LABEL}>parts (2 to 4)</legend>
          <span className="flex h-9 flex-wrap items-center gap-x-3">
            {PARTS.map((p) => (
              <label key={p} className="mono flex items-center gap-1 text-[11px] text-muted">
                <input type="checkbox" checked={parts.includes(p)} onChange={(e) => setParts(e.target.checked ? [...parts, p] : parts.filter((x) => x !== p))} />
                {partWords(p)}
              </label>
            ))}
          </span>
        </fieldset>
      )}
      {takesGoal && (
        <Field label={needsGoal ? "goal" : "goal (optional)"}>
          <select value={goal} onChange={(e) => setGoal(e.target.value)} className={`${SELECT} max-w-[14rem]`}>
            <option value="">{needsGoal ? "pick a goal" : "all steps"}</option>
            {goals.map((o) => (
              <option key={String(o.id)} value={String(o.id)}>
                {fit(o.objective, 32)}
              </option>
            ))}
          </select>
        </Field>
      )}
      <Field label={`max (1 to ${cap})`}>
        <input type="number" min={1} max={cap} value={max} onChange={(e) => setMax(Math.min(cap, Math.max(1, Number(e.target.value) || 1)))} className={NUM} />
      </Field>
      <Field label={`stall (1 to ${stallMax})`}>
        <input type="number" min={1} max={stallMax} value={stall} onChange={(e) => setStall(Math.min(stallMax, Math.max(1, Number(e.target.value) || 1)))} className={NUM} data-e2e="guidance-loop-stall" title="The same unmet reason this many times in a row ends the loop as stalled" />
      </Field>
      <Field label="budget s (optional)">
        <input type="number" min={1} max={86400} value={budgetS} onChange={(e) => setBudgetS(e.target.value)} placeholder="none" className={NUM} data-e2e="guidance-loop-budget-ms" />
      </Field>
      <Field label="budget tools (optional)">
        <input type="number" min={1} max={10000} value={budgetTools} onChange={(e) => setBudgetTools(e.target.value)} placeholder="none" className={NUM} data-e2e="guidance-loop-budget-tools" />
      </Field>
      <button type="submit" disabled={busy !== null || !prompt.trim() || (needsGoal && !goal) || !partsOk} className={ADD} data-e2e="guidance-loop-submit">
        add loop
      </button>
    </form>
  );
}

function LoopRow({ l, send, busy, onRemove }: { l: Loop; send: Send; busy: string | null; onRemove: () => void }) {
  const active = l.state === "active";
  const tone = loopTone(l.state);
  const budget = budgetWords(l);
  return (
    <li className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-x-3 border-b border-line/60 py-2" data-e2e="guidance-loop" data-id={l.id} data-state={l.state} data-iterations={l.iterations}>
      <Switch on={l.enabled} label={`Loop ${l.id} ${l.enabled ? "on" : "off"}`} busy={busy === `ls${l.id}`} onChange={(on) => void send({ action: on ? "enable" : "disable", kind: "loop", id: l.id }, `ls${l.id}`)} />
      <span className="grid min-w-0 gap-1">
        <span className={`block break-words text-xs ${active && l.enabled ? "text-muted" : "text-faint"}`}>{l.prompt}</span>
        <span className="flex min-w-0 items-center gap-2">
          <Pill tone={tone}>{l.state}</Pill>
          <span className="flex shrink-0 items-center gap-[2px]" role="img" aria-label={`${l.iterations} of ${l.max} iterations`} title={`${l.iterations} of ${l.max} iterations: one mark per allowed continuation`}>
            {Array.from({ length: Math.max(1, Math.min(LOOP_CAP, l.max)) }, (_, i) => (
              <i key={i} aria-hidden className={`block h-2.5 w-[5px] ${i < l.iterations ? (active ? "bg-sense-pale" : tone === "verified" ? "bg-verified" : "bg-attention") : "border border-line-3"}`} />
            ))}
          </span>
          <span className="mono text-[10.5px] text-muted">
            {l.iterations} of {l.max}
          </span>
        </span>
        <span className="mono text-[10px] text-quiet">
          {l.id} · {untilWords(l)} · {whoWords(l.by)}
          {l.until !== "manual" ? ` · stalls after ${l.stall ?? STALL_DEFAULT} same reasons in a row` : ""}
          {budget ? ` · ${budget}` : ""}
          {l.blocks ? ` · continued ${l.blocks}` : ""}
          {l.not_acted_antigravity ? ` · ${l.not_acted_antigravity} stops not acted on Antigravity` : ""}
        </span>
        {active && l.now && (
          <span className={`mono text-[10.5px] ${l.now.met ? "text-verified" : "text-faint"}`} title={l.now.session ? `newest session ${l.now.session}` : undefined}>
            now: {l.now.met ? "met" : "not met"}, {l.now.detail}
          </span>
        )}
        {!active && l.end_note && <span className="mono text-[10.5px] text-faint">{l.end_note}</span>}
      </span>
      <span className="flex items-center gap-3">
        {active ? (
          <button type="button" disabled={busy !== null} onClick={() => void send({ action: "done", kind: "loop", id: l.id }, `le${l.id}`)} className={ACT} data-e2e="guidance-loop-end">
            end
          </button>
        ) : (
          <button type="button" disabled={busy !== null} onClick={() => void send({ action: "reset", kind: "loop", id: l.id }, `lr${l.id}`)} className={ACT}>
            restart
          </button>
        )}
        <button type="button" disabled={busy !== null} onClick={onRemove} className={ACT}>
          remove
        </button>
      </span>
    </li>
  );
}

function StepAdd({ send, busy, steps, goals }: { send: Send; busy: string | null; steps: Step[]; goals: { id: number | string; objective: string }[] }) {
  const [title, setTitle] = useState("");
  const [after, setAfter] = useState("");
  const [afterAny, setAfterAny] = useState("");
  const [probe, setProbe] = useState(false);
  const [check, setCheck] = useState("none");
  const [paths, setPaths] = useState("");
  const [goal, setGoal] = useState("");
  return (
    <form
      className="flex min-w-0 flex-wrap items-end gap-x-3 gap-y-2"
      data-e2e="guidance-step-add"
      onSubmit={(e) => {
        e.preventDefault();
        if (!title.trim()) return;
        const body: Record<string, unknown> = { action: "add", kind: "step", title: title.trim(), depends_on: listOf(after), depends_any: listOf(afterAny), check, paths: listOf(paths), probe: check !== "none" && probe };
        if (goal) body.goal = Number(goal);
        void send(body, "step-add").then((ok) => {
          if (ok) {
            setTitle("");
            setAfter("");
            setAfterAny("");
          }
        });
      }}
    >
      <Field label="step" grow>
        <input value={title} onChange={(e) => setTitle(e.target.value)} maxLength={2000} placeholder="write the parser" className={INPUT} data-e2e="guidance-step-title" />
      </Field>
      <Field label="after">
        <input value={after} onChange={(e) => setAfter(e.target.value)} placeholder={steps.length ? steps.slice(-2).map((s) => s.id).join(", ") : "step ids"} className={`${INPUT} w-28`} list="guidance-step-ids" />
      </Field>
      <Field label="after one of">
        <input value={afterAny} onChange={(e) => setAfterAny(e.target.value)} placeholder="step ids" className={`${INPUT} w-28`} list="guidance-step-ids" data-e2e="guidance-step-after-any" />
      </Field>
      <datalist id="guidance-step-ids">
        {steps.map((s) => (
          <option key={s.id} value={s.id} />
        ))}
      </datalist>
      <Field label="done when">
        <select value={check} onChange={(e) => setCheck(e.target.value)} className={SELECT}>
          <option value="none">marked by hand</option>
          <option value="build">a build passes after edits</option>
          <option value="test">a test passes after edits</option>
        </select>
      </Field>
      {check !== "none" && (
        <Field label="under paths">
          <input value={paths} onChange={(e) => setPaths(e.target.value)} placeholder="src/**" className={`${INPUT} w-28`} />
        </Field>
      )}
      {check !== "none" && (
        <label className="mono flex h-9 items-center gap-1 self-end text-[11px] text-muted" title="A probe is done when its check passes while its dependencies are met, with no edit needed">
          <input type="checkbox" checked={probe} onChange={(e) => setProbe(e.target.checked)} data-e2e="guidance-step-probe" />
          probe (no edit needed)
        </label>
      )}
      {goals.length > 0 && (
        <Field label="goal">
          <select value={goal} onChange={(e) => setGoal(e.target.value)} className={`${SELECT} max-w-[12rem]`}>
            <option value="">none</option>
            {goals.map((o) => (
              <option key={String(o.id)} value={String(o.id)}>
                {fit(o.objective, 28)}
              </option>
            ))}
          </select>
        </Field>
      )}
      <button type="submit" disabled={busy !== null || !title.trim()} className={ADD} data-e2e="guidance-step-submit">
        add step
      </button>
    </form>
  );
}

function StepGraph({ steps, next, send, busy, onRemove }: { steps: Step[]; next: string | null; send: Send; busy: string | null; onRemove: (id: string) => void }) {
  const [picked, setPicked] = useState<string | null>(null);
  const g = useMemo(() => layout(steps), [steps]);
  const sel = steps.find((s) => s.id === picked) ?? null;
  return (
    <div className="grid min-w-0 gap-2" data-e2e="guidance-graph">
      {steps.length === 0 ? (
        <div className="night ticks grid min-h-28 place-items-center border border-line px-4 py-6">
          <span className="mono text-center text-[11px] leading-relaxed text-faint">
            No steps yet.
            <br />A step waits for the steps before it; the next ready one is given to the agent at prompts and stops.
          </span>
        </div>
      ) : (
        <div className="night ticks min-w-0 overflow-x-auto border border-line p-3">
          <svg role="img" aria-label={`Step graph: ${steps.map((s) => `${s.id} ${s.status}`).join(", ")}`} width={g.width} height={g.height} viewBox={`0 0 ${g.width} ${g.height}`} className="block">
            <defs>
              <marker id="guidance-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0 0 L8 4 L0 8 z" fill="var(--text-4)" />
              </marker>
              <marker id="guidance-arrow-any" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0 4 L4 0 L8 4 L4 8 z" fill="var(--sense-pale)" />
              </marker>
            </defs>
            {g.edges.map((e) => (
              <path
                key={`${e.from}>${e.to}`}
                d={edgePath(e)}
                fill="none"
                stroke={e.any ? "var(--sense-pale)" : "var(--text-4)"}
                strokeWidth={1}
                strokeDasharray={e.any ? "5 3" : undefined}
                markerEnd={e.any ? "url(#guidance-arrow-any)" : "url(#guidance-arrow)"}
                data-e2e="guidance-step-edge"
                data-join={e.any ? "any" : "all"}
              >
                <title>{e.any ? `${e.to} after one of its OR-join, here ${e.from}` : `${e.to} after ${e.from}`}</title>
              </path>
            ))}
            {g.nodes.map((n) => {
              const stroke = STATUS_STROKE[n.status] ?? "var(--text-4)";
              const on = n.id === picked;
              return (
                <g
                  key={n.id}
                  role="button"
                  tabIndex={0}
                  aria-label={`Step ${n.id}, ${n.status}: ${n.title}`}
                  data-e2e="guidance-step-node"
                  data-id={n.id}
                  data-status={n.status}
                  data-probe={String(n.probe)}
                  transform={`translate(${n.x} ${n.y})`}
                  className="fig-node"
                  aria-pressed={on}
                  onClick={() => setPicked(on ? null : n.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setPicked(on ? null : n.id);
                    }
                  }}
                >
                  <title>{`${n.id}: ${n.title} (${n.status}${n.probe ? ", probe" : ""})`}</title>
                  <rect className="fig-box" width={NODE_W} height={NODE_H} rx={n.probe ? NODE_H / 2 : 2} fill={on ? "var(--bg-3)" : "var(--bg-1)"} stroke={stroke} strokeWidth={n.id === next ? 2 : 1} strokeDasharray={n.status === "blocked" ? "3 2" : undefined} />
                  <text x={n.probe ? 12 : 8} y={13} fontSize={9} fontFamily="var(--font-mono, monospace)" fill={stroke}>
                    {n.id} · {n.status}
                    {n.probe ? " · probe" : ""}
                    {n.id === next ? " · next" : ""}
                  </text>
                  <text x={n.probe ? 12 : 8} y={26} fontSize={11} fill={n.status === "done" ? "var(--text-3)" : "var(--text-1)"}>
                    {fit(n.title, n.probe ? 19 : 22)}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      )}
      <ul className="mono flex flex-wrap items-center gap-x-5 gap-y-1 text-[10px] text-faint" aria-label="How the graph is drawn" data-e2e="guidance-graph-legend">
        <li className="flex items-center gap-1.5">
          <i aria-hidden className="inline-block h-2.5 w-4 border border-verified" />
          done
        </li>
        <li className="flex items-center gap-1.5">
          <i aria-hidden className="inline-block h-2.5 w-4 border border-sense-pale" />
          ready
        </li>
        <li className="flex items-center gap-1.5">
          <i aria-hidden className="inline-block h-2.5 w-4 border border-dashed border-quiet" />
          blocked
        </li>
        <li className="flex items-center gap-1.5" title="A probe is done when its check passes while its dependencies are met, with no edit needed">
          <i aria-hidden className="inline-block h-2.5 w-4 rounded-full border border-quiet" />
          probe
        </li>
        <li className="flex items-center gap-1.5" title="Plain edge: the step waits for every step it points from (AND-join)">
          <svg aria-hidden width="22" height="8" viewBox="0 0 22 8">
            <path d="M0 4h16" stroke="var(--text-4)" />
            <path d="M15 1l6 3-6 3z" fill="var(--text-4)" />
          </svg>
          after all
        </li>
        <li className="flex items-center gap-1.5" title="Dashed edge with a diamond head: the step waits for at least one of these (OR-join)">
          <svg aria-hidden width="22" height="8" viewBox="0 0 22 8">
            <path d="M0 4h14" stroke="var(--sense-pale)" strokeDasharray="4 2" />
            <path d="M14 4l3.5-3 3.5 3-3.5 3z" fill="var(--sense-pale)" />
          </svg>
          after one of
        </li>
        {steps.length > 0 && <li className="ml-auto">pick a step for its detail</li>}
      </ul>
      {sel && (
        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3 border-b border-line/60 py-2" data-e2e="guidance-step" data-id={sel.id} data-status={sel.status}>
          <span className="min-w-0">
            <span className={`block break-words text-xs ${sel.done ? "text-faint line-through" : "text-muted"}`}>{sel.title}</span>
            <span className="mono text-[10px] text-quiet">
              {sel.id} · {sel.status}
              {sel.depends_on.length ? ` · after ${sel.depends_on.join(", ")}` : ""}
              {(sel.depends_any ?? []).length ? ` · after one of ${(sel.depends_any ?? []).join(", ")}` : ""}
              {sel.check && sel.probe ? ` · probe: done when a ${sel.check} passes with its dependencies met, no edit needed` : ""}
              {sel.check && !sel.probe ? ` · done when a ${sel.check} passes${sel.paths.length ? ` after edits under ${sel.paths.join(", ")}` : " after edits"}` : ""}
              {sel.goal !== null ? ` · goal ${sel.goal}` : ""} · {whoWords(sel.by)}
              {sel.done && sel.done_by ? ` · done ${whoWords(sel.done_by).replace(/^set /, "")}` : ""}
            </span>
          </span>
          <span className="flex items-center gap-3">
            {sel.done ? (
              <button type="button" disabled={busy !== null} onClick={() => void send({ action: "reset", kind: "step", id: sel.id }, `sr${sel.id}`)} className={ACT}>
                open again
              </button>
            ) : (
              <button type="button" disabled={busy !== null} onClick={() => void send({ action: "done", kind: "step", id: sel.id }, `sd${sel.id}`)} className={ACT}>
                mark done
              </button>
            )}
            <button type="button" disabled={busy !== null} onClick={() => onRemove(sel.id)} className={ACT}>
              remove
            </button>
          </span>
        </div>
      )}
      {steps.some((s) => s.status === "ready") && (
        <ul className="flex flex-wrap gap-x-4 gap-y-1">
          {steps
            .filter((s) => s.status === "ready")
            .map((s) => (
              <li key={s.id} className="flex items-center gap-2">
                <span className="mono text-[10.5px] text-muted">{s.id}</span>
                <button type="button" disabled={busy !== null} onClick={() => void send({ action: "done", kind: "step", id: s.id }, `sd${s.id}`)} className={ACT} data-e2e="guidance-step-done" data-id={s.id}>
                  mark done
                </button>
              </li>
            ))}
        </ul>
      )}
    </div>
  );
}

function Outcomes({ list, now }: { list: Outcome[]; now: number }) {
  if (list.length === 0) return <div className={EMPTY}>No loop has ended yet. Each end is kept here with its condition, iterations, budget used and verdict.</div>;
  return (
    <ul className="grid">
      {list.map((o, i) => {
        const b = o.budget ?? {};
        const used = `${msWords(b.used_ms ?? 0)}, ${b.used_tools ?? 0} tool ${(b.used_tools ?? 0) === 1 ? "call" : "calls"}`;
        return (
          <li key={`${o.id}-${o.ended_ms ?? i}`} className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-3 border-b border-line/50 py-1.5" data-e2e="guidance-outcome" data-id={o.id} data-end={o.end}>
            <span className="mono text-[10px] text-faint" title={o.ended_ms ? new Date(o.ended_ms).toISOString() : undefined}>
              {now && o.ended_ms ? ago(o.ended_ms, now) : ""}
            </span>
            <span className="grid min-w-0 gap-0.5">
              <span className="flex min-w-0 flex-wrap items-center gap-2">
                <Pill tone={endTone(o.end)}>{endWords(o.end)}</Pill>
                <span className="mono text-[10.5px] text-muted">
                  {o.id} · {o.iterations} of {o.max ?? "?"} · used {used}
                </span>
              </span>
              <span className="mono text-[10px] text-quiet">
                {untilWords({ until: o.condition.until, goal: o.condition.goal ?? null, parts: o.condition.parts })}
                {o.verdicts?.state ? ` · ${o.verdicts.rule ?? "loop-met"} ${o.verdicts.state}` : ""}
                {o.by ? ` · ${whoWords(o.by)}` : ""}
              </span>
              {(o.unmet ?? []).length > 0 && (
                <span className="mono break-words text-[10px] text-faint" title={(o.unmet ?? []).join("\n")}>
                  unmet, {(o.unmet ?? []).length} distinct: {fit((o.unmet ?? [])[(o.unmet ?? []).length - 1] ?? "", 120)}
                </span>
              )}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function Deliveries({ list, now }: { list: Delivery[]; now: number }) {
  if (list.length === 0) return <div className={EMPTY}>Nothing sent to the agent yet. Each delivery is listed here with the verdict of the evidence ledger.</div>;
  return (
    <ul className="grid">
      {list.map((d, i) => (
        <li key={`${d.t}-${d.kind}-${d.id}-${i}`} className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-3 border-b border-line/50 py-1.5" data-e2e="guidance-delivery" data-kind={d.kind} data-channel={d.channel}>
          <span className="mono text-[10px] text-faint" title={Number.isFinite(d.t) ? new Date(d.t).toISOString() : undefined}>
            {now ? ago(d.t, now) : ""}
          </span>
          <span className="min-w-0">
            <span className="block break-words text-[11.5px] text-muted">{d.text}</span>
            <span className="mono text-[10px] text-quiet">
              {d.kind} {d.id} · {d.event} · {channelWords(d)}
              {d.session ? ` · session ${d.session.slice(0, 8)}` : ""}
            </span>
            {d.verdict && (
              <span className="mt-0.5 flex flex-wrap items-center gap-2" data-e2e="guidance-delivery-verdict" data-state={d.verdict.verdict.state}>
                <Pill tone={verdictTone(d.verdict)} title={`${d.verdict.verdict.rule}, window ${d.verdict.window}${d.verdict.closed ? ` (${d.verdict.closed})` : ""}`}>
                  {d.verdict.verdict.rule}
                </Pill>
                <span className="mono text-[10px] text-faint">{verdictWords(d.verdict)}</span>
              </span>
            )}
          </span>
        </li>
      ))}
    </ul>
  );
}

/** The rail's Guidance tab: what each part of the Guidance section means. */
export function GuidanceAbout() {
  const rows = (items: [string, string][]) => (
    <ul className="grid gap-2">
      {items.map(([k, v]) => (
        <li key={k} className="grid grid-cols-[6.5rem_minmax(0,1fr)] gap-3 text-[11.5px]">
          <span className="text-text">{k}</span>
          <span className="leading-relaxed text-faint">{v}</span>
        </li>
      ))}
    </ul>
  );
  return (
    <div className="grid" data-e2e="guidance-about">
      <RailSection title="The circuit · Fig. 3">
        {rows([
          ["hook events", "the eight events of your coding agent a reminder can be set at; one with an enabled reminder is bright and carries its count; picking one opens the reminder form at that event"],
          ["lines", "every event can reach the reminders; only the stop reaches the loops; prompts and stops carry the next ready step. A line is bright when the project's guidance uses it now, dotted when it does not"],
          ["guidance", "reminders on and set; loops active and ended, with the first active loop's iterations as marks; steps done of all, with the next ready step"],
          ["sent", "every delivery of this project, and the evidence ledger's verdicts on them as one bar"],
          ["setting off", "a kind switched off in Settings is drawn dashed; with memory off nothing is delivered"],
        ])}
      </RailSection>
      <RailSection title="Reminders">
        <p className="mb-2 text-[11.5px] leading-relaxed text-faint">
          A note for your coding agent, added to its context through the hooks at its trigger, whole, prefixed with who set it. It may ride with a prime in the same note; it never counts against the prime&apos;s never-twice rule.
        </p>
        {rows([
          ["when", "a session start, every prompt or every nth one, after a compaction, at a stop, before an edit of matching paths, after a failed build or test, after every nth tool call, or when the same recorded error comes back"],
          ["error repeats", "the error code (E0308) or a piece of its message, as the hook records it; fires when 2 to 5 of the session's tool results carry it, then counts again"],
          ["how often", "every time, once a session, or at most n times a session; reset counts again from now"],
          ["files", "optional: only when the event's files, or a file the session touched, match the glob"],
        ])}
      </RailSection>
      <RailSection title="Loops">
        <p className="mb-2 text-[11.5px] leading-relaxed text-faint">
          At a stop, while the condition is unmet and the loop is under its maximum, the Stop hook answers Claude Code with decision block and the loop&apos;s prompt, which makes the agent continue. At the maximum, or once the condition holds, the loop is marked exhausted or done and a final note says so.
        </p>
        {rows([
          ["check passes", "the last build or test after the last edit, in that session, passed"],
          ["goal closed", "the goal is done or dropped"],
          ["steps done", "every step (of the goal, when one is named) is done"],
          ["by hand", "until you or the agent end it"],
          ["all of, any of", "2 to 4 of the conditions above combined; the unmet reason names the parts not met"],
          ["maximum", "3 by default, never more than 20"],
          ["stall", "the same unmet reason this many times in a row (2 by default, 1 to 10) ends the loop as stalled, said once to the agent; a loop ended by hand only has no progress reason and does not stall"],
          ["budget", "optional: wall time or tool calls since the loop was set; when used, the loop ends as budget spent"],
          ["ended loops", "every end (met, maximum, stalled, budget spent, ended by hand, removed) is one row with the condition, iterations, budget used, the distinct unmet reasons in order and the verdict; kept as evidence for Learning, which does not read them yet in v0.1.0"],
          ["the exception", "the only case in which the Nano's hooks block a stop, and only to continue the agent; it acts only when a loop is set and on, and the loops setting switches every loop off"],
        ])}
      </RailSection>
      <RailSection title="Step graph">
        {rows([
          ["ready", "every step it waits for is done, and one of its OR-join when it has one"],
          ["blocked", "a step it waits for is not done"],
          ["after one of", "an OR-join (depends_any), drawn as a dashed edge with a diamond head; plain edges are the AND-join (depends_on)"],
          ["probe", "a step with a check, drawn with round ends: done when that check passes while its dependencies are met, with no edit needed"],
          ["cycles", "refused when a step is added or changed, over both kinds of edge"],
          ["nudges", "with reminders on, the next ready step is given at prompts and at stops, once per step per session unless it changes"],
          ["done by the Nano", "a step with a check is marked done when that build or test passes after an edit under its paths; the evidence is kept on the step"],
        ])}
      </RailSection>
      <RailSection title="Verdicts">
        <p className="mb-2 text-[11.5px] leading-relaxed text-faint">
          The evidence ledger judges each delivery over the rest of its session and writes the verdict once the session ends, is cleared, or goes an hour without activity. Until then it is shown as open.
        </p>
        {rows([
          ["step nudge", "followed when the step is marked done, or its check passes, later in the session; passed over otherwise"],
          ["loop", "each continuation is verified when the condition was met before the cap, refuted when the cap, a stall or a budget came first"],
          ["reminder", "a reminder tied to an edit glob is verified when the last build or test after the last edit under that glob passed, refuted when it failed; other reminders are not judged"],
        ])}
      </RailSection>
      <RailSection title="Hosts">
        {rows([
          ["Claude Code", "every trigger; loops continue the agent at a stop"],
          ["Antigravity", "reminders at session start and prompts only (the events that take injected steps); loops do not act there, and each stop they would have acted on is recorded"],
        ])}
      </RailSection>
      <RailSection title="Your agent sets its own">
        <p className="text-[11.5px] leading-relaxed text-faint">
          Over MCP, nm_goal with op reminder_add, loop_add, step_add, step_done, loop_end, guidance and the others. What the agent sets is shown as set by the agent. The command line has the same: neuromorph-nano guidance.
        </p>
      </RailSection>
      <RailSection title="Undo and removal">
        <p className="text-[11.5px] leading-relaxed text-faint">
          Every change, the Nano&apos;s own included, is a row on the Audit page that undoes it. Removing shows what goes and asks for the id to be typed first.
        </p>
      </RailSection>
    </div>
  );
}
