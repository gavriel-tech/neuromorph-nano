"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals ... you have to think of it as if it's an
// interactive system for the main parts the right menu are libraries and or discription tools etc.
// left menu are the main interface nav like it is now. ... We need the interface to be amazing."):
// Audit is now the log as a timeline. The middle: the tape of every event, the filters (kind,
// project, ok or failed, log or episodes) and the timeline by day, each event a mark on the spine
// with its time, kind, label and duration. Clicking an event puts it in the rail: what changed,
// Undo, the parts it ran and the full record. The rail also holds the stage times and retention.
// 2026-11-27 (his words: "The main center screen for every page should be easy to navigate and
// beautifully visual and interactive"): a mark on the tape opens its event and the open event
// stands out on it; the kinds are chips that wrap instead of one bar whose names broke in two;
// "undoable" is a filter of its own, and an undoable row carries its Undo beside it, so undo is
// one press from the timeline as well as from the event's detail.

import { useEffect, useRef, useState } from "react";
import { ControlRow, NoteLine } from "@/components/Controls";
import DeleteFlow, { type DeleteNote } from "@/components/DeleteFlow";
import { Tape } from "@/components/PagePlates";
import PrimesStudio, { PrimesRail } from "@/components/PrimesStudio";
import ProjectMenu from "@/components/ProjectMenu";
import SectionBar from "@/components/SectionBar";
import StageTimesPanel from "@/components/StageTimes";
import Workspace, { RailSection, useRail } from "@/components/Workspace";
import { CallText, Dot, Empty, Failed, Masked, Pill, Placeholder, Segmented, type Tone } from "@/components/ui";
import { msText } from "@/lib/activity";
import { ago, bytes, post, refreshAll, useApi, type Audit, type AuditRow, type Control, type Ledger } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { primeCounts, useProjectKey } from "@/lib/evidence";
import { setHash, useHash } from "@/lib/hash";
import { callLabel, captureMask, hide, maskName, maskPathsIn, reveal, useRevealed } from "@/lib/privacy";

type Note = { text: string; tone: "verified" | "error" };

// 2026-11-26: every kind the engine writes falls in one of these groups; the engine filters by a
// comma list of kinds (`/api/audit?kind=a,b`). Each group's words say what its rows are.
const KIND_GROUPS: Record<string, string[]> = {
  all: [],
  tool: ["tool"],
  priming: ["prime", "prime-config"],
  setting: ["setting", "nanoignore", "project"],
  outcome: ["outcome"],
  learning: ["learning-cycle", "training", "engine-rollback", "engine-sync", "distil", "unit"],
  "memory & guidance": ["memory", "goal", "guidance"],
  cartridges: ["library", "library-base", "library-base-remove", "library-merge", "library-rollback", "library-seat", "library-stacks", "library-sockets", "library-rack", "role-model"],
  "maps & filters": ["job-filter", "job-filter-build", "parameter-map", "parameter-map-save", "parameter-map-remove", "cartridge-parameters", "parameter-overlay"],
  backups: ["backup-repo"],
  download: ["download", "cartridge-source"],
  "forget & delete": ["project-forget", "memory-forget", "retention", "storage-delete", "storage-prune", "dense-clear", "privacy"],
  setup: ["setup", "first-content", "hook-fault"],
};
const KINDS = Object.keys(KIND_GROUPS);
const STATES = ["all", "ok", "failed", "undoable"] as const;
type StateFilter = (typeof STATES)[number];

/** Colour marks state only: a failure is an error, a call still running is the signal. */
const toneOf = (e: AuditRow): Tone => (e.ok === false ? "error" : e.state === "running" ? "signal" : e.kind === "outcome" && e.ok === true ? "verified" : "muted");
const SPINE: Record<Tone, string> = { error: "var(--error)", signal: "var(--signal)", verified: "var(--verified)", muted: "var(--text-4)", attention: "var(--attention)", learned: "var(--learned)" };

function titleOf(e: AuditRow): string {
  if (e.kind === "setting") return String(e.label ?? e.key ?? "Setting changed");
  if (e.kind === "library" && typeof e.label !== "string") {
    if (e.action === "use") return `Stack for ${String(e.use ?? "a use")} switched`;
    if (e.action === "trial") return `Stack ${String(e.stack ?? "")} trial changed`;
  }
  if (e.kind === "job-filter" && typeof e.label !== "string") return `Job filter ${String(e.role ?? "")}/${String(e.job ?? "")} ${String(e.action ?? "")}`;
  if (typeof e.label !== "string" && typeof e.line === "string") return e.line;
  return String(e.label ?? e.tool ?? e.kind);
}

/** The rows the engine can undo through POST /api/undo (crates/nmcore controls `undo_by`): every
 *  setting change; a library row that switched a stack for a use or its trial mark; a job filter
 *  plugged in, out or removed (not the undo records of job filters themselves). */
// 2026-11-27: the engine's undo also takes the rows below (controls.rs `undo_by`, Phase B and
// after); the page listed only the first three kinds, so a guidance, memory pin, goal, parameter
// map, backup repository or teacher data change had no Undo here although its note said Audit
// undoes it. `null` is any action of that kind.
const UNDO_KINDS: Record<string, string[] | null> = {
  "parameter-map": ["save", "import", "lineage", "archive", "restore"],
  unit: ["unregister"],
  "cartridge-parameters": ["edit"],
  "parameter-overlay": null,
  memory: ["pin", "unpin"],
  goal: ["add", "status"],
  guidance: null,
  "cartridge-source": ["add", "remove"],
  "backup-repo": ["add", "remove", "restore"],
  distil: ["import", "label", "values"],
  // 2026-11-27: the base cartridge a MicroModel serves, and the model a role runs on (rows logged
  // from this date keep their `from`; earlier rows have no action and are not offered).
  "library-base": ["choose"],
  "role-model": ["choose"],
  nanoignore: ["set", "add", "remove"],
  project: ["create", "rename"],
  "prime-config": null,
};
function undoable(e: AuditRow): boolean {
  if (e.kind === "setting") return true;
  if (e.kind === "library") return e.action === "use" || e.action === "trial";
  if (e.kind === "job-filter") return typeof e.action === "string" && e.action !== "undo";
  if (e.kind in UNDO_KINDS) {
    // A row that is itself an undo is not offered again.
    if (typeof e.undo_of === "string" || e.action === "undo") return false;
    const actions = UNDO_KINDS[e.kind];
    return actions === null || (typeof e.action === "string" && actions.includes(e.action));
  }
  return false;
}

/** What an undoable row changed, in words: the key or the use, from and to. */
function changeOf(e: AuditRow): { what: string; from: string | null; to: string | null } {
  const v = (x: unknown) => (x === null ? "off" : x === undefined ? null : String(x));
  if (e.kind === "library") return { what: e.action === "trial" ? `stack ${String(e.stack ?? "")}` : `stack for ${String(e.use ?? "")}`, from: v(e.from), to: v(e.to) };
  if (e.kind === "job-filter") return { what: `${String(e.role ?? "")}/${String(e.job ?? "")} ${String(e.action ?? "")}`, from: v(e.from), to: v(e.to) };
  if (e.kind === "library-base") return { what: `base cartridge for ${String(e.model ?? "")}`, from: v(e.from), to: v(e.to) };
  if (e.kind === "role-model") return { what: `model of the ${String(e.role ?? "")}`, from: v(e.from), to: v(e.to) };
  if (e.kind in UNDO_KINDS) return { what: `${e.kind}${typeof e.action === "string" ? ` · ${e.action}` : ""}${typeof e.item === "string" ? ` · ${e.item}` : ""}`, from: v(e.from), to: v(e.to) };
  return { what: String(e.key), from: v(e.from), to: v(e.to) };
}

function episodeTitle(e: unknown): string {
  if (e && typeof e === "object") {
    const o = e as Record<string, unknown>;
    for (const k of ["question", "label", "kind", "id"]) if (typeof o[k] === "string") return (o[k] as string).slice(0, 120);
  }
  return typeof e === "string" ? e.slice(0, 120) : "episode";
}

/** The meta line of a row: its project, mode, route, time taken and state. */
function metaOf(e: AuditRow): string {
  return [typeof e.project === "string" ? `project ${maskName(e.project)}` : null, e.mode, e.via, e.state === "running" ? "running" : e.state === "stale" ? "never ended" : e.kind === "tool" || typeof e.ms === "number" ? msText(e.ms) : null, e.ok === false ? "failed" : null].filter(Boolean).join(" · ");
}

/** One record in full, as the engine wrote it, with paths shortened, anything secret removed and
 *  the question held back unless it was revealed. */
function RecordView({ id, value }: { id: string; value: unknown }) {
  const a = useRevealed(`call:${id}`);
  const b = useRevealed(id);
  const on = a || b;
  const shown =
    !on && value && typeof value === "object"
      ? Object.fromEntries(
          Object.entries(value as Record<string, unknown>).map(([k, x]) => {
            if (typeof x !== "string") return [k, x];
            if (k === "label") return [k, `${callLabel(x).short}${callLabel(x).hidden ? " · question hidden" : ""}`];
            if (k === "question" || k === "prompt" || k === "query" || k === "text") return [k, "hidden"];
            if (k === "project") return [k, maskName(x)];
            return [k, x];
          }),
        )
      : value;
  return <pre className="mono fade-in max-h-[50vh] overflow-auto border-l border-line pl-3 text-[10.5px] leading-relaxed text-muted">{maskPathsIn(JSON.stringify(shown, null, 2))}</pre>;
}

/** A call's label as plain text for a row (a row is a button, so no show control inside it): the
 *  question held back unless it was shown in the rail. */
function CallWords({ e }: { e: AuditRow }) {
  const on = useRevealed(`call:${e.id}`);
  const { short, hidden } = callLabel(e.label, e.tool);
  return <>{hidden ? (on ? `${short}: ${hidden}` : `${short} · question hidden`) : short}</>;
}

function EpisodeWords({ id, e }: { id: string; e: unknown }) {
  const on = useRevealed(id);
  return <>{on ? episodeTitle(e) : "Episode · question hidden"}</>;
}

/** Inline show/hide button beside a row in the center timeline, synced with the side panel's Masked control. */
function RevealToggle({ id }: { id: string }) {
  const on = useRevealed(id);
  if (captureMask()) return null;
  return (
    <button
      type="button"
      onClick={() => (on ? hide(id) : reveal(id))}
      aria-label={on ? "Hide question" : "Show question"}
      data-e2e="audit-row-reveal"
      className="press mono ml-1 min-h-10 shrink-0 self-center border-b border-line-strong px-1.5 text-[10px] uppercase tracking-[0.12em] text-faint hover:border-sense-pale hover:text-text sm:ml-2"
    >
      {on ? "hide" : "show"}
    </button>
  );
}

/** Undo one row through the engine, then read again everything an undo can change. */
async function undoRow(id: string): Promise<Note> {
  try {
    await post("/api/undo", { id });
    for (const p of ["/api/audit", "/api/controls", "/api/primes", "/api/projects", "/api/nanoignore", "/api/cartridges", "/api/cartridges/library", "/api/surgery", "/api/parameter-maps", "/api/devices"]) refreshAll(p);
    return { text: "Undone. The undo is logged too.", tone: "verified" };
  } catch (e) {
    return { text: e instanceof Error ? e.message : String(e), tone: "error" };
  }
}

/** The rail's view of one event: what changed with Undo, the parts it ran with their time, and
 *  the full record. Read live from the same log, so an undo shows at once. */
function EventDetail({ path, row, onNote }: { path: string; row: AuditRow; onNote: (n: Note) => void }) {
  const { data } = useApi<Audit>(path, 3000);
  const now = useNow();
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<Note | null>(null);
  const rows = data?.audit ?? [];
  const e = rows.find((r) => r.id === row.id) ?? row;
  const undone = rows.some((r) => r.undo_of === e.id);
  const ch = changeOf(e);
  const steps = (e.steps ?? []).filter((s) => typeof s.ms === "number");
  const maxStep = Math.max(1, ...steps.map((s) => s.ms ?? 0));
  return (
    <div className="grid gap-0" data-e2e="audit-detail" data-id={e.id}>
      <RailSection title="Event">
        <dl className="mono grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-[11px]">
          <dt className="text-faint">kind</dt>
          <dd className="text-right text-muted">{e.kind}</dd>
          <dt className="text-faint">when</dt>
          <dd className="text-right text-muted">
            {new Date(e.time_ms).toLocaleString()}
            {now > 0 ? ` · ${ago(e.time_ms, now)}` : ""}
          </dd>
          {typeof e.project === "string" && (
            <>
              <dt className="text-faint">project</dt>
              <dd className="truncate text-right text-muted">{maskName(e.project)}</dd>
            </>
          )}
          {(e.kind === "tool" || typeof e.ms === "number") && (
            <>
              <dt className="text-faint">took</dt>
              <dd className="text-right text-text">{e.state === "running" ? "running" : e.state === "stale" ? "never ended" : msText(e.ms)}</dd>
            </>
          )}
          {(e.mode || e.via) && (
            <>
              <dt className="text-faint">mode</dt>
              <dd className="text-right text-muted">{[e.mode, e.via].filter(Boolean).join(" · ")}</dd>
            </>
          )}
          <dt className="text-faint">state</dt>
          <dd className={`text-right ${e.ok === false ? "text-error" : "text-muted"}`}>{e.ok === false ? "failed" : e.ok === true ? "ok" : (e.state ?? "recorded")}</dd>
        </dl>
        {typeof e.error === "string" && <div className="mono mt-2 border-l-2 border-error pl-2 text-[11px] text-error">{maskPathsIn(e.error)}</div>}
      </RailSection>
      {(undoable(e) || typeof e.undo_of === "string") && (
        <RailSection title="Change">
          <NoteLine note={note} />
          <div className="mono grid gap-2 text-[11px]">
            <span className="text-faint">{ch.what}</span>
            {(ch.from !== null || ch.to !== null) && (
              <span className="text-muted">
                {ch.from ?? "not recorded"} <span className="text-quiet">to</span> <span className="text-text">{ch.to ?? "not recorded"}</span>
              </span>
            )}
            <span className="flex flex-wrap items-center gap-3">
              {typeof e.undo_of === "string" && <Pill>an undo</Pill>}
              {undone ? (
                <span className="text-faint">undone</span>
              ) : undoable(e) ? (
                <button
                  type="button"
                  disabled={busy}
                  data-e2e="audit-undo"
                  onClick={async () => {
                    setBusy(true);
                    const n = await undoRow(e.id);
                    setBusy(false);
                    setNote(n);
                    onNote(n);
                  }}
                  className="press mono inline-flex min-h-10 items-center border-b border-text text-[11px] uppercase tracking-[0.12em] text-text disabled:opacity-50"
                >
                  {busy ? "Undoing" : "Undo"}
                </button>
              ) : null}
            </span>
          </div>
        </RailSection>
      )}
      {steps.length > 0 && (
        <RailSection title={`Parts it ran · ${steps.length}`}>
          <ul className="mono grid gap-1 text-[10.5px]">
            {steps.map((s, i) => (
              <li key={`${s.part}-${i}`} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-0.5">
                <span className="truncate text-muted">{s.part}</span>
                <span className="text-text">{msText(s.ms, s.timed)}</span>
                <span className="col-span-2 block h-1 w-full bg-line/50">
                  <span className="block h-full bg-sense-pale" style={{ width: `${Math.max(1, ((s.ms ?? 0) / maxStep) * 100)}%` }} />
                </span>
              </li>
            ))}
          </ul>
        </RailSection>
      )}
      <RailSection title="Record">
        <RecordView id={e.id} value={e} />
      </RailSection>
    </div>
  );
}

/** 2026-11-26 (Luis Jake Gabriel III: audit log retention): how many days of use the tool-call rows
 *  are kept, and trimming now through the delete flow. Setting rows are always kept (Undo). */
function Retention() {
  const controls = useApi<Control[]>("/api/controls", 15000);
  const ctl = controls.data?.find((c) => c.key === "audit.keep_days") ?? null;
  const [note, setNote] = useState<DeleteNote | null>(null);
  return (
    <div className="grid gap-3" data-e2e="audit-retention">
      <NoteLine note={note} />
      {ctl ? <ControlRow c={ctl} foldHelp onNote={setNote} /> : <p className="mono text-[11px] text-faint">{controls.error ? controls.error : "Reading the retention setting."}</p>}
      <DeleteFlow<{ rows?: number; bytes_before?: number; bytes_after?: number }>
        path="/api/audit/retention"
        body={{}}
        label="Trim the audit log now"
        title="Removes tool-call rows older than the days set; setting changes are always kept. Shows what goes first."
        done={(r) => `Audit log trimmed: ${r?.rows ?? 0} rows removed, ${bytes(Math.max(0, (r?.bytes_before ?? 0) - (r?.bytes_after ?? 0)))} freed. Logged in Audit.`}
        refresh={["/api/audit", "/api/storage"]}
        onNote={setNote}
      />
    </div>
  );
}

/** The side panel's Kinds tab: what each kind filter holds (the engine's own kind names) and
 *  which kinds of change can be undone. Reference for the chips above the timeline. */
function KindsRail() {
  return (
    <div className="grid gap-0" data-e2e="audit-kinds">
      <RailSection title="Kind filters">
        <dl className="grid gap-2 text-[11px]">
          {KINDS.filter((k) => k !== "all").map((k) => (
            <div key={k} className="grid gap-0.5">
              <dt className="mono uppercase tracking-[0.08em] text-muted">{k}</dt>
              <dd className="mono leading-relaxed text-faint">{(KIND_GROUPS[k] ?? []).join(" · ")}</dd>
            </div>
          ))}
        </dl>
      </RailSection>
      <RailSection title="What can be undone">
        <ul className="mono grid gap-1 text-[11px] text-faint">
          <li>
            <span className="text-muted">setting</span> · every change
          </li>
          <li>
            <span className="text-muted">library</span> · use · trial
          </li>
          <li>
            <span className="text-muted">job-filter</span> · plugged in, out or removed
          </li>
          {Object.entries(UNDO_KINDS).map(([k, actions]) => (
            <li key={k}>
              <span className="text-muted">{k}</span> · {actions === null ? "every change" : actions.join(" · ")}
            </li>
          ))}
        </ul>
        <p className="read mt-2 text-[11px] leading-relaxed text-faint">An undoable row carries Undo beside it and in its detail. The undo is a row too; a row already undone says so. The state filter &quot;undoable&quot; lists only what can still be undone.</p>
      </RailSection>
    </div>
  );
}

function AuditAbout() {
  return (
    <div className="grid gap-0">
      <RailSection title="What this is">
        <p className="read text-xs leading-relaxed text-muted">Everything the Nano does: each call and what ran, what went to the agent, what was remembered and why, and every change to settings, memory and cartridges. Setting changes, stack switches and job filter changes can be undone; the undo is logged too.</p>
      </RailSection>
      <RailSection title="Reading the timeline">
        <ul className="read grid gap-1.5 text-[11px] text-faint">
          <li>Newest first, by day. Each mark on the spine is one event; its colour is its state: red failed, blue running, green a checked outcome.</li>
          <li>The bar on the right is how long it ran, against the longest event shown.</li>
          <li>The tape above shows every event shown along time, its height by how long it ran. A mark opens its event; the open event stands out on the tape.</li>
          <li>Kind and project filters ask the engine (the newest 200 rows); ok or failed filters those rows.</li>
        </ul>
      </RailSection>
      <RailSection title="Episodes">
        <p className="read text-[11px] text-faint">Kept with memory on, one per answered question, for recall and outcomes. The question stays hidden until shown.</p>
      </RailSection>
      <RailSection title="Priming">
        <p className="read text-[11px] text-faint">Every verified prime sent to your coding agent through lifecycle hooks is logged as a <span className="mono text-muted">prime</span> row (kind, trigger, cited places, no file contents) and filterable under <span className="mono text-muted">priming</span>. Open the Primes tab to inspect follow-through verdicts and toggle the priming control.</p>
      </RailSection>
    </div>
  );
}

const AUDIT_VIEWS = [
  { id: "log", label: "Audit Log", about: "Every tool call, setting change, prime delivery, and undoable action." },
  { id: "primes", label: "Primes Studio", about: "Alarm-style switches for built-in primes, custom triggers, intervals, schedules, and presets." },
  { id: "episodes", label: "Episodes", about: "Recorded question episodes for recall and verified outcomes." },
] as const;
type AuditView = (typeof AUDIT_VIEWS)[number]["id"];

const dayOf = (ms: number) => new Date(ms).toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short", year: "numeric" });
const timeOf = (ms: number) => new Date(ms).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" });

/** The working middle: top section bar (Audit Log | Primes Studio | Episodes), tape, filters, and timeline. */
function AuditWork({ view, onPickView }: { view: AuditView; onPickView: (v: AuditView) => void }) {
  const rail = useRail();
  const { key: globalProjectKey } = useProjectKey();
  const [kind, setKind] = useState("all");
  const [allProjects, setAllProjects] = useState(false);
  const [state, setState] = useState<StateFilter>("all");
  const [msg, setMsg] = useState<Note | null>(null);
  const now = useNow();
  const activeProject = allProjects ? "" : (globalProjectKey ?? "");
  const q = new URLSearchParams();
  if (kind !== "all") q.set("kind", (KIND_GROUPS[kind] ?? [kind]).join(","));
  if (activeProject) q.set("project", activeProject);
  q.set("limit", "200");
  const path = `/api/audit?${q.toString()}`;
  const { data, error } = useApi<Audit>(path, 3000);

  const all = [...(data?.audit ?? [])].reverse();
  const undone = new Set(all.flatMap((e) => (typeof e.undo_of === "string" ? [e.undo_of] : [])));
  const canUndo = (e: AuditRow) => undoable(e) && !undone.has(e.id);
  const rows = state === "all" ? all : all.filter((e) => (state === "failed" ? e.ok === false : state === "undoable" ? canUndo(e) : e.ok !== false));
  const failed = all.filter((e) => e.ok === false).length;
  const undoCount = all.filter(canUndo).length;
  const [undoing, setUndoing] = useState<string | null>(null);
  const episodes = [...(data?.episodes ?? [])].reverse();
  const maxMs = Math.max(1, ...rows.map((e) => (typeof e.ms === "number" ? e.ms : 0)));
  const shownKey = rail.shown?.key ?? null;

  const days: { day: string; rows: AuditRow[] }[] = [];
  for (const e of rows) {
    const d = dayOf(e.time_ms);
    const last = days[days.length - 1];
    if (last && last.day === d) last.rows.push(e);
    else days.push({ day: d, rows: [e] });
  }

  const openedFromUrl = useRef<string | null>(null);

  const open = (e: AuditRow) =>
    rail.show({
      key: `audit:${e.id}`,
      title: (
        <span className="flex min-w-0 items-center gap-2">
          <Dot tone={toneOf(e)} />
          <span className="min-w-0 truncate">{undoable(e) || e.kind !== "tool" ? titleOf(e) : <CallText id={e.id} label={e.label} tool={e.tool} />}</span>
        </span>
      ),
      body: <EventDetail path={path} row={e} onNote={setMsg} />,
    });

  useEffect(() => {
    const onReset = (e: Event) => {
      const href = (e as CustomEvent<{ href?: string }>).detail?.href;
      if (href !== "/audit/") return;
      window.history.pushState(null, "", "/audit/");
      rail.clear();
      setKind("all");
      setAllProjects(false);
      setState("all");
      onPickView("log");
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("nm:nav-reset", onReset);
    return () => window.removeEventListener("nm:nav-reset", onReset);
  }, [rail, onPickView]);

  useEffect(() => {
    if (typeof window === "undefined" || all.length === 0) return;
    const params = new URLSearchParams(window.location.search);
    const targetId = params.get("id");
    if (!targetId || openedFromUrl.current === targetId) return;
    const hit = all.find((e) => e.id === targetId);
    if (!hit) return;
    openedFromUrl.current = targetId;
    open(hit);
    const el = document.querySelector(`[data-audit-id="${CSS.escape(targetId)}"]`);
    el?.scrollIntoView({ block: "center", behavior: "smooth" });
  });

  return (
    <div className="grid min-w-0 gap-5">
      <SectionBar sections={AUDIT_VIEWS} value={view} onPick={(v) => onPickView(v)} label="Audit sections" />
      {data && view === "log" && rows.length > 0 && (
        <section aria-label="Tape of events" className="min-w-0">
          <div className="topo night ticks border border-line px-3 py-3">
            <Tape rows={rows} now={now} onPick={open} picked={shownKey?.startsWith("audit:") ? shownKey.slice(6) : null} />
          </div>
          <div className="mono mt-1.5 text-[10px] uppercase tracking-[0.16em] text-faint">Fig. 1 · every event along time · the parts it named beneath · a mark opens its event</div>
        </section>
      )}
      {view !== "primes" && (
      <div className="grid min-w-0 gap-3" data-e2e="audit-filters">
        {view === "log" && (
        <div role="radiogroup" aria-label="Kind" className="flex min-w-0 flex-wrap items-end gap-x-1 gap-y-1 border-b border-line">
          {KINDS.map((k) => {
            const on = k === kind;
            return (
              <button key={k} type="button" role="radio" aria-checked={on} onClick={() => setKind(k)} title={k === "all" ? "Every kind of event" : `Kinds: ${(KIND_GROUPS[k] ?? []).join(", ")}`} className={`press mono min-h-9 shrink-0 border-b-2 px-2.5 text-[10.5px] uppercase tracking-[0.1em] whitespace-nowrap ${on ? "border-sense-pale bg-bg-2/40 text-text" : "border-transparent text-faint hover:bg-bg-2/30 hover:text-text"}`} data-e2e="audit-kind" data-kind={k}>
                {k}
              </button>
            );
          })}
        </div>
        )}
        <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-2.5">
          <ProjectMenu placement="down" label="project" onSelect={() => setAllProjects(false)} />
          <button
            type="button"
            aria-pressed={allProjects}
            data-e2e="audit-all-projects-toggle"
            onClick={() => setAllProjects((v) => !v)}
            className={`press mono h-9 shrink-0 border px-2.5 text-[10px] uppercase tracking-[0.12em] ${
              allProjects
                ? "border-sense-pale bg-bg-2/70 text-text"
                : "border-line-strong text-faint hover:border-sense-pale hover:text-text"
            }`}
          >
            {allProjects ? "Showing all projects" : "All projects"}
          </button>
          {view === "log" && (
            <div className="w-full sm:w-auto sm:min-w-[20rem]">
              <Segmented values={[...STATES]} value={state} label="State" onPick={(v) => setState(v as StateFilter)} render={(v) => (v === "failed" && data ? `failed ${failed}` : v === "undoable" && data ? `undoable ${undoCount}` : v)} />
            </div>
          )}
        </div>
      </div>
      )}
      {view === "primes" && <PrimesStudio onBack={() => onPickView("log")} />}
      <NoteLine note={msg} />
      {error && !data && <Failed error={error} />}
      {!data && !error && <Placeholder className="h-80" what="audit log" />}
      {data && view === "log" && (
        rows.length === 0 ? (
          <Empty title="No events match">{state === "undoable" ? "Nothing here can be undone. The side panel, Kinds, lists the changes that show under this filter until they are undone." : "Change the kind, the project or the state above."}</Empty>
        ) : (
          <div className="grid gap-5" data-e2e="audit-log">
            {days.map((d, di) => (
              <section key={`${d.day}-${di}`} aria-label={d.day} className="min-w-0">
                <div className="mono sticky top-[6.75rem] z-10 flex items-baseline justify-between gap-3 border-b border-line bg-bg/90 py-1.5 text-[10.5px] uppercase tracking-[0.14em] text-faint backdrop-blur lg:top-[2.75rem]">
                  <span>{d.day}</span>
                  <span className="text-quiet">{d.rows.length}</span>
                </div>
                <ol className="relative">
                  <span aria-hidden className="absolute top-0 bottom-0 left-[4.25rem] w-px bg-line sm:left-[5.1rem]" />
                  {d.rows.map((e) => {
                    const tone = toneOf(e);
                    const on = shownKey === `audit:${e.id}`;
                    const ms = typeof e.ms === "number" ? e.ms : null;
                    const hasHidden = !(undoable(e) || e.kind !== "tool") && Boolean(callLabel(e.label, e.tool).hidden);
                    return (
                      <li key={e.id} data-audit-id={e.id} className={`flex min-w-0 items-stretch ${on ? "bg-bg-2/50 shadow-[inset_2px_0_0_var(--sense-pale)]" : "hover:bg-bg-1/50"}`}>
                        <button
                          type="button"
                          onClick={() => open(e)}
                          aria-pressed={on}
                          data-e2e="audit-row"
                          data-kind={e.kind}
                          className="press group grid min-h-10 min-w-0 flex-1 grid-cols-[3.5rem_1rem_minmax(0,1fr)_auto] items-center gap-x-1.5 py-1.5 text-left sm:grid-cols-[4rem_1.2rem_minmax(0,1fr)_auto] sm:gap-x-2"
                        >
                          <span className="mono pl-0.5 text-[10px] text-faint tabular-nums sm:pl-1 sm:text-[10.5px]">{timeOf(e.time_ms)}</span>
                          <span className="relative flex justify-center">
                            <i aria-hidden className={`relative z-[1] block h-2 w-2 rounded-full border ${on ? "scale-125" : ""}`} style={{ background: tone === "muted" ? "var(--bg-0)" : SPINE[tone], borderColor: SPINE[tone] }} />
                          </span>
                          <span className="min-w-0">
                            <span className="flex min-w-0 items-center gap-2">
                              <span className="mono hidden shrink-0 text-[10px] uppercase tracking-[0.08em] text-quiet sm:inline">{e.kind}</span>
                              <span className={`mono min-w-0 truncate text-[11.5px] sm:text-[12px] ${on ? "text-text" : "text-muted group-hover:text-text"}`}>{undoable(e) || e.kind !== "tool" ? titleOf(e) : <CallWords e={e} />}</span>
                              {undone.has(e.id) ? (
                                <span className="mono shrink-0 text-[10px] text-quiet">undone</span>
                              ) : typeof e.undo_of === "string" ? (
                                <span className="mono shrink-0 text-[10px] text-quiet">an undo</span>
                              ) : null}
                            </span>
                            <span className="mono block truncate text-[10px] text-faint sm:text-[10.5px]">{metaOf(e)}</span>
                          </span>
                          <span className="flex shrink-0 items-center gap-2">
                            {ms !== null && (
                              <span aria-hidden className="hidden h-1 w-14 bg-line/50 md:block" title={msText(ms)}>
                                <span className="block h-full" style={{ width: `${Math.max(2, Math.sqrt(ms / maxMs) * 100)}%`, background: e.ok === false ? "var(--error)" : "var(--sense-pale)" }} />
                              </span>
                            )}
                            <span className="mono hidden w-14 text-right text-[10.5px] text-faint sm:inline-block">{now > 0 ? ago(e.time_ms, now) : ""}</span>
                          </span>
                        </button>
                        {hasHidden && <RevealToggle id={`call:${e.id}`} />}
                        {canUndo(e) && (
                          <button
                            type="button"
                            disabled={undoing !== null}
                            title="Undo this change through the engine; the undo is logged too"
                            data-e2e="audit-row-undo"
                            onClick={async () => {
                              setUndoing(e.id);
                              setMsg(await undoRow(e.id));
                              setUndoing(null);
                            }}
                            className="press mono ml-1 min-h-10 shrink-0 self-center border-b border-sense-pale px-1.5 text-[10px] uppercase tracking-[0.12em] text-sense-pale hover:text-text disabled:opacity-40 sm:ml-2"
                          >
                            {undoing === e.id ? "undoing" : "undo"}
                          </button>
                        )}
                      </li>
                    );
                  })}
                </ol>
              </section>
            ))}
          </div>
        )
      )}
      {data && view === "episodes" && (
        episodes.length === 0 ? (
          <Empty title="No episodes for this project" />
        ) : (
          <ol className="border-t border-line">
            {episodes.map((e, i) => {
              const id = `ep:${i}`;
              const on = shownKey === id;
              return (
                <li key={id} className="flex min-w-0 items-stretch border-b border-line/60">
                  <button
                    type="button"
                    onClick={() => rail.show({ key: id, title: <Masked id={id} shown="Episode · question hidden" full={episodeTitle(e)} />, body: <RecordView id={id} value={e} /> })}
                    aria-pressed={on}
                    className={`press flex min-h-10 min-w-0 flex-1 items-center gap-3 px-1 py-1.5 text-left ${on ? "bg-bg-1/80 text-text" : "hover:bg-bg-1/50 hover:text-text"}`}
                  >
                    <Dot />
                    <span className="mono min-w-0 flex-1 truncate text-[12px] text-muted">
                      <EpisodeWords id={id} e={e} />
                    </span>
                  </button>
                  <RevealToggle id={id} />
                </li>
              );
            })}
          </ol>
        )
      )}
    </div>
  );
}

export default function AuditPage() {
  const { data, error } = useApi<Audit>("/api/audit?limit=200", 3000);
  const { key: projKey } = useProjectKey();
  const ledger = useApi<Ledger>(projKey ? `/api/ledger?project=${encodeURIComponent(projKey)}` : "/api/ledger", 10000);
  const hash = useHash();
  const [pickedView, setPickedView] = useState<AuditView | null>(null);
  const view: AuditView = pickedView ?? (hash === "#primes" ? "primes" : hash === "#episodes" ? "episodes" : "log");
  const setView = (next: AuditView) => {
    setPickedView(next);
    if (typeof window !== "undefined") {
      setHash(next === "log" ? "" : `#${next}`, false);
    }
  };
  const pc = primeCounts(ledger.data);
  const rows = data?.audit ?? [];
  const failed = rows.filter((e) => e.ok === false).length;
  const canUndo = rows.filter((e) => undoable(e) && !rows.some((r) => r.undo_of === e.id)).length;
  return (
    <Workspace
      title="Audit"
      status={
        <>
          <span className="flex items-center gap-2">
            <Dot tone={error ? "error" : data ? "verified" : "muted"} />
            {error ? "not connected" : data ? "live" : "connecting"}
          </span>
          {data && <span className="text-text">{rows.length} events</span>}
          {data && <span className={failed > 0 ? "text-error" : undefined}>{failed} failed</span>}
          {data && <span title="Changes the engine can still undo: settings, stacks, job filters, parameter maps, memory pins, goals, guidance, sources, backup repositories, teacher data">{canUndo} undoable</span>}
        </>
      }
      rail={[
        {
          id: "primes-rail",
          label: "Primes",
          count: pc?.given,
          body: <PrimesRail inStudio={view === "primes"} onOpenStudio={() => setView("primes")} onOpenLog={() => setView("log")} />,
        },
        {
          id: "stages",
          label: "Stage times",
          // The figures of a row stay on one line in the narrow panel (790 ms, not 790 over ms).
          body: (
            <div className="[&_td]:whitespace-nowrap [&_th]:whitespace-nowrap">
              <StageTimesPanel />
            </div>
          ),
        },
        { id: "retention", label: "Retention", body: <Retention /> },
        { id: "kinds", label: "Kinds", body: <KindsRail /> },
      ]}
      about={<AuditAbout />}
    >
      <AuditWork view={view} onPickView={setView} />
    </Workspace>
  );
}
