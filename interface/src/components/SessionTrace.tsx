"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "says no claude activity. we dont use just cluade so
// thats misleading. also its just missing a lot of real details"): the session is the one an agent's
// hooks record (any agent that has them set up); with `calls={false}` the page lists every agent's
// calls elsewhere, so the empty state says only that no hook session is recorded.

/** The coding agent session in play, from `/api/session` (API 4): the task as a short masked line,
 *  the files read and edited, the checks with pass or fail, what is open, and what the Nano served
 *  and each prime given with what followed, drawn as lanes along the session's time and as one
 *  path per pack.
 *  Every mark is a record the engine reports; nothing is placed by guess. */
import AgentCalls from "@/components/AgentCalls";
import Link from "next/link";
import { useState, type ReactNode } from "react";
import { Dot, Masked, Reticle } from "@/components/ui";
import { useRail } from "@/components/Workspace";
import { ago, useApi, useApiLevel, type Join, type PackView, type SessionReply, type SessionState } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { API_SESSION, placeShort, primesOf, primeWord, stateBg, stateWord, useProjectKey } from "@/lib/evidence";
import { maskName, scrub } from "@/lib/privacy";


function Lane({ label, children, note }: { label: string; children?: ReactNode; note?: ReactNode }) {
  return (
    <div className="grid grid-cols-[72px_minmax(0,1fr)] items-center gap-2 md:grid-cols-[92px_minmax(0,1fr)]">
      <span className="mono truncate text-[10px] uppercase tracking-[0.12em] text-faint">{label}</span>
      <span className="relative block h-3.5 border-b border-line">
        {children}
        {note && <span className="mono absolute inset-y-0 right-0 left-0 block truncate leading-[14px] text-[10px] text-quiet">{note}</span>}
      </span>
    </div>
  );
}

/** A mark on a lane at a fraction of the session's span. */
function Mark({ at, className, title, h = "h-2.5" }: { at: number; className: string; title: string; h?: string }) {
  return <i aria-hidden title={title} className={`absolute bottom-0 block ${h} ${className}`} style={{ left: `calc(${(Math.min(1, Math.max(0, at)) * 100).toFixed(2)}% - 3px)` }} />;
}

function isCheck(c: { kind?: string; class?: string }): boolean {
  return /^(test|build|check|lint)$/.test(c.kind ?? "") || /test|build|check|clippy/.test(c.class ?? "");
}

/** Detailed side-panel view for one context pack / prime from the session trace. */
function PackPathDetail({ id, at, label, ask, view, now }: { id: string; at: number; label: string; ask?: string | null; view?: PackView | null; now: number }) {
  const joins: Join[] = view?.joins ?? [];
  const open = view?.window === "open" || (!view?.closed && joins.length === 0);
  const fullAsk = ask ? scrub(ask) : null;
  return (
    <div className="grid gap-3" data-e2e="pack-path-detail">
      <div className="mono grid gap-1 border-b border-line/60 pb-2.5 text-[11px]">
        <div className="text-text">{label}</div>
        <div className="text-[10px] text-faint">
          id: {id}
          {now > 0 ? ` · ${ago(at, now)}` : ""} · {new Date(at).toLocaleString()}
        </div>
        <div className="text-[10px] text-quiet">
          {open ? "window open · judged when the turn ends" : view?.closed ? `closed · ${view.closed}` : `${joins.length} outcome(s)`}
        </div>
      </div>
      <div className="grid gap-1">
        <span className="eyebrow text-muted">Question</span>
        {fullAsk ? (
          <p className="read whitespace-pre-wrap break-words border-l-2 border-sense-pale pl-3 text-[12px] leading-relaxed text-text">{fullAsk}</p>
        ) : (
          <p className="mono text-[11px] text-quiet">Question text was not recorded for this item.</p>
        )}
      </div>
      <div className="grid gap-1.5 border-t border-line/60 pt-2.5">
        <span className="eyebrow text-muted">What followed ({joins.length})</span>
        {joins.length === 0 ? (
          <p className="mono text-[10.5px] text-quiet">{open ? "Still waiting for the turn to end." : "No follow-up events recorded."}</p>
        ) : (
          <ul className="grid gap-2">
            {joins.map((j, i) => (
              <li key={i} className="mono grid gap-0.5 border-l border-line-strong pl-2.5 text-[10.5px]">
                <div className="flex items-center gap-1.5 text-text">
                  <i aria-hidden className={`inline-block h-1.5 w-1.5 ${stateBg(j.state)}`} />
                  <span>{j.rule}</span>
                  <span className="text-muted">· {stateWord(j.state)}</span>
                  {j.strength && <span className="text-faint">({j.strength})</span>}
                </div>
                {j.event?.place && <div className="text-[10px] text-faint">{placeShort(j.event.place)}</div>}
                {j.why && <div className="read text-[11px] text-muted">{j.why}</div>}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

/** What followed one pack, as a path: the pack, then each finding in the order the rules ran. */
function PackPath({ id, at, label, ask, view, now }: { id: string; at: number; label: string; ask?: string | null; view?: PackView | null; now: number }) {
  const rail = useRail();
  const joins: Join[] = view?.joins ?? [];
  const open = view?.window === "open" || (!view?.closed && joins.length === 0);
  const openDetail = () =>
    rail.show({
      key: `pack:${id}`,
      title: label,
      body: <PackPathDetail id={id} at={at} label={label} ask={ask} view={view} now={now} />,
    });
  return (
    <li className="flex min-w-0 flex-wrap items-center gap-x-0 gap-y-1 py-[3px]">
      <button
        type="button"
        onClick={openDetail}
        title="Open full question and follow-up details"
        className="press mono flex shrink-0 items-center gap-1.5 text-[10.5px] text-muted hover:text-text"
      >
        <i aria-hidden className="inline-block h-1.5 w-1.5 rotate-45 bg-sense-pale" />
        {label}
      </button>
      {ask !== undefined && (
        <span className="mono ml-2 min-w-0 max-w-[20rem] truncate text-[10.5px] text-faint">
          <Masked id={`ask:${id}`} shown="question hidden" full={ask ? scrub(ask) : null} onDetail={openDetail} />
        </span>
      )}
      {now > 0 && <span className="mono ml-2 shrink-0 text-[10px] text-quiet">{ago(at, now)}</span>}
      {joins.map((j, i) => (
        <span key={i} className="flex shrink-0 items-center" title={[j.rule, stateWord(j.state), j.strength, j.event?.place ? placeShort(j.event.place) : null, j.why].filter(Boolean).join(" · ")}>
          <span aria-hidden className="mx-1.5 inline-block h-px w-3 bg-line-strong" />
          <span className="mono flex items-center gap-1 text-[10.5px] text-muted">
            <i aria-hidden className={`inline-block h-1.5 w-1.5 ${stateBg(j.state)}`} />
            {j.rule} · {stateWord(j.state)}
            {j.strength ? <span className="text-faint"> {j.strength}</span> : null}
          </span>
        </span>
      ))}
      {open && (
        <span className="flex shrink-0 items-center">
          <span aria-hidden className="mx-1.5 inline-block h-px w-3 border-t border-dashed border-line-strong" />
          <span className="mono text-[10.5px] text-quiet">window open · judged when the turn ends</span>
        </span>
      )}
      {!open && joins.length === 0 && view?.closed && <span className="mono ml-2 text-[10.5px] text-quiet">closed · {view.closed}</span>}
    </li>
  );
}

/** One open matter as a short line: a failing check by class and package with its first located
 *  error, a tool failure by tool, or a file changed and not checked since. Folders held back. */
function openText(kind: "check" | "tool" | "change", v: unknown): string {
  if (typeof v === "string") return placeShort(v);
  if (!v || typeof v !== "object") return kind;
  const o = v as Record<string, unknown>;
  const str = (k: string) => (typeof o[k] === "string" ? (o[k] as string) : "");
  if (kind === "change") return placeShort(str("path") || str("file")) || "a change";
  if (kind === "tool") return [str("tool") || "tool", Array.isArray(o.paths) && o.paths.length ? placeShort(String(o.paths[0])) : ""].filter(Boolean).join(" ");
  const errs = Array.isArray(o.errors) ? (o.errors as Record<string, unknown>[]) : [];
  const e = errs[0];
  const at = e && typeof e.file === "string" ? `${placeShort(e.file)}${typeof e.line === "number" ? `:${e.line}` : ""}` : "";
  return [str("class") || str("kind") || "check", str("pkg"), e && typeof e.code === "string" ? e.code : "", at].filter(Boolean).join(" ");
}

function Readout({ k, children }: { k: string; children: ReactNode }) {
  return (
    <span className="mono flex shrink-0 items-baseline gap-1.5 text-[10.5px]">
      <span className="uppercase tracking-[0.12em] text-faint">{k}</span>
      <span className="tnum text-muted">{children}</span>
    </span>
  );
}

function Trace({ s, now, packs }: { s: SessionState; now: number; packs: number }) {
  const t0 = s.first_ms;
  const t1 = Math.max(s.last_ms, t0 + 1);
  const x = (t: number) => (t - t0) / (t1 - t0);
  const files = s.files ?? [];
  const cmds = s.commands ?? [];
  const checks = cmds.filter(isCheck);
  const served = s.served ?? [];
  const primes = primesOf(s);
  const open = s.open ?? {};
  const nOpen = (open.failing_checks?.length ?? 0) + (open.tool_failures?.length ?? 0) + (open.unchecked_changes?.length ?? 0);
  const paths = [
    ...served.map((p, i) => ({ id: p.pack?.id ?? `p${i}`, at: p.t, label: p.tool === "nm_context_pack" ? "context pack" : (p.tool ?? "pack"), ask: p.ask ?? null, view: p.pack })),
    ...primes.map((n, i) => ({ id: n.id ?? `n${i}`, at: n.t, label: `prime · ${primeWord(n.kind)}`, ask: undefined, view: n.ledger })),
  ].sort((a, b) => b.at - a.at);
  return (
    <>
      <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1">
        <Readout k="files">
          {files.filter((f) => (f.reads ?? 0) > 0).length} read · {files.filter((f) => (f.edits ?? 0) + (f.writes ?? 0) > 0).length} edited
        </Readout>
        <Readout k="checks">
          <span className="text-verified">{checks.filter((c) => c.ok === true).length} pass</span> · <span className={checks.some((c) => c.ok === false) ? "text-error" : undefined}>{checks.filter((c) => c.ok === false).length} fail</span>
        </Readout>
        <Readout k="open">
          <span className={nOpen ? "text-attention" : undefined}>
            {open.failing_checks?.length ?? 0} failing · {open.tool_failures?.length ?? 0} tool failures · {open.unchecked_changes?.length ?? 0} unchecked
          </span>
        </Readout>
        <Readout k="served">{served.length} packs · {primes.length} {primes.length === 1 ? "prime" : "primes"}</Readout>
      </div>
      {files.length > 0 && (
        <div className="mono mt-1.5 flex min-w-0 gap-2 text-[10.5px]">
          <span className="shrink-0 uppercase tracking-[0.12em] text-faint">files</span>
          <span className="min-w-0 truncate text-muted" title="newest first; folders held back">
            {[...files]
              .sort((a, b) => (b.last ?? 0) - (a.last ?? 0))
              .slice(0, 6)
              .map((f) => `${placeShort(f.path)}${(f.edits ?? 0) + (f.writes ?? 0) > 0 ? " edited" : " read"}`)
              .join(" · ")}
          </span>
        </div>
      )}
      {nOpen > 0 && (
        <div className="mono mt-1 flex min-w-0 gap-2 text-[10.5px]">
          <span className="shrink-0 uppercase tracking-[0.12em] text-attention">open</span>
          <span className="min-w-0 truncate text-muted">
            {[
              ...(open.failing_checks ?? []).map((v) => `failing ${openText("check", v)}`),
              ...(open.tool_failures ?? []).map((v) => `tool failed ${openText("tool", v)}`),
              ...(open.unchecked_changes ?? []).map((v) => `unchecked ${openText("change", v)}`),
            ].join(" · ")}
          </span>
        </div>
      )}
      <div className="mt-2.5 grid gap-[3px]" role="img" aria-label={`Session over ${Math.round((t1 - t0) / 1000)} s: ${files.length} files, ${checks.length} checks, ${served.length} packs`}>
        <Lane label="Task">
          {(s.starts ?? []).map((st, i) => (
            <Mark key={`s${i}`} at={x(st.t)} title={`session ${st.source}`} className="w-1.5 border border-faint" h="h-1.5" />
          ))}
          {s.task && <Mark at={x(s.task.t)} title="newest prompt" className="w-[2px] bg-text" h="h-3" />}
        </Lane>
        <Lane label="Files">
          {files.map((f, i) =>
            typeof f.last === "number" ? <Mark key={i} at={x(f.last)} title={`${placeShort(f.path)} · ${f.reads ?? 0} read · ${(f.edits ?? 0) + (f.writes ?? 0)} edited`} className={(f.edits ?? 0) + (f.writes ?? 0) > 0 ? "w-1.5 bg-sense-pale" : "w-1.5 border border-sense-pale"} h={(f.edits ?? 0) + (f.writes ?? 0) > 0 ? "h-2.5" : "h-1.5"} /> : null,
          )}
        </Lane>
        <Lane label="Checks">
          {cmds.map((c, i) => (
            <Mark key={i} at={x(c.t)} title={`${c.class ?? c.kind ?? "command"}${c.pkg ? ` ${c.pkg}` : ""} · ${c.ok === true ? "passed" : c.ok === false ? "failed" : "ran"}`} className={!isCheck(c) ? "w-px bg-line-3" : c.ok === false ? "w-1.5 bg-error" : c.ok === true ? "w-1.5 bg-verified" : "w-1.5 border border-faint"} h={isCheck(c) ? "h-2" : "h-1.5"} />
          ))}
        </Lane>
        <Lane label="Nano">
          {paths.map((p) =>
            (p.view?.joins ?? []).map((j, k) =>
              typeof j.event?.at === "number" ? (
                <i key={`${p.id}-${k}`} aria-hidden className="absolute bottom-[3px] block h-px bg-line-strong" style={{ left: `${(x(p.at) * 100).toFixed(2)}%`, width: `${(Math.max(0, x(j.event.at) - x(p.at)) * 100).toFixed(2)}%` }} />
              ) : null,
            ),
          )}
          {paths.map((p) =>
            (p.view?.joins ?? []).map((j, k) => (typeof j.event?.at === "number" ? <Mark key={`${p.id}-m${k}`} at={x(j.event.at)} title={`${j.rule} · ${stateWord(j.state)}`} className={`w-1.5 ${stateBg(j.state)}`} h="h-1.5" /> : null)),
          )}
          {paths.map((p) => (
            <i key={p.id} aria-hidden title={p.label} className={`absolute bottom-[1px] block h-2 w-2 rotate-45 ${p.label.startsWith("prime") ? "bg-behav" : "bg-sense-pale"}`} style={{ left: `calc(${(x(p.at) * 100).toFixed(2)}% - 4px)` }} />
          ))}
        </Lane>
        <div className="mono grid grid-cols-[72px_minmax(0,1fr)] gap-2 text-[10px] text-quiet md:grid-cols-[92px_minmax(0,1fr)]">
          <span />
          <span className="flex justify-between gap-3">
            <span>session start{now > 0 ? ` · ${ago(t0, now)}` : ""}</span>
            <span>last record{now > 0 ? ` · ${ago(t1, now)}` : ""}</span>
          </span>
        </div>
      </div>
      {paths.length > 0 ? (
        <ol className="mt-2 min-h-0 flex-1 overflow-y-auto">
          {paths.slice(0, packs).map((p) => (
            <PackPath key={p.id} id={p.id} at={p.at} label={p.label} ask={p.ask} view={p.view} now={now} />
          ))}
        </ol>
      ) : (
        <div className="mono mt-2 text-[10.5px] text-quiet">The Nano served nothing in this session yet.</div>
      )}
    </>
  );
}

/** Empty lanes, drawn so the strip keeps its shape while there is nothing to show. */
function Quiet({ children }: { children: ReactNode }) {
  return (
    <div className="mt-1 grid gap-[3px]">
      <Lane label="Task" note={children} />
      <Lane label="Files" />
      <Lane label="Checks" />
      <Lane label="Nano" />
    </div>
  );
}

export default function SessionTrace({ packs = 3, className = "", calls = true }: { packs?: number; className?: string; calls?: boolean }) {
  const level = useApiLevel();
  const { key } = useProjectKey();
  const [picked, setPicked] = useState<{ project: string | null; id: string | null }>({ project: null, id: null });
  const has = level !== null && level >= API_SESSION;
  const pickedId = picked.project === key ? picked.id : null;
  const url = has && key ? `/api/session?project=${encodeURIComponent(key)}${pickedId ? `&session=${encodeURIComponent(pickedId)}` : ""}` : null;
  const r = useApi<SessionReply>(url, 5000);
  const now = useNow();
  const sessions = [...(r.data?.sessions ?? [])].reverse();
  if (pickedId && r.data?.sessions && !r.data.sessions.some((x) => x.session === pickedId)) {
    setPicked({ project: key, id: null });
  }
  const s = r.data?.session ?? null;
  const activeSession = s?.session ?? pickedId ?? sessions[0]?.session ?? "";
  const task = s?.task;
  return (
    <section aria-label="Coding agent session" className={`flex min-h-0 min-w-0 flex-col ${className}`}>
      <div className="flex min-w-0 items-center justify-between gap-4 border-b border-line pb-2">
        <h2 className="eyebrow flex min-w-0 items-center gap-2 text-muted">
          <Reticle />
          <span className="shrink-0">Session in play</span>
          {key && (
            <span className="mono min-w-0 truncate normal-case tracking-normal text-faint">
              <Masked id={`project:${key}`} shown={key === "nano" ? "the Nano's own folder" : `project ${maskName(key)}`} full={key} />
            </span>
          )}
        </h2>
        <span className="flex shrink-0 items-center gap-3">
          {sessions.length > 1 && (
            <label className="relative">
              <span className="sr-only">Session</span>
              <select
                value={activeSession}
                onChange={(e) => setPicked({ project: key, id: e.target.value || null })}
                className="press mono h-7 max-w-[150px] appearance-none border-b border-line-strong bg-transparent pr-5 pl-1 text-[10.5px] text-muted hover:border-text focus-visible:border-sense-pale sm:max-w-[190px]"
              >
                {sessions.map((item, idx) => (
                  <option key={item.session} value={item.session}>
                    {idx === 0 ? "latest" : item.session.slice(0, 10)} · {item.records} rec{now > 0 ? ` · ${ago(item.last_ms, now)}` : ""}
                  </option>
                ))}
              </select>
              <svg aria-hidden viewBox="0 0 16 16" className="pointer-events-none absolute top-1/2 right-1 h-3 w-3 -translate-y-1/2 text-faint">
                <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.2" />
              </svg>
            </label>
          )}
          {s && (
            <span className="mono hidden text-[10.5px] text-faint sm:inline">
              {s.records} records{s.compactions ? ` · ${s.compactions} compactions` : ""}
            </span>
          )}
          <Link href="/learning/" className="link mono text-[11px]">
            Ledger
          </Link>
        </span>
      </div>
      <div className="mono mt-2 flex min-w-0 items-baseline gap-2 text-[11px]">
        <span className="shrink-0 uppercase tracking-[0.12em] text-faint">Task</span>
        <span className="min-w-0 truncate text-muted">
          {!s ? (
            <span className="text-quiet">–</span>
          ) : !task ? (
            <span className="text-quiet">no prompt recorded</span>
          ) : (
            <Masked id={`task:${s.session}`} shown={`prompt of ${task.chars ?? "?"} characters · text hidden`} full={task.text ? scrub(task.text) : null} />
          )}
          {s && task && !task.text && <span className="text-quiet"> · text not kept (prompts off or past its days)</span>}
        </span>
      </div>
      <div className="mt-2 flex min-h-0 flex-1 flex-col">
        {level === null ? (
          <Quiet>connecting</Quiet>
        ) : !has ? (
          <Quiet>this engine build (API {level}) does not report the session; it comes with API {API_SESSION}</Quiet>
        ) : r.error && !r.data ? (
          <Quiet>
            <span className="flex items-center gap-1.5">
              <Dot tone="error" /> the session did not answer ({r.error})
            </span>
          </Quiet>
        ) : !r.data ? (
          <Quiet>reading the session</Quiet>
        ) : !s && !calls ? (
          <Quiet>no hook session recorded yet · every agent&apos;s calls are under Agent calls</Quiet>
        ) : !s ? (
          // 2026-11-26: without a hook session, every coding agent's calls on this project.
          <AgentCalls project={key ?? null} />
        ) : (
          <Trace s={s} now={now} packs={packs} />
        )}
      </div>
    </section>
  );
}
