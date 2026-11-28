"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And
// that whole information database for it should be on a right side panel. The main core of the
// pages should be on actual work flow and core function visuals ... you have to think of it as if
// it's an interactive system for the main parts the right menu are libraries and or discription
// tools etc. left menu are the main interface nav like it is now. ... We need the interface to be
// amazing."): the continual learning loop as a working system (what came in, the gates, what is
// training or waiting, what was kept, the learned units in brief) and the rail's libraries (the
// evidence ledger, the primes, the training store, the cartridges).

import Link from "next/link";
import { useState, type ReactNode } from "react";
import { ControlRow, NoteLine } from "@/components/Controls";
import DeleteFlow from "@/components/DeleteFlow";
import { compartmentName, type Compartment, type LearnedUnitsReply } from "@/components/LearnedUnits";
import { Big, Block, Row } from "@/components/MemoryPanels";
import { useRail } from "@/components/Workspace";
import { setHash } from "@/lib/hash";
import { Empty, Fold, Pill, Switch } from "@/components/ui";
import { ago, bytes, post, refreshAll, useApi, type Cartridge, type ContinualLearning, type Control, type Examples, type Ledger, type LedgerRow } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { API_EXAMPLES, API_OUTCOMES, API_SESSION, domainShort, isGuidedRow, isPrimeRow, overall, packsOf, pct, placeShort, primeCounts, primeWord, RULES, stateBg, stateText, stateWord } from "@/lib/evidence";
import { gateLine, learnedLine, type IdleCycle, type Learned, type JobFilterRow } from "@/lib/learned";
import { maskName } from "@/lib/privacy";

type Note = { text: string; tone: "verified" | "error" };

/* ---------- Small figures ---------- */

/** A labelled figure on a hairline. `value` null while it is read. */
export function Fig({ label, value, caption, tone = "text-text" }: { label: string; value: string | null; caption: string; tone?: string }) {
  return (
    <div className="min-w-0 border-t border-line pt-2.5">
      <div className="eyebrow">{label}</div>
      <div className="mt-2">
        <Big value={value} size={24} tone={tone} />
      </div>
      <div className="mono mt-1 truncate text-[10.5px] text-faint">{caption}</div>
    </div>
  );
}

/** The order of the matrix's columns: what a rule found, strongest first within each state. */
const COLS = ["used/direct", "used/indirect", "used/weak", "verified/direct", "verified/indirect", "verified/weak", "refuted/direct", "refuted/indirect", "refuted/weak", "passed-over/indirect", "passed-over/weak", "none", "unknown"];

/** Counts by rule and by what it found: one row per rule, one column per state and strength. */
function RuleMatrix({ counts }: { counts: Record<string, Record<string, number>> }) {
  const seen = new Set(Object.values(counts).flatMap((m) => Object.keys(m)));
  const cols = [...COLS.filter((c) => seen.has(c)), ...[...seen].filter((c) => !COLS.includes(c)).sort()];
  const rules = [...RULES.filter(([r]) => r in counts), ...Object.keys(counts).filter((r) => !RULES.some(([x]) => x === r)).map((r) => [r, ""] as [string, string])];
  const max = Math.max(1, ...Object.values(counts).flatMap((m) => Object.values(m)));
  if (!cols.length) return null;
  return (
    <div className="overflow-x-auto">
      <div role="table" aria-label="Evidence by rule and finding" className="grid min-w-[480px]" style={{ gridTemplateColumns: `minmax(110px,1.4fr) repeat(${cols.length}, minmax(0,1fr))` }}>
        <div role="row" className="contents">
          <span role="columnheader" className="mono border-b border-line-strong pb-2 text-[10px] uppercase tracking-[0.12em] text-faint">
            rule
          </span>
          {cols.map((c) => {
            const [st, str] = c.split("/");
            return (
              <span role="columnheader" key={c} className="mono min-w-0 border-b border-line-strong pb-2 text-center text-[9.5px] leading-tight">
                <span className={`block truncate uppercase tracking-[0.06em] ${stateText(st)}`}>{stateWord(st)}</span>
                <span className="block truncate text-quiet">{str ?? " "}</span>
              </span>
            );
          })}
        </div>
        {rules.map(([r, about]) => (
          <div role="row" key={r} className="contents">
            <span role="rowheader" className="min-w-0 border-b border-line/60 py-1.5" title={about || undefined}>
              <span className="mono block truncate text-[10.5px] text-muted">{r}</span>
            </span>
            {cols.map((c) => {
              const v = counts[r]?.[c] ?? 0;
              const st = c.split("/")[0];
              return (
                <span role="cell" key={c} className="flex items-center justify-center gap-1 border-b border-line/60 py-1.5">
                  {v > 0 ? (
                    <>
                      <i aria-hidden className={`inline-block h-1.5 w-1.5 ${stateBg(st)}`} style={{ opacity: 0.35 + 0.65 * (v / max) }} />
                      <span className="mono tnum text-[11px] text-text">{v}</span>
                    </>
                  ) : (
                    <span className="mono text-[10px] text-quiet">·</span>
                  )}
                </span>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

/** One ledger row as a line of the tape: when, what kind, what it found and where. */
function TapeRow({ r, now }: { r: LedgerRow; now: number }) {
  const at = r.at ?? r.event?.at ?? r.t ?? 0;
  let mark = "bg-line-3";
  let text = r.row;
  let sub = "";
  if (r.row === "evidence") {
    mark = stateBg(r.state);
    text = `${r.rule} · ${stateWord(r.state)}${r.strength ? ` ${r.strength}` : ""}`;
    sub = [r.event?.place ? placeShort(r.event.place) : null, typeof r.steps_after === "number" ? `${r.steps_after} steps after` : null, r.closed ? `closed: ${r.closed}` : null].filter(Boolean).join(" · ");
  } else if (r.row === "served") {
    mark = "bg-sense-pale";
    text = "served · context pack";
    sub = [`${r.cited?.length ?? 0} places cited`, r.joined_by ? `joined by ${r.joined_by}` : null].filter(Boolean).join(" · ");
  } else if (isPrimeRow(r)) {
    mark = "bg-behav";
    text = `prime given · ${primeWord(typeof r.kind === "string" ? r.kind : undefined)}`;
    sub = `${Array.isArray(r.places) ? r.places.length : 0} places${r.hook ? ` · at ${r.hook}` : ""}`;
  } else if (isGuidedRow(r)) {
    mark = "bg-behav";
    text = `guidance delivered · ${typeof r.kind === "string" ? r.kind : "item"} ${typeof r.item === "string" ? r.item : ""}`.trim();
    sub = [r.hook ? `at ${r.hook}` : null, typeof r.channel === "string" && r.channel ? r.channel : null].filter(Boolean).join(" · ");
  } else if (r.row === "outcome") {
    const standing = typeof r.standing === "string" ? r.standing : typeof r.state === "string" ? r.state : "";
    mark = stateBg(standing);
    text = `outcome · ${typeof r.kind === "string" ? r.kind : "decided"}${standing ? ` · ${standing}` : ""}`;
    sub = "handed to memory";
  }
  return (
    <li className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-baseline gap-x-3 border-b border-line/60 py-1.5">
      <i aria-hidden className={`inline-block h-1.5 w-1.5 translate-y-[-1px] ${mark}`} />
      <span className="min-w-0">
        <span className="mono block truncate text-[11px] text-muted">{text}</span>
        {sub && <span className="mono block truncate text-[10px] text-faint">{sub}</span>}
      </span>
      <span className="mono shrink-0 text-[10px] text-faint">{now > 0 && at ? ago(at, now) : ""}</span>
    </li>
  );
}

/** Counts of one kind: name, count, a hairline under the name by its share. */
function Counts({ title, counts, word = (k: string) => k, tone = () => "bg-faint" }: { title: string; counts?: Record<string, number>; word?: (k: string) => string; tone?: (k: string) => string }) {
  const list = Object.entries(counts ?? {}).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...list.map(([, n]) => n));
  return (
    <div className="min-w-0">
      <div className="eyebrow mb-1">{title}</div>
      {list.length === 0 ? (
        <div className="mono text-[10.5px] text-quiet">none</div>
      ) : (
        <ul>
          {list.slice(0, 6).map(([k, n]) => (
            <li key={k} className="grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-3 py-0.5">
              <span className="relative min-w-0">
                <span className="mono block truncate text-[10.5px] text-muted">{word(k)}</span>
                <i aria-hidden className={`absolute bottom-0 left-0 block h-px ${tone(k)}`} style={{ width: `${(n / max) * 100}%` }} />
              </span>
              <span className="mono tnum text-[10.5px] text-text">{n}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ---------- Rail: evidence, primes, training store, cartridges ---------- */

/** The evidence ledger: what followed each context pack and each prime, by rule and strength,
 *  the newest rows as a tape, and the clear through the delete flow. */
export function LearningEvidence({ ledger, level, projectKey, pending }: { ledger: { data: Ledger | null; error: string | null }; level: number | null; projectKey: string | null; pending: boolean }) {
  const now = useNow();
  const hasLedger = level !== null && level >= API_SESSION;
  const L = ledger.data;
  const sum = overall(L);
  const empty = L && (L.rows?.length ?? 0) === 0 && (L.pending?.length ?? 0) === 0;
  const tape = [...(L?.rows ?? [])].sort((a, b) => (b.at ?? b.event?.at ?? b.t ?? 0) - (a.at ?? a.event?.at ?? a.t ?? 0)).slice(0, 14);
  return (
    <div className="grid gap-5">
      <section className="grid gap-3">
        <h3 className="eyebrow flex flex-wrap items-baseline justify-between gap-2">
          <span>Evidence ledger</span>
          {L?.chain && (
            <span className={`mono normal-case tracking-normal ${L.chain.intact ? "text-faint" : "text-error"}`}>
              {L.chain.rows} rows · chain {L.chain.intact ? "intact" : `broken: ${L.chain.fault ?? "fault"}`}
            </span>
          )}
        </h3>
        {level !== null && !hasLedger ? (
          <span className="mono text-[11px] text-faint">Not reported by this engine build (API {level}; the ledger comes with API {API_SESSION}).</span>
        ) : ledger.error && !L ? (
          <Empty title="The ledger did not answer">{ledger.error}</Empty>
        ) : !L ? (
          <span className="mono text-[11px] text-faint">{pending ? "reading the ledger" : "no ledger"}</span>
        ) : empty ? (
          <span className="mono text-[11px] text-faint" title="With the hooks on, each context pack and each prime is judged when its turn ends: read, edited, checked, undone or asked again.">
            No evidence yet in this project.
          </span>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-x-5 gap-y-3">
              <Fig label="Packs served" value={String(sum.served)} caption={`${sum.closed} judged · ${sum.open} open`} />
              <Fig label="Then used" value={pct(sum.used, sum.closed)} caption={`${sum.used} of ${sum.closed} read or edited`} tone="text-sense-pale" />
              <Fig label="Verified" value={pct(sum.verified, sum.closed)} caption={`${sum.verified} by a check or a prompt`} tone="text-verified" />
              <Fig label="Refuted" value={pct(sum.refuted, sum.closed)} caption={`${sum.refuted} · ${sum.passed} passed over`} tone={sum.refuted ? "text-error" : "text-text"} />
            </div>
            {typeof L.outcomes === "number" && level !== null && level >= API_OUTCOMES && (
              <span className="mono text-[10.5px] text-faint">
                {L.outcomes} decided {L.outcomes === 1 ? "outcome" : "outcomes"} handed to{" "}
                <Link href="/memory/" className="link">
                  Memory
                </Link>
              </span>
            )}
            {L.counts && Object.keys(L.counts).length > 0 && (
              <Fold title="Findings by rule and strength">
                <RuleMatrix counts={L.counts} />
              </Fold>
            )}
          </>
        )}
      </section>
      {L && !empty && (
        <section>
          <h3 className="eyebrow mb-1 flex items-baseline justify-between gap-2">
            <span>Recent evidence</span>
            <span className="mono normal-case tracking-normal text-faint">newest {tape.length}</span>
          </h3>
          <ol>
            {tape.map((r, i) => (
              <TapeRow key={`${r.seq ?? i}-${r.row}`} r={r} now={now} />
            ))}
          </ol>
          {(L.pending?.length ?? 0) > 0 && <div className="mono mt-1.5 text-[10px] text-quiet">{L.pending!.length} still open: judged when their turn ends.</div>}
        </section>
      )}
      {hasLedger && projectKey && L && (
        <div className="border-t border-line pt-3">
          <DeleteFlow<{ rows?: number; audit_id?: string }>
            key={projectKey}
            path="/api/ledger/clear"
            body={{ project: projectKey }}
            label="Clear evidence ledger"
            title="Removes this project's evidence ledger; older packs are not rewritten. Asks what would go first."
            done={(r) => `Evidence ledger cleared: ${r?.rows ?? 0} rows removed. Packs served before now will not be rewritten. Logged in Audit.`}
            refresh={["/api/ledger", "/api/session", "/api/outcomes"]}
          />
        </div>
      )}
    </div>
  );
}

/** Primes given and whether each was followed, with the priming control. */
export function LearningPrimes({ ledger, level, control }: { ledger: Ledger | null; level: number | null; control: Control | null }) {
  const [note, setNote] = useState<Note | null>(null);
  const pc = primeCounts(ledger);
  const primes = packsOf(ledger?.rows ?? []).filter((p) => p.prime);
  return (
    <div className="grid gap-3">
      <h3 className="eyebrow flex items-baseline justify-between gap-2">
        <span>Primes</span>
        {pc && <span className="mono normal-case tracking-normal text-faint">{pc.given ?? 0} given</span>}
      </h3>
      {level !== null && level < API_OUTCOMES ? (
        <span className="mono text-[11px] text-faint">Not reported by this engine build (API {level}; primes come with API {API_OUTCOMES}).</span>
      ) : !ledger ? (
        <span className="mono text-[11px] text-faint">reading</span>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-x-4">
            <Fig label="Followed" value={String(pc?.followed ?? 0)} caption="read or edited" tone="text-sense-pale" />
            <Fig label="Passed over" value={String(pc?.passed_over ?? 0)} caption="went elsewhere" />
            <Fig label="Open" value={String(pc?.window_open ?? 0)} caption="not judged yet" />
          </div>
          {primes.length === 0 ? (
            <span className="mono text-[11px] text-faint">No prime given in this project yet.</span>
          ) : (
            <ol>
              {primes.slice(0, 8).map((n) => (
                <li key={n.pack} className="grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-3 border-b border-line/60 py-1.5">
                  <span className="mono min-w-0 truncate text-[11px] text-muted">{primeWord(n.kind)}</span>
                  <span className={`mono shrink-0 text-[10px] ${stateText(n.followed)}`}>{n.followed ? (n.followed === "used" ? "followed" : stateWord(n.followed)) : "open"}</span>
                </li>
              ))}
            </ol>
          )}
        </>
      )}
      {control && (
        <div className="border-t border-line">
          <NoteLine note={note} />
          <ControlRow c={control} foldHelp onNote={setNote} />
        </div>
      )}
      <div className="grid gap-2 border-t border-line pt-3">
        <span className="eyebrow text-faint">How priming nudges your agent</span>
        <p className="text-[11.5px] leading-relaxed text-muted">
          Primes are short, verified nudges injected into your coding agent (Claude Code or Antigravity) through lifecycle hooks at the exact moment prior memory or live impact helps steer it. Silence is the default: every cited line is re-read fresh on disk, no prime repeats twice in a session, and any prime exceeding the self-calibrating latency deadline is dropped.
        </p>
        <ul className="grid gap-1.5 text-[11px] leading-relaxed text-faint">
          <li><span className="mono text-text">prompt-recall</span>: surfaces verified past answers at prompt submit.</li>
          <li><span className="mono text-text">failure-fix</span>: returns the verified fix when the same build or test error recurs.</li>
          <li><span className="mono text-text">edit-impact</span>: warns before edits touching a verified fix, a refuted outcome, or a function signature with live callers.</li>
          <li><span className="mono text-text">resume / core</span>: restores active session state after compaction or surfaces top <span className="mono">1-core</span> memories at session open.</li>
          <li><span className="mono text-text">open-loop</span>: flags unverified edits or failing checks at turn stop (<span className="mono">priming = on</span>).</li>
        </ul>
        <p className="text-[11px] leading-relaxed text-faint">
          Explicit objectives, completion loops, step-graph directions, and trigger reminders ride on the same hook channel and are configured on <span className="text-muted">Memory -&gt; Loops and graph</span> (or by the agent over MCP <span className="mono">nm_goal</span>).
        </p>
      </div>
    </div>
  );
}

/** The training store (API 7): what the Nano keeps from verified work for fitting later. */
export function LearningStore({ store, error, level, control, projectKey }: { store: Examples | null; error: string | null; level: number | null; control: Control | null; projectKey: string | null }) {
  const [note, setNote] = useState<Note | null>(null);
  const now = useNow();
  const f = store?.families;
  const fund = f?.fundamentals;
  const fundN = typeof fund === "number" ? fund : (fund?.hardware_rows ?? 0) + (fund?.references ?? 0);
  const total = (f?.learning ?? 0) + (f?.preference ?? 0) + fundN;
  const hasExamples = level === null || level >= API_EXAMPLES;
  return (
    <section className="grid gap-3">
      <h3 className="eyebrow flex items-baseline justify-between gap-2">
        <span>Training store</span>
        {store && (
          <span className="mono normal-case tracking-normal text-faint">
            keeps {store.setting ?? "?"} · {bytes(store.bytes ?? 0)}
          </span>
        )}
      </h3>
      {level !== null && level < API_EXAMPLES ? (
        <span className="mono text-[11px] text-faint">Not reported by this engine build (API {level}; the store comes with API {API_EXAMPLES}).</span>
      ) : error && !store ? (
        <Empty title="The training store did not answer">{error}</Empty>
      ) : !store ? (
        <span className="mono text-[11px] text-faint">reading</span>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-x-4">
            <Fig label="Learning" value={String(f?.learning ?? 0)} caption="from packs" tone="text-sense-pale" />
            <Fig label="Preference" value={String(f?.preference ?? 0)} caption="your way" />
            <Fig label="Fundamentals" value={String(fundN)} caption={fund && typeof fund === "object" ? `${fund.hardware_rows ?? 0} hw · ${fund.references ?? 0} refs` : "rows"} />
          </div>
          {total === 0 ? (
            <span className="mono text-[10.5px] text-faint">
              Nothing kept yet in this project{store.memory === "off" ? "; memory is off" : store.setting === "off" ? "; keeping is off" : ""}.
            </span>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-x-5 gap-y-3">
                <Counts title="verdict" counts={store.verdicts} word={stateWord} tone={(k) => stateBg(k === "corrected" ? "verified" : k)} />
                <Counts title="split" counts={store.splits} />
                <Counts title="domain" counts={store.domains} word={(d) => domainShort(d, maskName)} />
                <Counts title="candidate label" counts={store.labels} word={stateWord} tone={(k) => stateBg(k)} />
              </div>
              {(store.examples?.length ?? 0) > 0 && (
                <ol>
                  {store.examples!.slice(0, 6).map((x) => {
                    const at = typeof x.at === "number" ? x.at : typeof x.t === "number" ? x.t : 0;
                    return (
                      <li key={x.id} className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-baseline gap-x-3 border-b border-line/60 py-1.5">
                        <i aria-hidden className={`inline-block h-1.5 w-1.5 ${stateBg(x.verdict === "corrected" ? "verified" : x.verdict)}`} />
                        <span className="mono min-w-0 truncate text-[10.5px] text-muted">
                          {stateWord(x.verdict)}
                          {x.strength ? ` ${x.strength}` : ""} · {x.split ?? "?"}
                          {x.anchor ? ` · ${placeShort(x.anchor)}` : ""} · {x.candidates?.length ?? 0} judged
                        </span>
                        <span className="mono shrink-0 text-[10px] text-faint">{now > 0 && at ? ago(at, now) : ""}</span>
                      </li>
                    );
                  })}
                </ol>
              )}
            </>
          )}
          {(store.judged_waiting ?? 0) > 0 && <span className="mono text-[10px] text-quiet">{store.judged_waiting} judged sets waiting for their turn to close</span>}
        </>
      )}
      {hasExamples && projectKey && store && (
        <div className="border-t border-line pt-3">
          <DeleteFlow<{ rows?: number }>
            key={projectKey}
            path="/api/examples/clear"
            body={{ project: projectKey }}
            label="Clear training store"
            title="Removes this project's stored training examples and judged sets. Asks what would go first."
            done={(r) => `Training store cleared: ${r?.rows ?? 0} rows removed. Logged in Audit.`}
            refresh={["/api/examples"]}
          />
        </div>
      )}
      {control && (
        <div className="border-t border-line">
          <NoteLine note={note} />
          <ControlRow c={control} foldHelp onNote={setNote} />
        </div>
      )}
    </section>
  );
}

function CartList({ title, list, none }: { title: string; list: Cartridge[]; none: string }) {
  return (
    <div className="min-w-0">
      <div className="eyebrow mb-1.5">
        {title} · {list.length}
      </div>
      {list.length === 0 ? (
        <div className="text-[11px] leading-relaxed text-faint">{none}</div>
      ) : (
        <ul className="grid gap-1.5">
          {list.map((c, i) => {
            const ver = (c as { version?: number }).version;
            return (
              <li key={`${c.id}@${ver ?? i}`} className="flex items-center gap-2 text-[11px]">
                <span aria-hidden className={c.loaded ? "spectral-sw !h-4 !w-1" : "dither-slot !h-4 !w-2.5"} />
                <span className="mono min-w-0 flex-1 truncate text-muted">
                  {c.id}
                  {typeof ver === "number" ? <span className="text-faint"> v{ver}</span> : null}
                </span>
                <span className={c.loaded ? "text-verified" : "text-faint"}>{c.loaded ? "in use" : "in the cartridge library"}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

/** The cartridges: learned (learning and preference families) apart from the fundamentals. */
export function LearningCartridges({ learned, fund, loaded }: { learned: Cartridge[]; fund: Cartridge[]; loaded: boolean }) {
  return (
    <div className="grid gap-4">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="eyebrow">Cartridges</h3>
        <Link href="/cartridges/" className="link mono text-[11px]">
          all on Cartridges
        </Link>
      </div>
      {!loaded ? (
        <span className="mono text-[11px] text-faint">reading</span>
      ) : (
        <div className="grid gap-5" data-e2e="learning-cartridges">
          <CartList title="Learned cartridges in the cartridge library" list={learned} none="None in the cartridge library yet: preference and judge cartridges appear here when a cycle keeps one. Lenses, read-outs and learned pieces are learned units, on Cartridges." />
          <CartList title="Fundamentals · base and hardware" list={fund} none="None in the cartridge library." />
        </div>
      )}
    </div>
  );
}

/* ---------- The loop ---------- */

/** A measured share as a percent; the engine reports these in 0..1. */
const pc1 = (x: number) => `${(x * 100).toFixed(x >= 0.995 || x === 0 ? 0 : 1)}%`;

/** The figures this version does not measure yet: named, never shown as numbers. */
const NOT_MEASURED = ["context fidelity", "reconstruction accuracy", "policy effectiveness", "transfer to new tasks"];

/** What each continual learning cartridge holds. */
const CART_STANDING: Record<string, { word: string; tone: "muted" | "attention" | "verified" | "error" }> = {
  empty: { word: "empty", tone: "muted" },
  observed: { word: "holding, not cycled", tone: "attention" },
  candidate: { word: "candidate", tone: "attention" },
  under_evaluation: { word: "under evaluation", tone: "attention" },
  validated: { word: "validated", tone: "verified" },
  rejected: { word: "rejected", tone: "error" },
};

type Learner = { name: string; promoted?: boolean; ran?: boolean; gate?: string; held_out?: number; current?: number | null; candidate?: number | null; note?: string };
type CycleRow = { id?: string; at?: number; trigger?: string; ran?: string[]; rolled_back?: boolean; learners?: Learner[] };
/** Fields of `/api/learning` the shared type does not name yet. */
type Extra = { idle_cycle?: IdleCycle; backlog?: number; gated_measured?: number; gated_held_back?: number; held_out_n?: number; held_out_score?: number | null; held_out_baseline?: number | null };

const fx = (n: number | null | undefined, d = 3) => (typeof n === "number" ? n.toFixed(d) : "not measured");

function CycleDetail({ c, now }: { c: CycleRow; now: number }) {
  const learners = c.learners ?? [];
  return (
    <div className="grid gap-3">
      <div>
        <Row k="when">{c.at && now > 0 ? ago(c.at, now) : "not reported"}</Row>
        <Row k="trigger">{c.trigger ?? "not reported"}</Row>
        <Row k="rolled back">{c.rolled_back ? "yes" : "no"}</Row>
      </div>
      <div className="eyebrow">Learners · {learners.length}</div>
      {learners.length === 0 ? (
        <span className="mono text-[11px] text-faint">none reported</span>
      ) : (
        <ul className="grid gap-2">
          {learners.map((l, i) => (
            <li key={`${l.name}-${i}`} className="border-b border-line/60 pb-2">
              <div className="flex items-center justify-between gap-2">
                <span className="mono truncate text-[11.5px] text-text">{l.name}</span>
                <Pill tone={!l.ran ? "muted" : l.promoted ? "verified" : "attention"}>{!l.ran ? "did not run" : l.promoted ? "kept" : "held back"}</Pill>
              </div>
              {l.gate && <div className="mono mt-0.5 text-[10px] text-faint">gate: {l.gate}</div>}
              {(typeof l.current === "number" || typeof l.candidate === "number") && (
                <div className="mono text-[10px] text-faint">
                  current {fx(l.current)} · candidate {fx(l.candidate)}
                  {typeof l.held_out === "number" ? ` · ${l.held_out} held out` : ""}
                </div>
              )}
              {l.note && <div className="mt-0.5 text-[10.5px] text-faint">{l.note}</div>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function GateDetail({ d, learned }: { d: ContinualLearning & Extra; learned: Learned }) {
  const idle = d.idle_cycle;
  return (
    <div className="grid gap-3">
      <p className="text-[11.5px] leading-relaxed text-faint">
        A cycle is kept only when its candidate predicts held-out outcomes better than the current state; while the held-out part is too small the supervisor rule decides, and the cycle says so.
      </p>
      <div>
        <Row k="next automatic cycle">{idle && typeof idle.threshold === "number" ? `${idle.verified_since_last_cycle ?? 0} of ${idle.threshold} verified examples${idle.due ? " · due" : ""}` : "not reported"}</Row>
        <Row k="cycle mode">{d.cycle_mode ?? "not reported"}</Row>
        <Row k="cycles measured">{d.gated_measured ?? "not reported"}</Row>
        <Row k="held back by the gate">{d.gated_held_back ?? "not reported"}</Row>
        <Row k="held-out score">{typeof d.held_out_score === "number" ? `${d.held_out_score.toFixed(3)} on ${d.held_out_n ?? 0}` : "not measured"}</Row>
        <Row k="first measured">{fx(d.held_out_baseline)}</Row>
        <Row k="split gate">{gateLine(learned)}</Row>
      </div>
    </div>
  );
}

function KeptDetail({ d }: { d: ContinualLearning }) {
  const pols = d.policies ?? [];
  const pats = d.patterns ?? [];
  return (
    <div className="grid gap-4">
      <div>
        <div className="eyebrow mb-1">Scoped policies in use · {pols.length}</div>
        {pols.length === 0 ? (
          <span className="mono text-[11px] text-faint">none yet</span>
        ) : (
          <ul className="grid gap-2">
            {pols.map((p) => (
              <li key={p.id} className="border-l-2 border-sense-pale pl-3 text-[11px]">
                <div className="mono text-text">
                  {p.name} · scope {p.applies_to} · {p.evidence_count} evidence
                </div>
                <div className="mt-0.5 text-muted">{p.expected_improvement}</div>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div>
        <div className="eyebrow mb-1">Patterns · {pats.length}</div>
        {pats.length === 0 ? (
          <span className="mono text-[11px] text-faint">none yet</span>
        ) : (
          <ul className="grid gap-2">
            {pats.slice(0, 12).map((p) => (
              <li key={p.id} className="border-b border-line/60 pb-1.5 text-[11px]">
                <div className="mono text-text">
                  {p.kind} · {p.category} · {p.status}
                </div>
                <div className="mt-0.5 text-muted">{p.generalized_lesson}</div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

/** One step of the loop: a number with its caption, opening its detail. */
function Step({ n, label, value, caption, tone, onPick, title, last = false }: { n: number; label: string; value: string | null; caption: string; tone?: string; onPick: () => void; title: string; last?: boolean }) {
  return (
    <li className="relative min-w-0 border-r border-b border-line">
      <button type="button" onClick={onPick} title={title} className="press grid h-full w-full content-start gap-1.5 px-3 py-2.5 text-left hover:bg-bg-2/40">
        <span className="flex items-baseline justify-between gap-2">
          <span className="eyebrow truncate">
            <span className="text-quiet">{n}</span> {label}
          </span>
          {!last && (
            <span aria-hidden className="mono text-[11px] text-quiet">
              →
            </span>
          )}
        </span>
        <Big value={value} tone={tone} />
        <span className="mono line-clamp-2 text-[10px] leading-snug text-faint" title={caption}>
          {caption}
        </span>
      </button>
    </li>
  );
}

/** The continuous learning engine as a working loop: the actions, the six steps from verified
 *  work to learned units, what is waiting, the measured figures, the six cartridges and the
 *  cycles. Every step opens its detail in the rail. */
export function LearningEngine({
  data,
  error,
  projectKey,
  ledger,
  ledgerPending,
  store,
  storePending,
  units,
  learned,
  jobFilters,
  training,
}: {
  data: ContinualLearning | null;
  error: string | null;
  projectKey: string | null;
  ledger: Ledger | null;
  ledgerPending: boolean;
  store: Examples | null;
  storePending: boolean;
  units: LearnedUnitsReply | null;
  learned: Learned;
  jobFilters: JobFilterRow[] | null;
  training: ReactNode;
}) {
  const [busy, setBusy] = useState<"" | "cycle" | "rollback" | "sync">("");
  const [armBack, setArmBack] = useState(false);
  const [note, setNote] = useState<Note | null>(null);
  const now = useNow();
  const rail = useRail();

  const runAction = async (action: "cycle" | "rollback" | "sync") => {
    if (!projectKey) return;
    setBusy(action);
    try {
      const res = await post<ContinualLearning & { cycle?: { label?: string; ran?: string[]; learners?: { name: string; promoted?: boolean }[] } }>("/api/learning/cycle", { project: projectKey, action });
      const text =
        action === "sync"
          ? `${res.cycle?.label ?? "Stored examples brought in"}. Logged in Audit.`
          : action === "rollback"
            ? `Rolled back the newest learning cycle; engine now v${res.version}. Logged in Audit.`
            : `Learning cycle ran (engine v${res.version}, ${res.cycles_run} ${res.cycles_run === 1 ? "cycle" : "cycles"} in all${res.cycle?.ran?.length ? `; ran: ${res.cycle.ran.join(", ")}` : ""}). Logged in Audit.`;
      setNote({ text, tone: "verified" });
      refreshAll("/api/learning");
      refreshAll("/api/audit");
      refreshAll("/api/cartridges");
    } catch (e) {
      setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });
    }
    setBusy("");
    setArmBack(false);
  };

  const d = data as (ContinualLearning & Extra) | null;
  const idle = d?.idle_cycle ?? null;
  const cycles = (idle?.cycles ?? []) as CycleRow[];
  const live = cycles.filter((c) => !c.rolled_back);
  const measures = d?.measures ?? [];
  const held = measures.find((m) => m.key === "held_out");
  const backlog = d?.backlog ?? 0;
  const lv = d?.levels;
  const sum = overall(ledger);
  const kept = (lv?.knowledge?.patterns ?? 0) + (lv?.knowledge?.procedural ?? 0) + (lv?.behavioral?.policies ?? 0);
  const examples = store ? (store.families?.learning ?? 0) + (store.families?.preference ?? 0) : null;
  const off = d?.learning === "off";
  const btn = "press mono min-h-10 border-b px-1 text-[11px] uppercase tracking-[0.12em] disabled:opacity-40";

  return (
    <Block
      id="loop"
      title="Continuous learning engine"
      right={
        d ? (
          <Pill tone={off ? "muted" : "verified"}>
            {off ? "off" : "on"} · v{d.version} · {d.cycles_run} {d.cycles_run === 1 ? "cycle" : "cycles"} · {d.cycle_mode ?? "auto"}
          </Pill>
        ) : undefined
      }
      hint="Each judged outcome enters as an experience. A cycle turns experiences into patterns, procedures and scoped policies, and keeps them only when they predict held-out outcomes better than before."
    >
      <NoteLine note={note} />
      {error && !d ? (
        <Empty title="The learning engine did not answer">{error}</Empty>
      ) : !d ? (
        <span className="mono text-[11px] text-faint">reading the learning engine</span>
      ) : (
        <div className="grid gap-5">
          <div className="flex flex-wrap items-center gap-4" data-e2e="learning-actions">
            <button type="button" disabled={!!busy || !projectKey || off} onClick={() => void runAction("cycle")} className={`${btn} border-sense-pale text-sense-pale hover:text-text`}>
              {busy === "cycle" ? "Running cycle" : "Run a learning cycle now"}
            </button>
            <button
              type="button"
              disabled={!!busy || !projectKey || off || backlog === 0}
              title={backlog === 0 ? "Every stored example is already in the engine" : `${backlog} stored rows not in the engine yet`}
              onClick={() => void runAction("sync")}
              className={`${btn} border-line-strong text-muted hover:text-text`}
            >
              {busy === "sync" ? "Bringing in" : backlog > 0 ? `Sync ${backlog} stored` : "Synced"}
            </button>
            {live.length > 0 &&
              (!armBack ? (
                <button type="button" disabled={!!busy} onClick={() => setArmBack(true)} className={`${btn} border-attention text-attention hover:text-text`}>
                  Roll back the newest cycle
                </button>
              ) : (
                <>
                  <button type="button" disabled={!!busy} onClick={() => void runAction("rollback")} className={`${btn} border-error text-error`}>
                    {busy === "rollback" ? "Rolling back" : "Confirm roll back"}
                  </button>
                  <button type="button" disabled={!!busy} onClick={() => setArmBack(false)} className={`${btn} border-transparent text-faint hover:text-text`}>
                    Keep it
                  </button>
                </>
              ))}
            {off && <span className="mono text-[10.5px] text-faint">learning is off: switch it on to run a cycle</span>}
          </div>

          <ol className="grid grid-cols-2 border-t border-l border-line sm:grid-cols-3 2xl:grid-cols-6" data-e2e="learning-loop">
            <Step
              n={1}
              label="Verified"
              value={ledger ? String(sum.verified) : ledgerPending ? null : "not reported"}
              caption={ledger ? `of ${sum.closed} judged · ${sum.open} open` : "no ledger"}
              tone="text-verified"
              title="Context packs whose outcome was verified. Open the evidence ledger."
              onPick={() => rail.open("evidence")}
            />
            <Step
              n={2}
              label="Examples"
              value={examples !== null ? String(examples) : storePending ? null : "not reported"}
              caption={store ? `${store.judged_waiting ?? 0} judged sets waiting` : "no training store"}
              tone="text-sense-pale"
              title="Examples kept in the training store. Open the training data."
              onPick={() => rail.open("training")}
            />
            <Step
              n={3}
              label="Experiences"
              value={String(lv?.experiential?.episodes ?? 0)}
              caption={backlog > 0 ? `${backlog} stored rows not in yet` : "every stored row taken in"}
              title="Level 1: each judged outcome, live in the engine."
              onPick={() =>
                rail.show({
                  key: "step:experiences",
                  title: `Experiences · ${lv?.experiential?.episodes ?? 0}`,
                  body: (
                    <div>
                      <Row k="level 1 · experiences">{lv?.experiential?.episodes ?? 0}</Row>
                      <Row k="taken in">{lv?.experiential?.cycle ?? "not reported"}</Row>
                      <Row k="stored, not in yet">{backlog}</Row>
                      <Row k="level 2 · patterns">{lv?.knowledge?.patterns ?? 0}</Row>
                      <Row k="level 2 · procedures">{lv?.knowledge?.procedural ?? 0}</Row>
                      <Row k="level 2 · semantic">{lv?.knowledge?.semantic ?? 0}</Row>
                      <Row k="level 3 · policies">{lv?.behavioral?.policies ?? 0}</Row>
                      <Row k="level 3 · preferences">{lv?.behavioral?.preferences ?? 0}</Row>
                      <Row k="retrieval adapted">{lv?.behavioral?.retrieval_adapted ? "yes" : "no"}</Row>
                    </div>
                  ),
                })
              }
            />
            <Step
              n={4}
              label="Gate"
              value={idle && typeof idle.threshold === "number" ? `${idle.verified_since_last_cycle ?? 0}/${idle.threshold}` : "not reported"}
              caption={idle?.due ? "cycle due" : `${d.gated_measured ?? 0} measured · ${d.gated_held_back ?? 0} held back`}
              tone={idle?.due ? "text-attention" : "text-text"}
              title="Verified examples toward the next automatic cycle, and the held-out gate's record."
              onPick={() => rail.show({ key: "step:gate", title: "The gate", body: <GateDetail d={d} learned={learned} /> })}
            />
            <Step
              n={5}
              label="Kept"
              value={String(kept)}
              caption={`${lv?.knowledge?.patterns ?? 0} pat · ${lv?.knowledge?.procedural ?? 0} proc · ${lv?.behavioral?.policies ?? 0} pol`}
              tone="text-verified"
              title="Patterns, procedures and scoped policies the gate kept."
              onPick={() => rail.show({ key: "step:kept", title: `Kept · ${kept}`, body: <KeptDetail d={d} /> })}
            />
            <Step
              n={6}
              label="Units acting"
              value={units ? `${units.acting}/${units.total}` : "not reported"}
              caption={learnedLine(learned)}
              tone="text-learned"
              title="Learned units acting now. Their home is the Cartridges page."
              last
              onPick={() => setHash("#units", false)}
            />
          </ol>

          {ledger && sum.closed > 0 && (
            // 2026-11-26: how much of what was judged was verified, as a bar (counts from the ledger).
            <div className="grid gap-1" data-e2e="verified-bar">
              <div className="h-1 w-full bg-line" role="img" aria-label={`${sum.verified} of ${sum.closed} judged packs verified`}>
                <div className="h-full bg-verified" style={{ width: `${Math.round((sum.verified / sum.closed) * 100)}%` }} />
              </div>
              <span className="mono text-[10px] text-faint">
                {sum.verified} of {sum.closed} judged packs verified · {sum.refuted} refuted
              </span>
            </div>
          )}

          {backlog > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-3 border-l-2 border-attention pl-3" data-e2e="learning-backlog">
              <span className="mono text-[11px] text-muted">
                {backlog} stored {backlog === 1 ? "row waits" : "rows wait"} to enter the engine (next settle or cycle, or now)
              </span>
              <button type="button" disabled={!!busy || !projectKey || off} onClick={() => void runAction("sync")} className={`${btn} border-attention text-attention hover:text-text`}>
                {busy === "sync" ? "Bringing in" : "Bring them in"}
              </button>
            </div>
          )}

          <div className="grid gap-5 lg:grid-cols-2">
            <div className="min-w-0">
              <div className="eyebrow mb-1.5">Training or waiting</div>
              {training}
              <div className="mono mt-1.5 grid gap-0.5 text-[10.5px] text-faint">
                <span>
                  next automatic cycle:{" "}
                  {idle && typeof idle.threshold === "number" ? `${idle.verified_since_last_cycle ?? 0} of ${idle.threshold} verified${idle.due ? ", due" : ""}` : "not reported"}
                </span>
                <span>last cycle: {d.last_cycle_ms && now > 0 ? ago(d.last_cycle_ms, now) : "none yet"}</span>
                <span>windows open: {ledger ? sum.open : "not reported"} · judged sets waiting: {store ? (store.judged_waiting ?? 0) : "not reported"}</span>
              </div>
              {idle && typeof idle.threshold === "number" && idle.threshold > 0 && (
                <div aria-hidden className="mt-2 h-px w-full bg-line">
                  <div className="h-px bg-attention" style={{ width: `${Math.min(1, (idle.verified_since_last_cycle ?? 0) / idle.threshold) * 100}%` }} />
                </div>
              )}
            </div>
            <div className="min-w-0" data-e2e="learning-measures">
              <div className="eyebrow mb-1.5 flex flex-wrap items-baseline justify-between gap-2">
                <span>Measured</span>
                {held && (
                  <span className="mono normal-case tracking-normal text-faint">
                    held-out {held.value.toFixed(3)}
                    {held.baseline !== null ? ` · first ${held.baseline.toFixed(3)}` : ""} · {held.n} held out
                  </span>
                )}
              </div>
              {measures.filter((m) => m.key !== "held_out").length === 0 ? (
                <span className="mono text-[10.5px] text-faint">nothing measured yet; no figure is shown before it is measured</span>
              ) : (
                <ul>
                  {measures
                    .filter((m) => m.key !== "held_out")
                    .map((m) => (
                      <li key={m.key} className="grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-3 border-b border-line/60 py-1" title={m.what}>
                        <span className="min-w-0 truncate text-[11px] text-muted">{m.label}</span>
                        <span className="mono tnum text-[11px] text-text">
                          {pc1(m.value)} <span className="text-[10px] text-faint">of {m.n}</span>
                        </span>
                      </li>
                    ))}
                </ul>
              )}
              <div className="mono mt-1.5 text-[10px] text-quiet">not measured in this version: {NOT_MEASURED.join(" · ")}</div>
            </div>
          </div>

          {(d.cartridges?.length ?? 0) > 0 && (
            <div>
              <div className="eyebrow mb-1.5">Continual learning cartridges · {d.cartridges!.length}</div>
              <ol className="grid grid-cols-2 border-t border-l border-line sm:grid-cols-3 2xl:grid-cols-6">
                {d.cartridges!.map((c) => {
                  const st = CART_STANDING[c.status] ?? { word: c.status, tone: "muted" as const };
                  return (
                    <li key={c.id} className="min-w-0 border-r border-b border-line">
                      <button
                        type="button"
                        title={c.summary}
                        onClick={() =>
                          rail.show({
                            key: `clc:${c.id}`,
                            title: `${c.domain} · v${c.version}`,
                            body: (
                              <div>
                                <p className="mb-2 text-[11.5px] leading-relaxed text-faint">{c.summary}</p>
                                <Row k="standing">{st.word}</Row>
                                <Row k="entries">{c.entries}</Row>
                                <Row k="evidence">{c.evidence_count}</Row>
                                <Row k="confidence">{typeof c.confidence === "number" ? c.confidence.toFixed(2) : "not measured"}</Row>
                                <Row k="updated">{c.updated_ms && now > 0 ? ago(c.updated_ms, now) : "never"}</Row>
                                <Row k="archived">{c.archived ? "yes" : "no"}</Row>
                              </div>
                            ),
                          })
                        }
                        className="press grid w-full gap-1 px-2.5 py-2 text-left hover:bg-bg-2/40"
                      >
                        <span className="flex items-center justify-between gap-1">
                          <span className="mono truncate text-[10.5px] uppercase tracking-[0.08em] text-text">{c.domain}</span>
                          <span className="mono text-[10px] text-faint">v{c.version}</span>
                        </span>
                        <span className="mono text-[10px] text-muted">
                          {c.entries} {c.entries === 1 ? "entry" : "entries"} · {c.evidence_count} ev
                        </span>
                        <Pill tone={st.tone}>{st.word}</Pill>
                      </button>
                    </li>
                  );
                })}
              </ol>
            </div>
          )}

          <div data-e2e="learning-cycles">
            <div className="eyebrow mb-1.5 flex flex-wrap items-baseline justify-between gap-2">
              <span>Cycles · what was learned</span>
              <span className="mono normal-case tracking-normal text-faint">{cycles.length} kept in the record</span>
            </div>
            {cycles.length === 0 ? (
              <span className="mono text-[10.5px] text-faint">No cycle has run in this project yet.</span>
            ) : (
              <ol>
                {[...cycles]
                  .reverse()
                  .slice(0, 6)
                  .map((c, i) => {
                    const keptNames = (c.learners ?? []).filter((l) => l.promoted).map((l) => l.name);
                    const gates = (c.learners ?? []).filter((l) => l.ran && l.gate).map((l) => `${l.name}: ${l.gate}`);
                    return (
                      <li key={c.id ?? i}>
                        <button type="button" onClick={() => rail.show({ key: `cycle:${c.id ?? i}`, title: `Cycle · ${c.trigger ?? "cycle"}`, body: <CycleDetail c={c} now={now} /> })} className="press grid w-full grid-cols-[auto_minmax(0,1fr)_auto] items-baseline gap-x-3 border-b border-line/60 py-1.5 text-left hover:bg-bg-2/40">
                          <i aria-hidden className={`inline-block h-1.5 w-1.5 ${c.rolled_back ? "bg-quiet" : keptNames.length ? "bg-verified" : "bg-attention"}`} />
                          <span className="min-w-0">
                            <span className="mono block truncate text-[11px] text-muted">
                              {c.trigger ?? "cycle"} · {keptNames.length ? `kept ${keptNames.join(", ")}` : "nothing kept"}
                              {c.rolled_back ? " · rolled back" : ""}
                            </span>
                            {gates.length > 0 && <span className="mono block truncate text-[10px] text-faint">{gates.join(" · ")}</span>}
                          </span>
                          <span className="mono shrink-0 text-[10px] text-faint">{now > 0 && c.at ? ago(c.at, now) : ""}</span>
                        </button>
                      </li>
                    );
                  })}
              </ol>
            )}
          </div>

          {jobFilters && jobFilters.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="eyebrow mr-1">Job filters</span>
              {jobFilters.map((jf) => (
                <span key={`${jf.role}-${jf.cartridge}`} title={jf.enabled ? "plugged in" : "built, not plugged in"} className={`mono border border-line px-1.5 py-0.5 text-[10px] ${jf.enabled ? "text-text" : "text-quiet line-through"}`}>
                  {jf.role}/{jf.job}
                  {jf.base_cos !== undefined && jf.filtered_cos !== undefined ? ` (${jf.filtered_cos - jf.base_cos >= 0 ? "+" : ""}${(jf.filtered_cos - jf.base_cos).toFixed(4)} cos)` : ""}
                </span>
              ))}
              <Link href="/parameters/" className="link mono text-[10.5px]">
                on Parameters
              </Link>
            </div>
          )}
        </div>
      )}
    </Block>
  );
}

/* ---------- The learned units, in brief ---------- */

function UnitsDetail({ c }: { c: Compartment }) {
  return (
    <div className="grid gap-3">
      <p className="text-[11.5px] leading-relaxed text-faint">{c.holds}.</p>
      {c.units.length === 0 ? (
        <span className="mono text-[11px] text-faint">Nothing learned in this compartment yet.</span>
      ) : (
        <ul className="grid gap-2">
          {c.units.map((u) => (
            <li key={u.key} className="border-b border-line/60 pb-2">
              <div className="flex items-center justify-between gap-2">
                <span className="mono truncate text-[11px] text-text" title={u.id}>
                  {u.name ? `${u.name} · ` : ""}
                  {u.id}
                </span>
                {u.acts ? <Pill tone="verified">acting</Pill> : <Pill title={u.why_not ?? undefined}>{u.plugged ? "not acting" : "unplugged"}</Pill>}
              </div>
              {u.does && <div className="mt-0.5 text-[10.5px] text-muted">{u.does}</div>}
              {u.measured && <div className="mono mt-0.5 text-[10px] text-faint">measured: {u.measured}</div>}
            </li>
          ))}
        </ul>
      )}
      <Link href="/cartridges/#adapters" className="link mono text-[11px]">
        plug units in or out on Cartridges
      </Link>
    </div>
  );
}

/** Every compartment on one line each: how many units act, and its switch for all of them. The
 *  units themselves are managed on Cartridges. */
export function LearningUnits({ projectKey }: { projectKey: string | null }) {
  const reply = useApi<LearnedUnitsReply>("/api/learned-units", 10000);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [note, setNote] = useState<Note | null>(null);
  const [fresh, setFresh] = useState<LearnedUnitsReply | null>(null);
  const rail = useRail();
  const data = fresh ?? reply.data;
  const flip = async (c: Compartment, on: boolean) => {
    setBusyKey(c.switch_key);
    setNote(null);
    try {
      const r = await post<{ audit_id: string | null; learned_units: LearnedUnitsReply }>("/api/learned-units", { compartment: c.key, on });
      setFresh(r.learned_units);
      setNote({ text: r.audit_id ? `${on ? "Compartment switched on" : "Compartment switched off"}; applies at the next call of the serving process. Audit ${r.audit_id} undoes it.` : "Already so; nothing changed.", tone: "verified" });
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
  return (
    <Block
      id="units"
      title="Learned units"
      hint="Everything learned, kept beside the frozen models and never merged into them. A compartment switched off gives the base's outputs again, bit for bit."
      right={
        <span className="flex items-center gap-4">
          {data && (
            <span className="mono text-[10.5px] text-faint">
              {data.acting} of {data.total} acting
            </span>
          )}
          <Link href="/cartridges/#adapters" className="link mono text-[10.5px]">
            manage on Cartridges
          </Link>
        </span>
      }
    >
      {reply.error && !data ? (
        <Empty title="The learned units did not answer">{reply.error}</Empty>
      ) : !data ? (
        <span className="mono text-[11px] text-faint">reading</span>
      ) : (
        <ul className="grid grid-cols-1 border-t border-l border-line sm:grid-cols-2 2xl:grid-cols-3" data-e2e="learned-units">
          {comps.map((c) => {
            const acting = c.units.filter((u) => u.acts).length;
            return (
              <li key={c.key} className={`flex min-w-0 items-center justify-between gap-2 border-r border-b border-line pl-3 ${c.units.length === 0 ? "opacity-70" : ""}`} data-e2e="compartment" data-key={c.key}>
                <button type="button" onClick={() => rail.show({ key: `units:${c.key}`, title: `${compartmentName(c)} · ${c.units.length} units`, body: <UnitsDetail c={c} /> })} className="press min-w-0 flex-1 py-1.5 text-left" title={c.holds}>
                  <span className={`mono block truncate text-[11px] ${c.key === projectKey ? "text-text" : "text-muted"}`}>{compartmentName(c)}</span>
                  <span className="mono block truncate text-[10px] text-faint">
                    {c.units.length === 0 ? "no units yet" : `${acting} of ${c.units.length} acting`}
                    {c.on ? "" : " · off"}
                  </span>
                </button>
                <Switch on={c.on} busy={busyKey === c.switch_key} label={`Switch ${c.on ? "off" : "on"} every learned unit of ${compartmentName(c)}`} onChange={(on) => void flip(c, on)} />
              </li>
            );
          })}
        </ul>
      )}
      {note && (
        <p className={`mono mt-2 text-[10.5px] ${note.tone === "error" ? "text-error" : "text-verified"}`} role="status">
          {note.text}
        </p>
      )}
    </Block>
  );
}
