"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And
// that whole information database for it should be on a right side panel. The main core of the
// pages should be on actual work flow and core function visuals ... you have to think of it as if
// it's an interactive system for the main parts the right menu are libraries and or discription
// tools etc. left menu are the main interface nav like it is now. ... We need the interface to be
// amazing."): the working parts of the Memory page (what is held, the world model's levels, what
// came back for recent calls) and the rail's libraries (outcomes, projects, the files on disk, the
// routing record). Each item in the middle opens its detail in the rail.

import { Fragment, useEffect, useRef, type ReactNode } from "react";
import DeleteFlow from "@/components/DeleteFlow";
import DotText from "@/components/DotText";
import ProjectMenu from "@/components/ProjectMenu";
import { useRail } from "@/components/Workspace";
import { Empty, Masked, Pill, Placeholder } from "@/components/ui";
import { ago, bytes, projectLabel, useApi, type Control, type Ledger, type LedgerRow, type Outcomes as OutcomesReply, type Project, type Projects } from "@/lib/api";
import { combineText, headsText, levelsOf, pct as pctOf, routingByJob, slopeText, TAG_NAMES, unlevelled, type ComprehensionReply } from "@/lib/comprehension";
import { API_OUTCOMES, API_SESSION, packsOf, pickProject, placeShort, standingTone, stateBg, stateText, stateWord } from "@/lib/evidence";
import { useHash } from "@/lib/hash";
import type { LiveWriter } from "@/lib/livefeed";
import { maskPath, maskPathsIn, scrub } from "@/lib/privacy";

/* ---------- Shared by the Memory and Learning pages ---------- */

/** A block of the working system: a small heading in capitals on a hairline, a figure or control
 *  on the right, then the block. */
export function Block({ id, title, right, children, e2e, hint }: { id?: string; title: ReactNode; right?: ReactNode; children: ReactNode; e2e?: string; hint?: string }) {
  return (
    <section id={id} className="min-w-0 scroll-mt-6" data-e2e={e2e}>
      <div className="flex min-h-7 flex-wrap items-center justify-between gap-x-3 gap-y-1 border-b border-line pb-1.5">
        <h2 className="eyebrow flex items-center gap-2 text-muted" title={hint}>
          <span aria-hidden className="inline-block h-1 w-1 bg-sense-pale" />
          {title}
        </h2>
        {right}
      </div>
      <div className="pt-3">{children}</div>
    </section>
  );
}

/** The project the page reads, using the global ProjectMenu dropdown (switch, add, edit, delete). */
export function ProjectPick({ value, projects }: { value: string | null; projects: Projects | null }) {
  if (!value && !projects) return <span className="mono text-[11px] text-faint">reading projects</span>;
  return <ProjectMenu placement="down" label="project" />;
}

/** Deep links kept from the section-bar days (`/memory/#world`, `/learning/#training`): a name
 *  that is a rail tab opens that tab; a name that is a block of the middle scrolls to it. */
export function HashRoute({ tabs, blocks }: { tabs: readonly string[]; blocks: Record<string, string> }) {
  const hash = useHash().replace(/^#/, "");
  const { open } = useRail();
  // The rail's `open` changes with the tab shown; held in a ref so a tab picked by hand is not
  // pulled back to the one the address names. Only a change of the address acts.
  const openRef = useRef(open);
  useEffect(() => {
    openRef.current = open;
  });
  useEffect(() => {
    if (!hash) return;
    if (tabs.includes(hash)) openRef.current(hash);
    else if (blocks[hash]) document.getElementById(blocks[hash])?.scrollIntoView({ block: "start" });
  }, [hash, tabs, blocks]);
  return null;
}

/** A label and a value on one line, for details in the rail. */
export function Row({ k, children }: { k: ReactNode; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)] items-baseline gap-3 border-b border-line/50 py-1.5 text-[11.5px]">
      <span className="text-faint">{k}</span>
      <span className="mono min-w-0 break-words text-muted">{children}</span>
    </div>
  );
}

/** A number in the dot face when it is one, else the words in plain text. */
export function Big({ value, size = 26, tone = "text-text" }: { value: string | null; size?: number; tone?: string }) {
  if (value === null) return <span className="mono text-[11px] text-faint">reading</span>;
  if (/^[\d.,%+/-]+$/.test(value))
    return (
      <span className={tone}>
        <DotText text={value} size={size} />
      </span>
    );
  return <span className="mono text-[11px] text-faint">{value}</span>;
}

/* ---------- What is held ---------- */

export const TIERS: { key: "core" | "active" | "archive" | "temporary"; label: string; about: string }[] = [
  { key: "core", label: "Core", about: "Identity, objectives and the most used records." },
  { key: "active", label: "Active", about: "Topics in use, by subject." },
  { key: "archive", label: "Archive", about: "Older topics, never deleted." },
  { key: "temporary", label: "Temporary", about: "Three uses save a note; 24 hours unused clears it." },
];

type Item = { id: string; tier: string; topic: string; text: string; state: string; class: string; recorded: number | null };
type Items = { items?: Item[]; total?: number; error?: string };

/** The URL the memory list reads with no search, so the tier detail shares its reply. */
export const itemsUrl = (key: string | null) => (key ? `/api/memory/items?project=${encodeURIComponent(key)}&limit=2000` : null);

const tierOf = (t: string) => (/core/i.test(t) ? "core" : /activ/i.test(t) ? "active" : /arch/i.test(t) ? "archive" : /temp/i.test(t) ? "temporary" : t);

/** One tier's records, in the rail. */
function TierDetail({ tier, projectKey, now }: { tier: (typeof TIERS)[number]; projectKey: string | null; now: number }) {
  const q = useApi<Items>(itemsUrl(projectKey), 30000);
  const rail = useRail();
  const list = (q.data?.items ?? []).filter((m) => tierOf(m.tier) === tier.key);
  return (
    <div className="grid gap-3">
      <p className="text-[11.5px] leading-relaxed text-faint">{tier.about}</p>
      {q.error && !q.data ? (
        <Empty title="The memories did not answer">{q.error}</Empty>
      ) : !q.data ? (
        <span className="mono text-[11px] text-faint">reading</span>
      ) : list.length === 0 ? (
        <span className="mono text-[11px] text-faint">Nothing in this tier.</span>
      ) : (
        <ul className="grid">
          {list.slice(0, 40).map((m) => (
            <li key={m.id} className="border-b border-line/60 py-1.5">
              <span className="block break-words text-[11.5px] text-muted">{m.text}</span>
              <span className="mono text-[10px] text-faint">
                {m.topic}
                {m.class === "pinned" ? " · pinned" : ""}
                {m.recorded && now > 0 ? ` · ${ago(m.recorded, now)}` : ""}
              </span>
            </li>
          ))}
        </ul>
      )}
      <button type="button" onClick={() => rail.open("memories")} className="press mono justify-self-start border-b border-line-strong text-[10px] uppercase tracking-[0.12em] text-muted hover:text-text">
        add, pin or delete in Memories
      </button>
    </div>
  );
}

/** The four tiers of the picked project as meters; a tier opens its records in the rail. */
export function MemoryHeld({ tiers, projectKey, now, all, reading = false }: { tiers: Record<string, number> | null; projectKey: string | null; now: number; all: Record<string, number> | null; reading?: boolean }) {
  const rail = useRail();
  const max = Math.max(1, ...TIERS.map((t) => tiers?.[t.key] ?? 0));
  return (
    <ol className="grid grid-cols-2 border-t border-l border-line sm:grid-cols-4" data-e2e="memory-held">
      {TIERS.map((t) => {
        const n = tiers ? (tiers[t.key] ?? 0) : null;
        return (
          <li key={t.key} className="min-w-0 border-r border-b border-line">
            <button
              type="button"
              title={`${t.about} Open its records.`}
              onClick={() => rail.show({ key: `tier:${t.key}`, title: `${t.label} · ${n ?? "not reported"} records`, body: <TierDetail tier={t} projectKey={projectKey} now={now} /> })}
              className="press grid w-full gap-2 px-3 py-2.5 text-left hover:bg-bg-2/40"
            >
              <span className="eyebrow">{t.label}</span>
              <Big value={n === null ? (reading ? "reading" : "not reported") : String(n)} tone={n ? "text-text" : "text-faint"} />
              <span aria-hidden className="block h-px w-full bg-line">
                <span className="block h-px bg-sense-pale" style={{ width: `${((n ?? 0) / max) * 100}%` }} />
              </span>
              <span className="mono truncate text-[10px] text-faint">{all ? `${all[t.key] ?? 0} in all projects` : " "}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

/* ---------- The world model ---------- */

/** The comprehension reply for the picked project (shared by the levels and the routing tab). */
export const compUrl = (key: string | null) => (key ? `/api/comprehension?project=${encodeURIComponent(key)}` : "/api/comprehension");

function LevelDetail({ l, w }: { l: ReturnType<typeof levelsOf>[number]; w: { done: number | null; total: number | null; count: number | null } | null }) {
  return (
    <div className="grid gap-2">
      <p className="text-[11.5px] leading-relaxed text-faint">{l.about}.</p>
      <Row k="notes">{l.count}</Row>
      {l.tags.map((t) => (
        <Row key={t.tag} k={TAG_NAMES[t.tag] ?? t.tag}>
          {t.count}
        </Row>
      ))}
      <Row k="idle writer">{w ? (w.total ? `${w.done ?? 0} of ${w.total} done` : w.count !== null ? `${w.count} written` : "no progress reported") : "not reported"}</Row>
    </div>
  );
}

/** L1 to L4 with the idle writer's progress along each, the outline, and the refresh of the
 *  notes through the delete flow. */
export function MemoryLevels({ projectKey, writer, serving = null }: { projectKey: string | null; writer: LiveWriter | null; serving?: boolean | null }) {
  const q = useApi<ComprehensionReply>(compUrl(projectKey), 15000);
  const rail = useRail();
  const d = q.data;
  const levels = levelsOf(d);
  const other = d ? (d.unlevelled_notes ?? unlevelled(d.note_counts)) : 0;
  return (
    <div className="grid min-w-0 gap-4" data-e2e="comprehension">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="mono text-[10.5px] text-faint">
          {d ? `${d.total_notes} notes · ${d.base_backed_up}/${d.base_cartridges} bases backed up${other > 0 ? ` · ${other} of no level` : ""}` : q.error ? `comprehension did not answer (${q.error})` : "reading"}
        </span>
        {projectKey && (
          // 2026-11-26 (audit B3): the notes go through the delete flow; the idle writer writes
          // L1 to L4 again from the material.
          <DeleteFlow<{ note?: string; audit_id?: string }>
            path="/api/comprehension/refresh"
            body={{ project: projectKey }}
            label="refresh world model"
            title="Asks what would go first: the project's world-model notes, which the idle writer writes again from the material"
            startClass="press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.12em] text-muted hover:text-text disabled:opacity-40"
            done={(r) => r?.note ?? "World-model notes removed; the idle writer writes L1 to L4 again. Logged in Audit."}
            refresh={["/api/comprehension", "/api/memory", "/api/live"]}
            e2e="world-refresh"
          />
        )}
      </div>
      <section aria-label="Notes per level" data-e2e="comprehension-levels">
        <ol className="grid grid-cols-2 border-t border-l border-line sm:grid-cols-4">
          {levels.map((l) => {
            const w = writer?.levels[`L${l.level}` as "L1" | "L2" | "L3" | "L4"] ?? null;
            const share = w && w.total ? Math.min(1, (w.done ?? 0) / w.total) : null;
            return (
              <li key={l.level} className="min-w-0 border-r border-b border-line" data-level={l.level}>
                <button type="button" title={`${l.about}. Open its detail.`} onClick={() => rail.show({ key: `level:${l.level}`, title: `${l.label} · ${l.count} notes`, body: <LevelDetail l={l} w={w} /> })} className="press grid w-full gap-1.5 px-3 py-2.5 text-left hover:bg-bg-2/40">
                  <span className="eyebrow">{l.label}</span>
                  <Big value={d ? String(l.count) : null} tone={l.count ? "text-text" : "text-faint"} />
                  <span aria-hidden className="block h-px w-full bg-line">
                    {share !== null && <span className="block h-px bg-verified" style={{ width: `${share * 100}%` }} />}
                  </span>
                  <span className="mono line-clamp-2 min-h-[2lh] text-[10px] leading-snug text-faint">{w && w.total ? `writer ${w.done ?? 0}/${w.total}` : l.tags.map((t) => `${TAG_NAMES[t.tag] ?? t.tag} ${t.count}`).join(" · ")}</span>
                </button>
              </li>
            );
          })}
        </ol>
        <div className="mono mt-1.5 text-[10px] text-faint">{d ? (d.notes_store === null ? "no note store yet (NANO_WRITER_CACHE=on keeps notes on disk)" : d.notes_store ? `read from memory/${d.notes_store}` : "store not reported by this engine build") : " "}</div>
      </section>
      <div className="grid min-w-0 gap-x-8 gap-y-5 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <section aria-label="Outline preview" data-e2e="comprehension-outline" className="min-w-0">
          <div className="mb-2 flex items-baseline justify-between gap-3 border-b border-line/60 pb-1">
            <span className="eyebrow text-muted">L4 outline</span>
            <span className="mono text-[10px] text-faint">{d?.outline ? `${d.outline.split(/\r?\n/).length} lines` : "the project in one page"}</span>
          </div>
          {d?.outline ? (
            <pre className="mono max-h-72 overflow-auto border-l border-line pl-3 text-[11px] leading-relaxed whitespace-pre-wrap text-muted">{d.outline}</pre>
          ) : (
            <div className="grid gap-1 border-l border-line pl-3">
              <span className="mono text-[11px] text-muted">{d ? "No outline written yet for this project." : q.error ? "Comprehension did not answer." : "reading"}</span>
              {d && <span className="mono text-[10.5px] leading-relaxed text-faint">The idle writer writes it last, from the module notes: piece notes first, then file notes, module notes and the outline.</span>}
            </div>
          )}
        </section>
        <section aria-label="The idle writer" data-e2e="world-writer" className="min-w-0">
          <div className="mb-2 flex items-baseline justify-between gap-3 border-b border-line/60 pb-1">
            <span className="eyebrow text-muted">Idle writer</span>
            <span className={`mono text-[10px] ${writer?.in_session ? "text-signal" : "text-faint"}`}>{serving === false ? "no engine serving" : writer ? (writer.in_session ? "writing" : writer.held ? "held" : "idle") : "not reported"}</span>
          </div>
          {serving === false ? (
            <span className="mono text-[11px] leading-relaxed text-faint">No engine is serving, so no writer runs. The notes above are read from disk.</span>
          ) : writer ? (
            <dl className="mono grid grid-cols-[auto_minmax(0,1fr)] gap-x-6 gap-y-1 text-[11px]">
              <dt className="text-faint">queued</dt>
              <dd className="text-right text-text">{writer.queue ?? "not reported"}</dd>
              <dt className="text-faint">sessions</dt>
              <dd className="text-right text-muted">{writer.sessions ?? "not reported"}</dd>
              <dt className="text-faint">results</dt>
              <dd className="text-right text-muted">{writer.results ?? "not reported"}</dd>
              <dt className="text-faint">preempted</dt>
              <dd className="text-right text-muted">{writer.preempted ?? "not reported"}</dd>
              <dt className="text-faint">headroom</dt>
              <dd className="text-right text-muted">{writer.headroom_mb !== null ? `${Math.round(writer.headroom_mb)} MiB` : "not measured"}</dd>
              {writer.last_event && (
                <>
                  <dt className="text-faint">last event</dt>
                  <dd className="truncate text-right text-muted" title={writer.last_event}>
                    {writer.last_event}
                  </dd>
                </>
              )}
            </dl>
          ) : (
            <span className="mono text-[11px] text-faint">The live state reports no idle writer.</span>
          )}
        </section>
      </div>
    </div>
  );
}

/** The routing record and the drone heads each lens reads with: reference, in the rail. */
export function MemoryRouting({ projectKey }: { projectKey: string | null }) {
  const q = useApi<ComprehensionReply>(compUrl(projectKey), 15000);
  const d = q.data;
  const jobs = routingByJob(d?.routing);
  const heads = d?.heads ?? [];
  if (q.error && !d) return <Empty title="Comprehension did not answer">{q.error}</Empty>;
  if (!d) return <Placeholder className="h-20" what="comprehension" />;
  return (
    <div className="grid gap-6">
      <section aria-label="Routing record" data-e2e="comprehension-routing">
        <div className="eyebrow mb-1.5 flex flex-wrap items-baseline justify-between gap-2">
          <span>Routing record</span>
          <span className="mono normal-case tracking-normal text-faint">{d.routing?.present ? `measured${d.routing.made_utc ? ` ${d.routing.made_utc}` : ""}` : d.routing_record ? "measured" : "missing: fixed strengths in use"}</span>
        </div>
        {jobs.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="mono w-full text-[10.5px]">
              <thead>
                <tr className="text-left text-faint">
                  <th className="py-1 pr-2 font-normal">job</th>
                  <th className="py-1 pr-2 font-normal">model</th>
                  <th className="py-1 pr-2 text-right font-normal">acc</th>
                  <th className="py-1 pr-2 text-right font-normal">check</th>
                  <th className="py-1 pr-2 text-right font-normal">slope</th>
                  <th className="py-1 text-right font-normal">n</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <Fragment key={j.job}>
                    {j.rows.map((r, i) => (
                      <tr key={`${j.job}:${r.model ?? i}`} className="border-t border-line/60 text-muted">
                        <td className="py-1 pr-2 text-text">{i === 0 ? j.job : ""}</td>
                        <td className="max-w-[7rem] truncate py-1 pr-2" title={r.stack ?? r.model ?? undefined}>
                          {r.model ?? "?"}
                        </td>
                        <td className="py-1 pr-2 text-right">{pctOf(r.accuracy)}</td>
                        <td className="py-1 pr-2 text-right">{pctOf(r.check_accuracy)}</td>
                        <td className="py-1 pr-2 text-right">{slopeText(r.slope)}</td>
                        <td className="py-1 text-right">{r.n ?? "-"}</td>
                      </tr>
                    ))}
                    {j.combine && (
                      <tr className="text-faint">
                        <td />
                        <td colSpan={5} className="pb-1.5">
                          {combineText(j.combine)}
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-[11.5px] leading-relaxed text-faint">{d.routing ? "No measured rows: the strengths table uses its fixed values until nano-lab stage-bench routing-record writes designs/routing-record.json." : "Rows not reported by this engine build."}</p>
        )}
        {(d.routing?.lenses?.length ?? 0) > 0 && (
          <div className="mono mt-1.5 break-words text-[10px] text-faint">
            lenses {d.routing?.lenses?.join(" · ")}
            {(d.routing?.base_cartridges?.length ?? 0) > 0 ? ` · valid for ${d.routing?.base_cartridges?.join(", ")}` : ""}
          </div>
        )}
      </section>
      <section aria-label="Heads per lens" data-e2e="comprehension-heads">
        <div className="eyebrow mb-1.5 flex flex-wrap items-baseline justify-between gap-2">
          <span>Drone heads per lens</span>
          <span className="mono normal-case tracking-normal text-faint">
            {d.drone_default_heads} default · {d.drone_lens_sets} own sets
          </span>
        </div>
        {heads.length > 0 ? (
          <ul className="grid gap-2">
            {heads.map((h) => (
              <li key={`${h.population}:${h.lens}`} className="grid gap-0.5 text-[10.5px]">
                <span className="mono truncate text-text" title={h.population}>
                  {h.lens} <span className="text-faint">{h.count}</span>
                </span>
                <span className="mono break-words text-faint">{headsText(h.heads)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[11.5px] text-faint">{d.heads ? "No drone reader sets in designs/drones.json." : "Head lists not reported by this engine build."}</p>
        )}
        <div className="mono mt-1.5 text-[10px] text-faint">Heads as layer.head. A lens with no own set reads with the default set.</div>
      </section>
    </div>
  );
}

/* ---------- What came back for recent calls ---------- */

type Served = LedgerRow & { settings?: { memory?: string; dense_cache?: string; mode?: string; path?: string; cartridges?: string[]; via?: string }; steps?: string[]; episode?: string; withheld?: boolean };

/** The ledger the Memory and Learning pages share (one poller per project). */
export const ledgerUrl = (key: string | null, level: number | null) => (level !== null && level >= API_SESSION && key ? `/api/ledger?project=${encodeURIComponent(key)}&limit=300` : null);

function RecallDetail({ r, evidence, root, now }: { r: Served; evidence: LedgerRow[]; root: string | null; now: number }) {
  const s = r.settings ?? {};
  return (
    <div className="grid gap-4">
      <div>
        <Row k="when">{r.at && now > 0 ? ago(r.at, now) : "not reported"}</Row>
        <Row k="tool">{r.tool ?? "not reported"}</Row>
        <Row k="joined by">{r.joined_by ?? "not reported"}</Row>
        <Row k="memory then">{s.memory ?? "not reported"}</Row>
        <Row k="meaning index">{s.dense_cache ?? "not reported"}</Row>
        <Row k="mode · path">{[s.mode, s.path].filter(Boolean).join(" · ") || "not reported"}</Row>
        <Row k="cartridges">{s.cartridges?.length ? s.cartridges.join(", ") : "none"}</Row>
        <Row k="kept as episode">{r.episode ? `${r.episode.slice(0, 12)}…` : "no"}</Row>
        {r.withheld && <Row k="withheld">yes</Row>}
      </div>
      <div>
        <div className="eyebrow mb-1">Places brought back · {r.cited?.length ?? 0}</div>
        <ul className="grid gap-0.5">
          {(r.cited ?? []).map((p, i) => (
            <li key={i} className="mono truncate text-[10.5px] text-muted">
              <Masked id={`recall:${r.pack}:${i}`} shown={placeShort(p)} full={maskPath(p, root)} />
            </li>
          ))}
        </ul>
      </div>
      <div>
        <div className="eyebrow mb-1">What followed · {evidence.length}</div>
        {evidence.length === 0 ? (
          <span className="mono text-[10.5px] text-faint">not judged yet</span>
        ) : (
          <ul className="grid gap-0.5">
            {evidence.map((e, i) => (
              <li key={i} className="mono flex items-center gap-2 text-[10.5px] text-muted">
                <i aria-hidden className={`inline-block h-1.5 w-1.5 ${stateBg(e.state)}`} />
                {e.rule} · {stateWord(e.state)}
                {e.strength ? ` ${e.strength}` : ""}
              </li>
            ))}
          </ul>
        )}
      </div>
      {(r.steps?.length ?? 0) > 0 && (
        <div>
          <div className="eyebrow mb-1">Path through the parts</div>
          <div className="mono break-words text-[10px] leading-relaxed text-faint">{r.steps!.join(" → ")}</div>
        </div>
      )}
    </div>
  );
}

/** The newest context packs the ledger recorded for the picked project: when, how many places
 *  came back, whether memory was on, and what followed. One opens its detail in the rail. */
export function MemoryRecall({ projectKey, level, root, now }: { projectKey: string | null; level: number | null; root: string | null; now: number }) {
  const q = useApi<Ledger>(ledgerUrl(projectKey, level), 10000);
  const rail = useRail();
  const rows = q.data?.rows ?? [];
  const served = rows.filter((r) => r.row === "served").sort((a, b) => (b.at ?? 0) - (a.at ?? 0)) as Served[];
  const packs = new Map(packsOf(rows).map((p) => [p.pack, p]));
  if (level !== null && level < API_SESSION) return <Empty title="Not reported by this engine build" slot>The call record comes with API {API_SESSION}; this build answers API {level}.</Empty>;
  if (!projectKey) return <span className="mono text-[11px] text-faint">no project picked</span>;
  if (q.error && !q.data) return <Empty title="The call record did not answer">{q.error}</Empty>;
  if (!q.data) return <span className="mono text-[11px] text-faint">reading the call record</span>;
  if (served.length === 0) return <span className="mono text-[11px] text-faint">No call recorded in this project yet.</span>;
  return (
    <ol className="grid" data-e2e="memory-recall">
      {served.slice(0, 8).map((r, i) => {
        const p = r.pack ? packs.get(r.pack) : undefined;
        const st = p && p.evidence.length ? (p.states.has("verified") ? "verified" : p.states.has("refuted") ? "refuted" : p.states.has("used") ? "used" : [...p.states][0]) : null;
        const cited = r.cited ?? [];
        return (
          <li key={`${r.pack ?? i}`}>
            <button
              type="button"
              onClick={() => rail.show({ key: `recall:${r.pack}`, title: `${r.tool ?? "call"} · ${cited.length} places`, body: <RecallDetail r={r} evidence={p?.evidence ?? []} root={root} now={now} /> })}
              className="press grid w-full grid-cols-[auto_minmax(0,1fr)_auto] items-baseline gap-x-3 border-b border-line/60 py-2 text-left hover:bg-bg-2/40"
            >
              <i aria-hidden className={`inline-block h-1.5 w-1.5 translate-y-[-1px] ${st ? stateBg(st) : "bg-sense-pale"}`} />
              <span className="min-w-0">
                <span className="mono block truncate text-[11.5px] text-muted">
                  {r.tool ?? "call"} · {cited.length} {cited.length === 1 ? "place" : "places"} · memory {r.settings?.memory ?? "?"}
                  {r.episode ? " · kept" : ""}
                </span>
                <span className="mono block truncate text-[10px] text-faint">
                  {cited.slice(0, 4).map(placeShort).join("  ")}
                  {cited.length > 4 ? `  +${cited.length - 4}` : ""}
                </span>
              </span>
              <span className="text-right">
                <span className={`mono block text-[10px] uppercase tracking-[0.1em] ${st ? stateText(st) : "text-faint"}`}>{st ? stateWord(st) : "open"}</span>
                <span className="mono block text-[10px] text-faint">{r.at && now > 0 ? ago(r.at, now) : ""}</span>
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

/* ---------- Rail: outcomes, projects, files ---------- */

/** Verified and refuted outcomes kept in the picked project (API 6), each place read fresh. */
export function MemoryOutcomes({ projectKey, level, root }: { projectKey: string | null; level: number | null; root?: string | null }) {
  const has = level !== null && level >= API_OUTCOMES;
  const q = useApi<OutcomesReply>(has && projectKey ? `/api/outcomes?project=${encodeURIComponent(projectKey)}` : null, 15000);
  const list = q.data?.outcomes ?? [];
  const v = list.filter((o) => o.standing === "verified").length;
  const r = list.filter((o) => o.standing === "refuted").length;
  return (
    <div className="grid gap-2">
      <div className="eyebrow flex items-baseline justify-between gap-2">
        <span>Outcomes kept</span>
        {q.data && (
          <span className="mono normal-case tracking-normal text-faint">
            {v} verified · {r} refuted
          </span>
        )}
      </div>
      {level !== null && !has ? (
        <span className="mono text-[11px] text-faint">Not reported by this engine build (API {level}; outcomes come with API {API_OUTCOMES}).</span>
      ) : q.error && !q.data ? (
        <Empty title="Outcomes did not answer">{q.error}</Empty>
      ) : !q.data ? (
        <span className="mono text-[11px] text-faint">reading</span>
      ) : list.length === 0 ? (
        <span className="mono text-[11px] text-faint" title="A context pack whose cited lines were then read or edited, and verified or refuted by what followed, becomes an outcome here. So does a change that fixed a located failure.">
          No outcome kept yet.
        </span>
      ) : (
        <>
          {q.data.root_known === false && <div className="mono text-[10px] text-quiet">The project&apos;s folder is not known yet, so freshness could not be read.</div>}
          <ul>
            {list.slice(0, 16).map((o) => {
              const f = o.failure;
              const what = o.kind === "fixed" ? `fixed · ${[f?.check, f?.code].filter(Boolean).join(" ") || "a failure"}` : "answered a question";
              return (
                <li key={o.record} className="grid gap-0.5 border-b border-line/60 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="mono min-w-0 truncate text-[11px] text-muted">{what}</span>
                    <Pill tone={standingTone(o.standing)}>{o.standing ?? "undecided"}</Pill>
                  </div>
                  {o.kind === "answered" && o.question && (
                    <span className="mono min-w-0 truncate text-[10.5px] text-faint">
                      <Masked id={`outcome:${o.record}`} shown="question hidden" full={scrub(o.question)} />
                    </span>
                  )}
                  {o.kind === "fixed" && f?.msg && (
                    <span className="mono min-w-0 truncate text-[10.5px] text-faint">
                      <Masked id={`outcome:${o.record}`} shown="message hidden" full={scrub(f.msg)} />
                    </span>
                  )}
                  <div className="mono flex min-w-0 flex-wrap gap-x-3 text-[10px] text-faint">
                    {(o.places ?? []).slice(0, 4).map((p, i) => (
                      <span key={i} className={p.fresh ? "" : "text-quiet line-through decoration-dotted"} title={p.fresh ? "lines unchanged since verified" : "lines changed since, or not read"}>
                        <Masked id={`place:${o.record}:${i}`} shown={placeShort(p.place)} full={maskPath(p.place, root)} />
                      </span>
                    ))}
                    <span>
                      {o.verified ?? 0} verified · {o.refuted ?? 0} refuted{o.strength ? ` · ${o.strength}` : ""}
                    </span>
                    {!o.in_memory && <span className="text-quiet">waiting in the ledger for the journal</span>}
                  </div>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}

/** One folder per project; picking one makes the whole page read it. */
export function MemoryProjects({ list, current, projects, error, statusRoot, now }: { list: Project[]; current: string | null; projects: Projects | null; error: string | null; statusRoot?: string | null; now: number }) {
  return (
    <div className="grid gap-2">
      <div className="eyebrow flex items-baseline justify-between gap-2">
        <span>Projects</span>
        {projects && <span className="mono normal-case tracking-normal text-faint">{list.length}</span>}
      </div>
      {!projects && !error && <span className="mono text-[11px] text-faint">reading</span>}
      {error && !projects && <Empty title="Projects did not answer">{error}</Empty>}
      {projects && list.length === 0 && <span className="mono text-[11px] text-faint">No project memory yet: a folder is made the first time a project is served with memory on.</span>}
      {list.length > 0 && (
        <ul>
          {list.map((p) => (
            <li
              key={p.key}
              role="button"
              tabIndex={0}
              aria-pressed={p.key === current}
              onClick={() => pickProject(p.key)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  pickProject(p.key);
                }
              }}
              className="press relative grid cursor-pointer gap-0.5 border-b border-line/60 py-2 pl-3 hover:bg-bg-2/40"
            >
              {p.key === current && <span aria-hidden className="absolute top-2 bottom-2 left-0 w-[2px] bg-sense-pale" />}
              <span className="mono truncate text-[12px] text-text" title={p.key}>
                {projectLabel(p)}
              </span>
              <span className="mono truncate text-[10px] text-faint">
                <Masked id={`root:${p.key}`} shown={maskPath(p.root ?? p.dir, statusRoot)} full={p.root ?? p.dir} />
              </span>
              <span className="mono text-[10px] text-faint">
                {p.episodes} {p.episodes === 1 ? "episode" : "episodes"} · {bytes(p.bytes)}
                {p.last_ms && now > 0 ? ` · last served ${ago(p.last_ms, now)}` : ""}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** The picked project's files: one writer per file, the meaning index kept on disk (with its
 *  clear), and the Nano's own memory index. */
export function MemoryFiles({ project, projectKey, level, denseCtl, memoryOn, index, statusRoot, memoryDir }: { project: Project | null; projectKey: string | null; level: number | null; denseCtl: Control | null; memoryOn: boolean | null; index: { data: unknown; error: string | null; loading: boolean }; statusRoot?: string | null; memoryDir?: string | null }) {
  const overlap = project?.tiers_overlap;
  const mig = project?.migration ?? project?.migrated;
  const dense = project?.dense;
  return (
    <div className="grid gap-6">
      <section className="grid gap-1">
        <div className="eyebrow mb-1">One writer per file</div>
        {!project ? (
          <span className="mono text-[11px] text-faint">{projectKey === "nano" ? "The Nano's own folder: its records per tier are in the strata figure." : "No project picked."}</span>
        ) : (
          <>
            <Row k="layout">
              <span className={overlap ? "text-attention" : ""}>{overlap === undefined ? "not reported" : overlap ? `${overlap} records in two tiers` : "every record in one tier"}</span>
            </Row>
            {overlap ? <p className="text-[11px] leading-relaxed text-faint">Written by an older version; brought over to one writer per file the next time a server opens the folder, with a copy kept.</p> : null}
            {mig !== undefined && mig !== null && <Row k="migration">{typeof mig === "string" || typeof mig === "number" ? String(mig) : "reported"}</Row>}
            {project.tiers && <Row k="per tier">{Object.entries(project.tiers).map(([t, n]) => `${t} ${n}`).join(" · ")}</Row>}
          </>
        )}
      </section>
      <section className="grid gap-1">
        <div className="eyebrow mb-1" title="One content hash and one vector per piece of the material, never its text or file names, so a restart does not embed everything again.">
          Meaning index on disk
        </div>
        <Row k="kept">{denseCtl ? (denseCtl.value === "on" && memoryOn !== false ? "on, in the project's memory folder" : memoryOn === false ? "off, memory is off" : "off") : "not reported"}</Row>
        <Row k="size">{dense && typeof dense.bytes === "number" ? `${bytes(dense.bytes)}${typeof dense.rows === "number" ? ` · ${dense.rows} pieces` : ""}` : `not reported by this engine build (API ${level ?? "?"})`}</Row>
        {projectKey && (
          <div className="mt-2">
            <DeleteFlow<{ rows?: number; bytes?: number }>
              path="/api/dense/clear"
              body={{ project: projectKey }}
              label="Clear meaning index"
              title="Removes this project's stored vectors; the next start embeds the material again. Asks what would go first."
              done={(r) => `Meaning index cleared: ${r?.rows ?? 0} rows, ${bytes(r?.bytes ?? 0)}. The next start embeds the material again. Logged in Audit.`}
              refresh={["/api/projects"]}
            />
          </div>
        )}
      </section>
      <section>
        <div className="eyebrow mb-1">Index of the Nano&apos;s own memory</div>
        {index.data ? (
          <pre className="mono max-h-[50vh] overflow-auto border-l border-line pl-3 text-[10.5px] leading-relaxed text-muted">{maskPathsIn(JSON.stringify(index.data, null, 2), statusRoot)}</pre>
        ) : index.loading ? (
          <span className="mono text-[11px] text-faint">reading</span>
        ) : index.error ? (
          <Empty title="The memory index did not answer">{index.error}</Empty>
        ) : (
          <span className="mono text-[11px] text-faint">Nothing remembered yet. Folder: {maskPath(memoryDir, statusRoot) || "memory/"}</span>
        )}
      </section>
    </div>
  );
}
