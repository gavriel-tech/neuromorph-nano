"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "Dude. The discriptors should be mostly hidden ...
// And that whole information database for it should be on a right side panel. The main core of the
// pages should be on actual work flow and core function visuals ... you have to think of it as if
// it's an interactive system for the main parts the right menu are libraries and or discription
// tools etc. left menu are the main interface nav like it is now. ... We need the interface to be
// amazing."): the middle is the continual learning loop working (the switches, what came in, the
// gate, what is training or waiting, what was kept, the learned units in brief); the evidence
// ledger, the primes, the training data and the cartridges are rail tabs; the description is the
// About tab. The learned units are managed on Cartridges; here they are a compact summary.
// 2026-11-27 (his words: "The main center screen for every page should be easy to navigate and
// beautifully visual and interactive" ... "We should be able to visualize the system in a high
// level professional and technical way"): the middle has four tabs, kept in this browser: the
// loop (the figure of evidence gathering into a cartridge as the focal view, the switches beside
// it, then the six steps from verified work through the gate to units acting), distillation, the
// learned units, and the stated preferences. The project is picked on the tab bar.

import Link from "next/link";
import { useCallback, useMemo, useState } from "react";
import { ControlRow, NoteLine } from "@/components/Controls";
import { DistillationSection } from "@/components/DistilBuilder";
import { LearningCartridges, LearningEngine, LearningEvidence, LearningStore, LearningUnits } from "@/components/LearningPanels";
import type { LearnedUnitsReply } from "@/components/LearnedUnits";
import MainTabs from "@/components/MainTabs";
import { Block, HashRoute, ledgerUrl, ProjectPick } from "@/components/MemoryPanels";
import PlateCanvas from "@/components/PlateCanvas";
import PreferenceBuilder, { PreferenceNotes } from "@/components/PreferenceBuilder";
import { PrimesRail } from "@/components/PrimesStudio";
import { TrainingDataPanel } from "@/components/TrainingData";
import { useTopology } from "@/components/Topology";
import Workspace, { RailSection } from "@/components/Workspace";
import { Pill } from "@/components/ui";
import { useApi, useApiLevel, type Audit, type AuditRow, type Cartridge, type ContinualLearning, type Control, type Examples, type Ledger } from "@/lib/api";
import { API_EXAMPLES, API_SESSION, familyOf, useProjectKey } from "@/lib/evidence";
import { learningScene } from "@/lib/pagescenes";
import { gateLine, jobFiltersOf, learnedCounts, learnedLine, lensesOf, type IdleCycle } from "@/lib/learned";
import type { ComprehensionReply } from "@/lib/comprehension";
import { stageGroup } from "@/lib/livefeed";
import { useLive } from "@/lib/livehook";
import type { PaintOpts } from "@/lib/paint";

const PRINCIPLES = [
  ["Mistakes as mistakes", "What failed, where and why, never stored as facts."],
  ["Evidence on every piece", "Its source, its evidence and how confident it is."],
  ["Verified counts more", "A passed test, a successful build or an accepted change weighs more than anything unconfirmed."],
  ["Fades when it stops holding", "Contradicted or unused pieces lose weight and move to the archive."],
  ["Preferences apart from work", "Learning from work and learning your preferences are separate kinds of cartridge, never mixed."],
  ["Everything can be undone", "Any piece or cartridge can be switched off, rolled back or removed, alone or by source."],
];

/** The rail's tabs; the old section names open the matching place (`/learning/#training`). */
const TABS = ["evidence", "primes", "training", "cartridges", "preferences", "distil-notes"] as const;

/** The side panel's Distillation tab: what the builder in the middle fills, from what, and what
 *  the held-out gate asks before a cartridge is written. Reference; the builder shows the engine's
 *  own reason when a pair of kind and technique is not available. */
function DistilNotes() {
  const kinds: [string, string, string][] = [
    ["adapter cartridge", "lens · teacher-loop · unit · steer", "a learned unit on the MicroModel's deck; the Embedder's is a request lens, the Reader's may be a learning unit"],
    ["filter cartridge", "channel · teacher-loop", "a job filter; Writer, Reader and Reranker, not the Embedder (a filter would change the stored vectors)"],
    ["preference cartridge", "readout", "preference values for the whole engine, from judged pairs (accept or reject)"],
    ["read-out cartridge", "readout · teacher-loop", "a judge read-out on the Reranker, from teacher scores"],
  ];
  return (
    <div className="grid gap-0" data-e2e="distil-notes">
      <RailSection title="Distillation">
        <p className="text-[12px] leading-relaxed text-muted">A learning function: a teacher&apos;s outputs fill learned parameters into a learned cartridge of the kind you choose, written in that kind&apos;s own format, so it plugs in as cartridges of that kind do. Nothing leaves the machine.</p>
      </RailSection>
      <RailSection title="Kinds and techniques">
        <dl className="grid gap-2.5 text-[11px]">
          {kinds.map(([k, t, about]) => (
            <div key={k} className="grid gap-0.5">
              <dt className="flex flex-wrap items-baseline justify-between gap-x-3">
                <span className="text-text">{k}</span>
                <span className="mono text-[10.5px] text-muted">{t}</span>
              </dt>
              <dd className="leading-relaxed text-faint">{about}</dd>
            </div>
          ))}
        </dl>
      </RailSection>
      <RailSection title="Teacher data">
        <p className="text-[11.5px] leading-relaxed text-faint">A JSON lines file of teacher pairs on this machine (prompt, completion, teacher, licence), or a live teacher: your agent over MCP, or an installed open-weights model on the graphics card. Each pair&apos;s split is fixed by its prompt (80 fit, 10 tune, 10 held out). Only pairs labelled used or superseded are trained on. Audit undoes an import or a label.</p>
      </RailSection>
      <RailSection title="The held-out gate">
        <ul className="grid list-none gap-1 text-[11.5px] leading-relaxed text-faint">
          <li>the teacher-token loss falls and top-1 agreement rises on the held-out split</li>
          <li>a shuffled-target control gains at most a quarter as much</li>
          <li>drift on neutral text stays within its limit</li>
          <li>three repeated trials, each without a third of the fit pairs, all gain</li>
          <li>otherwise the build reports REFUSED with its figures and writes nothing</li>
        </ul>
      </RailSection>
      <RailSection title="After a build">
        <p className="text-[11.5px] leading-relaxed text-faint">A new adapter starts unplugged on its MicroModel&apos;s deck, a new job filter switched off on its lane, a new preference cartridge unplugged on the whole-engine rack. One build runs at a time and shows in Live Activity; each writes a report under runs/ and an audit row.</p>
      </RailSection>
      <RailSection title="Expected effect">
        <p className="text-[11.5px] leading-relaxed text-faint">The teacher&apos;s style, format and wording in the MicroModel&apos;s output, not new knowledge or reasoning. For the Reranker, its yes or no score moves toward the teacher&apos;s scores.</p>
      </RailSection>
    </div>
  );
}
const BLOCKS: Record<string, string> = {};

type Surgery = { job_filter_build?: { status?: string; job?: string; role?: string; message?: string; started_ms?: number }; roles?: Record<string, { job_filters?: { cartridge: string; job: string; enabled?: boolean; base_cos?: number; filtered_cos?: number; layers?: number }[] }> };

export default function Learning() {
  const controls = useApi<Control[]>("/api/controls");
  const carts = useApi<Cartridge[]>("/api/cartridges", 5000);
  const [note, setNote] = useState<{ text: string; tone: "verified" | "error" } | null>(null);
  const ctl = controls.data?.find((c) => c.key === "learning");
  const cycleCtl = controls.data?.find((c) => c.key === "learning.cycle");
  const examplesCtl = controls.data?.find((c) => c.key === "examples");
  const audit = useApi<Audit>("/api/audit?limit=300", 5000);
  const { tree, status, error: treeError } = useTopology();
  const level = useApiLevel();
  const { key, projects } = useProjectKey();
  const hasLedger = level !== null && level >= API_SESSION;
  const ledger = useApi<Ledger>(ledgerUrl(key, level), 10000);
  const continual = useApi<ContinualLearning>(key ? `/api/learning?project=${encodeURIComponent(key)}` : null, 10000);
  const surgery = useApi<Surgery>("/api/surgery", 15000);
  const units = useApi<LearnedUnitsReply>("/api/learned-units", 10000);
  const jobFilters = useMemo(() => jobFiltersOf(surgery.data), [surgery.data]);
  const comprehension = useApi<ComprehensionReply>(key ? `/api/comprehension?project=${encodeURIComponent(key)}` : "/api/comprehension", 15000);
  const live = useLive();
  // The learned pieces, counted from the engine's own records (never a fixed zero).
  const learned = useMemo(
    () =>
      learnedCounts({
        carts: carts.data,
        filters: jobFilters,
        lenses: lensesOf(comprehension.data?.routing ?? null, live.reply?.models ?? null),
        idle: (continual.data as unknown as { idle_cycle?: IdleCycle } | null)?.idle_cycle ?? null,
      }),
    [carts.data, jobFilters, comprehension.data, live.reply, continual.data],
  );
  const hasExamples = level !== null && level >= API_EXAMPLES;
  const store = useApi<Examples>(hasExamples && key ? `/api/examples?project=${encodeURIComponent(key)}&limit=8` : null, 15000);
  const ledgerPending = ledger.loading || level === null || (hasLedger && !key);
  const storePending = store.loading || level === null || (hasExamples && !key);
  const rows = useMemo(() => (audit.data?.audit ?? []) as AuditRow[], [audit.data]);
  const paint: PaintOpts = useMemo(() => ({ status, carts: carts.data ?? [], drones: new Map(), hide: new Set<string>() }), [status, carts.data]);
  const learnNode = tree?.byPath.get("learning") ?? tree?.root;
  const build = useCallback((W: number, H: number) => (learnNode ? learningScene(learnNode, rows, carts.data ?? [], W, H, learned) : null), [learnNode, rows, carts.data, learned]);
  const learnedCarts = (carts.data ?? []).filter((c) => familyOf(c) !== "fundamentals");
  const fundCarts = (carts.data ?? []).filter((c) => familyOf(c) === "fundamentals");

  // What is training now: the engine's live stages of the learning and filter groups, and the job
  // filter build the surgery route reports.
  const running = (live.reply?.active ?? []).filter((a) => {
    const g = stageGroup(a.stage);
    return g === "learning" || g === "filter";
  });
  const jfb = surgery.data?.job_filter_build;
  const training = (
    <ul className="grid gap-1" data-e2e="learning-training">
      {running.map((a, i) => (
        <li key={`${a.stage}-${i}`} className="mono flex items-center gap-2 text-[11px] text-signal">
          <i aria-hidden className="inline-block h-1.5 w-1.5 bg-signal" />
          {a.stage}
          {a.model ? ` · ${a.model}` : ""}
        </li>
      ))}
      <li className={`mono flex items-center gap-2 text-[11px] ${jfb?.status && jfb.status !== "idle" ? "text-signal" : "text-muted"}`}>
        <i aria-hidden className={`inline-block h-1.5 w-1.5 ${jfb?.status && jfb.status !== "idle" ? "bg-signal" : "bg-line-3"}`} />
        job filter build: {surgery.data ? (jfb?.status ?? "not reported") : surgery.error ? "did not answer" : "reading"}
        {jfb?.status && jfb.status !== "idle" && jfb.job ? ` · ${jfb.role}/${jfb.job}` : ""}
      </li>
      {running.length === 0 && (
        <li className="mono flex items-center gap-2 text-[11px] text-muted">
          <i aria-hidden className="inline-block h-1.5 w-1.5 bg-line-3" />
          learning stages: {live.reply ? "none running" : live.error ? "live state did not answer" : "reading"}
        </li>
      )}
    </ul>
  );

  const status_ = (
    <>
      {ctl ? (
        <Pill tone={ctl.value === "off" ? "muted" : "verified"}>
          learning {ctl.value}
          {typeof continual.data?.version === "number" ? ` · v${continual.data.version}` : ""}
        </Pill>
      ) : (
        <Pill tone="muted">{controls.error ? "learning not reported" : "reading"}</Pill>
      )}
      {continual.data && <span>{continual.data.cycles_run} cycles</span>}
      {units.data && (
        <span>
          {units.data.acting}/{units.data.total} units acting
        </span>
      )}
      <Link href="/cartridges/#flow" className="link normal-case tracking-normal" data-e2e="flow-link">
        components
      </Link>
    </>
  );

  const L = ledger.data;
  const about = (
    <div className="grid gap-0">
      <RailSection title="What this page is">
        <p className="text-[12px] leading-relaxed text-muted">
          Continuous learning: the Nano learns only what you want learned, from verified work with you and your coding agents (Claude Code, Cursor, VS Code, Antigravity, Windsurf), and from your preferences, into neuroplasticity cartridges. Its evidence comes first: the ledger records what followed each context pack and each prime. Learning is a separate system from memory.
        </p>
      </RailSection>
      <RailSection title="The loop">
        <p className="text-[12px] leading-relaxed text-muted">
          Each judged outcome enters as an experience. A learning cycle turns experiences into patterns, procedures and scoped policies, and keeps them only when they predict held-out outcomes better than what the engine knew before. What is kept acts as learned units, plugged in or out by project on Cartridges.
        </p>
        <p className="mono mt-2 text-[10.5px] text-faint">
          {learnedLine(learned)} · {gateLine(learned)}
        </p>
      </RailSection>
      <RailSection title="Learned units">
        <p className="text-[12px] leading-relaxed text-muted">Everything learned is kept beside the frozen models and never merged into them, in compartments: one per project, one shared, one of the Nano&apos;s own. A compartment switched off gives the base&apos;s outputs again, bit for bit. A compartment in the middle opens its units here.</p>
      </RailSection>
      <RailSection title="Priming &amp; agent nudging">
        <p className="text-[12px] leading-relaxed text-muted">
          Primes proactively nudge your coding agent in the right direction through lifecycle hooks (past verified answers, recurring failure fixes, edit-impact warnings, compaction resume, core memories, and unverified open loops), alongside user- or agent-set completion loops, step-graph directions, and reminders on Memory -&gt; Loops and graph. Every prime is verified fresh on disk and scored by the Evidence Ledger.
        </p>
      </RailSection>
      <RailSection title="The figure">
        <p className="text-[11.5px] leading-relaxed text-faint">One cluster per recorded call, gathering toward the cartridge slot at the right. The slot fills only when the engine reports a learned piece; nothing in it is drawn before it is measured.</p>
      </RailSection>
      <RailSection title="How the Nano learns · six principles">
        <ol className="grid gap-3">
          {PRINCIPLES.map(([t, d], i) => (
            <li key={t} className="grid grid-cols-[1.5rem_minmax(0,1fr)] gap-2">
              <span className="mono text-[10.5px] text-faint">0{i + 1}</span>
              <span>
                <span className="block text-[11.5px] uppercase tracking-[0.06em] text-text">{t}</span>
                <span className="block text-[11.5px] leading-relaxed text-muted">{d}</span>
              </span>
            </li>
          ))}
        </ol>
      </RailSection>
    </div>
  );

  return (
    <Workspace
      title="Learning"
      status={status_}
      about={about}
      defaultTab="evidence"
      rail={[
        { id: "evidence", label: "Evidence", count: L ? (L.rows?.length ?? 0) : undefined, body: <LearningEvidence ledger={{ data: L, error: ledger.error }} level={level} projectKey={key} pending={ledgerPending} /> },
        { id: "primes", label: "Primes", count: L?.primes?.given ?? L?.nudges?.given, body: <PrimesRail /> },
        {
          id: "training",
          label: "Training data",
          body: (
            <div className="grid gap-6">
              <LearningStore key={key ?? "none"} store={store.data} error={store.error} level={level} control={examplesCtl ?? null} projectKey={key} />
              <TrainingDataPanel projectKey={key} />
            </div>
          ),
        },
        { id: "cartridges", label: "Cartridges", count: carts.data ? learnedCarts.length : undefined, body: <LearningCartridges learned={learnedCarts} fund={fundCarts} loaded={!!carts.data} /> },
        { id: "preferences", label: "Preferences", body: <PreferenceNotes /> },
        { id: "distil-notes", label: "Distillation", body: <DistilNotes /> },
      ]}
    >
      <HashRoute tabs={TABS} blocks={BLOCKS} />
      <MainTabs
        storageKey="nm-main:learning"
        label="Learning views"
        right={<ProjectPick value={key} projects={projects} />}
        tabs={[
          {
            id: "loop",
            label: "Loop",
            aliases: ["overview", "engine"],
            count: continual.data ? `v${continual.data.version}` : undefined,
            title: "Evidence, the gate, what was kept and the units acting",
            body: (
              <>
                <div className="grid min-w-0 gap-x-6 gap-y-5 lg:grid-cols-[minmax(0,1.7fr)_minmax(15rem,1fr)]">
                  <section aria-label="Evidence gathering into a cartridge" className="min-w-0">
                    <PlateCanvas build={build} paint={paint} status={status} tree={tree} what="learning figure" failed={tree ? null : treeError} label={`Recorded calls flowing toward the cartridge slot; ${learnedLine(learned)}; ${gateLine(learned)}`} className="h-[clamp(230px,32vh,340px)] border border-line" />
                    <div className="mono mt-1.5 text-[10px] uppercase tracking-[0.16em] text-faint">Fig. 1 · recorded calls gathering toward the cartridge slot</div>
                    <div className="mono mt-1 text-[10.5px] leading-relaxed text-faint" title="One cluster per recorded call; the slot fills only when the engine reports a learned piece." data-e2e="learned-pieces">
                      {learnedLine(learned)} · {gateLine(learned)}
                    </div>
                  </section>
                  <Block title="Learning switches">
                    <NoteLine note={note} />
                    {ctl || cycleCtl ? (
                      <div className="grid grid-cols-1 border-t border-l border-line">
                        {[ctl, cycleCtl].filter((c): c is Control => !!c).map((c) => (
                          <div key={c.key} className="min-w-0 border-r border-b border-line px-3">
                            <ControlRow c={c} foldHelp onNote={setNote} />
                          </div>
                        ))}
                      </div>
                    ) : (
                      <span className="mono text-[11px] text-faint">{controls.error ? `controls did not answer (${controls.error})` : "reading"}</span>
                    )}
                  </Block>
                </div>

                <LearningEngine
                  key={`cl-${key ?? "none"}`}
                  data={continual.data}
                  error={continual.error}
                  projectKey={key}
                  ledger={L}
                  ledgerPending={ledgerPending}
                  store={store.data}
                  storePending={storePending}
                  units={units.data}
                  learned={learned}
                  jobFilters={jobFilters}
                  training={training}
                />
              </>
            ),
          },
          {
            id: "distillation",
            label: "Distillation",
            aliases: ["distil", "teacher"],
            title: "Teacher data, a live teacher and a build for every learned cartridge kind",
            body: <DistillationSection />,
          },
          {
            id: "units",
            label: "Learned units",
            count: units.data ? `${units.data.acting}/${units.data.total}` : undefined,
            title: "Every learned unit by compartment, switched on or off",
            body: <LearningUnits projectKey={key} />,
          },
          {
            id: "stated",
            label: "Preferences",
            aliases: ["preferences", "wants"],
            title: "What you want and what to avoid, as rule cartridges, with the draft check",
            body: (
              <Block id="preferences" title="Preferences · want and avoid" e2e="learning-preferences" hint="Your stated preferences as rule cartridges; the side panel's Preferences tab says how each rule is held">
                <PreferenceBuilder />
              </Block>
            ),
          },
        ]}
      />
    </Workspace>
  );
}
