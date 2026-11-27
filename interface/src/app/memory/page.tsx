"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "Dude. The discriptors should be mostly hidden ...
// And that whole information database for it should be on a right side panel. The main core of the
// pages should be on actual work flow and core function visuals ... you have to think of it as if
// it's an interactive system for the main parts the right menu are libraries and or discription
// tools etc. left menu are the main interface nav like it is now. ... We need the interface to be
// amazing."): the middle is the memory system working (the switch, what is held per tier, the
// strata, the world model's four levels and outline, what came back for recent calls); the
// memories, the goals, the outcomes and projects, the routing record and the files on disk are rail
// tabs; the description is the About tab.
// 2026-11-26 (his words: "loops can be in memory but have its own tab in memory maybe" ... "not in
// the right side bar" ... "the world model inside the memory section should have its own tab too"):
// the middle has three tabs: Memory (what is held, recent calls), World model, Loops and graph (the
// agent guidance: loops, the step graph, notes at their triggers, what was sent and its verdicts).
// 2026-11-27 (his words: "The main center screen for every page should be easy to navigate and
// beautifully visual and interactive ... the topology of the organism is a good example"): the tab
// bar carries the project and the memory switch, so the figure of each tab starts at the top; the
// world model is one wide figure of the four levels with the idle writer's queue beside it, then
// the levels, the outline and the writer's state.

import Link from "next/link";
import { useCallback, useMemo, useState } from "react";
import { NoteLine, setControl } from "@/components/Controls";
import GoalsPanel from "@/components/GoalsPanel";
import GuidancePanel, { GuidanceAbout } from "@/components/GuidancePanel";
import MainTabs from "@/components/MainTabs";
import { MemoryList } from "@/components/MemoryList";
import { Block, HashRoute, MemoryFiles, MemoryHeld, MemoryLevels, MemoryOutcomes, MemoryProjects, MemoryRecall, MemoryRouting, ProjectPick, TIERS } from "@/components/MemoryPanels";
import PlateCanvas from "@/components/PlateCanvas";
import { useTopology } from "@/components/Topology";
import Workspace, { RailSection } from "@/components/Workspace";
import { LiveBadge } from "@/components/LivePanels";
import { Failed, Pill, Switch } from "@/components/ui";
import { engineServing, NO_ENGINE, stageGroup, type LiveEvent } from "@/lib/livefeed";
import { useLive, useLiveEvents } from "@/lib/livehook";
import type { PaintOpts } from "@/lib/paint";
import { foldersOf, type Folder } from "@/lib/strata";
import { tiersOf, useActivity, useFade } from "@/lib/activity";
import { memoryScene, worldModelScene, type WorldLevel } from "@/lib/pagescenes";
import { refresh, useApi, useApiLevel, type Control, type Projects, type Status } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { useProjectKey } from "@/lib/evidence";
import { levelsOf, LEVELS, type ComprehensionReply } from "@/lib/comprehension";

type TiersReply = { nano?: Record<string, number>; projects?: Record<string, Record<string, number>>; total?: Record<string, number> };
type Goals = { open?: number; total?: number };
type ItemsCount = { total?: number; items?: unknown[] };

/** The rail's tabs; the old section names open the matching place (`/memory/#memories`). */
const TABS = ["memories", "goals", "guidance", "outcomes", "routing", "files"] as const;
const BLOCKS: Record<string, string> = {};

/** The memory setting as one switch on the tab bar: its state in words, and what it does on hover.
 *  The same setting, with its full description, is on Settings. */
function MemorySwitch({ c, onNote }: { c: Control; onNote: (n: { text: string; tone: "verified" | "error" }) => void }) {
  const [busy, setBusy] = useState(false);
  const on = c.value !== "off";
  return (
    <span className="inline-flex items-center gap-1.5" title={`${c.help || c.desc || ""} Applies ${c.applies}.`} data-e2e="memory-switch" data-value={c.value}>
      <span className="mono text-[10px] uppercase tracking-[0.12em] text-faint">memory</span>
      <Switch
        on={on}
        busy={busy}
        label={`Memory ${on ? "on" : "off"}`}
        onChange={async (v) => {
          setBusy(true);
          const n = await setControl(c.key, v ? "on" : "off");
          setBusy(false);
          onNote(n);
        }}
      />
      <span className={`mono w-6 text-[10.5px] uppercase tracking-[0.1em] ${on ? "text-verified" : "text-attention"}`}>{on ? "on" : "off"}</span>
      {c.source !== "default" && <Pill>changed</Pill>}
    </span>
  );
}

export default function Memory() {
  const status = useApi<Status>("/api/status", 5000);
  const index = useApi<unknown>("/api/memory", 10000);
  const projects = useApi<Projects>("/api/projects", 10000);
  const controls = useApi<Control[]>("/api/controls");
  const [note, setNote] = useState<{ text: string; tone: "verified" | "error" } | null>(null);
  const now = useNow();
  const memCtl = controls.data?.find((c) => c.key === "memory");
  const list = useMemo(() => [...(projects.data?.projects ?? [])].sort((a, b) => (b.last_ms ?? 0) - (a.last_ms ?? 0)), [projects.data]);
  const { tree, error: treeError } = useTopology();
  const memNode = tree?.byPath.get("memory") ?? null;
  const ownIndex = index.data && typeof index.data === "object" ? Object.keys(index.data as object).length : null;
  const act = useActivity(tree);
  const fade = useFade(act);
  const memTiers = useApi<TiersReply>("/api/memory/tiers", 10000);
  const level = useApiLevel();
  const { key } = useProjectKey();
  const picked = list.find((p) => p.key === key) ?? null;
  const denseCtl = controls.data?.find((c) => c.key === "dense_cache") ?? null;
  // The Nano's own memory, as one more column, when the engine reports its records per tier.
  const own = useMemo(() => tiersOf(memTiers.error ? null : memTiers.data), [memTiers.data, memTiers.error]);
  const folders = useMemo(() => {
    const f: Folder[] = foldersOf(list).slice(0, own ? 7 : 8);
    return own ? [{ key: "nano-own", label: "the Nano's own", episodes: 0, bytes: 0, tiers: own, own: true }, ...f] : f;
  }, [list, own]);
  const paint: PaintOpts = useMemo(() => ({ status: status.data, carts: [], drones: new Map(), hide: new Set<string>(), act, fade }), [status.data, act, fade]);
  const build = useCallback((W: number, H: number) => (memNode ? memoryScene(memNode, folders, W, H, own ? null : ownIndex) : null), [memNode, folders, ownIndex, own]);
  // The world model: notes per level (`/api/comprehension`), the idle writer's queue and progress
  // per level (`/api/live` `writer`). Writer and memory events read the tiers and levels again.
  const live = useLive();
  const compUrl = key ? `/api/comprehension?project=${encodeURIComponent(key)}` : "/api/comprehension";
  const comp = useApi<ComprehensionReply>(compUrl, 15000);
  // With no engine serving, the writer the live state reports is the interface process's own
  // (it runs none): it is then not reported, so the figure says so and never shows a zero.
  const writer = engineServing(live.reply) === false ? null : (live.reply?.writer ?? null);
  const worldLevels: WorldLevel[] = useMemo(
    () =>
      levelsOf(comp.data).map((l) => {
        const w = writer?.levels[`L${l.level}` as "L1" | "L2" | "L3" | "L4"] ?? null;
        return { level: l.level as 1 | 2 | 3 | 4, count: comp.data ? l.count : (w?.count ?? null), done: w?.done ?? null, total: w?.total ?? null };
      }),
    [comp.data, writer],
  );
  const onMemEvent = useCallback(
    (e: LiveEvent) => {
      const g = stageGroup(e.stage, e.kind);
      if (g === "writer" || g === "G3" || /memory|tier|episode/i.test(e.stage)) {
        void refresh("/api/memory/tiers");
        void refresh(compUrl);
      }
    },
    [compUrl],
  );
  useLiveEvents(onMemEvent);
  // Until the picked project's comprehension first answers (a project switch included), the
  // figure waits with "loading world model" rather than drawing levels as not reported.
  const compReading = comp.loading && !comp.error;
  const buildWorld = useCallback((W: number, H: number) => (memNode && !compReading ? worldModelScene(memNode, worldLevels, writer, W, H) : null), [memNode, worldLevels, writer, compReading]);

  // Records per tier of the picked project: the tier route first, else the project's own figures.
  const tiers: Record<string, number> | null = useMemo(() => {
    const t = memTiers.data;
    if (key === "nano") return t?.nano ?? own ?? null;
    if (key && t?.projects?.[key]) return t.projects[key];
    return picked?.tiers ?? null;
  }, [memTiers.data, key, own, picked]);
  const held = tiers ? TIERS.reduce((a, t) => a + (tiers[t.key] ?? 0), 0) : null;
  // Waiting on the first reply of the tiers or the projects: the counts are being read, not
  // missing.
  const tiersReading = tiers === null && !memTiers.error && !projects.error && (memTiers.loading || projects.loading);
  const goals = useApi<Goals>(key ? `/api/goals?project=${encodeURIComponent(key)}` : null, 10000);
  const items = useApi<ItemsCount>(key ? `/api/memory/items?project=${encodeURIComponent(key)}&limit=2000` : null, 30000);
  const writerWord = engineServing(live.reply) === false ? NO_ENGINE : writer ? `writer ${writer.in_session ? "writing" : writer.held ? "held" : "idle"} · ${writer.queue ?? "-"} queued` : live.error ? "live state did not answer" : live.reply ? "no idle writer reported" : "reading the idle writer";

  const about = (
    <div className="grid">
      <RailSection title="What this page is">
        <p className="text-[12px] leading-relaxed text-muted">
          What the Nano keeps across sessions, one folder per project: core records, active topics, the archive and temporary notes. Verified outcomes come back with the cited lines re-read fresh. Memory is a separate system from learning.
        </p>
      </RailSection>
      <RailSection title="Tiers">
        <ul className="grid gap-2">
          {TIERS.map((t) => (
            <li key={t.key} className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-3 text-[11.5px]">
              <span className="text-text">{t.label}</span>
              <span className="leading-relaxed text-faint">{t.about}</span>
            </li>
          ))}
        </ul>
      </RailSection>
      <RailSection title="The world model">
        <ul className="grid gap-2">
          {LEVELS.map((l) => (
            <li key={l.level} className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-3 text-[11.5px]">
              <span className="text-text">{l.label}</span>
              <span className="leading-relaxed text-faint">{l.about}</span>
            </li>
          ))}
        </ul>
        <p className="mt-2 text-[11.5px] leading-relaxed text-faint">The idle writer writes the notes when the machine has room; its progress shows along each level.</p>
      </RailSection>
      <RailSection title="The figures">
        <p className="text-[11.5px] leading-relaxed text-faint">
          Strata: four tiers as layers, each project folder a column through every tier, its footprint by size, one bright bead per verified episode, brightness from real memory activity. World model: one point per note, the idle writer&apos;s progress along each level&apos;s front edge, its queue as beads. Both are read again on each writer and memory event.
        </p>
      </RailSection>
      <RailSection title="Recent calls">
        <p className="text-[11.5px] leading-relaxed text-faint">The context packs the evidence ledger recorded for the picked project: the places each brought back, whether memory was on, whether it was kept as an episode, and what followed.</p>
      </RailSection>
    </div>
  );

  return (
    <Workspace
      title="Memory"
      status={
        <>
          <LiveBadge />
          {memCtl && <Pill tone={memCtl.value === "off" ? "muted" : "verified"}>memory {memCtl.value}</Pill>}
          <span>{held === null ? (tiersReading ? "reading records" : "records not reported") : `${held} ${held === 1 ? "record" : "records"} held`}</span>
          {engineServing(live.reply) !== false && <span className="normal-case tracking-normal">{writerWord}</span>}
        </>
      }
      about={about}
      defaultTab="memories"
      rail={[
        { id: "memories", label: "Memories", count: items.data?.total ?? items.data?.items?.length, body: <MemoryList projectKey={key} /> },
        { id: "goals", label: "Goals", count: goals.data?.open, body: <GoalsPanel projectKey={key} /> },
        { id: "guidance", label: "Guidance", body: <GuidanceAbout /> },
        {
          id: "outcomes",
          label: "Outcomes",
          body: (
            <div className="grid gap-6">
              <MemoryOutcomes projectKey={key} level={level} root={picked?.root ?? null} />
              <MemoryProjects list={list} current={key} projects={projects.data} error={projects.error} statusRoot={status.data?.root} now={now} />
            </div>
          ),
        },
        { id: "routing", label: "Routing", body: <MemoryRouting projectKey={key} /> },
        {
          id: "files",
          label: "Files",
          body: <MemoryFiles project={picked} projectKey={key} level={level} denseCtl={denseCtl} memoryOn={memCtl ? memCtl.value !== "off" : null} index={index} statusRoot={status.data?.root} memoryDir={status.data?.memory_dir} />,
        },
      ]}
    >
      <HashRoute tabs={TABS} blocks={BLOCKS} />
      {(index.error || projects.error) && !projects.data && <Failed error={index.error ?? projects.error ?? ""} />}
      <div className="grid min-w-0 gap-8">
        <NoteLine note={note} />

        <MainTabs
          storageKey="nm-main:memory"
          label="Memory views"
          right={
            <>
              <ProjectPick value={key} projects={projects.data} />
              {memCtl && <MemorySwitch c={memCtl} onNote={setNote} />}
              <Link href="/topology/#/memory" className="link mono text-[10.5px]">
                see in topology
              </Link>
            </>
          }
          tabs={[
            {
              id: "memory",
              label: "Memory",
              count: held ?? undefined,
              title: "What is held per tier, the strata, and what came back for recent calls",
              aliases: ["held", "recall"],
              body: (
                <>
                <Block title="Held" hint="Records per tier in the picked project. A tier opens its records." right={<span className="mono text-[10.5px] text-faint">{held === null ? (tiersReading ? "reading" : "not reported") : `${held} in this project`}</span>}>
                  <MemoryHeld tiers={tiers} reading={tiersReading} projectKey={key} now={now} all={memTiers.data?.total ?? null} />
                  <section aria-label="The pool in strata" className="mt-3">
                    <PlateCanvas build={build} paint={paint} status={status.data} tree={tree} what="memory tiers" failed={tree ? null : treeError} label="Memory in four tiers seen as strata, each project folder a column through them" className="h-[clamp(260px,42vh,480px)] border border-line" />
                    <div className="mono mt-1.5 text-[10px] uppercase tracking-[0.16em] text-faint">Fig. 1 · tiers as strata · one column per project folder</div>
                  </section>
                </Block>

                <Block id="recall" title="Recent calls · what came back" hint="The newest context packs recorded for the picked project. One opens its places and what followed.">
                  <MemoryRecall projectKey={key} level={level} root={picked?.root ?? null} now={now} />
                </Block>
                </>
              ),
            },
            {
              id: "world",
              label: "World model",
              count: comp.data ? comp.data.total_notes : undefined,
              title: "The four levels the idle writer builds, and the project outline",
              body: (
                <>
                <Block id="world" title="World model · four levels of comprehension" hint="The four levels the idle writer builds from the material, its progress along each, and the project outline." e2e="world-model" right={<span className="mono text-[10.5px] text-faint">{comp.data ? `${comp.data.total_notes} notes` : comp.error ? "comprehension did not answer" : "reading"}</span>}>
                  <div className="grid min-w-0 gap-5">
                    <section aria-label="The world model in four levels" className="min-w-0">
                      <PlateCanvas build={buildWorld} paint={paint} status={status.data} tree={tree} what="world model" failed={tree ? null : treeError} label={`The 4-level world model: ${worldLevels.map((l) => `L${l.level} ${l.count ?? "not reported"}`).join(", ")}; idle writer ${writer?.queue ?? "not reported"} queued`} className="h-[clamp(300px,46vh,520px)] border border-line" />
                      <div className="mono mt-1.5 flex flex-wrap justify-between gap-x-4 text-[10px] uppercase tracking-[0.16em] text-faint">
                        <span>Fig. 2 · four levels, architecture on top · one point per note</span>
                        <span className="normal-case tracking-normal">{engineServing(live.reply) === false ? "no engine serving: no writer runs" : writerWord}</span>
                      </div>
                    </section>
                    <MemoryLevels projectKey={key} writer={writer} serving={engineServing(live.reply)} />
                  </div>
                </Block>

                </>
              ),
            },
            {
              id: "loops",
              label: "Loops and graph",
              title: "Loops, the step graph, reminders at their triggers, what was sent and its verdicts",
              aliases: ["guidance", "guidance-work", "loops-graph"],
              body: (
                <>
                <Block id="guidance-work" title="Loops and graph · what the Nano tells your coding agent" hint="Through the hooks: loops that continue the agent at a stop until their condition holds, the step graph and its next ready step, notes at their triggers, and the evidence ledger's verdict on each delivery. The rail's Guidance tab says what each part means." e2e="guidance-block">
                  <GuidancePanel projectKey={key} />
                </Block>

                </>
              ),
            },
          ]}
        />
      </div>
    </Workspace>
  );
}
