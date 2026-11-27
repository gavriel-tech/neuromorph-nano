"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals ... the right menu are libraries and or
// discription tools etc."): the floorplan of the organism fills the middle; a block clicked there
// opens its details in the side panel, where the crates and the systems are listed too.

import { useCallback, useMemo } from "react";
import { SystemPanel } from "@/components/LivePanels";
import PlateCanvas from "@/components/PlateCanvas";
import { PartRail, SystemsRail, useTopology } from "@/components/Topology";
import { Failed, FigureLoading, Placeholder } from "@/components/ui";
import Workspace, { useRail } from "@/components/Workspace";
import { useActivity, useFade } from "@/lib/activity";
import { droneSets, type PaintOpts } from "@/lib/paint";
import { plateScene } from "@/lib/plates";
import type { TNode, Tree } from "@/lib/topology";

/** Shows a part of the organism in the side panel, by its path. */
function useShowPart(tree: Tree | null) {
  const { show: railShow, open } = useRail();
  return useCallback(
    // Named, so the panel's own links (a part's parts, its parent) open in the panel too.
    function show(path: string[]) {
      const key = path.join("/");
      const n = tree?.byPath.get(key);
      if (!n) return;
      if (path.length === 0) {
        open("systems");
        return;
      }
      railShow({ key: `part:${key}`, title: n.label, body: <PartRail path={key} topologyLink onGo={show} /> });
    },
    [tree, railShow, open],
  );
}

/** Each crate, with the place in the organism that holds it. */
function crateRows(tree: Tree) {
  const rows: { crate: string; node: TNode; subs: number; deps: string[]; kind: string }[] = [];
  for (const [crate, home] of tree.crateHome) {
    const card = home.card?.id === crate ? home.card : [...tree.byPath.values()].find((n) => n.card?.id === crate)?.card;
    if (!card) continue;
    const own = [...tree.byPath.values()].find((n) => n.role === "group" && n.crate === crate) ?? home;
    rows.push({ crate, node: own, subs: card.subcomponents.length, deps: card.deps, kind: card.kind });
  }
  rows.sort((a, b) => a.node.path.join("/").localeCompare(b.node.path.join("/")) || a.crate.localeCompare(b.crate));
  return rows;
}

/** The rail's list of engine crates: each with its system, what it depends on and its parts. */
function CratesRail() {
  const { tree, error } = useTopology();
  const show = useShowPart(tree);
  if (!tree) return error ? <Failed error={error} /> : <Placeholder className="h-40" what="anatomy" />;
  const rows = crateRows(tree);
  return (
    <div className="grid gap-1" data-e2e="anatomy-crates">
      <div className="mono grid grid-cols-[5.5rem_minmax(0,1fr)_2.5rem] gap-3 border-b border-line pb-2 text-[10px] uppercase tracking-[0.12em] text-faint">
        <span>Crate</span>
        <span>System</span>
        <span className="text-right">Parts</span>
      </div>
      <ul>
        {rows.map((r) => {
          const where = r.node.path.map((_, k) => tree.byPath.get(r.node.path.slice(0, k + 1).join("/"))?.label).filter(Boolean);
          return (
            <li key={r.crate} className="border-b border-line/60">
              <button type="button" onClick={() => show(r.node.path)} className="press group grid w-full grid-cols-[5.5rem_minmax(0,1fr)_2.5rem] items-baseline gap-x-3 py-2 text-left" title={r.deps.length ? `depends on ${r.deps.join(", ")}` : "depends on nothing"}>
                <span className="mono truncate text-[12px] text-text">{r.crate}</span>
                <span className="min-w-0">
                  <span className="block truncate text-[12px] text-muted group-hover:text-text">
                    <span>{where.join(" / ")}</span>
                  </span>
                  <span className="mono block truncate text-[10.5px] text-faint">
                    <span>{r.deps.length ? `on ${r.deps.join(", ")}` : "depends on nothing"}</span>
                  </span>
                </span>
                <span className="mono text-right text-[11px] text-faint">{r.subs}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** The rail's systems, each opening its details in the rail. */
function Systems() {
  const { tree } = useTopology();
  const show = useShowPart(tree);
  return <SystemsRail onGo={show} />;
}

/** The floorplan: each system a block, one band per component, traces for the real connections. */
function Floorplan() {
  const { tree, error, status, carts, settings } = useTopology();
  const drones = useMemo(() => droneSets(settings), [settings]);
  const act = useActivity(tree);
  const fade = useFade(act);
  const paint: PaintOpts = useMemo(() => ({ status, carts: carts ?? [], drones, hide: new Set<string>(), act, fade }), [status, carts, drones, act, fade]);
  const build = useCallback((W: number, H: number) => (tree ? plateScene(tree, tree.root, W, H, carts ?? [], status, drones) : null), [tree, carts, status, drones]);
  const show = useShowPart(tree);
  const open = useCallback(
    (key: string) => {
      const n = tree?.byPath.get(key);
      if (n) show(n.path);
    },
    [tree, show],
  );
  return (
    <>
      {error && !tree && <Failed error={error} />}
      {!tree && !error && (
        <div className="topo night ticks relative h-[clamp(420px,calc(100dvh-12rem),900px)] overflow-hidden border border-line">
          <FigureLoading what="anatomy floorplan" />
        </div>
      )}
      {tree && (
        <section aria-label="Floorplan of the whole system" data-e2e="anatomy-floorplan">
          <PlateCanvas build={build} paint={paint} status={status} onOpen={open} tree={tree} what="floorplan" label="Floorplan of the organism: each system a block, one band per component, traces for the real connections" className="ticks h-[clamp(420px,calc(100dvh-12rem),900px)] border border-line" />
          <div className="mono mt-3 text-[10.5px] uppercase tracking-[0.18em] text-faint">Fig. 1 · the organism as a floorplan · click a block for its details</div>
        </section>
      )}
    </>
  );
}

function StatusLine() {
  const { tree } = useTopology();
  if (!tree) return null;
  const comps = tree.root.children.reduce((a, c) => a + c.children.length, 0);
  return (
    <>
      <span>{tree.root.children.length} systems</span>
      <span>{comps} components</span>
      <span>{tree.crateHome.size} crates</span>
    </>
  );
}

const about = (
  <div className="grid gap-3 text-[12.5px] leading-relaxed text-muted">
    <p>What the organism is made of: its systems and their components, drawn as one floorplan from the engine&apos;s own anatomy, and every engine crate with the system that holds it.</p>
    <p>Click a block to see its details here: what it is, its ports, settings, controls and its own activity. Each detail links to the part in the topology, where it opens with its live activity.</p>
    <p>Blocks light as the engine reports their live stages; nothing lights without a real event.</p>
  </div>
);

export default function Anatomy() {
  return (
    <Workspace
      title="Anatomy"
      railKey="anatomy"
      defaultTab="crates"
      status={<StatusLine />}
      rail={[
        { id: "crates", label: "Crates", body: <CratesRail /> },
        { id: "systems", label: "Systems", body: <Systems /> },
        { id: "system", label: "Live", body: <SystemPanel /> },
      ]}
      about={about}
    >
      <Floorplan />
    </Workspace>
  );
}
