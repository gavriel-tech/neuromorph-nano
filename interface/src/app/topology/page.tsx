import type { Metadata } from "next";
import { SystemPanel } from "@/components/LivePanels";
import Topology, { LastCallRail, SystemsRail } from "@/components/Topology";
import Workspace from "@/components/Workspace";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The main core of the pages should be on actual work
// flow and core function visuals ... the right menu are libraries and or discription tools etc. left
// menu are the main interface nav like it is now."): the organism fills the middle; a picked part's
// details, the systems and the last call are in the side panel.

export const metadata: Metadata = { title: "Topology · Neuromorph Nano" };

const about = (
  <div className="grid gap-3 text-[12.5px] leading-relaxed text-muted">
    <p>The topology is the organism drawn from the engine&apos;s own anatomy: every system, the components inside it and the real connections between them.</p>
    <p>Click a part to see its details here: what it is, its ports, settings, controls and its own activity over the last hour and day. A part with parts inside opens into its own view; the path above the map goes back up, as does Escape.</p>
    <p>Scroll or use the + and - buttons (or the + - 0 keys) to zoom; drag to move while zoomed. A pinch outward opens the part under the fingers.</p>
    <p>Each call a coding agent makes lights the parts it used, in the order they ran; the engine&apos;s live stage events light their parts as they run. Nothing lights without a real event.</p>
  </div>
);

/** The topology full screen: the drill-down map, with the side panel of the picked part. */
export default function TopologyPage() {
  return (
    <Workspace
      title="Topology"
      railKey="topology"
      defaultTab="systems"
      rail={[
        { id: "systems", label: "Systems", body: <SystemsRail /> },
        { id: "last-call", label: "Last call", body: <LastCallRail /> },
        { id: "system", label: "Live", body: <SystemPanel /> },
      ]}
      about={about}
    >
      <div className="flex h-[calc(100dvh-14rem)] min-h-[340px] flex-col md:h-[calc(100dvh-14.5rem)] lg:h-[calc(100dvh-7.5rem)] lg:min-h-[420px]">
        <Topology variant="full" panel="rail" />
      </div>
    </Workspace>
  );
}
