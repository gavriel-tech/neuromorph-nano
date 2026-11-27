"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "Dude. The discriptors should be mostly hidden ... And
// that whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals ... you have to think of it as if it's an
// interactive system for the main parts the right menu are libraries and or discription tools etc.
// left menu are the main interface nav like it is now. ... We need the interface to be amazing." and
// "aslo the parameters main page ui should be way better. more visual and interactive with real
// fucntionality."): the Parameters page in the Workspace frame. The middle is the map atlas of every
// MicroModel (or the same maps as a grouped list), filtered as one; a picked map opens in the right
// rail with its lineage and its controls. The rail's tabs: Add (a job filter, built through the held-out
// split gate), Import (your own maps and cartridges), Archive (per MicroModel), Kinds, About.
// 2026-11-25 (directions of Luis Jake Gabriel III: the interface shows and controls the entire
// parameter mapping, including adding and editing maps). The same reads and actions are in MCP
// (nm_cartridges maps, map_remove, filter_*) and the API. Below the maps, the cartridge parameters
// (CartridgeParams): every object holding parameters, read and edited into new versions.

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { NoteLine } from "@/components/Controls";
import { roleTitle } from "@/components/MicroModel";
import MapAtlas, { AtlasFilters, MapSummary, type MapView } from "@/components/MapAtlas";
import CartridgeParams, { ParamKindsKey, linkedKey } from "@/components/CartridgeParams";
import { BackupsLine } from "@/components/CartridgeRail";
import ParamFocus from "@/components/ParamFocus";
import { ALL_STATES, MapsCtx, cleanable, sum, type MapFilter } from "@/components/MapCore";
import { useMapsState } from "@/components/MapsState";
import { ImportPanel } from "@/components/MapInspector";
import MapList from "@/components/MapList";
import { AddFilter, ArchiveTab, KindsTab } from "@/components/MapRail";
import Workspace, { RailSection } from "@/components/Workspace";
import { Failed, Placeholder } from "@/components/ui";
import { bytes } from "@/lib/api";

const ABOUT = (
  <div className="grid">
    <RailSection title="What this is">
      <p className="read text-[12px] leading-relaxed text-muted">Every parameter map each MicroModel&apos;s cartridges are cut from, and the job filters built over them, layer by layer. Open any map to read what is in it, edit the ones meant to be edited, build or rebuild a job filter, plug one in or out, import your own, archive or remove.</p>
    </RailSection>
    <RailSection title="Reading the atlas">
      <ul className="read grid gap-1.5 text-[11.5px] leading-relaxed text-faint">
        <li>A lane per MicroModel: its job filters as switches, the heat of its layers (maps per layer, stacked by kind), its maps grouped by what they are for, the cartridges they feed.</li>
        <li>A tile is a map: width is its size on disk, the top line its kind, the border its state, the strip the layers it covers. Experiments are small squares.</li>
        <li>Hover a tile, a layer or a cartridge to light its lineage; click a tile to open it here. Ctrl or shift click picks several for one archive or delete.</li>
      </ul>
    </RailSection>
    <RailSection title="Cartridge parameters">
      <p className="read text-[11.5px] leading-relaxed text-faint">Below the maps, or alone in the cartridges view, every object that holds learned parameters, per MicroModel: socket cartridges, job filters, project adapters (the lens and the read-out), learning units, preference counts and base cartridges, drawn as each MicroModel&apos;s chain from its open weights up; a tile&apos;s strip is the layer range its label records, and the small squares are its other versions. Open one to read its tensors (shape, norms, a histogram, values by page) and edit it: a strength, layers switched off, values set. Apply makes a new version beside it, on trial or plugged out; the source is never written, and Audit sets the new version aside. A cartridge holding no parameters says so.</p>
    </RailSection>
    <RailSection title="States">
      <ul className="read grid gap-1 text-[11.5px] leading-relaxed text-faint">
        <li>
          <span className="text-muted">current</span>: serves, or builds the base cartridge serving now.
        </li>
        <li>
          <span className="text-muted">stale</span>: made on an older base cartridge.
        </li>
        <li>
          <span className="text-muted">history</span>: built earlier versions; kept for the record and for rebuilding.
        </li>
        <li>
          <span className="text-muted">experiment</span>: sweeps and comparisons nothing reads.
        </li>
        <li>
          <span className="text-muted">log</span>, <span className="text-muted">origin not recorded</span>: surgery&apos;s log; maps that do not say what they were measured on.
        </li>
      </ul>
    </RailSection>
    <RailSection title="Where else">
      <p className="read text-[11.5px] leading-relaxed text-faint">The same reads and actions are in MCP (nm_cartridges maps, map_remove, filter_*) and the API; every write is logged in Audit, and Audit undoes it.</p>
    </RailSection>
  </div>
);

export default function Parameters() {
  const { q, surgery, roles, build, note, api, form } = useMapsState();
  const { role, setRole, job, setJob, layers, setLayers } = form;
  // The filter is shared by the atlas and the list (every state shown at first).
  const [filter, setFilter] = useState<MapFilter>({ role: "all", states: ALL_STATES, layer: null, text: "" });
  const [view, setView] = useState<MapView>("atlas");
  // Focus on one object's parameters (from Cartridges, "view parameters", or a lane row): the
  // middle becomes that object's working view; leaving it brings the atlas back.
  const [focus, setFocusState] = useState<string | null>(null);
  useEffect(() => {
    const read = () => setFocusState(linkedKey());
    read();
    window.addEventListener("popstate", read);
    window.addEventListener("hashchange", read);
    return () => {
      window.removeEventListener("popstate", read);
      window.removeEventListener("hashchange", read);
    };
  }, []);
  const setFocus = (key: string | null) => {
    window.history.pushState(null, "", key ? `/parameters/?cartridge=${encodeURIComponent(key)}` : "/parameters/");
    setFocusState(key);
    window.scrollTo({ top: 0 });
  };

  const totals = useMemo(() => {
    const rows = roles.flatMap((r) => r.maps);
    const jf = Object.values(surgery.data?.roles ?? {}).flatMap((r) => r.job_filters ?? []);
    const clean = rows.filter(cleanable);
    return { maps: rows.length, bytes: sum(rows), stale: rows.filter((m) => m.state === "stale").length, clean: sum(clean), filters: jf.length, plugged: jf.filter((f) => f.enabled !== false).length, archived: roles.reduce((a, r) => a + (r.archived?.length ?? 0), 0) };
  }, [roles, surgery.data]);

  return (
    <MapsCtx.Provider value={api}>
      <Workspace
        title="Parameters"
        status={
          <>
            {q.data && (
              <>
                <span title="Every map of every MicroModel, and their size on disk">
                  {totals.maps} maps · {bytes(totals.bytes)}
                </span>
                <span title="Job filters plugged in, of all built">
                  {totals.plugged} of {totals.filters} job filters plugged in
                </span>
                {totals.stale > 0 && (
                  <span className="text-attention" title="Made on an older base cartridge">
                    {totals.stale} stale
                  </span>
                )}
                <span title="Experiments, history, stale maps and maps of unrecorded origin that can be archived">{bytes(totals.clean)} cleanable</span>
                {build?.status === "running" && <span className="text-signal">building {roleTitle(build.role)}/{build.job}</span>}
              </>
            )}
            <Link href="/cartridges/#flow" className="link normal-case tracking-normal" data-e2e="flow-link">
              components
            </Link>
          </>
        }
        rail={[
          { id: "add", label: "Build", body: <AddFilter role={role} setRole={setRole} job={job} setJob={setJob} layers={layers} setLayers={setLayers} build={build} note={note?.at === "build" ? note.note : null} /> },
          { id: "import", label: "Import", body: <ImportPanel roles={roles.map((r) => r.role)} /> },
          {
            id: "archive",
            label: "Archive",
            count: totals.archived || undefined,
            body: (
              <div className="grid gap-5">
                <ArchiveTab />
                <RailSection title="Backups">
                  <BackupsLine parts={["maps", "cartridges"]} />
                </RailSection>
              </div>
            ),
          },
          {
            id: "kinds",
            label: "Kinds",
            body: (
              <div className="grid gap-5">
                <RailSection title="Kinds of maps">
                  <KindsTab filter={filter} setFilter={setFilter} />
                </RailSection>
                <RailSection title="Objects holding parameters">
                  <ParamKindsKey />
                </RailSection>
              </div>
            ),
          },
        ]}
        about={ABOUT}
      >
        {focus && <ParamFocus key={focus} k={focus} layerCount={api.layerCount} onLeave={() => setFocus(null)} onOpen={(k) => setFocus(k)} />}
        {!focus && q.error && !q.data && <Failed error={q.error} />}
        {!focus && !q.data && !q.error && <Placeholder className="h-72" what="parameter maps" />}
        {!focus && q.data && (
          <div className="grid min-w-0 gap-5">
            <NoteLine note={note?.note ?? null} />
            <MapSummary filter={filter} setFilter={setFilter} />
            <AtlasFilters filter={filter} setFilter={setFilter} view={view} setView={setView} />
            {/* Both map views stay in the page; the one not chosen is hidden. */}
            <div hidden={view !== "atlas"}>
              <MapAtlas filter={filter} setFilter={setFilter} />
            </div>
            <div hidden={view !== "list"}>
              <MapList filter={filter} />
            </div>
          </div>
        )}
        {/* The parameters of every object that holds them, cartridge or not: under the maps, or alone
            in the cartridges view. */}
        <div className="mt-5 min-w-0" hidden={focus !== null}>
          <CartridgeParams layerCount={api.layerCount} onFocus={(k) => setFocus(k)} role={filter.role} text={view === "cartridges" ? filter.text : ""} />
        </div>
      </Workspace>
    </MapsCtx.Provider>
  );
}
