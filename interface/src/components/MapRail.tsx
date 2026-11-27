"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "the right menu are libraries and or discription tools
// etc."): the Parameters page's right rail tabs. Add: build a job filter through the held-out split
// gate, its layers picked on the MicroModel's layer strip. Archive: every MicroModel's archived maps,
// restored or deleted. Kinds: what each kind of map is, with how many there are; picking one finds it.

import Link from "next/link";
import { maskPathsIn } from "@/lib/privacy";
import { useState, type MouseEvent } from "react";
import { NoteLine } from "@/components/Controls";
import DeleteFlow from "@/components/DeleteFlow";
import { roleTitle } from "@/components/MicroModel";
import { Pill, Segmented } from "@/components/ui";
import { bytes } from "@/lib/api";
import { MapsCtx, kindTone, sum, useMaps, type BuildState, type MapFilter, type Note } from "@/components/MapCore";
import { useMapsState } from "@/components/MapsState";
import PreferenceBuilder from "@/components/PreferenceBuilder";
import { DistilBuilder } from "@/components/DistilBuilder";

const linkBtn = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text disabled:opacity-40";
const delBtn = "press mono border-b border-line px-0.5 text-[10px] uppercase tracking-[0.1em] text-faint hover:text-error disabled:opacity-40";
const field = "mono h-9 min-w-0 border-b border-line-strong bg-transparent text-[12px] text-text outline-none placeholder:text-quiet focus-visible:border-sense-pale";

/** "all", "12-27" or "3,5,9-12" as a set of layers (null when it does not read). */
function parseLayers(spec: string, total: number): Set<number> | null {
  const s = spec.trim();
  if (s === "all") return new Set(Array.from({ length: total }, (_, i) => i));
  if (!/^[\d,-]+$/.test(s)) return null;
  const out = new Set<number>();
  for (const part of s.split(",").filter(Boolean)) {
    const m = /^(\d+)(?:-(\d+))?$/.exec(part);
    if (!m) return null;
    const a = Number(m[1]);
    const b = m[2] === undefined ? a : Number(m[2]);
    for (let i = Math.min(a, b); i <= Math.max(a, b); i++) out.add(i);
  }
  return out;
}

/** A set of layers written the way the engine takes it: "all", or ranges joined by commas. */
function writeLayers(set: Set<number>, total: number): string {
  const xs = [...set].filter((x) => x >= 0 && x < total).sort((a, b) => a - b);
  if (xs.length === total) return "all";
  const parts: string[] = [];
  for (let i = 0; i < xs.length; i++) {
    let j = i;
    while (j + 1 < xs.length && xs[j + 1] === xs[j] + 1) j++;
    parts.push(i === j ? `${xs[i]}` : `${xs[i]}-${xs[j]}`);
    i = j;
  }
  return parts.join(",");
}

/** Click a layer, then another, for a range; ctrl or shift click adds or takes out one layer. */
function LayerPicker({ total, value, onChange }: { total: number; value: string; onChange: (v: string) => void }) {
  const [anchor, setAnchor] = useState<number | null>(null);
  const set = parseLayers(value, total);
  const click = (i: number, e: MouseEvent) => {
    if (e.ctrlKey || e.metaKey || e.shiftKey) {
      const next = new Set(set ?? []);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      onChange(next.size ? writeLayers(next, total) : `${i}`);
      setAnchor(null);
    } else if (anchor === null) {
      setAnchor(i);
      onChange(`${i}`);
    } else {
      onChange(writeLayers(new Set(Array.from({ length: Math.abs(i - anchor) + 1 }, (_, k) => Math.min(i, anchor) + k)), total));
      setAnchor(null);
    }
  };
  return (
    <div className="grid gap-0.5">
      <div className="flex h-7 gap-px" role="group" aria-label="Layers to fit the filter on">
        {Array.from({ length: total }, (_, i) => (
          <button
            key={i}
            type="button"
            aria-pressed={set?.has(i) ?? false}
            onClick={(e) => click(i, e)}
            title={`layer ${i}${anchor !== null ? ` (range from ${anchor})` : ""}`}
            className={`press min-w-0 flex-1 transition-colors ${set?.has(i) ? "bg-sense-pale/80 hover:bg-sense-pale" : "bg-line/50 hover:bg-line-strong"} ${anchor === i ? "outline outline-1 outline-text" : ""}`}
          />
        ))}
      </div>
      <div className="mono flex justify-between text-[9px] text-quiet">
        <span>L0</span>
        <span>{anchor !== null ? `from L${anchor}: click end layer` : set?.size === total ? `fitting all ${total} layers` : `fitting ${set?.size ?? 0} of ${total} layers`}</span>
        <span>L{total - 1}</span>
      </div>
    </div>
  );
}

// 2026-11-26 (Luis Jake Gabriel III: "This should all be part of the neuromorph nano and integrated
// fully into the mcp the api and the interface"): the lab's three builders in one tab, the same route
// as MCP (nm_cartridges filter_build, map_build, base_build) and the API (/api/lab/build). One build
// runs at a time; a new base cartridge version is kept beside the others (library.upgrade keep).
const KINDS = ["filter", "map", "base"] as const;
type Kind = (typeof KINDS)[number];
const KIND_WORD: Record<Kind, string> = { filter: "job filter", map: "parameter map", base: "base cartridge" };
const KIND_INFO: Record<Kind, { creates: string; gate: string; how: string }> = {
  filter: {
    creates: "A small plug-in filter (job-filter-<job>.json) that improves this MicroModel on one task.",
    gate: "Only saved if it beats the base cartridge on a held-out split; shows up on Cartridges ready to plug in.",
    how: "Runs the compressed base cartridge alongside the upstream open weights on prompts for the chosen task, fits per-channel gain and bias corrections on the selected layers, and checks them on held-out prompts before keeping the filter.",
  },
  map: {
    creates: "A weight-sensitivity map (pmap-*.json) plus a new base cartridge version cut from it.",
    gate: "Checked against upstream open weights; saved beside your existing base versions (nothing deleted).",
    how: "Measures how much each weight matters across attention and feed-forward matrices, compresses out the chosen share, and writes a new base cartridge version you can choose on Cartridges -> Base versions.",
  },
  base: {
    creates: "A new base cartridge version with the highest-impact weights restored.",
    gate: "Checked against upstream open weights; saved beside your serving base (nothing overwritten).",
    how: "Restores the top high-sensitivity weights into the current serving base cartridge to tighten accuracy, keeping the new version beside your existing ones so you pick which serves.",
  },
};

export function AddFilter({ role, setRole, job, setJob, layers, setLayers, build, note, kinds = KINDS, e2e = "pmap-add" }: { role: string; setRole: (v: string) => void; job: string; setJob: (v: string) => void; layers: string; setLayers: (v: string) => void; build: BuildState | null; note: Note | null; kinds?: readonly Kind[]; e2e?: string }) {
  const { roles, layerCount, filtersOf, busy, run } = useMaps();
  const [kind, setKind] = useState<Kind>(kinds[0] ?? "filter");
  const [share, setShare] = useState("0.40");
  const [device, setDevice] = useState("gpu");
  const [showHow, setShowHow] = useState(false);
  const r = roles.find((x) => x.role === role);
  const total = r ? layerCount(r.model) : null;
  const jf = filtersOf(role);
  const okLayers = layers.trim() === "all" || (/^[\d,-]+$/.test(layers.trim()) && layers.trim().length <= 32);
  const shareN = Number(share);
  const okShare = Number.isFinite(shareN) && shareN >= 0.05 && shareN <= 0.6;
  const running = build?.status === "running";
  const info = KIND_INFO[kind];
  const existingJob = jf.find((f) => f.job === job.trim());
  const start = () => {
    if (kind === "filter") return void run("build", "/api/surgery/job-filter/build", { role, job: job.trim(), layers: layers.trim() || "all" }, `Building the job filter ${roleTitle(role)}/${job.trim()}; it passes the held-out split gate before it is kept.`);
    return void run("build", "/api/lab/build", kind === "map" ? { kind, role, share: shareN, device } : { kind, role }, `Building a ${KIND_WORD[kind]} for the ${roleTitle(role)}; a new base cartridge version is kept beside the others, and you choose what serves on Cartridges, Base versions.`);
  };
  const can = !busy && !running && (kind === "filter" ? !!job.trim() && okLayers : kind === "map" ? okShare : true);
  const buildLabel = running
    ? "building..."
    : kind === "filter"
      ? `${existingJob ? "rebuild" : "build"} ${roleTitle(role)}/${job.trim() || "job"} (${layers.trim() || "all"})`
      : kind === "map"
        ? `build ${roleTitle(role)} map (${Math.round((okShare ? shareN : 0.4) * 100)}%)`
        : `build ${roleTitle(role)} base cartridge`;
  const presetBtn = "press mono border border-line px-1.5 py-0.5 text-[9px] uppercase tracking-[0.08em] text-faint hover:border-sense-pale hover:text-text";
  return (
    <div className="grid gap-4" data-e2e={e2e}>
      <div className="grid gap-2" data-e2e="lab-kind">
        <span className="eyebrow">1 · what to build</span>
        {kinds.length > 1 && <Segmented label="What the lab builds" values={[...kinds]} value={kind} onPick={(v) => setKind(v as Kind)} render={(v) => KIND_WORD[v as Kind]} />}
        <div className="mono grid gap-1 border border-line bg-bg-1/70 p-2 text-[10px]">
          <div className="flex items-start justify-between gap-2">
            <span className="leading-snug text-muted">{info.creates}</span>
            <button type="button" onClick={() => setShowHow((v) => !v)} className={`${linkBtn} shrink-0`}>
              {showHow ? "hide" : "how it works"}
            </button>
          </div>
          <span className="text-[9.5px] text-quiet">{info.gate}</span>
          {showHow && <p className="read border-t border-line/50 pt-1.5 text-[10.5px] leading-relaxed text-faint">{info.how}</p>}
        </div>
      </div>

      <label className="grid gap-1">
        <span className="eyebrow">2 · target MicroModel</span>
        <select value={role} onChange={(e) => setRole(e.target.value)} className={`${field} w-full`}>
          {roles.map((x) => (
            <option key={x.role} value={x.role} className="bg-bg-1">
              {roleTitle(x.role)} · {x.serving_base ? `base cartridge ${x.serving_base}` : "open mode"}
            </option>
          ))}
        </select>
      </label>

      {kind === "map" && (
        <div className="grid gap-2.5 border border-line bg-bg-1/40 p-2.5" data-e2e="lab-map-options">
          <div className="flex items-baseline justify-between gap-2">
            <span className="eyebrow" title="The fraction of each attention and feed-forward matrix mapped out (0.05 to 0.60)">3 · share mapped out</span>
            <span className="mono text-[10.5px] text-text">
              {okShare ? `${Math.round(shareN * 100)}% cut · ${Math.round((1 - shareN) * 100)}% kept` : "invalid"}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={0.05}
              max={0.6}
              step={0.05}
              value={okShare ? shareN : 0.4}
              onChange={(e) => setShare(Number(e.target.value).toFixed(2))}
              aria-label="Share mapped out slider"
              className="flex-1 accent-[var(--sense-pale)]"
            />
            <input value={share} onChange={(e) => setShare(e.target.value)} aria-label="Share mapped out" className={`${field} w-16 text-right`} spellCheck={false} />
          </div>
          <div className="flex items-center justify-between gap-2">
            <span className="eyebrow text-quiet">compute device</span>
            <Segmented label="Device" values={["gpu", "cpu"]} value={device} onPick={setDevice} render={(v) => (v === "gpu" ? "graphics card" : "processor")} />
          </div>
          {!okShare && <span className="mono text-[10px] text-attention">enter a share between 0.05 and 0.60</span>}
        </div>
      )}

      {kind !== "filter" && (
        <p className="mono text-[10px] leading-relaxed text-faint">
          Requires upstream open weights on disk. The new version is kept beside existing ones; pick what serves on{" "}
          <Link href="/cartridges/#bases" className="link">
            Cartridges, Base versions
          </Link>
          .
        </p>
      )}

      {kind === "filter" && (
        <div className="grid gap-2">
          <label className="grid gap-1">
            <span className="eyebrow">3 · task / job name</span>
            <input value={job} onChange={(e) => setJob(e.target.value)} placeholder="e.g. code-slice, read, write, judge-code" className={`${field} w-full`} spellCheck={false} />
          </label>
          {jf.length > 0 && (
            <div className="grid gap-1">
              <span className="mono text-[9.5px] uppercase tracking-[0.1em] text-quiet">existing filters on {roleTitle(role)} (click to rebuild)</span>
              <ul className="grid gap-0.5">
                {jf.map((f) => (
                  <li key={f.job}>
                    <button type="button" onClick={() => setJob(f.job)} className={`press mono flex w-full items-baseline justify-between gap-3 border-b border-line/40 py-1 text-left text-[10.5px] hover:text-text ${job === f.job ? "text-sense-pale" : "text-muted"}`} title="Fill this job name to rebuild its filter on the layers selected below">
                      <span className="truncate">{f.job}</span>
                      <span className="shrink-0 text-faint">
                        {typeof f.layers === "number" ? `${f.layers}L · ` : ""}
                        {f.enabled !== false ? "plugged in" : "unplugged"}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {kind === "filter" && (
        <div className="grid gap-2 border border-line bg-bg-1/40 p-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="eyebrow">4 · layers to correct</span>
            <div className="flex flex-wrap items-center gap-1">
              <button type="button" onClick={() => setLayers("all")} className={`${presetBtn} ${layers.trim() === "all" ? "border-sense-pale text-text" : ""}`}>
                all
              </button>
              {total !== null && total >= 4 && (
                <>
                  <button type="button" onClick={() => setLayers(`0-${Math.floor(total / 2) - 1}`)} className={presetBtn}>
                    first half
                  </button>
                  <button type="button" onClick={() => setLayers(`${Math.floor(total / 2)}-${total - 1}`)} className={presetBtn}>
                    second half
                  </button>
                  <button type="button" onClick={() => setLayers(`${Math.max(0, total - 8)}-${total - 1}`)} className={presetBtn}>
                    last 8
                  </button>
                </>
              )}
            </div>
          </div>
          <span className="mono text-[9.5px] leading-snug text-quiet">click start + end bar for a range, or ctrl+click to toggle single layers</span>
          {total !== null && <LayerPicker key={role} total={total} value={layers} onChange={setLayers} />}
          <input value={layers} onChange={(e) => setLayers(e.target.value)} placeholder="all or 12-27" aria-label="Layers to fit" className={`${field} w-full`} spellCheck={false} />
          {!okLayers && <span className="mono text-[10px] text-attention">enter all, a range like 12-27, or a list like 3,5,9-12 (max 32 chars)</span>}
        </div>
      )}

      <div className="grid gap-1.5">
        <button
          type="button"
          disabled={!can}
          data-e2e="lab-build"
          onClick={start}
          className="press mono min-h-9 w-full border border-sense-pale/80 bg-sense/15 px-3 text-[10.5px] uppercase tracking-[0.12em] text-sense-pale hover:bg-sense/25 hover:text-text disabled:opacity-40"
          title={kind === "filter" ? "Fits the filter on the training split, then keeps it only if it beats the base cartridge on the held-out split" : "Runs the lab in the background; one build at a time"}
        >
          {buildLabel}
        </button>
      </div>
      {build && (
        <div className="mono grid gap-1 border-l-2 border-line-strong pl-3 text-[10.5px]">
          <span className="flex items-center gap-2">
            <Pill tone={build.status === "done" ? "verified" : build.status === "error" ? "error" : build.status === "stopped" ? "attention" : "signal"}>{build.status}</Pill>
            <span className="text-text">{build.kind && build.kind !== "filter" ? `${KIND_WORD[build.kind as Kind] ?? build.kind} · ${roleTitle(build.role)}` : `${roleTitle(build.role)}/${build.job}`}</span>
          </span>
          {build.message && <span className="text-faint">{maskPathsIn(build.message)}</span>}
        </div>
      )}
      <NoteLine note={note} />
    </div>
  );
}

// 2026-11-26 (Luis Jake Gabriel III, on where cartridge building belongs: "wouldn't it be under
// learning or cartridge build?"): the Cartridges page's Build tab, every builder whose result is a
// cartridge (job filters and base cartridge versions), through the same builder and routes as the
// Parameters page. What it builds appears on the decks and in Base versions.
// Every builder whose result is a cartridge, in one tab: the lab's job filter and base cartridge
// builders (the same form as Parameters), distillation and the stated preference builder (the same
// components as the Learning page). Each writes through its own route into the cartridge library.
const BUILDS = ["filter", "base", "distil", "preference"] as const;
type Build = (typeof BUILDS)[number];
const BUILD_WORD: Record<Build, string> = { filter: "job filter", base: "base cartridge", distil: "distillation", preference: "preference" };
const CARTRIDGE_READS = ["/api/cartridges", "/api/cartridges/bases", "/api/cartridges/compat", "/api/cartridges/library"];

export function CartridgeBuildTab() {
  const { build, note, api, form } = useMapsState(CARTRIDGE_READS);
  const [what, setWhat] = useState<Build>("filter");
  return (
    <MapsCtx.Provider value={api}>
      <div className="grid gap-4" data-e2e="cartridge-build">
        <div className="grid gap-1.5" data-e2e="cartridge-build-kind">
          <span className="eyebrow">build a cartridge</span>
          <select value={what} onChange={(e) => setWhat(e.target.value as Build)} aria-label="What to build" className="mono h-9 w-full border-b border-line-strong bg-transparent text-[12px] text-text outline-none focus-visible:border-sense-pale">
            {BUILDS.map((b) => (
              <option key={b} value={b} className="bg-bg-1">
                {BUILD_WORD[b]}
              </option>
            ))}
          </select>
        </div>
        {what === "preference" ? <PreferenceBuilder /> : what === "distil" ? <DistilBuilder e2e="cartridge-build-distil" /> : <AddFilter key={what} {...form} build={build} note={note?.at === "build" ? note.note : null} kinds={[what]} e2e="cartridge-build-lab" />}
      </div>
    </MapsCtx.Provider>
  );
}

export function ArchiveTab() {
  const { roles, busy, run, onNote } = useMaps();
  return (
    <div className="grid gap-5" data-e2e="pmap-archive">
      {roles.map((r) => {
        const arch = r.archived ?? [];
        const who = roleTitle(r.role);
        if (!arch.length)
          return (
            <div key={r.role} className="mono flex items-baseline justify-between gap-3 border-b border-line/40 pb-2 text-[10.5px]">
              <span className="text-muted">{who}</span>
              <span className="text-quiet">nothing archived</span>
            </div>
          );
        return (
          <section key={r.role} className="grid gap-2 border-b border-line/40 pb-3">
            <div className="flex items-baseline justify-between gap-3">
              <span className="mono text-[11px] text-text">{who}</span>
              <span className="mono text-[10px] text-faint">
                {arch.length} · {bytes(sum(arch))}
              </span>
            </div>
            <div className="mono flex flex-wrap items-start gap-x-4 gap-y-1 text-[10.5px]">
              <button type="button" disabled={busy} onClick={() => void run(`${r.role}:restore`, "/api/parameter-maps/restore", { role: r.role, files: arch.map((m) => m.file) }, `${arch.length} maps restored into the ${who}'s folder. Undo in Audit.`)} className={linkBtn}>
                restore all
              </button>
              <DeleteFlow<{ audit_id?: string }>
                path="/api/parameter-maps/remove"
                body={{ role: r.role, files: arch.map((m) => m.file), archived: true, confirm: true }}
                label="delete the archive"
                title="Asks what would go first, then asks you to type a word; this cannot be undone"
                startClass={delBtn}
                disabled={busy}
                done={() => `The ${who}'s archive deleted. Logged in Audit.`}
                refresh={["/api/parameter-maps"]}
                onNote={onNote}
              />
            </div>
            <ul className="grid gap-1">
              {arch.map((m) => (
                <li key={m.file} className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3 border-b border-line/30 py-1">
                  <span className="min-w-0">
                    <span className="mono flex min-w-0 items-center gap-1.5 text-[11px] text-muted" title={`${m.file}: ${m.about}`}>
                      <i aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 ${kindTone(m.kind)}`} />
                      <span className="truncate">{m.file}</span>
                    </span>
                    <span className="mono block truncate text-[9.5px] text-faint" title={m.lineage?.note}>
                      {m.kind} · {bytes(m.bytes)}
                      {m.lineage?.purpose ? ` · ${m.lineage.purpose}` : ""}
                      {m.lineage?.outcome ? ` · ${m.lineage.outcome}` : ""}
                    </span>
                  </span>
                  <span className="flex items-start gap-3">
                    <button type="button" disabled={busy} onClick={() => void run(`${r.role}/${m.file}`, "/api/parameter-maps/restore", { role: r.role, files: [m.file] }, `${m.file} restored. Undo in Audit.`)} className={linkBtn}>
                      restore
                    </button>
                    <DeleteFlow<{ audit_id?: string }>
                      path="/api/parameter-maps/remove"
                      body={{ role: r.role, files: [m.file], archived: true, confirm: true }}
                      label="delete"
                      startClass={delBtn}
                      disabled={busy}
                      done={() => `${m.file} deleted from the archive. Logged in Audit.`}
                      refresh={["/api/parameter-maps"]}
                      onNote={onNote}
                    />
                  </span>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
      <p className="mono text-[10px] leading-relaxed text-quiet">Nothing serves or builds from an archive. Archive from a map&apos;s detail, a list group&apos;s clean up, or several picked in the atlas.</p>
    </div>
  );
}

export function KindsTab({ filter, setFilter }: { filter: MapFilter; setFilter: (f: MapFilter) => void }) {
  const { roles } = useMaps();
  const rows = roles.flatMap((r) => r.maps);
  const kinds = [...new Map(rows.map((m) => [m.kind, m.about])).entries()].map(([kind, about]) => {
    const ms = rows.filter((m) => m.kind === kind);
    return { kind, about, n: ms.length, b: sum(ms) };
  });
  kinds.sort((a, b) => b.b - a.b);
  return (
    <ul className="grid gap-1">
      {kinds.map(({ kind, about, n, b }) => {
        const on = filter.text.trim().toLowerCase() === kind;
        return (
          <li key={kind}>
            <button type="button" aria-pressed={on} onClick={() => setFilter({ ...filter, text: on ? "" : kind })} className={`press grid w-full gap-1 border-b border-line/40 py-2 text-left ${on ? "bg-bg-2/60" : "hover:bg-bg-2/40"}`} title={on ? "Show every kind again" : "Find the maps of this kind"}>
              <span className="flex min-w-0 items-center gap-2">
                <i aria-hidden className={`inline-block h-2 w-2 shrink-0 ${kindTone(kind)}`} />
                <span className="mono min-w-0 truncate text-[11px] uppercase tracking-[0.12em] text-text">{kind}</span>
                <span className="mono ml-auto shrink-0 text-[10px] text-faint">
                  {n} · {bytes(b)}
                </span>
              </span>
              <span className="read pl-4 text-[11px] leading-relaxed text-faint">{about}</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
