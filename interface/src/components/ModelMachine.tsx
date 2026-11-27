"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "the activity of the micromodels on the model page
// seem off. like the activity is fake. jsut one micromodel is shown lit up and not dynamically or
// based on real activity" ... "also it doesnt let you control the micromodels base open source
// weights completely or show them on the ui for the micromodels" ... "the oipen weights used is not
// the micromodel. that should be clearly diferentiated but not overyly through text" ... "The main
// core of the pages should be on actual work flow and core function visuals"): one MicroModel as a
// live machine. Its layers are lit only by the real calls that ran it (the engine's activity record
// for its slots, refreshed as new calls land); its state dot follows the live feed (executing,
// resident, a fresh step just seen); the base cartridge it runs on (filled square) and the upstream
// open weights it was built from (hollow ring) sit under its role name; precision, VRAM, size and
// what is docked are figures; the precision and "Runs on" controls stay on the card. Pressing the
// machine shows its detail in the side panel (`ModelDetail`): what it runs on, its shape, live
// state, activity by layer, and its tensors and clusters.

import Link from "next/link";
import type { ReactNode } from "react";
import { ControlRow } from "@/components/Controls";
import DotText from "@/components/DotText";
import MicroModelName, { BaseMark, WeightsMark, roleTitle } from "@/components/MicroModel";
import MicroModelWeights from "@/components/MicroModelWeights";
import { Matrix } from "@/components/PagePlates";
import { useTopology } from "@/components/Topology";
import { WeightsReading, WeightsDrawer } from "@/components/WeightsDrawers";
import { RailSection, useRail } from "@/components/Workspace";
import { KV, Pill, Placeholder } from "@/components/ui";
import { modelLayers, msText, useActivity } from "@/lib/activity";
import { ago, bits, bytes, useApi, type Cartridge, type Control, type Model, type Status } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { engineServing, liveModelFor, modelState, type LiveModel } from "@/lib/livefeed";
import { useLive } from "@/lib/livehook";
import { useRoleMap } from "@/lib/roles";

export type Note = { text: string; tone: "verified" | "error" };
export type StatusModel = Model & { kv_heads?: number; hidden?: number; head_groups?: number; loaded?: boolean };
export type BaseVersion = { key: string; version: number; bytes: number; present: boolean; refused: string | null; serving: boolean; made?: string };
export type BaseModel = { model: string; serving: string; choice: string; versions: BaseVersion[] };
export type LayerAct = ReturnType<typeof modelLayers>;
/** The newest live step seen for a MicroModel since the page opened: when it arrived, and its stage. */
export type Pulse = { seen: number; stage: string };

type Downloads = { models?: { id: string; repo: string; revision: string; bytes: number; files: number; present: number; partial: number; on_disk: number; state: string }[]; job?: { what: string; running: boolean; done: number; total: number } | null };

/** How long a fresh live step keeps the machine's dot lit. */
export const FRESH_MS = 6000;

/** The live model of a released model: by its card id, else by the role it serves now. */
export function liveFor(models: LiveModel[] | null | undefined, id: string, roles: string[]): LiveModel | null {
  return liveModelFor(models, id) ?? (models ?? []).find((m) => roles.includes(m.role.toLowerCase())) ?? null;
}

/** The state word under the role name. */
export function stateWord(lm: LiveModel | null, liveReported: boolean, serving: boolean | null = null): string {
  // With no engine serving there is no MicroModel state to read: the word says so.
  if (serving === false) return "no engine serving";
  const st = modelState(lm);
  if (st !== "not reported") return st;
  return liveReported ? "not in the live state" : "live state not reported";
}

/** A small figure: a key over a value, a faint line under it. */
function Fig({ k, v, sub, title }: { k: string; v: ReactNode; sub?: ReactNode; title?: string }) {
  return (
    <div className="min-w-0" title={title}>
      <div className="mono text-[9.5px] uppercase tracking-[0.14em] text-quiet">{k}</div>
      <div className="mono mt-1 truncate text-[12px] text-text">{v}</div>
      {sub !== undefined && <div className="mono mt-0.5 truncate text-[10px] text-faint">{sub}</div>}
    </div>
  );
}

/** The layers' activity as a strip of bars, bottom layer first (read from the same levels as the
 *  matrix). */
export function LayerBars({ levels, className = "h-8" }: { levels: number[]; className?: string }) {
  const n = levels.length;
  return (
    <svg viewBox={`0 0 ${n * 4} 20`} preserveAspectRatio="none" className={`block w-full ${className}`} role="img" aria-label={`activity by layer over 24 hours, ${levels.filter((l) => l > 0).length} of ${n} layers ran`}>
      {levels.map((l, i) => (
        <rect key={i} x={i * 4 + 0.5} width={3} y={20 - Math.max(0.6, 20 * Math.min(1, l))} height={Math.max(0.6, 20 * Math.min(1, l))} fill={l > 0 ? "var(--sense-pale)" : "var(--line-3)"} opacity={l > 0 ? 0.35 + 0.65 * Math.min(1, l) : 1} />
      ))}
    </svg>
  );
}

export function ModelMachine({
  m,
  role,
  base,
  basesError,
  ml,
  cover,
  lm,
  liveReported,
  serving = null,
  drones,
  carts,
  onCard,
  ctl,
  filters,
  share,
  pulse,
  now,
  selected,
  onSelect,
  onInspect,
  onNote,
}: {
  m: StatusModel;
  role: string;
  base: BaseModel | undefined;
  basesError: boolean;
  ml: LayerAct;
  cover: string;
  lm: LiveModel | null;
  liveReported: boolean;
  /** Whether an engine is serving (null when not known): with none, live figures are not measured. */
  serving?: boolean | null;
  drones?: Set<string>;
  carts: Cartridge[];
  onCard?: number[];
  ctl?: Control;
  filters: string[];
  share: number;
  pulse: Pulse | null;
  now: number;
  selected: boolean;
  onSelect: () => void;
  onInspect: () => void;
  onNote: (n: Note) => void;
}) {
  const st = modelState(lm);
  const fresh = !!pulse && now > 0 && now - pulse.seen < FRESH_MS;
  const hot = st === "executing" || fresh;
  const own = carts.filter((c) => (c.model ?? c.base) === m.id && c.loaded);
  const socks = lm?.filters ?? [];
  const dockedNames = [...own.map((c) => `cartridge ${c.id}`), ...(socks.length ? socks.map((f) => `${f.kind} ${f.id}${f.on ? "" : " (off)"}`) : filters.map((f) => `job filter ${f}`)), ...(lm?.lens ? [`lens ${lm.lens}`] : [])];
  const v = base?.versions.find((x) => x.key === base.serving);
  const name = role ? roleTitle(role) : "MicroModel (no role)";
  const resident = onCard?.length ?? 0;
  return (
    <section className={`reveal min-w-0 border-t pt-3 transition-colors ${selected ? "border-sense-pale" : "border-line"}`} data-e2e="model-card" data-model={m.id} data-state={hot ? "active" : st}>
      <button type="button" onClick={onSelect} aria-pressed={selected} className="press block w-full min-w-0 text-left" title="Show what it runs on, its shape, live state, activity by layer and tensors in the side panel">
        <div className="flex items-center justify-between gap-3">
          <span className="flex min-w-0 items-center gap-2.5">
            <span aria-hidden className={`inline-block h-2 w-2 shrink-0 ${hot ? "bg-signal motion-safe:animate-pulse" : st === "resident" ? "bg-verified" : "border border-quiet"}`} />
            <h2 className={`mono truncate text-[13px] uppercase leading-none tracking-[0.2em] ${selected ? "text-sense-pale" : "text-text"}`}>{name}</h2>
          </span>
          <span className={`mono shrink-0 text-[10px] uppercase tracking-[0.12em] ${hot ? "text-signal" : st === "resident" ? "text-muted" : "text-faint"}`} title="From the live feed">
            {fresh && st !== "executing" ? "stepping" : stateWord(lm, liveReported, serving)}
          </span>
        </div>
        <div className="mono mt-2 flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5 text-[10.5px]" data-e2e="model-base">
          {base ? <BaseMark base={base.serving} /> : <span className="text-quiet">{basesError ? "base cartridge not reported" : "reading"}</span>}
          {v && <span className="text-faint">{bytes(v.bytes)}</span>}
          <WeightsMark model={m.id} />
        </div>
        <div className="mt-4 overflow-hidden">
          <Matrix id={m.id} cell={12} fluid drones={drones} carts={carts} onCard={onCard} levels={ml?.levels ?? null} live={lm} layers={m.layers} heads={m.heads} />
        </div>
      </button>
      <div className="mono mt-2 grid min-h-[3.1rem] content-start gap-0.5 text-[10.5px] text-faint" title={cover} data-e2e="model-activity">
        {typeof m.layers !== "number" ? (
          <span>layers not reported</span>
        ) : !ml ? (
          <span>layer activity not reported</span>
        ) : ml.calls === 0 ? (
          <span>no call ran it in the last 24 h · dark</span>
        ) : (
          <>
            <span className="text-muted">
              {ml.calls} {ml.calls === 1 ? "call" : "calls"} in 24 h · {ml.hour} in the last hour · {msText(ml.ms)}
            </span>
            <span>
              last ran {ml.last && now > 0 ? ago(ml.last, now) : "at an unknown time"}
              {ml.perLayer ? " · lit by layer" : " · lit alike (layers not reported)"}
            </span>
          </>
        )}
      </div>
      <div className="mono mt-1 flex min-h-4 flex-wrap items-baseline gap-x-3 text-[10.5px]" data-e2e="model-live">
        {fresh ? (
          <span className="text-signal">now · {pulse?.stage}</span>
        ) : pulse && now > 0 ? (
          <span className="text-faint">
            last live step {pulse.stage} · {ago(pulse.seen, now)}
          </span>
        ) : (
          <span className="text-quiet">no live step since this page opened</span>
        )}
        {/* Forward passes and their summed time since the serving process started, from the live feed. */}
        <span className={`w-full ${typeof lm?.forwards === "number" ? "text-faint" : "text-quiet"}`} data-e2e="model-forwards" title="Forward passes this MicroModel ran since the serving engine started, and the time they took in all (the engine's own count)">
          {typeof lm?.forwards === "number" ? `${lm.forwards.toLocaleString()} forwards${typeof lm.busy_ms === "number" ? ` · ${msText(lm.busy_ms)} busy` : ""}` : serving === false ? "forwards not measured: no engine serving" : "forwards not measured"}
        </span>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-line pt-3">
        <Fig k="precision" v={lm?.precision ?? bits(m.precision)} sub={lm?.precision ? "live" : "set"} title={m.precision_source ? `Set: ${bits(m.precision)} (${m.precision_source})` : undefined} />
        <Fig k="vram" v={lm && lm.vram_mb !== null ? `${Math.round(lm.vram_mb)} MiB` : "not measured"} sub={!lm ? (serving === false ? "no engine serving" : "not reported") : resident ? `${resident} layers on card` : lm.resident ? "resident" : "not held"} title="On the graphics card now, from the live feed" />
        <Fig k="size" v={<DotText text={(m.bytes / (1 << 30)).toFixed(2)} size={12} label={bytes(m.bytes)} />} sub={`GB · ${Math.round(share * 100)}%`} title={`${bytes(m.bytes)} of model files, ${Math.round(share * 100)}% of all; read only`} />
        <Fig
          k="docked"
          v={
            <Link href="/cartridges/" className="hover:text-sense-pale">
              {dockedNames.length}
            </Link>
          }
          sub={dockedNames.length ? "plugged in" : "nothing"}
          title={dockedNames.length ? dockedNames.join(", ") : "No cartridge, job filter or lens is plugged in on it"}
        />
      </div>
      {ctl && (
        <div className="mt-1 border-t border-line">
          <ControlRow c={ctl} compact onNote={onNote} />
        </div>
      )}
      <MicroModelWeights model={m.id} onNote={onNote} />
      <button type="button" onClick={onInspect} className="press mono mt-3 border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.12em] text-muted hover:text-text" title="Its tensors and parameter clusters, read only, in the side panel">
        Inspect tensors · {m.id}
      </button>
    </section>
  );
}

/** One MicroModel's detail for the side panel: what it runs on, its shape, live state, activity by
 *  layer, and its tensors and clusters. Reads the same shared routes as the page. */
export function ModelDetail({ model, tensors = false }: { model: string; tensors?: boolean }) {
  const status = useApi<Status>("/api/status", 5000);
  const bases = useApi<{ models?: BaseModel[]; applies?: string }>("/api/cartridges/bases", 15000);
  const dl = useApi<Downloads>("/api/downloads", 5000);
  const live = useLive();
  const roleMap = useRoleMap();
  const { tree } = useTopology();
  const act = useActivity(tree);
  const now = useNow();
  const rail = useRail();
  const m = status.data?.models.find((x) => x.id === model) as StatusModel | undefined;
  const b = bases.data?.models?.find((x) => x.model === model);
  const w = dl.data?.models?.find((x) => x.id === model);
  const job = dl.data?.job && dl.data.job.what === `models:${model}` ? dl.data.job : null;
  const lm = liveFor(live.reply?.models, model, roleMap.rolesOf(model));
  const ml = m && typeof m.layers === "number" ? modelLayers(act, tree, model, m.layers, roleMap.rolesOf(model)) : null;
  const loaded = status.data?.loaded?.find((x) => x.id === model);
  return (
    <div className="grid min-w-0" data-e2e="model-detail" data-model={model}>
      <div className="mb-3">
        <MicroModelName model={model} layout="stack" nameClass="text-[13px] uppercase tracking-[0.2em] text-text" />
      </div>
      <RailSection title="Runs on · base cartridge">
        {!b ? (
          <span className="mono text-[10.5px] text-faint">{bases.error ? "base cartridges not reported" : "reading"}</span>
        ) : (
          <div className="grid gap-1.5">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <BaseMark base={b.serving} />
              <span className="mono text-[10px] text-faint">chosen: {b.choice === "auto" ? "newest" : b.choice === "open" ? "open mode" : b.choice}</span>
            </div>
            <ul className="grid">
              {b.versions.map((v) => (
                <li key={v.key} className="mono flex min-w-0 items-baseline justify-between gap-2 border-b border-line/40 py-1 text-[10.5px]">
                  <span className={`truncate ${v.serving ? "text-text" : "text-muted"}`}>
                    {v.key}
                    {v.made ? <span className="text-quiet"> · {v.made}</span> : null}
                  </span>
                  <span className="flex shrink-0 items-baseline gap-2">
                    <span className="text-faint">{bytes(v.bytes)}</span>
                    <Pill tone={v.refused ? "error" : v.serving ? "verified" : v.present ? "muted" : "attention"} title={v.refused ?? undefined}>
                      {v.refused ? "refused" : v.serving ? "serving" : v.present ? "on disk" : "missing"}
                    </Pill>
                  </span>
                </li>
              ))}
            </ul>
            <span className="mono text-[10px] text-quiet">
              choose on the card · applies {bases.data?.applies ?? "at the next start"} ·{" "}
              <Link href="/cartridges/#bases" className="link">
                versions
              </Link>
            </span>
          </div>
        )}
      </RailSection>
      <RailSection title="Built from · upstream open weights">
        {!w ? (
          <span className="mono text-[10.5px] text-faint">{dl.error ? "downloads not reported" : "reading"}</span>
        ) : (
          <div className="grid gap-0.5">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <WeightsMark model={model} />
              <Pill tone={w.state === "installed" ? "verified" : w.state === "partial" ? "attention" : "muted"}>{w.state === "installed" ? "on disk" : w.state === "partial" ? "partly" : "not on disk"}</Pill>
            </div>
            <KV k="repository" mono>
              <span title={w.repo}>{w.repo}</span>
            </KV>
            <KV k="revision" mono>
              <span title={w.revision}>{w.revision.slice(0, 12)}</span>
            </KV>
            <KV k="licence" mono>
              Apache-2.0
            </KV>
            <KV k="files" mono>
              {w.present} of {w.files}
              {w.partial ? ` · ${w.partial} partial` : ""}
            </KV>
            <KV k="on disk" mono>
              {bytes(w.on_disk)}
              {w.bytes && w.on_disk < w.bytes ? ` of ${bytes(w.bytes)}` : ""}
            </KV>
            {job && (
              <div className="mono mt-1 grid gap-1 text-[10.5px]">
                <span className={job.running ? "text-signal" : "text-faint"}>
                  {job.running ? "downloading" : "last download"} · {job.done} of {job.total} files
                </span>
                <div className="h-1 w-full bg-line">
                  <div className="h-full bg-sense-pale transition-[width] duration-300" style={{ width: `${job.total ? Math.round((job.done / job.total) * 100) : 0}%` }} />
                </div>
              </div>
            )}
            <span className="mono mt-1 text-[10px] text-quiet" title="The base cartridge was built from these upstream open weights; they are never altered. Neuromorph Nano runs standalone on its base cartridges; upstream open weights are optional for fallback (open mode), output-parity reference checks, parameter mapping, and cartridge surgery.">
              not the MicroModel · optional ·{" "}
              <button type="button" onClick={() => rail.open("downloads")} className="link">
                downloads
              </button>
            </span>
          </div>
        )}
      </RailSection>
      <RailSection title="Shape and precision">
        {!m ? (
          <Placeholder className="h-16" what="shape and precision" />
        ) : (
          <div className="grid">
            <KV k="layers" mono>
              {typeof m.layers === "number" ? m.layers : "not reported"}
            </KV>
            <KV k="heads" mono>
              {typeof m.heads === "number" ? m.heads : "not reported"}
              {typeof m.kv_heads === "number" ? ` · ${m.kv_heads} key/value` : ""}
            </KV>
            {typeof m.hidden === "number" && (
              <KV k="hidden width" mono>
                {m.hidden}
              </KV>
            )}
            <KV k="weight card" mono>
              {m.card === true ? "checked" : m.card === false ? "none" : "not reported"}
            </KV>
            <KV k="main file" mono>
              <span title={m.main_file ?? undefined}>{m.main_file ?? "not reported"}</span>
            </KV>
            <KV k="precision" mono>
              <span title={m.precision_source}>
                {bits(m.precision)}
                {loaded?.effective_bits ? ` · running ${loaded.effective_bits}-bit` : ""}
              </span>
            </KV>
            <KV k="size" mono>
              {bytes(m.bytes)}
            </KV>
          </div>
        )}
      </RailSection>
      <RailSection title="Live">
        <div className="grid">
          <KV k="state" mono>
            {stateWord(lm, !!live.reply, live.reply ? engineServing(live.reply) : null)}
          </KV>
          {lm && (
            <>
              <KV k="vram" mono>
                {lm.vram_mb !== null ? `${Math.round(lm.vram_mb)} MiB` : "not reported"}
              </KV>
              <KV k="precision" mono>
                {lm.precision ?? "not held"}
              </KV>
              <KV k="lens" mono>
                {lm.lens ?? "none"}
              </KV>
            </>
          )}
          {(loaded?.layers_on_card?.length ?? 0) > 0 && (
            <KV k="layers on card" mono>
              {loaded?.layers_on_card?.length}
            </KV>
          )}
          {(lm?.filters ?? []).map((f) => (
            <KV key={`${f.kind}-${f.id}`} k={f.kind} mono>
              {f.id}
              {f.on ? "" : " (off)"}
            </KV>
          ))}
        </div>
      </RailSection>
      <RailSection title="Activity · 24 h">
        {!ml ? (
          <span className="mono text-[10.5px] text-faint">{m && typeof m.layers !== "number" ? "layers not reported" : "activity not reported"}</span>
        ) : (
          <div className="grid gap-2">
            <LayerBars levels={ml.levels} />
            <div className="mono flex justify-between text-[9.5px] text-quiet">
              <span>layer 0</span>
              <span>{ml.perLayer ? "by layer" : ml.calls ? "layers not reported: lit alike" : "no calls"}</span>
              <span>{ml.levels.length - 1}</span>
            </div>
            <div className="grid">
              <KV k="calls" mono>
                {ml.calls}
              </KV>
              <KV k="last hour" mono>
                {ml.hour}
              </KV>
              <KV k="time" mono>
                {msText(ml.ms)}
              </KV>
              <KV k="last ran" mono>
                {ml.last && now > 0 ? ago(ml.last, now) : ml.calls ? "unknown" : "not in 24 h"}
              </KV>
            </div>
            <span className="mono text-[10px] text-quiet">{act.cover}</span>
          </div>
        )}
      </RailSection>
      <RailSection title="Tensors and clusters · read only">{tensors ? <WeightsReading model={model} /> : <WeightsDrawer model={model} label="read the tensors" />}</RailSection>
    </div>
  );
}
