"use client";

// 2026-11-26 (Luis Jake Gabriel III, his words: "The discriptors should be mostly hidden ... And that
// whole information database for it should be on a right side panel. The main core of the pages
// should be on actual work flow and core function visuals ... it should be centered on the
// interactive cartridge system"): the cartridge system's state, read once from the engine and shared
// by the decks, the library and the detail in the side panel; the plug and eject actions; and the
// detail of a cartridge or of a deck, which reads its record fresh each time the engine answers.

import Link from "next/link";
import { useMemo, useState } from "react";
import DeleteFlow from "@/components/DeleteFlow";
import MicroModelName, { roleTitle } from "@/components/MicroModel";
import { NoteLine } from "@/components/Controls";
import { ModuleSwitches } from "@/components/RolesRouting";
import { useRail } from "@/components/Workspace";
import { CartridgeCard, KIND_WORD, STACK_RANK, fitOf, type Kart, type Link as CartLink } from "@/components/CartridgeBay";
import { BaseVersions, CartridgeSpec, change } from "@/components/CartridgeRail";
import { CartridgeInspector, ViewParams } from "@/components/CartridgeEdit";
import { compartmentName, type LearnedUnitsReply } from "@/components/LearnedUnits";
import { PREFS_PATH, type PrefList } from "@/components/PreferenceBuilder";
import { Empty, KV, Pill, Placeholder, Switch } from "@/components/ui";
import { ago, bytes, useApi, useApiLevel, type Cartridge, type Library, type Status } from "@/lib/api";
import { useNow } from "@/lib/clock";
import { API_LIBRARY, cartridgeWords, familyOf } from "@/lib/evidence";
import { engineServing, liveModelFor } from "@/lib/livefeed";
import { useLive } from "@/lib/livehook";
import { useRoleMap } from "@/lib/roles";

export type Note = { text: string; tone: "verified" | "error" };

// 2026-11-25 (directions of Luis Jake Gabriel III): base cartridge versions per MicroModel.
export type BaseVersion = { key: string; version: number; made: string; bytes: number; present: boolean; refused: string | null; serving: boolean };
export type BaseModel = { model: string; serving: string; choice: string; versions: BaseVersion[] };
export type BasesPayload = { policy?: string; applies?: string; models?: BaseModel[]; error?: string };
export type CompatRow = { key: string; name: string; kind: string; made_for: string; role: string | null; fits: string[]; not: Record<string, string> };
export type CompatPayload = { models?: string[]; cartridges?: CompatRow[] };

export interface SurgeryJobFilter {
  cartridge: string;
  job: string;
  base: string;
  layers?: number;
  base_cos?: number;
  filtered_cos?: number;
  base_max_diff?: number;
  filtered_max_diff?: number;
  enabled?: boolean;
  /** The model the filter was built for, as the engine records it. */
  model?: string;
}
export interface JobFilterBuildStatus {
  status: string;
  role: string;
  job: string;
  started_ms: number;
  finished_ms: number;
  message: string;
}
export interface SurgeryPayload {
  roles: Record<string, { job_filters?: SurgeryJobFilter[] }>;
  job_filter_build?: JobFilterBuildStatus;
}

/** The whole cartridge system as the engine reports it: the MicroModels, every cartridge as a card
 *  (base versions, job filters, learned units, compensation, preference and engine cartridges),
 *  each one's fit, and whether the serving engine reports it connected. */
export function useDecks() {
  const status = useApi<Status>("/api/status", 5000);
  const basesQ = useApi<BasesPayload>("/api/cartridges/bases", 10000);
  const surgeryQ = useApi<SurgeryPayload>("/api/surgery", 3000);
  const cartsQ = useApi<Cartridge[]>("/api/cartridges", 2000);
  const compatQ = useApi<CompatPayload>("/api/cartridges/compat", 10000);
  const unitsQ = useApi<LearnedUnitsReply>("/api/learned-units", 10000);
  // Stated preference cartridges: plugged in and out by their stack's use, the library switch.
  const prefsQ = useApi<PrefList>(PREFS_PATH, 10000);
  const level = useApiLevel();
  const hasLib = level !== null && level >= API_LIBRARY;
  const libQ = useApi<Library>(hasLib ? "/api/cartridges/library" : null, 10000);
  const live = useLive();
  const roleMap = useRoleMap();
  const all = useMemo(() => cartsQ.data ?? [], [cartsQ.data]);
  const models = useMemo(() => status.data?.models ?? [], [status.data]);

  const karts = useMemo<Kart[]>(() => {
    const out: Kart[] = [];
    for (const m of basesQ.data?.models ?? []) {
      for (const v of m.versions) {
        out.push({
          key: `base:${v.key}`,
          name: v.key.replace(/@\d+$/, ""),
          kind: "base",
          model: m.model,
          version: `v${v.version}`,
          meta: bytes(v.bytes),
          plugged: v.serving,
          ready: v.present && !v.refused,
          note: !v.present ? "its file is not in the cartridge library yet; run the installer again" : v.refused ? cartridgeWords(v.refused.replace(`${v.key}: `, "")) : undefined,
          plug: { path: "/api/cartridges/bases", body: { model: m.model, choice: v.key }, done: `${v.key} seated on ${roleTitle(roleMap.roleOf(m.model)) || m.model}; it serves from the next start.` },
          eject: { path: "/api/cartridges/bases", body: { model: m.model, choice: "open" }, done: `${v.key} taken out of ${roleTitle(roleMap.roleOf(m.model)) || m.model}; its open weights serve from the next start.` },
          src: { type: "base", model: m.model, key: v.key },
        });
      }
    }
    for (const [role, info] of Object.entries(surgeryQ.data?.roles ?? {})) {
      for (const f of info.job_filters ?? []) {
        out.push({
          key: `filter:${f.cartridge}`,
          name: f.job,
          kind: "filter",
          model: f.model ?? roleMap.modelOf(role),
          version: `${f.layers ?? "all"} layers`,
          meta: role,
          plugged: f.enabled !== false,
          ready: true,
          plug: { path: "/api/surgery/job-filter", body: { role, job: f.job, action: "enable" }, done: `Job filter ${role}/${f.job} plugged in; the next call uses it.` },
          eject: { path: "/api/surgery/job-filter", body: { role, job: f.job, action: "disable" }, done: `Job filter ${role}/${f.job} unplugged; the next call runs without it.` },
          src: { type: "filter", role, job: f.job, id: f.cartridge },
        });
      }
    }
    const unitIds = new Set<string>();
    for (const c of unitsQ.data?.compartments ?? []) {
      for (const u of c.units) {
        unitIds.add(u.id);
        out.push({
          key: `unit:${u.key}`,
          name: u.name ?? u.id,
          kind: u.id.includes("lens") ? "lens" : "adapter",
          model: u.model ?? (u.role ? roleMap.modelOf(u.role) : null),
          version: typeof u.version === "number" ? `v${u.version}` : u.kind,
          meta: compartmentName(c),
          plugged: u.plugged,
          ready: c.on,
          note: c.on ? undefined : `its compartment (${compartmentName(c)}) is switched off`,
          plug: { path: "/api/learned-units", body: { unit: u.key, on: true }, done: `${u.id} plugged in; applies at the next call. Logged in Audit.` },
          eject: { path: "/api/learned-units", body: { unit: u.key, on: false }, done: `${u.id} unplugged; the outputs are the base's again from the next call. Logged in Audit.` },
          src: { type: "unit", key: u.key },
        });
      }
    }
    for (const lm of live.reply?.models ?? []) {
      if (!lm.lens || unitIds.has(lm.lens)) continue;
      out.push({ key: `lens:${lm.lens}`, name: lm.lens, kind: "lens", model: lm.card ?? roleMap.modelOf(lm.role), version: "project", meta: lm.role, plugged: true, ready: true, src: { type: "lens", role: lm.role, id: lm.lens } });
    }
    for (const c of all) {
      if (c.kind === "tensors" && (c.model ?? "").startsWith("qwen3")) continue; // base versions come from /api/cartridges/bases
      const fam = familyOf(c);
      const id = `${c.id}@${c.version}`;
      const settings = c.model === "settings" || c.kind === "settings";
      const kind = settings ? "settings" : fam === "preference" ? "preference" : c.kind === "affine" ? "comp" : "learned";
      const model = settings ? null : (c.model ?? (c.base || null));
      const switchable = !settings && !(c.library ?? c.book);
      // A stated preference cartridge's newest version plugs in through its stack's use (the
      // library switch every stack uses; audited, undone in Audit).
      const stated = (prefsQ.data?.cartridges ?? []).find((p) => p.key === id && p.use && p.stack);
      if (stated?.use && stated.stack) {
        out.push({
          key: `cart:${id}`,
          name: stated.name || c.id,
          kind: "preference",
          model: null,
          version: `v${c.version}`,
          meta: `${stated.rules.length} rule(s)${stated.learned_values ? " · learned values" : ""}`,
          plugged: stated.plugged,
          ready: true,
          plug: { path: "/api/cartridges", body: { library: stated.use, stack: stated.stack }, done: `Preference cartridge ${c.id} plugged in. Logged in Audit, where it can be undone.` },
          eject: { path: "/api/cartridges", body: { library: stated.use, stack: null }, done: `Preference cartridge ${c.id} unplugged. Logged in Audit, where it can be undone.` },
          src: { type: "cart", id: c.id, version: c.version },
        });
        continue;
      }
      out.push({
        key: `cart:${id}`,
        name: c.id,
        kind,
        model,
        version: `v${c.version}`,
        meta: typeof c.bytes === "number" ? bytes(c.bytes) : "",
        plugged: c.loaded,
        ready: true,
        plug: switchable ? { path: "/api/cartridges", body: { id: c.id, on: true }, done: `Cartridge ${c.id} plugged in. Logged in Audit, where it can be undone.` } : undefined,
        eject: switchable ? { path: "/api/cartridges", body: { id: c.id, on: false }, done: `Cartridge ${c.id} unplugged. Logged in Audit, where it can be undone.` } : undefined,
        note: switchable ? undefined : "plugged in or out by its stack for a use (Stacks view)",
        src: { type: "cart", id: c.id, version: c.version },
      });
    }
    // 2026-11-25: each cartridge's fit, from the engine's compatibility read.
    const rows = new Map((compatQ.data?.cartridges ?? []).map((r) => [r.key, r]));
    for (const k of out) {
      const r = rows.get(k.key.startsWith("filter:") ? k.key : k.key.replace(/^(base|cart):/, ""));
      if (!r) continue;
      k.fits = r.fits;
      k.why = r.not;
      k.fitText = r.made_for === "engine" ? "fits the whole engine" : r.fits.length ? `fits ${r.fits.map((m) => roleTitle(roleMap.roleOf(m)) || m).join(", ")}` : `made for ${r.made_for} · fits no installed MicroModel`;
      if (k.model && !fitOf(k, k.model).fits && !k.plugged) {
        k.ready = false;
        k.note = fitOf(k, k.model).why ?? k.note;
      }
    }
    return out.sort((a, b) => STACK_RANK[a.kind] - STACK_RANK[b.kind]);
  }, [basesQ.data, surgeryQ.data, unitsQ.data, live.reply, all, compatQ.data, roleMap, prefsQ.data]);

  // 2026-11-25 (directions of Luis Jake Gabriel III: every part shows it is properly connected): a
  // plugged cartridge is connected when the running engine reports it on its MicroModel.
  const serving = engineServing(live.reply);
  const seats = (libQ.data as unknown as { last_start?: { seats?: Record<string, { seated?: string | null; refused?: string[] }> } } | undefined)?.last_start?.seats ?? {};
  const units = unitsQ.data;
  const linkOf = (k: Kart): CartLink => {
    if (serving !== true) return { state: "unknown", why: "no engine is serving; start one to check the connection" };
    const lm = (live.reply?.models ?? []).find((m) => m.card === k.model || (!m.card && roleMap.modelOf(m.role) === k.model));
    const src = k.src;
    if (src?.type === "base") {
      const seat = seats[src.model];
      return seat?.seated === src.key ? { state: "on", why: `seated on ${src.model} by the serving engine` } : { state: "off", why: seat?.refused?.length ? seat.refused.join("; ") : `the serving engine did not seat ${src.key}; it serves from the next start` };
    }
    if (src?.type === "filter") {
      if (lm?.filters.some((f) => f.id === src.id && f.on)) return { state: "on", why: `loaded on the ${lm.role} by the engine` };
      if (lm?.filters.some((f) => f.id === src.id && !f.on)) return { state: "off", why: `the engine reports it on the ${lm.role} but switched off` };
      return { state: "off", why: lm ? `the live state does not list it on the ${lm.role}; job filters load at the MicroModel's first call` : "the live state does not report this MicroModel" };
    }
    if (src?.type === "unit") {
      const u = units?.compartments.flatMap((c) => c.units).find((x) => x.key === src.key);
      return u?.acts ? { state: "on", why: "acting: the engine reads it at the next call of its project" } : { state: "off", why: u?.why_not ?? "not acting" };
    }
    if (src?.type === "lens") return lm?.lens === src.id ? { state: "on", why: "the embedder reads through it" } : { state: "off", why: "the engine does not report this lens" };
    return k.plugged ? { state: "on", why: "loaded by the engine" } : { state: "off" };
  };

  /** The newest call on a MicroModel in the live record, in ms; null when the record shows none. */
  const lastCall = (model: string): number | null => {
    const roles = roleMap.rolesOf(model);
    let best: number | null = null;
    for (const e of live.events) if (e.model && (e.model === model || roles.includes(e.model)) && (best === null || e.at_ms > best)) best = e.at_ms;
    return best;
  };

  return {
    status,
    models,
    basesQ,
    surgeryQ,
    cartsQ,
    compatQ,
    unitsQ,
    libQ,
    hasLib,
    level,
    live,
    roleMap,
    all,
    karts,
    serving,
    linkOf,
    lastCall,
    liveFor: (model: string) => liveModelFor(live.reply?.models, model),
  };
}

export type Decks = ReturnType<typeof useDecks>;

/** Plug and eject through the engine, with the card busy meanwhile and a short landing animation. */
export function usePlug() {
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [landing, setLanding] = useState<string | null>(null);
  const [note, setNote] = useState<Note | null>(null);
  const act = async (k: Kart, what: "plug" | "eject") => {
    const a = what === "plug" ? k.plug : k.eject;
    if (!a) return;
    setBusyKey(k.key);
    const ok = await change(a.path, a.body, a.done, setNote);
    setBusyKey(null);
    if (ok && what === "plug") {
      setLanding(k.key);
      window.setTimeout(() => setLanding((x) => (x === k.key ? null : x)), 900);
    }
  };
  const run = async (key: string, path: string, body: unknown, done: string) => {
    setBusyKey(key);
    await change(path, body, done, setNote);
    setBusyKey(null);
  };
  return { busyKey, landing, note, setNote, act, run };
}

const linkCls = "press mono border-b border-line-strong px-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:text-text";

/** A cartridge opened in the side panel: its card with its plug, its record read fresh, and the
 *  controls it has (version choice, remove, inspect and edit values). */
export function KartDetail({ kartKey }: { kartKey: string }) {
  const d = useDecks();
  const plug = usePlug();
  const rail = useRail();
  const k = d.karts.find((x) => x.key === kartKey);
  if (!k) {
    const reading = !d.basesQ.data || !d.cartsQ.data || !d.surgeryQ.data;
    return <div data-e2e="cartridge-drawer">{reading ? <Placeholder className="h-24" what="cartridge" /> : <Empty title="Not reported any more">It was unplugged or removed since it was opened.</Empty>}</div>;
  }
  const src = k.src;
  const link = d.linkOf(k);
  const fit = k.model ? fitOf(k, k.model) : null;
  const f = src?.type === "filter" ? (d.surgeryQ.data?.roles[src.role]?.job_filters ?? []).find((x) => x.cartridge === src.id) : undefined;
  const u = src?.type === "unit" ? d.unitsQ.data?.compartments.flatMap((c) => c.units).find((x) => x.key === src.key) : undefined;
  const c = src?.type === "cart" ? d.all.find((x) => x.id === src.id && x.version === src.version) : undefined;
  // The key its parameters answer to, across every cartridge kind (base, job filter, adapter/unit, lens, library cartridge).
  const paramKey =
    src?.type === "filter"
      ? `filter:${src.role}/${src.job}`
      : src?.type === "base"
        ? src.key
        : src?.type === "unit"
          ? u
            ? u.kind === "adapter" && typeof u.version === "number"
              ? `${u.id}@${u.version}`
              : u.id
            : null
          : src?.type === "lens"
            ? src.id
            : src?.type === "cart"
              ? `${src.id}@${src.version}`
              : null;
  const cartRole = src?.type === "filter" ? src.role : src?.type === "lens" ? src.role : k.model ? d.roleMap.roleOf(k.model) : "";
  return (
    <div className="grid gap-4" data-e2e="cartridge-drawer" data-key={k.key}>
      <NoteLine note={plug.note} />
      <CartridgeCard k={k} busy={plug.busyKey === k.key} link={k.plugged ? link : undefined} onPlug={k.plug ? () => void plug.act(k, "plug") : undefined} onEject={k.eject ? () => void plug.act(k, "eject") : undefined} landing={plug.landing === k.key} />
      {paramKey && <ViewParams paramKey={paramKey} role={cartRole || undefined} withOptions />}
      <div className="grid">
        <KV k="Kind">{KIND_WORD[k.kind]}</KV>
        <KV k="MicroModel">{k.model ? <MicroModelName model={k.model} /> : "the whole engine"}</KV>
        <KV k="State">{k.plugged ? <span className="text-verified">plugged in</span> : <span className="text-muted">in the cartridge library</span>}</KV>
        {fit && <KV k="Fit">{fit.fits ? <span className="text-verified">fits</span> : <span className="text-attention">{fit.why}</span>}</KV>}
        {k.fitText && <KV k="Fits">{k.fitText}</KV>}
        {k.note && <KV k="Note">{k.note}</KV>}
        {k.plugged && <KV k="Connection">{link.state === "on" ? <span className="text-verified">connected</span> : link.state === "off" ? <span className="text-attention">not connected</span> : "not checked: no engine serving"}</KV>}
      </div>
      {/* Why the deck shows it connected or not, in words (on the deck it is a hover). */}
      {k.plugged && link.why && (
        <p className="read -mt-2 text-[11px] leading-relaxed text-faint" data-e2e="cart-link-why">
          {link.why}
        </p>
      )}
      {src?.type === "base" && <BaseVersions only={src.model} onNote={plug.setNote} />}
      {f && src?.type === "filter" && (
        <div className="grid">
          <KV k="Job" mono>{f.job}</KV>
          <KV k="Built over" mono>{f.base}</KV>
          <KV k="Layers" mono>{f.layers ?? "every layer"}</KV>
          <KV k="Cosine to the open weights" mono>{f.base_cos !== undefined && f.filtered_cos !== undefined ? `${f.base_cos.toFixed(4)} to ${f.filtered_cos.toFixed(4)}` : "not measured"}</KV>
          <KV k="Largest change" mono>{f.base_max_diff !== undefined && f.filtered_max_diff !== undefined ? `${f.base_max_diff.toFixed(2)} to ${f.filtered_max_diff.toFixed(2)}` : "not measured"}</KV>
          <KV k="File" mono>{f.cartridge}</KV>
          <div className="mt-2 flex items-center justify-between gap-3">
            <Link href="/parameters/" className="link mono text-[10.5px]">
              build another on Parameters
            </Link>
            <DeleteFlow<{ audit_id?: string }>
              path="/api/surgery/job-filter"
              body={{ role: src.role, job: f.job, action: "delete" }}
              label="remove"
              title="Asks what would go first; a removed filter can be put back from Audit."
              startClass="press mono border-b border-line px-0.5 text-[10px] uppercase tracking-[0.08em] text-faint hover:text-error"
              done={() => `Job filter ${src.role}/${f.job} removed (kept aside; Undo in Audit puts it back).`}
              refresh={["/api/surgery", "/api/live"]}
              onNote={plug.setNote}
              onDone={() => rail.clear()}
            />
          </div>
        </div>
      )}
      {u && (
        <div className="grid">
          {u.does && <p className="read mb-1 text-[11.5px] leading-relaxed text-muted">{u.does}</p>}
          <KV k="Acting">{u.acts ? <span className="text-verified">acting</span> : <span className="text-attention">{u.why_not ?? "not acting"}</span>}</KV>
          {u.acts_on && <KV k="Acts on">{u.acts_on}</KV>}
          {u.measured && <KV k="Measured" mono>{u.measured}</KV>}
          {u.made_from && <KV k="Made from">{u.made_from}</KV>}
          {u.loads && <KV k="Loads">{u.loads}</KV>}
          <KV k="Version" mono>{typeof u.version === "number" ? `v${u.version}${u.versions && u.versions > 1 ? ` of ${u.versions} kept` : ""}` : "not reported"}</KV>
          <KV k="Size" mono>{typeof u.bytes === "number" ? bytes(u.bytes) : "not reported"}</KV>
          <div className="mt-2">
            <button type="button" className={linkCls} onClick={() => rail.open("adapters")}>
              every learned unit, by compartment
            </button>
          </div>
        </div>
      )}
      {src?.type === "lens" && <p className="read text-[11.5px] leading-relaxed text-muted">The engine reports this lens on the {src.role}. Its plug and versions are under Learned.</p>}
      {c && (
        <>
          <CartridgeSpec c={c} onNote={plug.setNote} />
          <CartridgeInspector cartKey={`${c.id}@${c.version}`} />
        </>
      )}
    </div>
  );
}

/** A MicroModel's deck opened in the side panel: its name and parts, its model and precision
 *  switches, its base versions and every cartridge it holds or could hold. */
export function DeckDetail({ model }: { model: string }) {
  const d = useDecks();
  const plug = usePlug();
  const rail = useRail();
  const now = useNow();
  const m = d.models.find((x) => x.id === model);
  const roles = d.roleMap.rolesOf(model);
  const lm = d.liveFor(model);
  const last = d.lastCall(model);
  const mine = d.karts.filter((k) => k.model === model);
  // As the deck stacks them: the base first, then bottom up.
  const docked = mine.filter((k) => k.plugged);
  const waiting = mine.filter((k) => !k.plugged);
  const row = (k: Kart, n?: number) => (
    <li key={k.key} className="flex min-w-0 items-center gap-2 border-b border-line/50 py-1.5">
      {n !== undefined && <span className="mono w-4 shrink-0 text-right text-[10px] text-quiet">{n}</span>}
      <button type="button" className="press mono min-w-0 flex-1 truncate text-left text-[11px] text-text hover:text-sense-pale" onClick={() => rail.show({ title: k.name, key: k.key, body: <KartDetail kartKey={k.key} /> })}>
        {k.name} <span className="text-faint">{KIND_WORD[k.kind]} · {k.version}</span>
      </button>
      {k.plug || k.eject ? (
        <Switch on={k.plugged} busy={plug.busyKey === k.key} disabled={!k.plugged && !k.ready} label={`${k.plugged ? "Unplug" : "Plug in"} ${k.name}`} onChange={(on) => void plug.act(k, on ? "plug" : "eject")} />
      ) : (
        <Pill tone={k.plugged ? "learned" : "muted"}>{k.plugged ? "in" : "out"}</Pill>
      )}
    </li>
  );
  return (
    <div className="grid gap-4" data-e2e="deck-detail" data-model={model}>
      <NoteLine note={plug.note} />
      <MicroModelName model={model} layout="stack" nameClass="text-[13px] uppercase tracking-[0.16em] text-text" />
      <div className="grid">
        <KV k="Engine">{d.serving === true ? "serving" : d.serving === false ? "not serving" : "not known yet"}</KV>
        <KV k="On the card">{lm?.executing ? <span className="text-signal">working</span> : lm?.resident ? "held" : "not held"}</KV>
        <KV k="Last call" mono>{last ? ago(last, now) : "not in the live record"}</KV>
        <KV k="Graphics memory" mono>{lm?.vram_mb ? `${lm.vram_mb.toLocaleString()} MiB` : "not measured"}</KV>
        <KV k="Layers" mono>{typeof m?.layers === "number" ? m.layers : "not reported"}</KV>
        <KV k="Heads" mono>{typeof m?.heads === "number" ? m.heads : "not reported"}</KV>
      </div>
      <div className="mono flex flex-wrap items-center gap-x-3 gap-y-1 border-y border-line py-2 text-[10px]">
        <span className="text-[9.5px] uppercase tracking-[0.12em] text-quiet">parameters</span>
        {roles.map((r) => (
          <Link key={r} href={`/parameters/?role=${encodeURIComponent(r)}`} className={linkCls}>
            {roleTitle(r)} maps &amp; objects
          </Link>
        ))}
        <Link href="/parameters/#add" className={linkCls}>
          build
        </Link>
      </div>
      {roles.length ? (
        <div className="grid gap-2">
          {roles.map((r) => (
            <ModuleSwitches key={r} role={r} onNote={plug.setNote} />
          ))}
        </div>
      ) : (
        <div className="mono border border-dashed border-line px-2.5 py-2 text-[10.5px] text-quiet">no role runs on this model now; choose it for a role on Models</div>
      )}
      <BaseVersions only={model} onNote={plug.setNote} />
      <div className="grid gap-1" data-e2e="deck-order">
        <div className="eyebrow" title="The order the deck stacks them, bottom up: the base cartridge, compensation, job filters, adapters and lenses, then preference">
          Docked · in the order they apply
        </div>
        <ol className="grid">
          {docked.map((k, i) => row(k, i + 1))}
          {docked.length === 0 && <li className="mono py-2 text-[11px] text-quiet">nothing docked; its open weights serve</li>}
        </ol>
      </div>
      {waiting.length > 0 && (
        <div className="grid gap-1">
          <div className="eyebrow">In the cartridge library</div>
          <ul className="grid">{waiting.map((k) => row(k))}</ul>
        </div>
      )}
    </div>
  );
}
