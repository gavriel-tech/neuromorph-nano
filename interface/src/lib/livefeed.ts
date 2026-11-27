/* The engine's live state (`GET /api/live`) and its event stream (`GET /api/live/stream`, Server-
 * Sent Events of timeline items). Pure parsers and mappings, shared by the pages and the tests.
 *
 * Every field is read defensively: an engine build that does not serve the route yet, or sends a
 * field in another shape, gives an empty view, never a guessed one. Nothing here invents activity:
 * a stage lights only when the engine names it in `active` or in a timeline event. */

export type LiveStage = { stage: string; model: string | null; since_ms: number | null; elapsed_ms: number | null; label?: string | null };
/** A socket docked on a MicroModel: a job-path filter (`nmengn-sockets/1`), a lens or a preference
 *  cartridge. Strings from the engine are read as `{ id }`. */
export type LiveSocket = { id: string; kind: "filter" | "lens" | "preference" | "other"; on: boolean };
export type LiveModel = {
  role: string;
  card: string | null;
  resident: boolean;
  executing: boolean;
  vram_mb: number | null;
  precision: string | null;
  filters: LiveSocket[];
  lens: string | null;
  /** Forward passes this MicroModel ran since the serving process started, and their summed time
   *  (`forwards`, `busy_ms`); null when the engine does not report them. */
  forwards?: number | null;
  busy_ms?: number | null;
};
export type LiveVram = { total_mb: number | null; used_mb: number | null; free_mb: number | null; reserve_mb: number | null; reserve_measured?: boolean | null; by_model: Record<string, number> };
/** One world-model level's progress in the idle writer: a count, or done of total. */
export type LiveLevel = { count: number | null; done: number | null; total: number | null };
export type LiveWriter = {
  queue: number | null;
  held: boolean | null;
  sessions: number | null;
  results: number | null;
  preempted: number | null;
  kept_resident: boolean | null;
  headroom_mb: number | null;
  levels: Record<"L1" | "L2" | "L3" | "L4", LiveLevel | null>;
  /** The writer's newest event: its summary, else its kind (`kept`, `preempted`, ...). */
  last_event: string | null;
  in_session?: boolean | null;
  /** What holds the idle writer back now (`call`, `quiet`, `index`, `memory`); null when nothing
   *  does or the engine does not say. */
  waiting?: string | null;
};

/** Why the idle Writer is held back, in plain words. */
export const WRITER_WAITS: Record<string, string> = {
  call: "held back: a call is in flight",
  quiet: "held back: waiting for quiet after the last call",
  index: "held back: the index is still building",
  memory: "held back: not enough free memory to load it",
};

/** The idle Writer's state now in a few words: writing, why it is held back, or whether its
 *  weights sit on the card. */
export function writerState(w: LiveWriter | null | undefined): string | null {
  if (!w) return null;
  if (w.in_session) return "writing now";
  if (w.waiting) return WRITER_WAITS[w.waiting] ?? `held back: ${w.waiting}`;
  return w.held ? "on the card, not writing" : "not loaded";
}
/** The machine's capacity (`nmengn::capacity::to_value`): the graphics card (null on a machine
 *  without one), host memory, threads and the ceilings homeostasis keeps to. */
export type LiveCard = { name: string | null; total_mb: number | null; free_mb: number | null; compute: string | null; reserve_mb: number | null; reserve_measured: boolean | null };
export type LiveCapacity = {
  card: LiveCard | null;
  host_available_mb: number | null;
  host_total_mb: number | null;
  threads: number | null;
  /** The processor's hardware threads (`hardware_threads`); `threads` is what the engine plans on. */
  hardware_threads?: number | null;
  gpu_ceiling_mb: number | null;
  ram_ceiling_mb: number | null;
  /** The user's hardware limiters in force now (`capacity.limits`). Null when the reply carries
   *  none. */
  limits?: LiveLimits | null;
};
/** The graphics card limiter in force (`capacity.limits.vram`, `/api/limits` `effective.vram`). */
export type LiveLimitVram = {
  setting: string | null;
  cap_mb: number | null;
  total_mb: number | null;
  used_mb: number | null;
  driver_free_mb: number | null;
  effective_free_mb: number | null;
  cpu_only_now: boolean | null;
  cpu_only_next_start: boolean | null;
  note: string | null;
};
/** The system memory limiter in force (`limits.ram`). `used_mb` is the Nano's own memory. */
export type LiveLimitRam = { setting: string | null; cap_mb: number | null; available_mb: number | null; room_mb: number | null; used_mb: number | null; cache_ceiling_mb: number | null };
/** The processor limiter in force (`limits.cpu`). */
export type LiveLimitCpu = { setting: string | null; cap: number | null; effective_threads: number | null; hardware_threads: number | null };
/** The hardware limiters in force. `vram_mb`, `ram_mb` and `cpu_threads` are the values in force
 *  (`auto`, or a whole number as a string; `0` for VRAM when the processor runs everything now);
 *  `vram`, `ram` and `cpu` carry the engine's detail (what auto resolves to, what is in use) and
 *  `mode` how the MicroModels sit on the card. A field the engine does not give is null. */
export type LiveLimits = {
  vram_mb: string | null;
  ram_mb: string | null;
  cpu_threads: string | null;
  mode?: string | null;
  vram?: LiveLimitVram | null;
  ram?: LiveLimitRam | null;
  cpu?: LiveLimitCpu | null;
};
/** A serving process the interface found (`processes`). */
export type LiveProcess = { pid: number | null; project: string | null; state: string | null };
/** A background job's progress (`progress`): the dense cache build, for one. */
export type LiveProgress = { kind: string; done: number | null; total: number | null; since_ms: number | null };
export type LiveEvent = {
  at_ms: number;
  kind: string;
  stage: string;
  model: string | null;
  ms: number | null;
  summary: string;
  label?: string | null;
  /** The tentacles a `tentacle.read` stage read (0 is the first), from its `tentacle_mask` count;
   *  null when the event carries no mask (another stage, or an engine build without it). */
  tentacles?: number[] | null;
};
export type LiveReply = {
  state: "idle" | "busy";
  active: LiveStage[];
  models: LiveModel[];
  vram: LiveVram | null;
  capacity: LiveCapacity | null;
  writer: LiveWriter | null;
  timeline: LiveEvent[];
  backend: "cuda" | "cpu" | null;
  /** The backend in plain words (which card and why, or why the processor). */
  backend_note?: string | null;
  /** How many serving processes the interface found; 0 when no engine is serving (the state is
   *  then the interface process's own, with no MicroModel). Null when the reply does not say. */
  serving?: number | null;
  processes?: LiveProcess[];
  progress?: LiveProgress[];
  project?: string | null;
  pid?: number | null;
  /** 2026-11-27: each memory area's setting, budget and use, as the engine sends it
   *  (`parseMemoryAreas` in `memoryareas.ts` reads it). */
  memory_areas?: unknown;
};

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
/** A count that the engine may send as a number or a flag (`preempted: true`). */
const count = (v: unknown): number | null => (typeof v === "boolean" ? (v ? 1 : 0) : num(v));
const flag = (v: unknown): boolean | null => (typeof v === "boolean" ? v : typeof v === "number" ? v > 0 : null);

function socketKind(raw: string | null): LiveSocket["kind"] {
  const k = (raw ?? "").toLowerCase();
  if (/lens/.test(k)) return "lens";
  if (/pref/.test(k)) return "preference";
  if (/filter|socket|job/.test(k)) return "filter";
  return "other";
}

/** A socket from a string (`"reranker/code-slice"`) or an object (`{id|cartridge|name, kind, enabled}`). */
export function parseSocket(v: unknown, fallback: LiveSocket["kind"] = "filter"): LiveSocket | null {
  const s = str(v);
  if (s) return { id: s, kind: fallback, on: true };
  const o = obj(v);
  if (!o) return null;
  const id = str(o.id) ?? str(o.cartridge) ?? str(o.name) ?? (str(o.job) ? `${str(o.role) ? `${str(o.role)}/` : ""}${str(o.job)}` : null);
  if (!id) return null;
  const k = str(o.kind) ?? str(o.schema) ?? str(o.family);
  const on = flag(o.enabled) ?? flag(o.on) ?? flag(o.loaded) ?? true;
  return { id, kind: k ? socketKind(k) === "other" ? fallback : socketKind(k) : fallback, on };
}

function parseLevel(v: unknown): LiveLevel | null {
  const n = num(v);
  if (n !== null) return { count: n, done: null, total: null };
  const o = obj(v);
  if (!o) return null;
  const done = num(o.done) ?? num(o.written);
  const total = num(o.total) ?? num(o.of);
  const c = num(o.count) ?? num(o.notes) ?? done;
  if (c === null && done === null && total === null) return null;
  return { count: c, done, total };
}

/** One hardware limiter value: `auto` (any case), a non-negative whole number (string or number,
 *  leading zeros dropped), else null. */
export function limitValue(v: unknown): string | null {
  if (typeof v === "number") return Number.isSafeInteger(v) && v >= 0 ? String(v) : null;
  const s = str(v);
  if (!s) return null;
  if (s.toLowerCase() === "auto") return "auto";
  if (/^\d+$/.test(s) && Number.isSafeInteger(Number(s))) return String(Number(s));
  return null;
}

const settingOf = (v: unknown): string | null => limitValue(v) ?? str(v);

/** The limiters block, or null when the reply carries none. Reads the engine's nested shape
 *  (`{vram: {setting, cap_mb, ...}, ram: {...}, cpu: {...}, mode}`) and a flat one
 *  (`{vram_mb, ram_mb, cpu_threads}`). */
export function parseLiveLimits(v: unknown): LiveLimits | null {
  const o = obj(v);
  if (!o) return null;
  const vr = obj(o.vram);
  const rm = obj(o.ram);
  const cp = obj(o.cpu);
  const vram: LiveLimitVram | null = vr
    ? {
        setting: settingOf(vr.setting),
        cap_mb: num(vr.cap_mb),
        total_mb: num(vr.total_mb),
        used_mb: num(vr.used_mb),
        driver_free_mb: num(vr.driver_free_mb),
        effective_free_mb: num(vr.effective_free_mb),
        cpu_only_now: flag(vr.cpu_only_now),
        cpu_only_next_start: flag(vr.cpu_only_next_start),
        note: str(vr.note),
      }
    : null;
  const ram: LiveLimitRam | null = rm ? { setting: settingOf(rm.setting), cap_mb: num(rm.cap_mb), available_mb: num(rm.available_mb), room_mb: num(rm.room_mb), used_mb: num(rm.used_mb), cache_ceiling_mb: num(rm.cache_ceiling_mb) } : null;
  const cpu: LiveLimitCpu | null = cp ? { setting: settingOf(cp.setting), cap: num(cp.cap), effective_threads: num(cp.effective_threads), hardware_threads: num(cp.hardware_threads) } : null;
  // The value in force: a cap when one applies, `0` while the processor runs everything, else the
  // setting (`auto`).
  const vramNow = vram ? (vram.cap_mb !== null ? limitValue(vram.cap_mb) : vram.cpu_only_now ? "0" : limitValue(vram.setting)) : null;
  const ramNow = ram ? (ram.cap_mb !== null ? limitValue(ram.cap_mb) : limitValue(ram.setting)) : null;
  const cpuNow = cpu ? (cpu.cap !== null ? limitValue(cpu.cap) : limitValue(cpu.setting)) : null;
  return {
    vram_mb: vramNow ?? limitValue(o.vram_mb) ?? limitValue(o["limits.vram_mb"]),
    ram_mb: ramNow ?? limitValue(o.ram_mb) ?? limitValue(o["limits.ram_mb"]),
    cpu_threads: cpuNow ?? limitValue(o.cpu_threads) ?? limitValue(o["limits.cpu_threads"]),
    mode: str(o.mode),
    vram,
    ram,
    cpu,
  };
}

/** The capacity block. Null when the reply carries none. A machine without a graphics card
 *  gives `card: null`, never a guessed card. */
export function parseCapacity(v: unknown): LiveCapacity | null {
  const o = obj(v);
  if (!o) return null;
  const c = obj(o.card);
  const card: LiveCard | null = c
    ? { name: str(c.name), total_mb: num(c.total_mb), free_mb: num(c.free_mb), compute: str(c.compute) ?? (num(c.compute) !== null ? String(num(c.compute)) : null), reserve_mb: num(c.reserve_mb), reserve_measured: flag(c.reserve_measured) }
    : null;
  const hw = num(o.hardware_threads);
  return { card, host_available_mb: num(o.host_available_mb), host_total_mb: num(o.host_total_mb), threads: num(o.threads) ?? hw, hardware_threads: hw, gpu_ceiling_mb: num(o.gpu_ceiling_mb), ram_ceiling_mb: num(o.ram_ceiling_mb), limits: parseLiveLimits(o.limits) };
}

/** One timeline item (or one Server-Sent Event's data). Null when it names no time. */
export function parseLiveEvent(v: unknown): LiveEvent | null {
  const o = obj(v);
  if (!o) return null;
  const at = num(o.at_ms) ?? num(o.at) ?? num(o.time_ms);
  if (at === null) return null;
  const kind = str(o.kind) ?? "event";
  const stage = str(o.stage) ?? kind;
  // The tentacles that read, from the bit mask the engine counts; the mask is said in words in
  // the summary instead of as a number.
  const mask = num(obj(o.counts)?.tentacle_mask);
  const tentacles = mask === null ? null : Array.from({ length: 32 }, (_, i) => i).filter((i) => (mask >> i) & 1);
  let summary = str(o.summary) ?? "";
  if (tentacles) summary = summary.replace(/\s*tentacle_mask \d+/, tentacles.length ? ` tentacles ${tentacles.map((i) => `T${i + 1}`).join(" ")}` : "");
  return { at_ms: at, kind, stage, model: str(o.model), ms: num(o.ms), summary, label: str(o.label), tentacles };
}

/** The whole live reply. Null when the reply is not an object (an engine without the route). */
export function parseLive(raw: unknown): LiveReply | null {
  const o = obj(raw);
  if (!o) return null;
  const active: LiveStage[] = arr(o.active)
    .map((a): LiveStage | null => {
      const x = obj(a);
      const stage = x ? str(x.stage) : str(a);
      if (!stage) return null;
      const label = x ? str(x.label) : null;
      return { stage, model: x ? str(x.model) : null, since_ms: x ? num(x.since_ms) : null, elapsed_ms: x ? num(x.elapsed_ms) : null, ...(label ? { label } : {}) };
    })
    .filter((x): x is LiveStage => x !== null);
  const models: LiveModel[] = arr(o.models)
    .map((m): LiveModel | null => {
      const x = obj(m);
      if (!x) return null;
      const role = str(x.role) ?? str(x.card);
      if (!role) return null;
      const filters = arr(x.filters)
        .map((f) => parseSocket(f, "filter"))
        .filter((f): f is LiveSocket => f !== null);
      for (const p of arr(x.preferences)) {
        const s = parseSocket(p, "preference");
        if (s) filters.push({ ...s, kind: "preference" });
      }
      const lensObj = parseSocket(x.lens, "lens");
      return {
        role,
        card: str(x.card),
        resident: flag(x.resident) ?? false,
        executing: flag(x.executing) ?? false,
        vram_mb: num(x.vram_mb),
        precision: str(x.precision),
        filters,
        lens: lensObj?.id ?? null,
        forwards: num(x.forwards),
        busy_ms: num(x.busy_ms),
      };
    })
    .filter((x): x is LiveModel => x !== null);
  const v = obj(o.vram);
  const by: Record<string, number> = {};
  for (const [k, val] of Object.entries(obj(v?.by_model) ?? {})) {
    const n = num(val) ?? num(obj(val)?.mb) ?? num(obj(val)?.vram_mb);
    if (n !== null) by[k] = n;
  }
  const vram: LiveVram | null = v ? { total_mb: num(v.total_mb), used_mb: num(v.used_mb), free_mb: num(v.free_mb), reserve_mb: num(v.reserve_mb), reserve_measured: flag(v.reserve_measured), by_model: by } : null;
  const w = obj(o.writer);
  const lv = obj(w?.levels);
  const writer: LiveWriter | null = w
    ? {
        queue: num(w.queue),
        held: flag(w.held),
        sessions: num(w.sessions),
        results: num(w.results),
        preempted: count(w.preempted),
        kept_resident: flag(w.kept_resident),
        headroom_mb: num(w.headroom_mb),
        levels: { L1: parseLevel(lv?.L1), L2: parseLevel(lv?.L2), L3: parseLevel(lv?.L3), L4: parseLevel(lv?.L4) },
        last_event: str(w.last_event) ?? str(obj(w.last_event)?.summary) ?? str(obj(w.last_event)?.kind),
        in_session: flag(w.in_session),
        waiting: str(w.waiting),
      }
    : null;
  const timeline = mergeTimeline([], arr(o.timeline).map(parseLiveEvent).filter((e): e is LiveEvent => e !== null));
  const backend = o.backend === "cuda" || o.backend === "cpu" ? o.backend : null;
  const busy = o.state === "busy" || (o.state !== "idle" && (active.length > 0 || models.some((m) => m.executing)));
  const processes: LiveProcess[] = arr(o.processes)
    .map((p) => obj(p))
    .filter((p): p is Record<string, unknown> => p !== null)
    .map((p) => ({ pid: num(p.pid), project: str(p.project), state: str(p.state) }));
  const progress: LiveProgress[] = arr(o.progress)
    .map((p) => obj(p))
    .filter((p): p is Record<string, unknown> => p !== null && str(p.kind) !== null)
    .map((p) => ({ kind: str(p.kind) as string, done: num(p.done), total: num(p.total), since_ms: num(p.since_ms) }));
  return {
    state: busy ? "busy" : "idle",
    active,
    models,
    vram,
    capacity: parseCapacity(o.capacity),
    writer,
    timeline,
    backend,
    backend_note: str(o.backend_note),
    memory_areas: o.memory_areas ?? null,
    serving: num(o.serving),
    processes,
    progress,
    project: str(o.project) ?? str(o.project_key),
    pid: num(o.pid),
  };
}

/** Whether an engine is serving: the reply's `serving` count when it gives one, else true when it
 *  names a backend. Null when not known (before the live state answers, or a reply that says
 *  neither). */
export function engineServing(r: Pick<LiveReply, "serving" | "backend"> | null | undefined): boolean | null {
  if (!r) return null;
  if (typeof r.serving === "number") return r.serving > 0;
  return r.backend !== null ? true : null;
}

/** The plain line shown where a live figure would be when no engine is serving. */
export const NO_ENGINE = "No engine is serving right now; start one with neuromorph-nano serve or daemon";

/** One identity per event, so the stream and the polled timeline never count an event twice. */
export function eventKey(e: LiveEvent): string {
  return `${e.at_ms}|${e.kind}|${e.stage}|${e.model ?? ""}|${e.ms ?? ""}`;
}

/** Timelines merged oldest first, one per identity, the newest `max` kept. */
export function mergeTimeline(prev: LiveEvent[], incoming: LiveEvent[], max = 200): LiveEvent[] {
  const by = new Map<string, LiveEvent>();
  for (const e of prev) by.set(eventKey(e), e);
  for (const e of incoming) by.set(eventKey(e), e);
  const out = [...by.values()].sort((a, b) => a.at_ms - b.at_ms);
  return out.length > max ? out.slice(out.length - max) : out;
}

/* ---------- Stages: which ganglion level or background job an event belongs to ---------- */

export type StageGroup = "G0" | "G1" | "G2" | "G3" | "writer" | "preempt" | "dense" | "filter" | "learning" | "memory" | "rebind" | "backup" | "guidance" | "other";

/** Plain names of the groups, for labels. */
export const GROUP_NAMES: Record<StageGroup, string> = {
  G0: "G0 intake and routing",
  G1: "G1 search",
  G2: "G2 reads and judging",
  G3: "G3 pack",
  writer: "idle writer",
  preempt: "preemption",
  dense: "dense cache build",
  filter: "lab build",
  learning: "learning cycle",
  memory: "memory write",
  rebind: "parameter overlay rebind",
  backup: "backup",
  guidance: "guidance delivered",
  other: "other",
};

/** The group of a stage or kind name. Order matters: the more specific names are tried first. */
export function stageGroup(stage: string, kind = ""): StageGroup {
  const s = `${kind} ${stage}`.toLowerCase();
  const bare = stage.trim().toLowerCase();
  if (/preempt/.test(s)) return "preempt";
  // One timed event each: a memory write, a parameter overlay binding a MicroModel again, a backup.
  if (bare === "memory.write" || kind === "memory") return "memory";
  if (bare === "overlay.rebind" || kind === "rebind") return "rebind";
  if (bare === "backup" || kind === "backup") return "backup";
  // 2026-11-27: guidance the Nano gave the agent through a hook (a reminder, a loop continuation,
  // a step nudge), one event per delivery, written by the hook process.
  if (bare.startsWith("guidance.") || kind === "guidance") return "guidance";
  // A lab build the interface started (a parameter map, a job filter, a distillation, a teaching run).
  if (bare === "lab.build" || bare === "filter.build") return "filter";
  // A bare `dense` stage (and its `progress` events) is the background dense cache build;
  // `G1.dense` is a question's dense search.
  if (bare === "dense" || bare === "dense_cache") return "dense";
  if (/\bl[1-4]\b|writer|comprehen|idle_?write|outline|file_note|module_note/.test(s)) return "writer";
  if (/job_?filter|split_?gate|socket|filter[._]?build/.test(s)) return "filter";
  if (/dense_?cache|embed_?build|vector_?build/.test(s)) return "dense";
  if (/learn|cycle|rollback|pref_?fit/.test(s)) return "learning";
  if (/\bg3\b|pack|muscle/.test(s)) return "G3";
  if (/\bg2\b|tentacle|judge|duel|rerank|deep_check|order_reads/.test(s)) return "G2";
  if (/\bg1\b|dense|lexical|tree|evidence|search/.test(s)) return "G1";
  if (/\bg0\b|intake|rout|signals|sense|drone_kind|compose|meaning/.test(s)) return "G0";
  return "other";
}

/** The group of an event. A writer event labelled `preempted` is a preemption: its stage is only
 *  `writer`, so the label decides. */
export function eventGroup(e: Pick<LiveEvent, "stage" | "kind"> & { label?: string | null; summary?: string }): StageGroup {
  if (e.kind === "writer" && (e.label === "preempted" || /\bpreempted\b/.test(e.summary ?? ""))) return "preempt";
  return stageGroup(e.stage, e.kind);
}

/** The writer level an event names (`L1` to `L4`), or null. */
export function writerLevel(stage: string, summary = ""): 1 | 2 | 3 | 4 | null {
  const m = /\bL([1-4])\b/i.exec(`${stage} ${summary}`);
  return m ? (Number(m[1]) as 1 | 2 | 3 | 4) : null;
}

/** The MicroModel slot a model name or role belongs to (anatomy ids `nmbrain.slot#...`). The
 *  reader reads the task, so it sits in the router slot. */
export function slotOf(model: string | null | undefined): string | null {
  const m = (model ?? "").toLowerCase();
  if (!m) return null;
  if (m.includes("embed")) return "nmbrain.slot#embedder";
  if (m.includes("rerank")) return "nmbrain.slot#reranker";
  if (m.includes("writer")) return "nmbrain.slot#writer";
  if (m.includes("reader") || m.includes("router")) return "nmbrain.slot#router";
  return null;
}

/** The anatomy ids an event lights in the topology: its ganglion level, the tentacle a read names,
 *  the MicroModel slot that ran it. Resolved against the tree by `resolvePart`. */
export function partIdsFor(e: Pick<LiveEvent, "stage" | "kind" | "model" | "summary"> & { label?: string | null; tentacles?: number[] | null }): string[] {
  const g = eventGroup(e);
  const ids: string[] = [];
  const slot = slotOf(e.model);
  switch (g) {
    case "G0":
      ids.push("nmbrain.g0");
      break;
    case "G1":
      ids.push("nmbrain.g1");
      break;
    case "G2": {
      ids.push("nmbrain.g2");
      // Only the tentacles the engine names light: the ones in the event's mask, else one named in
      // the stage or its summary. A read that names none lights no tentacle (the Reader that ran it
      // still lights).
      if (/tentacle/i.test(e.stage)) {
        const t = /\bt(?:entacle)?[ _#-]?(\d+)\b/i.exec(`${e.stage} ${e.summary}`);
        if (e.tentacles) for (const i of e.tentacles) ids.push(`nmpod.unit#t${i}`);
        else if (t) ids.push(`nmpod.unit#t${t[1]}`);
      }
      if (/judge|duel|rerank/i.test(e.stage)) ids.push("nmbrain.slot#reranker");
      break;
    }
    case "G3":
      ids.push("nmmusc#g3");
      break;
    case "writer":
      ids.push("nmbrain.slot#writer", "nmmem.hierarchy");
      break;
    case "preempt":
      ids.push("nmbrain.slot#writer", "nmbrain.homeostasis");
      break;
    case "dense":
      ids.push("nmbrain.slot#embedder", "nmsubs");
      break;
    case "filter":
      ids.push("nmlearn.cartridge");
      break;
    case "learning":
      ids.push("nmlearn");
      break;
    case "memory":
      ids.push("nmmem.gate");
      break;
    // A rebind lights the MicroModel it binds again (the slot below). A backup and a guidance
    // delivery have no part in the anatomy (it has no hooks or agent part): they show in the
    // timeline, the sidebar system strip and the System panel only.
    default:
      break;
  }
  if (slot && !ids.includes(slot)) ids.push(slot);
  return ids;
}

/** How long an event keeps its parts lit (ms): its own time, held at least long enough to see and
 *  at most a few seconds; a preemption is the short pulse it is. */
export function holdFor(e: Pick<LiveEvent, "stage" | "kind" | "ms"> & { label?: string | null; summary?: string }): number {
  if (eventGroup(e) === "preempt") return 300;
  const ms = e.ms ?? 0;
  return Math.max(450, Math.min(3000, ms));
}

/* ---------- The latency waterfall of the newest burst of events ---------- */

export type WaterBar = { stage: string; group: StageGroup; model: string | null; start: number; ms: number; running: boolean };

/** The newest burst of timed events as a waterfall: events closer than `gap` ms belong to one burst
 *  (one question's stages). Each event ends at its `at_ms` and began `ms` before. Active stages are
 *  added as running bars, ending now. Background writer events are left out of a question's burst.
 *  Returns the bars from the burst's start and its span; empty when nothing was timed. */
export function waterfall(events: LiveEvent[], active: LiveStage[] = [], now = 0, gap = 1500): { bars: WaterBar[]; span: number; at: number | null } {
  const timed = events.filter((e) => e.ms !== null && e.ms >= 0 && stageGroup(e.stage, e.kind) !== "writer").sort((a, b) => a.at_ms - b.at_ms);
  const burst: LiveEvent[] = [];
  for (let i = timed.length - 1; i >= 0; i--) {
    const e = timed[i];
    const prev = burst[burst.length - 1];
    if (prev) {
      const prevStart = prev.at_ms - (prev.ms ?? 0);
      if (prevStart - e.at_ms > gap) break;
    }
    burst.push(e);
  }
  burst.reverse();
  const run = active.filter((a) => stageGroup(a.stage) !== "writer");
  const starts = [...burst.map((e) => e.at_ms - (e.ms ?? 0)), ...run.map((a) => a.since_ms ?? (now && a.elapsed_ms !== null ? now - a.elapsed_ms : Number.NaN))].filter(Number.isFinite);
  if (!starts.length) return { bars: [], span: 0, at: null };
  const t0 = Math.min(...starts);
  const bars: WaterBar[] = burst.map((e) => ({ stage: e.stage, group: stageGroup(e.stage, e.kind), model: e.model, start: e.at_ms - (e.ms ?? 0) - t0, ms: e.ms ?? 0, running: false }));
  for (const a of run) {
    const s = a.since_ms ?? (now && a.elapsed_ms !== null ? now - a.elapsed_ms : null);
    if (s === null) continue;
    bars.push({ stage: a.stage, group: stageGroup(a.stage), model: a.model, start: s - t0, ms: Math.max(0, a.elapsed_ms ?? (now ? now - s : 0)), running: true });
  }
  const span = Math.max(1, ...bars.map((b) => b.start + b.ms));
  const at = burst.length ? burst[burst.length - 1].at_ms : null;
  return { bars, span, at };
}

/** The live model that runs a released model, by its card id or its role. */
export function liveModelFor(models: LiveModel[] | null | undefined, id: string, roleOf?: (id: string) => string | null | undefined): LiveModel | null {
  const list = models ?? [];
  const exact = list.find((m) => m.card === id);
  if (exact) return exact;
  // 2026-11-26: the upstream weights ids (qwen3-0.6b) do not carry the role name; the role map does.
  const role = roleOf?.(id);
  if (role) {
    const roles = role.toLowerCase().split(/\s*\+\s*/);
    const byRole = list.find((m) => roles.includes(m.role.toLowerCase()));
    if (byRole) return byRole;
  }
  const lower = id.toLowerCase();
  return list.find((m) => lower.includes(m.role.toLowerCase())) ?? null;
}

/** The status word of a live model: executing, resident or not held. */
export function modelState(m: LiveModel | null): "executing" | "resident" | "not held" | "not reported" {
  if (!m) return "not reported";
  if (m.executing) return "executing";
  return m.resident ? "resident" : "not held";
}

/** The one line the header shows: what runs now, or Idle / Listening. */
export function liveLine(r: LiveReply | null): string {
  if (!r) return "live state not reported";
  if (engineServing(r) === false) return "No engine serving";
  const now = nowLine(r);
  if (now) return now;
  return r.state === "busy" ? "busy" : "Idle / Listening";
}

/** What runs now in one line, or null when nothing does: the stage in flight (a stage of the
 *  call before the call itself), else the call by its tool, else the MicroModels executing, else
 *  the idle Writer's open session. */
export function nowLine(r: LiveReply | null): string | null {
  if (!r) return null;
  const staged = r.active.filter((a) => stageGroup(a.stage) !== "other");
  const a = staged[staged.length - 1];
  if (a) return `${GROUP_NAMES[stageGroup(a.stage)]} · ${a.stage}${a.model ? ` · ${a.model}` : ""}${staged.length > 1 ? ` +${staged.length - 1}` : ""}`;
  const call = r.active.find((x) => x.stage === "call");
  if (call) return `call${call.label ? ` · ${call.label}` : ""}`;
  if (r.active.length) return r.active[0].stage;
  const exec = r.models.filter((m) => m.executing).map((m) => m.role);
  if (exec.length) return `${exec.join(", ")} executing`;
  if (r.writer?.in_session) return "idle Writer writing";
  return null;
}
