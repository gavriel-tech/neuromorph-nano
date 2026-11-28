"use client";

import { useCallback, useSyncExternalStore } from "react";

/** The engine serves this interface and its API from the same place (127.0.0.1:4870).
 *  Under `npm run dev` (port 3000) the API is reached on the engine's port unless overridden
 *  by `window.__NANO_API_BASE__` or `NEXT_PUBLIC_API_BASE`. */
function base(): string {
  if (typeof window !== "undefined") {
    const winBase = (window as { __NANO_API_BASE__?: string }).__NANO_API_BASE__?.trim();
    if (winBase) return winBase.replace(/\/+$/, "");
  }
  const envBase = typeof process !== "undefined" ? process.env?.NEXT_PUBLIC_API_BASE?.trim() : "";
  if (envBase) return envBase.replace(/\/+$/, "");
  if (typeof window === "undefined") return "";
  return window.location?.port === "3000" ? "http://127.0.0.1:4870" : "";
}

/** The engine's origin for requests that do not go through `useApi` (the live event stream). */
export function apiBase(): string {
  return base();
}

/** A precision width in bits, or "default" for the part's own default. */
export type Width = number | "default";
/** `layers` is optional: the engine may add it from the model card. */
/** `precision_source`: where `precision` comes from ("serving process", or the saved setting that
 *  applies at the next start). */
export type Model = { id: string; bytes: number; card: boolean; main_file: string | null; precision: Width; layers?: number; heads?: number; precision_source?: string };
/** A model the engine holds right now (optional, asked of the engine). */
export type LoadedModel = { id: string; card?: string; roles?: string[]; precision?: Width; effective_bits?: number; bytes?: number; ram_bytes?: number; gpu_bytes?: number; device?: string; layers_on_card?: number[]; project?: string };
export type Precision = { base: Width; writer: Width; reranker: Width; embedder: Width };
export type Status = {
  name: string;
  version: string;
  root: string;
  models_dir: string;
  models: Model[];
  parts: string[];
  precision: Precision;
  memory_dir: string;
  time_ms: number;
  /** Optional, asked of the engine: models held now, resident memory, effective precision. */
  loaded?: LoadedModel[];
  resident_bytes?: number;
  gpu_bytes?: number;
  /** API version 2: serving processes with a fresh live file, and what the most active one holds. */
  api?: number;
  serving?: number;
  substrate?: { chunks?: number; files?: number } | null;
};
export type Port = { name: string; kind: string; required: boolean };
export type Part = {
  id: string;
  kind: string;
  version: number;
  parent?: string | null;
  ports_in: Port[];
  ports_out: Port[];
  deps: string[];
  config_default: Record<string, unknown>;
  subcomponents: { id: string; version: number; hash: string }[];
  source?: string;
  hash: string;
};
/** A learned adapter cartridge, by its label (`nm-cartridge/1`, `nmlearn::cartridge::Label`). */
export type Cartridge = {
  schema: string;
  id: string;
  kind: string;
  data_class: string;
  base: string;
  base_sha256: string;
  components: string[];
  scopes: string[];
  quantization: string;
  description: string;
  version: number;
  check: string;
  date: string;
  loaded: boolean;
  /** Optional, asked of the engine. */
  bytes?: number;
  loaded_bytes?: number;
  model?: string;
  class?: "learning" | "preference" | "self";
  /** The cartridge's family when the engine names it: fundamentals, learning or preference. */
  family?: string;
  layers?: [number, number] | null;
  verified?: boolean;
  /** API 5: a cartridge of the library carries its place in the library. Some builds name this
   *  field `book`; both are read, and it is always shown as a cartridge. */
  library?: LibraryMark;
  book?: LibraryMark;
};
export type LibraryMark = { key?: string; domain?: string; target?: string; trial?: boolean; in_use?: string; stacks?: string[]; hash?: string; evidence_rows?: number };

/* ---------- API 4 to 7: the session in play, the evidence ledger, outcomes, the library, the training store ---------- */

/** An event of Claude Code's work seen through the hooks. */
export type HookEvent = { at?: number; event?: string; kind?: string; tool?: string; place?: string; ok?: boolean };
/** One finding of the ledger about a pack or a prime. */
export type Join = { rule: string; state: string; strength?: string | null; steps_after?: number; event?: HookEvent | null; why?: string; detail?: Record<string, unknown> };
/** The ledger's view of one pack (or prime) in the session state. */
export type PackView = { id?: string; at?: number; joined_by?: string; cited?: string[]; window?: string; closed?: string | null; in_ledger?: boolean; joins?: Join[] };
export type SessionFile = { path: string; reads?: number; read_lines?: [number, number][]; edits?: number; writes?: number; edit_lines?: [number, number][]; last?: number; sensitive?: boolean };
export type SessionCommand = { t: number; kind?: string; class?: string; cmd?: string; pkg?: string; ok?: boolean; exit?: number; tests?: [number, number] };
export type SessionServed = { t: number; tool?: string; ask?: string; episode?: string; pack?: PackView | null };
/** A prime given in the session: a short note the Nano added to Claude Code's context unasked. */
export type SessionPrime = { t: number; event?: string; kind?: string; id?: string; places?: string[]; from?: string[]; ledger?: PackView | null };
export type SessionState = {
  session: string;
  first_ms: number;
  last_ms: number;
  records: number;
  model?: string | null;
  starts?: { t: number; source: string }[];
  compactions?: number;
  task?: { t: number; sha?: string; chars?: number; text?: string | null } | null;
  prompts?: number;
  files?: SessionFile[];
  commands?: SessionCommand[];
  open?: { failing_checks?: unknown[]; tool_failures?: unknown[]; unchecked_changes?: unknown[] };
  served?: SessionServed[];
  /** API 7; API 6 gave the same list as `nudges`. Read through `primesOf`. */
  primes?: SessionPrime[];
  nudges?: SessionPrime[];
};
export type SessionReply = { project: string; sessions?: { session: string; first_ms: number; last_ms: number; records: number }[]; session: SessionState | null };

/** A ledger row: `served`, `primed` (API 6 wrote `nudged`), `evidence` or `outcome`; other fields by kind. */
export type LedgerRow = {
  row: string;
  seq?: number;
  pack?: string;
  session?: string;
  at?: number;
  t?: number;
  rule?: string;
  state?: string;
  strength?: string | null;
  steps_after?: number;
  event?: HookEvent | null;
  closed?: string;
  kind?: string;
  hook?: string;
  places?: unknown[];
  cited?: string[];
  tool?: string;
  joined_by?: string;
  why?: string;
  [k: string]: unknown;
};
export type Ledger = {
  project: string;
  chain?: { rows: number; intact: boolean; fault: string | null; torn: string | null };
  packs?: { served: number; window_open: number };
  /** API 7; API 6 gave the same counts as `nudges`. */
  primes?: PrimeCounts;
  nudges?: PrimeCounts;
  outcomes?: number;
  states?: Record<string, number>;
  counts?: Record<string, Record<string, number>>;
  rows?: LedgerRow[];
  pending?: { pack: string; session?: string; at?: number; joined_by?: string; joins?: Join[] }[];
};
export type PrimeCounts = { given?: number; window_open?: number; followed?: number; passed_over?: number };
export type Outcome = {
  record: string;
  kind: "answered" | "fixed" | string;
  question?: string | null;
  failure?: { check?: string; code?: string; file?: string; line?: number; msg?: string } | null;
  places?: { place: string; fresh: boolean }[];
  verified?: number;
  refuted?: number;
  strength?: string | null;
  standing?: "verified" | "refuted" | string;
  in_memory?: boolean;
};
export type Outcomes = { project: string; root_known?: boolean; outcomes: Outcome[] };

/** A cartridge as the library's catalogue lists it. */
export type LibCartridge = { id: string; version?: number; family?: string; domain?: string; model?: string; layers?: [number, number] | null; kind?: string; target?: string; hash?: string; trial?: boolean; description?: string; evidence?: unknown[]; files?: unknown[] };
export type LibStack = { id: string; version?: number; use: string; piece?: { name?: string; identity?: string; kind?: string } | string | null; cartridges: string[]; hash?: string; trial?: boolean; description?: string };
export type LibUse = { use: string; piece?: string | null; stack?: string | null; trial?: boolean; why?: string };
export type LibInUse = {
  machine?: { pieces?: { kind?: string; identity?: string; name?: string; family_key?: string }[] };
  uses?: LibUse[];
  loaded?: { key: string; trial?: boolean; hardware?: string; refused?: string | null }[];
  set?: Record<string, { value: string; from: string }>;
  trial?: { key: string; value: string; from: string }[];
  options?: string[];
  refused?: string[];
};
export type Library = {
  exists?: boolean;
  error?: string;
  catalogue?: { cartridges?: LibCartridge[]; books?: LibCartridge[]; stacks?: LibStack[]; active?: Record<string, string> };
  in_use?: LibInUse;
  last_start?: LibInUse | null;
};
/** One example of the training store (API 7), as `/api/examples` lists it. */
export type Example = {
  id: string;
  family?: string;
  verdict?: string;
  strength?: string | null;
  decided_by?: string;
  split?: string;
  domains?: string[];
  anchor?: string | null;
  path?: string;
  at?: number;
  t?: number;
  candidates?: { place?: string; served?: boolean; rank?: number | null; label?: string; strength?: string | null }[];
  [k: string]: unknown;
};
/** The training store of a project (API 7): what is kept for fitting cartridges later. */
export type Examples = {
  project: string;
  setting?: string;
  memory?: string;
  families?: { learning?: number; preference?: number; fundamentals?: { hardware_rows?: number; references?: number } | number };
  judged_waiting?: number;
  verdicts?: Record<string, number>;
  strengths?: Record<string, number>;
  splits?: Record<string, number>;
  domains?: Record<string, number>;
  labels?: Record<string, number>;
  engines?: Record<string, number>;
  preference_kinds?: Record<string, number>;
  bytes?: number;
  examples?: Example[];
};
/** The 3-level Continuous Learning Engine & Supervisor state (`/api/learning`). */
export type ContinualLearning = {
  project: string;
  learning?: string;
  cycle_mode?: string;
  version: number;
  cycles_run: number;
  last_cycle_ms: number;
  levels?: {
    experiential?: { episodes?: number; cycle?: string };
    knowledge?: { patterns?: number; semantic?: number; procedural?: number; cycle?: string };
    behavioral?: { policies?: number; preferences?: number; retrieval_adapted?: boolean };
  };
  patterns?: {
    id: string;
    category: string;
    kind: string;
    observation: string;
    generalized_lesson: string;
    recommended_procedure: string[];
    evidence_ids: string[];
    confidence: number;
    scope: string;
    status: string;
  }[];
  policies?: {
    id: string;
    name: string;
    condition: string;
    applies_to: string;
    does_not_apply_to: string;
    previous_behavior: string;
    observed_problem: string;
    new_behavior: string[];
    expected_improvement: string;
    confidence: number;
    evidence_count: number;
    status: string;
  }[];
  retrieval?: {
    semantic_weight: number;
    lexical_weight: number;
    structural_weight: number;
    recency_weight: number;
    relationship_weight: number;
    similarity_threshold: number;
    retrieval_depth: number;
    retrieved_total: number;
    used_total: number;
    irrelevant_total: number;
    missing_essential_total: number;
  };
  /** 2026-11-25: a figure is null until it is measured, never a starting default. */
  baseline_eval?: Record<string, number | null>;
  current_eval?: Record<string, number | null>;
  metrics?: {
    learning_retention: number | null;
    correction_recurrence: number | null;
    retrieval_precision: number | null;
    retrieval_recall: number | null;
    context_fidelity: number | null;
    reconstruction_accuracy: number | null;
    policy_effectiveness: number | null;
    transfer_performance: number | null;
    regression_rate: number | null;
    learning_efficiency: number | null;
    corrections_total: number;
    recurrences_total: number;
    last_recurrence_stage: string;
  };
  cartridges?: {
    id: string;
    domain: string;
    version: number;
    status: string;
    entries: number;
    evidence_count: number;
    confidence: number | null;
    summary: string;
    updated_ms: number;
    archived: boolean;
  }[];
  rollback_versions?: number[];
  /** Every figure the engine has measured, with the count it rests on. */
  measures?: { key: string; label: string; what: string; value: number; n: number; baseline: number | null }[];
  /** Stored training examples the engine has not taken in yet. */
  backlog?: number;
};
/** One row of `memory/audit.jsonl`. Fields beyond these depend on the row's kind. */
export type AuditRow = {
  id: string;
  kind: string;
  time_ms: number;
  label?: string;
  tool?: string;
  mode?: string;
  via?: string;
  ms?: number;
  ok?: boolean;
  error?: string;
  project?: string;
  parts?: string[];
  key?: string;
  from?: string;
  to?: string;
  undo_of?: string;
  /** Optional, asked of the engine: "running" while the call is in progress. */
  state?: "running" | "done" | "stale";
  /** Optional, asked of the engine: each part in run order with its timing. */
  steps?: { part: string; start_ms?: number; ms?: number | null; us?: number; timed?: boolean; inside?: string; cells?: string[]; counts?: Record<string, number>; layers?: [number, number]; shared?: string; job?: string; [k: string]: unknown }[];
  [k: string]: unknown;
};
export type Audit = { audit: AuditRow[]; episodes: unknown[] };
export type Project = {
  key: string;
  name?: string;
  custom_name?: string | null;
  description?: string | null;
  custom?: boolean;
  root_exists?: boolean;
  is_scratch?: boolean;
  root: string | null;
  first_ms: number | null;
  last_ms: number | null;
  episodes: number;
  bytes: number;
  dir: string;
  serving?: boolean;
  tiers?: Record<string, number>;
  tiers_overlap?: number;
  /** Read when an engine build reports them: the meaning index kept on disk and the memory
   *  layout's migration. */
  dense?: { rows?: number; bytes?: number } | null;
  migrated?: unknown;
  migration?: unknown;
};
export type Projects = { dir: string; projects: Project[]; current?: string | null; serving?: { project?: string; project_key?: string }[] };

/** Human-readable display name for a project (custom name or folder basename, falling back to key). */
export function projectLabel(p: Pick<Project, "key" | "name"> | null | undefined): string {
  if (!p) return "nano";
  const n = p.name?.trim();
  return n ? n : p.key;
}

/** Resolves a project key to its display name from the loaded projects list. */
export function projectLabelByKey(projects: Projects | null | undefined, key: string | null | undefined): string {
  if (!key || key === "nano") return "The Nano itself";
  const found = projects?.projects?.find((p) => p.key === key);
  return found ? projectLabel(found) : key;
}

export type PrimeSchedule = {
  days?: string[];
  after?: string;
  before?: string;
};

export type BuiltinPrime = {
  id: string;
  label: string;
  trigger: string;
  summary: string;
  enabled: boolean;
};

export type CustomPrime = {
  id: string;
  name: string;
  text: string;
  enabled: boolean;
  trigger: string;
  every: number;
  interval_min: number;
  schedule: PrimeSchedule | null;
  repeat: string;
  max: number;
  glob: string | null;
  error: string | null;
  error_count: number;
  loop_id: string | null;
  by: string;
  fired?: number;
  last_ms?: number;
};

export type PrimePreset = {
  name: string;
  description: string;
  builtin: boolean;
  mode: string;
  builtins: Record<string, boolean>;
  custom: CustomPrime[];
};

export type PrimesReply = {
  project: string;
  mode: string;
  builtins: BuiltinPrime[];
  custom: CustomPrime[];
  presets: PrimePreset[];
  triggers: string[];
  repeats: string[];
  counts: PrimeCounts | null;
  loops: { id?: string; name?: string; prompt?: string; enabled?: boolean; status?: string; iterations?: number; max?: number; condition?: string }[];
  steps: { id?: string; title?: string; status?: string }[];
  reminders: { id?: string; text?: string; trigger?: string; enabled?: boolean }[];
  error?: string;
};
export type Control = {
  key: string;
  group?: string;
  label: string;
  help: string;
  desc?: string;
  kind?: "enum" | "integer";
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  values: string[];
  default: string;
  value: string;
  source: string;
  wired: boolean;
  applies: string;
  /** On `mcp.enabled_tools`: every MCP tool the engine has, in its coding priority order. */
  all_tools?: string[];
};

export type PairedDevice = {
  id: string;
  name: string;
  paired_ms: number;
  last_seen_ms: number;
  revoked: boolean;
  revoked_ms: number;
};

export type DevicesReply = {
  network: "off" | "view" | "control" | string;
  addresses: string[];
  devices: PairedDevice[];
  code_open: boolean;
  code_expires_ms: number;
  locked_until_ms: number;
  encryption: string;
};

export type Loaded<T> = { data: T | null; error: string | null; loading: boolean; at: number | null };

/* ---------- One shared store: every path is fetched by one poller, whatever reads it. ----------
 * Pages read the cache at once (navigation is instant), and an unchanged reply keeps the same
 * object, so nothing re-renders while the engine is idle. */
type Entry = { data: unknown; error: string | null; at: number | null; sig: string };
const store = new Map<string, Entry>();
const listeners = new Map<string, Set<() => void>>();
const wants = new Map<string, Map<symbol, number>>();
const timers = new Map<string, ReturnType<typeof setTimeout>>();
const inflight = new Map<string, Promise<void>>();
const rerun = new Set<string>();

function emit(path: string) {
  listeners.get(path)?.forEach((f) => f());
}

/** Fetches one path now and stores it; the stored object is replaced only when the reply changed
 *  (the status clock `time_ms` is left out of that comparison). If a fetch is already in flight
 *  for `path`, queues one follow-up refresh so post-write invalidations are never dropped. */
export function refresh(path: string): Promise<void> {
  const cur = inflight.get(path);
  if (cur) {
    rerun.add(path);
    return cur.then(() => inflight.get(path) ?? Promise.resolve());
  }
  const run = (async () => {
    try {
      const r = await fetch(base() + path, { cache: "no-store" });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const text = await r.text();
      const sig = text.replace(/"time_ms"\s*:\s*\d+,?/g, "");
      const prev = store.get(path);
      if (!prev || prev.sig !== sig || prev.error) {
        store.set(path, { data: JSON.parse(text), error: null, at: Date.now(), sig });
        emit(path);
      } else {
        prev.at = Date.now();
      }
    } catch (e) {
      const prev = store.get(path);
      const error = e instanceof Error ? e.message : String(e);
      if (prev?.error !== error) {
        store.set(path, { data: prev?.data ?? null, error, at: prev?.at ?? null, sig: prev?.sig ?? "" });
        emit(path);
      }
    } finally {
      inflight.delete(path);
      if (rerun.delete(path)) void refresh(path);
    }
  })();
  inflight.set(path, run);
  return run;
}

/** Refreshes every stored path whose name starts with `prefix` (after a write). */
export function refreshAll(prefix: string): Promise<void> {
  const jobs: Promise<void>[] = [];
  for (const p of store.keys()) if (p.startsWith(prefix)) jobs.push(refresh(p));
  return Promise.all(jobs).then(() => undefined);
}

/** Reads the cached entry for `path` without mounting a React hook. */
export function readCache<T>(path: string): Loaded<T> {
  const entry = store.get(path);
  return {
    data: (entry?.data as T) ?? null,
    error: entry?.error ?? null,
    loading: !entry,
    at: entry?.at ?? null,
  };
}

function schedule(path: string) {
  const t = timers.get(path);
  if (t) clearTimeout(t);
  timers.delete(path);
  const w = wants.get(path);
  if (!w || w.size === 0) return;
  const every = Math.min(...w.values());
  if (!Number.isFinite(every)) return;
  timers.set(
    path,
    setTimeout(async () => {
      if (typeof document === "undefined" || !document.hidden) await refresh(path);
      schedule(path);
    }, every),
  );
}

/** Reads one API path, refreshing every `everyMs` when given. Readers of the same path share
 *  one request; the fastest reader sets the pace. */
export function useApi<T>(path: string | null, everyMs?: number): Loaded<T> {
  const subscribe = useCallback(
    (cb: () => void) => {
      // No path: nothing is asked (an endpoint this engine build does not have).
      if (path === null) return () => {};
      const me = Symbol(path);
      let set = listeners.get(path);
      if (!set) listeners.set(path, (set = new Set()));
      set.add(cb);
      let w = wants.get(path);
      if (!w) wants.set(path, (w = new Map()));
      w.set(me, everyMs ?? Infinity);
      const e = store.get(path);
      if (!e || e.at === null || Date.now() - e.at > (everyMs ?? 30_000)) void refresh(path);
      schedule(path);
      return () => {
        set.delete(cb);
        w.delete(me);
        schedule(path);
      };
    },
    [path, everyMs],
  );
  const entry = useSyncExternalStore(
    subscribe,
    () => (path === null ? undefined : store.get(path)),
    () => undefined,
  );
  if (path === null) return { data: null, error: null, loading: false, at: null };
  return { data: (entry?.data as T) ?? null, error: entry?.error ?? null, loading: !entry, at: entry?.at ?? null };
}

/** The API version of the running engine, or null until `/api/status` answers. */
export function useApiLevel(): number | null {
  const { data } = useApi<Status>("/api/status", 5000);
  if (!data) return null;
  return typeof data.api === "number" ? data.api : 1;
}

export function bytes(n: number): string {
  if (n >= 1 << 30) return `${(n / (1 << 30)).toFixed(2)} GB`;
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(0)} MB`;
  if (n >= 1 << 10) return `${(n / (1 << 10)).toFixed(0)} KB`;
  return `${n} B`;
}

/** Precision as shown: "16-bit", or "default" when the part runs at its own default. */
export function bits(w: Width | undefined): string {
  if (w === undefined) return "";
  return w === "default" ? "default" : `${w}-bit`;
}

/** "12s ago", "4m ago", from a time in ms and the time now. */
export function ago(ms: number, now: number): string {
  const s = Math.max(0, Math.round((now - ms) / 1000));
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

/** The engine's local write token. The engine places it in every page it serves
 *  (`<meta name="nano-token">`); it is read once, when this module loads, because client-side
 *  navigation may replace the head. Under `npm run dev` the pages come from Next.js, so the token
 *  is fetched from the engine, which answers only when started with `ui --dev`. */
let pageToken: string | null =
  typeof document === "undefined" ? null : document.querySelector('meta[name="nano-token"]')?.getAttribute("content") ?? null;
let refreshingToken: Promise<string | null> | null = null;

function extractMetaToken(html: string): string | null {
  const m1 = html.match(/<meta\b[^>]*\bname=["']nano-token["'][^>]*\bcontent=["']([^"']+)["'][^>]*>/i);
  if (m1?.[1]) return m1[1];
  const m2 = html.match(/<meta\b[^>]*\bcontent=["']([^"']+)["'][^>]*\bname=["']nano-token["'][^>]*>/i);
  return m2?.[1] ?? null;
}

async function fetchFreshToken(): Promise<string | null> {
  if (refreshingToken) return refreshingToken;
  refreshingToken = (async () => {
    const origin = base();
    if (origin !== "") {
      const r = await fetch(origin + "/api/token", { cache: "no-store" }).catch(() => null);
      if (r?.ok) {
        const data = (await r.json().catch(() => null)) as { token?: unknown } | null;
        if (data && typeof data.token === "string" && data.token) {
          pageToken = data.token;
          return pageToken;
        }
      }
    }
    if (typeof window !== "undefined" || origin !== "") {
      const pagePath = typeof window !== "undefined" ? window.location?.pathname || "/" : "/";
      const hr = await fetch(origin + pagePath, { cache: "no-store" }).catch(() => null);
      if (hr?.ok) {
        const html = await hr.text().catch(() => "");
        const tok = extractMetaToken(html);
        if (tok) {
          pageToken = tok;
          if (typeof document !== "undefined") {
            document.querySelector('meta[name="nano-token"]')?.setAttribute("content", tok);
          }
          return pageToken;
        }
      }
    }
    return null;
  })().finally(() => {
    refreshingToken = null;
  });
  return refreshingToken;
}

async function token(): Promise<string> {
  if (pageToken) return pageToken;
  if (typeof document !== "undefined") {
    const meta = document.querySelector('meta[name="nano-token"]')?.getAttribute("content");
    if (meta) {
      pageToken = meta;
      return pageToken;
    }
  }
  const fresh = await fetchFreshToken();
  if (fresh) return fresh;
  if (base() !== "") {
    throw new Error("the engine gives the development server its write token only under `neuromorph-nano ui --dev`");
  }
  throw new Error("no write token in this page; open the interface from the engine (neuromorph-nano ui)");
}

async function parseError(r: Response): Promise<Error> {
  const text = await r.text().catch(() => "");
  let parsed: { error?: unknown; message?: unknown } | null = null;
  try {
    parsed = text ? (JSON.parse(text) as { error?: unknown; message?: unknown }) : null;
  } catch {
    parsed = null;
  }
  const msg =
    (parsed && typeof parsed.error === "string" && parsed.error) ||
    (parsed && typeof parsed.message === "string" && parsed.message) ||
    text.trim() ||
    `${r.status} ${r.statusText}`;
  return new Error(msg);
}

/** Sends one write to the engine (controls, undo). Writes are accepted only from this machine,
 *  with the engine's local token. Automatically refreshes the token once on 403. */
export async function post<T>(path: string, body: unknown): Promise<T> {
  const sendOnce = (tok: string) =>
    fetch(base() + path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Nano-Token": tok },
      body: JSON.stringify(body),
    });

  const tok = await token();
  let r = await sendOnce(tok);
  if (r.status === 403) {
    pageToken = null;
    const fresh = await fetchFreshToken();
    if (fresh && fresh !== tok) {
      r = await sendOnce(fresh);
    }
  }
  if (!r.ok) throw await parseError(r);
  const text = await r.text();
  return (text ? JSON.parse(text) : null) as T;
}

/* ---------- Confirmed deletes (2026-11-26) ----------
 * Every destructive write goes in two steps. Step 1 posts the body without `confirm_token`: the
 * engine deletes nothing and answers with a preview (what would go, its size, whether it can be
 * undone), a single-use token bound to that exact request, its expiry and the word the user must
 * type. Step 2 posts the same body with the token; only then is anything deleted. */

export type DeletePreviewItem = { label: string; path: string | null; bytes: number | null; files: number | null };
export type DeletePreview = {
  title: string;
  items: DeletePreviewItem[];
  bytes: number | null;
  files: number | null;
  undo: string;
  note: string | null;
  confirm_token: string;
  expires_ms: number;
  confirm_word: string;
  next: string | null;
};

/** What an engine build without confirmed deletes is told. */
export const NO_CONFIRMED_DELETES = "this engine build does not support confirmed deletes; update it";

/** Whether an `undo` line says the delete is permanent. */
export function cannotUndo(undo: string | null | undefined): boolean {
  return !undo || /cannot|can't|can not|not be undone|no undo|permanent/i.test(undo);
}

/** The body of step 1: the request as given, without any confirmation. */
export function previewBody(body: Record<string, unknown>): Record<string, unknown> {
  const rest = { ...body };
  delete rest.confirm;
  delete rest.confirm_token;
  return rest;
}

/** The body of step 2: the same request plus the token (and `confirm: true`, which the engine
 *  accepts alongside). */
export function confirmBody(body: Record<string, unknown>, token: string): Record<string, unknown> {
  return { ...previewBody(body), confirm_token: token, confirm: true };
}

const numOrNull = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const strOrNull = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v : null);

/** A step-1 reply, checked. Null when it carries no token (an engine build that would have
 *  deleted at once, or refused). */
export function parseDeletePreview(raw: unknown): DeletePreview | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const o = raw as Record<string, unknown>;
  const token = strOrNull(o.confirm_token);
  const word = strOrNull(o.confirm_word);
  if (!token || !word) return null;
  const p = o.preview && typeof o.preview === "object" && !Array.isArray(o.preview) ? (o.preview as Record<string, unknown>) : {};
  const items: DeletePreviewItem[] = [];
  for (const it of Array.isArray(p.items) ? p.items : []) {
    if (!it || typeof it !== "object") continue;
    const x = it as Record<string, unknown>;
    items.push({ label: strOrNull(x.label) ?? strOrNull(x.path) ?? "item", path: strOrNull(x.path), bytes: numOrNull(x.bytes), files: numOrNull(x.files) });
  }
  return {
    title: strOrNull(p.title) ?? "Delete",
    items,
    bytes: numOrNull(p.bytes),
    files: numOrNull(p.files),
    undo: strOrNull(p.undo) ?? "cannot be undone",
    note: strOrNull(p.note),
    confirm_token: token,
    expires_ms: numOrNull(o.expires_ms) ?? 0,
    confirm_word: word,
    next: strOrNull(o.next),
  };
}

/** Whether the engine has confirmed deletes. An older build would delete at step 1 for some
 *  endpoints (it ignores the missing token), so it is asked first, through a read only the new
 *  builds serve (`GET /api/memory/items`): "unknown endpoint" means an older build. */
let confirmedDeletes = false;
async function supportsConfirmedDeletes(): Promise<boolean> {
  if (confirmedDeletes) return true;
  const r = await fetch(base() + "/api/memory/items?project=nano&limit=1", { cache: "no-store" });
  if (r.status === 404) {
    const text = await r.text().catch(() => "");
    if (/unknown endpoint/i.test(text) || !text.trim()) return false;
  }
  confirmedDeletes = true;
  return true;
}

/** Step 1: asks the engine what `body` would delete. Nothing is deleted. */
export async function previewDelete(path: string, body: Record<string, unknown>): Promise<DeletePreview> {
  if (!(await supportsConfirmedDeletes())) throw new Error(NO_CONFIRMED_DELETES);
  const raw = await post<unknown>(path, previewBody(body));
  const p = parseDeletePreview(raw);
  if (!p) throw new Error(NO_CONFIRMED_DELETES);
  return p;
}

/** Step 2: deletes, with the token step 1 gave for this exact request. */
export async function confirmDelete<T>(path: string, body: Record<string, unknown>, token: string): Promise<T> {
  return post<T>(path, confirmBody(body, token));
}

/** Clears internal caches, timers, and cached token state for unit tests. */
export function _resetApiForTest(): void {
  for (const t of timers.values()) clearTimeout(t);
  timers.clear();
  wants.clear();
  listeners.clear();
  store.clear();
  inflight.clear();
  rerun.clear();
  pageToken = null;
  refreshingToken = null;
  confirmedDeletes = false;
}
