import "./register-ts.mjs";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

/* UI integration tests for the job-filter, comprehension and learning-cycle endpoints.
 *
 * The pages call `post()` / `useApi()` from src/lib/api.ts. These tests drive that client code
 * against an in-process engine double (`makeEngine`) that ports the request handling of
 * crates/nmcore/src/bin/ui.rs (`write_api`, `update_job_filter`, `start_job_filter_build`,
 * `comprehension_view`, `surgery_view`) and examples.rs (`learning_cycle_api`). The "contract"
 * tests below read the Rust and page sources and fail if the shapes, validators or error strings
 * the double relies on drift. A last group talks to a running engine and is skipped when none is
 * reachable. */

const api = await import("../src/lib/api.ts");

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (p) => readFileSync(resolve(HERE, p), "utf8");
const UI_RS = read("../../crates/nmcore/src/bin/ui.rs");
const EXAMPLES_RS = read("../../crates/nmcore/src/bin/examples.rs");
const LEDGER_RS = read("../../crates/nmcore/src/bin/ledger.rs");
const NMJSON_RS = read("../../crates/nmjson/src/json.rs");
const NMJSON_ERR_RS = read("../../crates/nmjson/src/error.rs");
const ENGINE_RS = read("../../crates/nmlearn/src/engine.rs");
// 2026-11-26: the Cartridges page is split into the page, its decks and its rail.
const CARTRIDGES_PAGE = ["../src/app/cartridges/page.tsx", "../src/components/CartridgeDecks.tsx", "../src/components/CartridgeBay.tsx", "../src/components/CartridgeRail.tsx"].map(read).join(" ");
// 2026-11-26: the Parameters page and its atlas, detail, list and rail.
const PARAMETERS_PAGE = ["../src/app/parameters/page.tsx", "../src/components/MapAtlas.tsx", "../src/components/MapDetail.tsx", "../src/components/MapList.tsx", "../src/components/MapRail.tsx", "../src/components/MapCore.tsx", "../src/components/MapsState.tsx"].map(read).join(" ");
// 2026-11-26: the Learning page and its panels (LearningPanels, MemoryPanels).
const LEARNING_PAGE = ["../src/app/learning/page.tsx", "../src/components/LearningPanels.tsx", "../src/components/MemoryPanels.tsx"].map(read).join(" ");
// 2026-11-26: the Memory page and its panels (MemoryPanels).
const MEMORY_PAGE = ["../src/app/memory/page.tsx", "../src/components/MemoryPanels.tsx"].map(read).join(" ");
const COMPREHENSION_LIB = read("../src/lib/comprehension.ts");

/* ---------- Messages the engine sends, copied from the Rust sources (pinned by the contract tests). ---------- */

const MSG = {
  slug: "role and job must be alphanumeric with '-' or '_'",
  layers: "layers must be 'all' or a numeric range/list like '12-27'",
  // 2026-11-26: the lab runs inside the program; this is the error only when neither it nor a nano-lab program can be reached.
  noLab: "the lab is not reachable: this program's path is unreadable and no nano-lab program was found",
  running: "a lab build is already running",
  write403: "writes are accepted only from this machine's interface, as JSON, with its local token",
  unknown: "unknown endpoint",
  noSnapshot: "no previous continuous learning snapshot available to roll back to",
};
/** `nmjson::Error::invalid(detail)` as Display prints it. */
const nmjsonInvalid = (detail) =>
  `prtdef-001: ${detail}. Supply a valid part card, cgraph or typed value. Recovery: fix then retry. Validation side effects: none; completed calls/log appends remain committed`;

/* ---------- Validators ported from the Rust (pinned by the contract tests). ---------- */

/** ui.rs `safe_slug`. */
const safeSlug = (s) => s.length > 0 && s.length <= 64 && /^[A-Za-z0-9_-]+$/.test(s);
/** ui.rs `safe_layers`. */
const safeLayers = (spec) => spec === "all" || (spec.length > 0 && spec.length <= 32 && /^[0-9,-]+$/.test(spec));
/** ledger.rs `project_memory` key rule (the folder must also exist). */
const projectKeyOk = (key) => key.length > 0 && !key.startsWith(".") && /^[A-Za-z0-9._-]+$/.test(key);

/* ---------- The engine double. ---------- */

const STATUS_TEXT = { 200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed", 500: "Internal Server Error" };
function reply(status, body, ctype = "application/json") {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return new Response(text, { status, statusText: STATUS_TEXT[status] ?? "", headers: { "Content-Type": ctype } });
}

function makeFilter(role, job, extra = {}) {
  return {
    cartridge: `${role}-jf-${job}`,
    job,
    base: role,
    model: `${role}-model`,
    base_cos: 0.9812,
    filtered_cos: 0.9934,
    base_max_diff: 3.1,
    filtered_max_diff: 1.4,
    parts: Array.from({ length: 16 }, (_, i) => ({ layer: 12 + i })),
    ...extra,
  };
}

function makeEngine(opts = {}) {
  const state = {
    token: opts.token ?? "tok-test",
    labPresent: opts.labPresent ?? true,
    build: opts.build ?? null,
    // library/parameter-maps/<role>/job-filter-<job>.json, by role then job.
    filters: opts.filters ?? {
      writer: { "code-slice": makeFilter("writer", "code-slice"), "docs-only": makeFilter("writer", "docs-only", { enabled: false }) },
      reranker: { triage: makeFilter("reranker", "triage") },
    },
    projects: new Set(opts.projects ?? ["nano", "alpha", "beta.v2"]),
    learning: opts.learning ?? { version: 3, cycles_run: 2, history: [1, 2] },
    comprehension: opts.comprehension ?? {
      nano: { total_notes: 7, note_counts: { outline: 1, file: 6 }, outline: "nano outline", routing_record: true, drone_default_heads: 4, drone_lens_sets: 8, base_cartridges: 3, base_backed_up: 2 },
      alpha: { total_notes: 2, note_counts: { file: 2 }, outline: null, routing_record: false, drone_default_heads: 4, drone_lens_sets: 8, base_cartridges: 3, base_backed_up: 2 },
    },
    failNext: null,
  };
  const calls = [];
  let audit = 0;

  const buildJson = () =>
    state.build ?? { status: "idle", role: "", job: "", started_ms: 0, finished_ms: 0, message: "" };

  function surgeryView() {
    const roles = {};
    for (const role of Object.keys(state.filters).sort()) {
      const jobs = Object.keys(state.filters[role]).sort();
      if (jobs.length === 0) continue;
      roles[role] = {
        super_weights: 0,
        ledger: [],
        job_filters: jobs.map((job) => {
          const jf = state.filters[role][job];
          const summary = {};
          for (const k of ["cartridge", "job", "base", "model", "base_cos", "filtered_cos", "base_max_diff", "filtered_max_diff", "enabled"]) {
            if (k in jf) summary[k] = jf[k];
          }
          if (!("enabled" in summary)) summary.enabled = true;
          if (Array.isArray(jf.parts)) summary.layers = jf.parts.length;
          return summary;
        }),
      };
    }
    return { roles, ledger: [], job_filter_build: buildJson() };
  }

  function learningView(key) {
    return {
      version: state.learning.version,
      cycles_run: state.learning.cycles_run,
      last_cycle_ms: 1_700_000_000_000 + state.learning.cycles_run,
      levels: { experiential: { episodes: 5, cycle: "immediate" }, knowledge: { patterns: 1, semantic: 0, procedural: 0, cycle: "consolidation" }, behavioral: { policies: 0, preferences: 0, retrieval_adapted: false } },
      rollback_versions: [...state.learning.history],
      project: key,
      learning: "on",
      cycle_mode: "idle",
    };
  }

  /** `write_api` for the three routes under test; `Err` becomes 400 `{"error": ...}`. */
  function writeApi(path, raw) {
    let v;
    try {
      if (raw === "" || raw === undefined) throw new Error("truncated JSON");
      v = JSON.parse(raw);
    } catch (e) {
      throw `body: ${nmjsonInvalid(e.message === "truncated JSON" ? "truncated JSON" : "invalid JSON")}`;
    }
    const isObj = v !== null && typeof v === "object" && !Array.isArray(v);
    const field = (k) => {
      if (!isObj) throw `${k}: ${nmjsonInvalid("expected JSON object")}`;
      if (!(k in v)) throw `${k}: ${nmjsonInvalid(`missing JSON key ${k}`)}`;
      if (typeof v[k] !== "string") throw `${k}: ${nmjsonInvalid("expected JSON string")}`;
      return v[k];
    };
    const optString = (k) => (isObj && typeof v[k] === "string" ? v[k] : null);
    const optInteger = (k) => (isObj && Number.isInteger(v[k]) && v[k] >= 0 ? v[k] : null);

    if (path === "/api/surgery/job-filter") {
      const role = field("role");
      const job = field("job");
      const action = field("action");
      if (!safeSlug(role) || !safeSlug(job)) throw MSG.slug;
      const jf = state.filters[role]?.[job];
      if (!jf) throw `job filter ${role}/${job} not found`;
      if (action === "delete") delete state.filters[role][job];
      else if (action === "enable" || action === "disable") jf.enabled = action === "enable";
      else throw `unknown action ${action} (use enable, disable, or delete)`;
      return { role, job, action, surgery: surgeryView() };
    }
    if (path === "/api/surgery/job-filter/build") {
      const role = field("role");
      const job = field("job");
      const layers = optString("layers") ?? "all";
      if (!safeSlug(role) || !safeSlug(job)) throw MSG.slug;
      if (!safeLayers(layers)) throw MSG.layers;
      if (!state.labPresent) throw MSG.noLab;
      if (state.build?.status === "running") throw MSG.running;
      state.build = { status: "running", role, job, started_ms: 1_700_000_000_000, finished_ms: 0, message: `Running held-out split-gate fit for ${role}/${job} (layers: ${layers})...` };
      return { role, job, layers, job_filter_build: buildJson() };
    }
    if (path === "/api/learning/cycle") {
      const key = field("project");
      const action = optString("action") ?? "cycle";
      const target = optInteger("version");
      if (!(key === "nano" || (projectKeyOk(key) && state.projects.has(key)))) throw `project: no project ${key}`;
      const L = state.learning;
      if (action === "rollback") {
        const idx = target === null ? L.history.length - 1 : L.history.lastIndexOf(target);
        if (idx < 0) throw `nmlearn.engine [policy] ${MSG.noSnapshot}`;
        L.version = L.history.splice(idx, 1)[0];
      } else {
        L.history.push(L.version);
        L.version += 1;
        L.cycles_run += 1;
      }
      return { ...learningView(key), audit_id: `a${++audit}` };
    }
    throw MSG.unknown;
  }

  /** `api()` for the read routes under test; anything else is 404 `unknown endpoint`. */
  function readApi(path, query) {
    if (path === "/api/surgery") return surgeryView();
    if (path === "/api/comprehension") {
      const raw = new URLSearchParams(query).get("project");
      const key = raw !== null && (raw === "nano" || (projectKeyOk(raw) && state.projects.has(raw))) ? raw : "nano";
      return state.comprehension[key] ?? state.comprehension.nano;
    }
    return undefined;
  }

  async function fetchImpl(url, init = {}) {
    const u = new URL(String(url), "http://127.0.0.1:4870");
    const method = init.method ?? "GET";
    const headers = init.headers ?? {};
    calls.push({ method, path: u.pathname, query: u.search.slice(1), url: String(url), headers, body: init.body });
    if (state.failNext) {
      const f = state.failNext;
      state.failNext = null;
      if (f === "network") throw new TypeError("fetch failed");
      return reply(f.status, f.body, f.ctype);
    }
    if (method === "POST") {
      const ctype = headers["Content-Type"] ?? "";
      const okToken = headers["X-Nano-Token"] === state.token;
      const tooBig = typeof init.body === "string" && init.body.length > 64 * 1024;
      if (!okToken || !ctype.startsWith("application/json") || tooBig) return reply(403, MSG.write403, "text/plain");
      try {
        return reply(200, writeApi(u.pathname, init.body));
      } catch (e) {
        return reply(400, { error: String(e) });
      }
    }
    if (method !== "GET" && method !== "HEAD") return reply(405, "GET, HEAD or POST", "text/plain");
    const v = readApi(u.pathname, u.search.slice(1));
    return v === undefined ? reply(404, { error: MSG.unknown }) : reply(200, v);
  }

  return { fetch: fetchImpl, state, calls, surgeryView };
}

/* ---------- Environment harness (same isolation as api-controls.test.mjs). ---------- */

async function withEngine(opts, fn) {
  const origFetch = globalThis.fetch;
  const origWindow = globalThis.window;
  const origDocument = globalThis.document;
  const origEnv = process.env.NEXT_PUBLIC_API_BASE;
  api._resetApiForTest();
  const engine = makeEngine(opts);
  delete globalThis.window;
  delete process.env.NEXT_PUBLIC_API_BASE;
  // The engine-served page carries its write token in <meta name="nano-token">.
  globalThis.document = {
    querySelector: (sel) =>
      sel === 'meta[name="nano-token"]' ? { getAttribute: (n) => (n === "content" ? engine.state.token : null), setAttribute() {} } : null,
  };
  globalThis.fetch = engine.fetch;
  try {
    await fn(engine);
  } finally {
    api._resetApiForTest();
    globalThis.fetch = origFetch;
    if (origWindow === undefined) delete globalThis.window;
    else globalThis.window = origWindow;
    if (origDocument === undefined) delete globalThis.document;
    else globalThis.document = origDocument;
    if (origEnv === undefined) delete process.env.NEXT_PUBLIC_API_BASE;
    else process.env.NEXT_PUBLIC_API_BASE = origEnv;
  }
}

const lastPost = (engine) => engine.calls.filter((c) => c.method === "POST").at(-1);
const ws = (s) => s.replace(/\s+/g, " ");

/* =====================================================================================
 * Contract: the Rust handlers and the pages still match what the double and the tests assume.
 * ===================================================================================== */

test("contract: ui.rs routes the three writes with the required fields the double enforces", () => {
  const src = ws(UI_RS);
  assert.match(src, /if path == "\/api\/surgery\/job-filter" \{ let role = field\("role"\)\?; let job = field\("job"\)\?; let action = field\("action"\)\?; return update_job_filter\(home, role, job, action(, by)?\);/);
  assert.match(src, /if path == "\/api\/surgery\/job-filter\/build" \{ let role = field\("role"\)\?; let job = field\("job"\)\?; let layers = v\.get\("layers"\)\.and_then\(Value::string\)\.unwrap_or\("all"\); return start_job_filter_build\(home, role, job, layers(, by)?\);/);
  assert.match(src, /if path == "\/api\/learning\/cycle" \{ let key = field\("project"\)\?; let action = v\.get\("action"\)\.and_then\(Value::string\)\.unwrap_or\("cycle"\); let target_ver = v\.get\("version"\)\.and_then\(Value::integer\)\.ok\(\);/);
  assert.match(src, /\.ok_or_else\(\|\| format!\("project: no project \{key\}"\)\)\?; return crate::examples::learning_cycle_api_at\(/);
  assert.match(src, /let field = \|k: &str\| v\.get\(k\)\.and_then\(Value::string\)\.map_err\(\|e\| format!\("\{k\}: \{e\}"\)\);/);
  // 2026-11-26: the body is parsed mutable so the two-step delete gate can confirm it.
  assert.match(src, /let mut v = nmjson::parse\(body\)\.map_err\(\|e\| format!\("body: \{e\}"\)\)\?;/);
  assert.match(src, /if let Some\(preview\) = crate::forget::gate\(home, path, &mut v\)\? \{ return Ok\(preview\); \}/);
  assert.match(src, /_ => return Err\("unknown endpoint"\.to_owned\(\)\),/);
});

test("contract: ui.rs serves GET /api/surgery and /api/comprehension, and has no GET /api/surgery/job-filter", () => {
  assert.match(UI_RS, /"\/api\/surgery" => Some\(surgery_view\(home\)\),/);
  assert.match(UI_RS, /"\/api\/comprehension" => Some\(comprehension_view\(home, query\)\),/);
  assert.doesNotMatch(UI_RS, /"\/api\/surgery\/job-filter(\/build)?" =>/, "job filters are listed through GET /api/surgery");
  assert.ok(UI_RS.includes(`None => send(stream, head_only, "404 Not Found", "application/json", b"{\\"error\\":\\"unknown endpoint\\"}", cors),`));
});

test("contract: ui.rs answers a failed write with 400 {error} and a refused write with 403 text/plain", () => {
  const src = ws(UI_RS);
  assert.ok(src.includes(`Err(e) => send_json_error(stream, "400 Bad Request", &e, cors),`));
  assert.ok(src.includes(`let body = obj(vec![("error", s(msg))]).encode().unwrap_or_default();`));
  assert.ok(src.includes(`"403 Forbidden", "text/plain", b"${MSG.write403}",`));
  assert.ok(src.includes("ctype.is_some_and(|c| c.starts_with(\"application/json\"))"));
  assert.ok(src.includes("const MAX_BODY: usize = 64 * 1024;"));
});

test("contract: slug, layers and project-key validators match the ported checks", () => {
  assert.ok(UI_RS.includes("!s.is_empty() && s.len() <= 64 && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))"));
  assert.ok(UI_RS.includes("spec == \"all\" || (!spec.is_empty() && spec.len() <= 32 && spec.chars().all(|c| c.is_ascii_digit() || matches!(c, ',' | '-')))"));
  assert.ok(LEDGER_RS.includes("!key.is_empty() && !key.starts_with('.') && key.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))"));
  assert.ok(LEDGER_RS.includes(`if key == "nano" {`));
  assert.ok(ws(UI_RS).includes("if !safe_slug(role) || !safe_slug(job) { return Err(\"role and job must be alphanumeric with '-' or '_'\".to_owned()); }"));
});

test("contract: every engine error string the tests expect exists verbatim in the Rust", () => {
  for (const m of [MSG.slug, MSG.layers, MSG.noLab, MSG.running]) assert.ok(UI_RS.includes(m), `ui.rs lost: ${m}`);
  assert.ok(UI_RS.includes('Err(format!("job filter {role}/{job} not found"))'));
  assert.ok(UI_RS.includes('Err(format!("unknown action {other} (use enable, disable, or delete)"))'));
  assert.ok(ENGINE_RS.includes(MSG.noSnapshot));
  assert.ok(NMJSON_RS.includes('format!("missing JSON key {key}")'));
  assert.ok(NMJSON_RS.includes('Error::invalid("expected JSON string")'));
  assert.ok(NMJSON_RS.includes('Error::invalid("expected JSON object")'));
  assert.ok(NMJSON_RS.includes('Error::invalid("truncated JSON")'));
  assert.ok(NMJSON_ERR_RS.includes('Self::new("prtdef-001", detail, "Supply a valid part card, cgraph or typed value")'));
  assert.ok(NMJSON_ERR_RS.includes('"{}: {}. {}. Recovery: fix then retry. Validation side effects: none; completed calls/log appends remain committed"'));
});

test("contract: response fields the pages read are the fields the Rust writes", () => {
  // ComprehensionReply (src/lib/comprehension.ts, read by the memory page) <- comprehension_view.
  assert.ok(MEMORY_PAGE.includes('type ComprehensionReply } from "@/lib/comprehension";'));
  for (const k of ["total_notes", "note_counts", "outline", "routing_record", "drone_default_heads", "drone_lens_sets", "base_cartridges", "base_backed_up", "notes_store", "levels", "unlevelled_notes", "outline_key", "routing", "heads"]) {
    assert.ok(UI_RS.includes(`("${k}", `), `comprehension_view lost ${k}`);
    assert.match(COMPREHENSION_LIB,new RegExp(`\\b${k}\\??:`), `ComprehensionReply lost ${k}`);
  }
  // JobFilterBuildStatus (cartridges page) <- job_filter_build_json.
  for (const k of ["status", "role", "job", "started_ms", "finished_ms", "message"]) {
    assert.ok(UI_RS.includes(`("${k}", s`) || UI_RS.includes(`("${k}", n`), `job_filter_build_json lost ${k}`);
  }
  // SurgeryJobFilter summary keys, the `enabled` default and the `layers` count.
  assert.ok(UI_RS.includes(`for k in ["cartridge", "job", "base", "model", "base_cos", "filtered_cos", "base_max_diff", "filtered_max_diff", "enabled"]`));
  assert.ok(UI_RS.includes(`summary.insert("enabled".to_owned(), Value::Bool(true));`));
  assert.ok(UI_RS.includes(`summary.insert("layers".to_owned(), Value::Number(parts.len().to_string()));`));
  // The job filter reply: role, job, action (and, since the audit row, its audit_id), then the surgery view.
  assert.match(ws(UI_RS), /\("role", s\(role\)\), \("job", s\(job\)\), \("action", s\(action\)\),( \("audit_id", s\(&audit_id\)\),)? \("surgery", surgery_view\(home\)\),/);
  assert.match(ws(UI_RS), /\("role", s\(role\)\), \("job", s\(job\)\), \("layers", s\(layers\)\), \("job_filter_build", job_filter_build_json\((home)?\)\),/);
  // ContinualLearning (learning page) <- learning_cycle_api + engine.to_value.
  for (const k of ["audit_id", "project", "learning", "cycle_mode"]) assert.ok(EXAMPLES_RS.includes(`m.insert("${k}".into()`), `learning_cycle_api lost ${k}`);
  for (const k of ["version", "cycles_run", "last_cycle_ms"]) assert.ok(ENGINE_RS.includes(`("${k}", n(self.${k}))`), `engine.to_value lost ${k}`);
});

test("contract: the pages send the request shapes these tests use and surface errors as notes", () => {
  // 2026-11-26: job filters are built, plugged in or out and removed on Parameters (Cartridges
  // keeps plug and eject on its decks and the remove in its drawer, and links to Parameters).
  const cart = ws(CARTRIDGES_PAGE);
  assert.ok(cart.includes(`path="/api/surgery/job-filter"`) && cart.includes(`action: "delete" }}`));
  assert.ok(cart.includes(`body: { role, job: f.job, action: "enable" }`) && cart.includes(`body: { role, job: f.job, action: "disable" }`));
  assert.ok(cart.includes(`useApi<SurgeryPayload>("/api/surgery", 3000)`));
  assert.ok(cart.includes(`href="/parameters/"`));
  const params = ws(PARAMETERS_PAGE);
  assert.ok(params.includes(`"/api/surgery/job-filter/build", { role, job: job.trim(), layers: layers.trim() || "all" }`));
  assert.ok(params.includes(`"/api/surgery/job-filter", { role: r.role, job: f.job, action: on ? "disable" : "enable" }`));
  assert.ok(params.includes(`path={f ? "/api/surgery/job-filter" : "/api/parameter-maps/remove"}`));
  // The write path is shared by every builder (MapsState): the page reads, then the caller page's own.
  assert.ok(params.includes(`const REFRESH = ["/api/parameter-maps", "/api/surgery", "/api/audit", "/api/live"];`));
  assert.ok(params.includes(`for (const p of [...REFRESH, ...extra]) refreshAll(p);`));
  assert.ok(params.includes(`setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });`) || params.includes(`{ note: { text: e instanceof Error ? e.message : String(e), tone: "error" }`));

  const learn = ws(LEARNING_PAGE);
  // 2026-11-25: a cycle, a rollback of the newest cycle, or a sync of the stored examples.
  assert.ok(learn.includes(`("/api/learning/cycle", { project: projectKey, action });`));
  assert.ok(learn.includes(`const runAction = async (action: "cycle" | "rollback" | "sync") => {`));
  assert.ok(learn.includes(`setNote({ text: e instanceof Error ? e.message : String(e), tone: "error" });`));
  assert.ok(learn.includes(`refreshAll("/api/learning");`));

  // 2026-11-26: the same request, now built in MemoryPanels (compUrl) and read on the page.
  assert.ok(MEMORY_PAGE.includes('`/api/comprehension?project=${encodeURIComponent(key)}` : "/api/comprehension"'));
  assert.ok(/useApi<ComprehensionReply>\(\w+, 15000\)/.test(MEMORY_PAGE));
});

/* =====================================================================================
 * /api/surgery/job-filter: list (GET /api/surgery), on/off (enable/disable), remove (delete).
 * ===================================================================================== */

test("job-filter list: GET /api/surgery lists filters per role with enabled defaults and layer counts", async () => {
  await withEngine({}, async () => {
    await api.refresh("/api/surgery");
    const { data, error } = api.readCache("/api/surgery");
    assert.equal(error, null);
    assert.deepEqual(Object.keys(data.roles), ["reranker", "writer"]);
    const writer = data.roles.writer.job_filters;
    assert.deepEqual(writer.map((j) => j.job), ["code-slice", "docs-only"]);
    assert.equal(writer[0].enabled, true, "a filter file without `enabled` is listed as on");
    assert.equal(writer[1].enabled, false);
    assert.equal(writer[0].layers, 16);
    assert.equal("parts" in writer[0], false, "the raw parts list is summarised, not sent");
    assert.deepEqual(data.job_filter_build, { status: "idle", role: "", job: "", started_ms: 0, finished_ms: 0, message: "" });
  });
});

test("job-filter on/off/remove: valid writes send role/job/action with the token and refresh the list", async () => {
  await withEngine({}, async (engine) => {
    await api.refresh("/api/surgery");
    const before = api.readCache("/api/surgery").data;

    const off = await api.post("/api/surgery/job-filter", { role: "writer", job: "code-slice", action: "disable" });
    const sent = lastPost(engine);
    assert.equal(sent.path, "/api/surgery/job-filter");
    assert.equal(sent.headers["Content-Type"], "application/json");
    assert.equal(sent.headers["X-Nano-Token"], "tok-test");
    assert.deepEqual(JSON.parse(sent.body), { role: "writer", job: "code-slice", action: "disable" });
    assert.equal(off.action, "disable");
    assert.equal(off.surgery.roles.writer.job_filters[0].enabled, false, "the reply carries the updated surgery view");

    await api.refreshAll("/api/surgery");
    const after = api.readCache("/api/surgery").data;
    assert.notEqual(after, before, "a changed list replaces the cached object");
    assert.equal(after.roles.writer.job_filters[0].enabled, false);

    const on = await api.post("/api/surgery/job-filter", { role: "writer", job: "docs-only", action: "enable" });
    assert.equal(on.surgery.roles.writer.job_filters.find((j) => j.job === "docs-only").enabled, true);

    const gone = await api.post("/api/surgery/job-filter", { role: "reranker", job: "triage", action: "delete" });
    assert.deepEqual([gone.role, gone.job, gone.action], ["reranker", "triage", "delete"]);
    assert.equal(gone.surgery.roles.reranker, undefined, "a role with nothing left drops out of the list");
    await api.refreshAll("/api/surgery");
    assert.equal(api.readCache("/api/surgery").data.roles.reranker, undefined);
  });
});

test("job-filter: invalid role/job slugs are refused with the engine's message and change nothing", async () => {
  await withEngine({}, async (engine) => {
    const bad = [
      ["../writer", "code-slice"],
      ["writer", "code slice"],
      ["writer", ""],
      ["", "code-slice"],
      ["writer", "x".repeat(65)],
      ["writer", "café"],
      ["writer", "a/b"],
      ["writer", "code-slice.json"],
    ];
    for (const [role, job] of bad) {
      await assert.rejects(
        () => api.post("/api/surgery/job-filter", { role, job, action: "delete" }),
        (e) => e instanceof Error && e.message === MSG.slug,
        `${JSON.stringify([role, job])} must be refused`,
      );
    }
    assert.equal(Object.keys(engine.state.filters.writer).length, 2);
    // The edge of the slug rule is accepted (and then fails only on existence).
    await assert.rejects(() => api.post("/api/surgery/job-filter", { role: "writer", job: "x".repeat(64), action: "delete" }), {
      message: `job filter writer/${"x".repeat(64)} not found`,
    });
  });
});

test("job-filter: malformed bodies surface the field-level parse error", async () => {
  await withEngine({}, async (engine) => {
    await assert.rejects(() => api.post("/api/surgery/job-filter", { role: "writer", job: "code-slice" }), {
      message: `action: ${nmjsonInvalid("missing JSON key action")}`,
    });
    await assert.rejects(() => api.post("/api/surgery/job-filter", { job: "code-slice", action: "enable" }), {
      message: `role: ${nmjsonInvalid("missing JSON key role")}`,
    });
    await assert.rejects(() => api.post("/api/surgery/job-filter", { role: "writer", job: 7, action: "enable" }), {
      message: `job: ${nmjsonInvalid("expected JSON string")}`,
    });
    await assert.rejects(() => api.post("/api/surgery/job-filter", ["writer", "code-slice", "enable"]), {
      message: `role: ${nmjsonInvalid("expected JSON object")}`,
    });
    // post(path, undefined) sends no body at all: the engine reports truncated JSON.
    await assert.rejects(() => api.post("/api/surgery/job-filter", undefined), { message: `body: ${nmjsonInvalid("truncated JSON")}` });
    assert.equal(lastPost(engine).body, undefined);
    // An action outside on/off/remove, and a filter that does not exist.
    await assert.rejects(() => api.post("/api/surgery/job-filter", { role: "writer", job: "code-slice", action: "toggle" }), {
      message: "unknown action toggle (use enable, disable, or delete)",
    });
    await assert.rejects(() => api.post("/api/surgery/job-filter", { role: "writer", job: "missing", action: "enable" }), {
      message: "job filter writer/missing not found",
    });
    assert.equal(engine.state.filters.writer["code-slice"].enabled, undefined, "no refused write touched the file");
  });
});

test("job-filter: server errors reach the caller (refused token, 500 text, network failure, GET on the write path)", async () => {
  await withEngine({}, async (engine) => {
    engine.state.token = "tok-rotated"; // the page still holds the old token
    globalThis.document = { querySelector: () => ({ getAttribute: () => "tok-stale", setAttribute() {} }) };
    await assert.rejects(() => api.post("/api/surgery/job-filter", { role: "writer", job: "code-slice", action: "enable" }), { message: MSG.write403 });
    assert.equal(engine.calls.filter((c) => c.method === "POST").length, 1, "no fresh token was found, so the write is not retried");
  });
  await withEngine({}, async (engine) => {
    engine.state.failNext = { status: 500, body: "encode: value too deep", ctype: "text/plain" };
    await assert.rejects(() => api.post("/api/surgery/job-filter", { role: "writer", job: "code-slice", action: "enable" }), { message: "encode: value too deep" });
    engine.state.failNext = { status: 500, body: "", ctype: "text/plain" };
    await assert.rejects(() => api.post("/api/surgery/job-filter", { role: "writer", job: "code-slice", action: "enable" }), { message: "500 Internal Server Error" });
    engine.state.failNext = "network";
    await assert.rejects(() => api.post("/api/surgery/job-filter", { role: "writer", job: "code-slice", action: "enable" }), { message: "fetch failed" });

    // There is no GET on the write path; the client records the 404 instead of throwing.
    await api.refresh("/api/surgery/job-filter");
    const r = api.readCache("/api/surgery/job-filter");
    assert.equal(r.data, null);
    assert.equal(r.error, "404 Not Found");
  });
});

/* =====================================================================================
 * /api/surgery/job-filter/build: build + held-out split-gate.
 * ===================================================================================== */

test("job-filter build: the page's body starts a split-gate build and the status shows as running", async () => {
  await withEngine({}, async (engine) => {
    // What the page sends from its inputs: trimmed role and job, empty layers -> "all".
    const [buildRole, buildJob, buildLayers] = [" writer ", "code-slice ", "   "];
    const res = await api.post("/api/surgery/job-filter/build", { role: buildRole.trim(), job: buildJob.trim(), layers: buildLayers.trim() || "all" });
    assert.deepEqual(JSON.parse(lastPost(engine).body), { role: "writer", job: "code-slice", layers: "all" });
    assert.deepEqual([res.role, res.job, res.layers], ["writer", "code-slice", "all"]);
    assert.equal(res.job_filter_build.status, "running");
    assert.equal(res.job_filter_build.message, "Running held-out split-gate fit for writer/code-slice (layers: all)...");

    await api.refreshAll("/api/surgery");
    await api.refresh("/api/surgery");
    assert.equal(api.readCache("/api/surgery").data.job_filter_build.status, "running");
  });
  for (const layers of ["12-27", "0,3,5-7", "1".repeat(32)]) {
    await withEngine({}, async () => {
      const res = await api.post("/api/surgery/job-filter/build", { role: "writer", job: "code-slice", layers });
      assert.equal(res.layers, layers);
    });
  }
  // `layers` is optional, and a non-string value falls back to "all" as the engine reads it.
  await withEngine({}, async () => {
    assert.equal((await api.post("/api/surgery/job-filter/build", { role: "writer", job: "code-slice" })).layers, "all");
  });
  await withEngine({}, async () => {
    assert.equal((await api.post("/api/surgery/job-filter/build", { role: "writer", job: "code-slice", layers: 12 })).layers, "all");
  });
});

test("job-filter build: invalid slugs, layer specs and malformed bodies are refused before anything runs", async () => {
  await withEngine({}, async (engine) => {
    for (const [role, job] of [["writer", ""], ["wri ter", "x"], ["writer", "../../bin"], ["writer", "a;b"]]) {
      await assert.rejects(() => api.post("/api/surgery/job-filter/build", { role, job, layers: "all" }), { message: MSG.slug });
    }
    for (const layers of ["", "12..27", "1;rm -rf", "L12-27", "1".repeat(33), "ALL"]) {
      await assert.rejects(() => api.post("/api/surgery/job-filter/build", { role: "writer", job: "code-slice", layers }), { message: MSG.layers }, `layers ${JSON.stringify(layers)}`);
    }
    await assert.rejects(() => api.post("/api/surgery/job-filter/build", { role: "writer" }), { message: `job: ${nmjsonInvalid("missing JSON key job")}` });
    await assert.rejects(() => api.post("/api/surgery/job-filter/build", "writer/code-slice"), { message: `role: ${nmjsonInvalid("expected JSON object")}` });
    await assert.rejects(() => api.post("/api/surgery/job-filter/build", null), { message: `role: ${nmjsonInvalid("expected JSON object")}` });
    assert.equal(engine.state.build, null, "no refused build changed the build state");
  });
});

test("job-filter build: engine-side failures surface as the page's error note text", async () => {
  await withEngine({ labPresent: false }, async () => {
    await assert.rejects(() => api.post("/api/surgery/job-filter/build", { role: "writer", job: "code-slice", layers: "all" }), { message: MSG.noLab });
  });
  await withEngine({}, async () => {
    await api.post("/api/surgery/job-filter/build", { role: "writer", job: "code-slice", layers: "all" });
    // A second click while the first build runs.
    let note = null;
    try {
      await api.post("/api/surgery/job-filter/build", { role: "reranker", job: "triage", layers: "all" });
    } catch (e) {
      note = { text: e instanceof Error ? e.message : String(e), tone: "error" };
    }
    assert.deepEqual(note, { text: MSG.running, tone: "error" });
  });
  await withEngine({}, async (engine) => {
    engine.state.failNext = { status: 403, body: MSG.write403, ctype: "text/plain" };
    await assert.rejects(() => api.post("/api/surgery/job-filter/build", { role: "writer", job: "code-slice" }), { message: MSG.write403 });
  });
});

/* =====================================================================================
 * GET /api/comprehension
 * ===================================================================================== */

test("comprehension: GET returns the readiness shape, per project, with the key URL-encoded", async () => {
  await withEngine({}, async (engine) => {
    await api.refresh("/api/comprehension");
    const nano = api.readCache("/api/comprehension").data;
    assert.deepEqual(Object.keys(nano).sort(), ["base_backed_up", "base_cartridges", "drone_default_heads", "drone_lens_sets", "note_counts", "outline", "routing_record", "total_notes"]);
    assert.equal(nano.total_notes, 7);
    assert.equal(typeof nano.routing_record, "boolean");

    const key = "alpha";
    const url = `/api/comprehension?project=${encodeURIComponent(key)}`;
    await api.refresh(url);
    const alpha = api.readCache(url).data;
    assert.equal(alpha.total_notes, 2);
    assert.equal(alpha.outline, null);
    assert.notEqual(alpha, nano, "each project URL is its own cache entry");

    // Keys are encoded exactly as the memory page encodes them.
    const odd = "beta.v2";
    await api.refresh(`/api/comprehension?project=${encodeURIComponent(odd)}`);
    assert.equal(engine.calls.at(-1).query, "project=beta.v2");
    await api.refresh(`/api/comprehension?project=${encodeURIComponent("a b&c")}`);
    assert.equal(engine.calls.at(-1).query, "project=a%20b%26c");
  });
});

test("comprehension: an invalid or unknown project key falls back to the Nano's own memory (no error)", async () => {
  await withEngine({}, async () => {
    for (const key of ["../etc", ".hidden", "a b&c", "no-such-project", ""]) {
      const url = `/api/comprehension?project=${encodeURIComponent(key)}`;
      await api.refresh(url);
      const r = api.readCache(url);
      assert.equal(r.error, null, `key ${JSON.stringify(key)}`);
      assert.equal(r.data.total_notes, 7, `key ${JSON.stringify(key)} reads the home memory`);
    }
  });
});

test("comprehension: server errors are recorded on the cache entry and the last good data is kept", async () => {
  await withEngine({}, async (engine) => {
    const url = "/api/comprehension?project=alpha";
    await api.refresh(url);
    const good = api.readCache(url).data;

    engine.state.failNext = { status: 500, body: "encode: bad number", ctype: "text/plain" };
    await api.refresh(url);
    let r = api.readCache(url);
    assert.equal(r.error, "500 Internal Server Error");
    assert.equal(r.data, good, "the panel keeps showing the last good reply");

    engine.state.failNext = "network";
    await api.refresh(url);
    r = api.readCache(url);
    assert.equal(r.error, "fetch failed");
    assert.equal(r.data, good);

    await api.refresh(url);
    r = api.readCache(url);
    assert.equal(r.error, null, "the next good reply clears the error");
    assert.deepEqual(r.data, good);

    // A path the engine does not know (an older engine build, or a typo).
    await api.refresh("/api/comprehension/notes");
    assert.equal(api.readCache("/api/comprehension/notes").error, "404 Not Found");
    // A reply that is not JSON is an error, not a crash.
    engine.state.failNext = { status: 200, body: "<html>not found; build the interface</html>", ctype: "text/html" };
    await api.refresh("/api/comprehension?project=nano");
    assert.match(api.readCache("/api/comprehension?project=nano").error, /JSON/);
  });
});

/* =====================================================================================
 * POST /api/learning/cycle
 * ===================================================================================== */

test("learning cycle: the page's cycle and rollback bodies run and return the engine state", async () => {
  await withEngine({}, async (engine) => {
    const projectKey = "alpha";
    const runAction = (action, version) =>
      api.post("/api/learning/cycle", { project: projectKey, action, ...(typeof version === "number" ? { version } : {}) });

    const c = await runAction("cycle");
    assert.deepEqual(JSON.parse(lastPost(engine).body), { project: "alpha", action: "cycle" });
    assert.equal(c.version, 4);
    assert.equal(c.cycles_run, 3);
    assert.equal(c.project, "alpha");
    assert.match(c.audit_id, /^a\d+$/);
    assert.equal(typeof c.learning, "string");
    assert.equal(typeof c.cycle_mode, "string");
    assert.deepEqual(c.rollback_versions, [1, 2, 3]);

    const r = await runAction("rollback", 2);
    assert.deepEqual(JSON.parse(lastPost(engine).body), { project: "alpha", action: "rollback", version: 2 });
    assert.equal(r.version, 2);
    assert.deepEqual(r.rollback_versions, [1, 3]);

    // Without a version the latest snapshot is used; without an action a cycle runs.
    assert.equal((await runAction("rollback")).version, 3);
    const d = await api.post("/api/learning/cycle", { project: "nano" });
    assert.equal(d.project, "nano");
  });
});

test("learning cycle: invalid project keys and malformed bodies are refused with the engine's message", async () => {
  await withEngine({}, async (engine) => {
    for (const key of ["../alpha", ".hidden", "a b", "alpha/../../x", "no-such-project", ""]) {
      await assert.rejects(() => api.post("/api/learning/cycle", { project: key, action: "cycle" }), { message: `project: no project ${key}` });
    }
    await assert.rejects(() => api.post("/api/learning/cycle", { action: "cycle" }), { message: `project: ${nmjsonInvalid("missing JSON key project")}` });
    await assert.rejects(() => api.post("/api/learning/cycle", { project: 42 }), { message: `project: ${nmjsonInvalid("expected JSON string")}` });
    await assert.rejects(() => api.post("/api/learning/cycle", undefined), { message: `body: ${nmjsonInvalid("truncated JSON")}` });
    assert.equal(engine.state.learning.version, 3, "no refused call ran a cycle");
  });
});

test("learning cycle: rollback failures and server errors surface as the page's error note text", async () => {
  await withEngine({ learning: { version: 1, cycles_run: 0, history: [] } }, async (engine) => {
    const runAction = async (action, version) => {
      try {
        await api.post("/api/learning/cycle", { project: "alpha", action, ...(typeof version === "number" ? { version } : {}) });
        return null;
      } catch (e) {
        return { text: e instanceof Error ? e.message : String(e), tone: "error" };
      }
    };
    let note = await runAction("rollback");
    assert.equal(note.tone, "error");
    assert.ok(note.text.includes(MSG.noSnapshot), note.text);

    await runAction("cycle");
    note = await runAction("rollback", 99);
    assert.ok(note.text.includes(MSG.noSnapshot), "an unknown version has no snapshot either");

    engine.state.failNext = { status: 400, body: { error: "learning-cycle: audit log is held by another writer" } };
    assert.deepEqual(await runAction("cycle"), { text: "learning-cycle: audit log is held by another writer", tone: "error" });
    engine.state.failNext = { status: 403, body: "remote device is view-only", ctype: "text/plain" };
    assert.deepEqual(await runAction("cycle"), { text: "remote device is view-only", tone: "error" });
    engine.state.failNext = "network";
    assert.deepEqual(await runAction("cycle"), { text: "fetch failed", tone: "error" });
  });
});

/* =====================================================================================
 * A running engine (skipped unless one answers on NANO_LIVE_API_BASE or 127.0.0.1:4870).
 * Only reads and writes that the engine refuses during validation are sent: nothing is changed.
 * ===================================================================================== */

const LIVE_BASE = (process.env.NANO_LIVE_API_BASE || "http://127.0.0.1:4870").replace(/\/+$/, "");
const liveUp = await fetch(LIVE_BASE + "/api/status", { signal: AbortSignal.timeout(800) })
  .then((r) => r.ok)
  .catch(() => false);
const liveSkip = liveUp ? false : `no engine reachable at ${LIVE_BASE} (start \`neuromorph-nano ui\` or set NANO_LIVE_API_BASE to run these)`;

async function withLive(fn) {
  const origEnv = process.env.NEXT_PUBLIC_API_BASE;
  const origDocument = globalThis.document;
  const origWindow = globalThis.window;
  api._resetApiForTest();
  delete globalThis.document;
  delete globalThis.window;
  process.env.NEXT_PUBLIC_API_BASE = LIVE_BASE;
  try {
    await fn();
  } finally {
    api._resetApiForTest();
    if (origEnv === undefined) delete process.env.NEXT_PUBLIC_API_BASE;
    else process.env.NEXT_PUBLIC_API_BASE = origEnv;
    if (origDocument !== undefined) globalThis.document = origDocument;
    if (origWindow !== undefined) globalThis.window = origWindow;
  }
}

/** The write token the engine puts in its served page, or null (interface not built). */
async function liveToken() {
  const html = await fetch(LIVE_BASE + "/", { cache: "no-store" }).then((r) => r.text()).catch(() => "");
  return html.match(/<meta\b[^>]*\bname=["']nano-token["'][^>]*\bcontent=["']([^"']+)["']/i)?.[1] ?? null;
}

test("live: GET /api/surgery and /api/comprehension answer with the shapes the pages read", { skip: liveSkip }, async () => {
  await withLive(async () => {
    await api.refresh("/api/surgery");
    const s = api.readCache("/api/surgery");
    assert.equal(s.error, null);
    assert.equal(typeof s.data.roles, "object");
    assert.ok(["idle", "running", "done", "error"].includes(s.data.job_filter_build.status));
    for (const info of Object.values(s.data.roles)) {
      for (const jf of info.job_filters ?? []) {
        assert.equal(typeof jf.job, "string");
        assert.equal(typeof jf.enabled, "boolean");
      }
    }
    await api.refresh("/api/comprehension");
    const c = api.readCache("/api/comprehension");
    assert.equal(c.error, null);
    for (const k of ["total_notes", "note_counts", "routing_record", "drone_default_heads", "drone_lens_sets", "base_cartridges", "base_backed_up"]) {
      assert.ok(k in c.data, `missing ${k}`);
    }
    await api.refresh(`/api/comprehension?project=${encodeURIComponent("../etc")}`);
    assert.equal(api.readCache(`/api/comprehension?project=${encodeURIComponent("../etc")}`).error, null);
    await api.refresh("/api/surgery/job-filter");
    assert.equal(api.readCache("/api/surgery/job-filter").error, "404 Not Found");
  });
});

test("live: invalid writes are refused with the engine's messages (nothing is changed)", { skip: liveSkip }, async (t) => {
  if (!(await liveToken())) {
    t.skip("the engine's page carries no write token (interface/out not built)");
    return;
  }
  await withLive(async () => {
    await assert.rejects(() => api.post("/api/surgery/job-filter", { role: "../x", job: "y", action: "enable" }), { message: MSG.slug });
    await assert.rejects(() => api.post("/api/surgery/job-filter", { role: "x" }), /^Error: job: prtdef-001: missing JSON key job/);
    await assert.rejects(() => api.post("/api/surgery/job-filter/build", { role: "x", job: "y z" }), { message: MSG.slug });
    await assert.rejects(() => api.post("/api/surgery/job-filter/build", { role: "x", job: "y", layers: "1;2" }), { message: MSG.layers });
    await assert.rejects(() => api.post("/api/learning/cycle", { project: "../nope", action: "cycle" }), { message: "project: no project ../nope" });
    await assert.rejects(() => api.post("/api/learning/cycle", {}), /^Error: project: prtdef-001: missing JSON key project/);
  });
});
