import "./register-ts.mjs";
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

/* Tests for the 4-level comprehension view (`/api/comprehension`, src/lib/comprehension.ts and the
 * memory page), the stage-times view (`/api/stage-times`, src/lib/stagetimes.ts and the audit
 * page), the per-MicroModel graphics card meter (src/lib/vram.ts and the settings page) and the
 * route list of scripts/e2e-ui-verify.mjs. Contract tests read the Rust sources and fail when the
 * field names or tags the pages rely on drift. */

const api = await import("../src/lib/api.ts");
const comp = await import("../src/lib/comprehension.ts");
const st = await import("../src/lib/stagetimes.ts");
const { vramSegments, mb } = await import("../src/lib/vram.ts");

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (p) => readFileSync(resolve(HERE, p), "utf8");
const UI_RS = read("../../crates/nmcore/src/bin/ui.rs");
const WRITER_RS = read("../../crates/nmapi/src/writer_idle.rs");
const STAGE_BENCH_RS = read("../../crates/nmlab/src/stage_bench.rs");
// 2026-11-26: the Memory page and its panels (MemoryPanels).
const MEMORY_PAGE = ["../src/app/memory/page.tsx", "../src/components/MemoryPanels.tsx"].map(read).join(" ");
const AUDIT_PAGE = read("../src/app/audit/page.tsx");
const SETTINGS_PAGE = read("../src/app/settings/page.tsx");
const STAGE_PANEL = read("../src/components/StageTimes.tsx");
const E2E = read("../scripts/e2e-ui-verify.mjs");
const ROUTING = JSON.parse(read("../../designs/routing-record.json"));
const DRONES = JSON.parse(read("../../designs/drones.json"));

/* ---------- Ports of ui.rs `routing_view` and `heads_view` (pinned by the contract tests). ---------- */

const pick = (v, keys) => Object.fromEntries(keys.filter((k) => k in v).map((k) => [k, v[k]]));
function routingView(rec) {
  if (!rec) return { present: false };
  const rows = [];
  const combine = [];
  for (const r of rec.rows ?? []) {
    if ("model" in r) rows.push(pick(r, ["job", "kind", "model", "stack", "accuracy", "slope", "n", "correct", "check_accuracy", "check_n", "strength"]));
    else if ("combine" in r) combine.push({ decisive: null, ...pick(r, ["job", "kind", "combine", "decisive"]) });
  }
  const lens = rec.lens ?? "";
  return {
    present: true,
    made_utc: rec.made_utc ?? null,
    version: rec.version ?? null,
    rows,
    combine,
    lens,
    lenses: lens.split(",").map((s) => s.trim()).filter(Boolean),
    base_cartridges: Object.keys(rec.base_cartridges ?? {}).sort(),
  };
}
function headsView(drones) {
  const out = [];
  for (const pop of Object.keys(drones?.populations ?? {}).sort()) {
    const p = drones.populations[pop];
    if (p.default) out.push({ population: pop, lens: "default", heads: p.default, count: p.default.length });
    for (const lens of Object.keys(p.tentacles ?? {}).sort()) out.push({ population: pop, lens, heads: p.tentacles[lens], count: p.tentacles[lens].length });
  }
  return out;
}

/* =====================================================================================
 * Contracts
 * ===================================================================================== */

test("contract: ui.rs reads the note store by the writer's field names (k, by, t), never key/text", () => {
  // The writer appends {"k", "by", "t"} lines (writer_idle.rs `keep` and `rewrite_store`).
  assert.ok(WRITER_RS.includes('[("k", key), ("by", by), ("t", text)]'), "writer_idle.rs keep() changed its fields");
  assert.match(UI_RS, /let \(Some\(k\), Some\(_\), Some\(t\)\) = \(get\("k"\), get\("by"\), get\("t"\)\) else \{ continue \};/);
  const body = UI_RS.slice(UI_RS.indexOf("fn read_notes("), UI_RS.indexOf("fn note_store("));
  assert.doesNotMatch(body, /get\("key"\)|get\("text"\)/);
  // The new store first, the old one as fallback.
  assert.ok(UI_RS.includes('["world-model/notes.jsonl", "writer/notes.jsonl"]'));
});

test("contract: the level of every note tag is the same in ui.rs and src/lib/comprehension.ts, and every writer tag has a level", () => {
  const m = UI_RS.match(/const LEVELS: \[\(u64, &str, &\[&str\]\); 4\] = \[(.*)\];/);
  assert.ok(m, "ui.rs LEVELS not found");
  const rust = [...m[1].matchAll(/\((\d), "(\w+)", &\[([^\]]*)\]\)/g)].map(([, l, name, tags]) => ({
    level: Number(l),
    name,
    tags: [...tags.matchAll(/"(\w+)"/g)].map((x) => x[1]),
  }));
  assert.deepEqual(rust, comp.LEVELS.map(({ level, name, tags }) => ({ level, name, tags })));
  const writerTags = [...WRITER_RS.slice(WRITER_RS.indexOf("pub const fn tag(self)"), WRITER_RS.indexOf("pub const fn tokens(self)")).matchAll(/=> "(\w+)"/g)].map((x) => x[1]);
  assert.deepEqual(writerTags.sort(), ["file", "module", "names", "note", "outline", "sum"]);
  for (const t of writerTags) assert.notEqual(comp.levelOf(t), null, `writer tag ${t} has no level`);
  assert.deepEqual(
    ["note", "names", "sum", "file", "module", "outline", "q", ""].map(comp.levelOf),
    [1, 1, 1, 2, 3, 4, null, null],
  );
});

test("contract: routing and heads fields the panel reads are the fields ui.rs writes", () => {
  const src = UI_RS.replace(/\s+/g, " ");
  assert.ok(src.includes('pick(r, &["job", "kind", "model", "stack", "accuracy", "slope", "n", "correct", "check_accuracy", "check_n", "strength"])'));
  assert.ok(src.includes('pick(r, &["job", "kind", "combine", "decisive"])'));
  for (const k of ["present", "made_utc", "rows", "combine", "lens", "lenses", "base_cartridges"]) assert.ok(src.includes(`("${k}", `), `routing_view lost ${k}`);
  for (const k of ["population", "lens", "heads", "count"]) assert.ok(src.includes(`("${k}", `), `heads_view lost ${k}`);
  for (const k of ["notes_store", "levels", "unlevelled_notes", "outline_key", "routing", "heads"]) assert.ok(src.includes(`("${k}", `));
  // The memory page draws every part of it.
  for (const e of ["comprehension-levels", "comprehension-outline", "comprehension-routing", "comprehension-heads"]) assert.ok(MEMORY_PAGE.includes(`data-e2e="${e}"`), e);
  for (const f of ["levelsOf(d)", "routingByJob(d?.routing)", "headsText(h.heads)", "combineText(j.combine)"]) assert.ok(MEMORY_PAGE.includes(f), f);
});

test("contract: /api/stage-times is routed, parses the tables nano-lab writes, and is drawn on /audit", () => {
  assert.match(UI_RS, /"\/api\/stage-times" => Some\(stage_times_view\(home, query\)\),/);
  assert.ok(STAGE_BENCH_RS.includes('"  stages, {label} (ms per warm question: median / p95 / n)"'), "nano-lab changed its table head");
  assert.ok(UI_RS.includes('t.strip_prefix("stages, ")'));
  assert.ok(UI_RS.includes('name.strip_prefix("STAGE-BENCH-")?.strip_suffix(".txt")?.parse().ok()'));
  for (const k of ["source", "live", "file", "stamp", "header", "sets", "files", "hook"]) assert.ok(UI_RS.includes(`("${k}", `), `stage_times_view lost ${k}`);
  for (const k of ["set", "summary", "warm", "stages", "stage", "median_ms", "p95_ms"]) assert.ok(UI_RS.includes(`("${k}", `), `stage row lost ${k}`);
  assert.ok(AUDIT_PAGE.includes("<StageTimesPanel />"));
  assert.ok(STAGE_PANEL.includes('useApi<StageTimes>("/api/stage-times", 30000)'));
  // Every stage name drone_path.rs reports through nano-lab has a plain name.
  const m = STAGE_BENCH_RS.match(/const STAGES: \[&str; \d+\] = \[([^\]]*)\]/);
  if (m) for (const name of [...m[1].matchAll(/"(\w+)"/g)].map((x) => x[1])) assert.ok(name in st.STAGE_NAMES, `no plain name for stage ${name}`);
});

test("contract: the settings page draws one meter row per MicroModel from the live models, else vramSegments", () => {
  assert.ok(SETTINGS_PAGE.includes("const vram = liveModels.length ? liveVramSegments(liveModels, gpuCeilingMb) : vramSegments(status, gpuCeilingMb);"));
  assert.ok(SETTINGS_PAGE.includes('data-e2e="vram-models"'));
});

/* =====================================================================================
 * Comprehension mapping
 * ===================================================================================== */

/** The real `GET /api/comprehension` reply (tests/fixtures, captured from a running server). */
const COMP_REAL = JSON.parse(read("fixtures/comprehension.json"));

test("comprehension: the real reply reads into the panel's helpers", () => {
  const levels = comp.levelsOf(COMP_REAL);
  assert.deepEqual(
    levels.map((l) => [l.label, l.count, l.tags.map((t) => t.tag).join(",")]),
    [
      ["L1 piece", 0, "note,names,sum"],
      ["L2 file", 0, "file"],
      ["L3 module", 0, "module"],
      ["L4 system", 0, "outline"],
    ],
  );
  assert.equal(COMP_REAL.total_notes, 0);
  assert.equal(COMP_REAL.notes_store, null);
  assert.equal(comp.unlevelled(COMP_REAL.note_counts), 0);
  const jobs = comp.routingByJob(COMP_REAL.routing);
  assert.deepEqual(
    jobs.map((j) => [j.job, j.rows.map((r) => r.model).join(","), comp.combineText(j.combine)]),
    [
      ["code-close", "embedder,judge", "combined, decisive at 2.15"],
      ["prose-close", "embedder,judge", "combined, no decisive threshold"],
    ],
  );
  assert.equal(comp.pct(jobs[0].rows[0].accuracy), "78.1%");
  assert.equal(comp.slopeText(jobs[0].rows[1].slope), "-1.17");
  assert.deepEqual(COMP_REAL.routing.lenses, ["embedder-lens-5270b394a1b2@9", "none"]);
  assert.deepEqual(
    COMP_REAL.heads.map((h) => [h.lens, h.count, h.heads.length]),
    [
      ["default", 16, 16],
      ["words", 16, 16],
    ],
  );
  assert.equal(comp.headsText(COMP_REAL.heads[0].heads, 2), "L13.6 L14.6 +14");
});

test("comprehension: levelsOf uses the engine's levels when sent, in L1 to L4 order with labels", () => {
  const levels = [
    { level: 3, name: "module", count: 4, tags: [{ tag: "module", count: 4 }] },
    { level: 1, name: "piece", count: 12, tags: [{ tag: "note", count: 7 }, { tag: "names", count: 3 }, { tag: "sum", count: 2 }] },
    { level: 2, name: "file", count: 5, tags: [{ tag: "file", count: 5 }] },
    { level: 4, name: "system", count: 1, tags: [{ tag: "outline", count: 1 }] },
  ];
  const out = comp.levelsOf({ note_counts: {}, levels });
  assert.deepEqual(out.map((l) => [l.label, l.count]), [["L1 piece", 12], ["L2 file", 5], ["L3 module", 4], ["L4 system", 1]]);
  assert.equal(out[0].tags.length, 3);
});

test("comprehension: levelsOf computes the levels from note_counts for an older build", () => {
  const out = comp.levelsOf({ note_counts: { note: 7, names: 2, file: 6, outline: 1, q: 9 } });
  assert.deepEqual(out.map((l) => l.count), [9, 6, 0, 1]);
  assert.deepEqual(out[0].tags, [{ tag: "note", count: 7 }, { tag: "names", count: 2 }, { tag: "sum", count: 0 }]);
  assert.equal(comp.unlevelled({ note: 7, q: 9, x: 1 }), 10);
  // No reply yet: four empty levels, never a crash.
  assert.deepEqual(comp.levelsOf(null).map((l) => l.count), [0, 0, 0, 0]);
});

test("comprehension: the real routing record groups by job with its model rows and combine flag", () => {
  const r = routingView(ROUTING);
  const jobs = comp.routingByJob(r);
  assert.ok(jobs.length >= 1);
  for (const j of jobs) {
    assert.ok(j.rows.length >= 1, `${j.job} has model rows`);
    for (const row of j.rows) {
      assert.equal(typeof row.accuracy, "number");
      assert.equal(typeof row.n, "number");
      assert.equal(typeof row.slope, "number");
      assert.match(comp.pct(row.accuracy), /^\d+(\.\d)?%$/);
    }
    assert.ok(j.combine, `${j.job} has a combine flag`);
  }
  const decisive = jobs.find((j) => typeof j.combine?.decisive === "number");
  if (decisive) assert.match(comp.combineText(decisive.combine), /^combined, decisive at \d+\.\d\d$/);
  assert.ok(r.lenses.length >= 1);
  assert.deepEqual(comp.routingByJob({ present: false }), []);
  assert.deepEqual(comp.routingByJob(null), []);
});

test("comprehension: formatting of percentages, slopes, combine lines and head lists", () => {
  assert.equal(comp.pct(0.7812), "78.1%");
  assert.equal(comp.pct(undefined), "-");
  assert.equal(comp.slopeText(13.292735), "+13.29");
  assert.equal(comp.slopeText(-1.167455), "-1.17");
  assert.equal(comp.slopeText(null), "-");
  assert.equal(comp.combineText({ combine: false }), "not combined");
  assert.equal(comp.combineText({ combine: true, decisive: null }), "combined, no decisive threshold");
  assert.equal(comp.combineText({ combine: true, decisive: 2.148641 }), "combined, decisive at 2.15");
  assert.equal(comp.headsText([[13, 6], [14, 6]]), "L13.6 L14.6");
  assert.equal(comp.headsText([[1, 1], [2, 2], [3, 3]], 2), "L1.1 L2.2 +1");
  assert.equal(comp.headsText(null), "");
});

test("comprehension: the real drone sets give a default list and each lens's own list", () => {
  const heads = headsView(DRONES);
  const def = heads.find((h) => h.lens === "default");
  assert.ok(def, "a default set");
  assert.equal(def.count, def.heads.length);
  assert.equal(comp.headsText(def.heads, 999).split(" ").length, def.count);
  for (const h of heads) assert.ok(h.heads.every((x) => Array.isArray(x) && x.length === 2));
});

/* =====================================================================================
 * Stage times
 * ===================================================================================== */

const SET = {
  set: "large-code",
  warm: "median 3349 p95 4933 min 2370 max 6056 (n 70)",
  stages: [
    { stage: "sense", median_ms: 916, p95_ms: 1173, n: 70 },
    { stage: "g1", median_ms: 1405, p95_ms: 1535, n: 70 },
    { stage: "pack", median_ms: 2, p95_ms: 4, n: 70 },
    { stage: "total", median_ms: 3319, p95_ms: 4902, n: 70 },
  ],
};

test("stage times: bars take their share of the total median and leave the total out", () => {
  const { rows, total } = st.stageBars(SET);
  assert.deepEqual(rows.map((r) => r.stage), ["sense", "g1", "pack"]);
  assert.equal(total.median_ms, 3319);
  assert.ok(Math.abs(rows[1].share - 1405 / 3319) < 1e-9);
  // No total row: shares of the sum of medians.
  const noTotal = st.stageBars({ set: "x", stages: SET.stages.slice(0, 2) });
  assert.equal(noTotal.total, null);
  assert.ok(Math.abs(noTotal.rows[0].share + noTotal.rows[1].share - 1) < 1e-9);
  assert.deepEqual(st.stageBars(null), { rows: [], total: null });
  assert.equal(st.slowest(SET).stage, "g1");
  assert.equal(st.slowest({ set: "x", stages: [] }), null);
});

test("stage times: the newest live call (when published) reads as stage rows", () => {
  assert.equal(st.lastLiveCall(null), null);
  assert.equal(st.lastLiveCall({ live_calls: [] }), null);
  assert.deepEqual(st.lastLiveCall({ live_calls: [{ sense: 1 }, { at_ms: 5, sense: 900, g1: 1400, total: 3300, note: "x" }] }), [
    { stage: "sense", ms: 900 },
    { stage: "g1", ms: 1400 },
  ]);
  assert.match(UI_RS, /\("live_calls", Value::Array\(live_calls\)\)/);
  assert.ok(UI_RS.includes('p.get("stage_times")'));
});

test("stage times: milliseconds and stamps read plainly", () => {
  assert.equal(st.msShort(414), "414 ms");
  assert.equal(st.msShort(3319), "3.32 s");
  assert.equal(st.msShort(undefined), "-");
  assert.equal(st.stampText(1795489519750), "2026-11-24 03:05 UTC");
  assert.equal(st.stampText(null), "");
  for (const k of ["signals_meaning", "evidence", "drone_kind", "sense", "compose", "g1", "tentacle_read", "deep_check", "g2_own", "g2_order_reads", "judge_duels", "judge_close", "pack", "total"]) {
    assert.ok(k in st.STAGE_NAMES, k);
  }
});

test("stage times: the newest report in runs/ (when there is one) has tables the server's rules can read", (t) => {
  const runs = resolve(HERE, "../../runs");
  const files = existsSync(runs) ? readdirSync(runs).filter((f) => /^STAGE-BENCH-\d+\.txt$/.test(f)) : [];
  if (files.length === 0) {
    t.skip("no runs/STAGE-BENCH-*.txt here");
    return;
  }
  files.sort((a, b) => Number(b.slice(12, -4)) - Number(a.slice(12, -4)));
  // The rule of ui.rs parse_stage_bench: rows of `<stage> <median> <p95> <n>` under a `stages, ` head.
  const withTables = files.find((f) => statSync(resolve(runs, f)).size > 0 && /^\s+stages, /m.test(readFileSync(resolve(runs, f), "utf8")));
  if (!withTables) {
    t.skip("no stage-bench report with stage tables");
    return;
  }
  const lines = readFileSync(resolve(runs, withTables), "utf8").split(/\r?\n/);
  const at = lines.findIndex((l) => l.trim().startsWith("stages, "));
  const row = lines[at + 1].trim().split(/\s+/);
  assert.equal(row.length, 4);
  assert.ok(Number.isFinite(Number(row[1])) && Number.isFinite(Number(row[2])) && Number.isInteger(Number(row[3])));
});

/* =====================================================================================
 * The api client against an engine double for both GET views.
 * ===================================================================================== */

async function withReplies(routes, fn) {
  const origFetch = globalThis.fetch;
  const origWindow = globalThis.window;
  const origDocument = globalThis.document;
  const origEnv = process.env.NEXT_PUBLIC_API_BASE;
  api._resetApiForTest();
  delete globalThis.window;
  delete globalThis.document;
  delete process.env.NEXT_PUBLIC_API_BASE;
  globalThis.fetch = async (url) => {
    const u = new URL(String(url), "http://127.0.0.1:4870");
    const body = routes[u.pathname];
    if (body === undefined) return new Response('{"error":"unknown endpoint"}', { status: 404, statusText: "Not Found" });
    return new Response(JSON.stringify(typeof body === "function" ? body(u) : body), { status: 200, statusText: "OK", headers: { "Content-Type": "application/json" } });
  };
  try {
    await fn();
  } finally {
    api._resetApiForTest();
    globalThis.fetch = origFetch;
    if (origWindow !== undefined) globalThis.window = origWindow;
    if (origDocument !== undefined) globalThis.document = origDocument;
    if (origEnv === undefined) delete process.env.NEXT_PUBLIC_API_BASE;
    else process.env.NEXT_PUBLIC_API_BASE = origEnv;
  }
}

test("client: /api/comprehension with levels, routing and heads reads into the panel's helpers", async () => {
  const reply = {
    total_notes: 9,
    note_counts: { note: 4, names: 2, file: 2, outline: 1 },
    notes_store: "world-model/notes.jsonl",
    levels: comp.LEVELS.map((l) => ({ level: l.level, name: l.name, count: l.tags.reduce((a, t) => a + ({ note: 4, names: 2, file: 2, outline: 1 }[t] ?? 0), 0), tags: l.tags.map((t) => ({ tag: t, count: { note: 4, names: 2, file: 2, outline: 1 }[t] ?? 0 })) })),
    unlevelled_notes: 0,
    outline: "crates: nmcore, nmapi",
    outline_key: "outline:abc",
    routing_record: true,
    routing: routingView(ROUTING),
    drone_default_heads: 16,
    drone_lens_sets: 1,
    heads: headsView(DRONES),
    base_cartridges: 4,
    base_backed_up: 4,
  };
  await withReplies({ "/api/comprehension": reply }, async () => {
    await api.refresh("/api/comprehension?project=nano");
    const { data, error } = api.readCache("/api/comprehension?project=nano");
    assert.equal(error, null);
    assert.deepEqual(comp.levelsOf(data).map((l) => l.count), [6, 2, 0, 1]);
    assert.equal(comp.routingByJob(data.routing).length, comp.routingByJob(reply.routing).length);
    assert.equal(data.heads.length, reply.heads.length);
  });
});

test("client: /api/stage-times answers or reports a 404 from an older build", async () => {
  const reply = { source: "stage-bench", live: false, file: "STAGE-BENCH-1.txt", stamp: 1, header: "stage-bench base", sets: [SET], files: [{ file: "STAGE-BENCH-1.txt", stamp: 1 }], hook: "h" };
  await withReplies({ "/api/stage-times": reply }, async () => {
    await api.refresh("/api/stage-times");
    const { data } = api.readCache("/api/stage-times");
    assert.equal(data.live, false);
    assert.equal(st.stageBars(data.sets[0]).rows.length, 3);
  });
  await withReplies({}, async () => {
    await api.refresh("/api/stage-times");
    assert.equal(api.readCache("/api/stage-times").error, "404 Not Found");
    assert.ok(STAGE_PANEL.includes('q.error.startsWith("404")'));
  });
});

/* =====================================================================================
 * Graphics card meter per MicroModel
 * ===================================================================================== */

test("vram: one segment per listed model, held bytes summed over processes, shares of the ceiling", () => {
  const MiB = 1024 * 1024;
  const status = {
    models: [{ id: "embedder" }, { id: "reranker" }, { id: "reader" }, { id: "writer" }],
    loaded: [
      { id: "embedder", gpu_bytes: 600 * MiB, device: "cuda" },
      { id: "writer", gpu_bytes: 1000 * MiB, device: "cuda" },
      { id: "writer", gpu_bytes: 200 * MiB },
      { id: "extra", gpu_bytes: 100 * MiB },
    ],
  };
  const seg = vramSegments(status, 4000);
  assert.deepEqual(seg.map((s) => s.id), ["embedder", "reranker", "reader", "writer", "extra"]);
  assert.deepEqual(seg.map((s) => s.held), [true, false, false, true, true]);
  assert.equal(mb(seg[3].gpuBytes), 1200);
  assert.equal(seg[3].device, "cuda");
  assert.ok(Math.abs(seg[0].share - 0.15) < 1e-9);
  assert.equal(seg[1].share, 0);
  assert.deepEqual(vramSegments(null, 3500), []);
  // A ceiling of 0 does not divide by zero.
  assert.ok(Number.isFinite(vramSegments(status, 0)[0].share));
});

/* =====================================================================================
 * e2e-ui-verify.mjs covers every static route and the new controls
 * ===================================================================================== */

test("e2e script: visits every static route of src/app, /parts included, and checks the new views", () => {
  const appDir = resolve(HERE, "../src/app");
  const routes = ["/", ...readdirSync(appDir).filter((d) => existsSync(resolve(appDir, d, "page.tsx"))).map((d) => `/${d}/`)];
  // 2026-11-25: /parameters/ joined (the whole parameter mapping, directions of Luis Jake Gabriel III).
  assert.equal(routes.length, 12);
  const m = E2E.match(/const ROUTES = \[([^\]]*)\]/);
  assert.ok(m, "ROUTES list");
  const listed = [...m[1].matchAll(/"([^"]+)"/g)].map((x) => x[1]);
  assert.deepEqual([...listed].sort(), [...routes].sort());
  for (const needle of ["/api/surgery/job-filter/build", "/api/surgery/job-filter", "comprehension-levels", "comprehension-outline", "comprehension-routing", "comprehension-heads", "vram-models", "stage-times", "Rollback", "/api/learning/cycle", "no server answers"]) {
    assert.ok(E2E.includes(needle), `e2e script lost ${needle}`);
  }
});
