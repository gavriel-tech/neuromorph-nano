import "./register-ts.mjs";
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

/* Contract tests for Phase 4.5 items 1 and 4 (UI side): the live feed (`GET /api/live`, its
 * stream), the storage view and controls (`/api/storage`, prune, delete, settings), the learned
 * pieces that replace the fixed "0 learned pieces", the graphics card ceiling control that now
 * defaults to `auto`, and the pages that draw them. Every parser is fed the contract's shape, an
 * empty reply, an error reply and junk: an engine build without a route must give an empty view,
 * never a guessed one. */

const live = await import("../src/lib/livefeed.ts");
const storage = await import("../src/lib/storage.ts");
const learnedLib = await import("../src/lib/learned.ts");
const controls = await import("../src/lib/controls.ts");
const vram = await import("../src/lib/vram.ts");
const scenes = await import("../src/lib/pagescenes.ts");
const { buildTree, resolvePart } = await import("../src/lib/topology.ts");
const hook = await import("../src/lib/livehook.ts");

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (p) => readFileSync(resolve(HERE, p), "utf8");
const src = (p) => read(`../src/${p}`);

/* ---------- The real replies (tests/fixtures, captured from a running server). ---------- */

const fixture = (name) => JSON.parse(read(`fixtures/${name}.json`));
/** `GET /api/live` with one engine serving (CUDA, a GTX 1650, the four MicroModels seated, a dense
 *  cache build running). */
const LIVE = fixture("live");
/** `GET /api/live` with no engine serving, as the interface builds it (live.rs `merged`): its own
 *  process's state, `serving: 0`, no processes, `backend` null, no MicroModel seated and no card
 *  opened. Derived from the real reply. */
const LIVE_OFF = {
  ...LIVE,
  state: "idle",
  active: [],
  progress: [],
  serving: 0,
  processes: [],
  backend: null,
  backend_note: "No graphics card has been opened yet in this process.",
  models: LIVE.models.map((m) => ({ ...m, card: null, resident: false, executing: false, vram_mb: 0, precision: null, busy_ms: 0, forwards: 0 })),
  vram: { ...LIVE.vram, total_mb: null, used_mb: null, free_mb: null, by_model: { embedder: 0, reranker: 0, reader: 0, writer: 0, filters: 0, kv_windows: 0 } },
  capacity: { ...LIVE.capacity, card: null, gpu_ceiling_mb: null, limits: { ...LIVE.capacity.limits, mode: "processor" } },
};
const STORAGE_REAL = fixture("storage");
const STORAGE_SETTINGS_REAL = fixture("storage-settings");

/* ---------- A reply with every optional form the parsers also read (socket strings and objects,
 * preference cartridges, level objects, a writer summary). ---------- */

const VARIANT = {
  state: "busy",
  active: [{ stage: "judge_duels", model: "reranker-base@5", since_ms: 1_000_000, elapsed_ms: 42 }],
  models: [
    { role: "embedder", card: "embedder-base@4", resident: true, executing: false, vram_mb: 610, precision: "bmq8", filters: [], lens: "project-lens-a" },
    { role: "reranker", card: "reranker-base@5", resident: true, executing: true, vram_mb: 1200, precision: "q8", filters: ["reranker/code-slice", { id: "pref-1", kind: "preference", enabled: false }], lens: null },
    { role: "reader", card: "reader-base@3", resident: true, executing: false, vram_mb: 900, precision: "dp4a", filters: [], lens: null },
    { role: "writer", card: "writer-base@4", resident: false, executing: false, vram_mb: null, precision: "q8", filters: [], lens: null },
  ],
  vram: { total_mb: 8192, used_mb: 3500, free_mb: 4692, reserve_mb: 1024, by_model: { embedder: 610, reranker: 1200, reader: 900, writer: 0, kv_windows: 500 } },
  capacity: { card: { name: "Test Card", total_mb: 8192, free_mb: 4692, compute: "8.6", reserve_mb: 1024, reserve_measured: true }, host_available_mb: 20000, host_total_mb: 32768, threads: 16, gpu_ceiling_mb: 7168, ram_ceiling_mb: 24576 },
  writer: { queue: 7, held: true, sessions: 3, results: 120, preempted: 2, kept_resident: false, headroom_mb: 900, levels: { L1: { done: 40, total: 100 }, L2: 12, L3: { count: 3 }, L4: null }, last_event: "L2 file note written" },
  timeline: [
    { at_ms: 999_000, kind: "stage", stage: "signals_meaning", model: "reader-base@3", ms: 30, summary: "" },
    { at_ms: 999_200, kind: "stage", stage: "tentacle_read", model: null, ms: 150, summary: "t3 read" },
    { at_ms: 999_000, kind: "stage", stage: "signals_meaning", model: "reader-base@3", ms: 30, summary: "" },
    { at_ms: 990_000, kind: "writer", stage: "L1", model: "writer-base@4", ms: 800, summary: "piece notes" },
    { kind: "junk without time" },
  ],
  backend: "cuda",
};

test("live: the real reply parses in full", () => {
  const r = live.parseLive(LIVE);
  assert.equal(r.state, "busy");
  assert.equal(r.backend, "cuda");
  assert.equal(r.backend_note, "Graphics card in use: NVIDIA GeForce GTX 1650 (compute 7.5, 4095 MB).");
  assert.equal(r.serving, 1);
  assert.equal(live.engineServing(r), true);
  assert.deepEqual(r.processes, [{ pid: 10940, project: "repo-6f34c1f1", state: "busy" }]);
  assert.deepEqual(r.progress, [{ kind: "dense", done: 640, total: 2999, since_ms: 1795577349657 }]);
  assert.equal(r.project, "repo-6f34c1f1");
  assert.equal(r.pid, 10940);
  assert.deepEqual(r.active[0], { stage: "dense", model: "embedder", since_ms: 1795577349657, elapsed_ms: 143799 });
  // The four MicroModels, in the engine's order.
  assert.deepEqual(
    r.models.map((m) => [m.role, m.card, m.resident, m.executing, m.vram_mb, m.precision]),
    [
      ["embedder", "qwen3-embedding-0.6b", true, true, 381, "q8"],
      ["reranker", "qwen3-reranker-0.6b", true, false, 382, "q8"],
      ["reader", null, false, false, 0, null],
      ["writer", "qwen3-1.7b", true, false, 1196, "q8"],
    ],
  );
  assert.deepEqual(r.vram, { total_mb: 4095, used_mb: 3397, free_mb: 698, reserve_mb: 140, reserve_measured: true, by_model: { embedder: 381, filters: 0, kv_windows: 0, reader: 0, reranker: 382, writer: 1196 } });
  // Capacity: the card under `capacity.card`, the limits under `capacity.limits`.
  assert.deepEqual(r.capacity.card, { name: "NVIDIA GeForce GTX 1650", total_mb: 4095, free_mb: 698, compute: "7.5", reserve_mb: 140, reserve_measured: true });
  assert.equal(r.capacity.gpu_ceiling_mb, 3955);
  assert.equal(r.capacity.host_total_mb, 32436);
  assert.equal(r.capacity.host_available_mb, 10316);
  assert.equal(r.capacity.ram_ceiling_mb, 24327);
  assert.equal(r.capacity.threads, 12);
  assert.equal(r.capacity.hardware_threads, 12);
  assert.equal(r.capacity.limits.mode, "four seated");
  assert.equal(r.capacity.limits.vram.effective_free_mb, 698);
  // The writer: levels as plain counts, the last event as {at_ms, kind}.
  assert.equal(r.writer.queue, 253);
  assert.equal(r.writer.held, true);
  assert.equal(r.writer.in_session, false);
  assert.equal(r.writer.kept_resident, true);
  assert.equal(r.writer.headroom_mb, 140);
  assert.equal(r.writer.preempted, 2);
  assert.equal(r.writer.sessions, 2);
  assert.equal(r.writer.results, 3);
  assert.equal(r.writer.last_event, "kept");
  assert.deepEqual(r.writer.levels.L1, { count: 3, done: null, total: null });
  assert.deepEqual(r.writer.levels.L4, { count: 0, done: null, total: null });
  // The timeline: every event kept, oldest first.
  assert.equal(r.timeline.length, LIVE.timeline.length);
  assert.equal(r.timeline[0].summary, "progress dense start done 0 total 2999");
  assert.equal(r.timeline.at(-1).stage, "call");
  // The header line names the dense cache build, not a search.
  assert.equal(live.liveLine(r), "dense cache build · dense · embedder");
});

test("live: with no engine serving the reply says so and no live figure is shown", () => {
  const r = live.parseLive(LIVE_OFF);
  assert.equal(r.serving, 0);
  assert.equal(live.engineServing(r), false);
  assert.equal(r.backend, null);
  assert.equal(live.liveLine(r), "No engine serving");
  assert.equal(vram.vramBreakdown(r.vram), null, "no card opened: no breakdown");
  assert.equal(live.NO_ENGINE, "No engine is serving right now; start one with neuromorph-nano serve or daemon");
  assert.ok(r.models.every((m) => !m.resident));
  // An older reply without `serving` falls back to whether it names a backend.
  assert.equal(live.engineServing({ serving: null, backend: "cpu" }), true);
  assert.equal(live.engineServing({ serving: null, backend: null }), null, "neither said: not known");
  assert.equal(live.engineServing(null), null);
});

test("live: the optional forms (socket strings and objects, preferences, level objects) still parse", () => {
  const r = live.parseLive(VARIANT);
  const rr = r.models[1];
  assert.equal(rr.executing, true);
  assert.deepEqual(rr.filters, [
    { id: "reranker/code-slice", kind: "filter", on: true },
    { id: "pref-1", kind: "preference", on: false },
  ]);
  assert.equal(r.models[0].lens, "project-lens-a");
  assert.equal(r.vram.by_model.kv_windows, 500);
  assert.equal(r.writer.preempted, 2);
  assert.equal(r.writer.last_event, "L2 file note written");
  assert.deepEqual(r.writer.levels.L1, { count: 40, done: 40, total: 100 });
  assert.deepEqual(r.writer.levels.L2, { count: 12, done: null, total: null });
  assert.equal(r.writer.levels.L4, null);
  assert.equal(r.timeline.length, 3, "one per identity, junk left out");
  assert.deepEqual(
    r.timeline.map((e) => e.at_ms),
    [990_000, 999_000, 999_200],
  );
});

test("live: empty, error and junk replies give no view, never a guessed one", () => {
  assert.equal(live.parseLive(null), null);
  assert.equal(live.parseLive("404 Not Found"), null);
  assert.equal(live.parseLive([1, 2]), null);
  const e = live.parseLive({});
  assert.equal(e.state, "idle");
  assert.deepEqual(e.active, []);
  assert.deepEqual(e.models, []);
  assert.equal(e.vram, null);
  assert.equal(e.capacity, null);
  assert.equal(e.writer, null);
  assert.deepEqual(e.timeline, []);
  assert.equal(e.backend, null);
  assert.equal(live.liveLine(null), "live state not reported");
  assert.equal(live.liveLine(e), "Idle / Listening");
  // A machine without a graphics card: card null, nothing invented.
  const cpu = live.parseLive({ state: "idle", capacity: { card: null, host_available_mb: 1000, host_total_mb: 8000, threads: 4, gpu_ceiling_mb: null, ram_ceiling_mb: 6000 }, backend: "cpu" });
  assert.equal(cpu.capacity.card, null);
  assert.equal(cpu.capacity.gpu_ceiling_mb, null);
  assert.equal(cpu.backend, "cpu");
});

test("live: idle shows Idle / Listening with the recent real timeline; busy names the stage", () => {
  const idle = live.parseLive({ ...LIVE, state: "idle", active: [], progress: [], models: LIVE.models.map((m) => ({ ...m, executing: false })) });
  assert.equal(idle.state, "idle");
  assert.equal(live.liveLine(idle), "Idle / Listening");
  assert.equal(idle.timeline.length, LIVE.timeline.length);
  const busy = live.parseLive(VARIANT);
  assert.match(live.liveLine(busy), /^G2 reads and judging · judge_duels · reranker-base@5/);
  // No state field: busy only when something is active or executing.
  assert.equal(live.parseLive({ active: [] }).state, "idle");
  assert.equal(live.parseLive({ active: ["pack"] }).state, "busy");
});

test("live: stage groups and the anatomy ids each event lights (G0 to G3, tentacles, duels, writer L1 to L4, preemption)", () => {
  const cases = [
    ["signals_meaning", "G0"],
    ["compose", "G0"],
    ["g1", "G1"],
    ["evidence", "G1"],
    ["tentacle_read", "G2"],
    ["judge_duels", "G2"],
    ["judge_close", "G2"],
    ["g2_order_reads", "G2"],
    ["pack", "G3"],
    ["L1", "writer"],
    ["L4", "writer"],
    ["preempt", "preempt"],
    ["dense_cache", "dense"],
    ["job_filter_build", "filter"],
    ["learning_cycle", "learning"],
    ["something_else", "other"],
    // The engine's own stage names.
    ["G0", "G0"],
    ["G1.dense", "G1"],
    ["G1.tree", "G1"],
    ["G2.close", "G2"],
    ["G2.duel", "G2"],
    ["tentacle.read", "G2"],
    ["dense", "dense"],
  ];
  for (const [stage, g] of cases) assert.equal(live.stageGroup(stage), g, stage);
  assert.equal(live.stageGroup("filter.build", "filter"), "filter");
  assert.equal(live.stageGroup("learn.cycle", "learn"), "learning");
  assert.equal(live.stageGroup("dense", "progress"), "dense", "the dense cache build's progress events");
  // Every stage of the real timeline has a group.
  for (const e of live.parseLive(LIVE).timeline) assert.ok(live.GROUP_NAMES[live.stageGroup(e.stage, e.kind)], e.stage);
  assert.deepEqual(live.partIdsFor({ stage: "signals_meaning", kind: "stage", model: "reader-base@3", summary: "" }), ["nmbrain.g0", "nmbrain.slot#router"]);
  assert.deepEqual(live.partIdsFor({ stage: "tentacle_read", kind: "stage", model: null, summary: "tentacle 3" }), ["nmbrain.g2", "nmpod.unit#t3"]);
  assert.deepEqual(live.partIdsFor({ stage: "judge_duels", kind: "stage", model: "reranker-base@5", summary: "" }), ["nmbrain.g2", "nmbrain.slot#reranker"]);
  assert.deepEqual(live.partIdsFor({ stage: "L2", kind: "writer", model: "writer-base@4", summary: "" }), ["nmbrain.slot#writer", "nmmem.hierarchy"]);
  assert.deepEqual(live.partIdsFor({ stage: "preempt", kind: "preempt", model: null, summary: "" }), ["nmbrain.slot#writer", "nmbrain.homeostasis"]);
  // Only the tentacles the engine names light: the mask of a head read, never the whole neuropod.
  const read = live.parseLiveEvent({ at_ms: 5, kind: "stage", stage: "tentacle.read", model: "reader", ms: 40, counts: { reads: 2, tentacle_mask: 5 }, summary: "stage tentacle.read reader 40 ms reads 2 tentacle_mask 5" });
  assert.deepEqual(read.tentacles, [0, 2]);
  assert.equal(read.summary, "stage tentacle.read reader 40 ms reads 2 tentacles T1 T3");
  assert.deepEqual(live.partIdsFor(read), ["nmbrain.g2", "nmpod.unit#t0", "nmpod.unit#t2", "nmbrain.slot#router"]);
  assert.deepEqual(live.partIdsFor({ stage: "tentacle.read", kind: "stage", model: "reader", summary: "stage tentacle.read reader reads 0" }), ["nmbrain.g2", "nmbrain.slot#router"], "a read that names no tentacle lights none");
  // One timed event each: a memory write, an overlay rebind, a backup, a lab build.
  assert.equal(live.stageGroup("memory.write", "memory"), "memory");
  assert.deepEqual(live.partIdsFor({ stage: "memory.write", kind: "memory", model: null, summary: "" }), ["nmmem.gate"]);
  assert.equal(live.stageGroup("overlay.rebind", "rebind"), "rebind");
  assert.deepEqual(live.partIdsFor({ stage: "overlay.rebind", kind: "rebind", model: "embedder", summary: "" }), ["nmbrain.slot#embedder"]);
  assert.equal(live.stageGroup("backup", "backup"), "backup");
  assert.deepEqual(live.partIdsFor({ stage: "backup", kind: "backup", model: null, summary: "" }), [], "a backup has no part in the anatomy");
  // A guidance delivery (one event per hook reply) has no part either: strip and System panel only.
  for (const stage of ["guidance.reminder", "guidance.loop", "guidance.step"]) {
    assert.equal(live.stageGroup(stage, "guidance"), "guidance");
    assert.deepEqual(live.partIdsFor({ stage, kind: "guidance", model: null, summary: "" }), [], "the anatomy has no hooks or agent part");
  }
  assert.equal(live.GROUP_NAMES.guidance, "guidance delivered");
  assert.equal(live.stageGroup("lab.build", "filter"), "filter");
  assert.deepEqual(live.partIdsFor({ stage: "lab.build", kind: "filter", model: "writer", summary: "" }), ["nmlearn.cartridge", "nmbrain.slot#writer"]);
  // A writer event labelled preempted is the short preemption pulse.
  assert.equal(live.eventGroup({ stage: "writer", kind: "writer", label: "preempted" }), "preempt");
  assert.equal(live.holdFor({ stage: "writer", kind: "writer", label: "preempted", ms: 0 }), 300);
  assert.equal(live.eventGroup({ stage: "writer", kind: "writer", label: "session" }), "writer");
  // The idle Writer in words: writing, why it is held back, or only loaded.
  assert.equal(live.writerState({ in_session: true, held: true }), "writing now");
  assert.equal(live.writerState({ in_session: false, held: false, waiting: "memory" }), "held back: not enough free memory to load it");
  assert.equal(live.writerState({ in_session: false, held: true, waiting: null }), "on the card, not writing");
  assert.equal(live.writerState({ in_session: false, held: false }), "not loaded");
  assert.equal(live.writerState(null), null);
  // What runs now: a stage of the call before the call itself; nothing at rest.
  assert.equal(live.nowLine({ active: [{ stage: "call", model: null, label: "nm_context_pack" }, { stage: "G1.dense", model: "embedder" }], models: [], writer: null }), "G1 search · G1.dense · embedder");
  assert.equal(live.nowLine({ active: [{ stage: "call", model: null, label: "nm_context_pack" }], models: [], writer: null }), "call · nm_context_pack");
  assert.equal(live.nowLine({ active: [], models: [{ role: "reader", executing: true }], writer: null }), "reader executing");
  assert.equal(live.nowLine({ active: [], models: [{ role: "reader", executing: false }], writer: { in_session: false } }), null);
  assert.equal(live.writerLevel("L3", ""), 3);
  assert.equal(live.writerLevel("module_note", "L3 module notes"), 3);
  assert.equal(live.holdFor({ stage: "preempt", kind: "", ms: 5000 }), 300);
  assert.equal(live.holdFor({ stage: "pack", kind: "", ms: 10 }), 450);
  assert.equal(live.holdFor({ stage: "pack", kind: "", ms: 99_999 }), 3000);
});

test("live: every id an event lights resolves to a drawn part of the real anatomy", () => {
  const CRATES = resolve(HERE, "../../crates");
  const parts = Object.fromEntries(
    readdirSync(CRATES)
      .filter((d) => existsSync(resolve(CRATES, d, "part.json")))
      .map((d) => [d, JSON.parse(readFileSync(resolve(CRATES, d, "part.json"), "utf8"))]),
  );
  const anatomy = JSON.parse(read("../../designs/anatomy.json"));
  const tree = buildTree({ anatomy, parts, status: null, cartridges: [] });
  for (const stage of ["signals_meaning", "g1", "tentacle_read", "judge_duels", "pack", "L1", "preempt", "dense_cache", "job_filter_build", "learning_cycle"]) {
    const ids = live.partIdsFor({ stage, kind: "", model: "reader-base@3", summary: "tentacle 1" });
    assert.ok(ids.length > 0, `${stage} lights something`);
    for (const id of ids) assert.ok(resolvePart(tree, id).length > 0, `${stage}: ${id} resolves to a part`);
  }
});

test("live: timelines merge once per event; the waterfall of the newest burst", () => {
  const a = { at_ms: 100, kind: "stage", stage: "g1", model: null, ms: 40, summary: "" };
  const b = { at_ms: 200, kind: "stage", stage: "pack", model: null, ms: 50, summary: "" };
  assert.deepEqual(live.mergeTimeline([a], [a, b]), [a, b]);
  assert.equal(live.mergeTimeline([], Array.from({ length: 300 }, (_, i) => ({ ...a, at_ms: i })), 200).length, 200);
  const old = { at_ms: -10_000, kind: "stage", stage: "g1", model: null, ms: 10, summary: "" };
  const w = live.waterfall([old, a, b]);
  assert.equal(w.bars.length, 2, "an older burst is left out");
  assert.equal(w.bars[0].start, 0);
  assert.equal(w.span, 140, "from the first start (100 - 40) to the last end (200)");
  assert.equal(w.at, 200);
  // A running stage grows from its start to now.
  const run = live.waterfall([a], [{ stage: "pack", model: null, since_ms: 150, elapsed_ms: null }], 260);
  assert.equal(run.bars.at(-1).running, true);
  assert.equal(run.bars.at(-1).ms, 110);
  assert.deepEqual(live.waterfall([]), { bars: [], span: 0, at: null });
  // Idle writer passes are not a question's stages.
  assert.equal(live.waterfall([{ ...a, stage: "L1", kind: "writer" }]).bars.length, 0);
});

test("live: the feed primes on history and hands only fresh events on; a missing route is polled slowly", () => {
  hook._resetLiveForTest();
  const got = [];
  const off = hook.onFresh((e) => got.push(e.stage));
  const h1 = { at_ms: 1, kind: "stage", stage: "g1", model: null, ms: 5, summary: "" };
  const h2 = { at_ms: 2, kind: "stage", stage: "pack", model: null, ms: 5, summary: "" };
  hook.ingest([h1], true);
  assert.deepEqual(got, [], "history present when the page opens is not replayed");
  hook.ingest([h1, h2], true);
  assert.deepEqual(got, ["pack"]);
  hook.ingest([h2], true);
  assert.deepEqual(got, ["pack"], "an event is handed on once");
  off();
  hook._resetLiveForTest();
  assert.equal(hook.pollEvery("polling", null), 1000);
  assert.equal(hook.pollEvery("stream", null), 3000);
  assert.equal(hook.pollEvery("polling", "404 Not Found"), 30000);
  assert.equal(hook.pollEvery("polling", "500 Internal"), 1000);
});

test("live: the model of a released id, its state word", () => {
  const real = live.parseLive(LIVE);
  assert.equal(live.modelState(live.liveModelFor(real.models, "qwen3-embedding-0.6b")), "executing");
  assert.equal(live.modelState(live.liveModelFor(real.models, "qwen3-reranker-0.6b")), "resident");
  assert.equal(live.modelState(live.liveModelFor(real.models, "qwen3-1.7b")), "resident");
  assert.equal(live.liveModelFor(real.models, "qwen3-1.7b").role, "writer");
  const r = live.parseLive(VARIANT);
  assert.equal(live.liveModelFor(r.models, "reranker-base@5").role, "reranker");
  assert.equal(live.modelState(live.liveModelFor(r.models, "reranker-base@5")), "executing");
  assert.equal(live.modelState(live.liveModelFor(r.models, "embedder-base@4")), "resident");
  assert.equal(live.modelState(live.liveModelFor(r.models, "writer-base@4")), "not held");
  assert.equal(live.modelState(live.liveModelFor([], "writer-base@4")), "not reported");
  assert.equal(live.modelState(live.liveModelFor(null, "x")), "not reported");
});

/* ---------- VRAM breakdown and the machine's capacity ---------- */

test("vram: the 4-model breakdown, KV and windows, the free reserve and what is free", () => {
  const r = live.parseLive(LIVE);
  const parts = vram.vramBreakdown(r.vram);
  assert.deepEqual(
    parts.map((p) => [p.id, p.label, p.mb]),
    [
      ["embedder", "embedder", 381],
      ["reranker", "reranker", 382],
      ["reader", "reader", 0],
      ["writer", "writer", 1196],
      ["filters", "job filters", 0],
      ["kv_windows", "KV and windows", 0],
      ["other", "other use", 3397 - (381 + 382 + 1196)],
      ["reserve", "free reserve", 140],
      ["free", "free", 698 - 140],
    ],
  );
  const sum = parts.reduce((a, p) => a + p.share, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, "the shares add up to the card");
  // The variant with KV windows in use.
  const v = vram.vramBreakdown(live.parseLive(VARIANT).vram);
  assert.equal(v.find((p) => p.id === "kv_windows").mb, 500);
  assert.equal(vram.vramBreakdown(null), null);
  assert.equal(vram.vramBreakdown({ total_mb: null, used_mb: null, free_mb: null, reserve_mb: null, by_model: {} }), null);
});

test("vram: one meter row per MicroModel from the live models", () => {
  const r = live.parseLive(LIVE);
  const seg = vram.liveVramSegments(r.models, 3955);
  assert.deepEqual(
    seg.map((s) => [s.role, s.id, s.held, vram.mb(s.gpuBytes), s.precision, s.device]),
    [
      ["embedder", "qwen3-embedding-0.6b", true, 381, "q8", null],
      ["reranker", "qwen3-reranker-0.6b", true, 382, "q8", null],
      ["reader", "reader", false, 0, null, null],
      ["writer", "qwen3-1.7b", true, 1196, "q8", null],
    ],
  );
  assert.ok(Math.abs(seg[3].share - 1196 / 3955) < 1e-9);
  assert.deepEqual(vram.liveVramSegments(null, 3955), []);
});

test("vram: budget.gpu_mb_max auto follows this card; a number is used as it is", () => {
  const cap = live.parseLive(LIVE).capacity;
  assert.deepEqual(vram.gpuCeiling("auto", cap), { mb: 3955, auto: true, label: "auto (follows this card) · 3955 MiB" });
  assert.deepEqual(vram.gpuCeiling(undefined, cap).auto, true, "a missing value is auto (the default)");
  assert.deepEqual(vram.gpuCeiling("auto", null), { mb: null, auto: true, label: "auto (follows this card) · ceiling not reported" });
  assert.deepEqual(vram.gpuCeiling("auto", { gpu_ceiling_mb: null, card: { total_mb: 8000, reserve_mb: 1000 } }).mb, null, "no ceiling reported: none is invented from the card");
  assert.deepEqual(vram.gpuCeiling("81920", cap), { mb: 81920, auto: false, label: "81920 MiB" });
  assert.deepEqual(vram.gpuCeiling("3500", null), { mb: 3500, auto: false, label: "3500 MiB" });
});

test("contract: no live panel says a figure is not reported by the engine build; no engine serving is said plainly", () => {
  for (const f of ["components/LivePanels.tsx", "components/HardwareLimits.tsx", "app/settings/page.tsx", "app/memory/page.tsx"]) {
    assert.ok(!src(f).includes("not reported live by this engine build"), f);
  }
  const panels = src("components/LivePanels.tsx");
  assert.ok(panels.includes("No engine is serving right now; start one with <code"));
  assert.ok(panels.includes('"free not measured"'));
  assert.ok(src("app/memory/page.tsx").includes("? NO_ENGINE"));
  assert.ok(src("lib/livehook.ts").includes('addEventListener("live"'), "the stream's named events are read");
});

/* ---------- The auto control ---------- */

const gpuControl = (over = {}) => ({ key: "budget.gpu_mb_max", label: "Graphics card ceiling", help: "", kind: "integer", min: 1024, max: 81920, step: 256, unit: "MiB", values: [], default: "auto", value: "auto", source: "default", wired: true, applies: "now", ...over });

test("controls: budget.gpu_mb_max shows auto (follows this card) and takes auto or a number up to 81,920 MiB", () => {
  assert.equal(controls.formatControlValue("auto", "budget.gpu_mb_max", "MiB"), "auto (follows this card)");
  assert.equal(controls.formatControlValue("auto", "budget.gpu_mb_max"), "auto (follows this card)");
  assert.equal(controls.formatControlValue("6144", "budget.gpu_mb_max", "MiB"), "6144 MiB");
  assert.equal(controls.formatControlValue("default", "precision.base"), "auto");
  const c = gpuControl();
  assert.equal(controls.isNumericControl(c), true);
  assert.equal(controls.acceptsAuto(c), true);
  assert.deepEqual(controls.validateIntegerControl(c, "auto"), { ok: true, value: null, normalized: "auto" });
  assert.deepEqual(controls.validateIntegerControl(c, " AUTO "), { ok: true, value: null, normalized: "auto" });
  assert.deepEqual(controls.validateIntegerControl(c, "81920"), { ok: true, value: 81920, normalized: "81920" });
  assert.deepEqual(controls.validateIntegerControl(c, "81921"), { ok: false, error: "Must be at most 81920 MiB" });
  assert.deepEqual(controls.validateIntegerControl(c, "abc"), { ok: false, error: "Must be a whole number or auto" });
  assert.deepEqual(controls.validateIntegerControl(c, ""), { ok: false, error: "Enter a whole number or auto" });
  // Stepping from auto starts at what auto resolves to on this card, else at the floor.
  assert.equal(controls.stepIntegerControl(c, "auto", 1, 7168), "7424");
  assert.equal(controls.stepIntegerControl(c, "auto", -1, 7168), "6912");
  assert.equal(controls.stepIntegerControl(c, "auto", 1), "1280");
  // A control without auto keeps its old messages.
  const plain = gpuControl({ key: "budget.pack_tokens", default: "4000", value: "4000", min: 1000, max: 16000, unit: "tokens" });
  assert.equal(controls.acceptsAuto(plain), false);
  assert.deepEqual(controls.validateIntegerControl(plain, "auto"), { ok: false, error: "Must be a whole number" });
});

test("contract: the settings page resolves the ceiling through gpuCeiling, not a fixed 3500", () => {
  const page = src("app/settings/page.tsx");
  assert.ok(page.includes('gpuCeiling(byKey.get("budget.gpu_mb_max"), live.reply?.capacity)'));
  assert.ok(!page.includes('Number(byKey.get("budget.gpu_mb_max") ?? "3500") || 3500'));
  assert.ok(page.includes("<VramBar"));
  assert.ok(page.includes("<Capacity"));
  assert.ok(page.includes("<StageTimesPanel />"));
  const ctl = src("components/Controls.tsx");
  assert.ok(ctl.includes("acceptsAuto(c)"));
  assert.ok(ctl.includes('formatControlValue("auto", c.key)'));
});

/* ---------- Learned pieces (the fixed "0 learned pieces" is gone) ---------- */

const CYCLES = {
  verified_since_last_cycle: 3,
  threshold: 20,
  due: false,
  cycles: [
    { id: "c1", learners: [{ name: "g3-priors", ran: true, promoted: true }, { name: "preference-cartridge", ran: true, promoted: false }, { name: "drones", ran: false }], rolled_back: false },
    { id: "c2", learners: [{ name: "g3-priors", ran: true, promoted: true }, { name: "judge-readout", ran: true, promoted: true }], rolled_back: true },
    { id: "c3", learners: [{ name: "judge-readout", ran: true, promoted: true }], rolled_back: false },
  ],
};
const SURGERY = { roles: { reranker: { job_filters: [{ cartridge: "jf-a", job: "code-slice", enabled: true }, { cartridge: "jf-b", job: "tests", enabled: false }] }, reader: {} } };
const cart = (over) => ({ schema: "", id: "c", kind: "learning", data_class: "", base: "", base_sha256: "", components: [], scopes: [], quantization: "", description: "", version: 1, check: "", date: "", loaded: true, ...over });

test("learned: real counts of job filters, lenses, judge read-outs, preference cartridges and G3 muscle updates, with split-gate telemetry", () => {
  const filters = learnedLib.jobFiltersOf(SURGERY);
  assert.equal(filters.length, 2);
  const lenses = learnedLib.lensesOf({ lens: "base", lenses: ["proj-a", "proj-b"] }, [{ lens: "proj-a" }, { lens: null }]);
  assert.deepEqual(lenses.sort(), ["proj-a", "proj-b"]);
  const carts = [cart({ id: "p1", class: "preference" }), cart({ id: "j1", kind: "judge-readout" }), cart({ id: "l1" })];
  const l = learnedLib.learnedCounts({ carts, filters, lenses, idle: CYCLES });
  assert.equal(l.filters, 2);
  assert.equal(l.filtersOn, 1);
  assert.equal(l.lenses, 2);
  assert.equal(l.preference, 1);
  assert.equal(l.judges, 2, "one judge read-out cartridge, one promoted judge learner not rolled back");
  assert.equal(l.muscle, 1, "the G3 update of a rolled-back cycle does not count");
  assert.equal(l.gatePass, 4);
  assert.equal(l.gateHeld, 1);
  assert.equal(l.rollbacks, 1);
  assert.equal(l.cycles, 3);
  assert.equal(l.total, 2 + 2 + 2 + 1 + 1);
  assert.equal(learnedLib.learnedLine(l), "8 learned pieces (2 job filters, 2 lenses, 2 judge read-outs, 1 preference cartridge, 1 G3 muscle update)");
  assert.equal(learnedLib.gateLine(l), "split gate 4 passed · 1 held back · 1 rolled back · 3 cycles");
});

test("learned: a source that did not answer is not reported, never zero", () => {
  const none = learnedLib.learnedCounts({});
  assert.equal(none.total, null);
  assert.equal(learnedLib.learnedLine(none), "learned pieces not reported");
  assert.equal(learnedLib.gateLine(none), "split gate not reported");
  assert.equal(learnedLib.jobFiltersOf(null), null);
  assert.equal(learnedLib.lensesOf(null, null), null);
  const zero = learnedLib.learnedCounts({ carts: [], filters: [], lenses: [], idle: { cycles: [] } });
  assert.equal(zero.total, 0);
  assert.equal(learnedLib.learnedLine(zero), "0 learned pieces (0 job filters, 0 lenses, 0 judge read-outs, 0 preference cartridges, 0 G3 muscle updates)");
});

const CRATES_DIR = resolve(HERE, "../../crates");
const PARTS = Object.fromEntries(
  readdirSync(CRATES_DIR)
    .filter((d) => existsSync(resolve(CRATES_DIR, d, "part.json")))
    .map((d) => [d, JSON.parse(readFileSync(resolve(CRATES_DIR, d, "part.json"), "utf8"))]),
);
const TREE = buildTree({ anatomy: JSON.parse(read("../../designs/anatomy.json")), parts: PARTS, status: null, cartridges: [] });

test("learned: the learning funnel's caption carries the real counts, not a fixed 0", () => {
  assert.ok(!src("lib/pagescenes.ts").includes("0 learned pieces"), "the hardcoded caption is gone");
  const node = TREE.byPath.get("learning") ?? TREE.root;
  const l = learnedLib.learnedCounts({ carts: [], filters: learnedLib.jobFiltersOf(SURGERY), lenses: ["proj-a"], idle: CYCLES });
  const S = scenes.learningScene(node, [], [], 1200, 500, l);
  const texts = S.texts.map((t) => t.text);
  assert.ok(texts.some((t) => t.includes(learnedLib.learnedLine(l))), "learned line drawn");
  assert.ok(texts.some((t) => t === learnedLib.gateLine(l)), "split-gate line drawn");
  assert.ok(texts.some((t) => /^\d+ learned/.test(t)), "the beads beside the slot are labelled");
  // Without the learned record, the cartridges alone are counted, never a guessed number.
  const bare = scenes.learningScene(node, [], [], 1200, 500);
  assert.ok(bare.texts.some((t) => t.text.includes("0 learned pieces (0 judge read-outs, 0 preference cartridges)")));
  const page = src("app/learning/page.tsx");
  assert.ok(page.includes("learningScene(learnNode, rows, carts.data ?? [], W, H, learned)"));
  assert.ok(page.includes('data-e2e="learned-pieces"'));
});

/* ---------- World model figure ---------- */

test("world model: four levels with real counts, the writer's progress and queue; empty says so", () => {
  const node = TREE.byPath.get("memory") ?? TREE.root;
  const w = live.parseLive(VARIANT).writer;
  const levels = [
    { level: 1, count: 120, done: 40, total: 100 },
    { level: 2, count: 12, done: null, total: null },
    { level: 3, count: 3 },
    { level: 4, count: 1 },
  ];
  const S = scenes.worldModelScene(node, levels, w, 1200, 520);
  const t = S.texts.map((x) => x.text).join("\n");
  for (const name of ["L1 piece", "L2 file", "L3 module", "L4 architecture", "idle writer queue"]) assert.ok(t.includes(name), name);
  assert.ok(t.includes("writer 40 of 100"));
  assert.ok(t.includes("7 queued · writer held · 2 preempted"));
  // The real writer: 253 queued, held, preempted twice; its level counts from the live state.
  const real = live.parseLive(LIVE).writer;
  const R = scenes.worldModelScene(node, [1, 2, 3, 4].map((level) => ({ level, count: real.levels[`L${level}`].count })), real, 1200, 520);
  assert.ok(R.texts.map((x) => x.text).join("\n").includes("253 queued · writer held · 2 preempted"));
  const E = scenes.worldModelScene(node, [1, 2, 3, 4].map((level) => ({ level, count: null })), null, 1200, 520);
  const et = E.texts.map((x) => x.text).join("\n");
  assert.ok(et.includes("no world-model notes yet"));
  assert.ok(et.includes("queue not reported"));
  const page = src("app/memory/page.tsx") + src("components/MemoryPanels.tsx");
  assert.ok(page.includes("worldModelScene(memNode, worldLevels, writer, W, H)"));
  assert.ok(page.includes('data-e2e="world-model"') || page.includes('e2e="world-model"'));
});

/* ---------- Storage ---------- */

const STORAGE = {
  root: "C:/nano",
  free_bytes: 5_000_000_000,
  categories: [
    { id: "weights", label: "Parameter Cartridges and Weights", total_bytes: 9000, files: [{ path: "C:/nano/weights/reader.bin", bytes: 8000, modified_ms: 1, kind: "read-only" }, { path: "C:/nano/library/parameter-maps/jf-a.json", bytes: 1000, modified_ms: 2, kind: "removable" }] },
    { id: "memory", label: "Memory and World Model", files: [{ path: "C:/nano/memory/world-model/l1.jsonl", bytes: 300, modified_ms: 3, kind: "removable" }, { path: "C:/nano/memory/writer/notes.jsonl", bytes: 700, modified_ms: 4, kind: "weird" }] },
    { id: "compression", label: "Compression and Quantized Copies", total_bytes: 50, files: [{ path: "C:/nano/cache/q8/reader.q8", bytes: 50, modified_ms: null, kind: "rebuildable" }] },
    { id: "vectors", label: "Dense Vectors and Caches", total_bytes: 0, files: [] },
    { label: "no id but a label", files: [{ bytes: 5 }] },
    "junk",
  ],
};

test("storage: the real reply: categories by name, items with their class, counts and the disk", () => {
  const r = storage.parseStorage(STORAGE_REAL);
  assert.deepEqual(
    r.categories.map((c) => [c.id, c.label, c.total_bytes, c.file_count]),
    [
      ["memory", "Memory & World Model", 23519530, 189],
      ["vectors", "Dense Vectors & Index Caches", 338019077, STORAGE_REAL.categories[1].files],
      ["compression", "Compression & Quantized Copies", 12744996, STORAGE_REAL.categories[2].files],
      ["cartridges", "Parameter Cartridges & Weights", 15253057953, STORAGE_REAL.categories[3].files],
    ],
  );
  assert.equal(r.total_bytes, 15627341556);
  assert.equal(r.free_bytes, 6276251648, "from disk.free_bytes");
  assert.equal(r.disk_total_bytes, 1000169668608);
  assert.equal(r.total_files, 2730);
  assert.equal(r.floor_bytes, 4294967296);
  assert.equal(r.below_floor, false);
  assert.equal(r.root, STORAGE_REAL.home);
  const [mem, vec, comp, carts] = r.categories;
  // Each item: its class from `class`, file or folder from `kind`, its label, project and count.
  const audit = mem.files[0];
  assert.equal(audit.kind, "read-only");
  assert.equal(audit.entry, "file");
  assert.equal(audit.label, "audit log");
  assert.equal(audit.project, "nano");
  assert.ok(mem.files.every((f) => f.kind === "read-only"));
  assert.ok(vec.files.every((f) => f.kind === "rebuildable"));
  assert.match(vec.note, /^Auxiliary note vectors are held in RAM/);
  assert.ok(comp.files.some((f) => f.entry === "folder" && typeof f.count === "number" && f.count > 1));
  // Prune frees the engine's rebuildable bytes; only removable items are offered for delete.
  assert.equal(storage.prunable(vec), 338019077);
  assert.equal(storage.prunable(comp), 11932598);
  assert.equal(storage.prunable(mem), 0);
  const jf = carts.files.find((f) => f.label === "job filter");
  assert.equal(jf.kind, "removable");
  assert.equal(storage.canDelete(jf), true);
  assert.equal(storage.canDelete(carts.files.find((f) => f.kind === "read-only")), false);
  assert.equal(carts.removable_bytes, 8968740);
  // The breakdown's shares add up; a file's bytes are found by its id in the path.
  assert.ok(Math.abs(storage.shares(r).reduce((a, s) => a + s.share, 0) - 1) < 1e-9);
  assert.equal(storage.bytesFor(r, "substrate-cache-4ead3ffdb8db"), 241525460);
});

test("storage: the real settings reply: the disk floor is the one settable key; folders are read only", () => {
  const v = storage.parseStorageSettingsView(STORAGE_SETTINGS_REAL);
  assert.deepEqual(v.settable, [{ key: "storage.disk_floor_gb", value: 4, default: 4, min: 1, max: 1000, unit: "GB" }]);
  assert.deepEqual(
    v.paths.map((p) => p.id),
    ["home", "memory_dir", "projects_dir", "library_dir", "models_dir"],
  );
  assert.equal(v.memory_dirs.length, 7);
  assert.equal(v.memory_dirs[0].project, "nano");
  assert.match(v.memory_dir_note, /^The memory folder is fixed by the Nano's layout/);
  assert.equal(v.free_bytes, 6276120576);
  assert.equal(v.floor_bytes, 4294967296);
  assert.equal(v.below_floor, false);
  // Only the settable key is offered for editing, never the folder paths.
  assert.deepEqual(storage.parseStorageSettings(STORAGE_SETTINGS_REAL), { "storage.disk_floor_gb": 4 });
  // The engine reads the value as a string.
  assert.deepEqual(storage.settingsBody("storage.disk_floor_gb", 8), { key: "storage.disk_floor_gb", value: "8" });
  assert.equal(storage.parseStorageSettingsView(null), null);
  assert.deepEqual(storage.parseStorageSettingsView({}).settable, []);
});

test("storage: categories in the four's order, files largest first, totals summed, unknown kinds read only", () => {
  const r = storage.parseStorage(STORAGE);
  assert.deepEqual(
    r.categories.map((c) => c.id),
    ["memory", "vectors", "compression", "weights", "no id but a label"],
  );
  const mem = r.categories[0];
  assert.equal(mem.total_bytes, 1000, "summed where the engine left it out");
  assert.equal(mem.files[0].path, "C:/nano/memory/writer/notes.jsonl");
  assert.equal(mem.files[0].kind, "read-only", "an unknown kind is never offered for removal");
  assert.equal(r.categories[4].files.length, 0, "a file without a path is left out");
  assert.equal(r.root, "C:/nano");
  assert.equal(r.free_bytes, 5_000_000_000);
  assert.equal(r.total_bytes, 1000 + 0 + 50 + 9000 + 0);
  const sh = storage.shares(r);
  assert.ok(Math.abs(sh.reduce((a, s) => a + s.share, 0) - 1) < 1e-9);
  assert.equal(storage.prunable(r.categories[2]), 50);
  assert.equal(storage.prunable(r.categories[3]), 0);
  assert.equal(storage.canDelete(r.categories[3].files[1]), true);
  assert.equal(storage.canDelete(r.categories[3].files[0]), false, "released weights are read only");
  assert.equal(storage.canDelete(r.categories[2].files[0]), false, "rebuildable files go through prune");
  assert.equal(storage.bytesFor(r, "jf-a"), 1000);
  assert.equal(storage.bytesFor(r, "missing"), null);
  assert.equal(storage.bytesFor(null, "jf-a"), null);
});

test("storage: empty and error replies give an empty view; every destructive body carries confirm: true", () => {
  assert.equal(storage.parseStorage(null), null);
  assert.equal(storage.parseStorage("404 Not Found"), null);
  assert.deepEqual(storage.parseStorage({}).categories, []);
  assert.deepEqual(storage.shares(null), []);
  assert.deepEqual(storage.pruneBody("vectors"), { category: "vectors", confirm: true });
  assert.deepEqual(storage.deleteBody("C:/nano/x"), { path: "C:/nano/x", confirm: true });
  assert.deepEqual(storage.settingsBody("disk_floor_gb", 4), { key: "disk_floor_gb", value: "4" });
  assert.deepEqual(storage.parseStorageSettings({ settings: { memory_dir: "D:/m", disk_floor_mb: 4096, prune_on_low: true, nested: { a: 1 } } }), { memory_dir: "D:/m", disk_floor_mb: 4096, prune_on_low: true });
  assert.equal(storage.parseStorageSettings(null), null);
  assert.equal(storage.categoryOf("x", "Dense Vectors"), "vectors");
  assert.equal(storage.categoryOf("bmq8", ""), "compression");
});

test("contract: the storage page asks before every prune and delete, saves a setting with an audit row, and is in the navigation", () => {
  const page = src("app/storage/page.tsx");
  // 2026-11-26: a settings change saves at once (one post) and is undone in Audit; no dialog.
  // 2026-11-26: the prune, each delete and forgetting a project go through the multi-step delete
  // flow (the engine's preview, the typed word, then the final button); none posts directly.
  const posts = [...page.matchAll(/await post\(/g)].length;
  assert.equal(posts, 1);
  assert.ok(!page.includes('role="alertdialog"') && !page.includes("window.confirm"), "no dialog");
  for (const needle of ["Logged in Audit, where it can be undone", "path={STORAGE_PRUNE}", "body={pruneBody(c.id)}", "path={STORAGE_DELETE}", "body={deleteBody(deleting.path)}", 'path="/api/projects/forget"', "STORAGE_SETTINGS, settingsBody(", "Copy path", 'data-e2e="storage-breakdown"', 'data-e2e="storage-files"', "Not served by this engine build"]) {
    assert.ok(page.includes(needle), needle);
  }
  assert.ok(src("components/Shell.tsx").includes('{ href: "/storage/", label: "Storage"'));
  assert.ok(read("../scripts/e2e-ui-verify.mjs").includes('"/storage/"'));
});

/* ---------- Zero synthetic activity ---------- */

test("audit: no random, looping or placeholder activity in the interface sources", () => {
  const files = [];
  const walk = (d) => {
    for (const f of readdirSync(d, { withFileTypes: true })) {
      const p = resolve(d, f.name);
      if (f.isDirectory()) walk(p);
      else if (/\.(ts|tsx)$/.test(f.name)) files.push(p);
    }
  };
  walk(resolve(HERE, "../src"));
  for (const f of files) {
    const s = readFileSync(f, "utf8");
    assert.ok(!/Math\.random\s*\(/.test(s), `${f} uses Math.random`);
    assert.ok(!/0 learned pieces/.test(s) || /learned\.ts$/.test(f), `${f} hardcodes 0 learned pieces`);
  }
  const css = read("../src/app/globals.css");
  assert.ok(!/animation:[^;]*infinite/.test(css), "no looping CSS animation");
  // The topology and the anatomy plate light only from live events.
  const topo = src("components/Topology.tsx");
  assert.ok(topo.includes("useLiveEvents(onLiveEvent)"));
  assert.ok(src("app/anatomy/page.tsx").includes("tree={tree}"));
  assert.ok(src("lib/livelights.ts").includes("useLiveEvents(onEvent)"));
});
