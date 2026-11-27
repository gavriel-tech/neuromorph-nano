import "./register-ts.mjs";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const api = await import("../src/lib/api.ts");
const controls = await import("../src/lib/controls.ts");
const limits = await import("../src/lib/limits.ts");
const live = await import("../src/lib/livefeed.ts");

const HERE = dirname(fileURLToPath(import.meta.url));
const src = (p) => readFileSync(resolve(HERE, "../src", p), "utf8");

/* The real replies, captured from a running server (tests/fixtures): `GET /api/limits` keys
 * `current` by full control keys, nests `effective` as {vram, ram, cpu, mode}, names the
 * recommendation `recommendation` with one {key, value, reason} per limiter plus `notes`, and gives
 * this machine's `profile`. `GET /api/live` carries the serving engine's capacity and limits. */
const fixture = (name) => JSON.parse(readFileSync(resolve(HERE, "fixtures", `${name}.json`), "utf8"));
const LIMITS = fixture("limits");
const LIVE = fixture("live");
/* The same profile as the server now gives it with no engine serving: the card read through the
 * driver without a context, so free equals total and the reserve is not measured. */
const PROFILE_CARD = { ...LIMITS.profile, card: { name: "NVIDIA GeForce GTX 1650", compute: "7.5", total_mb: 4095, free_mb: 4095, reserve_mb: 140, reserve_measured: false }, gpu_ceiling_mb: 3955 };

const control = (over = {}) => ({ key: "limits.vram_mb", label: "VRAM limit", help: "", kind: "integer", min: 256, max: 81920, step: 256, unit: "MiB", values: [], default: "auto", value: "auto", source: "default", wired: true, applies: "now", ...over });

/* ---------- Parser ---------- */

test("limits: parseLimits reads the real reply", () => {
  const r = limits.parseLimits(LIMITS);
  // `current` is keyed `limits.vram_mb` and so on.
  assert.deepEqual(r.current, { vram_mb: "auto", ram_mb: "auto", cpu_threads: "auto" });
  // `effective` is nested: the values in force are read from each block's setting and cap.
  assert.deepEqual(r.effective, { vram_mb: "auto", ram_mb: "auto", cpu_threads: "auto" });
  assert.equal(r.effectiveDetail.mode, "processor");
  assert.equal(r.effectiveDetail.cpu.effective_threads, 12);
  assert.equal(r.effectiveDetail.ram.room_mb, 10458);
  assert.equal(r.effectiveDetail.vram.driver_free_mb, null);
  assert.match(r.effectiveDetail.vram.note, /^auto: the Nano uses what the driver reports free/);
  // `recommendation`, one {key, value, reason} per limiter, the tier and its label, the notes.
  assert.equal(r.recommended.tier, "cpu-only");
  assert.equal(r.recommended.tier_label, "Processor only (no usable graphics card)");
  assert.deepEqual(r.recommended.values, { vram_mb: "0", ram_mb: "auto", cpu_threads: "11" });
  assert.match(r.recommended.reasons.cpu_threads, /^Every MicroModel runs on the processor here\. 11 of the 12 threads/);
  assert.match(r.recommended.reasons.vram_mb, /0 means processor only/);
  assert.match(r.recommended.reasons.ram_mb, /32436 MiB/);
  assert.equal(r.recommended.notes.length, 1);
  assert.match(r.recommended.notes[0], /Prune rebuildable storage/);
  assert.equal(r.recommended.summary, null);
  // `profile`: this machine, whether or not an engine is serving.
  assert.equal(r.profile.card, null);
  assert.equal(r.profile.host_total_mb, 32436);
  assert.equal(r.profile.threads, 12);
  assert.equal(r.profile.hardware_threads, 12);
  assert.equal(r.profile.ram_ceiling_mb, 24327);
  assert.match(r.applies, /^live, with no restart/);
  assert.equal(limits.tierBand(r.recommended.tier), 0);
});

test("limits: the profile card read with no engine serving parses as normal, its reserve unmeasured", () => {
  const r = limits.parseLimits({ ...LIMITS, profile: PROFILE_CARD });
  assert.equal(r.profile.card.name, "NVIDIA GeForce GTX 1650");
  assert.equal(r.profile.card.total_mb, 4095);
  assert.equal(r.profile.card.free_mb, 4095);
  assert.equal(r.profile.card.reserve_measured, false);
  assert.equal(limits.detectedFor("vram_mb", r.profile), 4095);
  assert.equal(limits.detectedWords("vram_mb", r.profile), "NVIDIA GeForce GTX 1650 · 4095 MiB");
  // With no engine serving, the machine shown is the profile; with one, the live capacity.
  const liveCap = live.parseLive(LIVE).capacity;
  assert.equal(limits.machineCapacity(liveCap, false, r.profile).source, "profile");
  assert.equal(limits.machineCapacity(liveCap, false, r.profile).cap.card.reserve_measured, false);
  assert.equal(limits.machineCapacity(liveCap, true, r.profile).source, "live");
  assert.equal(limits.machineCapacity(liveCap, true, r.profile).cap.card.reserve_measured, true);
  assert.equal(limits.machineCapacity(null, false, null).cap, null);
  assert.equal(limits.machineCapacity(liveCap, false, null).source, "live");
});

test("limits: parseLimits is defensive and never guesses", () => {
  assert.equal(limits.parseLimits(null), null);
  assert.equal(limits.parseLimits("nope"), null);
  assert.equal(limits.parseLimits([1, 2]), null);
  const e = limits.parseLimits({});
  assert.deepEqual(e.current, { vram_mb: null, ram_mb: null, cpu_threads: null });
  assert.equal(e.recommended, null);
  assert.equal(e.profile, null);
  // Numbers as numbers, AUTO in capitals, leading zeros, and junk.
  const m = limits.parseLimits({ current: { vram_mb: 4096, ram_mb: "AUTO", cpu_threads: "008" }, effective: { vram_mb: "-1", ram_mb: 1.5, cpu_threads: "lots" }, recommended: { tier: "  ", reasons: "x" } });
  assert.deepEqual(m.current, { vram_mb: "4096", ram_mb: "auto", cpu_threads: "8" });
  assert.deepEqual(m.effective, { vram_mb: null, ram_mb: null, cpu_threads: null });
  assert.equal(m.recommended.tier, null);
  assert.deepEqual(m.recommended.reasons, { vram_mb: null, ram_mb: null, cpu_threads: null });
});

test("limits: /api/live capacity carries the limits in force, nested, with the residency mode", () => {
  const r = live.parseLive(LIVE);
  const l = r.capacity.limits;
  assert.equal(l.vram_mb, "auto");
  assert.equal(l.ram_mb, "auto");
  assert.equal(l.cpu_threads, "auto");
  assert.equal(l.mode, "four seated");
  assert.deepEqual(
    { total: l.vram.total_mb, used: l.vram.used_mb, free: l.vram.driver_free_mb, eff: l.vram.effective_free_mb, now: l.vram.cpu_only_now, next: l.vram.cpu_only_next_start },
    { total: 4095, used: 2105, free: 698, eff: 698, now: false, next: false },
  );
  assert.equal(l.ram.used_mb, 8884);
  assert.equal(l.cpu.hardware_threads, 12);
  // What auto resolves to, and the Nano's own use.
  assert.equal(limits.limitResolved("vram_mb", l), 698);
  assert.equal(limits.limitResolved("ram_mb", l), 10316);
  assert.equal(limits.limitResolved("cpu_threads", l), 12);
  assert.equal(limits.limitUsed("vram_mb", l), 2105);
  assert.equal(limits.limitUsed("ram_mb", l), 8884);
  assert.equal(limits.inForceShort("vram_mb", "auto", l), "auto · 698 MiB free");
  assert.equal(limits.inForceShort("ram_mb", "auto", l), "auto · 10316 MiB room");
  assert.equal(limits.inForceShort("cpu_threads", "auto", l), "auto · 12 threads");
  assert.equal(limits.inForceShort("vram_mb", "4096", l), "4096 MiB");
  assert.equal(limits.inForceShort("vram_mb", "auto", null), "auto");
  // A cap in force, and the processor running everything now.
  const capped = live.parseLiveLimits({ vram: { setting: "4096", cap_mb: 4096 }, ram: { setting: "auto", cap_mb: null }, cpu: { setting: "6", cap: 6 } });
  assert.deepEqual([capped.vram_mb, capped.ram_mb, capped.cpu_threads], ["4096", "auto", "6"]);
  assert.equal(live.parseLiveLimits({ vram: { setting: "0", cap_mb: null, cpu_only_now: true } }).vram_mb, "0");
  // A flat block still reads, and a reply without limits gives none.
  assert.deepEqual(
    (({ vram_mb, ram_mb, cpu_threads }) => ({ vram_mb, ram_mb, cpu_threads }))(live.parseLiveLimits({ vram_mb: "4096", ram_mb: "auto", cpu_threads: 6 })),
    { vram_mb: "4096", ram_mb: "auto", cpu_threads: "6" },
  );
  assert.equal(live.parseLive({ state: "idle", capacity: { ...LIVE.capacity, limits: undefined } }).capacity.limits, null);
  // Residency: the engine's mode first, then the writer's kept_resident.
  assert.equal(limits.residency(r.writer, l.mode).mode, "seated");
  assert.equal(limits.residency(null, "evict and restore (the writer loads per idle session)").mode, "swapping");
  assert.equal(limits.residency(null, "none seated").mode, "none");
  assert.equal(limits.residency(null, "processor").mode, "processor");
  assert.equal(limits.residency(r.writer).mode, "seated", "kept_resident true");
  assert.equal(limits.residency({ kept_resident: false }).mode, "swapping");
  assert.equal(limits.residency({ kept_resident: null }), null);
  assert.equal(limits.residency(null), null);
});

test("limits: words, detected hardware and the VRAM cap on the bar", () => {
  assert.equal(limits.limitWords("vram_mb", "auto"), "auto (follows this machine)");
  assert.equal(limits.limitWords("vram_mb", "0"), "0 MiB (processor only from the next start)");
  assert.equal(limits.limitWords("ram_mb", "8192"), "8192 MiB");
  assert.equal(limits.limitWords("cpu_threads", "1"), "1 thread");
  assert.equal(limits.limitWords("cpu_threads", "6"), "6 threads");
  assert.equal(limits.limitWords("ram_mb", null), "not reported");
  assert.equal(limits.limitShort("vram_mb", "0"), "CPU only");
  const p = live.parseCapacity(LIVE.capacity);
  assert.equal(limits.detectedFor("vram_mb", p), 4095);
  assert.equal(limits.detectedFor("ram_mb", p), 32436);
  assert.equal(limits.detectedFor("cpu_threads", p), 12);
  assert.equal(limits.detectedWords("ram_mb", p), "32436 MiB · 10316 free");
  assert.equal(limits.detectedFor("vram_mb", live.parseCapacity(LIMITS.profile)), 0, "the profile of a process with no card");
  assert.equal(limits.detectedFor("vram_mb", null), null);
  assert.equal(limits.vramCapShare("4096", 8192), 0.5);
  assert.equal(limits.vramCapShare("auto", 8192), null);
  assert.equal(limits.vramCapShare("0", 8192), null);
  assert.equal(limits.vramCapShare("16384", 8192), 1);
  assert.equal(limits.vramCapShare("4096", null), null);
});

test("limits: tiers map onto the four spectral bands", () => {
  assert.equal(limits.tierBand("cpu-only"), 0);
  assert.equal(limits.tierBand("4 GB"), 1);
  assert.equal(limits.tierBand("gpu_4gb"), 1);
  assert.equal(limits.tierBand("8-12 GB"), 2);
  assert.equal(limits.tierBand("gpu_8_12gb"), 2);
  assert.equal(limits.tierBand("16-24+ GB"), 3);
  assert.equal(limits.tierBand("gpu_16_24gb"), 3);
  assert.equal(limits.tierBand(null), -1);
  assert.equal(limits.tierBand("unheard of"), -1);
});

test("limits: validateLimit takes auto or a whole number inside the machine", () => {
  assert.deepEqual(limits.validateLimit("vram_mb", " Auto "), { ok: true, value: "auto" });
  assert.deepEqual(limits.validateLimit("vram_mb", "0"), { ok: true, value: "0" });
  assert.deepEqual(limits.validateLimit("vram_mb", "4096", 8192), { ok: true, value: "4096" });
  assert.deepEqual(limits.validateLimit("vram_mb", "0004096"), { ok: true, value: "4096" });
  assert.equal(limits.validateLimit("vram_mb", "100").ok, false);
  assert.equal(limits.validateLimit("vram_mb", "9000", 8192).ok, false);
  assert.equal(limits.validateLimit("ram_mb", "0").ok, false);
  assert.equal(limits.validateLimit("cpu_threads", "0").ok, false);
  assert.deepEqual(limits.validateLimit("cpu_threads", "4", 16), { ok: true, value: "4" });
  assert.equal(limits.validateLimit("cpu_threads", "17", 16).ok, false);
  for (const bad of ["", "  ", "-1", "1.5", "1e3", "lots"]) assert.equal(limits.validateLimit("ram_mb", bad).ok, false, bad);
});

/* ---------- The three controls take auto; every other control is unchanged ---------- */

test("controls: the three limit keys accept auto, sit in Hardware limits and validate 0 for VRAM", () => {
  for (const key of ["limits.vram_mb", "limits.ram_mb", "limits.cpu_threads"]) {
    // Even when the engine names no auto preset and the value is a number, auto is accepted.
    const c = control({ key, default: "4096", value: "4096", values: [], kind: undefined });
    assert.equal(controls.acceptsAuto(c), true, key);
    assert.equal(controls.isNumericControl(c), true, key);
    assert.equal(controls.isLimitControl(c), true, key);
    assert.equal(controls.controlGroupName(c), "Hardware limits", key);
    assert.deepEqual(controls.validateIntegerControl(c, "auto"), { ok: true, value: null, normalized: "auto" });
    assert.deepEqual(controls.validateIntegerControl(c, "AUTO"), { ok: true, value: null, normalized: "auto" });
    assert.equal(controls.formatControlValue("auto", key), "auto (follows this machine)");
  }
  assert.ok(controls.CONTROL_GROUP_ORDER.includes("Hardware limits"));
  const vram = control();
  assert.deepEqual(controls.validateIntegerControl(vram, "0"), { ok: true, value: 0, normalized: "0" });
  assert.deepEqual(controls.validateIntegerControl(vram, "100"), { ok: false, error: "Must be at least 256 MiB" });
  assert.deepEqual(controls.validateIntegerControl(vram, "4096"), { ok: true, value: 4096, normalized: "4096" });
  assert.equal(controls.formatControlValue("0", "limits.vram_mb"), "0 (processor only from the next start)");
  assert.equal(controls.formatControlValue("6", "limits.cpu_threads", "threads"), "6 threads");
  assert.equal(controls.formatControlValue("8192", "limits.ram_mb", "MiB"), "8192 MiB");
  const ram = control({ key: "limits.ram_mb", min: 512 });
  assert.deepEqual(controls.validateIntegerControl(ram, "0"), { ok: false, error: "Must be at least 512 MiB" });
  // Stepping from auto starts at what auto resolves to.
  assert.equal(controls.stepIntegerControl(vram, "auto", 1, 7168), "7424");
});

test("controls: keys outside the limiters keep their behaviour", () => {
  const plain = control({ key: "budget.pack_tokens", default: "4000", value: "4000", values: ["1000", "4000"], min: 1000, max: 16000, unit: "tokens", step: 250 });
  assert.equal(controls.acceptsAuto(plain), false);
  assert.equal(controls.isLimitControl(plain), false);
  assert.equal(controls.controlGroupName({ ...plain, group: undefined }), "Budgets and allowances");
  assert.deepEqual(controls.validateIntegerControl(plain, "auto"), { ok: false, error: "Must be a whole number" });
  assert.deepEqual(controls.validateIntegerControl(plain, "0"), { ok: false, error: "Must be at least 1000 tokens" });
  assert.equal(controls.formatControlValue("auto", "budget.gpu_mb_max"), "auto (follows this card)");
  assert.equal(controls.formatControlValue("0", "quota.session_tokens"), "unlimited");
  // A control without a key (the old call shape) still works.
  assert.equal(controls.acceptsAuto({ values: [], default: "auto", value: "1" }), true);
  assert.equal(controls.acceptsAuto({ values: [], default: "1", value: "1" }), false);
});

/* ---------- Apply Recommended Defaults: what it lists, and only after confirmation ---------- */

test("limits: recommendedChanges lists each limiter that would change", () => {
  const r = limits.parseLimits(LIMITS);
  const ch = limits.recommendedChanges(r);
  assert.deepEqual(
    ch.map((c) => [c.key, c.from, c.to]),
    [
      ["limits.vram_mb", "auto", "0"],
      ["limits.cpu_threads", "auto", "11"],
    ],
  );
  assert.equal(ch[0].label, "Graphics card memory (VRAM)");
  assert.deepEqual(limits.recommendedChanges(limits.parseLimits({ ...LIMITS, current: { "limits.vram_mb": "0", "limits.ram_mb": "auto", "limits.cpu_threads": "11" } })), []);
  assert.deepEqual(limits.recommendedChanges(limits.parseLimits({ current: LIMITS.current })), []);
  assert.deepEqual(limits.recommendedChanges(null), []);
  assert.deepEqual(limits.APPLY_RECOMMENDED_BODY, { confirm: true });
  assert.ok(Object.isFrozen(limits.APPLY_RECOMMENDED_BODY));
  assert.deepEqual(limits.setLimitBody("ram_mb", "8192"), { key: "limits.ram_mb", value: "8192" });
});

/** A mock engine for the limits routes, answering in the real reply's shape: GET /api/limits,
 *  POST /api/limits {key, value}, POST /api/limits/apply-recommended {confirm: true}. Writes need
 *  the local token. */
async function withEngine(fn) {
  const origFetch = globalThis.fetch;
  const origDocument = globalThis.document;
  api._resetApiForTest();
  const state = JSON.parse(JSON.stringify(LIMITS));
  const posts = [];
  globalThis.document = { querySelector: (sel) => (sel === 'meta[name="nano-token"]' ? { getAttribute: () => "tok", setAttribute() {} } : null) };
  const json = (status, body) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  globalThis.fetch = async (url, init = {}) => {
    const path = String(url);
    if (!init.method || init.method === "GET") {
      if (path === "/api/limits") return json(200, state);
      return json(404, { error: "unknown endpoint" });
    }
    if (init.headers?.["X-Nano-Token"] !== "tok") return new Response("forbidden", { status: 403 });
    let body;
    try {
      body = JSON.parse(init.body);
    } catch {
      return json(400, { error: "body: truncated JSON" });
    }
    posts.push({ path, body });
    if (path === "/api/limits") {
      if (!(body?.key in state.current)) return json(400, { error: `unknown limit ${body?.key}` });
      const v = String(body.value);
      if (v !== "auto" && !/^\d+$/.test(v)) return json(400, { error: `${body.key}: expected auto or a whole number` });
      state.current[body.key] = v;
      return json(200, { key: body.key, value: v, note: "applies live, with no restart", current: state.current });
    }
    if (path === "/api/limits/apply-recommended") {
      if (body?.confirm !== true) return json(400, { error: "confirm: apply-recommended needs {confirm: true}" });
      for (const f of ["vram_mb", "ram_mb", "cpu_threads"]) state.current[state.recommendation[f].key] = state.recommendation[f].value;
      return json(200, state);
    }
    return json(404, { error: "unknown endpoint" });
  };
  try {
    await fn({ state, posts });
  } finally {
    api._resetApiForTest();
    globalThis.fetch = origFetch;
    if (origDocument === undefined) delete globalThis.document;
    else globalThis.document = origDocument;
  }
}

test("contract: GET /api/limits is read through the shared store and parsed", async () => {
  await withEngine(async () => {
    await api.refresh("/api/limits");
    const r = limits.parseLimits(api.readCache("/api/limits").data);
    assert.equal(r.recommended.tier, "cpu-only");
    assert.equal(limits.recommendedChanges(r).length, 2);
  });
});

test("contract: POST /api/limits sets one limiter by its control key; refusals surface as the engine's message", async () => {
  await withEngine(async ({ state, posts }) => {
    const r = await api.post("/api/limits", limits.setLimitBody("vram_mb", "4096"));
    assert.deepEqual(posts.at(-1), { path: "/api/limits", body: { key: "limits.vram_mb", value: "4096" } });
    assert.equal(limits.parseLimits(r).current.vram_mb, "4096");
    await api.post("/api/limits", limits.setLimitBody("vram_mb", "auto"));
    assert.equal(state.current["limits.vram_mb"], "auto");
    await api.post("/api/limits", limits.setLimitBody("vram_mb", "0"));
    assert.equal(state.current["limits.vram_mb"], "0");
    await assert.rejects(() => api.post("/api/limits", { key: "limits.disk_mb", value: "1" }), { message: "unknown limit limits.disk_mb" });
    await assert.rejects(() => api.post("/api/limits", { key: "limits.ram_mb", value: "lots" }), { message: "limits.ram_mb: expected auto or a whole number" });
  });
});

test("contract: apply-recommended is sent with {confirm: true} and is refused without it", async () => {
  await withEngine(async ({ state, posts }) => {
    await assert.rejects(() => api.post("/api/limits/apply-recommended", {}), { message: "confirm: apply-recommended needs {confirm: true}" });
    await assert.rejects(() => api.post("/api/limits/apply-recommended", { confirm: "yes" }), { message: "confirm: apply-recommended needs {confirm: true}" });
    assert.equal(state.current["limits.vram_mb"], "auto", "no refused call changed a limit");
    const r = await api.post("/api/limits/apply-recommended", limits.APPLY_RECOMMENDED_BODY);
    assert.deepEqual(posts.at(-1), { path: "/api/limits/apply-recommended", body: { confirm: true } });
    const after = limits.parseLimits(r);
    assert.deepEqual(after.current, { vram_mb: "0", ram_mb: "auto", cpu_threads: "11" });
    assert.deepEqual(limits.recommendedChanges(after), [], "nothing left to apply");
  });
});

test("contract: the panel posts only from its confirmations and the page carries it", () => {
  const panel = src("components/HardwareLimits.tsx");
  // The apply body is sent only by applyRecommended, which only the confirm button of the dialog
  // calls; the first button opens the dialog.
  assert.equal(panel.split("APPLY_RECOMMENDED_BODY").length - 1, 2, "imported once, sent once");
  assert.ok(panel.includes("await post<unknown>(`${LIMITS_API}/apply-recommended`, APPLY_RECOMMENDED_BODY);"));
  assert.ok(panel.includes("onClick={() => setOpen(true)}"));
  assert.ok(panel.includes('role="alertdialog" aria-label="Confirm Apply Recommended Defaults"'));
  assert.ok(panel.includes("onClick={go}"));
  assert.equal(panel.split("await applyRecommended()").length - 1, 1, "applied from one place: the dialog's go");
  assert.ok(panel.includes("{changes.map((c) =>"), "the dialog lists what will change");
  // One limiter: Set limit arms a confirmation; only Confirm change posts.
  assert.ok(panel.includes("await post<unknown>(LIMITS_API, setLimitBody(field, value));"));
  assert.equal(panel.split("setLimit(field, armed)").length - 1, 1);
  assert.ok(panel.includes("{busy ? \"Saving\" : \"Confirm change\"}"));
  const page = src("app/settings/page.tsx");
  assert.ok(page.includes("<HardwareLimitsPanel />"));
  assert.ok(page.includes("capMb={vramCapMb && vramCapMb > 0 ? vramCapMb : null}"));
  assert.ok(page.includes("residency(live.reply?.writer, live.reply?.capacity?.limits?.mode)"));
  assert.ok(page.includes("<Capacity cap={machine.cap}"), "the machine falls back to the /api/limits profile");
  const bar = src("components/LivePanels.tsx");
  assert.ok(bar.includes('data-e2e="vram-cap"'));
});
